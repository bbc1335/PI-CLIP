import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
from model.PSPNet import OneModel as PSPNet
from einops import rearrange

# add
import clip
import math
from model.get_cam import get_img_cam
from pytorch_grad_cam import GradCAM
from clip.clip_text import new_class_names, new_class_names_coco

# ============================================================================
# [既有方案 D 纯净版接口开始] 多粒度文本-视觉相似度金字塔
# 核心实现全部位于 model/text_visual_pyramid_pure.py。本文件只负责：
#   1. 复用原 forward 已经提取的 query CLIP 多层特征；
#   2. 复用初始化阶段已经缓存的 foreground/background 文本特征；
#   3. 将固定先验图作为一个额外通道接入原分割流程。
# ============================================================================
from model.text_visual_pyramid_pure import (
    CLIP_LAYER_IDS,
    TextVisualSimilarityPyramidPure,
)

# ============================================================================
# [MI-CoCluster VVP 新接口开始]
# 核心实现全部位于 model/mi_cocluster_vvp.py。该接口根据配置用 MI-CoCluster
# VVP 替换原 CNN stage-4/5 VVP，不改变文本-视觉先验分支和旧权重协议。
# ============================================================================
from model.mi_cocluster_vvp import MICoClusterVVP
# ============================================================================
# [既有方案 D 与 MI-CoCluster VVP 接口结束]
# ============================================================================


def _flatten_integer_values(value):
    """[方案 D 新增] 将 class_name 或 cat_idx 中的类别索引统一展平。"""
    if value is None:
        return []
    if torch.is_tensor(value):
        if value.numel() == 0:
            return []
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_integer_values(item))
        return flattened
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []

def _split_classname_synonyms(classname):
    """将类别名中的逗号同义词拆成独立自然短语。

    例如：
        "person with clothes,people,human"
    会变成：
        ["person with clothes", "people", "human"]

    没有逗号的普通类别名仍然返回单元素列表，因此不会改变原有类别行为。
    """
    if not isinstance(classname, str):
        return [classname]
    variants = [item.strip() for item in classname.split(',') if item.strip()]
    return variants if variants else [classname]


# 通用背景模板只扩展背景提示，不改变前景模板。
# 多个背景模板的文本特征会在 zeroshot_classifier 中先归一化、再平均，
# 最终仍然得到一个 [num_classes, text_dim] 的背景特征矩阵。
GENERIC_BACKGROUND_TEMPLATES = [
    'a photo without {}.',
    'a background scene without {}.',
    'a scene with no {}.',
]


def zeroshot_classifier(classnames, templates, model):
    """生成类别文本特征。

    参数：
        classnames:
            类别名称列表。函数会固定将逗号分隔的同义词拆成多个自然提示。
        templates:
            提示模板列表。
        model:
            CLIP 文本编码器。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            # 同义词分开生成文本，避免把多个词拼成不自然的
            # "person with clothes,people,human" 长句子。
            classname_variants = _split_classname_synonyms(classname)
            texts = [
                template.format(classname_variant)
                for classname_variant in classname_variants
                for template in templates
            ]
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # 对同义词和背景模板的 embedding 做平均，再重新归一化。
            # 这样输出形状仍是 [1, text_dim]，下游 VTP 接口不需要修改。
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask):
    """
    计算查询特征与支持集特征之间的空间余弦相似度图（VVP: Vision-Visual Prototype）
    
    参数:
        q: 查询图像特征 [bs, dim, h, w]
        s: 支持集图像特征 [bs, dim, h, w]
        mask: 支持集前景掩码 [bs, 1, h, w] 或 [bs, h, w]
    
    返回:
        similarity: 空间相似度图 [bs, 1, h, w]，值域[0,1]，表示每个位置与支持集前景的相似度
    """
    # ==================== 掩码预处理 ====================
    # 如果掩码是3D张量[bs, h, w]，添加通道维度变为[bs, 1, h, w]
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    # 将掩码二值化（只有值为1的才是前景）并插值到与查询特征相同的空间尺寸
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    
    # ==================== 支持集特征掩蔽 ====================
    cosine_eps = 1e-7  # 余弦相似度计算的小常数，防止除零
    s = s * mask  # 用掩码过滤支持集特征，只保留前景区域，忽略背景干扰
    
    # ==================== 特征展平与归一化 ====================
    bsize, ch_sz, sp_sz, _ = q.size()[:]  # 获取batch_size、通道数、空间尺寸
    
    # 查询特征：展平为 [bs, dim, h*w]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    # 计算查询特征的L2范数 [bs, 1, h*w]，用于余弦相似度归一化
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    
    # 支持特征：先展平为 [bs, dim, h*w]，再转置为 [bs, h*w, dim]
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    # 计算支持特征的L2范数 [bs, h*w, 1]
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    
    # ==================== 批量矩阵乘法计算余弦相似度 ====================
    # 公式：similarity = (s · q) / (||s|| * ||q|| + ε)
    # tmp_supp [bs, h*w, dim] @ tmp_query [bs, dim, h*w] -> [bs, h*w, h*w]
    # 分子：支持集每个位置与查询集每个位置的点积
    # 分母：两者的L2范数乘积，实现余弦归一化
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    
    # ==================== 最大池化聚合 ====================
    # 对支持集维度取最大值：对每个查询位置，找到与支持集最相似的位置
    # similarity.max(1)[0] -> [bs, h*w]，获取每个查询位置的最大相似度响应
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    # 重塑为2D空间图 [bs, 1, h, w]
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class OneModel(nn.Module):
    """
    PI-CLIP: 基于CLIP和元学习的少样本分割模型
    整体思想：结合CNN的精细特征、CLIP的视觉-文本语义、Transformer的跨模态交互，以及基类知识迁移
    """
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 类别类型: 'Base' 基类 或 'Novel' 新类
        self.dataset = args.data_set  # 数据集名称: 'pascal' 或 'coco'
        if self.dataset == 'pascal':
            self.base_classes = 15  # PASCAL数据集基类数量
        elif self.dataset == 'coco':
            self.base_classes = 60  # COCO数据集基类数量
        self.low_fea_id = args.low_fea[-1]  # 低层特征的层级标识(用于Gram矩阵计算)

        assert args.layers in [50, 101, 152]  # ResNet层数必须是50/101/152
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)  # 损失函数
        self.shot = args.shot  # 少样本的样本数量(1-shot, 5-shot等)

        # [方案 D 纯净版接口] 原来的 VVP 相似度通道保持不变；启用先验时，
        # query_merge 额外接收文本视觉先验通道，Transformer 仍只使用 VVP。
        self.use_text_visual_pyramid = bool(
            getattr(args, "use_text_visual_pyramid", False)
        )
        # [MI-CoCluster VVP 接口] 新方法只替换原 VVP 的相似度来源，
        # 不改变文本视觉先验和 Transformer 的通道协议。默认关闭，
        # 因此不修改配置时仍可复现原来的训练行为。
        self.use_mi_cocluster_vvp = bool(
            getattr(args, "use_mi_cocluster_vvp", False)
        )
        self.visual_similarity_channels = self.shot * 2
        self.similarity_channels = self.visual_similarity_channels + int(
            self.use_text_visual_pyramid
        )
        self.vgg = args.vgg  # 是否使用VGG backbone
        models.BatchNorm = BatchNorm

        # ==================== 加载预训练的PSPNet模型 ====================
        PSPNet_ = PSPNet(args)
        new_param = torch.load(args.pre_weight, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:
            # 处理state_dict键名不匹配的情况(去掉module.前缀)
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        
        # 从PSPNet中获取各个组件
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm  # Pyramid Pooling Module金字塔池化模块
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])  # 分类器前几层
        self.base_learnear = nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])  # 基类学习器

        # ==================== 特征降维模块 ====================
        if self.vgg:
            fea_dim = 512 + 256  # VGG的feature维度
        else:
            fea_dim = 1024 + 512  # ResNet的feature维度
        
        # 支持集特征降维网络:将多尺度CNN特征压缩到256维
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        # 查询集特征降维网络
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        # ==================== 特征融合模块 ====================
        channel = 512 + self.similarity_channels
        # 查询特征融合网络:将CNN特征、CLIP特征、原型等融合
        self.query_merge = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # 支持特征融合网络
        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # ==================== Transformer交互模块 ====================
        self.transformer = Transformer(shot=self.shot)  # 跨注意力Transformer

        # ==================== Gram矩阵融合模块 ====================
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble: 学习器集成
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # ==================== K-Shot重加权模块 ====================
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                # 简单的线性重加权
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                # 带非线性变换的重加权网络
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        # ==================== CLIP相关模块 ====================
        self.annotation_root = args.annotation_root  # 标注根目录
        self.clip_model, _ = clip.load(args.clip_path)  # 加载预训练CLIP模型
        
        # 预计算文本特征（零样本分类器）。
        # 同义词拆分和多背景模板固定启用，作为文本提示增强方法的一部分，
        # 不再暴露为独立消融开关。
        background_templates = GENERIC_BACKGROUND_TEMPLATES
        if self.dataset == 'pascal':
            # PASCAL 背景文本特征：
            # 同义词拆成自然短语，并加入多个通用背景模板。
            self.bg_text_features = zeroshot_classifier(
                new_class_names,
                background_templates,
                self.clip_model,
            )
            # PASCAL 前景文本特征：同义词拆成自然短语。
            self.fg_text_features = zeroshot_classifier(
                new_class_names,
                ['a photo of {}.'],
                self.clip_model,
            )
        elif self.dataset == 'coco':
            # COCO 背景文本特征：
            # 同义词拆成自然短语，并加入多个通用背景模板。
            self.bg_text_features = zeroshot_classifier(
                new_class_names_coco,
                background_templates,
                self.clip_model,
            )
            # COCO 前景文本特征：同义词拆成自然短语。
            self.fg_text_features = zeroshot_classifier(
                new_class_names_coco,
                ['a photo of {}.'],
                self.clip_model,
            )

        # ====================================================================
        # [方案 D 纯净版接口] 创建固定文本-视觉金字塔，并复用上面已经
        # 计算好的 fg_text_features / bg_text_features。该模块没有可训练
        # 参数，文本投影和融合卷积在初始化后都保持冻结。
        # ====================================================================
        self.text_visual_pyramid = None
        if self.use_text_visual_pyramid:
            visual_proj = getattr(self.clip_model.visual, "proj", None)
            if visual_proj is None:
                raise RuntimeError(
                    "TextVisualSimilarityPyramidPure requires a CLIP ViT "
                    "visual projection layer."
                )
            projection_init = torch.linalg.pinv(
                visual_proj.detach().float()
            ).t().contiguous()
            self.text_visual_pyramid = TextVisualSimilarityPyramidPure(
                vision_dim=visual_proj.shape[0],
                text_dim=self.fg_text_features.shape[-1],
                projection_init=projection_init,
            )
        # ====================================================================
        # [MI-CoCluster VVP 接口] 使用 CLIP 第 9、10 层 patch 特征生成
        # 新的 VVP 相似度图。默认关闭可学习融合时保持固定的 0.5C + 0.5U，
        # 不新增可训练参数；开启后两层各有一个可学习 logit（存放在长度为 2
        # 的参数中），用于学习凸组合。
        # ====================================================================
        self.mi_cocluster_vvp = None
        if self.use_mi_cocluster_vvp:
            self.mi_cocluster_vvp = MICoClusterVVP(
                shot=self.shot,
                rank=16,
                num_iters=4,
                use_learnable_fusion=args.use_learnable_fusion,
            )
        # ====================================================================
        # [方案 D 纯净版接口结束]
        # ====================================================================

    @staticmethod
    def _reshape_clip_feature_layers(
        clip_feature_layers,
        image_h,
        image_w,
        patch_size=16,
    ):
        """[MI-CoCluster VVP 新增] 将 CLIP 多层 token 转成空间特征图。

        CLIP 的 ``extract=True`` 返回的每一层特征形状为
        ``[N_patch + 1, B, C]``，其中第 0 个 token 是 [CLS]。新 VVP
        只需要 patch token，因此先去掉 [CLS]，再恢复为 ``[B, C, H, W]``。
        """
        patch_tokens = [
            layer[1:, :, :] for layer in clip_feature_layers
        ]
        grid_h = int(image_h) // int(patch_size)
        grid_w = int(image_w) // int(patch_size)
        num_patches = grid_h * grid_w

        feature_maps = []
        for layer in patch_tokens:
            # [N_patch, B, C] -> [B, C, N_patch]
            layer = layer.permute(1, 2, 0)
            if layer.shape[-1] != num_patches:
                raise ValueError(
                    "CLIP patch token number does not match the input "
                    f"grid: got {layer.shape[-1]}, expected {num_patches}."
                )
            feature_maps.append(
                layer.reshape(
                    layer.shape[0],
                    layer.shape[1],
                    grid_h,
                    grid_w,
                ).float()
            )
        return feature_maps

    def _build_cnn_vvp(
        self,
        query_feat_4,
        query_feat_5,
        supp_feat_4,
        supp_feat_5,
        mask,
    ):
        """[MI-CoCluster VVP 对比接口] 复用原 CNN stage-4/5 VVP。

        1-shot 时输出 [B, 2, H, W]，K-shot 时输出 [B, 2*K, H, W]。
        输出通道协议与 MI-CoCluster VVP 完全一致，可用于统一接口。
        """
        if self.shot == 1:
            similarity1 = get_similarity(query_feat_5, supp_feat_5, mask)
            similarity2 = get_similarity(query_feat_4, supp_feat_4, mask)
            return torch.cat([similarity1, similarity2], dim=1)

        mask = rearrange(mask, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_5 = rearrange(
            supp_feat_5, "(b n) c h w -> b n c h w", n=self.shot
        )
        supp_feat_4 = rearrange(
            supp_feat_4, "(b n) c h w -> b n c h w", n=self.shot
        )
        clip_similarity_1 = [
            get_similarity(
                query_feat_5,
                supp_feat_5[:, i, ...],
                mask=mask[:, i, ...],
            )
            for i in range(self.shot)
        ]
        clip_similarity_2 = [
            get_similarity(
                query_feat_4,
                supp_feat_4[:, i, ...],
                mask=mask[:, i, ...],
            )
            for i in range(self.shot)
        ]
        similarity1 = torch.cat(clip_similarity_1, dim=1)
        similarity2 = torch.cat(clip_similarity_2, dim=1)
        return torch.cat([similarity1, similarity2], dim=1)

    def _resolve_class_indices(self, class_name, cat_idx, batch_size, device):
        """[方案 D 新增] 解析当前 batch 每个查询图像对应的类别文本索引。"""
        indices = _flatten_integer_values(class_name)
        if not indices:
            indices = _flatten_integer_values(cat_idx)
        if not indices:
            raise ValueError(
                "Unable to resolve a class index from class_name or cat_idx."
            )

        if len(indices) == 1:
            resolved = indices * batch_size
        elif len(indices) == batch_size:
            resolved = indices
        elif len(indices) % batch_size == 0:
            stride = len(indices) // batch_size
            resolved = indices[::stride]
        else:
            raise ValueError(
                f"Received {len(indices)} class indices for batch size "
                f"{batch_size}."
            )

        num_classes = self.fg_text_features.shape[0]
        if any(index < 0 or index >= num_classes for index in resolved):
            raise IndexError(
                f"Class indices must be in [0, {num_classes}), got {resolved}."
            )
        return torch.tensor(resolved, dtype=torch.long, device=device)

    def _build_text_visual_prior(
        self,
        query_clip_feature_layers,
        class_name,
        cat_idx,
        output_size,
    ):
        """[方案 D 纯净版] 复用 query CLIP 特征生成文本视觉先验图。"""
        if self.text_visual_pyramid is None:
            raise RuntimeError(
                "TextVisualSimilarityPyramidPure is not enabled."
            )

        if len(query_clip_feature_layers) <= max(CLIP_LAYER_IDS):
            raise ValueError(
                "query_clip_feature_layers does not contain all layers "
                f"required by scheme D: {CLIP_LAYER_IDS}."
            )

        # 原始 CLIP 特征列表包含全部 12 层，当前方案 D 选择人工编号
        # 第 10、11、12 层（零基索引 9、10、11）。
        query_pyramid = tuple(
            query_clip_feature_layers[layer_id] for layer_id in CLIP_LAYER_IDS
        )

        batch_size = query_pyramid[0].shape[0]
        class_indices = self._resolve_class_indices(
            class_name,
            cat_idx,
            batch_size,
            query_pyramid[0].device,
        )
        fg_text_features = self.fg_text_features.to(
            query_pyramid[0].device
        ).index_select(0, class_indices)
        bg_text_features = self.bg_text_features.to(
            query_pyramid[0].device
        ).index_select(
            0,
            class_indices,
        )

        # 文本特征沿用原模型初始化阶段已有的 fg/bg_text_features，不重新
        # 运行文本编码器。
        # 金字塔模块输出 [B, 1, H, W] 的前景先验图。
        return self.text_visual_pyramid(
            query_pyramid,
            fg_text_features,
            bg_text_features,
            output_size=output_size,
        )

    def forward(self, x, x_cv2, que_name, class_name, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        # ==================== 输入预处理 ====================
        # 将支持集标签 s_y 从 [b, n, h, w] 重排为 [(b*n), 1, h, w]，并将标签值 1 转为浮点数 1.0
        mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        mask = (mask == 1).float()
        h, w = x.shape[-2:]  # 获取查询图像的空间尺寸
        # 将支持集图像从 [b, n, c, h, w] 重排为 [(b*n), c, h, w]，合并 shot 维度以批量处理
        s_x = rearrange(s_x, "b n c h w -> (b n) c h w")

        # ==================== CNN 特征提取 ====================
        # 提取查询图像和支持图像的 CNN 多尺度特征（仅使用 stage 2~5）
        _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(x)
        _, _, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x, mask)

        # 拼接 stage 2 和 stage 3 的特征，并通过下采样投影层融合
        supp_feat_cnn = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat_cnn = self.down_supp(supp_feat_cnn)
        query_feat_cnn = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat_cnn = self.down_query(query_feat_cnn)

        # 根据配置选择低层特征（low_fea_id），用于后续 Gram 矩阵相似度计算
        supp_feat_item = eval('supp_feat_' + self.low_fea_id)
        supp_feat_item = rearrange(supp_feat_item, "(b n) c h w -> b n c h w", n=self.shot)
        # 按 shot 维度拆分为独立列表，便于逐样本处理
        supp_feat_list_ori = [supp_feat_item[:, i, ...] for i in range(self.shot)]

        # ==================== CLIP 特征提取 ====================
        # 文本视觉先验只需要 query CLIP 特征；MI-CoCluster VVP 还需要
        # support CLIP 特征。两个分支可以同时开启，共享同一次 query
        # CLIP 前向，避免重复提取 query 特征。
        # 这些分支没有可训练参数，因此显式关闭梯度以降低显存和计算开销。
        que_clip_feat_all = None
        supp_clip_feat_all = None
        if self.use_text_visual_pyramid or self.use_mi_cocluster_vvp:
            with torch.no_grad():
                tmp_que_clip_fts, _ = self.clip_model.encode_image(x, h, w, extract=True,)[:]

                patch_size = getattr(self.clip_model.visual, "patch_size", 16, )
                que_clip_feat_all = self._reshape_clip_feature_layers(
                    tmp_que_clip_fts,
                    h,
                    w,
                    patch_size=patch_size,
                )

                if self.use_mi_cocluster_vvp:
                    support_h, support_w = s_x.shape[-2:]
                    tmp_supp_clip_fts, _ = self.clip_model.encode_image(
                        s_x,
                        support_h,
                        support_w,
                        extract=True,
                    )[:]
                    supp_clip_feat_all = self._reshape_clip_feature_layers(
                        tmp_supp_clip_fts,
                        support_h,
                        support_w,
                        patch_size=patch_size,
                    )

        # ==================== VVP 相似度生成 ====================
        # MI-CoCluster VVP 与原 CNN VVP 二选一，不使用残差相加。
        if self.use_mi_cocluster_vvp:
            # 使用 CLIP 第 9、10 层 patch 特征生成 MI-VVP。
            similarity = self.mi_cocluster_vvp(
                query_layers=(
                    que_clip_feat_all[8],
                    que_clip_feat_all[9],
                ),
                support_layers=(
                    supp_clip_feat_all[8],
                    supp_clip_feat_all[9],
                ),
                support_mask=mask,
                output_size=query_feat_cnn.shape[-2:],
            )
        else:
            # 原 CNN stage-4/5 VVP 回退分支。
            similarity = self._build_cnn_vvp(
                query_feat_4,
                query_feat_5,
                supp_feat_4,
                supp_feat_5,
                mask,
            )

        # ====================================================================
        # 文本-视觉多粒度相似度金字塔
        # ====================================================================
        text_visual_prior = None
        if self.use_text_visual_pyramid:
            text_visual_prior = self._build_text_visual_prior(
                que_clip_feat_all,
                class_name,
                cat_idx,
                output_size=similarity.shape[-2:],
            )

        # ==================== 支持集特征处理 ====================
        # 对支持集 CNN 特征进行加权全局平均池化（以 mask 为权重），得到原型向量
        supp_pro = Weighted_GAP(supp_feat_cnn, \
                                F.interpolate(mask, size=(supp_feat_cnn.size(2), supp_feat_cnn.size(3)),
                                              mode='bilinear', align_corners=True))
        # 将原型向量扩展为空间特征图，与空间特征拼接
        supp_feat_bin = supp_pro.repeat(1, 1, supp_feat_cnn.shape[-2], supp_feat_cnn.shape[-1])
        # 融合 CNN 空间特征与原型特征，得到最终的支持集表示
        supp_feat = self.supp_merge(torch.cat([supp_feat_cnn, supp_feat_bin],
                                              dim=1))

        # ==================== K-Shot 重加权 ====================
        # 通过 Gram 矩阵差异度量各支持样本与查询样本之间的风格/纹理相似度
        bs = x.shape[0]
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id))
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list_ori:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            # 归一化的 Gram 差异作为该支持样本的估计权重值
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))
        est_val_total = torch.cat(est_val_list, 1)
        if self.shot > 1:
            # 多样本时：对差异值排序，通过可学习网络 kshot_rw 为每个样本分配权重，再经 softmax 归一化
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            # 单样本时：权重为 1，无需重加权
            weight_soft = torch.ones_like(est_val_total)
        # 加权求和得到最终的估计值，用于后续调节背景/前景图
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        # 将多 shot 的支持集原型特征取平均，得到单一的原型表示
        supp_feat_bin = rearrange(supp_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_bin = torch.mean(supp_feat_bin, dim=1)

        # 拼接查询特征、支持原型、VVP 相似度和文本视觉先验，送入 query_merge。
        # 原 VVP 相似度、支持原型和 CNN 特征的训练信号。
        query_merge_features = [query_feat_cnn, supp_feat_bin, similarity * 10]
        if text_visual_prior is not None:
            query_merge_features.append(
                (text_visual_prior * 10).to(dtype=query_feat_cnn.dtype)
            )
        query_feat = self.query_merge(
            torch.cat(query_merge_features, dim=1)
        )

        # ==================== Transformer 解码 & 基类分类器 ====================
        # Transformer 仅使用 VVP 相似度，VTP 只在 query_merge 中作为先验。
        meta_out, weights = self.transformer(
            query_feat,
            supp_feat,
            mask,
            similarity=similarity,
        )

        # 基类分类器使用最高层 CNN 特征预测基类概率
        base_out = self.base_learnear(query_feat_5)
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # ==================== BAM 风格背景融合 ====================
        # 参考 BAM (https://github.com/chunbolang/BAM) 的实现
        # 分离元学习分支的背景图和前景图
        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]
        if self.training and self.cls_type == 'Base':
            # 训练阶段且为基类：排除当前目标类，聚合其他基类的背景响应
            c_id_array = torch.arange(self.base_classes + 1).cuda()
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)
        else:
            # 推理阶段或新类：聚合所有基类的背景响应
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        map_h, map_w = meta_map_bg.shape[-2], meta_map_bg.shape[-1]
        base_map = F.interpolate(base_map, size=(map_h, map_w), mode='bilinear', align_corners=True)

        # 将 Gram 估计值扩展到与前景图相同的尺寸
        est_map = est_val.expand_as(meta_map_fg)

        # 将 Gram 估计值分别融入背景图和前景图，利用风格差异信息修正预测
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        # 融合元学习背景图和基类背景图，通过 cls_merge 生成最终背景预测
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

        # 拼接最终背景和元学习前景，得到完整的分割输出
        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # ==================== 输出上采样 ====================
        # 将所有输出上采样回原始输入图像尺寸
        meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # ==================== 损失计算 ====================
        if self.training:
            # 主损失：最终融合输出的交叉熵损失
            main_loss = self.criterion(final_out, y_m.long())
            # 辅助损失 1：元学习分支的交叉熵损失
            aux_loss1 = self.criterion(meta_out, y_m.long())
            # 辅助损失 2：基类分类器的交叉熵损失（使用基类标签 y_b）
            aux_loss2 = self.criterion(base_out, y_b.long())

            # 知识蒸馏损失：利用 Transformer 各层的注意力权重，逐层蒸馏到前一层的预测
            weight_t = (y_m == 1).float()
            weight_t = torch.masked_fill(weight_t, weight_t == 0, -1e9)
            for i, weight in enumerate(weights):
                if i == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                weight_t = weight.detach()

            return final_out.max(1)[1], main_loss + aux_loss1, distil_loss / 3, aux_loss2
        else:
            # 推理模式：返回最终输出、元学习输出和基类输出
            return final_out, meta_out, base_out

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss

    def get_optim(self, model, args, LR):
        parameter_groups = [
            {'params': model.transformer.mix_transformer.parameters()},
            {'params': model.supp_merge.parameters(), "lr": LR * 10},
            {'params': model.query_merge.parameters(), "lr": LR * 10},
            {'params': model.cls_merge.parameters(), "lr": LR * 10},
            {'params': model.down_supp.parameters(), "lr": LR * 10},
            {'params': model.down_query.parameters(), "lr": LR * 10},
            {'params': model.gram_merge.parameters(), "lr": LR * 10},
        ]
        if model.mi_cocluster_vvp is not None and getattr(
            args, "use_learnable_fusion", False
        ):
            parameter_groups.append(
                {'params': model.mi_cocluster_vvp.parameters()}
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=LR,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learnear.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results
