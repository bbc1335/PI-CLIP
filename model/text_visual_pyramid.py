# ============================================================================
# [方案 D 新文件] 多粒度文本-视觉相似度金字塔
#
# 本文件与 model/PI_CLIP.py 中的原有分割流程分离：
#   1. 本文件保存方案 D 的全部核心实现。
#   2. PI_CLIP.py 只负责创建本模块，并在原 forward 中调用接口。
#   3. 不修改原有 GradCAM、VVP、query_merge 和 Transformer 的内部实现。
#
# 对原项目已有 CLIP 特征提取代码的复用方式：
#   - PI_CLIP.py 的原 forward 已经分别对 support 和 query 执行一次 CLIP
#     视觉编码，并生成：
#       supp_clip_feat_all = 去掉 CLS 且恢复二维网格的 support 多层特征。
#       que_clip_feat_all  = 去掉 CLS 且恢复二维网格的 query 多层特征。
#   - 本文件不调用 clip_model.encode_image，也不重新处理原始图像。
#   - 本文件从上述两份特征中分别选择第 11、12 层，并复用原 s_y 掩码。
#
# 方案 D 的核心逻辑：
#   - 将 CLIP support 前景特征图保留为“视觉原型集合”，而不是压缩成
#     单个全局向量。
#   - 对每个 query patch，在 support 前景原型集合上做最大余弦匹配，
#     得到 support 视觉证据图。
#   - 使用该证据图校准 query patch 与前景/背景文本的相似度。
#   - 最后在 logit 空间融合第 11、12 层的校准结果，并只做一次
#     Sigmoid，输出单通道先验图。
# ============================================================================

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# 采用 CLIP 的两个深层特征：
#   10 -> 第 11 个 Transformer Block（零基索引），语义开始完整。
#   11 -> 第 12 个 Transformer Block（零基索引），语义最完整。
CLIP_LAYER_IDS = (10, 11)

# 融合层数必须与 CLIP_LAYER_IDS 保持一致。
NUM_PYRAMID_LEVELS = len(CLIP_LAYER_IDS)


class TextVisualSimilarityPyramid(nn.Module):
    """[方案 D 新增] 使用 support 视觉原型校准文本-query 相似度。

    处理流程：
        1. 对 query 和 support 的每一层特征做相同 LayerNorm 与 L2 归一化。
        2. 保留 support 前景 patch token，构成每层的视觉原型集合。
        3. 每个 query patch 与 support 原型集合做最大余弦匹配。
        4. 用 support 匹配证据校准“前景文本减背景文本”的相似度 logit。
        5. 在 logit 空间融合第 11、12 层，最后只做一次 Sigmoid。
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        projection_init: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        # [方案 D 新增] 文本投影层：将 CLIP 文本维度投影到视觉特征维度。
        # 初始权重由 PI_CLIP.py 使用 CLIP visual.proj 的伪逆转置传入，
        # 后续可以在训练中微调。
        self.text_projection = nn.Linear(text_dim, vision_dim, bias=False)
        if projection_init is None:
            nn.init.xavier_normal_(self.text_projection.weight)
        else:
            if projection_init.shape != self.text_projection.weight.shape:
                raise ValueError(
                    "projection_init must have shape "
                    f"{tuple(self.text_projection.weight.shape)}, "
                    f"got {tuple(projection_init.shape)}."
                )
            with torch.no_grad():
                self.text_projection.weight.copy_(
                    projection_init.detach().to(
                        device=self.text_projection.weight.device,
                        dtype=self.text_projection.weight.dtype,
                    )
                )

        # [方案 D 新增] 每一层使用独立的 LayerNorm。query 和 support 必须
        # 共用同一层 LayerNorm，才能保证二者的 patch 向量分布一致。
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(vision_dim) for _ in range(NUM_PYRAMID_LEVELS)]
        )

        # [方案 D 新增] 每一层一个可学习校准强度。使用 Softplus 保证权重
        # 始终为正，门控图为正时增强文本响应，为负时抑制文本响应。
        # 初始值约为 Softplus(-2.0) = 0.127，使训练初期只做轻微校准。
        self.calibration_logits = nn.Parameter(
            torch.full((NUM_PYRAMID_LEVELS,), -2.0)
        )

        # [D1 新增] 在 logit 空间进行卷积融合：
        #   [B, 2, H, W] -> Conv3x3 -> Conv1x1
        # 这里不加入中间激活，避免 ReLU 裁掉负 logit。最终只在前向传播
        # 返回前执行一次 Sigmoid。
        self.fusion = nn.Sequential(
            nn.Conv2d(
                NUM_PYRAMID_LEVELS,
                NUM_PYRAMID_LEVELS,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Conv2d(NUM_PYRAMID_LEVELS, 1, kernel_size=1, bias=True),
        )
        self._initialize_fusion()

    def _initialize_fusion(self) -> None:
        """[D1 新增] 将融合层初始化为两层校准 logit 的近似平均。"""
        conv3x3 = self.fusion[0]
        conv1x1 = self.fusion[1]
        with torch.no_grad():
            # 3x3 卷积初始只保留中心位置，并让每个通道保持自身响应。
            conv3x3.weight.zero_()
            conv3x3.weight[:, :, 1, 1] = 1.0
            conv3x3.bias.zero_()

            # 1x1 卷积将两个通道平均压缩为一个先验通道。
            conv1x1.weight.fill_(1.0 / NUM_PYRAMID_LEVELS)
            conv1x1.bias.zero_()

    def _project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """[方案 D 新增] 投影并 L2 归一化文本特征。"""
        if text_features.ndim != 2 or text_features.shape[-1] != self.text_dim:
            raise ValueError(
                f"Text features must have shape [B, {self.text_dim}], "
                f"got {tuple(text_features.shape)}."
            )
        text_features = text_features.to(
            device=self.text_projection.weight.device,
            dtype=self.text_projection.weight.dtype,
        )
        projected = self.text_projection(text_features)
        return F.normalize(projected, p=2, dim=-1, eps=1e-6)

    @staticmethod
    def _prepare_support_mask(
        support_mask: torch.Tensor,
        batch_size: int,
        shot: int,
    ) -> torch.Tensor:
        """[方案 D 新增] 将 support mask 统一为 [B*shot, 1, H, W]。

        原 PI_CLIP.py 中的 s_y 形状为 [B, shot, H, W]。本函数只在方案 D
        内部展平 shot 维度，不修改原 forward 中后续流程使用的 mask。
        """
        if support_mask.ndim == 4 and support_mask.shape[:2] == (
            batch_size,
            shot,
        ):
            support_mask = support_mask.reshape(
                batch_size * shot,
                1,
                *support_mask.shape[-2:],
            )
        elif support_mask.ndim == 4 and support_mask.shape[0] == batch_size * shot:
            # 兼容已经展开成 [B*shot, 1, H, W] 的 mask。
            support_mask = support_mask.reshape(
                batch_size * shot,
                1,
                *support_mask.shape[-2:],
            )
        elif support_mask.ndim == 3 and support_mask.shape[0] == batch_size * shot:
            # 兼容没有通道维度的 [B*shot, H, W] mask。
            support_mask = support_mask.unsqueeze(1)
        else:
            raise ValueError(
                "support_mask must have shape [B, shot, H, W], "
                "[B*shot, 1, H, W], or [B*shot, H, W], got "
                f"{tuple(support_mask.shape)}."
            )

        return (support_mask == 1).float()

    def _compute_support_evidence(
        self,
        query_tokens: torch.Tensor,
        support_patch_features: torch.Tensor,
        support_mask: torch.Tensor,
        level: int,
        batch_size: int,
        shot: int,
    ) -> torch.Tensor:
        """[方案 D 新增] 计算 query 与 support 前景原型集合的匹配图。

        输入:
            query_tokens:
                当前层已归一化的 query token，形状 [B, N_q, C]。
            support_patch_features:
                当前层 support 特征图，形状 [B*shot, C, h, w]。
            support_mask:
                已展平的二值 support mask，形状 [B*shot, 1, H, W]。

        输出:
            support_evidence:
                每个 query patch 的 support 视觉匹配分数，形状 [B, h, w]。

        关键设计：
            不把 support 前景特征平均成一个全局向量，而是保留全部前景
            token 作为视觉原型集合。每个 query patch 与全部有效 support
            前景 token 做余弦匹配，并直接取最大响应。最大值聚合不依赖
            温度超参数，同时保留与 query patch 最相似的原型位置。
        """
        expected_support_batch = batch_size * shot
        if support_patch_features.ndim != 4:
            raise ValueError(
                "support features must have shape [B*shot, C, h, w], got "
                f"{tuple(support_patch_features.shape)}."
            )
        if support_patch_features.shape[0] != expected_support_batch:
            raise ValueError(
                f"Support batch size is {support_patch_features.shape[0]}, "
                f"expected {expected_support_batch}."
            )
        if support_patch_features.shape[1] != self.vision_dim:
            raise ValueError(
                f"Support level {level} has "
                f"{support_patch_features.shape[1]} channels, expected "
                f"{self.vision_dim}."
            )

        # query 和 support 共用当前层的 LayerNorm，随后都进行 L2 归一化。
        support_tokens = support_patch_features.permute(0, 2, 3, 1).contiguous()
        support_tokens = support_tokens.to(
            device=self.layer_norms[level].weight.device,
            dtype=self.layer_norms[level].weight.dtype,
        )
        support_tokens = self.layer_norms[level](support_tokens)
        support_tokens = F.normalize(support_tokens, p=2, dim=-1, eps=1e-6)

        # [B*shot, h, w, C] -> [B, shot, N_s, C]。
        _, channels, grid_h, grid_w = support_patch_features.shape
        support_tokens = support_tokens.reshape(
            batch_size,
            shot,
            grid_h * grid_w,
            channels,
        )

        # 将原始分辨率的 support mask 对齐到当前层的 patch 网格。
        level_mask = F.interpolate(
            support_mask,
            size=(grid_h, grid_w),
            mode="nearest",
        )
        level_mask = (
            level_mask.reshape(batch_size, shot, grid_h * grid_w) > 0.5
        )

        support_evidence_per_shot = []
        for shot_index in range(shot):
            prototype_tokens = support_tokens[:, shot_index, :, :]
            prototype_mask = level_mask[:, shot_index, :]

            # query_tokens: [B, N_q, C]；prototype_tokens: [B, N_s, C]。
            # similarity: [B, N_q, N_s]，每个元素都是余弦相似度。
            similarity = torch.einsum(
                "bqc,bsc->bqs",
                query_tokens,
                prototype_tokens,
            )

            # 无效 support 位置不参与最大匹配。若某个样本没有任何前景
            # token，下面的 has_foreground 会把结果置零，保证数值稳定。
            masked_similarity = similarity.masked_fill(
                ~prototype_mask.unsqueeze(1),
                torch.finfo(similarity.dtype).min,
            )
            shot_evidence = masked_similarity.max(
                dim=-1,
            ).values
            has_foreground = prototype_mask.any(dim=-1, keepdim=True).float()
            shot_evidence = shot_evidence * has_foreground
            support_evidence_per_shot.append(shot_evidence)

        # 多 shot 先分别匹配，再求平均，避免提前平均 support 视觉特征。
        support_evidence = torch.stack(
            support_evidence_per_shot,
            dim=1,
        ).mean(dim=1)
        return support_evidence.reshape(batch_size, grid_h, grid_w)

    def forward(
        self,
        visual_features: Sequence[torch.Tensor],
        support_features: Sequence[torch.Tensor],
        support_mask: torch.Tensor,
        fg_text_features: torch.Tensor,
        bg_text_features: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """生成单通道的 support 校准文本视觉先验图。

        visual_features:
            CLIP 第 11、12 层的 query 特征，每层形状为
            [B, C, H/16, W/16]。
        support_features:
            CLIP 第 11、12 层的 support 特征，每层形状为
            [B*shot, C, H/16, W/16]。
        support_mask:
            原始 support 掩码，形状优先为 [B, shot, H, W]。
        fg_text_features:
            目标类别的前景文本特征，形状 [B, text_dim]。
        bg_text_features:
            目标类别的背景文本特征，形状 [B, text_dim]。
        output_size:
            最终先验图需要对齐到的 CNN 特征尺寸 (H, W)。
        """
        if len(visual_features) != NUM_PYRAMID_LEVELS:
            raise ValueError(
                f"Expected {NUM_PYRAMID_LEVELS} query feature levels, "
                f"got {len(visual_features)}."
            )
        if len(support_features) != NUM_PYRAMID_LEVELS:
            raise ValueError(
                f"Expected {NUM_PYRAMID_LEVELS} support feature levels, "
                f"got {len(support_features)}."
            )
        if fg_text_features.shape != bg_text_features.shape:
            raise ValueError(
                "Foreground and background text features must have the same shape."
            )

        batch_size = visual_features[0].shape[0]
        support_batch_size = support_features[0].shape[0]
        if batch_size <= 0 or support_batch_size % batch_size != 0:
            raise ValueError(
                "Support batch size must be a positive integer multiple of "
                f"query batch size, got {support_batch_size} and {batch_size}."
            )
        shot = support_batch_size // batch_size
        support_mask = self._prepare_support_mask(
            support_mask,
            batch_size,
            shot,
        )

        fg_text_features = self._project_text(fg_text_features)
        bg_text_features = self._project_text(bg_text_features)
        calibrated_logit_maps = []

        for level, (patch_features, support_patch_features) in enumerate(
            zip(visual_features, support_features)
        ):
            if patch_features.ndim != 4:
                raise ValueError(
                    f"Query level {level} must have shape [B, C, H, W], "
                    f"got {tuple(patch_features.shape)}."
                )
            if patch_features.shape[0] != fg_text_features.shape[0]:
                raise ValueError(
                    f"Query level {level} batch size does not match text "
                    "features."
                )
            if patch_features.shape[1] != self.vision_dim:
                raise ValueError(
                    f"Query level {level} has {patch_features.shape[1]} "
                    f"channels, expected {self.vision_dim}."
                )

            # LayerNorm 在最后一维进行，因此先转换为 [B, H, W, C]。
            patch_tokens = patch_features.permute(0, 2, 3, 1).contiguous()
            patch_tokens = patch_tokens.to(
                device=self.layer_norms[level].weight.device,
                dtype=self.layer_norms[level].weight.dtype,
            )
            patch_tokens = self.layer_norms[level](patch_tokens)
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1, eps=1e-6)
            grid_h, grid_w = patch_tokens.shape[1:3]
            query_tokens = patch_tokens.reshape(
                batch_size,
                grid_h * grid_w,
                self.vision_dim,
            )

            # 1. 使用 support 前景特征图生成 query 的视觉匹配证据。
            support_evidence = self._compute_support_evidence(
                query_tokens,
                support_patch_features,
                support_mask,
                level,
                batch_size,
                shot,
            )

            # 2. 计算 query patch 的文本判别 logit：
            #    cos(query, foreground_text) - cos(query, background_text)。
            fg_similarity = torch.einsum(
                "bqc,bc->bq",
                query_tokens,
                fg_text_features,
            )
            bg_similarity = torch.einsum(
                "bqc,bc->bq",
                query_tokens,
                bg_text_features,
            )
            text_logits = (fg_similarity - bg_similarity).reshape(
                batch_size,
                grid_h,
                grid_w,
            )

            # 3. 对当前层的 support 证据做逐样本标准化。标准化消除不同
            #    样本间相似度尺度差异，tanh 再生成有界且正负对称的门控。
            evidence_mean = support_evidence.mean(
                dim=(-2, -1),
                keepdim=True,
            )
            evidence_std = support_evidence.std(
                dim=(-2, -1),
                keepdim=True,
                unbiased=False,
            )
            centered_evidence = (
                support_evidence - evidence_mean
            ) / (evidence_std + 1e-6)
            calibration_gate = torch.tanh(centered_evidence)

            # 4. 使用 Softplus 后的正校准强度残差调整文本 logit。
            calibration_strength = F.softplus(
                self.calibration_logits[level]
            )
            calibrated_logits = (
                text_logits + calibration_strength * calibration_gate
            )
            calibrated_logit_map = calibrated_logits.unsqueeze(1)

            # 5. 保留 logit，不提前做 Sigmoid。不同层进入融合前只统一尺寸。
            if calibrated_logit_map.shape[-2:] != output_size:
                calibrated_logit_map = F.interpolate(
                    calibrated_logit_map,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            calibrated_logit_maps.append(calibrated_logit_map)

        # [B, 1, H, W] x 2 -> [B, 2, H, W] -> [B, 1, H, W] logits。
        # D1 整个模块只在融合完成后执行一次 Sigmoid。
        fused_logits = self.fusion(
            torch.cat(calibrated_logit_maps, dim=1)
        )
        return torch.sigmoid(fused_logits)
