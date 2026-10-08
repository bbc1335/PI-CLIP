# ============================================================================
# 文本-视觉多粒度相似度先验
# CLIP 特征复用方式：
#   PI_CLIP.py 的 forward 已经调用 encode_image(..., extract=True) 取得
#   query 的多层 CLIP patch 特征。本文件从该结果中读取第 10、11、12 层
#   （Python 零基索引 9、10、11），不会再次运行 CLIP 图像编码器。
# ============================================================================

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# 必须与 PI_CLIP.py 中传入的 query 特征层顺序一致。
CLIP_LAYER_IDS = (9, 10, 11)
NUM_PYRAMID_LEVELS = len(CLIP_LAYER_IDS)


class TextVisualSimilarityPyramidPure(nn.Module):
    """固定权重的文本-query 多粒度相似度金字塔。

    每个尺度上的计算步骤：
        1. 对 query patch token 做 LayerNorm，使不同 CLIP Block 的特征尺度
           尽量一致；
        2. 对 query token 和文本向量做 L2 归一化；
        3. 分别计算 query 与前景文本、背景文本的余弦相似度；
        4. 使用 S_text = cos(query, foreground) - cos(query, background)，
           得到具有正负区分的文本 logit；
        5. 对每一层 S_text 做空间标准化，消除不同层和不同样本之间的均值、
           方差差异；
        6. 将各层相似度图上采样到统一尺寸，再用固定的 1x1 卷积做等权平均；
        7. 最后只执行一次 Sigmoid，输出范围 [0, 1] 的前景先验图。

    数学形式：
        q_l,i_norm = normalize(LayerNorm(q_l,i))
        t_fg_norm  = normalize(P @ t_fg)
        t_bg_norm  = normalize(P @ t_bg)

        s_l,i = q_l,i_norm · t_fg_norm - q_l,i_norm · t_bg_norm
        z_l,i = (s_l,i - mean(s_l)) / (std(s_l) + eps)
        M     = sigmoid((1 / L) * sum_l z_l)

    其中 l 表示层，i 表示 query patch，P 是固定的文本到视觉维度投影。
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

        if projection_init is None:
            # 独立测试模块时才使用随机投影。正式接入 PI-CLIP 时，必须传入
            # CLIP visual.proj 的伪逆投影，否则中间层特征和文本空间不对齐。
            projection = torch.randn(vision_dim, text_dim)
        else:
            projection = projection_init.detach().float().clone()

        expected_shape = (vision_dim, text_dim)
        if tuple(projection.shape) != expected_shape:
            raise ValueError(
                f"projection_init must have shape {expected_shape}, "
                f"got {tuple(projection.shape)}."
            )

        # 使用 buffer 而不是 Parameter，表示该投影只参与前向计算，不参与训练。
        # 这里保存的形状是 [vision_dim, text_dim]，对应 F.linear 的权重布局。
        self.register_buffer(
            "text_projection",
            projection,
            persistent=True,
        )

        # 融合方式仍然是卷积融合，但权重固定为 K 个尺度的等权平均：
        #   M = Conv1x1([S_10, S_11, S_12])
        # 1x1 卷积不是简单拼接后又接一个全连接，而是保持二维空间结构，
        # 因此可以对该通道融合结果直接做 Sigmoid 得到热力图。
        self.fusion = nn.Conv2d(
            NUM_PYRAMID_LEVELS,
            1,
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.fusion.weight.fill_(1.0 / NUM_PYRAMID_LEVELS)
        self.fusion.weight.requires_grad_(False)

    def _project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """将 CLIP 文本特征投影到视觉特征维度，并做 L2 归一化。"""
        if text_features.ndim != 2 or text_features.shape[-1] != self.text_dim:
            raise ValueError(
                f"Text features must have shape [B, {self.text_dim}], "
                f"got {tuple(text_features.shape)}."
            )

        text_features = text_features.to(
            device=self.text_projection.device,
            dtype=self.text_projection.dtype,
        )

        # F.linear(text, weight) 等价于 text @ weight.T。
        # 文本由 CLIP 的 512 维空间映射到 ViT 中间层的 768 维空间。
        projected = F.linear(text_features, self.text_projection)
        return F.normalize(projected, p=2, dim=-1, eps=1e-6)

    def _compute_layer_similarity(
        self,
        patch_features: torch.Tensor,
        foreground_text: torch.Tensor,
        background_text: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """计算一个 CLIP 层的文本判别相似度图。"""
        if patch_features.ndim != 4:
            raise ValueError(
                f"Query level {level} must have shape [B, C, H, W], "
                f"got {tuple(patch_features.shape)}."
            )
        if patch_features.shape[1] != self.vision_dim:
            raise ValueError(
                f"Query level {level} has {patch_features.shape[1]} channels, "
                f"expected {self.vision_dim}."
            )
        if patch_features.shape[0] != foreground_text.shape[0]:
            raise ValueError(
                "Query features and text features must have the same batch size."
            )

        batch_size, _, grid_h, grid_w = patch_features.shape

        # [B, C, h, w] -> [B, h, w, C]，因为 LayerNorm 在最后一维执行。
        patch_tokens = patch_features.permute(0, 2, 3, 1).to(
            device=self.text_projection.device,
            dtype=self.text_projection.dtype,
        )

        # 每个 patch 独立地在通道维做归一化。这里不设置可学习仿射参数，
        # 保证不同深度的 CLIP 特征只被校准尺度，不引入新的训练参数。
        patch_tokens = F.layer_norm(
            patch_tokens,
            normalized_shape=(self.vision_dim,),
            eps=1e-6,
        )
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1, eps=1e-6)

        # 展平为 [B, N, C]，随后与文本向量做逐 patch 余弦相似度。
        query_tokens = patch_tokens.reshape(
            batch_size,
            grid_h * grid_w,
            self.vision_dim,
        )

        # 余弦相似度，不是点积。F.normalize 后，点积的结果就是余弦值。
        foreground_similarity = torch.einsum(
            "bqc,bc->bq",
            query_tokens,
            foreground_text,
        )
        background_similarity = torch.einsum(
            "bqc,bc->bq",
            query_tokens,
            background_text,
        )

        # 前景相似度减背景相似度，构成有正负方向的文本 logit。
        text_logits = (
            foreground_similarity - background_similarity
        ).reshape(batch_size, grid_h, grid_w)

        # 对每个样本、每个尺度独立做空间标准化。这样不会把某一层固有的
        # 相似度均值差异直接传入卷积融合，也不需要额外的 temperature。
        logit_mean = text_logits.mean(dim=(-2, -1), keepdim=True)
        logit_std = text_logits.std(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False,
        )
        return (text_logits - logit_mean) / (logit_std + 1e-6)

    @torch.no_grad()
    def forward(
        self,
        query_features: Sequence[torch.Tensor],
        fg_text_features: torch.Tensor,
        bg_text_features: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """生成 [B, 1, output_h, output_w] 的文本视觉先验图。

        参数：
            query_features:
                CLIP 第 10、11、12 层的 query patch 特征，顺序必须与
                CLIP_LAYER_IDS=(9, 10, 11) 一致。每层形状为
                [B, vision_dim, h, w]。
            fg_text_features:
                当前批次每个查询图像对应的前景文本特征，形状
                [B, text_dim]。
            bg_text_features:
                当前批次每个查询图像对应的背景文本特征，形状
                [B, text_dim]。
            output_size:
                最终融合图需要对齐到的 CNN 特征尺寸 (H, W)。

        返回值：
            单通道先验图，形状 [B, 1, output_h, output_w]，值域为 [0, 1]。
        """
        if len(query_features) != NUM_PYRAMID_LEVELS:
            raise ValueError(
                f"Expected {NUM_PYRAMID_LEVELS} query feature levels, "
                f"got {len(query_features)}."
            )
        if fg_text_features.shape != bg_text_features.shape:
            raise ValueError(
                "Foreground and background text features must have the "
                "same shape."
            )
        if len(output_size) != 2:
            raise ValueError(
                f"output_size must contain (H, W), got {output_size}."
            )

        foreground_text = self._project_text(fg_text_features)
        background_text = self._project_text(bg_text_features)
        similarity_maps = []

        for level, patch_features in enumerate(query_features):
            layer_similarity = self._compute_layer_similarity(
                patch_features,
                foreground_text,
                background_text,
                level,
            ).unsqueeze(1)

            # 各 CLIP Block 的 patch 网格通常已经是同一分辨率；这里仍做一次
            # 尺寸对齐，使接口对任意 ViT patch 网格保持稳健。
            if layer_similarity.shape[-2:] != output_size:
                layer_similarity = F.interpolate(
                    layer_similarity,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            similarity_maps.append(layer_similarity)

        # [B, 1, H, W] * K -> [B, K, H, W]
        pyramid = torch.cat(similarity_maps, dim=1)

        # 固定 1x1 卷积将 K 个尺度等权融合为一个通道。
        fused_prior = self.fusion(pyramid)

        # 所有层在 logit/similarity 空间完成融合后，只做一次概率映射。
        return torch.sigmoid(fused_prior)
