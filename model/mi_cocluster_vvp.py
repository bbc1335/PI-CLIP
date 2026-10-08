# ============================================================================
# [MI-CoCluster VVP] CLIP 第 9、10 层 MI-CoCluster 视觉相似度
#
# 该模块根据 CLIP 支持集/查询集 patch 特征生成 MI-inspired VVP。
# 在当前 PI-CLIP 调用中，输入为 CLIP extract=True 得到的第 9、10 层
# （Python 零基索引 8、9）patch 特征；残差模式下会与原 CNN VVP 相加。
#
# 默认使用固定的 C/U 等权融合，可选启用按层学习的凸组合。
# ============================================================================

import torch
import torch.nn.functional as F
from torch import nn


class MICoClusterVVP(nn.Module):
    """基于 CLIP 第 9、10 层特征的 MI-CoCluster VVP。"""

    def __init__(self, shot=1, rank=8, num_iters=4,
                 use_learnable_fusion=False):
        super().__init__()
        self.shot = int(shot)
        self.rank = int(rank)
        self.num_iters = int(num_iters)
        self.use_learnable_fusion = bool(use_learnable_fusion)

        # 四通道顺序为 [C_10, U_10, C_9, U_9]。
        # 默认固定 alpha=0.5；开启后可学习 alpha=sigmoid(fusion_logit)。
        if self.use_learnable_fusion:
            self.fusion_logit = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer(
                "fusion_alpha",
                torch.full((2,), 0.5),
                persistent=False,
            )

    @staticmethod
    def _normalize_patch_features(feature):
        """将 [C, H, W] CLIP patch 特征转换为 L2 归一化 token。"""
        channels, height, width = feature.shape
        tokens = feature.permute(1, 2, 0).reshape(height * width, channels)
        tokens = F.layer_norm(tokens, (channels,), eps=1e-6)
        tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
        return tokens, height, width

    @staticmethod
    def _foreground_indices(support_mask, support_h, support_w):
        """将支持 mask 对齐到 CLIP patch 网格并返回前景索引。"""
        if support_mask.dim() == 2:
            support_mask = support_mask[None, None]
        elif support_mask.dim() == 3:
            support_mask = support_mask[None]
        elif support_mask.dim() != 4:
            raise ValueError(
                "support_mask must have shape [H, W], [1, H, W], "
                "or [B, 1, H, W]."
            )

        support_mask = F.interpolate(
            support_mask.float(),
            size=(support_h, support_w),
            mode="nearest",
        )
        foreground_indices = torch.nonzero(
            support_mask.reshape(-1) > 0.5,
            as_tuple=False,
        ).reshape(-1)

        # mask 正常情况下不会为空。这里做保护，避免异常样本导致前向崩溃。
        if foreground_indices.numel() == 0:
            foreground_indices = torch.arange(
                support_h * support_w,
                device=support_mask.device,
            )
        return foreground_indices

    def _nmf_reconstruct(self, affinity):
        """使用乘法更新近似求解 A ≈ U G V。"""
        eps = 1e-6
        affinity = affinity.float().clamp_min(eps)
        num_query, num_support = affinity.shape
        rank = min(self.rank, num_query, num_support)
        if rank <= 0:
            return affinity

        # SVD 初始化是确定性的，避免随机初始化造成聚类结果每步变化。
        try:
            u_svd, s_svd, v_svd = torch.linalg.svd(
                affinity,
                full_matrices=False,
            )
            u = u_svd[:, :rank].abs() + eps
            g = torch.diag(s_svd[:rank].clamp_min(eps))
            v = v_svd[:rank, :].abs() + eps
        except RuntimeError:
            u = torch.ones(
                num_query,
                rank,
                dtype=affinity.dtype,
                device=affinity.device,
            )
            g = torch.eye(
                rank,
                dtype=affinity.dtype,
                device=affinity.device,
            )
            v = torch.ones(
                rank,
                num_support,
                dtype=affinity.dtype,
                device=affinity.device,
            )

        for _ in range(self.num_iters):
            numerator_u = affinity @ v.t() @ g.t()
            denominator_u = u @ g @ v @ v.t() @ g.t() + eps
            u = (u * numerator_u / denominator_u).clamp_min(eps)

            numerator_g = u.t() @ affinity @ v.t()
            denominator_g = u.t() @ u @ g @ v @ v.t() + eps
            g = (g * numerator_g / denominator_g).clamp_min(eps)

            numerator_v = g.t() @ u.t() @ affinity
            denominator_v = g.t() @ u.t() @ u @ g @ v + eps
            v = (v * numerator_v / denominator_v).clamp_min(eps)

        return (u @ g @ v).to(dtype=affinity.dtype)

    @torch.no_grad()
    def _build_pair(self, q11, q12, s11, s12, support_mask):
        """构建单个 shot 的 [C_10, U_10, C_9, U_9] 四通道图。"""
        if q11.shape[-2:] != q12.shape[-2:]:
            raise ValueError("CLIP layers 9 and 10 must share spatial size.")
        if s11.shape[-2:] != s12.shape[-2:]:
            raise ValueError(
                "Support CLIP layers 9 and 10 must share spatial size."
            )

        q11_tokens, query_h, query_w = self._normalize_patch_features(q11)
        q12_tokens, _, _ = self._normalize_patch_features(q12)
        s11_tokens, support_h, support_w = self._normalize_patch_features(s11)
        s12_tokens, _, _ = self._normalize_patch_features(s12)

        foreground_indices = self._foreground_indices(
            support_mask,
            support_h,
            support_w,
        )
        s11_foreground = s11_tokens[foreground_indices]
        s12_foreground = s12_tokens[foreground_indices]

        # L2 归一化后的矩阵乘法等价于余弦相似度。
        cosine11 = q11_tokens @ s11_foreground.t()
        cosine12 = q12_tokens @ s12_foreground.t()

        # 原始 VVP 部分：每个 query patch 在支持前景中取最大余弦。
        c11 = cosine11.max(dim=1).values.reshape(query_h, query_w)
        c12 = cosine12.max(dim=1).values.reshape(query_h, query_w)

        # NMF 要求非负输入，将余弦范围 [-1, 1] 映射到 [0, 1]。
        affinity11 = ((cosine11 + 1.0) * 0.5).clamp_min(1e-6)
        affinity12 = ((cosine12 + 1.0) * 0.5).clamp_min(1e-6)
        reconstruction11 = self._nmf_reconstruct(affinity11)
        reconstruction12 = self._nmf_reconstruct(affinity12)

        # 支持前景列等权聚合，避免硬选择一个目标簇造成物体区域缺失。
        u11 = reconstruction11.mean(dim=1).reshape(query_h, query_w)
        u12 = reconstruction12.mean(dim=1).reshape(query_h, query_w)
        u11 = torch.tanh(
            (u11 - u11.median()) / (u11.std(unbiased=False) + 1e-6)
        )
        u12 = torch.tanh(
            (u12 - u12.median()) / (u12.std(unbiased=False) + 1e-6)
        )

        return torch.stack([c12, u12, c11, u11], dim=0)

    def forward(self, query_layers, support_layers, support_mask, output_size):
        """
        query_layers:
            (CLIP index 8, CLIP index 9)，对应人工编号第 9、10 层，
            形状分别为 [B, C, H, W]。
        support_layers:
            (CLIP index 8, CLIP index 9)，对应人工编号第 9、10 层，
            形状分别为 [B * shot, C, H, W]。
        support_mask:
            形状为 [B * shot, 1, H, W] 或 [B * shot, H, W]。
        output_size:
            最终对齐到 query_feat_cnn 的空间尺寸。
        """
        if len(query_layers) != 2 or len(support_layers) != 2:
            raise ValueError("query_layers and support_layers must each have 2 items.")

        q11, q12 = query_layers
        s11, s12 = support_layers
        batch_size = q11.shape[0]
        expected_support = batch_size * self.shot
        if s11.shape[0] != expected_support:
            raise ValueError(
                f"support batch must be {expected_support}, got {s11.shape[0]}."
            )

        shot_outputs = []
        for shot_index in range(self.shot):
            pair_list = []
            for batch_index in range(batch_size):
                support_index = batch_index * self.shot + shot_index
                pair_list.append(
                    self._build_pair(
                        q11=q11[batch_index],
                        q12=q12[batch_index],
                        s11=s11[support_index],
                        s12=s12[support_index],
                        support_mask=support_mask[support_index],
                    )
                )

            pairs = torch.stack(pair_list, dim=0)
            if self.use_learnable_fusion:
                alpha = torch.sigmoid(self.fusion_logit)
            else:
                alpha = self.fusion_alpha
            alpha = alpha.to(dtype=pairs.dtype).view(1, 2, 1, 1)
            cosine = torch.stack([pairs[:, 0], pairs[:, 2]], dim=1)
            cocluster = torch.stack([pairs[:, 1], pairs[:, 3]], dim=1)
            shot_outputs.append(
                (1.0 - alpha) * cosine + alpha * cocluster
            )

        shot_outputs = [
            F.interpolate(
                item,
                size=output_size,
                mode="bilinear",
                align_corners=True,
            )
            for item in shot_outputs
        ]

        # 保持原 VVP 的通道顺序：每个 shot 的第二个输入层在前，
        # 第一个输入层在后。当前调用使用零基索引 8、9，因此顺序为
        # [第 10 层 shot1, ..., 第 10 层 shotK,
        #  第 9 层 shot1, ..., 第 9 层 shotK]。
        # 下面的 layer12/layer11 变量名是历史命名，保留以兼容原有接口。
        layer12 = torch.cat([item[:, 0:1] for item in shot_outputs], dim=1)
        layer11 = torch.cat([item[:, 1:2] for item in shot_outputs], dim=1)
        return torch.cat([layer12, layer11], dim=1)
