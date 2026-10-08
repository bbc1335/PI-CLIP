# ============================================================================
# [MI-CoCluster VVP] CLIP 第 11、12 层 MI-CoCluster 视觉相似度
#
# 该模块用于替换原 VVP 中 stage 4/5 CNN 特征的余弦最大相似度。
# 输入为 CLIP extract=True 得到的第 11、12 层 patch 特征。
#
# [Cluster-Max C 修改] 原始 C 只取支持集前景中的单个最大余弦响应，
# Top-k 平均又会把多个不同语义部件的响应混在一起。当前实现先用
# NMF 的系数矩阵 V 对支持前景 patch 做硬分簇，再在每个簇内部取最大
# 余弦响应，最后对所有非空簇求平均：
#   C_i = mean_r( max_{j in cluster_r} cos(q_i, s_j) )
#
# 这样既能保留多部件物体的局部判别能力，又能避免单个异常 patch
# 主导整张相似度图。旧的 Top-k 实现保留在下方注释中，仅用于消融实验。
#
# 为兼容已经训练好的文本-视觉金字塔权重，这里的融合卷积使用
# persistent=False 的固定 buffer，不向 state_dict 增加任何参数。
# 因此不会改变原模型结构的可加载性。
# ============================================================================

import torch
import torch.nn.functional as F
from torch import nn


class MICoClusterVVP(nn.Module):
    """基于 CLIP 第 11、12 层特征的 MI-CoCluster VVP。"""

    def __init__(self, shot=1, rank=8, num_iters=4, top_k=5):
        super().__init__()
        self.shot = int(shot)
        self.rank = int(rank)
        self.num_iters = int(num_iters)
        # top_k 仅保留给下方注释掉的旧 Top-k 消融实现；当前 Cluster-Max
        # 前向不使用这个值，也不会把它写入 state_dict。
        self.top_k = max(1, int(top_k))

        # 四通道顺序为 [C_12, U_12, C_11, U_11]。
        # 使用固定的 1x1 卷积把每一层的“Cluster-Max 余弦响应 C”和
        # “NMF 共聚类响应 U”做等权融合：
        #   output 0 = 0.5 * C_12 + 0.5 * U_12
        #   output 1 = 0.5 * C_11 + 0.5 * U_11
        #
        # 这里故意使用固定权重，而不是新增可训练卷积。这样 U 会真正参与
        # VVP 计算，同时该 buffer 不进入 state_dict，旧检查点可以继续加载。
        # persistent=False 保证该 buffer 不进入 state_dict，旧权重不受影响。
        fusion_weight = torch.zeros(2, 4, 1, 1)
        fusion_weight[0, 0, 0, 0] = 0.5
        fusion_weight[0, 1, 0, 0] = 0.5
        fusion_weight[1, 2, 0, 0] = 0.5
        fusion_weight[1, 3, 0, 0] = 0.5
        self.register_buffer(
            "fusion_weight",
            fusion_weight,
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

    # ------------------------------------------------------------------
    # [旧 Top-k C 实现，保留但不参与当前前向]
    # 仅用于验证 Cluster-Max 与 Top-k 聚类的差异，避免丢失消融实验代码。
    # ------------------------------------------------------------------
    # def _topk_mean_similarity(self, cosine_similarity):
    #     """对每个 query patch 取 Top-k 支持前景余弦响应后求平均。"""
    #     num_support = cosine_similarity.shape[1]
    #     if num_support == 0:
    #         raise ValueError(
    #             "cosine_similarity must contain at least one support patch."
    #         )
    #     current_k = min(self.top_k, num_support)
    #     return cosine_similarity.topk(
    #         k=current_k,
    #         dim=1,
    #         largest=True,
    #         sorted=False,
    #     ).values.mean(dim=1)

    @staticmethod
    def _cluster_max_similarity(cosine_similarity, cluster_assignment):
        """按支持 patch 分簇，在每个簇内取最大余弦后求平均。

        Args:
            cosine_similarity: [N_query, N_support_foreground] 的余弦矩阵。
            cluster_assignment: [N_support_foreground] 的整数簇编号。

        Returns:
            [N_query] 的 Cluster-Max 响应。

        说明：
            这里只对“非空簇”求平均。某些 NMF 簇可能在 argmax 硬分配后
            没有获得任何支持 patch，此时不能把空簇当作零响应加入平均，
            否则会人为压低查询侧响应。
        """
        if cosine_similarity.dim() != 2:
            raise ValueError(
                "cosine_similarity must have shape [N_query, N_support]."
            )
        if cluster_assignment.dim() != 1:
            raise ValueError(
                "cluster_assignment must have shape [N_support]."
            )
        if cosine_similarity.shape[1] == 0:
            raise ValueError(
                "cosine_similarity must contain at least one support patch."
            )
        if cosine_similarity.shape[1] != cluster_assignment.shape[0]:
            raise ValueError(
                "The number of support columns and cluster assignments "
                "must match."
            )

        cluster_ids = torch.unique(cluster_assignment)
        if cluster_ids.numel() == 0:
            raise ValueError("cluster_assignment must contain at least one item.")

        cluster_responses = []
        for cluster_id in cluster_ids:
            cluster_indices = torch.nonzero(
                cluster_assignment == cluster_id,
                as_tuple=False,
            ).reshape(-1)
            cluster_cosine = cosine_similarity.index_select(
                1,
                cluster_indices,
            )
            # 每个 query patch 在该簇内部只保留最强响应；簇内最大值能
            # 保留多部件物体的局部证据，同时不会被其他簇的 patch 干扰。
            cluster_responses.append(cluster_cosine.max(dim=1).values)

        return torch.stack(cluster_responses, dim=0).mean(dim=0)

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

    def _nmf_factors(self, affinity):
        """使用乘法更新近似求解 A ≈ U G V，并返回三个非负因子。

        返回的 ``v`` 是支持 patch 到 NMF 簇的系数矩阵，形状为
        ``[rank, N_support]``。后续 Cluster-Max 使用 ``v.argmax(dim=0)``
        得到每个支持前景 patch 的硬簇编号。
        """
        eps = 1e-6
        affinity = affinity.float().clamp_min(eps)
        num_query, num_support = affinity.shape
        rank = min(self.rank, num_query, num_support)
        if rank <= 0:
            raise ValueError(
                "NMF requires at least one query patch and one support patch."
            )

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

        return u, g, v

    def _nmf_reconstruct(self, affinity):
        """兼容旧接口：只返回 NMF 重构矩阵 A_hat ≈ U G V。"""
        u, g, v = self._nmf_factors(affinity)
        return (u @ g @ v).to(dtype=affinity.dtype)

    @torch.no_grad()
    def _build_pair(self, q11, q12, s11, s12, support_mask):
        """构建单个 shot 的 [C_12, U_12, C_11, U_11] 四通道图。"""
        if q11.shape[-2:] != q12.shape[-2:]:
            raise ValueError("CLIP layer 11 and layer 12 must share spatial size.")
        if s11.shape[-2:] != s12.shape[-2:]:
            raise ValueError(
                "Support CLIP layer 11 and 12 must share spatial size."
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

        # NMF 要求非负输入，将余弦范围 [-1, 1] 映射到 [0, 1]。
        affinity11 = ((cosine11 + 1.0) * 0.5).clamp_min(1e-6)
        affinity12 = ((cosine12 + 1.0) * 0.5).clamp_min(1e-6)

        # 先分解得到支持侧系数矩阵 V，再根据 V 的 argmax 做支持前景
        # patch 的硬分簇。第 11、12 层分别保留自己的簇结构，避免不同
        # 语义层级之间的 patch 被强行共享同一个簇编号。
        u11_factor, g11, v11 = self._nmf_factors(affinity11)
        u12_factor, g12, v12 = self._nmf_factors(affinity12)
        cluster11 = v11.argmax(dim=0)
        cluster12 = v12.argmax(dim=0)

        # [Cluster-Max C] 每个 query patch 先在每个支持簇内取最大余弦，
        # 再对所有非空簇求平均，避免 Top-k 平均把不同部件混在一起。
        c11 = self._cluster_max_similarity(
            cosine11,
            cluster11,
        ).reshape(query_h, query_w)
        c12 = self._cluster_max_similarity(
            cosine12,
            cluster12,
        ).reshape(query_h, query_w)

        reconstruction11 = (u11_factor @ g11 @ v11).to(
            dtype=affinity11.dtype
        )
        reconstruction12 = (u12_factor @ g12 @ v12).to(
            dtype=affinity12.dtype
        )

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
            (CLIP index 10, CLIP index 11)，形状分别为 [B, C, H, W]。
        support_layers:
            (CLIP index 10, CLIP index 11)，形状分别为 [B * shot, C, H, W]。
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
            shot_outputs.append(
                F.conv2d(pairs, self.fusion_weight.to(dtype=pairs.dtype))
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

        # 保持原 VVP 的通道顺序：
        # [layer12_shot1, ..., layer12_shotK,
        #  layer11_shot1, ..., layer11_shotK]
        layer12 = torch.cat([item[:, 0:1] for item in shot_outputs], dim=1)
        layer11 = torch.cat([item[:, 1:2] for item in shot_outputs], dim=1)
        return torch.cat([layer12, layer11], dim=1)
