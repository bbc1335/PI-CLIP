import numpy as np
import torch
import tqdm
from typing import Callable, List
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import replace_layer_recursive
from pytorch_grad_cam.ablation_layer import AblationLayer


"""
AblationCAM 实现
论文：Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization (WACV 2020)
链接：https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf

核心思想：
  逐个通道清零（消融），观察目标类别分数下降了多少。
  下降越多 → 该通道越重要 → 权重越大。
  
  公式：α^c_k = (y^c_original - y^c_ablated(k)) / y^c_original

与 GradCAM 的区别：
  - GradCAM：通过反向传播计算梯度 ∂y^c/∂A^k 作为权重
  - AblationCAM：通过消融实验测量分数下降量作为权重（无需梯度）

优点：
  - 无梯度，不受梯度饱和/噪声影响
  - 可解释性强，直观反映通道重要性

缺点：
  - 计算量大：C 个通道需要 C/batch_size 次额外前向传播
  - 不适合通道数多的模型（如 ViT 的 768 通道）

实现细节：
  - 目标层的激活值会被缓存，避免重复计算
  - 但目标层之前的层不会被缓存，如果目标层是较大的模块（如 VGG 的 features），可以节省大量时间
  - ratio_channels_to_ablate 参数控制消融比例（默认 1.0 表示消融所有通道）
"""


class AblationCAM(BaseCAM):
    """
    无梯度的 CAM 可视化方法。
    
    通过消融目标层的每个通道，测量目标类别分数的变化量，
    以此作为通道权重，生成类激活图。
    """
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 ablation_layer: torch.nn.Module = AblationLayer(),
                 batch_size: int = 32,
                 ratio_channels_to_ablate: float = 1.0) -> None:
        """
        Args:
            model: 要分析的 PyTorch 模型
            target_layers: 目标层列表（要分析激活值的层）
            use_cuda: 是否使用 GPU
            reshape_transform: 将 Transformer 输出 reshape 为 CNN 格式的函数
            ablation_layer: 消融层（负责将指定通道的激活值置零）
            batch_size: 每次并行消融的通道数（越大越快，但占用更多显存）
            ratio_channels_to_ablate: 消融比例（0.0~1.0），1.0 表示消融所有通道，
                                     <1.0 时只消融激活值大/小的通道以加速
        """
        super(AblationCAM, self).__init__(model,
                                          target_layers,
                                          use_cuda,
                                          reshape_transform,
                                          uses_gradients=False)  # AblationCAM 不使用梯度
        self.batch_size = batch_size
        self.ablation_layer = ablation_layer
        self.ratio_channels_to_ablate = ratio_channels_to_ablate

    def save_activation(self, module, input, output) -> None:
        """
        Forward hook 回调函数：缓存目标层的原始激活值。
        
        在阶段 1 的前向传播中，通过此 hook 保存目标层的输出，
        后续 AblationLayer 直接使用缓存的激活值，避免重复计算。
        """
        self.activations = output

    def assemble_ablation_scores(self,
                             new_scores: list,
                             original_score: float ,
                             ablated_channels: np.ndarray,
                             number_of_channels: int) -> np.ndarray:
        """
        组装消融后的分数：
        - 对于实际做过消融实验的通道，用消融后的分数
        - 对于跳过消融（加速）的通道，直接使用原始分数（相当于不影响权重计算）
        
        当 ratio_channels_to_ablate < 1.0 时（只消融部分通道以加速），
        需要将稀疏的消融结果填充到一个完整的通道长度数组中。
        
        Args:
            new_scores: 每个消融通道对应的消融后 logit 分数（按 ablated_channels 的顺序给出）
            original_score: 目标类别的原始 logit 分数（没有任何消融）
            ablated_channels: 实际做过消融的通道索引列表（通常会包含原始激活大的通道，跳过小的）
            number_of_channels: 目标层的总通道数
        
        Returns:
            np.ndarray: shape [number_of_channels], 每个通道消融后的分数数组
        """
        index = 0          # 指针：指向当前要处理的 ablated_channels 位置
        result = []        # 最终结果数组
        
        # 按通道索引排序：因为 ablated_channels 可能是乱序的（排序后方便顺序比较）
        sorted_indices = np.argsort(ablated_channels)  # 升序排列后的索引
        ablated_channels = ablated_channels[sorted_indices]  # 通道索引按从小到大排序
        new_scores = np.float32(new_scores)[sorted_indices]  # 分数也跟着重新排序

        # 遍历所有通道索引（0 ~ number_of_channels-1）
        for i in range(number_of_channels):
            # 如果当前通道 i 在 ablated_channels 中且 index 没有越界 → 说明做过消融
            if index < len(ablated_channels) and ablated_channels[index] == i:
                weight = new_scores[index]  # 使用消融后的分数
                index = index + 1           # 指针前进到下一个消融通道
            else:
                # 当前通道 i 没有做过消融 → 使用原始分数
                # 权重计算为 (original_score - new_score)/original_score，
                # 这里 new_score = original_score → 权重为 0，表示该通道对目标无贡献
                weight = original_score
            result.append(weight)

        return np.array(result)

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layer: torch.nn.Module,
                        targets: List[Callable],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        """
        通过消融实验计算每个通道的重要性权重（无需梯度）。
        
        核心思想：逐个通道清零（消融），观察目标类别分数下降了多少。
        下降越多 → 该通道越重要 → 权重越大。
        
        公式：α^c_k = (y^c_original - y^c_ablated(k)) / y^c_original
        
        Args:
            input_tensor: 输入图像 tensor
            target_layer: 目标层（要分析激活值的层）
            targets: 目标类别选择器（从 logits 中取出目标类别的分数）
            activations: 目标层的激活值（shape: [batch, channels, H, W]）
            grads: 梯度（AblationCAM 不使用，但基类接口需要）
        
        Returns:
            np.ndarray: 通道权重数组，shape [batch, channels]
        """
        
        # ==================== 阶段 1：记录原始分数（Baseline） ====================
        # 注册 forward hook：在前向传播时缓存目标层的激活值
        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)  # 完整前向传播
            handle.remove()  # 移除 hook，避免后续干扰
            # 提取目标类别的原始 logit 分数（作为比较基准线）
            original_scores = np.float32(
                [target(output).cpu().item() for target, output in zip(targets, outputs)]
            )

        # ==================== 阶段 2：替换目标层为 AblationLayer ====================
        # 将模型中的目标层临时替换为 AblationLayer（用于消融特定通道）
        # 计算结束后会替换回来，保证模型不受影响
        ablation_layer = self.ablation_layer
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        # ==================== 阶段 3：逐通道消融 + 测量分数下降 ====================
        number_of_channels = activations.shape[1]  # 目标层的通道数
        weights = []
        
        # AblationCAM 是"无梯度"方法，全程只需要前向传播
        with torch.no_grad():
            # 遍历 batch 中的每张图像
            for batch_index, (target, tensor) in enumerate(zip(targets, input_tensor)):
                new_scores = []
                # 将单张图像复制 batch_size 份，用于并行消融多个通道
                batch_tensor = tensor.repeat(self.batch_size, 1, 1, 1)

                # 选择要消融的通道：
                #   - ratio_channels_to_ablate=1.0 → 消融所有通道
                #   - ratio_channels_to_ablate<1.0 → 只消融激活值大的通道（加速）
                channels_to_ablate = ablation_layer.activations_to_be_ablated(
                    activations[batch_index, :], self.ratio_channels_to_ablate)
                number_channels_to_ablate = len(channels_to_ablate)

                # 按 batch_size 分批消融（加速）
                for i in tqdm.tqdm(range(0, number_channels_to_ablate, self.batch_size)):
                    # 处理最后一批（可能不足 batch_size 个通道）
                    if i + self.batch_size > number_channels_to_ablate:
                        batch_tensor = batch_tensor[:(number_channels_to_ablate - i)]

                    # 设置 AblationLayer：指定当前 batch 要消融哪些通道
                    # 内部会记录通道索引，前向传播时将这些通道的激活值置零
                    ablation_layer.set_next_batch(input_batch_index=batch_index,
                                                  activations=self.activations,
                                                  num_channels_to_ablate=batch_tensor.size(0))
                    # 前向传播 → 得到消融后的目标类别分数
                    score = [target(o).cpu().item() for o in self.model(batch_tensor)]
                    new_scores.extend(score)
                    # 更新已处理的通道索引列表
                    ablation_layer.indices = ablation_layer.indices[batch_tensor.size(0):]

                # 将稀疏的消融结果填充为完整的通道长度数组
                # 未消融的通道直接使用 original_score（权重为 0）
                new_scores = self.assemble_ablation_scores(new_scores,
                                                           original_scores[batch_index],
                                                           channels_to_ablate,
                                                           number_of_channels)
                weights.extend(new_scores)

        # ==================== 阶段 4：计算权重 ====================
        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])  # [batch, channels]
        original_scores = original_scores[:, None]         # [batch, 1] → 广播用
        
        # 权重公式：α^c_k = (y^c_original - y^c_ablated(k)) / y^c_original
        #   - 消融后分数大幅下降 → 权重接近 1（重要通道）
        #   - 消融后分数不变 → 权重为 0（无关通道）
        #   - 消融后分数上升 → 权重为负（抑制性通道）
        weights = (original_scores - weights) / original_scores

        # ==================== 阶段 5：恢复模型 ====================
        # 将 AblationLayer 替换回原始的目标层，保证模型恢复原状
        replace_layer_recursive(self.model, ablation_layer, target_layer)
        return weights
