# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os

from tqdm import tqdm
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import parse_xml_to_dict, scoremap2bbox
from clip.clip_text import class_names, new_class_names, new_class_names_coco#, imagenet_templates
import argparse
from lxml import etree
import torch.nn.functional as F
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def split_dataset(dataset, n_splits):
    if n_splits == 1:
        return [dataset]
    part = len(dataset) // n_splits
    dataset_list = []
    for i in range(n_splits - 1):
        dataset_list.append(dataset[i*part:(i+1)*part])
    dataset_list.append(dataset[(i+1)*part:])

    return dataset_list

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_resize(h, w):
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def img_ms_and_flip(que_img, ori_height, ori_width, scales=[1.0], patch_size=16):

    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        que_img = que_img.cpu().detach().numpy().astype(np.uint8)
        que_img = Image.fromarray(que_img.transpose(1,2,0))
        image = preprocess(que_img)
    return image


def get_img_cam(que_img, tmp_que_name, que_class, model, bg_text_features, fg_text_features, cam, annotation_root, flag=None):
    """
    视觉-文本原型CAM生成函数 (VTP: Vision-Text Prototype)

    核心功能：
    1. 用CLIP的文本特征（前景/背景描述）引导GradCAM生成类别激活图
    2. 利用CLIP Transformer的自注意力权重，通过随机游走扩散机制精炼CAM
    3. 返回每个查询图像对应目标类别的精炼CAM热力图

    参数:
        que_img: 查询图像张量 [bs, 3, H, W]
        tmp_que_name: 查询图像文件名列表 [bs]
        que_class: 目标类别ID列表 [bs]
        model: CLIP模型
        bg_text_features: 背景文本特征矩阵 [num_classes, dim]  ("a photo without {xxx}")
        fg_text_features: 前景文本特征矩阵 [num_classes, dim]  ("a photo of {xxx}")
        cam: GradCAM对象 (在PI_CLIP.py中传入)
        annotation_root: 标注文件根目录
        flag: None=训练, False/True=推理（决定验证/训练集标注路径）

    返回:
        refined_cam_all_scales: 精炼后的CAM图列表 [bs, ori_H, ori_W]
                              如果目标类别不在图像中，返回全255的(64,64)张量作为"无目标"标记
    """
    model = model.cuda()
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()
    refined_cam_all_scales = []

    # ==================== 逐图像处理 ====================
    for i in range(0, len(tmp_que_name)):
        que_name = tmp_que_name[i]

        # ============== 步骤1: 读取标注，获取图像中所有物体 ==============
        if 'VOC' in annotation_root:
            # ----- PASCAL VOC 数据集: 从XML读取标注 -----
            xmlfile = os.path.join(annotation_root, str(que_name))
            xmlfile = xmlfile.replace('.jpg', '.xml')
            with open(xmlfile) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]

            ori_width = int(data['size']['width'])
            ori_height = int(data['size']['height'])

            # 收集图像中所有物体的标签名和ID
            label_list = []
            label_id_list = []
            for obj in data["object"]:
                # 将原始类别名映射到new_class_names中的新名称
                obj["name"] = new_class_names[class_names.index(obj["name"])]
                if obj["name"] not in label_list:
                    label_list.append(obj["name"])
                    label_id_list.append(new_class_names.index(obj["name"]))
        else:
            # ----- COCO 数据集: 从PNG标签图读取标注 -----
            ori_height, ori_width = np.asarray(que_img[i].cpu().detach()).shape[1:]
            if flag == False:
                # 验证集val2014
                tmp_label_img_path = os.path.join(annotation_root, 'val2014')
                label_img_path = os.path.join(tmp_label_img_path, que_name).replace('jpg', 'png')
            else:
                # 训练集train2014
                tmp_label_img_path = os.path.join(annotation_root, 'train2014')
                label_img_path = os.path.join(tmp_label_img_path, que_name).replace('jpg', 'png')
            label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
            label_id_list = np.unique(label_img).tolist()     # 所有出现的类别ID
            if 0 in label_id_list:
                label_id_list.remove(0)                       # 移除背景类(0)
            if 255 in label_id_list:
                label_id_list.remove(255)                     # 移除void标签(255)
            label_id_list = [x - 1 for x in label_id_list]    # COCO标签从1开始，转0-index

            label_list = []
            for lid in label_id_list:
                label_list.append(new_class_names_coco[int(lid)])

        # ============== 步骤2: 检查目标类别是否存在 ==============
        # 如果目标类别（que_class[i]）不在当前图像的标签列表中，返回全255的占位图
        # 这个检查很重要——CAM生成需要目标类别在图像中真实存在，否则GradCAM会生成无意义的噪声
        if que_class[i] not in label_id_list:
            return [torch.full((64,64),255).float().cuda()]

        # ============== 步骤3: 图像预处理与CLIP编码 ==============
        # 将图像缩放到patch_size的整数倍（CLIP ViT要求）
        image = img_ms_and_flip(que_img[i], ori_height, ori_width, scales=[1.0])
        image = image.unsqueeze(0)                            # [1, 3, h, w]
        h, w = image.shape[-2], image.shape[-1]
        image = image.cuda()
        # 通过CLIP的视觉编码器，返回多层特征和自注意力权重
        image_features, attn_weight_list = model.encode_image(image, h, w)

        # ============== 步骤4: 构建文本特征 ==============
        # 选取图像中实际出现的标签对应的前景/背景文本特征
        bg_features_temp = bg_text_features[label_id_list].cuda()
        fg_features_temp = fg_text_features[label_id_list].cuda()
        # 前景文本在前，背景文本在后，拼接为完整的文本特征矩阵
        text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
        input_tensor = [image_features, text_features_temp, h, w]

        # ============== 步骤5: GradCAM生成 + 自注意力精炼 ==============
        for idx, label in enumerate(label_list):
            if 'VOC' in annotation_root:
                label_id = new_class_names.index(label)
            else:
                label_id = new_class_names_coco.index(label)
            # 只处理与目标类别ID匹配的标签
            if label_id == que_class[i]:
                # --- 5a: GradCAM计算 ---
                targets = [ClipOutputTarget(label_list.index(label))]
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                                   targets=targets,
                                                                                   target_size=None)
                grayscale_cam = grayscale_cam[0, :]            # [14, 14] (CLIP patch网格)

                # --- 5b: 聚合注意力权重 ---
                # 将最后一层的注意力权重添加到前面的注意力列表中
                attn_weight_list.append(attn_weight_last)
                # 去掉CLS token对应的注意力，只保留patch-to-patch的注意力 [14*14, 14*14]
                attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
                attn_weight = torch.stack(attn_weight, dim=0)[-8:]  # 取最后8层
                attn_weight = torch.mean(attn_weight, dim=0)        # 跨层平均
                attn_weight = attn_weight[0].detach()               # 取第一个batch
                attn_weight = attn_weight.float()

                # --- 5c: 基于CAM阈值的边界框掩码 ---
                # 将CAM按阈值0.4转为矩形边界框，框住高响应区域（目标物体所在位置）
                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1                # 框内区域=1，框外=0
                aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])

                # --- 5d: 构建扩散转移矩阵 (随机游走) ---
                aff_mat = attn_weight                              # [N, N], N=14*14

                # 双向对称归一化: 行和列分别归一化，保证转移矩阵的对称性
                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

                # 对称化: 取转移矩阵与其转置的Hadamard乘积后的逐元素最大值
                # 这步确保矩阵是双向可达的——如果A可以到达B，B也可以到达A
                H_trans_mat = trans_mat @ trans_mat.t()
                trans_mat = torch.max(trans_mat, H_trans_mat)

                # 幂次扩散: 矩阵乘法模拟多步随机游走，让信息沿高注意力路径传播
                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                # 用边界框掩码裁剪: 只允许在目标物体区域内传播
                trans_mat = trans_mat * aff_mask

                # --- 5e: 用扩散转移矩阵精炼CAM ---
                cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
                cam_to_refine = cam_to_refine.view(-1, 1)         # [N, 1], N=196

                # 核心公式: cam_refined = T * cam_original
                # 转移矩阵 T [N,N] 乘 CAM 向量 [N,1]
                # 效果：每个像素的CAM值被更新为"它所能到达的所有像素的CAM值的加权和"
                # 权重由注意力强度决定——相邻且语义相似的patch会互相增强
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)

        # ============== 步骤6: 上采样到原始尺寸 ==============
        # 将(14,14)的CAM图上采样回原始图像大小 (ori_width, ori_height)
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
        refined_cam_all_scales.append(torch.tensor(cam_refined_highres).cuda())
    
    return refined_cam_all_scales

