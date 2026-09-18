import os
import datetime
import time
import cv2
import numpy as np
import argparse
import math
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from model import PI_CLIP
from util import dataset
from util import transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, \
    is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='PI_CLIP') # type: ignore
    parser.add_argument('--viz', action='store_true', default=False)
    # parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50_manet.yaml',
    #                     help='config file')
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50_manet.yaml',
                        help='config file')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):
    model = eval(args.arch).OneModel(args, cls_type='Base')
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    if hasattr(model, 'freeze_modules'):
        model.freeze_modules(model)

    if args.distributed:
        # Initialize Process Group
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model, device_ids=[args.local_rank])
    else:
        model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(resume_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        print('Number of Parameters: %d' % (total_number))
        print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer


def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))


def main():
    """
    训练流程的主入口函数
    
    负责以下任务：
    1. 解析配置参数和初始化日志
    2. 创建模型和优化器
    3. 构建训练集和验证集的数据加载器
    4. 执行训练-验证循环
    5. 保存检查点和记录最佳结果
    """
    global args, logger, writer
    
    # ==================== 初始化配置 ====================
    args = get_parser()  # 解析命令行参数和配置文件
    logger = get_logger()  # 初始化日志记录器
    args.distributed = True if torch.cuda.device_count() > 1 else False  # 根据GPU数量判断是否使用分布式训练
    if main_process():
        print(args)  # 打印配置参数（仅在主进程）

    # 设置随机种子以保证实验可重复性
    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    # 参数合法性检查
    assert args.classes > 1  # 类别数必须大于1
    assert args.zoom_factor in [1, 2, 4, 8]  # 缩放因子必须是特定值（与网络下采样倍数匹配）
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0  # 输入尺寸需满足网络要求

    # ==================== 模型构建 ====================
    if main_process():
        logger.info("=> creating model ...")
    model, optimizer = get_model(args)  # 创建模型和优化器（包含resume逻辑）
    if main_process():
        logger.info(model)  # 打印模型结构
    if main_process() and args.viz:
        writer = SummaryWriter(args.result_path)  # 初始化TensorBoard可视化工具

    # ==================== 数据预处理配置 ====================
    # ImageNet数据集的归一化参数（RGB通道）
    value_scale = 255
    mean = [0.485, 0.456, 0.406]  # ImageNet均值
    mean = [item * value_scale for item in mean]  # 转换到[0,255]范围
    std = [0.229, 0.224, 0.225]  # ImageNet标准差
    std = [item * value_scale for item in std]  # 转换到[0,255]范围
    
    # ----------------------  训练集数据增强  ----------------------
    # 标准训练变换（用于查询图像）
    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),  # 随机缩放
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),  # 随机旋转
        transform.RandomGaussianBlur(),  # 随机高斯模糊
        transform.RandomHorizontalFlip(),  # 随机水平翻转
        transform.Resize([args.train_h, args.train_w]),  # 调整到固定尺寸
        transform.ToTensor(),  # 转换为Tensor
        transform.Normalize(mean=mean, std=std)])  # 归一化
    
    # 三元组训练变换（用于支持图像，保持与查询图像相同的增强策略）
    train_transform_tri = transform_tri.Compose([
        transform_tri.RandScale([args.scale_min, args.scale_max]),
        transform_tri.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform_tri.RandomGaussianBlur(),
        transform_tri.RandomHorizontalFlip(),
        transform_tri.Resize([args.train_h, args.train_w]),
        transform_tri.ToTensor(),
        transform_tri.Normalize(mean=mean, std=std)])
    
    # 构建训练数据集
    if args.data_set == 'pascal' or args.data_set == 'coco':
        train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                     base_data_root=args.base_data_root, data_list=args.train_list, \
                                     transform=train_transform, transform_tri=train_transform_tri, mode='train', \
                                     data_set=args.data_set, use_split_coco=args.use_split_coco)
    
    # 分布式训练采样器（确保每个GPU看到不同的数据）
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, \
                                               pin_memory=True, sampler=train_sampler, drop_last=True, \
                                               shuffle=False if args.distributed else True)  # 分布式时使用sampler控制shuffle
    
    # ----------------------  验证集数据预处理  ----------------------
    if args.evaluate:
        # 根据配置选择验证集的resize策略
        if args.resized_val:
            # 直接resize到指定尺寸（可能改变宽高比）
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri.Compose([
                transform_tri.Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std)])
        else:
            # 保持宽高比的resize（短边对齐，长边padding）
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            val_transform_tri = transform_tri.Compose([
                transform_tri.test_Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std)])
        
        # 构建验证数据集
        if args.data_set == 'pascal' or args.data_set == 'coco':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                       base_data_root=args.base_data_root, data_list=args.val_list, \
                                       transform=val_transform, transform_tri=val_transform_tri, mode='val', \
                                       data_set=args.data_set, use_split_coco=args.use_split_coco)
        # 验证集dataloader（不shuffle，不使用分布式采样）
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ==================== 训练循环初始化 ====================
    # 初始化全局最佳性能指标
    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    global best_miou_m, best_miou_b, best_FBiou_m
    best_miou = 0.  # 最佳平均IoU
    best_FBiou = 0.  # 最佳前景背景IoU
    best_piou = 0.  # 最佳原型IoU
    best_epoch = 0  # 达到最佳性能的epoch
    keep_epoch = 0  # 连续未改进的epoch计数（用于早停）
    val_num = 0  # 验证次数
    best_miou_m = 0.  # 最佳多尺度平均IoU
    best_miou_b = 0.  # 最佳边界IoU
    best_FBiou_m = 0.  # 最佳多尺度前景背景IoU

    start_time = time.time()  # 记录训练开始时间

    # ==================== 主训练循环 ====================
    for epoch in range(args.start_epoch, args.epochs):
        # 早停机制：如果连续stop_interval个epoch没有改进，则停止训练
        if keep_epoch == args.stop_interval:
            break
        
        # 每个epoch使用不同的随机种子（可选）
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1  # 用于日志的epoch编号（从1开始）
        keep_epoch += 1  # 增加未改进计数
        
        # 分布式训练：每个epoch重新设置采样器种子
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # ----------------------  训练一个epoch  ----------------------
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, val_loader, model, optimizer, epoch)

        # 记录训练指标到TensorBoard
        if main_process() and args.viz:
            writer.add_scalar('FBIoU_train', mIoU_train, epoch_log)

        # 定期保存检查点（用于断点续训）
        if (epoch % args.save_freq == 0) and (epoch > 0) and main_process():
            filename = args.snapshot_path + '/epoch_{}.pth'.format(epoch)
            logger.info('Saving checkpoint to: ' + filename)
            if osp.exists(filename):
                os.remove(filename)  # 删除旧文件
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       filename)

        # -----------------------  验证  -----------------------
        # 每个epoch都进行验证（epoch % 1 == 0 恒为True）
        if args.evaluate and epoch % 1 == 0:
            loss_val, FBIoU, FBIoU_m, mIoU, mIoU_m, mIoU_b, pIoU = validate(val_loader, model)
            val_num += 1  # 验证次数加1
            
            # 记录验证指标到TensorBoard
            if main_process() and args.viz:
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('FBIoU_val', FBIoU, epoch_log)
                writer.add_scalar('mIoU_val', mIoU, epoch_log)
                writer.add_scalar('mIoU_val_m', mIoU_m, epoch_log)
                writer.add_scalar('mIoU_val_b', mIoU_b, epoch_log)
                writer.add_scalar('FBIoU_val_m', FBIoU_m, epoch_log)

            # 如果当前性能超过历史最佳，保存最佳模型（用于最终测试）
            if mIoU > best_miou:
                best_miou, best_FBiou, best_piou, best_epoch = mIoU, FBIoU, pIoU, epoch
                best_miou_m, best_miou_b, best_FBiou_m = mIoU_m, mIoU_b, FBIoU_m
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(epoch) + '_{:.4f}'.format(
                        best_miou) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                               filename)

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(args.arch + '\t Group:{} \t Best_step:{}'.format(args.split, best_epoch))
        print('mIoU:{:.4f} \t mIoU_m:{:.4f} \t mIoU_b:{:.4f}'.format(best_miou, best_miou_m, best_miou_b))
        print('FBIoU:{:.4f} \t FBIoU_m:{:.4f} \t pIoU:{:.4f}'.format(best_FBiou, best_FBiou_m, best_piou))
        print('>' * 80)
        print('%s' % datetime.datetime.now())


def train(train_loader, val_loader, model, optimizer, epoch):
    """
    训练一个epoch的主函数
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 待训练的模型
        optimizer: 优化器
        epoch: 当前训练轮数
    
    Returns:
        main_loss_meter.avg: 平均主损失
        mIoU: 平均交并比
        mAcc: 平均类别准确率
        allAcc: 全局准确率
    """
    # 声明全局变量，用于跟踪最佳模型性能指标
    global best_miou, best_FBiou, best_piou, best_epoch, keep_epoch, val_num
    global best_miou_m, best_miou_b, best_FBiou_m
    
    # 初始化各种统计指标的平均值计算器
    batch_time = AverageMeter()  # 批次处理时间
    data_time = AverageMeter()  # 数据加载时间
    main_loss_meter = AverageMeter()  # 主损失值
    aux_loss_meter1 = AverageMeter()  # 辅助损失1
    aux_loss_meter2 = AverageMeter()  # 辅助损失2
    loss_meter = AverageMeter()  # 总损失
    intersection_meter = AverageMeter()  # 预测与标签的交集
    union_meter = AverageMeter()  # 预测与标签的并集
    target_meter = AverageMeter()  # 真实标签统计

    model.train()  # 设置模型为训练模式
    if args.fix_bn:
        model.apply(fix_bn)  # 固定BatchNorm层的参数，防止其在训练过程中更新

    end = time.time()  # 记录起始时间
    val_time = 0.  # 验证过程耗时
    max_iter = args.epochs * len(train_loader)  # 计算最大迭代次数
    if main_process():
        print('Warmup: {}'.format(args.warmup))  # 打印学习率预热信息

    # 遍历训练数据加载器，i为批次索引
    for i, (input, input_name, target, target_b, s_input, s_mask, subcls, class_name, img_cv2) in enumerate(train_loader):

        data_time.update(time.time() - end)  # 更新数据加载时间
        current_iter = epoch * len(train_loader) + i + 1  # 计算当前迭代次数

        # 使用多项式学习率衰减策略更新学习率
        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                           index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        # 将所有数据移动到GPU，non_blocking=True允许异步数据传输以提高效率
        s_input = s_input.cuda(non_blocking=True)  # 支持样本图像
        s_mask = s_mask.cuda(non_blocking=True)  # 支持样本掩码
        input = input.cuda(non_blocking=True)  # 查询图像
        img_cv2 = img_cv2.cuda(non_blocking=True)  # 原始图像（OpenCV格式）
        target = target.cuda(non_blocking=True)  # 查询图像的标签
        target_b = target_b.cuda(non_blocking=True)  # 边界标签

        # 前向传播：模型接收支持集和查询集，输出预测结果和损失
        # s_x: 支持图像, que_name: 查询图像名称, s_y: 支持掩码
        # x: 查询图像, x_cv2: 查询图像原始格式, y_m: 查询掩码, y_b: 边界掩码
        # cat_idx: 类别索引, class_name: 类别名称
        output, main_loss, aux_loss1, aux_loss2 = model(s_x=s_input, que_name=input_name, s_y=s_mask, x=input, x_cv2=img_cv2, y_m=target, y_b=target_b, cat_idx=subcls, class_name=class_name)

        # 计算总损失：主损失 + 加权辅助损失
        loss = main_loss + args.aux_weight1 * aux_loss1 + args.aux_weight2 * aux_loss2

        # 反向传播优化步骤
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        n = input.size(0)  # 获取批次大小

        # 计算预测结果与真实标签的交集、并集（用于计算IoU）
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        # 将GPU张量转换为CPU numpy数组以便统计
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        # 计算当前批次的准确率（所有类别的平均准确率）
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # allAcc

        # 更新各项损失指标
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter1.update(aux_loss1.item(), n)
        aux_loss_meter2.update(aux_loss2.item(), n)
        loss_meter.update(loss.item(), n)

        # 更新批次处理时间（扣除验证时间）
        batch_time.update(time.time() - end - val_time)
        end = time.time()  # 更新结束时间

        # 计算剩余训练时间
        remain_iter = max_iter - current_iter  # 剩余迭代次数
        remain_time = remain_iter * batch_time.avg  # 预估剩余时间（秒）
        t_m, t_s = divmod(remain_time, 60)  # 转换为分钟和秒
        t_h, t_m = divmod(t_m, 60)  # 转换为小时和分钟
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))  # 格式化为 HH:MM:SS

        # 按指定频率打印训练日志（仅在主进程打印）
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss1 {aux_loss_meter1.val:.4f} '
                        'AuxLoss2 {aux_loss_meter2.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter1=aux_loss_meter1,
                                                          aux_loss_meter2=aux_loss_meter2,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
            # 如果启用可视化，将损失值写入TensorBoard
            if args.viz:
                writer.add_scalar('loss_train', loss_meter.val, current_iter)
                writer.add_scalar('loss_train_main', main_loss_meter.val, current_iter)
                writer.add_scalar('loss_train_aux1', aux_loss_meter1.val, current_iter)
                writer.add_scalar('loss_train_aux2', aux_loss_meter2.val, current_iter)

        # -----------------------  子epoch验证  -----------------------
        # 在训练中途进行验证的条件：
        # 1. 启用评估模式
        # 2. 启用子epoch验证
        # 3. 总epoch数<=100且当前epoch>0
        # 4. 当前批次为训练数据的一半位置
        if args.evaluate and args.SubEpoch_val and (args.epochs <= 100 and epoch % 1 == 0 and epoch > 0) and (
                i == round(len(train_loader) / 2)):  # <if> max_epoch<=100 <do> half_epoch Val
            # 执行验证并获取各项指标
            loss_val, FBIoU, FBIoU_m, mIoU, mIoU_m, mIoU_b, pIoU = validate(val_loader, model)
            val_num += 1  # 验证次数加1
            
            # 如果当前mIoU超过历史最佳值，则保存模型
            if mIoU > best_miou:
                # 更新最佳性能指标
                best_miou, best_FBiou, best_piou, best_epoch = mIoU, FBIoU, pIoU, (epoch - 0.5)
                best_miou_m, best_miou_b, best_FBiou_m = mIoU_m, mIoU_b, FBIoU_m
                keep_epoch = 0  # 重置保持计数器
                
                # 根据shot数（支持样本数量）生成不同的文件名
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch - 0.5) + '_{:.4f}'.format(
                        best_miou) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(
                        epoch - 0.5) + '_{:.4f}'.format(best_miou) + '.pth'
                
                # 在主进程中保存模型检查点
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch - 0.5, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename)

            # 验证结束后恢复模型到训练模式
            model.train()
            if args.fix_bn:
                model.apply(fix_bn)  # 重新固定BatchNorm层

    # 一个epoch训练结束，计算最终的评估指标
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)  # 各类别的IoU
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)  # 各类别的准确率
    mIoU = np.mean(iou_class)  # 平均IoU
    mAcc = np.mean(accuracy_class)  # 平均类别准确率
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)  # 全局准确率

    # 在主进程中打印训练结果
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
        # 打印每个类别的详细结果
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()  # final
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    intersection_meter_m = AverageMeter()  # meta
    union_meter_m = AverageMeter()
    target_meter_m = AverageMeter()

    if args.data_set == 'pascal':
        test_num = 1000  # 5000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = 1000  # 20000
        split_gap = 20

    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap
    class_intersection_meter_m = [0] * split_gap
    class_union_meter_m = [0] * split_gap
    class_intersection_meter_b = [0] * split_gap * 3
    class_union_meter_b = [0] * split_gap * 3
    class_target_meter_b = [0] * split_gap * 3

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, input_name, target, target_b, s_input, s_mask, subcls, class_name, ori_label, ori_label_b,
                img_cv2) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            img_cv2 = img_cv2.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_b = target_b.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            ori_label_b = ori_label_b.cuda(non_blocking=True)
            start_time = time.time()

            output, meta_out, base_out = model(s_x=s_input, x_cv2=img_cv2, que_name=input_name, s_y=s_mask, x=input,
                                               y_m=target, y_b=target_b,
                                               cat_idx=subcls, class_name=class_name)
            model_time.update(time.time() - start_time)

            H, W = target.shape[-2:]
            if args.ori_resize:  # 真值转化为方形
                H, W = ori_label.size(1), ori_label.size(2)
                target = map_to_square(ori_label).long()
                target_b = map_to_square(ori_label_b).long()

            output = map_to_square(F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True))
            meta_out = map_to_square(F.interpolate(meta_out, size=(H, W), mode='bilinear', align_corners=True))
            base_out = map_to_square(F.interpolate(base_out, size=(H, W), mode='bilinear', align_corners=True))

            loss = criterion(output, target)

            output = output.max(1)[1]
            meta_out = meta_out.max(1)[1]
            base_out = base_out.max(1)[1]

            subcls = subcls[0].cpu().numpy()[0]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            class_intersection_meter[subcls] += intersection[1]
            class_union_meter[subcls] += union[1]

            intersection, union, new_target = intersectionAndUnionGPU(meta_out, target, args.classes, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter_m.update(intersection), union_meter_m.update(union), target_meter_m.update(new_target)
            class_intersection_meter_m[subcls] += intersection[1]
            class_union_meter_m[subcls] += union[1]

            intersection, union, new_target = intersectionAndUnionGPU(base_out, target_b, split_gap * 3 + 1,
                                                                      args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            for idx in range(1, len(intersection)):
                class_intersection_meter_b[idx - 1] += intersection[idx]
                class_union_meter_b[idx - 1] += union[idx]
                class_target_meter_b[idx - 1] += new_target[idx]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % round((test_num / 100)) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))
    val_time = time.time() - val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou_class_m = intersection_meter_m.sum / (union_meter_m.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mIoU_m = np.mean(iou_class_m)

    class_iou_class = []
    class_iou_class_m = []
    class_iou_class_b = []
    class_miou = 0
    class_miou_m = 0
    class_miou_b = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
        class_iou = class_intersection_meter_m[i] / (class_union_meter_m[i] + 1e-10)
        class_iou_class_m.append(class_iou)
        class_miou_m += class_iou
    for i in range(len(class_intersection_meter_b)):
        class_iou = class_intersection_meter_b[i] / (class_union_meter_b[i] + 1e-10)
        class_iou_class_b.append(class_iou)
        class_miou_b += class_iou

    target_b = np.array(class_target_meter_b)

    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    class_miou_m = class_miou_m * 1.0 / len(class_intersection_meter)
    class_miou_b = class_miou_b * 1.0 / (
                len(class_intersection_meter_b) - len(target_b[target_b == 0]))  # filter the results with GT mIoU=0

    if main_process():
        logger.info('meanIoU---Val result: mIoU_f {:.4f}.'.format(class_miou))  # final
        logger.info('meanIoU---Val result: mIoU_m {:.4f}.'.format(class_miou_m))  # meta
        logger.info('meanIoU---Val result: mIoU_b {:.4f}.'.format(class_miou_b))  # base

        logger.info('<<<<<<< Novel Results <<<<<<<')
        for i in range(split_gap):
            logger.info('Class_{} Result: iou_f {:.4f}.'.format(i + 1, class_iou_class[i]))
            logger.info('Class_{} Result: iou_m {:.4f}.'.format(i + 1, class_iou_class_m[i]))
        logger.info('<<<<<<< Base Results <<<<<<<')
        for i in range(split_gap * 3):
            if class_target_meter_b[i] == 0:
                logger.info('Class_{} Result: iou_b None.'.format(i + 1 + split_gap))
            else:
                logger.info('Class_{} Result: iou_b {:.4f}.'.format(i + 1 + split_gap, class_iou_class_b[i]))

        logger.info('FBIoU---Val result: FBIoU_f {:.4f}.'.format(mIoU))
        logger.info('FBIoU---Val result: FBIoU_m {:.4f}.'.format(mIoU_m))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou_f {:.4f}.'.format(i, iou_class[i]))
            logger.info('Class_{} Result: iou_m {:.4f}.'.format(i, iou_class_m[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

        print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return loss_meter.avg, mIoU, mIoU_m, class_miou, class_miou_m, class_miou_b, iou_class[1]


def map_to_square(x):
    H, W = x.shape[-2:]
    longerside = max(H, W)
    assert len(x.shape) in (3, 4)
    if len(x.shape) == 3:
        backmask = torch.ones(x.shape[0], longerside, longerside, device='cuda') * 255
        backmask[0, :x.shape[-2], :x.shape[-1]] = x
    else:
        backmask = torch.ones(x.shape[0], x.shape[1], longerside, longerside, device='cuda') * 255
        backmask[0, :, :x.shape[-2], :x.shape[-1]] = x

    return backmask


if __name__ == '__main__':
    main()