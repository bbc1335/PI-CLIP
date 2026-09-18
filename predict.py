"""Run few-shot segmentation inference and save predicted masks.

This script keeps the model/data pipeline used by test.py, but replaces the
metric evaluation loop with a prediction loop.

Examples:
    python predict.py --config config/pascal/pascal_split0_resnet50_manet.yaml \
        --shot 1 --weight train_epoch_32_0.7788.pth

    python predict.py --config config/pascal/pascal_split0_resnet50_manet.yaml \
        --shot 5 --weight /path/to/5shot_checkpoint.pth
"""

import argparse
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from util import config, dataset, transform, transform_tri
from util.util import check_makedirs, get_save_path, setup_seed

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description="PI-CLIP few-shot segmentation prediction")
    parser.add_argument("--arch", type=str, default="PI_CLIP")
    parser.add_argument("--config", type=str, default="config/pascal/pascal_split0_resnet50_manet.yaml", help="Path to the YAML configuration file.")
    parser.add_argument(
        "--shot",
        dest="shot_override",
        type=int,
        choices=(1, 5),
        default=None,
        help="Number of support samples. Overrides Train.shot in the config.",
    )
    parser.add_argument(
        "--weight",
        dest="checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint filename or path. A filename is resolved under the "
            "snapshot directory; otherwise Test_Finetune.weight is used."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for predicted masks.")
    parser.add_argument("--num-samples", type=int, default=-1, help="Maximum number of queries to predict; -1 means all queries.")
    parser.add_argument(
        "--save-probability",
        action="store_true",
        help="Also save the foreground probability as a float32 .npy file.",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Also save the prediction overlaid on the query image.",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional config overrides. This option must be placed last.",
    )

    cli_args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli_args.config)
    cfg = config.merge_cfg_from_args(cfg, cli_args)
    if cli_args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, cli_args.opts)

    if cli_args.shot_override is not None:
        cfg.shot = cli_args.shot_override
    if cfg.shot not in (1, 5):
        raise ValueError("--shot must be 1 or 5, got {}".format(cfg.shot))
    if cfg.num_samples == 0 or cfg.num_samples < -1:
        raise ValueError("--num-samples must be a positive integer or -1")

    return cfg


def resolve_checkpoint(args):
    checkpoint = args.checkpoint or getattr(args, "weight", None)
    if not checkpoint:
        raise ValueError(
            "No checkpoint configured. Pass --weight or set "
            "Test_Finetune.weight in the config."
        )

    candidates = [checkpoint]
    if not osp.isabs(checkpoint):
        candidates.append(osp.join(args.snapshot_path, checkpoint))

    for path in candidates:
        if osp.isfile(path):
            return path

    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n  {}".format("\n  ".join(candidates))
    )


def build_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PI-CLIP inference requires CUDA because the model contains "
            "device-specific .cuda() operations."
        )

    # Import here so argument parsing and dataset checks do not require the
    # full model dependency stack to be available.
    from model import PI_CLIP  # noqa: F401

    model = eval(args.arch).OneModel(args, cls_type="Base")
    model = model.cuda()

    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = dict(state_dict)

    # This tensor is created dynamically by CLIP's variable-resolution
    # positional embedding and is not present in a newly built model.
    state_dict.pop("clip_model.visual.positional_embedding_new", None)

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module.") :]: value for key, value in state_dict.items()
        }

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint '{}'. Make sure --shot {} matches "
            "the number of shots used to train this checkpoint.".format(
                checkpoint_path, args.shot
            )
        ) from exc

    model.eval()
    return model, checkpoint_path, checkpoint.get("epoch")


def build_val_loader(args):
    value_scale = 255
    mean = [item * value_scale for item in [0.485, 0.456, 0.406]]
    std = [item * value_scale for item in [0.229, 0.224, 0.225]]

    if args.resized_val:
        val_transform = transform.Compose(
            [
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std),
            ]
        )
        val_transform_tri = transform_tri.Compose(
            [
                transform_tri.Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std),
            ]
        )
    else:
        val_transform = transform.Compose(
            [
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std),
            ]
        )
        val_transform_tri = transform_tri.Compose(
            [
                transform_tri.test_Resize(size=args.val_size),
                transform_tri.ToTensor(),
                transform_tri.Normalize(mean=mean, std=std),
            ]
        )

    val_data = dataset.SemData(
        split=args.split,
        shot=args.shot,
        data_root=args.data_root,
        base_data_root=args.base_data_root,
        data_list=args.val_list,
        transform=val_transform,
        transform_tri=val_transform_tri,
        mode="val",
        data_set=args.data_set,
        use_split_coco=args.use_split_coco,
    )
    return torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )


def batch_name(input_name):
    if isinstance(input_name, (list, tuple)):
        return input_name[0]
    return input_name


def save_prediction(args, output_dir, name, prediction, probability, image):
    stem = osp.splitext(osp.basename(name))[0]
    mask_path = osp.join(output_dir, stem + ".png")
    cv2.imwrite(mask_path, prediction.astype(np.uint8) * 255)

    if args.save_probability:
        np.save(
            osp.join(output_dir, stem + "_prob.npy"),
            probability.astype(np.float32),
        )

    if args.save_overlay:
        image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        image = cv2.resize(
            image,
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        overlay = image.copy()
        foreground = prediction.astype(bool)
        color = np.array([255, 0, 0], dtype=np.float32)
        overlay[foreground] = (
            0.5 * overlay[foreground] + 0.5 * color
        ).astype(np.uint8)
        overlay_path = osp.join(output_dir, stem + "_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return mask_path


def main():
    args = get_parser()
    get_save_path(args)
    check_makedirs(args.result_path)

    if args.output_dir is None:
        args.output_dir = osp.join(
            args.result_path, "predict", "{}shot".format(args.shot)
        )
    check_makedirs(args.output_dir)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    print("Configuration:")
    print(args)
    print("Using {} support sample(s).".format(args.shot))

    model, checkpoint_path, epoch = build_model(args)
    print("Loaded checkpoint: {}".format(checkpoint_path))
    if epoch is not None:
        print("Checkpoint epoch: {}".format(epoch))

    val_loader = build_val_loader(args)
    total = len(val_loader)
    if args.num_samples > 0:
        total = min(total, args.num_samples)

    processed = 0
    progress = tqdm(val_loader, total=total, desc="Predict")
    for batch in progress:
        (
            input_tensor,
            input_name,
            _,
            _,
            support_image,
            support_mask,
            subcls,
            class_name,
            original_label,
            _,
            image_cv2,
        ) = batch

        input_tensor = input_tensor.cuda(non_blocking=True)
        support_image = support_image.cuda(non_blocking=True)
        support_mask = support_mask.cuda(non_blocking=True)
        image_cv2 = image_cv2.cuda(non_blocking=True)

        # Keep gradients enabled because the VTP branch uses GradCAM inside
        # the model, even in evaluation mode.
        output, _, _ = model(
            x_cv2=image_cv2,
            que_name=input_name,
            s_x=support_image,
            s_y=support_mask,
            x=input_tensor,
            cat_idx=subcls,
            class_name=class_name,
        )

        height, width = original_label.shape[-2:]
        output = F.interpolate(
            output,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        probability = F.softmax(output, dim=1)[0, 1].detach().cpu().numpy()
        prediction = output.argmax(dim=1)[0].detach().cpu().numpy()

        name = batch_name(input_name)
        save_prediction(
            args,
            args.output_dir,
            name,
            prediction,
            probability,
            image_cv2[0],
        )

        processed += 1
        if args.num_samples > 0 and processed >= args.num_samples:
            break

    print(
        "Saved {} prediction(s) to {}".format(processed, args.output_dir)
    )


if __name__ == "__main__":
    main()
