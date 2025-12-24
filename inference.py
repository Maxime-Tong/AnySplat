import os
from pathlib import Path
import sys
import json
import gzip
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from einops import rearrange

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from misc.image_io import save_image, save_interpolated_video
from src.utils.image import process_image

from src.model.model.anysplat import AnySplat
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

class CUDATimer:
    def __init__(self, print_prefix=""):
        self.print_prefix = print_prefix
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print(f"{self.print_prefix} Elapsed time: {self.elapsed_time_ms:.2f} ms")

def setup_args():
    """Set up command-line arguments for the eval NVS script."""
    parser = argparse.ArgumentParser(description='Test AnySplat on NVS evaluation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NVS dataset')
    parser.add_argument('--llffhold', type=int, default=8, help='LLFF holdout')
    parser.add_argument('--output_path', type=str, default="outputs/nvs", help='Path to output directory')
    parser.add_argument('--views', type=int, default=64, help='Number of context views')
    parser.add_argument('--save', action='store_true', help='Whether to save the output images and videos')
    return parser.parse_args()

def compute_metrics(pred_image, image):
    psnr = compute_psnr(pred_image, image)
    ssim = compute_ssim(pred_image, image)
    lpips = compute_lpips(pred_image, image)
    return psnr, ssim, lpips

def evaluate(args: argparse.Namespace):
    model = AnySplat.from_pretrained("lhjiang/anysplat", cache_dir="./pretrained_models/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    os.makedirs(args.output_path, exist_ok=True)

    # load images
    image_folder = args.data_dir
    image_names = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if args.views != -1:
        num_ctx = args.views
        num_tgt = args.views // args.llffhold
        num_frames = num_ctx + num_tgt
        image_names = image_names[:num_frames]
        
    images = [process_image(img_path) for img_path in image_names]
    ctx_indices = [idx for idx, name in enumerate(image_names) if idx % args.llffhold != 0][:num_ctx]
    tgt_indices = [idx for idx, name in enumerate(image_names) if idx % args.llffhold == 0][:num_tgt]

    ctx_images = torch.stack([images[i] for i in ctx_indices], dim=0).unsqueeze(0).to(device)
    tgt_images = torch.stack([images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)
    ctx_images = (ctx_images+1)*0.5
    tgt_images = (tgt_images+1)*0.5
    b, v, _, h, w = tgt_images.shape

    # run inference
    with CUDATimer("[Running inference]"):
        encoder_output = model.encoder(
            ctx_images,
            global_step=0,
            visualization_dump={},
        )
    gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose

    num_context_view = ctx_images.shape[1]
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(vggt_input_image, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(pred_all_pose_enc, vggt_input_image.shape[-2:])

    extrinsic_padding = torch.tensor([0, 0, 0, 1], device=pred_all_extrinsic.device, dtype=pred_all_extrinsic.dtype).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
    pred_all_context_extrinsic, pred_all_target_extrinsic = pred_all_extrinsic[:, :num_context_view], pred_all_extrinsic[:, num_context_view:]
    pred_all_context_intrinsic, pred_all_target_intrinsic = pred_all_intrinsic[:, :num_context_view], pred_all_intrinsic[:, num_context_view:]

    scale_factor = pred_context_pose['extrinsic'][:, :, :3, 3].mean() / pred_all_context_extrinsic[:, :, :3, 3].mean()
    pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
    pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor
    print("scale_factor:", scale_factor)
    
    with CUDATimer("[Rendering video]"):
        output = model.decoder.forward(
            gaussians,
            pred_all_target_extrinsic,
            pred_all_target_intrinsic.float(),
            torch.ones(1, v, device=device) * 0.01,
            torch.ones(1, v, device=device) * 100,
            (h, w)
            )

    if args.save:
        save_interpolated_video(pred_all_context_extrinsic, pred_all_context_intrinsic, b, h, w, gaussians, args.output_path, model.decoder)
    
    # Save original images
    save_path = Path(args.output_path)
    # os.makedirs(save_path, exist_ok=True)
    # for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
    #     save_image(gt_image, save_path / "gt" / f"{idx:0>6}.jpg")
    #     save_image(pred_image, save_path / "pred" / f"{idx:0>6}.jpg")

    # compute metrics
    psnr, ssim, lpips = compute_metrics(output.color[0], tgt_images[0])
    print(f"PSNR: {psnr.mean():.2f}, SSIM: {ssim.mean():.3f}, LPIPS: {lpips.mean():.3f}")

if __name__ == "__main__":
    args = setup_args()
    evaluate(args)