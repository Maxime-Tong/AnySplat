
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_video
from src.model.ply_export import export_ply
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image

import argparse

OUT_FOLDER = "output/inference"
os.makedirs(OUT_FOLDER, exist_ok=True)

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
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print(f"{self.print_prefix}Elapsed time: {elapsed_time_ms:.2f} ms")

def render_video(
    pred_extrinsics, pred_intrinsics, b, h, w, gaussians, save_path, decoder_func, SAVE_FLAG=False
):
    # Update K to reflect the new number of frames
    num_frames = pred_extrinsics.shape[1]
    
    with torch.no_grad():
        with CUDATimer("[Rendering video] "):
            output = decoder_func.forward(
                gaussians,
                pred_extrinsics,
                pred_intrinsics.float(),
                torch.ones(1, num_frames, device=pred_extrinsics.device) * 0.1,
                torch.ones(1, num_frames, device=pred_extrinsics.device) * 100,
                (h, w),
            )

    # Convert to video format
    video = output.color[0].clip(min=0, max=1)
    depth = output.depth[0]
    
    # Normalize depth for visualization
    # to avoid `quantile() input tensor is too large`
    num_views = pred_extrinsics.shape[1] 
    depth_norm = (depth - depth[::num_views].quantile(0.01)) / (
        depth[::num_views].quantile(0.99) - depth[::num_views].quantile(0.01)
    )
    depth_norm = plt.cm.turbo(depth_norm.cpu().numpy())
    depth_colored = (
        torch.from_numpy(depth_norm[..., :3]).permute(0, 3, 1, 2).to(depth.device)
    )
    depth_colored = depth_colored.clip(min=0, max=1)

    if SAVE_FLAG:
        # Save depth video
        save_video(depth_colored, os.path.join(save_path, f"depth.mp4"))
        # Save video
        save_video(video, os.path.join(save_path, f"rgb.mp4"))
        
    return os.path.join(save_path, f"rgb.mp4"), os.path.join(save_path, f"depth.mp4")

def sample_images(images, num_views):
    total_views = len(images)
    if total_views <= 100:
        return images
    interval = total_views / 100
    selected_indices = [int(i * interval) for i in range(100)][:num_views]
    sampled_images = [images[i] for i in selected_indices]
    return sampled_images

def calc_metric(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    psnr = -10 * torch.log10(torch.tensor(mse)).item()
    print(f"[Metric] PSNR: {psnr} dB")
    return psnr

def main(args):
    SAVE_FLAG = args.save
    # Load the model from Hugging Face
    model = AnySplat.from_pretrained("lhjiang/anysplat", cache_dir="./pretrained_models/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Load Images
    image_folder = args.data
    images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = sample_images(images, args.v)
    
    print(f"Using {len(images)} images for inference.")
    
    images = [process_image(img_path) for img_path in images]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape
    
    # Run Inference
    with torch.no_grad():
        with CUDATimer("[Running inference] "):
            gaussians, pred_context_pose = model.inference((images+1)*0.5)

    # Save the results
    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']
    
    print("Rendering video...")
    render_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, args.out, model.decoder, SAVE_FLAG=args.save)
    if args.save:
        export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(args.out) / "gaussians.ply")
        
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="AnySplat Inference Script")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset images")
    parser.add_argument("--v", type=int, default=64, help="Number of views")
    parser.add_argument("--out", type=str, default=OUT_FOLDER, help="Output folder")
    parser.add_argument("--save", action="store_true", help="Whether to save the outputs")
    args = parser.parse_args()
    main(args)