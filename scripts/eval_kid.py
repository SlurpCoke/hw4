"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda

The script prints a markdown table with KID mean ± std for each
(method, num_steps) combination.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)  # [0,1]
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


def generate_samples_batch(
    method: str,
    n_samples: int,
    step_count: int,
    sde: VPSDE,
    vp_model: torch.nn.Module,
    flow: RectifiedFlow,
    rf_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Generate n_samples using the specified method and step count."""
    shape_per_batch = (batch_size, 1, 28, 28)
    all_samples = []
    generated = 0

    while generated < n_samples:
        bs = min(batch_size, n_samples - generated)
        shape = (bs, 1, 28, 28)

        if method == "rectflow":
            batch = flow.euler_sample(rf_model, shape, num_steps=step_count, device=device)
        elif method == "ddim":
            # DDIM is deterministic — use EM with no noise (zero-noise limit)
            # Implement deterministic reverse ODE: dx = [½β(t)x - β(t)*eps/σ(t)] dt
            @torch.no_grad()
            def ddim_sample(sde, model, shape, num_steps, device):
                dt = 1.0 / num_steps
                B = shape[0]
                t1 = torch.ones(B, device=device)
                sigma_1 = sde.sigma(t1)
                x = sigma_1[:, None, None, None] * torch.randn(shape, device=device)
                ts = torch.linspace(1.0, dt, num_steps, device=device)
                for t_val in ts:
                    t = torch.full((B,), t_val.item(), device=device)
                    bt = sde.beta(t)
                    st = sde.sigma(t)
                    eps_pred = model(x, t)
                    bt4 = bt[:, None, None, None]
                    st4 = st[:, None, None, None]
                    # Deterministic ODE (no stochastic term)
                    drift = 0.5 * bt4 * x - bt4 * eps_pred / st4
                    x = x + drift * dt
                return x

            batch = ddim_sample(sde, vp_model, shape, step_count, device)
        elif method == "em":
            batch = sde.euler_maruyama(vp_model, shape, num_steps=step_count, device=device)
        else:
            raise ValueError(f"Unknown method: {method}")

        all_samples.append(batch.cpu())
        generated += bs

    return torch.cat(all_samples, dim=0)[:n_samples]


def main():
    args = get_args()
    device = torch.device(args.device)

    # Load models
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()

    flow = RectifiedFlow()
    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()

    # Save real FashionMNIST images for KID computation
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    real_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    real_images = torch.stack([real_ds[i][0] for i in range(min(args.n_samples, len(real_ds)))])

    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        save_samples_to_dir(real_images, real_dir)

        # Table header
        print(f"\n{'Steps':<8} | {'Flow Matching':<25} | {'DDIM':<25} | {'DDPM EM':<25}")
        print("-" * 90)

        results = {}
        for step_count in STEP_COUNTS:
            row = {"steps": step_count}
            for method in METHODS:
                # EM with 1000 steps serves as "baseline" (unavailable for other step counts)
                if method == "em" and step_count < 1000:
                    row[method] = "—"
                    continue

                gen_dir = os.path.join(tmpdir, f"{method}_{step_count}")
                samples = generate_samples_batch(
                    method, args.n_samples, step_count,
                    sde, vp_model, flow, rf_model, device,
                )
                save_samples_to_dir(samples, gen_dir)

                try:
                    metrics = compute_kid(gen_dir, real_dir)
                    kid_mean = metrics.get("kernel_inception_distance_mean", float("nan"))
                    kid_std = metrics.get("kernel_inception_distance_std", float("nan"))
                    row[method] = f"{kid_mean:.4f} ± {kid_std:.4f}"
                except Exception as e:
                    row[method] = f"ERR: {e}"

            method_map = {"rectflow": "Flow Matching", "ddim": "DDIM", "em": "DDPM EM"}
            print(
                f"{step_count:<8} | "
                f"{row.get('rectflow', '—'):<25} | "
                f"{row.get('ddim', '—'):<25} | "
                f"{row.get('em', '—'):<25}"
            )
            results[step_count] = row

    print("\nNote: DDPM EM at 1000 steps is the baseline.")


if __name__ == "__main__":
    main()
