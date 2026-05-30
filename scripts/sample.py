"""
scripts/sample.py  —  Generate and compare samples (Parts 5C, 6B, 6D)
=======================================================================

Usage::
    # EM samples  (5.C.iii)
    python scripts/sample.py --method em --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000

    # PC samples  (5.C.iv)
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 1
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 3

    # Rectified Flow Euler  (6.B)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow/best.pt \\
        --num_steps 100

    # One-step reflow  (6.C)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow_reflow/best.pt \\
        --num_steps 1

    # Side-by-side grid  (6.D): pass a fixed seed file
    python scripts/sample.py --method all --vp_checkpoint runs/vp/best.pt \\
        --rf_checkpoint runs/rectflow/best.pt \\
        --reflow_checkpoint runs/rectflow_reflow/best.pt \\
        --seed 42 --out comparison_grid.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def save_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a (B,1,H,W) tensor as an image grid."""
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow)
    plt.figure(figsize=(nrow, samples.size(0) // nrow + 1))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",      type=str, default="em",
                   choices=["em", "pc", "rectflow", "all"],
                   help="Sampler to run (or 'all' for side-by-side grid).")
    # VP checkpoints
    p.add_argument("--checkpoint",    type=str, default=None)
    p.add_argument("--vp_checkpoint", type=str, default=None)
    # Rect-flow checkpoints
    p.add_argument("--rf_checkpoint",     type=str, default=None)
    p.add_argument("--reflow_checkpoint", type=str, default=None)
    # VP schedule
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--T",        type=int,   default=1000)
    # Sampler params
    p.add_argument("--num_steps",   type=int, default=1000)
    p.add_argument("--n_corrector", type=int, default=1)
    p.add_argument("--snr",         type=float, default=0.16)
    p.add_argument("--n_samples",   type=int, default=64)
    # Output
    p.add_argument("--out",    type=str, default="samples.png")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vp_model(
    checkpoint: str,
    device,
    beta_min: float = 0.01,
    beta_max: float = 5.0,
    T: int = 1000,
) -> tuple[VPSDE, UNet]:
    """Load a trained VP score model from checkpoint."""
    sde = VPSDE(beta_min=beta_min, beta_max=beta_max, T=T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return sde, model


def load_rf_model(checkpoint: str, device) -> tuple[RectifiedFlow, UNet]:
    """Load a trained Rectified Flow model from checkpoint."""
    flow = RectifiedFlow()
    model = UNet(in_channels=1, base_channels=64).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return flow, model


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape = (args.n_samples, 1, 28, 28)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.method == "em":
        ckpt = args.checkpoint or args.vp_checkpoint
        assert ckpt, "--checkpoint required for method=em"
        sde, model = load_vp_model(ckpt, device, args.beta_min, args.beta_max, args.T)
        samples = sde.euler_maruyama(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out, title=f"EM Sampler ({args.num_steps} steps)")

    elif args.method == "pc":
        ckpt = args.checkpoint or args.vp_checkpoint
        assert ckpt, "--checkpoint required for method=pc"
        sde, model = load_vp_model(ckpt, device, args.beta_min, args.beta_max, args.T)
        samples = sde.predictor_corrector(
            model, shape,
            num_steps=args.num_steps,
            n_corrector=args.n_corrector,
            snr=args.snr,
            device=device,
        )
        save_grid(
            samples, args.out,
            title=f"PC Sampler ({args.num_steps} steps, {args.n_corrector} corrector steps)"
        )

    elif args.method == "rectflow":
        ckpt = args.checkpoint or args.rf_checkpoint
        assert ckpt, "--checkpoint required for method=rectflow"
        flow, model = load_rf_model(ckpt, device)
        samples = flow.euler_sample(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out, title=f"Rectified Flow ({args.num_steps} steps)")

    elif args.method == "all":
        # Problem 6.D: 4×8 side-by-side grid with same 8 initial noise vectors
        assert args.vp_checkpoint, "--vp_checkpoint required for method=all"
        assert args.rf_checkpoint, "--rf_checkpoint required for method=all"
        assert args.reflow_checkpoint, "--reflow_checkpoint required for method=all"

        n_seeds = 8
        seed_shape = (n_seeds, 1, 28, 28)

        sde, vp_model = load_vp_model(
            args.vp_checkpoint, device, args.beta_min, args.beta_max, args.T
        )
        flow, rf_model = load_rf_model(args.rf_checkpoint, device)
        _, reflow_model = load_rf_model(args.reflow_checkpoint, device)

        # Fix the same 8 initial noise vectors for all methods
        torch.manual_seed(args.seed)
        fixed_noise = torch.randn(seed_shape, device=device)

        all_rows = []

        # Row 1: DDPM EM, 1000 steps
        torch.manual_seed(args.seed)
        row1 = sde.euler_maruyama(vp_model, seed_shape, num_steps=1000, device=device)
        all_rows.append(row1)

        # Row 2: Rect. Flow, 100 steps
        torch.manual_seed(args.seed)
        row2 = flow.euler_sample(rf_model, seed_shape, num_steps=100, device=device)
        all_rows.append(row2)

        # Row 3: Rect. Flow, 1 step (no reflow)
        torch.manual_seed(args.seed)
        row3 = flow.euler_sample(rf_model, seed_shape, num_steps=1, device=device)
        all_rows.append(row3)

        # Row 4: Reflow, 1 step
        torch.manual_seed(args.seed)
        row4 = flow.euler_sample(reflow_model, seed_shape, num_steps=1, device=device)
        all_rows.append(row4)

        # Stack into (4*8, 1, 28, 28) and create a 4×8 grid
        all_samples = torch.cat(all_rows, dim=0)  # (32, 1, 28, 28)
        grid = make_grid(all_samples.clamp(-1, 1) * 0.5 + 0.5, nrow=n_seeds)

        row_labels = [
            "DDPM EM (1000 steps)",
            "Rect. Flow (100 steps)",
            "Rect. Flow (1 step)",
            "Reflow (1 step)",
        ]

        fig, ax = plt.subplots(figsize=(n_seeds * 1.2, 4 * 1.2 + 1))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title("Side-by-Side Qualitative Comparison (4 methods × 8 seeds)", fontsize=12)

        # Add row labels on the left
        img_h = grid.shape[1] // 4
        for i, label in enumerate(row_labels):
            ax.text(
                -5, img_h * i + img_h / 2,
                label, ha="right", va="center", fontsize=8,
                transform=ax.transData,
            )

        plt.tight_layout()
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved side-by-side grid: {args.out}")


if __name__ == "__main__":
    main()
