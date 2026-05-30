"""
scripts/plot_coefficient.py  —  Part 1.8
=========================================
Plot the DDPM loss coefficient
    β_t² / (2 σ_t² α_t (1 - ᾱ_t))
vs. t on a log-scale y-axis.

Usage::
    python scripts/plot_coefficient.py --T 1000 --beta_start 1e-4 --beta_end 0.02
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


def linear_schedule(T: int, beta_start: float, beta_end: float):
    return np.linspace(beta_start, beta_end, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",          type=int,   default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end",   type=float, default=0.02)
    parser.add_argument("--out",        type=str,   default="coefficient_plot.png")
    args = parser.parse_args()

    T = args.T
    betas = linear_schedule(T, args.beta_start, args.beta_end)    # shape (T,)
    alphas = 1.0 - betas                                           # α_t = 1 - β_t
    alpha_bars = np.cumprod(alphas)                                # ᾱ_t = Π_{s=1}^t α_s

    # σ_t² = β_t  (the simplified choice used in DDPM simplified objective)
    sigma_sq = betas

    # Coefficient: β_t² / (2 σ_t² α_t (1 - ᾱ_t))
    # Note: σ_t² = β_t, so β_t² / (2 β_t α_t (1 - ᾱ_t)) = β_t / (2 α_t (1 - ᾱ_t))
    coeff = betas ** 2 / (2.0 * sigma_sq * alphas * (1.0 - alpha_bars))

    t_vals = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(t_vals, coeff, linewidth=1.5, color="steelblue")
    ax.set_xlabel("$t$", fontsize=13)
    ax.set_ylabel(
        r"$\dfrac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}$",
        fontsize=13,
    )
    ax.set_title("DDPM Loss Coefficient vs. $t$ (log scale)", fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved coefficient plot to {args.out}")


if __name__ == "__main__":
    main()
