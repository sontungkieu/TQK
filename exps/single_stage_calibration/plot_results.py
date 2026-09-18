#!/usr/bin/env python3
"""Generate the six pre-registered Phase-1 figures from search-split evidence."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP / "plots"))
from orx_figstyle import (  # noqa: E402
    BASELINE,
    COLUMN,
    DIVERGING,
    MUTED,
    PALETTE,
    SEQUENTIAL,
    TEXT,
    annotate_matrix,
    figure,
    figure_grid,
    save,
    use_style,
)


def save_figure(fig, name: str) -> None:
    outputs = save(fig, str(EXP / "plots" / name))
    print(f"figure {name}: {outputs}")


def main() -> None:
    use_style()
    search = pd.read_csv(EXP / "replay/search_results.csv")
    oracle = pd.read_csv(EXP / "replay/oracle_curve_search.csv")
    reliability = pd.read_csv(EXP / "replay/ranking_reliability_search.csv")
    psp = float(search.psp_IR_mean.iloc[0])
    colors = {1: PALETTE["blue"], 2: PALETTE["orange"], 3: PALETTE["green"]}

    fig, ax = figure(width=COLUMN, ratio=0.76)
    for k, group in search.groupby("K"):
        group = group.sort_values("checkpoint")
        ax.plot(group.checkpoint, group.final_IR_mean, marker="o", color=colors[k], label=f"K={k}")
    ax.axhline(psp, color=BASELINE, linestyle="--", label="PSP")
    ax.set_xlabel("Pruning checkpoint (denoising step)")
    ax.set_ylabel("Mean final ImageReward Q")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "a_quality_frontier")

    fig, ax = figure(width=COLUMN, ratio=0.76)
    ax.plot(oracle.M, oracle.oracle_IR, marker="o", color=PALETTE["blue"])
    ax.set_xlabel("Initial pool size M (candidates)")
    ax.set_ylabel("Oracle pool quality O(M)")
    save_figure(fig, "b_oracle_quality")

    fig, ax = figure(width=COLUMN, ratio=0.76)
    for k, group in search.groupby("K"):
        group = group.sort_values("checkpoint")
        ax.plot(group.checkpoint, group.selection_regret_mean, marker="o", color=colors[k], label=f"K={k}")
    ax.set_xlabel("Pruning checkpoint (denoising step)")
    ax.set_ylabel("Mean selection regret R")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, "c_selection_regret")

    fig, axes = figure_grid(nrows=1, ncols=3, width=TEXT, ratio=0.34, sharey=True)
    for panel, (k, group) in zip(axes, search.groupby("K")):
        group = group.sort_values("checkpoint")
        x = group.checkpoint.to_numpy()
        q = group.final_IR_mean.to_numpy()
        o = group.oracle_IR.to_numpy()
        panel.plot(x, o, color=PALETTE["orange"], marker="o", label="O(M)")
        panel.plot(x, q, color=PALETTE["blue"], marker="s", label="Q")
        panel.fill_between(x, q, o, color=MUTED, alpha=0.35, label="R")
        panel.set_xlabel("Checkpoint t")
        panel.text(0.04, 0.95, f"K={k}", transform=panel.transAxes, va="top", fontweight="bold")
    axes[0].set_ylabel("Mean ImageReward")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, "d_breadth_regret_decomposition")

    fig, ax = figure(width=COLUMN, ratio=0.76)
    ax.plot(reliability.checkpoint, reliability.mean_spearman, marker="o", color=PALETTE["blue"], label="Spearman")
    ax.plot(reliability.checkpoint, reliability.mean_kendall, marker="s", color=PALETTE["green"], label="Kendall τ")
    ax.set_xlabel("Checkpoint t (denoising step)")
    ax.set_ylabel("Mean prompt-level rank correlation")
    ax.legend(frameon=False)
    save_figure(fig, "e_ranking_reliability")

    checkpoints = sorted(search.checkpoint.unique())
    ks = sorted(search.K.unique())
    grid = np.full((len(ks), len(checkpoints)), np.nan)
    for _, row in search.iterrows():
        grid[ks.index(int(row.K)), checkpoints.index(int(row.checkpoint))] = row.delta_vs_psp
    limit = float(np.nanmax(np.abs(grid)))
    fig, ax = figure(width=TEXT, ratio=0.36)
    image = ax.imshow(grid, cmap=DIVERGING, vmin=-limit, vmax=limit, interpolation="nearest", aspect="auto")
    annotate_matrix(ax, grid, fmt="{:+.3f}", image=image)
    best = search.sort_values("final_IR_mean", ascending=False).iloc[0]
    best_row, best_col = ks.index(int(best.K)), checkpoints.index(int(best.checkpoint))
    ax.add_patch(Rectangle((best_col - 0.5, best_row - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.4))
    ax.set_xticks(range(len(checkpoints)), [str(value) for value in checkpoints])
    ax.set_yticks(range(len(ks)), [str(value) for value in ks])
    ax.set_xlabel("Pruning checkpoint t")
    ax.set_ylabel("Final survivors K")
    ax.grid(visible=False)
    fig.colorbar(image, ax=ax, label="Mean Δ ImageReward vs PSP", fraction=0.046, pad=0.03)
    save_figure(fig, "f_schedule_heatmap")


if __name__ == "__main__":
    main()
