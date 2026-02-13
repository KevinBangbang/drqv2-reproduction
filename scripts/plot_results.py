"""
Plot DrQ-v2 N-step ablation results.

Scans exp_local/ for experiment directories, reads eval.csv and .hydra/config.yaml,
and generates:
  - figures/main_results.png   (n=3 default config, mean +/- std over seeds)
  - figures/ablation_nstep.png (all n-step variants per task)

Usage:
  python scripts/plot_results.py
  python scripts/plot_results.py --root_dir exp_local --output_dir figures
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


# Paper reference scores (approximate final performance)
PAPER_REF = {
    "cartpole_swingup": 860,
    "walker_walk": 940,
}

TASK_DISPLAY = {
    "cartpole_swingup": "Cartpole Swingup",
    "walker_walk": "Walker Walk",
}

NSTEP_COLORS = {1: "#e74c3c", 3: "#2ecc71", 5: "#3498db", 10: "#9b59b6"}
NSTEP_ORDER = [1, 3, 5, 10]


def load_experiments(root_dir: Path):
    """Scan all experiment directories and return structured data.

    Returns:
        dict: {(task_name, nstep, seed): DataFrame}
    """
    experiments = {}
    # Walk all date subdirectories
    for date_dir in sorted(root_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for exp_dir in sorted(date_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / ".hydra" / "config.yaml"
            eval_path = exp_dir / "eval.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            task_name = cfg.get("task_name")
            nstep = cfg.get("nstep")
            seed = cfg.get("seed")
            if task_name is None or nstep is None or seed is None:
                continue
            df = pd.read_csv(eval_path)
            if df.empty:
                continue
            experiments[(task_name, int(nstep), int(seed))] = df
    return experiments


def aggregate(experiments, task, nstep):
    """Get mean and std of episode_reward across seeds for a (task, nstep)."""
    seed_dfs = []
    for (t, n, s), df in experiments.items():
        if t == task and n == nstep:
            seed_dfs.append(df[["frame", "episode_reward"]].copy())
    if not seed_dfs:
        return None, None, None

    # Align on common frames
    common_frames = seed_dfs[0]["frame"].values
    for df in seed_dfs[1:]:
        common_frames = np.intersect1d(common_frames, df["frame"].values)
    if len(common_frames) == 0:
        return None, None, None

    rewards = []
    for df in seed_dfs:
        mask = df["frame"].isin(common_frames)
        rewards.append(df.loc[mask, "episode_reward"].values)
    rewards = np.array(rewards)
    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)
    return common_frames, mean, std


def fmt_frames(x, _):
    """Format x-axis tick as '0.1M', '0.2M', etc."""
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    elif x >= 1e4:
        return f"{x / 1e6:.2f}M"
    elif x == 0:
        return "0"
    else:
        return f"{int(x)}"


def get_tasks(experiments):
    """Get sorted list of unique tasks."""
    tasks = sorted(set(t for t, _, _ in experiments.keys()))
    # Prefer canonical order
    order = ["cartpole_swingup", "walker_walk"]
    return [t for t in order if t in tasks] + [t for t in tasks if t not in order]


def plot_main_results(experiments, output_dir):
    """Plot n=3 (default) results with paper reference lines."""
    tasks = get_tasks(experiments)
    if not tasks:
        print("No data found for main_results plot.")
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        frames, mean, std = aggregate(experiments, task, nstep=3)
        if frames is None:
            ax.set_title(f"{TASK_DISPLAY.get(task, task)} (no data)")
            continue

        ax.plot(frames, mean, color=NSTEP_COLORS[3], linewidth=2, label="DrQ-v2 (n=3)")
        ax.fill_between(frames, mean - std, mean + std,
                        color=NSTEP_COLORS[3], alpha=0.2)

        # Paper reference
        if task in PAPER_REF:
            ax.axhline(y=PAPER_REF[task], color="gray", linestyle="--",
                       linewidth=1, label=f"Paper ref ({PAPER_REF[task]})")

        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=14)
        ax.set_xlabel("Environment Frames", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_frames))
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("DrQ-v2 Reproduction (n=3, default config)", fontsize=15, y=1.02)
    fig.tight_layout()
    out_path = output_dir / "main_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_ablation(experiments, output_dir):
    """Plot n-step ablation with all variants per task."""
    tasks = get_tasks(experiments)
    if not tasks:
        print("No data found for ablation plot.")
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        has_data = False
        for n in NSTEP_ORDER:
            frames, mean, std = aggregate(experiments, task, nstep=n)
            if frames is None:
                continue
            has_data = True
            lw = 2.5 if n == 3 else 1.5
            label = f"n={n} (default)" if n == 3 else f"n={n}"
            ax.plot(frames, mean, color=NSTEP_COLORS[n], linewidth=lw, label=label)
            ax.fill_between(frames, mean - std, mean + std,
                            color=NSTEP_COLORS[n], alpha=0.15)

        if not has_data:
            ax.set_title(f"{TASK_DISPLAY.get(task, task)} (no data)")
            continue

        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=14)
        ax.set_xlabel("Environment Frames", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_frames))
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("N-step Return Ablation", fontsize=15, y=1.02)
    fig.tight_layout()
    out_path = output_dir / "ablation_nstep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot DrQ-v2 ablation results")
    parser.add_argument("--root_dir", type=str, default="exp_local",
                        help="Root directory containing experiment outputs")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save figures")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)

    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root_dir} for experiments...")
    experiments = load_experiments(root_dir)
    print(f"Found {len(experiments)} experiment runs:")
    summary = defaultdict(list)
    for (task, nstep, seed) in sorted(experiments.keys()):
        summary[(task, nstep)].append(seed)
    for (task, nstep), seeds in sorted(summary.items()):
        print(f"  {task}  n={nstep}  seeds={seeds}")

    print("\nGenerating plots...")
    plot_main_results(experiments, output_dir)
    plot_ablation(experiments, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
