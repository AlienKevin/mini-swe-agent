"""Plot eval throughput over time from trajectory file timestamps.

Usage:
    python scripts/plot_throughput.py results/kimi-k2.5-multilingual
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_throughput(results_dir: str, window_minutes: int = 10):
    traj_files = sorted(Path(results_dir).glob("*.traj.json"), key=lambda f: f.stat().st_mtime)
    if not traj_files:
        print(f"No .traj.json files found in {results_dir}")
        sys.exit(1)

    mtimes = np.array([f.stat().st_mtime for f in traj_files])
    t0 = mtimes[0]
    elapsed_hours = (mtimes - t0) / 3600
    cumulative = np.arange(1, len(mtimes) + 1)

    # Rolling throughput: tasks completed in sliding window
    window_sec = window_minutes * 60
    rolling_rate = []
    for t in mtimes:
        count_in_window = np.sum((mtimes >= t - window_sec) & (mtimes <= t))
        rolling_rate.append(count_in_window / window_minutes * 60)  # tasks/hr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Cumulative completions
    ax1.plot(elapsed_hours, cumulative, linewidth=2)
    ax1.set_ylabel("Tasks completed")
    ax1.set_title(f"Eval throughput — {Path(results_dir).name} ({len(traj_files)} tasks)")
    ax1.grid(True, alpha=0.3)

    # Rolling throughput
    ax2.plot(elapsed_hours, rolling_rate, linewidth=2, color="tab:orange")
    ax2.set_xlabel("Elapsed time (hours)")
    ax2.set_ylabel(f"Tasks/hr ({window_minutes}min rolling)")
    ax2.grid(True, alpha=0.3)

    # Overall average
    if elapsed_hours[-1] > 0:
        avg = len(traj_files) / elapsed_hours[-1]
        ax2.axhline(avg, color="tab:red", linestyle="--", alpha=0.7, label=f"Average: {avg:.1f} tasks/hr")
        ax2.legend()

    plt.tight_layout()
    out_path = Path(results_dir) / "throughput.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/kimi-k2.5-multilingual"
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    plot_throughput(results_dir, window)
