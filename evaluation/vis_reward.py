"""Plot ep/mean_reward from an episode_metrics CSV."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="c2g-bench-neurips2026/outputs/sac_default/default_macro/seed_200_best/2026-04-23_20-06-52/episode_metrics.csv",
        help="Path to episode_metrics.csv",
    )
    parser.add_argument("--output", default="c2g-bench-neurips2026/outputs/sac_default/default_macro/seed_200_best/2026-04-23_20-06-52/mean_reward_defaulthigh_saclow", help="Save figure to file instead of showing")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw data with low opacity
    ax.plot(
        df["episode"], df["ep/mean_reward"],
        marker="o", markersize=8, alpha=0.35,
        linewidth=0.8, color=sns.color_palette()[0],
        label="Per-episode",
    )
    # Smoothed trend (rolling mean)
    window = max(1, len(df) // 20)
    smoothed = df["ep/mean_reward"].rolling(window, min_periods=1, center=True).mean()
    ax.plot(
        df["episode"], smoothed,
        linewidth=2.5, color="tab:blue",
        label=f"Rolling mean",
    )

    ax.axhline(0, color="black", linestyle="-", linewidth=1.2, zorder=5)
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Episode")
    ax.set_title("Default High level - SAC Low level", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=11)
    sns.despine(right=True, bottom=True)

    fig.tight_layout()

    if args.output:
        from pathlib import Path
        out = Path(args.output).with_suffix(".pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
