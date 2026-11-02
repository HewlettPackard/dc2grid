"""Visualise mean_reward from a benchmark results CSV."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(_REPO_ROOT / "outputs/sac_default/Rule_macro/benchmarkresults/results.csv"),
        help="Path to benchmark results CSV",
    )
    parser.add_argument("--output", default=str(_REPO_ROOT / "outputs/sac_default/Rule_macro/benchmarkresults/eval_benchmark_rulemacro_v2"), help="Save figure to file instead of showing")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["agent"] != "rule_macro"]
    df["agent"] = df["agent"].str.replace("rule_macro+", "", regex=False)

    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.barplot(
        data=df, x="agent", y="mean_reward",
        palette="viridis", edgecolor="black", linewidth=0.8,
        width=0.5, ax=ax,
    )

    # Add error bars if std column exists
    if "mean_reward_std" in df.columns:
        ax.errorbar(
            range(len(df)), df["mean_reward"], yerr=df["mean_reward_std"],
            fmt="none", color="black", capsize=5, capthick=1.2,
        )

    ax.axhline(0, color="black", linestyle="-", linewidth=1.2, zorder=5)
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Low-level Agent")
    ax.set_title("Evaluation with Rule-based High-level Controller", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    sns.despine(right=True, bottom=True)

    fig.tight_layout()

    if args.output:
        out = Path(args.output).with_suffix(".pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
