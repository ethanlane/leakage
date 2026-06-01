import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def get_nll_pair(df, metric):
    if metric == "independent":
        required = ["orig_two_logp", "sub_two_independent_logp"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")

        orig = -df["orig_two_logp"].dropna().astype(float)
        changed = -df["sub_two_independent_logp"].dropna().astype(float)
        xlabel = "Two changed-token NLL"
        subtitle = "Independent original-prefix score"

    elif metric == "sequential":
        required = ["orig_two_logp", "sub_two_sequential_logp"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")

        orig = -df["orig_two_logp"].dropna().astype(float)
        changed = -df["sub_two_sequential_logp"].dropna().astype(float)
        xlabel = "Two changed-token NLL"
        subtitle = "Sequential substituted-context score"

    elif metric == "whole":
        required = ["orig_context_mean_nll", "sub_context_mean_nll"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")

        orig = df["orig_context_mean_nll"].dropna().astype(float)
        changed = df["sub_context_mean_nll"].dropna().astype(float)
        xlabel = "Average full-sentence NLL"
        subtitle = "Whole-sentence average NLL"

    else:
        raise ValueError("metric must be one of: independent, sequential, whole")

    return orig, changed, xlabel, subtitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", choices=["independent", "sequential", "whole"], required=True)
    parser.add_argument("--dataset_label", default="Dataset")
    parser.add_argument("--title", default=None)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--density", action="store_true")

    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    orig, changed, xlabel, subtitle = get_nll_pair(df, args.metric)

    orig_mean = orig.mean()
    changed_mean = changed.mean()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(
        orig,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Unchanged/original, mean={orig_mean:.3f}",
    )

    plt.hist(
        changed,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Changed/substituted, mean={changed_mean:.3f}",
    )

    plt.axvline(
        orig_mean,
        linestyle="--",
        linewidth=2,
        label=f"Original mean={orig_mean:.3f}",
    )

    plt.axvline(
        changed_mean,
        linestyle=":",
        linewidth=2,
        label=f"Changed mean={changed_mean:.3f}",
    )

    title = args.title if args.title else f"{args.dataset_label}: {subtitle}"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density" if args.density else "Count")

    if args.xmin is not None or args.xmax is not None:
        plt.xlim(left=args.xmin, right=args.xmax)

    if args.ymin is not None or args.ymax is not None:
        plt.ylim(bottom=args.ymin, top=args.ymax)

    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"Saved plot to: {args.out}")
    print(f"n = {len(df)}")
    print(f"Original mean NLL = {orig_mean:.6f}")
    print(f"Changed mean NLL = {changed_mean:.6f}")
    print(f"Changed - original mean NLL = {changed_mean - orig_mean:.6f}")


if __name__ == "__main__":
    main()