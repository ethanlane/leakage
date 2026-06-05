#!/usr/bin/env python3
import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def require_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}. Existing columns: {list(df.columns)}")


def apply_filters(df, args):
    if "scored" in df.columns:
        df = df[df["scored"] == True].copy()

    if args.require_tests_pass:
        require_columns(df, ["original_test_passed", "perturbed_test_passed"])
        df = df[
            (df["original_test_passed"] == True)
            & (df["perturbed_test_passed"] == True)
        ].copy()

    if args.same_token_count:
        require_columns(df, ["orig_num_tokens", "sub_num_tokens"])
        df = df[df["orig_num_tokens"] == df["sub_num_tokens"]].copy()

    if args.drop_negative_raw:
        require_columns(df, ["orig_mean_nll", "sub_mean_nll"])
        df = df[
            (df["orig_mean_nll"].astype(float) >= 0)
            & (df["sub_mean_nll"].astype(float) >= 0)
        ].copy()

    return df


def plot_orig_vs_sub(df, args):
    require_columns(df, ["orig_mean_nll", "sub_mean_nll"])

    orig = df["orig_mean_nll"].dropna().astype(float)
    sub = df["sub_mean_nll"].dropna().astype(float)

    orig_mean = orig.mean()
    sub_mean = sub.mean()
    diff = sub_mean - orig_mean

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(
        orig,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Original variable mean NLL, mean={orig_mean:.3f}",
    )

    plt.hist(
        sub,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Substitute variable mean NLL, mean={sub_mean:.3f}",
    )

    plt.axvline(
        orig_mean,
        linestyle="--",
        linewidth=2,
        label=f"Original mean={orig_mean:.3f}",
    )

    plt.axvline(
        sub_mean,
        linestyle=":",
        linewidth=2,
        label=f"Substitute mean={sub_mean:.3f}",
    )

    title = args.title if args.title else os.path.basename(args.input)
    plt.title(title)
    plt.xlabel("Variable-name mean NLL")
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
    print(f"Original variable mean NLL = {orig_mean:.6f}")
    print(f"Substitute variable mean NLL = {sub_mean:.6f}")
    print(f"Mean difference, substitute - original = {diff:.6f}")
    print("Positive difference means the model prefers the original variable name.")


def plot_delta(df, args):
    require_columns(df, ["delta_mean_nll"])

    delta = df["delta_mean_nll"].dropna().astype(float)

    delta_mean = delta.mean()
    delta_median = delta.median()
    frac_positive = (delta > 0).mean()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(
        delta,
        bins=args.bins,
        alpha=0.75,
        density=args.density,
        label=f"Delta mean NLL, mean={delta_mean:.3f}",
    )

    plt.axvline(
        0,
        linestyle="-",
        linewidth=1.5,
        label="No preference",
    )

    plt.axvline(
        delta_mean,
        linestyle="--",
        linewidth=2,
        label=f"Mean={delta_mean:.3f}",
    )

    plt.axvline(
        delta_median,
        linestyle=":",
        linewidth=2,
        label=f"Median={delta_median:.3f}",
    )

    title = args.title if args.title else os.path.basename(args.input)
    plt.title(title)
    plt.xlabel(
        "Delta mean NLL: "
        "mean NLL(substitute variable) - mean NLL(original variable)"
    )
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
    print(f"Mean delta mean NLL = {delta_mean:.6f}")
    print(f"Median delta mean NLL = {delta_median:.6f}")
    print(f"Fraction positive delta mean NLL = {frac_positive:.6f}")
    print("Positive delta means the model prefers the original variable name.")


def plot_side_by_side_delta(df1, df2, args):
    require_columns(df1, ["delta_mean_nll"])
    require_columns(df2, ["delta_mean_nll"])

    delta1 = df1["delta_mean_nll"].dropna().astype(float)
    delta2 = df2["delta_mean_nll"].dropna().astype(float)

    mean1 = delta1.mean()
    mean2 = delta2.mean()
    median1 = delta1.median()
    median2 = delta2.median()
    frac1 = (delta1 > 0).mean()
    frac2 = (delta2 > 0).mean()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(
        delta1,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"{args.label1}, mean={mean1:.3f}",
    )

    plt.hist(
        delta2,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"{args.label2}, mean={mean2:.3f}",
    )

    plt.axvline(
        0,
        linestyle="-",
        linewidth=1.5,
        label="No preference",
    )

    plt.axvline(
        mean1,
        linestyle="--",
        linewidth=2,
        label=f"{args.label1} mean={mean1:.3f}",
    )

    plt.axvline(
        mean2,
        linestyle=":",
        linewidth=2,
        label=f"{args.label2} mean={mean2:.3f}",
    )

    title = args.title if args.title else "Delta mean NLL comparison"
    plt.title(title)
    plt.xlabel(
        "Delta mean NLL: "
        "mean NLL(substitute variable) - mean NLL(original variable)"
    )
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
    print(f"{args.label1}: n={len(df1)}, mean={mean1:.6f}, median={median1:.6f}, frac_positive={frac1:.6f}")
    print(f"{args.label2}: n={len(df2)}, mean={mean2:.6f}, median={median2:.6f}, frac_positive={frac2:.6f}")
    print(f"Mean difference, {args.label1} - {args.label2} = {mean1 - mean2:.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Result JSONL, e.g. quixbugs_var_scores.jsonl",
    )
    parser.add_argument(
        "--input2",
        default=None,
        help="Optional second result JSONL for comparison plot.",
    )
    parser.add_argument("--out", required=True, help="Output plot path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--density", action="store_true")

    parser.add_argument(
        "--plot_type",
        choices=["orig_vs_sub", "delta", "compare_delta"],
        default="orig_vs_sub",
        help=(
            "orig_vs_sub overlays original/substitute mean NLL. "
            "delta plots delta_mean_nll for one file. "
            "compare_delta overlays delta_mean_nll from two files."
        ),
    )

    parser.add_argument(
        "--require_tests_pass",
        action="store_true",
        help="Only keep rows where original_test_passed and perturbed_test_passed are both True.",
    )

    parser.add_argument(
        "--same_token_count",
        action="store_true",
        help="Only keep rows where orig_num_tokens == sub_num_tokens.",
    )

    parser.add_argument(
        "--drop_negative_raw",
        action="store_true",
        help="Drop rows with negative raw orig/sub mean NLL. Should normally remove nothing with the fixed scorer.",
    )

    parser.add_argument("--label1", default="Dataset 1")
    parser.add_argument("--label2", default="Dataset 2")

    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    args = parser.parse_args()

    df = load_jsonl(args.input)
    df = apply_filters(df, args)

    if len(df) == 0:
        raise ValueError("No rows left after filtering for input.")

    if args.plot_type == "orig_vs_sub":
        plot_orig_vs_sub(df, args)

    elif args.plot_type == "delta":
        plot_delta(df, args)

    elif args.plot_type == "compare_delta":
        if not args.input2:
            raise ValueError("--input2 is required for --plot_type compare_delta")

        df2 = load_jsonl(args.input2)
        df2 = apply_filters(df2, args)

        if len(df2) == 0:
            raise ValueError("No rows left after filtering for input2.")

        plot_side_by_side_delta(df, df2, args)

    else:
        raise ValueError(f"Unknown plot_type: {args.plot_type}")


if __name__ == "__main__":
    main()