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

    if args.transform_type:
        require_columns(df, ["transform_type"])
        df = df[df["transform_type"] == args.transform_type].copy()

    if args.require_tests_pass:
        require_columns(df, ["original_test_passed", "perturbed_test_passed"])
        df = df[
            (df["original_test_passed"] == True)
            & (df["perturbed_test_passed"] == True)
        ].copy()

    return df


def plot_original_vs_perturbed(df, args):
    require_columns(df, ["orig_full_mean_nll", "pert_full_mean_nll"])

    orig = df["orig_full_mean_nll"].dropna().astype(float)
    pert = df["pert_full_mean_nll"].dropna().astype(float)

    orig_mean = orig.mean()
    pert_mean = pert.mean()
    diff = pert_mean - orig_mean

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(
        orig,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Original code, mean={orig_mean:.3f}",
    )

    plt.hist(
        pert,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Perturbed code, mean={pert_mean:.3f}",
    )

    plt.axvline(
        orig_mean,
        linestyle="--",
        linewidth=2,
        label=f"Original mean={orig_mean:.3f}",
    )

    plt.axvline(
        pert_mean,
        linestyle=":",
        linewidth=2,
        label=f"Perturbed mean={pert_mean:.3f}",
    )

    title = args.title
    if title is None:
        title = os.path.basename(args.input)
        if args.transform_type:
            title += f" | {args.transform_type}"

    plt.title(title)
    plt.xlabel("Full-code mean NLL")
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
    print(f"Perturbed mean NLL = {pert_mean:.6f}")
    print(f"Mean difference, perturbed - original = {diff:.6f}")
    print("Positive difference means the model prefers the original code.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)

    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--density", action="store_true")

    parser.add_argument(
        "--transform_type",
        default=None,
        help="Optional filter, e.g. comparison_mirror",
    )

    parser.add_argument(
        "--require_tests_pass",
        action="store_true",
        help="Only keep rows where original and perturbed tests both pass.",
    )

    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    args = parser.parse_args()

    df = load_jsonl(args.input)
    df = apply_filters(df, args)

    if len(df) == 0:
        raise ValueError("No rows left after filtering.")

    plot_original_vs_perturbed(df, args)


if __name__ == "__main__":
    main()