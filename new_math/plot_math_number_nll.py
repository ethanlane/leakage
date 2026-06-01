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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Result JSONL, e.g. results/hendrycks_seen_olmo1b.jsonl")
    parser.add_argument("--out", required=True, help="Output plot path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Only keep same-token-count rows")

    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    args = parser.parse_args()

    df = load_jsonl(args.input)

    required = ["orig_num_nll", "pert_num_nll"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}. Existing columns: {list(df.columns)}")

    if args.clean:
        df = df[
            (df["count_mismatch"] == False)
            & (df["orig_num_tokens"] == df["pert_num_tokens"])
        ].copy()

    orig = df["orig_num_nll"].dropna().astype(float)
    pert = df["pert_num_nll"].dropna().astype(float)

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
        label=f"Original number NLL, mean={orig_mean:.3f}",
    )

    plt.hist(
        pert,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Perturbed number NLL, mean={pert_mean:.3f}",
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

    title = args.title if args.title else os.path.basename(args.input)
    plt.title(title)
    plt.xlabel("Number NLL")
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
    print(f"Original number mean NLL = {orig_mean:.6f}")
    print(f"Perturbed number mean NLL = {pert_mean:.6f}")
    print(f"Mean difference, pert - orig = {diff:.6f}")


if __name__ == "__main__":
    main()