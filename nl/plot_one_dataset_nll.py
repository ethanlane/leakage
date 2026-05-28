import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Result CSV, e.g. results/swords_olmo1b.csv")
    parser.add_argument("--out", required=True, help="Output plot path, e.g. plots/swords_nll.png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--density", action="store_true", help="Plot density instead of raw counts")
    parser.add_argument("--xmax", type=float, default=None, help="Optional max x-axis value")
    parser.add_argument("--xmin", type=float, default=None, help="Optional min x-axis value")
    parser.add_argument("--ymax", type=float, default=None, help="Optional max y-axis value")
    parser.add_argument("--ymin", type=float, default=None, help="Optional min y-axis value")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = ["orig_nll", "sub_nll"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}. Existing columns: {list(df.columns)}")

    orig = df["orig_nll"].dropna().astype(float)
    sub = df["sub_nll"].dropna().astype(float)

    orig_mean = orig.mean()
    sub_mean = sub.mean()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(
        orig,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Original NLL, mean={orig_mean:.3f}",
    )

    plt.hist(
        sub,
        bins=args.bins,
        alpha=0.55,
        density=args.density,
        label=f"Substitute NLL, mean={sub_mean:.3f}",
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
    plt.xlabel("NLL of target token")
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
    print(f"Substitute mean NLL = {sub_mean:.6f}")
    print(f"Mean difference, sub - orig = {sub_mean - orig_mean:.6f}")


if __name__ == "__main__":
    main()