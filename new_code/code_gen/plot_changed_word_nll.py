import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_metric(rows, key):
    vals = []
    for r in rows:
        if r.get("score_ok", True) and key in r and r[key] is not None:
            vals.append(float(r[key]))
    return np.array(vals, dtype=float)


def compute_common_limits_and_bins(datasets, bins=30):
    """
    datasets: list of tuples (orig_array, pert_array)
    Return:
      bin_edges, xlim, ylim
    """
    all_vals = []
    for orig, pert in datasets:
        all_vals.extend(orig.tolist())
        all_vals.extend(pert.tolist())

    all_vals = np.array(all_vals, dtype=float)

    xmin = float(np.min(all_vals))
    xmax = float(np.max(all_vals))

    # add a little padding
    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 0.5
    xlow = xmin - xpad
    xhigh = xmax + xpad

    bin_edges = np.linspace(xlow, xhigh, bins + 1)

    ymax = 0.0
    for orig, pert in datasets:
        d1, _ = np.histogram(orig, bins=bin_edges, density=True)
        d2, _ = np.histogram(pert, bins=bin_edges, density=True)
        ymax = max(ymax, float(np.max(d1)), float(np.max(d2)))

    ypad = 0.08 * ymax if ymax > 0 else 0.1
    ylim = (0, ymax + ypad)
    xlim = (xlow, xhigh)

    return bin_edges, xlim, ylim


def plot_dataset(
    orig,
    pert,
    title,
    out_path,
    bin_edges,
    xlim,
    ylim,
):
    orig_mean = float(np.mean(orig))
    pert_mean = float(np.mean(pert))

    plt.figure(figsize=(8, 5.5))

    plt.hist(
        orig,
        bins=bin_edges,
        alpha=0.6,
        density=True,
        label="Original changed-word NLL",
    )
    plt.hist(
        pert,
        bins=bin_edges,
        alpha=0.6,
        density=True,
        label="Substituted changed-word NLL",
    )

    plt.axvline(
        orig_mean,
        linestyle="--",
        linewidth=2,
        label=f"Original mean = {orig_mean:.2f}",
    )
    plt.axvline(
        pert_mean,
        linestyle="--",
        linewidth=2,
        label=f"Substituted mean = {pert_mean:.2f}",
    )

    plt.title(title)
    plt.xlabel("Changed-word NLL")
    plt.ylabel("Density")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--humaneval",
        default="results/humaneval_qwen25coder7b_changed_word_nll.jsonl",
    )
    parser.add_argument(
        "--lcb",
        default="results/lcb_qwen25coder7b_changed_word_nll.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        default="plots",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    he_rows = load_jsonl(args.humaneval)
    lcb_rows = load_jsonl(args.lcb)

    he_orig = extract_metric(he_rows, "orig_changed_sum_nll")
    he_pert = extract_metric(he_rows, "pert_changed_sum_nll")

    lcb_orig = extract_metric(lcb_rows, "orig_changed_sum_nll")
    lcb_pert = extract_metric(lcb_rows, "pert_changed_sum_nll")

    datasets = [
        (he_orig, he_pert),
        (lcb_orig, lcb_pert),
    ]

    bin_edges, xlim, ylim = compute_common_limits_and_bins(
        datasets,
        bins=args.bins,
    )

    plot_dataset(
        he_orig,
        he_pert,
        title="HumanEval: Original vs Substituted Changed-word NLL",
        out_path=out_dir / "humaneval_changed_word_nll_dist.png",
        bin_edges=bin_edges,
        xlim=xlim,
        ylim=ylim,
    )

    plot_dataset(
        lcb_orig,
        lcb_pert,
        title="LiveCodeBench: Original vs Substituted Changed-word NLL",
        out_path=out_dir / "lcb_changed_word_nll_dist.png",
        bin_edges=bin_edges,
        xlim=xlim,
        ylim=ylim,
    )

    summary = {
        "humaneval_n": len(he_orig),
        "lcb_n": len(lcb_orig),
        "common_xlim": xlim,
        "common_ylim": ylim,
        "humaneval_orig_mean": float(np.mean(he_orig)),
        "humaneval_pert_mean": float(np.mean(he_pert)),
        "lcb_orig_mean": float(np.mean(lcb_orig)),
        "lcb_pert_mean": float(np.mean(lcb_pert)),
    }

    with open(out_dir / "plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()