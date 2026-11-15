"""Plot native HiCache request latency and adjacent-pair speedup."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", required=True, type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    train = {}
    for path in (args.curve, args.train):
        if path:
            train.update({key: r["whole-2:whole-8"] for key, r in
                          json.loads(path.read_text())["paired"].items()})
    hold = {}
    if args.heldout:
        hold = {key: r["observed_8"] for key, r in
                json.loads(args.heldout.read_text())["cases"].items()}
    sizes = sorted({int(k.split("/")[0]) for k in train}, reverse=True)
    fig, axes = plt.subplots(2, len(sizes), sharex="col", squeeze=False,
                             figsize=(5.4*len(sizes), 6.4), layout="constrained")
    for col, prefix in enumerate(sizes):
        top, bottom = axes[:, col]
        observed = {**train, **hold}
        points = sorted((int(key.split("/")[1]), row) for key, row in observed.items()
                        if int(key.split("/")[0]) == prefix)
        x_points = [token for token, _ in points]
        for key, color in (("baseline_p50_us", "#234a78"), ("candidate_p50_us", "#ad4917")):
            top.plot(x_points, [r[key]/1e3 for _, r in points], color=color,
                     alpha=.6, linewidth=1.2)
        bottom.plot(x_points, [r["median_gain_us"]/1e3 for _, r in points],
                    color="#217263", alpha=.6, linewidth=1.2)
        for source, data in (("Calibration", train), ("Held out", hold)):
            entries = sorted((int(key.split("/")[1]), row) for key, row in data.items()
                             if int(key.split("/")[0]) == prefix)
            if not entries:
                continue
            x, rows = zip(*entries)
            symbol = "o" if source == "Calibration" else "D"
            style = "none"
            for i, key in enumerate(("baseline_p50_us", "candidate_p50_us")):
                color = ("#234a78", "#ad4917")[i]
                quota = (2, 8)[i]
                top.plot(x, [m[key]/1e3 for m in rows], marker=symbol,
                         color=color, linestyle=style, label=f"{quota} CTA · {source}")
            y = [m["median_gain_us"]/1e3 for m in rows]
            low = [max(0., value-m["block_bootstrap_95_us"][0]/1e3) for value,m in zip(y,rows)]
            high = [max(0., m["block_bootstrap_95_us"][1]/1e3-value) for value,m in zip(y,rows)]
            bottom.errorbar(x, y, yerr=[low, high], marker=symbol, linestyle=style,
                            color="#217263", capsize=3, label=source)
        top.set_title(f"{prefix:,} cached tokens from CPU")
        top.set_ylabel("One-token API latency, median (ms)")
        bottom.axhline(0, linewidth=.85, color="#777")
        bottom.set_ylabel("Paired gain of 8 vs 2 CTA (ms)")
        bottom.set_xlabel("Added tokens")
        for ax in (top, bottom):
            ax.grid(axis="y", alpha=.24)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Native SGLang HiCache · A100 PCIe · Qwen3-14B BF16", fontsize=14)
    fig.savefig(args.output, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
