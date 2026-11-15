"""Plot frozen budget-vs-regret data without touching model selection."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory / "selection.json").read_text())["summary"]
    panels = (
        ("new-shape", "New shape"),
        ("new-structure", "New dependency graph"),
        ("new-structure-and-order", "New graph and candidate orders"),
    )
    methods = {
        "direct-search": ("Direct search", "#36759c", "o"),
        "compose-dag": ("ComposeDES, functional DAG", "#777a80", "x"),
        "order-regression": ("Order-feature regression", "#188254", "D"),
        "compose-admission": ("ComposeDES, admission model", "#b45331", "s"),
    }
    budgets = ("0", "1", "2", "4", "8", "16", "all")
    x = range(len(budgets))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.25), sharey=True)
    for axis, (key, title) in zip(axes, panels):
        entry = data[key]
        for method, (label, color, marker) in methods.items():
            axis.plot(x, [entry["budget"][method][b] for b in budgets], label=label,
                      color=color, marker=marker, ms=4.5, lw=1.5)
        axis.axhline(entry["zero"]["fixed-global"]["mean_regret_pct"],
                     color="#333333", ls=":", label="Frozen fixed combination")
        axis.set_title(f"{title} (n={entry['n']})", fontsize=11)
        axis.set_xticks(x, budgets)
        axis.grid(axis="y", alpha=.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean gap to measured best (%)")
    axes[0].set_ylim(bottom=-.06)
    fig.supxlabel("Target configurations measured (16 replays per configuration)", y=.15)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=.27, top=.88, left=.07, right=.98, wspace=.16)
    fig.savefig(args.directory / "budget-regret.png", dpi=190)


if __name__ == "__main__":
    main()
