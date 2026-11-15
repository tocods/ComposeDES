"""Plot shared-point learning and end-to-end adaptive search results."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    stats = json.loads((args.directory/"active-results.json").read_text())
    results = {(r["protocol"],r["test"],r["method"],r["budget"]):r for r in stats["summary"]}
    supplemental_file = args.directory/"acquisition-controls.json"
    supplemental = ("engine-order-search", "static-top-score") if supplemental_file.is_file() else ()
    if supplemental:
        controls = json.loads(supplemental_file.read_text())["summary"]
        results.update({("active",r["test"],r["method"],r["budget"]):r
                        for r in controls if r["method"] in supplemental})
    methods = {
        "concurrent-expanded": ("Concurrent admission", "#ab482f", "s"),
        "concurrent-expanded-frozen": ("Admission, frozen weights", "#ac8277", "^"),
        "original-expanded": ("Original admission", "#8064a2", "v"),
        "regression-expanded": ("Score regression", "#188254", "D"),
        "direct-adaptive": ("Adaptive direct search", "#36759c", "o"),
        "engine-order-search": ("Engine-first direct search*", "#304451", "x"),
        "static-top-score": ("Frozen-prior search*", "#c4912b", "P"),
    }
    fig, axes = plt.subplots(1,2,figsize=(10.6,4.1),sharey=False)
    panels = (
        ("shared","unseen_regret","Shared observations, unseen choice",
         ("concurrent-expanded","concurrent-expanded-frozen","original-expanded","regression-expanded")),
        ("active","regret","Equal budget, adaptive measurements",
         ("concurrent-expanded","original-expanded","regression-expanded","direct-adaptive")+supplemental),
    )
    budgets = (0,1,2,4,8)
    ticks = list(range(len(budgets)))
    legends = {}
    for axis, (protocol,col,title,items) in zip(axes,panels):
        for method in items:
            label,color,marker = methods[method]
            plot, = axis.plot(ticks,[results[protocol,"cross-process",method,n][col]
                                    for n in budgets],label=label,color=color,marker=marker,lw=1.5)
            legends[label] = plot
        axis.set_title(title)
        axis.set_xticks(ticks,budgets)
        axis.grid(axis="y",alpha=.25)
        axis.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Gap to best unmeasured candidate (%)")
    axes[1].set_ylabel("Gap to best candidate (%)")
    fig.supxlabel("Number of target configurations measured (8 replays each)",y=.22)
    fig.legend(legends.values(),legends.keys(),ncol=3,loc="lower center",
               bbox_to_anchor=(.5,.032),frameon=False,fontsize=8.5)
    if supplemental:
        fig.text(.5,.012,"* Supplementary controls, evaluated after the frozen comparison.",
                 ha="center",fontsize=8)
    fig.subplots_adjust(bottom=.32,top=.89,left=.08,right=.98,wspace=.32)
    fig.savefig(args.directory/"active-budget.png",dpi=190)


if __name__ == "__main__":
    main()
