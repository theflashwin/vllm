"""Resume-TTFT CDFs for live replay arms, with p50/p90/p99 marked.

Each --arm is LABEL=DIR[,DIR...]; the runs listed for one label are pooled
(e.g. both run orders). The left panel is the whole CDF on a log x-axis, the
right panel the tail above p90.

Example:
    python analysis/live_cdf.py -o figures/live35_policy.png \
        --arm "LRU=results/live/live35_6971095/tier_lru,results/live/live35_rev/tier_lru" \
        --arm "Ours=results/live/live35_6971095/tier_reuse,results/live/live35_rev/tier_reuse"
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

# Categorical slots 1-4 of the dataviz reference palette, light mode.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
QUANTILES = (0.5, 0.9, 0.99)
X_TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


def resume_ttfts(run_dir: str) -> list[float]:
    with open(Path(run_dir) / "requests.csv") as f:
        return [
            float(r["ttft_s"])
            for r in csv.DictReader(f)
            if r["kind"] == "turn" and int(r["turn"]) > 0 and r["ttft_s"]
        ]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Resume TTFT (s, log scale)", color=MUTED, fontsize=9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", action="append", required=True, help="LABEL=DIR[,DIR]")
    p.add_argument("--title", default="")
    p.add_argument(
        "--paired",
        action="store_true",
        help="arms come in pairs: same color, second one dashed",
    )
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    arms = []
    for spec in args.arm:
        label, _, dirs = spec.partition("=")
        x = np.sort([t for d in dirs.split(",") for t in resume_ttfts(d)])
        arms.append((label, x, len(dirs.split(","))))

    fig, (full, tail) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    print(f"{'arm':<28}{'runs':>5}{'n':>7}" + "".join(f"{'p' + str(round(q * 100)):>8}" for q in QUANTILES))
    tail_left = []
    for i, (label, x, runs) in enumerate(arms):
        color = COLORS[i // 2 if args.paired else i]
        dash = "--" if args.paired and i % 2 else "-"
        y = np.arange(1, len(x) + 1) / len(x)
        qs = np.quantile(x, QUANTILES)
        tail_left.append(qs[1])
        print(f"{label:<28}{runs:>5}{len(x):>7}" + "".join(f"{q:>8.3f}" for q in qs))
        for ax in (full, tail):
            ax.step(x, y, where="post", color=color, linewidth=2, linestyle=dash,
                    label=label)
            ax.plot(qs, QUANTILES, "o", color=color, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    for ax in (full, tail):
        style(ax)
    full.set_ylim(0, 1.01)
    full.set_ylabel("Fraction of resumes", color=MUTED, fontsize=9)
    tail.set_ylim(0.9, 1.001)
    tail.set_xlim(left=min(tail_left) * 0.8)
    tail.set_title("Tail above p90", color=INK, fontsize=9, loc="left")
    for q in QUANTILES:
        for ax in (full, tail):
            ax.axhline(q, color=GRID, linewidth=0.8, zorder=0)
        tail.annotate(f"p{round(q * 100)}", (1, q), xycoords=("axes fraction", "data"),
                      xytext=(3, 0), textcoords="offset points", va="center",
                      color=MUTED, fontsize=8)
    full.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    if args.title:
        fig.suptitle(args.title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.16, top=0.83, wspace=0.22)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
