"""
Visualization of Triton GLiNER Guard benchmark results.
Run: /opt/anaconda3/bin/python3 triton-serve/results/plot_results.py
Outputs: 3 PNG files in triton-serve/results/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "/Users/annareshetnyak/TRITON/triton-serve/results"

STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8f9fb",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#222222",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "text.color":       "#111111",
    "grid.color":       "#dddddd",
    "grid.linestyle":   "--",
    "grid.alpha":       0.8,
    "font.family":      "sans-serif",
    "font.size":        12,
}
plt.rcParams.update(STYLE)

BLUE   = "#2563eb"
ORANGE = "#d97706"
GREEN  = "#16a34a"
RED    = "#dc2626"
PURPLE = "#7c3aed"

# ── График 1: Encoder scaling ──────────────────────────────────────────────────
counts = [1, 2, 4]
rps    = [86.0, 124.2, 134.6]
p50    = [1200, 790, 700]

fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()

ax1.plot(counts, rps, color=BLUE,   marker="o", markersize=9,
         linewidth=2.5, label="RPS", zorder=3)
ax2.plot(counts, p50, color=ORANGE, marker="s", markersize=9,
         linewidth=2.5, linestyle="--", label="P50 latency (ms)", zorder=3)

for x, y in zip(counts, rps):
    ax1.annotate(f"{y}", (x, y), textcoords="offset points",
                 xytext=(8, 6), color=BLUE, fontsize=11, fontweight="bold")
for x, y in zip(counts, p50):
    ax2.annotate(f"{y}ms", (x, y), textcoords="offset points",
                 xytext=(8, -14), color=ORANGE, fontsize=11, fontweight="bold")

ax1.annotate("+44.5%", xy=(1.5, 105), color=GREEN, fontsize=12,
             fontweight="bold", ha="center")
ax1.annotate("+8.4%", xy=(3.0, 130), color=GREEN, fontsize=11,
             fontweight="bold", ha="center")

ax1.set_xlabel("Encoder instance count", labelpad=8)
ax1.set_ylabel("Throughput (RPS)", color=BLUE, labelpad=8)
ax2.set_ylabel("P50 Latency (ms)", color=ORANGE, labelpad=8)
ax1.set_xticks(counts)
ax1.set_ylim(60, 160)
ax2.set_ylim(500, 1500)
ax1.tick_params(axis="y", colors=BLUE)
ax2.tick_params(axis="y", colors=ORANGE)
ax1.grid(True, axis="y")

lines = [
    mpatches.Patch(color=BLUE,   label="RPS"),
    mpatches.Patch(color=ORANGE, label="P50 latency (ms)"),
]
ax1.legend(handles=lines, loc="upper left",
           facecolor="#ffffff", edgecolor="#cccccc", labelcolor="#111111")

plt.title("Encoder scaling: throughput vs latency\n"
          "gliner_ensemble · REST · 100 users · 15 min · A100 80GB",
          pad=14, fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot1_encoder_scaling.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot1_encoder_scaling.png")


# ── График 2: REST vs gRPC tail latency ───────────────────────────────────────
labels    = ["P50", "P95", "P99"]
guard_rest = [590, 1000, 1900]
guard_grpc = [620, 1000, 1600]

x   = np.arange(len(labels))
w   = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - w/2, guard_rest, w, label="REST", color=BLUE,   alpha=0.85)
b2 = ax.bar(x + w/2, guard_grpc, w, label="gRPC", color=PURPLE, alpha=0.85)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25,
            f"{int(bar.get_height())}ms", ha="center", va="bottom",
            fontsize=10, color=BLUE, fontweight="bold")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25,
            f"{int(bar.get_height())}ms", ha="center", va="bottom",
            fontsize=10, color=PURPLE, fontweight="bold")

ax.annotate("", xy=(x[2]+w/2, guard_grpc[2]),
            xytext=(x[2]+w/2, guard_rest[2]),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2))
ax.text(x[2]+w/2 + 0.22, (guard_rest[2]+guard_grpc[2])/2,
        "-16%", color=GREEN, fontsize=12, fontweight="bold", va="center")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel("Latency (ms)", labelpad=8)
ax.set_ylim(0, 2300)
ax.grid(True, axis="y")
ax.legend(facecolor="#ffffff", edgecolor="#cccccc", labelcolor="#111111", fontsize=12)

rps_text = "Throughput: REST 152.2 RPS  ≈  gRPC 148.2 RPS"
ax.text(0.5, 0.96, rps_text, transform=ax.transAxes,
        ha="center", va="top", fontsize=11,
        color="#666666", style="italic")

plt.title("REST vs gRPC: latency percentiles\n"
          "gliner_guard · 100 users · 15 min · A100 80GB",
          pad=14, fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot2_rest_vs_grpc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot2_rest_vs_grpc.png")


# ── График 3: graph_optimization_level effect on tail latency ─────────────────
metrics   = ["P50", "P95", "P99"]
before    = [700,  1100, 1500]
after     = [730,   790,  970]

x = np.arange(len(metrics))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - w/2, before, w, label="enc=4, fp32 (before)", color=BLUE,  alpha=0.85)
b2 = ax.bar(x + w/2, after,  w, label="enc=4 + graph_opt=3",  color=GREEN, alpha=0.85)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{int(bar.get_height())}ms", ha="center", va="bottom",
            fontsize=10, color=BLUE, fontweight="bold")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{int(bar.get_height())}ms", ha="center", va="bottom",
            fontsize=10, color=GREEN, fontweight="bold")

deltas = [(b-a)/b*100 for b, a in zip(before, after)]
for i, (xi, bval, aval, d) in enumerate(zip(x, before, after, deltas)):
    if abs(d) > 3:
        color = GREEN if d > 0 else RED
        sign  = "-" if d > 0 else "+"
        ax.annotate("", xy=(xi+w/2, aval), xytext=(xi+w/2, bval),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2))
        ax.text(xi + w/2 + 0.22, (bval+aval)/2,
                f"{sign}{abs(d):.0f}%", color=color,
                fontsize=12, fontweight="bold", va="center")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13)
ax.set_ylabel("Latency (ms)", labelpad=8)
ax.set_ylim(0, 1700)
ax.grid(True, axis="y")
ax.legend(facecolor="#ffffff", edgecolor="#cccccc", labelcolor="#111111", fontsize=12)

rps_text = "Throughput: 134.6 → 136.1 RPS  (+1.1%)"
ax.text(0.5, 0.96, rps_text, transform=ax.transAxes,
        ha="center", va="top", fontsize=11,
        color="#666666", style="italic")

plt.title("ORT graph_optimization_level=3: tail latency improvement\n"
          "gliner_ensemble · REST · 100 users · 15 min · A100 80GB",
          pad=14, fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot3_graph_opt.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot3_graph_opt.png")

print("\nAll done. Files in:", OUT_DIR)
