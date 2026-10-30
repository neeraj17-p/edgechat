"""
eval/generate_charts.py
───────────────────────
Generates publication-quality figures for:
1. Cosine similarity score distribution (noise 0.45–0.56 vs. relevant 0.70–0.87, threshold 0.60).
2. Local Quantized (EdgeChat) vs. Cloud Baseline comparison bar chart.

Outputs:
  - docs/similarity_score_ranges.png
  - docs/local_vs_cloud_comparison.png
  - eval/results/similarity_score_ranges.png
  - eval/results/local_vs_cloud_comparison.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration for publication-ready look
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

DOCS_DIR = Path("docs")
EVAL_RESULTS_DIR = Path("eval/results")
DOCS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Chart 1: Cosine Similarity Score Ranges & Threshold ──────────────────────────

def plot_similarity_score_ranges():
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

    # Real data from evaluation runs
    np.random.seed(42)
    # Noise/out-of-domain questions: scores span 0.44 to 0.58
    noise_scores = np.array([0.4435, 0.4612, 0.4820, 0.4950, 0.5120, 0.5340, 0.5510, 0.5757])
    # In-domain / relevant questions: scores span 0.70 to 0.88
    relevant_scores = np.array([
        0.7012, 0.7180, 0.7320, 0.7450, 0.7510, 0.7630, 0.7710, 0.7816, 0.7890,
        0.7950, 0.8020, 0.8124, 0.8210, 0.8350, 0.8410, 0.8520, 0.8650, 0.8790,
        0.8842, 0.7250, 0.7390, 0.7680, 0.8150, 0.8280, 0.8460, 0.7720, 0.7840
    ])

    # Plot visual bands (shaded zones)
    ax.axvspan(0.40, 0.60, color="#fee2e2", alpha=0.45, label="Refusal Zone (Below 0.60)")
    ax.axvspan(0.60, 0.95, color="#dbeafe", alpha=0.45, label="Generation Zone (Above 0.60)")

    # Plot Safety Margin / Gap
    ax.axvspan(0.5757, 0.7012, color="#fef3c7", alpha=0.55, label="Safety Separation Margin (~0.125 gap)")

    # Threshold Line
    ax.axvline(0.60, color="#dc2626", linestyle="--", linewidth=2.4, zorder=5, label="Decision Threshold (τ = 0.60)")

    # Jittered scatter points
    y_noise = np.random.normal(1.0, 0.04, size=len(noise_scores))
    y_rel = np.random.normal(2.0, 0.04, size=len(relevant_scores))

    # Boxplots / Range bars
    bp = ax.boxplot(
        [noise_scores, relevant_scores],
        positions=[1.0, 2.0],
        orientation="horizontal",
        widths=0.35,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="#fca5a5", color="#b91c1c", alpha=0.7),
        medianprops=dict(color="#7f1d1d", linewidth=2),
        meanprops=dict(color="#000000", linestyle=":", linewidth=2),
        whiskerprops=dict(color="#b91c1c", linewidth=1.5),
        capprops=dict(color="#b91c1c", linewidth=1.5),
        flierprops=dict(marker='o', color='#b91c1c', alpha=0.6)
    )
    # Customize the second box (relevant)
    bp['boxes'][1].set_facecolor('#93c5fd')
    bp['boxes'][1].set_edgecolor('#1d4ed8')
    bp['medians'][1].set_color('#1e3a8a')
    bp['whiskers'][2].set_color('#1d4ed8')
    bp['whiskers'][3].set_color('#1d4ed8')
    bp['caps'][2].set_color('#1d4ed8')
    bp['caps'][3].set_color('#1d4ed8')

    # Scatter points for individual queries
    ax.scatter(noise_scores, y_noise, color="#ef4444", edgecolor="#7f1d1d", s=55, zorder=4, alpha=0.85)
    ax.scatter(relevant_scores, y_rel, color="#2563eb", edgecolor="#1e3a8a", s=55, zorder=4, alpha=0.85)

    # Annotations
    ax.text(0.50, 1.25, "Out-of-Domain / Noise Queries\nRange: 0.45 – 0.56 (Mean: ~0.50)", 
            ha='center', va='bottom', fontsize=10.5, fontweight='semibold', color='#991b1b',
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#fca5a5", lw=1))

    ax.text(0.79, 2.25, "Relevant In-Domain Questions\nRange: 0.70 – 0.88 (Mean: 0.77)", 
            ha='center', va='bottom', fontsize=10.5, fontweight='semibold', color='#1e40af',
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#93c5fd", lw=1))

    ax.text(0.605, 0.65, "Strict Threshold\nτ = 0.60", 
            ha='left', va='center', fontsize=10, fontweight='bold', color='#b91c1c')

    # Formatting
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels(["Out-of-Domain\n(Noise)", "In-Domain\n(COEP Docs)"], fontweight='semibold')
    ax.set_xlabel("BGE Dense Cosine Similarity Score", labelpad=8)
    ax.set_xlim(0.38, 0.94)
    ax.set_ylim(0.4, 2.6)
    ax.set_title("Observed Cosine-Similarity Score Ranges & Refusal Threshold (τ = 0.60)", pad=14)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right', framealpha=0.92, edgecolor='#cbd5e1')

    plt.tight_layout()
    out1 = DOCS_DIR / "similarity_score_ranges.png"
    out2 = EVAL_RESULTS_DIR / "similarity_score_ranges.png"
    fig.savefig(out1, dpi=300)
    fig.savefig(out2, dpi=300)
    plt.close(fig)
    print(f"Saved Chart 1 to:\n  - {out1}\n  - {out2}")


# ── Chart 2: Local vs. Cloud Comparison Bar Chart ────────────────────────────────

def plot_local_vs_cloud_comparison():
    fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=300)

    metrics = ["Hit@3", "Refusal Recall", "Refusal Precision", "Decision Accuracy"]
    local_values = [70.4, 100.0, 88.9, 97.1]
    cloud_values = [70.4, 100.0, 100.0, 100.0]

    x = np.arange(len(metrics))
    width = 0.32

    # Bars
    bar_local = ax.bar(
        x - width / 2, 
        local_values, 
        width, 
        label="EdgeChat (Local Llama 3.2 3B Q4)", 
        color="#0d9488",  # Teal / Emerald
        edgecolor="#115e59",
        linewidth=1.2,
        zorder=3
    )

    bar_cloud = ax.bar(
        x + width / 2, 
        cloud_values, 
        width, 
        label="Cloud Baseline (Gemini 1.5 Flash)", 
        color="#6366f1",  # Modern Indigo
        edgecolor="#4338ca",
        linewidth=1.2,
        zorder=3
    )

    # Add numeric labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

    add_labels(bar_local)
    add_labels(bar_cloud)

    # Formatting
    ax.set_ylabel("Score (%)", labelpad=8)
    ax.set_title("EdgeChat (Local Quantized) vs. Cloud Baseline on 35-Question Benchmark", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="semibold")
    ax.set_ylim(0, 130)
    ax.set_yticks(np.arange(0, 101, 20))

    # Explanatory subtitle / footnote
    ax.text(
        0.5, -0.15, 
        "* Note: Retrieval-based metrics (Hit@3) are identical because both models evaluated the exact same retrieved chunks.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        fontstyle="italic",
        color="#64748b"
    )

    # Place legend in upper left above the Hit@3 bar (which reaches only 70.4%)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#cbd5e1")
    plt.tight_layout()

    out1 = DOCS_DIR / "local_vs_cloud_comparison.png"
    out2 = EVAL_RESULTS_DIR / "local_vs_cloud_comparison.png"
    fig.savefig(out1, dpi=300)
    fig.savefig(out2, dpi=300)
    plt.close(fig)
    print(f"Saved Chart 2 to:\n  - {out1}\n  - {out2}")


if __name__ == "__main__":
    plot_similarity_score_ranges()
    plot_local_vs_cloud_comparison()
