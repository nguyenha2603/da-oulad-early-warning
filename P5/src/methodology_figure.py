"""Hình 1.1 – Quy trình dự án theo Data Science Methodology (Rollins), gắn với milestone P1–P5.
Chạy: python3 P5/src/methodology_figure.py  -> P5/figures/p5_methodology.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Times New Roman"
OUT = Path(__file__).resolve().parents[1] / "figures/p5_methodology.png"
NAVY, MUTED = "#1B2A4A", "#55606E"
STAGES = [
    ("P1", "From Problem\nto Approach", "Business Understanding\n→ Analytic Approach",
     "Business goal → SMART\n→ KPI → AQ1–AQ5\n→ phân loại D1–D4", "#E3F2FD"),
    ("P2", "From Requirements\nto Collection", "Data Requirements\n→ Data Collection",
     "AQ–Data Requirement\nMatrix; 5 dataset,\ncổng must-have\n→ chọn OULAD", "#E8F5E9"),
    ("P3", "From Understanding\nto Preparation", "Data Understanding\n→ Data Preparation",
     "23 vấn đề chất lượng;\nthiếu ≠ xóa;\npipeline raw → processed;\nchia train/test theo kỳ", "#FFF3CD"),
    ("P4", "From Modeling\nto Evaluation", "Modeling\n→ Evaluation",
     "Plan chốt trước (SHA-256);\nbaseline M0/M1 → LR;\nPR-AUC, Recall,\nPrecision@200", "#FCE4EC"),
    ("P5", "From Deployment\nto Feedback", "Deployment\n→ Feedback",
     "Dashboard + lý do;\ncố vấn duyệt; giám sát;\nthử nghiệm có đối chứng", "#EDE7F6"),
]

fig, ax = plt.subplots(figsize=(12, 4.6), dpi=200)
ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
w, gap, x0 = 2.12, 0.30, 0.18
for i, (ms, cluster, steps, out, fill) in enumerate(STAGES):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.55), w, 3.55, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec="#B0B8C4", lw=1))
    ax.text(x + w / 2, 3.83, ms, ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(x + w / 2, 3.28, cluster, ha="center", va="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(x + w / 2, 2.45, steps, ha="center", va="center", fontsize=10, style="italic", color=MUTED)
    ax.plot([x + 0.18, x + w - 0.18], [2.0, 2.0], color="#B0B8C4", lw=0.8)
    ax.text(x + w / 2, 1.28, out, ha="center", va="center", fontsize=9.6, color="#222222", linespacing=1.3)
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.02, 2.33), (x + w + gap - 0.02, 2.33),
                                     arrowstyle="-|>", mutation_scale=14, color=NAVY, lw=1.4))
# vòng lặp phản hồi: chạy bên dưới các ô
ax.set_ylim(-0.55, 4.6)
xs, xe = x0 + w / 2, x0 + 4 * (w + gap) + w / 2
ax.plot([xe, xe, xs], [0.53, 0.05, 0.05], color="#C62828", lw=1.3, ls="--")
ax.add_patch(FancyArrowPatch((xs, 0.05), (xs, 0.53), arrowstyle="-|>", mutation_scale=14, color="#C62828", lw=1.3))
ax.text(6.0, -0.28, "Feedback – lặp lại khi đánh giá phát hiện vấn đề (ví dụ: C08 sửa dữ liệu thành v1.2 sau lần chạy đầu; PD-03)",
        ha="center", va="center", fontsize=10, color="#C62828")
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
