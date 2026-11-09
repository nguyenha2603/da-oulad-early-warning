"""Bộ 7 biểu đồ kể chuyện (data storytelling) khổ 16:9 cho slide và báo cáo.
Nguyên tắc: tiêu đề = thông điệp; một màu nhấn, còn lại xám; nhãn trực tiếp, không legend; ít chữ.
Chạy: python3 P5/src/story_charts.py  ->  P5/story_charts/s1_*.png … s7_*.png + storyboard.png
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIN = Path(__file__).resolve().parents[2]
P3, P4, OUT = FIN / "P3", FIN / "P4", FIN / "P5/story_charts"
OUT.mkdir(exist_ok=True)
J = lambda p: json.loads(Path(p).read_text())
A2, A3, A4, FE = (J(P4 / f"results/{f}.json") for f in ["aq2_results", "aq3_results", "aq4_results", "fairness_error_results"])
U = {r["indicator"]: r for r in A2["unadjusted"]}

INK, GREY, LIGHT, ACC, BLUE = "#1f1f1f", "#8a8984", "#d9d8d3", "#e4572e", "#2a78d6"   # cam = cần hỗ trợ; xanh = hoàn thành
BG = "white"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.edgecolor": LIGHT, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.labelcolor": GREY, "figure.facecolor": BG, "savefig.facecolor": BG})
vn = lambda x, k=0: f"{x:,.{k}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def frame(title, sub, src="Nguồn: OULAD (BBB + FFF, 2013–2014), nhóm N05 tính toán"):
    fig = plt.figure(figsize=(10, 5.625))
    fig.text(0.05, 0.94, title, fontsize=19, weight="bold", color=INK, va="top", linespacing=1.25)
    fig.text(0.05, 0.94 - (0.165 if "\n" in title else 0.09), sub, fontsize=11.5, color=GREY, va="top")
    fig.text(0.05, 0.03, src, fontsize=8, color=GREY)
    return fig


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# 1 — Bối cảnh
ctx = pd.read_csv(P3 / "data/processed/context_all_registrations.csv")
n_all = len(ctx); early = int(ctx.unregistered_by_d28.sum() + ctx.registered_after_d28.sum())
act = ctx[ctx.active_d28 == 1]; nc = int(act.target_noncompletion.sum()); ok = len(act) - nc
fig = frame("Hơn 1/6 sinh viên rời đi trước tuần 4 –\n4/10 người còn lại vẫn không hoàn thành", f"{vn(n_all)} lượt đăng ký, chia theo tình trạng ở cuối ngày 28 và kết quả cuối khóa")
ax = fig.add_axes([0.05, 0.16, 0.9, 0.3]); ax.axis("off")
parts = [(early, LIGHT, "Đã rời đi trước tuần 4", GREY), (ok, BLUE, "Còn học → hoàn thành", "white"), (nc, ACC, "Còn học → không hoàn thành", "white")]
x = 0
for v, c, lab, tc in parts:
    ax.barh(0, v, left=x, color=c, height=0.8, edgecolor="white", linewidth=3)
    ax.text(x + v / 2, 0, f"{v / n_all * 100:.0f}%", ha="center", va="center", fontsize=22, weight="bold", color=tc if tc != GREY else INK)
    ax.text(x + v / 2, -0.75, f"{lab}\n{vn(v)} SV", ha="center", va="top", fontsize=10.5, color=INK)
    x += v
ax.set_xlim(0, n_all); ax.set_ylim(-1.6, 0.5)
fig.text(0.05, 0.56, "Hệ thống cảnh báo ở tuần 4 chỉ có thể giúp nhóm còn học – đó là trọng tâm của dự án.", fontsize=11, color=INK)
save(fig, "s1_boi_canh.png")

# 2 — Vấn đề: năng lực
mod = A4["test"]["a_k200"]["per_module"]["FFF-2014J"]
fig = frame("Khoảng 700 sinh viên cần hỗ trợ,\nnhưng cố vấn chỉ gọi được 200", "Một module-kỳ điển hình (FFF-2014J), số sinh viên còn học ở cuối ngày 28")
ax = fig.add_axes([0.3, 0.12, 0.64, 0.56])
vals = [mod["cohort"], mod["noncompl_in_cohort"], 200]; labs = ["Sinh viên còn học", "Sau này không hoàn thành", "Năng lực liên hệ"]
cols = [LIGHT, ACC, INK]
b = ax.barh(labs[::-1], vals[::-1], color=cols[::-1], height=0.6)
for rect, v in zip(b, vals[::-1]):
    ax.text(v + 25, rect.get_y() + rect.get_height() / 2, vn(v), va="center", fontsize=16, weight="bold", color=INK)
ax.set_xticks([]); ax.tick_params(axis="y", labelsize=12, colors=INK, length=0); ax.set_xlim(0, mod["cohort"] * 1.12)
save(fig, "s2_nang_luc.png")

# 3 — Phát hiện 1: khác biệt từ tuần 1
sw = pd.read_csv(P3 / "data/processed/aq1_student_week.csv")
sw["grp"] = np.where(sw.label3 == "Pass", "Hoàn thành", "Không hoàn thành")
m = sw.groupby(["grp", "week"]).clicks.median().unstack()
fig = frame("Khác biệt xuất hiện ngay từ tuần 1 –\nkhông phải “tụt dần”", "Trung vị số click trên VLE mỗi tuần, 13.020 sinh viên còn học ở ngày 28")
ax = fig.add_axes([0.08, 0.12, 0.78, 0.56])
for g, c in [("Hoàn thành", BLUE), ("Không hoàn thành", ACC)]:
    y = m.loc[g].values
    ax.plot([1, 2, 3, 4], y, color=c, lw=3.5, marker="o", ms=9)
    ax.text(4.12, y[-1], g, color=c, fontsize=12, weight="bold", va="center")
    for i, v in enumerate(y):
        ax.text(i + 1, v + (4 if c == BLUE else -8), f"{v:.0f}", ha="center", fontsize=11, color=c)
ratio = m.loc["Hoàn thành", 1] / m.loc["Không hoàn thành", 1]
ax.annotate("", xy=(1, m.loc["Hoàn thành", 1] - 2), xytext=(1, m.loc["Không hoàn thành", 1] + 2), arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.2))
ax.text(1.08, (m.loc["Hoàn thành", 1] + m.loc["Không hoàn thành", 1]) / 2, f"gấp {vn(ratio, 1)} lần\nngay tuần 1", fontsize=11, color=INK, va="center")
ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["Tuần 1", "Tuần 2", "Tuần 3", "Tuần 4"], fontsize=11)
ax.set_yticks([]); ax.set_xlim(0.7, 4.9); ax.set_ylim(0, m.values.max() * 1.15)
save(fig, "s3_tuan_1.png")

# 4 — Phát hiện 2: TMA1
no_nc = 1 - U["tma1_submitted_by_d28"]["median_noncompl"]; no_ok = 1 - U["tma1_submitted_by_d28"]["median_compl"]
fig = frame("Chưa nộp bài đầu tiên là tín hiệu rõ nhất", "Tỷ lệ sinh viên chưa nộp bài đánh giá đầu tiên (TMA1) tính đến ngày 28")
ax = fig.add_axes([0.1, 0.12, 0.8, 0.62])
b = ax.bar(["Sau này hoàn thành", "Sau này không hoàn thành"], [no_ok * 100, no_nc * 100], color=[LIGHT, ACC], width=0.5)
for rect, v, t in zip(b, [no_ok, no_nc], [f"≈ 1/{round(1 / no_ok)}", f"≈ 1/{round(1 / no_nc)}"]):
    ax.text(rect.get_x() + rect.get_width() / 2, v * 100 + 0.8, f"{v * 100:.0f}%", ha="center", fontsize=22, weight="bold", color=INK)
    ax.text(rect.get_x() + rect.get_width() / 2, v * 100 + 4.2, t + " sinh viên", ha="center", fontsize=11, color=GREY)
ax.set_yticks([]); ax.tick_params(axis="x", labelsize=12, colors=INK, length=0); ax.set_ylim(0, no_nc * 100 * 1.45)
save(fig, "s4_tma1.png")

# 5 — Giải pháp: danh sách vs ngẫu nhiên
pm = A4["test"]["a_k200"]["per_module"]
fig = frame("Gọi theo danh sách: 3/4 đến 7/8 cuộc gọi đến đúng người –\ngấp đôi gọi ngẫu nhiên", "Tỷ lệ cuộc gọi đến đúng sinh viên sau này không hoàn thành, 200 ca mỗi module, kỳ 2014J (mô hình chưa từng thấy)")
ax = fig.add_axes([0.08, 0.14, 0.86, 0.5])
mods = ["BBB-2014J", "FFF-2014J"]; xx = np.arange(2); w = 0.34
rnd = [pm[mm]["noncompl_in_cohort"] / pm[mm]["cohort"] for mm in mods]; lst = [pm[mm]["precision"] for mm in mods]
ax.bar(xx - w / 2, np.array(rnd) * 100, w, color=LIGHT); ax.bar(xx + w / 2, np.array(lst) * 100, w, color=ACC)
for i in range(2):
    ax.text(xx[i] - w / 2, rnd[i] * 100 + 2, f"{rnd[i] * 100:.0f}%", ha="center", fontsize=15, color=GREY, weight="bold")
    ax.text(xx[i] + w / 2, lst[i] * 100 + 2, f"{lst[i] * 100:.0f}%", ha="center", fontsize=18, color=INK, weight="bold")
    ax.text(xx[i] - w / 2, -7, "Gọi ngẫu nhiên", ha="center", fontsize=10, color=GREY)
    ax.text(xx[i] + w / 2, -7, "Theo danh sách", ha="center", fontsize=10, color=INK)
ax.set_xticks(xx); ax.set_xticklabels(["Module BBB", "Module FFF"], fontsize=12, color=INK); ax.tick_params(axis="x", pad=22, length=0)
ax.set_yticks([]); ax.set_ylim(0, 100)
save(fig, "s5_danh_sach.png")

# 6 — Đánh đổi theo k
fig = frame("Gọi thêm người, mỗi cuộc gọi kém hiệu quả hơn", "Tỷ lệ cuộc gọi đến đúng người khi tăng số ca liên hệ mỗi module (kỳ 2014J)")
ax = fig.add_axes([0.08, 0.12, 0.78, 0.62])
ks = [100, 200, 300]
for mm, c, lab in [("BBB-2014J", GREY, "BBB"), ("FFF-2014J", ACC, "FFF")]:
    y = [A4["test"][f"a_k{k}"]["per_module"][mm]["precision"] * 100 for k in ks]
    ax.plot(ks, y, color=c, lw=3.5, marker="o", ms=9)
    ax.text(312, y[-1], lab, color=c, fontsize=13, weight="bold", va="center")
    for k, v in zip(ks, y):
        up = (mm == "BBB-2014J") == (k == 100)
        ax.text(k, v + (2.2 if up else -4.5), f"{v:.0f}%", ha="center", fontsize=11, color=c)
ax.axvline(200, color=INK, lw=1, ls=":"); ax.text(203, 52, "Năng lực hiện có\n(200 ca)", fontsize=10, color=INK)
ax.set_xticks(ks); ax.set_xticklabels([f"{k} ca" for k in ks], fontsize=11); ax.set_yticks([]); ax.set_ylim(50, 100); ax.set_xlim(80, 335)
save(fig, "s6_danh_doi.png")

# 7 — Giới hạn
e = A3["extras"]["recall_test_ci95"]; err = FE["error"]
caught, missed = e["tp"], e["pos"] - e["tp"]
fig = frame("Mô hình vẫn bỏ sót khoảng 3/10 sinh viên –\nchủ yếu là nhóm gặp khó muộn", "Sinh viên sau này không hoàn thành ở kỳ 2014J, theo việc mô hình có nhận ra ở tuần 4 hay không")
ax = fig.add_axes([0.05, 0.4, 0.9, 0.18]); ax.axis("off")
tot = caught + missed
ax.barh(0, caught, color=ACC, height=0.8, edgecolor="white", linewidth=3); ax.barh(0, missed, left=caught, color=LIGHT, height=0.8, edgecolor="white", linewidth=3)
ax.text(caught / 2, 0, f"Nhận ra: {caught / tot * 100:.0f}%", ha="center", va="center", fontsize=17, weight="bold", color="white")
ax.text(caught + missed / 2, 0, f"Bỏ sót: {missed / tot * 100:.0f}%", ha="center", va="center", fontsize=17, weight="bold", color=INK)
ax.set_xlim(0, tot)
ax2 = fig.add_axes([0.05, 0.12, 0.9, 0.24]); ax2.axis("off")
fn, tp = err["fn_profile_median"], err["tp_profile_median"]
ax2.text(0, 0.75, "Ở tuần 4, nhóm bị bỏ sót trông như sinh viên bình thường:", fontsize=12, color=INK, transform=ax2.transAxes)
ax2.text(0, 0.35, f"• {fn['active_days_0_28']:.0f} ngày đăng nhập (nhóm được nhận ra: {tp['active_days_0_28']:.0f} ngày)     • đã nộp bài đầu tiên", fontsize=12, color=GREY, transform=ax2.transAxes)
ax2.text(0, 0.0, "→ Cần thêm một mốc cảnh báo sau bài đánh giá thứ hai.", fontsize=12, color=ACC, weight="bold", transform=ax2.transAxes)
save(fig, "s7_gioi_han.png")

# storyboard ghép (cho báo cáo)
from PIL import Image
files = ["s1_boi_canh.png", "s2_nang_luc.png", "s3_tuan_1.png", "s4_tma1.png", "s5_danh_sach.png", "s6_danh_doi.png"]
ims = [Image.open(OUT / f) for f in files]
w, h = ims[0].size; pad = 30
board = Image.new("RGB", (2 * w + 3 * pad, 3 * h + 4 * pad), "#e9e8e4")
for i, im in enumerate(ims):
    board.paste(im, (pad + (i % 2) * (w + pad), pad + (i // 2) * (h + pad)))
board.thumbnail((2400, 4000)); board.save(OUT / "storyboard_6.png")
print("story charts saved:", len(list(OUT.glob("s*.png"))))
