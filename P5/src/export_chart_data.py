"""Xuất dữ liệu cho 11 chart + dashboard (CSV để vẽ lại bằng Power BI/Superset) và vẽ dashboard wireframe.
Input: P3 interim/processed + P4 results (đã sinh bằng pipeline). Chạy: python3 P5/src/export_chart_data.py
Output: P5/chart_data/*.csv, P5/chart_data/bundle.json, P5/figures/p5_dashboard_wireframe.png
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIN = Path(__file__).resolve().parents[2]
P3, P4, P5 = FIN / "P3", FIN / "P4", FIN / "P5"
OUT, FIG = P5 / "chart_data", P5 / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
J = lambda p: json.loads(Path(p).read_text())
S3 = J(P3 / "data/metadata/profiling_summary.json")
A1, A2, A4, A5, FE = (J(P4 / f"results/{f}.json") for f in ["aq1_results", "aq2_results", "aq4_results", "aq5_results", "fairness_error_results"])
B = {}


def put(name, df):
    df.to_csv(OUT / f"{name}.csv", index=False)
    B[name] = df.to_dict("records")


# c01 missingness
put("c01_missingness", pd.DataFrame([{"variable": k, "n_missing": v["n"], "missing_rate": v["rate"]} for k, v in S3["missing"].items()]))
# c02 target balance (active d28)
ctx = pd.read_csv(P3 / "data/processed/context_all_registrations.csv")
act = ctx[ctx.active_d28 == 1]
t = act.groupby(["code_module", "code_presentation", "label3"]).size().rename("n").reset_index()
t["share"] = (t.n / t.groupby(["code_module", "code_presentation"]).n.transform("sum")).round(4)
put("c02_target_balance", t)
# c03 daily active students
daily = pd.read_parquet(P3 / "data/interim/vle_student_day.parquet")
put("c03_daily_active", daily.groupby(["code_module", "date"]).id_student.nunique().rename("active_students").reset_index())
# c04 weekly clicks by result
put("c04_weekly_clicks", pd.DataFrame(A1["table"])[["label3", "week", "n", "clicks_median", "clicks_ci_low", "clicks_ci_high", "clicks_mean", "active_days_mean"]])
# c05 clicks distribution (per student)
m = pd.read_csv(P3 / "data/processed/aq3_model_table.csv")
d2 = pd.read_csv(P3 / "data/processed/aq2_diagnostic_table.csv")[["code_module", "code_presentation", "id_student", "label3"]]
c5 = m[["code_module", "code_presentation", "id_student", "clicks_0_28"]].merge(d2, on=["code_module", "code_presentation", "id_student"]).drop(columns="id_student")
put("c05_clicks_per_student", c5)
# c06 unregistration timing (raw rows)
put("c06_unregistration", ctx[ctx.date_unregistration.notna()][["code_module", "code_presentation", "final_result", "date_unregistration"]])
# c07 AQ2 effect size by week
rows = []
for r in A2["unadjusted"]:
    for mt in ["clicks", "active_days"]:
        for w in range(1, 5):
            if r["indicator"] == f"{mt}_w{w}":
                rows.append({"measure": mt, "week": w, "cliffs_delta": r["effect"], "p_holm": r["p_holm"], "significant": r["significant"]})
put("c07_aq2_effect_by_week", pd.DataFrame(rows))
put("c07b_aq2_indicators", pd.DataFrame(A2["unadjusted"])[["indicator", "type", "median_noncompl", "median_compl", "effect", "p_holm", "significant"]])
# c08 PR curve + calibration (test)
pred = pd.read_csv(P4 / "results/test_predictions.csv")
pr, cal = [], []
for mname, col in [("M1 quy tắc", "p_M1"), ("M2 logistic", "p_M2"), ("M3 gradient boosting", "p_M3")]:
    p, r, _ = precision_recall_curve(pred.target_noncompletion, pred[col])
    idx = np.unique(np.linspace(0, len(p) - 1, 150).astype(int))
    pr += [{"model": mname, "recall": round(float(r[i]), 4), "precision": round(float(p[i]), 4)} for i in idx]
    if col != "p_M1":
        fp, mp = calibration_curve(pred.target_noncompletion, pred[col], n_bins=10, strategy="quantile")
        cal += [{"model": mname, "predicted": round(float(a), 4), "observed": round(float(b), 4)} for a, b in zip(mp, fp)]
put("c08_pr_curve", pd.DataFrame(pr)); put("c08b_calibration", pd.DataFrame(cal))
# c09 precision@k
rows = []
for key, v in A4["test"].items():
    rule, k = key.split("_k")
    if "per_module" in v and rule != "d":
        for mp, x in v["per_module"].items():
            rows.append({"rule": rule, "k": int(k), "module_presentation": mp, "precision": x["precision"], "true_positive": x["tp"]})
    elif rule == "d":
        rows.append({"rule": rule, "k": int(k), "module_presentation": "gộp BBB+FFF", "precision": v["precision"], "true_positive": v["tp"]})
put("c09_precision_at_k", pd.DataFrame(rows))
# c10 importance
put("c10_permutation_importance", pd.DataFrame([{"feature": k, "pr_auc_drop": v} for k, v in A5["permutation_importance_top10"].items()]))
# c11 fairness
rows = []
for attr, v in FE["fairness"].items():
    for g, x in v["groups"].items():
        rows.append({"attribute": attr, "group": g, **x})
put("c11_fairness", pd.DataFrame(rows))
# ------------------------------------------------------------------ dashboard data (test 2014J, vận hành – không hiển thị kết quả thật)
TYPE_VN = {"homepage": "trang chủ", "oucontent": "nội dung bài học", "resource": "tài liệu", "subpage": "trang con",
           "forumng": "diễn đàn", "quiz": "quiz", "url": "liên kết"}
trm = pd.read_csv(P3 / "data/processed/aq3_model_table.csv").query("split == 'train'")
mtest = pd.read_csv(P3 / "data/processed/aq3_model_table.csv").query("split == 'test'").set_index(["code_module", "id_student"])
med_tr = trm.median(numeric_only=True)


def tag(f, row):
    """Nhãn lý do ngắn cho dashboard (≤ 4 từ)."""
    if not isinstance(f, str) or not f:
        return ""
    v = row.get(f, np.nan)
    low = pd.notna(v) and pd.notna(med_tr.get(f, np.nan)) and v < med_tr[f]
    if f == "tma1_submitted_by_d28": return "Chưa nộp TMA1"
    if f == "tma1_on_time": return "TMA1 chưa đúng hạn"
    if f == "tma1_days_before_deadline": return "Nộp TMA1 sát hạn"
    if f == "no_vle_activity_0_28": return "Chưa hoạt động VLE"
    if f == "days_since_last_active_d28": return "Lâu chưa đăng nhập"
    if f == "clicks_change_w1_w4": return "Tương tác giảm" if low else "Tương tác tăng"
    if f.startswith("active_days"): return "Ít ngày đăng nhập" if low else "Nhiều ngày đăng nhập"
    if f in ("clicks_pre_start", "active_days_pre_start"): return "Ít chuẩn bị trước kỳ" if low else "Chuẩn bị trước kỳ"
    if f == "n_sites_0_28": return "Ít tài nguyên đã mở" if low else "Nhiều tài nguyên đã mở"
    for t, vn in TYPE_VN.items():
        if f == f"clicks_{t}_0_28": return f"Ít dùng {vn}" if low else f"Dùng nhiều {vn}"
    if f.startswith("clicks_") or f == "log_clicks_0_28": return "Ít tương tác" if low else "Tương tác nhiều"
    return f


pred = pd.read_csv(P4 / "results/test_predictions.csv")
pred = pred.sort_values(["code_module", "rank_in_module"]).reset_index(drop=True)
for i in (1, 2, 3):
    pred[f"tag{i}"] = [tag(r[f"reason_f{i}"], mtest.loc[(r.code_module, r.id_student)]) for _, r in pred.iterrows()]
top = pred.rank_in_module <= 300
pred["ma_sv"] = np.where(top, pred.rank_in_module.map(lambda r: f"SV-{r:03d}"), "")   # pseudonymisation theo thứ hạng trong module
dash = pred[top].copy()
dash["muc_uu_tien"] = pd.cut(dash.rank_in_module, [0, 50, 120, 200, 300], labels=["Rất cao", "Cao", "Trung bình", "Mở rộng (k>200)"])
put("dashboard_top300", dash[["code_module", "code_presentation", "rank_in_module", "ma_sv", "muc_uu_tien", "p_M2",
                              "clicks_w1", "clicks_w2", "clicks_w3", "clicks_w4", "active_days_0_28", "tma1_submitted_by_d28",
                              "tag1", "tag2", "tag3", "reason_1", "reason_2", "reason_3", "target_noncompletion"]].round({"p_M2": 3}))
# phân bố điểm rủi ro cả lớp (không định danh)
put("dashboard_cohort_scores", pred[["code_module", "rank_in_module", "p_M2", "tma1_submitted_by_d28"]].round({"p_M2": 4}))
# tương tác theo tuần: nhóm trong danh sách (top-200) vs phần còn lại
rows = []
for m_, g in pred.groupby("code_module"):
    for grp, gg in [("Trong danh sách (top-200)", g[g.rank_in_module <= 200]), ("Phần còn lại của lớp", g[g.rank_in_module > 200])]:
        for w in range(1, 5):
            rows.append({"code_module": m_, "group": grp, "week": w, "median_clicks": float(gg[f"clicks_w{w}"].median())})
put("dashboard_weekly_list_vs_rest", pd.DataFrame(rows))
med = pd.read_csv(P3 / "data/processed/aq1_student_week.csv").query("split == 'test'").groupby(["code_module", "week"]).clicks.median().reset_index()
put("dashboard_class_median_weekly", med)
B["_meta"] = {"source": "N05 – OULAD BBB+FFF; P3 prepared data; P4 results (test 2014J)", "rows": {k: len(v) for k, v in B.items() if isinstance(v, list)}}
(OUT / "bundle.json").write_text(json.dumps(B, ensure_ascii=False, default=float))

# ------------------------------------------------------------------ dashboard wireframe PNG (KPI + 5 biểu đồ + bảng gọn)
INK, INK2, LINE, ACC, C2, C3, C4 = "#0b0b0b", "#52514e", "#d6d5cf", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MOD, K = "FFF", 200
g = pred[pred.code_module == MOD]; lst = g[g.rank_in_module <= K]
fig = plt.figure(figsize=(13, 8.2), facecolor="#fbfbf9")
gs = fig.add_gridspec(3, 3, height_ratios=[0.34, 1, 1.15], hspace=0.42, wspace=0.3, left=0.04, right=0.98, top=0.88, bottom=0.05)
fig.text(0.04, 0.95, "Danh sách ưu tiên liên hệ – cuối tuần 4", fontsize=16, weight="bold", color=INK)
fig.text(0.04, 0.915, "Module-kỳ: FFF-2014J ▾     Mốc dữ liệu: ngày 28     Hạn mức: 200 ca ▾     (gợi ý – cố vấn duyệt từng ca)", fontsize=9.5, color=INK2)
# KPI row
kp = fig.add_subplot(gs[0, :]); kp.axis("off")
kpis = [("Sĩ số còn học", f"{len(g):,}".replace(",", ".")), ("Đề xuất liên hệ", f"{K}"), ("Đã duyệt", f"0 / {K}"),
        ("Chưa nộp TMA1: danh sách / cả lớp", f"{(1 - lst.tma1_submitted_by_d28.mean()) * 100:.0f}% / {(1 - g.tma1_submitted_by_d28.mean()) * 100:.0f}%")]
for i, (lab, val) in enumerate(kpis):
    x0 = i * 0.25 + 0.005
    kp.add_patch(plt.Rectangle((x0, 0.05), 0.235, 0.9, transform=kp.transAxes, fc="white", ec=LINE, lw=1))
    kp.text(x0 + 0.015, 0.72, lab, transform=kp.transAxes, fontsize=9, color=INK2)
    kp.text(x0 + 0.015, 0.22, val, transform=kp.transAxes, fontsize=20, weight="bold", color=INK)


def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left", weight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK2); ax.grid(color="#ecebe7", lw=0.6); ax.set_axisbelow(True)


# (1) histogram điểm rủi ro + vạch cắt
ax = fig.add_subplot(gs[1, 0]); style(ax, "① Phân bố điểm rủi ro của lớp")
thr = lst.p_M2.min()
ax.hist(g.p_M2, bins=30, color="#c9dbf3", edgecolor="white")
ax.hist(lst.p_M2, bins=30, color=ACC, edgecolor="white", range=(g.p_M2.min(), g.p_M2.max()))
ax.axvline(thr, color=INK, ls="--", lw=1); ax.text(thr - 0.01, ax.get_ylim()[1] * 0.92, "cắt top-200 ", fontsize=8, ha="right")
ax.set_xlabel("Điểm rủi ro (mô hình)", fontsize=8); ax.set_ylabel("Số SV", fontsize=8)
# (2) lý do phổ biến
ax = fig.add_subplot(gs[1, 1]); style(ax, "② Lý do phổ biến trong danh sách")
tags = pd.concat([lst.tag1, lst.tag2, lst.tag3]); tags = tags[tags != ""].value_counts().head(6)[::-1]
ax.barh(tags.index, tags.values, color=ACC, height=0.6)
for i, v in enumerate(tags.values): ax.text(v + 2, i, str(v), va="center", fontsize=8, color=INK2)
ax.set_xlabel("Số SV", fontsize=8); ax.grid(axis="y", visible=False)
# (3) tương tác theo tuần: danh sách vs phần còn lại
ax = fig.add_subplot(gs[1, 2]); style(ax, "③ Tương tác theo tuần (trung vị click)")
rest = g[g.rank_in_module > K]
for d_, c, lab in [(lst, C2, "Trong danh sách"), (rest, ACC, "Phần còn lại")]:
    ax.plot([1, 2, 3, 4], [d_[f"clicks_w{w}"].median() for w in range(1, 5)], color=c, lw=2, marker="o", ms=5, label=lab)
ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["T1", "T2", "T3", "T4"]); ax.legend(frameon=False, fontsize=8); ax.set_ylim(0)
# bảng danh sách gọn
ax = fig.add_subplot(gs[2, 0]); ax.axis("off"); ax.set_title("Danh sách (8/200 dòng đầu)", fontsize=10, color=INK, loc="left", weight="bold")
cols = [("Mã SV", 0.0), ("Ưu tiên", 0.24), ("Lý do chính", 0.46), ("Duyệt", 0.93)]
for c, x in cols: ax.text(x, 0.95, c, fontsize=8.5, weight="bold", color=INK2, transform=ax.transAxes)
for i, (_, r) in enumerate(lst.head(8).iterrows()):
    y = 0.84 - i * 0.11
    if i == 0: ax.add_patch(plt.Rectangle((-0.02, y - 0.04), 1.04, 0.1, transform=ax.transAxes, fc="#e8f1fc", ec="none"))
    pr = "Rất cao" if r.rank_in_module <= 50 else "Cao"
    for txt, x in [(r.ma_sv, 0.0), (pr, 0.24), (r.tag1, 0.46), ("☐", 0.93)]:
        ax.text(x, y, txt, fontsize=8.5, color=INK, transform=ax.transAxes)
# (4) chi tiết SV chọn
r0 = lst.iloc[0]
ax = fig.add_subplot(gs[2, 1]); style(ax, f"④ {r0.ma_sv}: tương tác so với trung vị lớp")
mf = med[med.code_module == MOD]
ax.plot([1, 2, 3, 4], [r0[f"clicks_w{w}"] for w in range(1, 5)], color=C2, lw=2, marker="o", ms=5, label=r0.ma_sv)
ax.plot(mf.week, mf.clicks, color=INK2, ls="--", lw=1.4, marker="o", ms=3, label="Trung vị lớp")
ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["T1", "T2", "T3", "T4"]); ax.legend(frameon=False, fontsize=8); ax.set_ylim(0)
chips = " · ".join(t for t in [r0.tag1, r0.tag2, r0.tag3] if t)
ax.text(0, -0.32, "Lý do: " + chips, transform=ax.transAxes, fontsize=8.5, color=INK)
ax.text(0, -0.45, "[ Duyệt liên hệ ]   [ Bỏ qua ]   [ Ghi chú ]", transform=ax.transAxes, fontsize=8.5, color=ACC, weight="bold")
# (5) giám sát công bằng
fr = pd.DataFrame(B["c11_fairness"])
fr = fr[fr.attribute.isin(["gender", "disability", "imd_group"])]
fr["lab"] = fr.attribute.map({"gender": "Giới", "disability": "Khuyết tật", "imd_group": "IMD"}) + " " + fr.group.astype(str)
ax = fig.add_subplot(gs[2, 2]); style(ax, "⑤ Giám sát công bằng: tỷ trọng nhóm")
yy = np.arange(len(fr)); h = 0.38
ax.barh(yy - h / 2, fr.share_population * 100, height=h, color="#c9c8c2", label="Trong lớp")
ax.barh(yy + h / 2, fr.share_top200 * 100, height=h, color=C3, label="Trong danh sách")
ax.set_yticks(yy); ax.set_yticklabels(fr.lab, fontsize=8); ax.invert_yaxis(); ax.set_xlabel("%", fontsize=8)
ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.grid(axis="y", visible=False)
fig.savefig(FIG / "p5_dashboard_wireframe.png", dpi=200, facecolor=fig.get_facecolor())
plt.close(fig)
print("wireframe saved")

# ------------------------------------------------------------------ fairness figure (P5 Hình 10.6)
fr = pd.DataFrame(B["c11_fairness"])
fr["lab"] = fr.attribute.map({"gender": "Giới", "age_band": "Tuổi", "disability": "Khuyết tật", "imd_group": "IMD"}) + ": " + fr.group.astype(str)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
yy = np.arange(len(fr))
for ax, col, title, c in [(axes[0], "recall", "Recall (ca không hoàn thành được phát hiện)", "#2a78d6"), (axes[1], "fpr", "FPR (ca hoàn thành bị gắn cờ nhầm)", "#eb6834")]:
    ax.barh(yy, fr[col], color=c, height=0.6)
    for i, v in enumerate(fr[col]):
        ax.text(v + 0.01, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=7, color="#52514e")
    ax.set_title(title, fontsize=9); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[0].set_yticks(yy); axes[0].set_yticklabels(fr.lab, fontsize=8); axes[0].invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "p5_fairness.png", dpi=200, bbox_inches="tight")
print("fairness figure saved")
