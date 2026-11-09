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
# dashboard list (top-200 each module, test 2014J)
dash = pred[pred.rank_in_module <= 300].sort_values(["code_module", "rank_in_module"])
dash["ma_sv"] = [f"SV-{i:03d}" for i in range(1, len(dash) + 1)]   # pseudonymisation – không hiển thị id gốc
dash["muc_uu_tien"] = pd.cut(dash.rank_in_module, [0, 50, 120, 200, 300], labels=["Rất cao", "Cao", "Trung bình", "Mở rộng (k>200)"])
put("dashboard_top300", dash[["code_module", "code_presentation", "rank_in_module", "ma_sv", "muc_uu_tien", "p_M2",
                              "clicks_w1", "clicks_w2", "clicks_w3", "clicks_w4", "active_days_0_28", "tma1_submitted_by_d28",
                              "reason_1", "reason_2", "reason_3", "target_noncompletion"]].round({"p_M2": 3}))
med = pd.read_csv(P3 / "data/processed/aq1_student_week.csv").groupby(["code_module", "week"]).clicks.median().reset_index()
put("dashboard_class_median_weekly", med)
B["_meta"] = {"source": "N05 – OULAD BBB+FFF; P3 N05-prepared-v1.0; P4 results (test 2014J)", "rows": {k: len(v) for k, v in B.items() if isinstance(v, list)}}
(OUT / "bundle.json").write_text(json.dumps(B, ensure_ascii=False, default=float))

# ------------------------------------------------------------------ wireframe PNG
INK, INK2, LINE, FILL, ACC = "#0b0b0b", "#52514e", "#b9b8b2", "#f3f2ee", "#2a78d6"
fig = plt.figure(figsize=(11, 6.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 110); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, title=None, fill="white", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fill, ec=LINE, lw=lw))
    if title:
        ax.text(x + 1, y + h - 1.6, title, fontsize=8.5, color=INK2, weight="bold", va="top")


box(1, 1, 108, 64, fill="#fbfbf9")
ax.text(3, 62.2, "Danh sách ưu tiên liên hệ – cuối tuần 4", fontsize=13, weight="bold", color=INK, va="top")
for i, (lab, val) in enumerate([("Module-kỳ", "FFF-2014J ▾"), ("Mốc dữ liệu", "Ngày 28"), ("Hạn mức", "200 ca ▾")]):
    box(55 + i * 18, 58.5, 16.5, 5); ax.text(56 + i * 18, 62.3, lab, fontsize=6.5, color=INK2, va="top"); ax.text(56 + i * 18, 60.2, val, fontsize=8.5, color=INK, va="center")
fff = pred[pred.code_module == "FFF"]
kpis = [("Sĩ số còn học", f"{len(fff):,}".replace(",", ".")), ("Đề xuất liên hệ", "200"), ("Đã duyệt", "0 / 200"), ("Đã bỏ qua", "0")]
for i, (lab, val) in enumerate(kpis):
    box(3 + i * 13, 49.5, 12, 7); ax.text(4 + i * 13, 55.2, lab, fontsize=6.5, color=INK2, va="top"); ax.text(4 + i * 13, 51.6, val, fontsize=12, weight="bold", color=INK)
# list table
box(3, 3, 52, 45, "Danh sách (xếp theo mức ưu tiên)")
cols = [("#", 4.5), ("Mã SV (giả danh)", 7.5), ("Ưu tiên", 20.5), ("Lý do chính", 28.5), ("Duyệt", 49.5)]
for c, x in cols:
    ax.text(x, 43.2, c, fontsize=6.8, weight="bold", color=INK2)
ax.plot([4, 54], [42.3, 42.3], color=LINE, lw=0.8)
top = dash[dash.code_module == "FFF"].head(9)
short = lambda s: (s[:34] + "…") if len(s) > 35 else s
for i, (_, r) in enumerate(top.iterrows()):
    y = 40.2 - i * 4.1
    if i == 0:
        ax.add_patch(FancyBboxPatch((3.6, y - 1.5), 50.8, 3.6, boxstyle="round,pad=0.1", fc="#e8f1fc", ec="none"))
    ax.text(4.5, y, str(int(r.rank_in_module)), fontsize=7, color=INK)
    ax.text(7.5, y, r.ma_sv, fontsize=7, color=INK)
    ax.text(20.5, y, str(r.muc_uu_tien), fontsize=7, color=INK)
    ax.text(28.5, y, short(r.reason_1), fontsize=6.3, color=INK)
    ax.text(49.5, y, "☐", fontsize=9, color=INK2, va="center")
ax.text(4.5, 4.2, "… 200 dòng · cố vấn duyệt từng ca, không gửi tự động", fontsize=6.5, color=INK2, style="italic")
# detail panel
r0 = top.iloc[0]
box(57, 3, 50, 45, f"Chi tiết sinh viên {r0.ma_sv} – mức ưu tiên {r0.muc_uu_tien}")
ax.text(59, 43, "Tương tác theo tuần (so với trung vị lớp)", fontsize=7.5, color=INK2)
sub = fig.add_axes([0.555, 0.40, 0.4, 0.22])
mf = med[med.code_module == "FFF"]
sub.plot(range(1, 5), [r0[f"clicks_w{i}"] for i in range(1, 5)], color=ACC, lw=2, marker="o", ms=4, label="Sinh viên")
sub.plot(mf.week, mf.clicks, color=INK2, lw=1.2, ls="--", marker="o", ms=3, label="Trung vị lớp")
sub.set_xticks([1, 2, 3, 4]); sub.set_xticklabels(["T1", "T2", "T3", "T4"], fontsize=7); sub.tick_params(labelsize=7)
for s in ("top", "right"): sub.spines[s].set_visible(False)
sub.legend(frameon=False, fontsize=6.5); sub.set_ylabel("click", fontsize=7)
ax.text(59, 21.5, "Lý do (do mô hình giải thích, ngôn ngữ trung tính):", fontsize=7.5, color=INK2)
for i, rs in enumerate([r0.reason_1, r0.reason_2, r0.reason_3]):
    if isinstance(rs, str) and rs:
        ax.text(59.5, 18.8 - i * 3.0, "• " + (rs[:70] + "…" if len(rs) > 71 else rs), fontsize=6.6, color=INK)
for i, (lab, fc) in enumerate([("Duyệt liên hệ", ACC), ("Bỏ qua", "white"), ("Ghi chú", "white")]):
    ax.add_patch(FancyBboxPatch((59 + i * 15, 4.5), 13, 3.6, boxstyle="round,pad=0.2", fc=fc, ec=ACC if fc == "white" else fc))
    ax.text(65.5 + i * 15, 6.3, lab, fontsize=7.5, color="white" if fc == ACC else ACC, ha="center", va="center", weight="bold")
fig.savefig(FIG / "p5_dashboard_wireframe.png", dpi=200, bbox_inches="tight")
print("exported", len(B) - 1, "tables; wireframe saved")


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
