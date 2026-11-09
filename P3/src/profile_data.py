"""Data Profiling & Data Quality Assessment – Milestone 3 (Task 1–2).

Đọc raw (chỉ đọc) + interim do prepare_data.py tạo; ghi data/metadata/profiling_summary.json và hình vào
reports/milestone3/figures/. Chạy sau prepare_data.py:  python3 src/profile_data.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *  # noqa: E402,F403

KEY = ["code_module", "code_presentation", "id_student"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Pass": "#2a78d6", "Fail": "#eb6834", "Withdrawn": "#1baf7a"}   # categorical slots 1–3
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight"})
S = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def main():
    raw = {f: pd.read_csv(RAW / f"{f}.csv", na_values=NA) for f in
           ["studentInfo", "studentRegistration", "assessments", "studentAssessment", "vle", "courses"]}
    sc = lambda df: df[df.code_module.isin(MODULES)] if "code_module" in df else df
    a = sc(raw["assessments"])
    sa = raw["studentAssessment"][raw["studentAssessment"].id_assessment.isin(a.id_assessment)]
    sa = sa.merge(a, on="id_assessment")
    reg = pd.read_parquet(INTERIM / "registrations_scope.parquet")
    feat = pd.read_parquet(INTERIM / "registration_features_all.parquet")
    daily = pd.read_parquet(INTERIM / "vle_student_day.parquet")
    courses = sc(raw["courses"])

    # ---- shape & schema
    S["shape"] = {k: {"rows_all": int(len(v)), "rows_scope": int(len(sc(v))) if "code_module" in v else None,
                      "cols": int(v.shape[1])} for k, v in raw.items()}
    S["course_length"] = courses.set_index(["code_module", "code_presentation"]).module_presentation_length.to_dict()
    S["course_length"] = {f"{m}-{p}": int(x) for (m, p), x in S["course_length"].items()}

    # ---- missingness (scope)
    miss = {}
    for name, df in [("studentInfo", sc(raw["studentInfo"])), ("studentRegistration", sc(raw["studentRegistration"])),
                     ("assessments", a), ("studentAssessment", sa[raw["studentAssessment"].columns]), ("vle", sc(raw["vle"]))]:
        for c in df.columns:
            if df[c].isna().any():
                miss[f"{name}.{c}"] = {"n": int(df[c].isna().sum()), "rate": round(float(df[c].isna().mean()), 4)}
    S["missing"] = miss
    # imd_band missing by group (MAR?)
    reg["imd_miss"] = reg.imd_band_missing
    S["imd_missing_by_region_top"] = (reg.groupby("region").imd_miss.mean().sort_values(ascending=False).head(3).round(3).to_dict())
    S["imd_missing_by_result"] = reg.groupby("final_result").imd_miss.mean().round(4).to_dict()
    S["score_missing_by_type"] = sa.groupby("assessment_type").score.apply(lambda s: int(s.isna().sum())).to_dict()
    # v1.1 – cơ chế thiếu dữ liệu (MCAR?) bằng chi-square: thiếu có phụ thuộc biến khác không
    from scipy import stats
    mech = {}
    for c in ["region", "final_result", "code_module", "gender", "age_band"]:
        ct = pd.crosstab(reg[c], reg.imd_miss); chi = stats.chi2_contingency(ct)
        nobs = ct.values.sum(); k = min(ct.shape) - 1
        mech[c] = {"chi2": round(float(chi[0]), 1), "dof": int(chi[2]), "p": float(chi[1]),
                   "cramers_v": round(float((chi[0] / (nobs * k)) ** 0.5), 3)}
    S["imd_missing_mechanism"] = mech
    S["imd_missing_by_region_all"] = reg.groupby("region").imd_miss.mean().round(4).sort_values(ascending=False).to_dict()
    S["date_registration_missing_outcome"] = reg[reg.date_registration_missing == 1].final_result.value_counts().to_dict()
    sn = sa[sa.score.isna()]
    S["score_missing_detail"] = {"n": int(len(sn)), "has_date_submitted": int(sn.date_submitted.notna().sum()), "banked": int(sn.is_banked.sum()),
                                 "score_zero_n": int((sa.score == 0).sum())}
    S["submitted_before_start"] = {"n": int((sa.date_submitted < 0).sum()), "banked": int(((sa.date_submitted < 0) & (sa.is_banked == 1)).sum()),
                                   "banked_total": int(sa.is_banked.sum())}
    vv = sc(raw["vle"])
    S["vle_week_missing_by_type"] = vv.assign(m=vv.week_from.isna()).groupby("activity_type").m.mean().round(2).to_dict()

    # ---- duplicates & key integrity
    S["orphans"] = {
        "studentAssessment_student_not_registered": int((~sa.set_index(["code_module", "code_presentation", "id_student"]).index
                                                         .isin(reg.set_index(KEY).index)).sum()),
        "registrations_without_any_vle": int(feat.no_vle_record_any.sum()),
        "registrations_without_vle_0_28": int(feat.no_vle_activity_0_28.sum()),
        "active_d28_without_vle_0_28": int(feat.loc[feat.active_d28 == 1, "no_vle_activity_0_28"].sum()),
    }
    S["no_vle_by_result"] = feat[feat.no_vle_record_any == 1].final_result.value_counts().to_dict()

    # ---- validity
    S["validity"] = {
        "score_out_of_0_100": int(((sa.score < 0) | (sa.score > 100)).sum()),
        "submitted_after_course_end": int((sa.merge(courses, on=["code_module", "code_presentation"])
                                           .pipe(lambda d: d.date_submitted > d.module_presentation_length)).sum()),
        "unregistration_before_registration": int((reg.date_unregistration < reg.date_registration).sum()),
        "vle_date_min": int(daily.date.min()), "vle_date_max": int(daily.date.max()),
        "vle_after_course_end": int(daily.merge(courses, on=["code_module", "code_presentation"])
                                    .pipe(lambda d: d.date > d.module_presentation_length).sum()),
        "vle_week_from_missing_rate": round(float(sc(raw["vle"]).week_from.isna().mean()), 3),
        "imd_band_levels": sorted(reg.imd_band.unique().tolist()),
    }
    # ---- distribution & outliers
    pop = feat[feat.active_d28 == 1]
    c = pop.clicks_0_28
    S["clicks_0_28"] = {"mean": round(float(c.mean()), 1), "median": float(c.median()), "p99": float(c.quantile(.99)),
                        "max": int(c.max()), "zero_rate": round(float((c == 0).mean()), 4),
                        "skew": round(float(c.skew()), 2), "iqr_outliers": int((c > c.quantile(.75) + 1.5 * (c.quantile(.75) - c.quantile(.25))).sum())}
    S["daily_clicks"] = {"p99": float(daily.clicks.quantile(.99)), "p999": float(daily.clicks.quantile(.999)),
                         "max": int(daily.clicks.max()), "n_days_over_1000": int((daily.clicks > 1000).sum())}
    S["studied_credits"] = {"median": float(pop.studied_credits.median()), "max": int(pop.studied_credits.max()),
                            "n_over_240": int((pop.studied_credits > 240).sum())}
    # ---- target balance
    S["final_result_all"] = reg.final_result.value_counts().to_dict()
    S["noncompletion_all"] = round(float(reg.target_noncompletion.mean()), 4)
    S["noncompletion_active"] = round(float(pop.target_noncompletion.mean()), 4)
    S["n_active"] = int(len(pop)); S["n_all"] = int(len(reg))
    S["unregistered_by_d28"] = int(reg.unregistered_by_d28.sum()); S["registered_after_d28"] = int(reg.registered_after_d28.sum())
    S["unreg_day_quantiles"] = reg.date_unregistration.quantile([.25, .5, .75]).round(0).to_dict()
    S["withdrawn_by_d28_share_of_withdrawn"] = round(float(((reg.final_result == "Withdrawn") & (reg.unregistered_by_d28 == 1)).sum()
                                                           / (reg.final_result == "Withdrawn").sum()), 4)
    # ---- representativeness
    S["demographics_active"] = {v: pop[v].value_counts(normalize=True).round(3).to_dict()
                                for v in ["gender", "age_band", "disability", "highest_education"]}
    S["module_domain"] = {"BBB": "Social Sciences", "FFF": "STEM"}
    S["rate_by_imd"] = pop.groupby("imd_band").target_noncompletion.mean().round(3).to_dict()
    S["rate_by_disability"] = pop.groupby("disability").target_noncompletion.mean().round(3).to_dict()
    # ---- temporal coverage
    S["active_students_per_week"] = (daily[(daily.date >= 0)].assign(w=lambda d: d.date // 7 + 1)
                                     .groupby("w").id_student.nunique().head(6).to_dict())

    # ---- quick group contrasts for AQ fit (descriptive only)
    S["median_by_label3_active"] = pop.groupby("label3")[["clicks_0_28", "active_days_0_28"]].median().to_dict()
    S["tma1_submit_rate_by_label3"] = pop.groupby("label3").tma1_submitted_by_d28.mean().round(3).to_dict()

    figures(reg, feat, pop, daily, miss)
    json.dump(S, open(META / "profiling_summary.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(json.dumps(S, indent=1, ensure_ascii=False, default=str)[:3000])


def figures(reg, feat, pop, daily, miss):
    # F1 missingness
    fig, ax = plt.subplots(figsize=(7, 2.6))
    items = sorted(miss.items(), key=lambda kv: kv[1]["rate"])
    ax.barh([k for k, _ in items], [v["rate"] * 100 for _, v in items], color="#2a78d6", height=0.55)
    for i, (_, v) in enumerate(items):
        ax.text(v["rate"] * 100 + 1, i, f"{v['rate']*100:.1f}% (n={v['n']:,})".replace(",", "."), va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Tỷ lệ thiếu (%) – phạm vi BBB+FFF"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v["rate"] for v in miss.values()) * 100 * 1.35)
    save(fig, "fig1_missingness.png")

    # F2 target balance by module-presentation (active d28) – stacked 100%
    t = pop.groupby(["code_module", "code_presentation"]).label3.value_counts(normalize=True).unstack()[list(COL)]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    lbl = [f"{m}\n{p}" for m, p in t.index]
    left = np.zeros(len(t))
    for k in COL:
        ax.bar(lbl, t[k] * 100, bottom=left, color=COL[k], label=k, width=0.62, edgecolor="white", linewidth=1.2)
        left += t[k].values * 100
    ax.set_ylabel("% sinh viên còn học ở ngày 28"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig2_target_balance.png")

    # F3 temporal coverage – active students per day
    d = daily.groupby(["code_module", "date"]).id_student.nunique().reset_index()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for m, c in zip(MODULES, ["#2a78d6", "#eb6834"]):
        x = d[d.code_module == m]
        ax.plot(x.date, x.id_student, color=c, lw=1.4, label=m)
    ax.axvline(0, color=INK2, lw=0.8, ls=":"); ax.axvline(PREDICTION_DAY, color=INK, lw=1.2, ls="--")
    ax.text(PREDICTION_DAY + 3, ax.get_ylim()[1] * 0.9, "mốc dự báo ngày 28", fontsize=8, color=INK)
    ax.set_xlabel("Ngày tương đối so với ngày khai giảng (cộng 4 kỳ)"); ax.set_ylabel("Số SV có hoạt động")
    ax.legend(frameon=False)
    save(fig, "fig3_temporal_coverage.png")

    # F4 weekly clicks by label3 (median)
    sw = pd.read_csv(PROCESSED / "aq1_student_week.csv")
    m = sw.groupby(["label3", "week"]).clicks.median().unstack(0)[list(COL)]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for k in COL:
        ax.plot(m.index, m[k], color=COL[k], lw=2, marker="o", ms=6, label=k)
        ax.text(4.08, m[k].iloc[-1], k, color=INK, fontsize=8, va="center")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["Tuần 1\n(0–6)", "Tuần 2\n(7–13)", "Tuần 3\n(14–20)", "Tuần 4\n(21–28)"])
    ax.set_ylabel("Trung vị số click / tuần"); ax.set_xlim(0.8, 4.5); ax.set_ylim(0)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig4_weekly_clicks_by_result.png")

    # F5 distribution of clicks 0–28 (log scale) – zero inflation
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hist(np.log1p(pop.clicks_0_28), bins=50, color="#2a78d6", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("log(1 + tổng click ngày 0–28)"); ax.set_ylabel("Số sinh viên"); ax.grid(axis="x", visible=False)
    z = (pop.clicks_0_28 == 0).sum()
    ax.annotate(f"{z} SV = 0 click", xy=(0.05, z), xytext=(1.2, z * 1.6 if z > 50 else 300), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    save(fig, "fig5_clicks_distribution.png")

    # F6 unregistration timing
    u = reg.date_unregistration.dropna()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.hist(u, bins=np.arange(-200, 280, 7), color="#1baf7a", edgecolor="white", linewidth=0.5)
    ax.axvline(PREDICTION_DAY, color=INK, lw=1.2, ls="--")
    n28 = int((u <= PREDICTION_DAY).sum())
    ax.text(PREDICTION_DAY + 5, ax.get_ylim()[1] * 0.85, f"{n28:,} lượt hủy ≤ ngày 28".replace(",", "."), fontsize=8)
    ax.set_xlabel("Ngày hủy đăng ký (tương đối với khai giảng)"); ax.set_ylabel("Số lượt"); ax.grid(axis="x", visible=False)
    save(fig, "fig6_unregistration_timing.png")


if __name__ == "__main__":
    main()
