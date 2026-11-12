"""Level 2 – Prototype Experiment (Milestone 4, N05). Chạy đúng Evaluation Plan đã chốt (evaluation/evaluation_plan.json).
Chạy:  python3 src/run_prototype.py   -> results/*.json, results/experiment_log.json, reports/milestone4/figures/*.png
"""
import hashlib, json, platform, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score, recall_score, f1_score,
                             brier_score_loss, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
P4 = Path(__file__).resolve().parents[1]
P3 = P4.parent / "P3"
PROC, RES, FIG = P3 / "data/processed", P4 / "results", P4 / "reports/milestone4/figures"
SEED = 42
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
LOG = []


def log(msg):
    LOG.append({"t": datetime.now(timezone.utc).strftime("%H:%M:%S"), "msg": msg}); print(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_inputs():
    lock = json.loads((P4 / "evaluation/evaluation_plan.lock").read_text())
    assert sha(P4 / "evaluation/evaluation_plan.json") == lock["sha256"], "Evaluation plan đã bị sửa sau khi chốt"
    man = json.loads((P3 / "data/metadata/prepared_manifest.json").read_text())
    for f in ["aq1_student_week.csv", "aq2_diagnostic_table.csv", "aq3_model_table.csv", "aq3_audit_attributes.csv"]:
        assert sha(PROC / f) == man["files"][f]["sha256"], f"prepared file changed: {f}"
    log(f"Input OK: plan {lock['sha256'][:12]} frozen {lock['frozen_utc']}; data {man['prepared_version']}")
    return lock, man


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / name); plt.close(fig)


# ============================================================== AQ1
def aq1():
    sw = pd.read_csv(PROC / "aq1_student_week.csv")
    rng = np.random.default_rng(SEED)
    out = []
    for (lab, w), g in sw.groupby(["label3", "week"]):
        row = {"label3": lab, "week": int(w), "n": int(len(g))}
        for m in ["clicks", "active_days"]:
            v = g[m].to_numpy()
            boots = np.median(rng.choice(v, size=(1000, len(v)), replace=True), axis=1)
            row.update({f"{m}_median": float(np.median(v)), f"{m}_mean": round(float(v.mean()), 2),
                        f"{m}_ci_low": float(np.percentile(boots, 2.5)), f"{m}_ci_high": float(np.percentile(boots, 97.5))})
        out.append(row)
    pop = sw.drop_duplicates(["code_module", "code_presentation", "id_student"])
    checks = {"rows_eq_4x": bool(len(sw) == 4 * len(pop)),
              "each_student_4_weeks": bool(sw.groupby(["code_module", "code_presentation", "id_student"]).size().eq(4).all()),
              "group_counts": pop.label3.value_counts().to_dict(),
              "zero_click_share_by_week": sw.assign(z=sw.clicks.eq(0)).groupby(["label3", "week"]).z.mean().round(3).unstack().to_dict()}
    by_mod = sw.groupby(["code_module", "label3", "week"]).clicks.median().unstack().round(1)
    res = {"table": out, "checks": checks, "median_clicks_by_module": {f"{m}|{l}": r.to_dict() for (m, l), r in by_mod.iterrows()}}
    t = pd.DataFrame(out)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    for ax, m, lab in zip(axes, ["clicks", "active_days"], ["Trung vị click / tuần (dải: KTC 95%)", "Trung bình số ngày hoạt động / tuần"]):
        for k, c in zip(["Pass", "Fail", "Withdrawn"], [C1, C2, C3]):
            d = t[t.label3 == k]
            if m == "clicks":
                ax.fill_between(d.week, d[f"{m}_ci_low"], d[f"{m}_ci_high"], color=c, alpha=0.18, lw=0)
                ax.plot(d.week, d[f"{m}_median"], color=c, lw=2, marker="o", ms=5, label=k)
            else:
                ax.plot(d.week, d[f"{m}_mean"], color=c, lw=2, marker="o", ms=5, label=k)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Tuần học"); ax.set_ylabel(lab); ax.set_ylim(0)
    axes[0].set_title("Số click mỗi tuần", fontsize=9); axes[1].set_title("Số ngày hoạt động mỗi tuần", fontsize=9)
    fig.suptitle("AQ1 – Tương tác 4 tuần đầu theo kết quả cuối khóa", x=0.02, ha="left", fontsize=11, fontweight="bold", y=1.17)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.30))
    save(fig, "p4_fig1_aq1_weekly.png")
    log(f"AQ1 done: checks {checks['rows_eq_4x']} {checks['each_student_4_weeks']}")
    return res


# ============================================================== AQ2
def cliffs_delta(x, y):
    # x: non-completers, y: completers. delta > 0 => non-completers larger
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 2 * u.statistic / (len(x) * len(y)) - 1, u.pvalue


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1, run)
    return adj


def aq2(plan):
    d = pd.read_csv(PROC / "aq2_diagnostic_table.csv")
    P = plan["AQ2"]; rows = []
    for v in P["indicators_continuous"]:
        x = d.loc[d.target_noncompletion == 1, v].dropna(); y = d.loc[d.target_noncompletion == 0, v].dropna()
        delta, p = cliffs_delta(x.to_numpy(), y.to_numpy())
        rows.append({"indicator": v, "type": "continuous", "median_noncompl": float(x.median()), "median_compl": float(y.median()),
                     "effect": round(float(delta), 3), "effect_name": "Cliff's delta", "p": p, "n1": len(x), "n0": len(y)})
    for v in P["indicators_binary"]:
        ct = pd.crosstab(d[v], d.target_noncompletion)
        chi = stats.chi2_contingency(ct)
        r1 = d.loc[d.target_noncompletion == 1, v].mean(); r0 = d.loc[d.target_noncompletion == 0, v].mean()
        rows.append({"indicator": v, "type": "binary", "median_noncompl": round(float(r1), 3), "median_compl": round(float(r0), 3),
                     "effect": round(float(r1 - r0), 3), "effect_name": "risk difference (rate1 - rate0)", "p": chi.pvalue,
                     "n1": int((d.target_noncompletion == 1).sum()), "n0": int((d.target_noncompletion == 0).sum())})
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p)
    thr = np.where(t.type == "continuous", 0.147, 0.05)
    t["significant"] = (t.p_holm < 0.05) & (t.effect.abs() >= thr)
    first_week = None
    for w in range(1, 5):
        s = t.set_index("indicator").significant
        if s[f"clicks_w{w}"] and s[f"active_days_w{w}"]:
            first_week = w; break
    # adjusted logistic regression
    X = d[["active_days_0_28", "clicks_change_w1_w4", "tma1_submitted_by_d28", "num_of_prev_attempts", "studied_credits",
           "highest_education_ord", "days_since_last_active_d28"]].copy()
    X["log_clicks_0_28"] = np.log1p(d.clicks_0_28)
    Xs = (X - X.mean()) / X.std()
    fe = pd.get_dummies(d.code_module + "-" + d.code_presentation, drop_first=True, dtype=float)
    mod = sm.Logit(d.target_noncompletion, sm.add_constant(pd.concat([Xs, fe], axis=1))).fit(disp=0)
    ci = mod.conf_int()
    adj = [{"predictor": c, "OR_per_SD": round(float(np.exp(mod.params[c])), 3), "ci_low": round(float(np.exp(ci.loc[c, 0])), 3),
            "ci_high": round(float(np.exp(ci.loc[c, 1])), 3), "p": float(mod.pvalues[c])} for c in Xs.columns]
    vif_corr = Xs.corr().round(2)
    res = {"unadjusted": t.round(6).to_dict("records"), "first_week": first_week, "adjusted": adj,
           "adjusted_pseudo_r2": round(float(mod.prsquared), 3), "n": int(len(d)),
           "max_abs_corr_predictors": float(vif_corr.where(~np.eye(len(vif_corr), dtype=bool)).abs().max().max())}
    # figure: effect size by week
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for m, c, lab in [("clicks", C1, "Số click"), ("active_days", C2, "Số ngày hoạt động")]:
        e = [t.set_index("indicator").effect[f"{m}_w{w}"] for w in range(1, 5)]
        ax.plot(range(1, 5), e, color=c, lw=2, marker="o", ms=6, label=lab)
    ax.axhline(-0.147, color=INK2, lw=0.8, ls="--"); ax.text(4.05, -0.147, "ngưỡng nhỏ\n|δ| = 0,147", fontsize=7, va="center", color=INK2)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels([f"Tuần {w}" for w in range(1, 5)])
    ax.set_ylabel("Cliff's δ (âm = nhóm không\nhoàn thành thấp hơn)"); ax.set_xlabel("Tuần học"); ax.set_xlim(0.8, 4.6)
    ax.set_title("AQ2 – Độ lớn khác biệt theo tuần (Cliff's δ)", loc="left", fontsize=11, fontweight="bold", pad=24)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "p4_fig2_aq2_effect_by_week.png")
    log(f"AQ2 done: first week {first_week}; significant {int(t.significant.sum())}/{len(t)}")
    return res


# ============================================================== AQ3
def feature_lists(man, with_score):
    fs = man["feature_sets"]
    feats = [f for f in fs["behavioural"] if f != "clicks_0_28"] + fs["academic_d28"] + fs["background"]
    if with_score:
        feats = feats + fs["conditional_score"]
    counts = [f for f in feats if f.startswith(("clicks_", "active_days_pre")) and f != "clicks_change_w1_w4"]
    return feats, counts


def make_lr(feats, counts):
    other = [f for f in feats if f not in counts]
    pre = ColumnTransformer([
        ("cnt", Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("sc", StandardScaler())]), counts),
        ("oth", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), other)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=2000))])


def make_gb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)


def thr_for_recall(y, p, target=0.70):
    prec, rec, th = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(th[ok.max()])


def metrics(y, p, t):
    yhat = (p >= t).astype(int)
    return {"pr_auc": round(float(average_precision_score(y, p)), 4), "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "recall": round(float(recall_score(y, yhat)), 4), "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
            "f1": round(float(f1_score(y, yhat)), 4), "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4),
            "flagged_share": round(float(yhat.mean()), 4)}


def aq3(plan, man):
    df = pd.read_csv(PROC / "aq3_model_table.csv")
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ["train", "validation", "test"])
    y = {k: v.target_noncompletion.to_numpy() for k, v in [("train", tr), ("validation", va), ("test", te)]}
    R, P, models = {}, {}, {}
    # M0 majority/prevalence
    prev = y["train"].mean()
    for s, d in [("validation", va), ("test", te)]:
        P[("M0", s)] = np.full(len(d), prev)
    # M1 simple rule (no fitting): not submitted TMA1 first, then fewer active days
    rule = lambda d: (1 - d.tma1_submitted_by_d28) + (29 - d.active_days_0_28) / 30.0
    for s, d in [("validation", va), ("test", te)]:
        P[("M1", s)] = rule(d).to_numpy() / 2.0
    for ws in [False, True]:
        feats, counts = feature_lists(man, ws)
        tag = "+score" if ws else ""
        lr = make_lr(feats, counts).fit(tr[feats], y["train"])
        gb = make_gb().fit(tr[feats], y["train"])
        models[f"M2{tag}"], models[f"M3{tag}"] = (lr, feats), (gb, feats)
        for s, d in [("validation", va), ("test", te)]:
            P[(f"M2{tag}", s)] = lr.predict_proba(d[feats])[:, 1]
            P[(f"M3{tag}", s)] = gb.predict_proba(d[feats])[:, 1]
    names = ["M0", "M1", "M2", "M3", "M2+score", "M3+score"]
    for mname in names:
        t = 0.5 if mname == "M0" else thr_for_recall(y["validation"], P[(mname, "validation")])
        R[mname] = {"threshold_from_validation": round(t, 4),
                    "validation": metrics(y["validation"], P[(mname, "validation")], t),
                    "test": metrics(y["test"], P[(mname, "test")], t)}
    # M0 PR-AUC = prevalence; recall at 0.5 meaningless -> note
    gain = R["M3"]["validation"]["pr_auc"] - R["M2"]["validation"]["pr_auc"]
    primary = "M3" if gain >= 0.02 else "M2"
    gain_s = R["M3+score"]["validation"]["pr_auc"] - R["M2+score"]["validation"]["pr_auc"]
    sel = {"validation_gain_M3_over_M2": round(gain, 4), "primary": primary,
           "validation_gain_with_score": round(gain_s, 4)}
    ok = (R[primary]["test"]["recall"] >= 0.70 and
          R[primary]["test"]["pr_auc"] > max(R["M0"]["test"]["pr_auc"], R["M1"]["test"]["pr_auc"]))
    sel["success_AQ3"] = bool(ok)
    log(f"AQ3: primary {primary} (gain {gain:+.4f}); test {R[primary]['test']}")

    # calibration + PR curves figure
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for mname, c in [("M1", C4), ("M2", C1), ("M3", C2)]:
        pr, rc, _ = precision_recall_curve(y["test"], P[(mname, "test")])
        axes[0].plot(rc, pr, color=c, lw=1.8, label=f"{mname} (PR-AUC {R[mname]['test']['pr_auc']:.3f})".replace(".", ","))
    axes[0].axhline(y["test"].mean(), color=INK2, ls=":", lw=1); axes[0].text(0.02, y["test"].mean() + 0.02, "M0 = tỷ lệ nền", fontsize=7, color=INK2)
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].legend(frameon=False, fontsize=8); axes[0].set_title("Precision–Recall (test 2014J)", fontsize=9)
    for mname, c in [("M2", C1), ("M3", C2)]:
        fp, mp = calibration_curve(y["test"], P[(mname, "test")], n_bins=10, strategy="quantile")
        axes[1].plot(mp, fp, color=c, lw=1.8, marker="o", ms=4, label=mname)
    axes[1].plot([0, 1], [0, 1], color=INK2, ls="--", lw=0.8)
    axes[1].set_xlabel("Xác suất dự báo"); axes[1].set_ylabel("Tỷ lệ thực tế"); axes[1].legend(frameon=False, fontsize=8); axes[1].set_title("Calibration (test 2014J)", fontsize=9)
    fig.suptitle("AQ3 – Chất lượng xếp hạng và hiệu chỉnh xác suất", x=0.02, ha="left", fontsize=11, fontweight="bold", y=1.04)
    save(fig, "p4_fig3_aq3_pr_calibration.png")
    return df, tr, va, te, y, P, R, sel, models


# ============================================================== AQ4
def aq4(plan, va, te, P, R, primary):
    t = R[primary]["threshold_from_validation"]
    out = {}
    for s, d in [("validation", va), ("test", te)]:
        d = d.assign(p=P[(primary, s)], mp=d.code_module + "-" + d.code_presentation)
        res = {}
        for k in plan["AQ4"]["k"]:
            for rule in ["a", "b", "d"]:
                if rule == "d":
                    sel = d.nlargest(2 * k, "p")
                    per = {mp: {"n_selected": int((sel.mp == mp).sum()), "tp": int(sel[sel.mp == mp].target_noncompletion.sum())} for mp in d.mp.unique()}
                    tp = int(sel.target_noncompletion.sum())
                    res[f"{rule}_k{k}"] = {"precision": round(tp / (2 * k), 4), "tp": tp, "per_module": per}
                    continue
                per = {}
                for mp, g in d.groupby("mp"):
                    if rule == "a":
                        sel = g.nlargest(k, "p")
                    else:
                        cand = g[g.p >= t].sort_values("clicks_change_w1_w4")
                        sel = cand.head(k)
                    tp = int(sel.target_noncompletion.sum())
                    per[mp] = {"n_selected": int(len(sel)), "tp": tp, "precision": round(tp / max(len(sel), 1), 4),
                               "expected_changed_14pct": round(tp * 0.14, 1), "cohort": int(len(g)),
                               "noncompl_in_cohort": int(g.target_noncompletion.sum())}
                prec = np.mean([v["precision"] for v in per.values()])
                res[f"{rule}_k{k}"] = {"mean_precision": round(float(prec), 4), "per_module": per}
        out[s] = res
    val200 = {r: (out["validation"][f"{r}_k200"].get("mean_precision") or out["validation"][f"{r}_k200"]["precision"]) for r in ["a", "b", "d"]}
    chosen_as_frozen = max(val200, key=val200.get)
    # PD-01 (plan deviation): kế hoạch quên kiểm tra tính khả thi. AQ4/C05 ràng buộc tối đa k ca MỖI module-kỳ.
    feasible = {}
    for r in ["a", "b", "d"]:
        per = out["validation"][f"{r}_k200"]["per_module"]
        feasible[r] = all(v["n_selected"] <= 200 for v in per.values())
    chosen = max([r for r in val200 if feasible[r]], key=val200.get)
    test_chosen = out["test"][f"{chosen}_k200"]
    if chosen == "d":
        succ = test_chosen["precision"] >= 0.70
    else:
        succ = all(v["precision"] >= 0.70 for v in test_chosen["per_module"].values())
    out["selection"] = {"validation_precision_at_200": val200, "chosen_by_frozen_rule": chosen_as_frozen,
                        "feasible_under_capacity_200_per_module": feasible, "chosen_rule": chosen,
                        "plan_deviation": "PD-01: loại quy tắc vi phạm trần 200 ca/module-kỳ (C05) trước khi chọn",
                        "success_AQ4": bool(succ)}
    # scenario on SMART: per module-presentation
    sc = {}
    if chosen != "d":
        for mp, v in test_chosen["per_module"].items():
            base = v["noncompl_in_cohort"] / v["cohort"]
            sc[mp] = {"baseline_rate": round(base, 4), "after_rate_if_14pct": round((v["noncompl_in_cohort"] - v["tp"] * 0.14) / v["cohort"], 4),
                      "drop_pp": round(v["tp"] * 0.14 / v["cohort"] * 100, 2)}
    out["smart_scenario_test"] = sc
    # figure precision@k
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ks = plan["AQ4"]["k"]; w = 0.25
    for i, (rule, c, lab) in enumerate([("a", C1, "(a) theo xác suất, hạn mức/module"), ("b", C3, "(b) suy giảm tuần 1→4"), ("d", C4, "(d) gộp – vượt trần 200 ca/module")]):
        v = [out["test"][f"{rule}_k{k}"].get("mean_precision", out["test"][f"{rule}_k{k}"].get("precision")) for k in ks]
        ax.bar(np.arange(3) + (i - 1) * w, v, width=w * 0.92, color=c, label=lab)
    ax.axhline(0.70, color=INK, ls="--", lw=1); ax.text(2.45, 0.71, "mục tiêu 0,70", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"k = {k}" for k in ks]); ax.set_ylabel("Precision@k (test 2014J)"); ax.set_xlabel("Hạn mức k (số ca liên hệ mỗi module – kỳ)"); ax.set_ylim(0, 1)
    ax.set_title("AQ4 – Precision@k theo quy tắc lập danh sách", loc="left", fontsize=11, fontweight="bold", pad=28)
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "p4_fig4_aq4_precision_at_k.png")
    log(f"AQ4: chosen {chosen}, val P@200 {val200}, success {succ}")
    return out


# ============================================================== AQ5
LABEL = {"active_days_0_28": "số ngày có hoạt động trên VLE trong 4 tuần đầu", "log_clicks_0_28": "tổng lượt tương tác 4 tuần đầu",
         "days_since_last_active_d28": "số ngày kể từ lần hoạt động gần nhất", "tma1_submitted_by_d28": "việc nộp bài đánh giá đầu tiên",
         "tma1_on_time": "việc nộp bài đánh giá đầu tiên đúng hạn", "tma1_days_before_deadline": "số ngày nộp trước hạn bài đánh giá đầu tiên",
         "clicks_change_w1_w4": "mức thay đổi tương tác giữa tuần 1 và tuần 4", "num_of_prev_attempts": "số lần đã học lại module",
         "studied_credits": "tổng tín chỉ đang học song song", "highest_education_ord": "trình độ học vấn đầu vào",
         "registration_lead_days": "số ngày đăng ký trước khai giảng", "no_vle_activity_0_28": "việc không có hoạt động VLE nào",
         "n_sites_0_28": "số tài nguyên học tập đã mở", "date_registration_missing": "thiếu ngày đăng ký"}


def label_of(f):
    if f in LABEL: return LABEL[f]
    if f.startswith("clicks_w"): return f"lượt tương tác tuần {f[-1]}"
    if f.startswith("active_days_w"): return f"số ngày hoạt động tuần {f[-1]}"
    if f.startswith("clicks_") and f.endswith("_0_28"): return f"lượt tương tác với tài nguyên {f[7:-5]}"
    if "pre_start" in f: return "hoạt động trước khai giảng"
    return f


def fmt(v):
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}".replace(".", ",")


def sentence(f, v, med):
    """Câu mô tả hành vi quan sát được, trung tính, kèm giá trị thật và trung vị tập train (không quy kết năng lực)."""
    if f == "tma1_submitted_by_d28":
        return "Chưa nộp bài đánh giá đầu tiên (TMA1) tính đến ngày 28." if v == 0 else "Đã nộp bài đánh giá đầu tiên."
    if f == "tma1_on_time":
        return "Bài đánh giá đầu tiên chưa được nộp đúng hạn." if v == 0 else "Bài đánh giá đầu tiên được nộp đúng hạn."
    if f == "no_vle_activity_0_28":
        return "Không có hoạt động nào trên VLE trong 4 tuần đầu."
    if f == "days_since_last_active_d28":
        return f"Đã {fmt(v)} ngày không có hoạt động trên VLE tính đến ngày 28 (trung vị: {fmt(med)} ngày)."
    if pd.isna(v):
        return f"Không có dữ liệu về {label_of(f)}."
    direction = "thấp hơn" if v < med else "cao hơn"
    lab = label_of(f)
    return f"{lab[:1].upper() + lab[1:]} {direction} mức thường gặp ({fmt(v)} so với trung vị {fmt(med)})."


def aq5(man, te, P, models, primary_feats_model, sens):
    lr, feats = models["M2"]
    pre, clf = lr.named_steps["pre"], lr.named_steps["lr"]
    Z = pre.transform(te[feats]); names = [c.split("__", 1)[1] for c in pre.get_feature_names_out()]
    contrib = Z * clf.coef_[0]
    logit = clf.intercept_[0] + contrib.sum(1)
    faithful_err = float(np.abs(logit - lr.decision_function(te[feats])).max())
    te = te.assign(p=P[("M2", "test")], mp=te.code_module + "-" + te.code_presentation)
    trdf = pd.read_csv(PROC / "aq3_model_table.csv").query("split == 'train'")
    med = trdf[feats].median()
    raw = te.copy(); raw["log_clicks_0_28"] = te["log_clicks_0_28"]
    BACKGROUND = set(man["feature_sets"]["background"])

    def pick(idx):
        """Lý do hiển thị: chỉ biến hành vi/học thuật (cố vấn có thể tác động), bỏ lý do TMA1 trùng lặp; tối đa 3."""
        c = contrib[idx]; order = np.argsort(-c); out = []
        for j in order:
            f = names[j]
            if c[j] <= 0 or len(out) == 3:
                break
            if f in BACKGROUND:
                continue
            if f in ("tma1_on_time", "tma1_days_before_deadline") and any(g == "tma1_submitted_by_d28" for g, _ in out) and raw.loc[idx, "tma1_submitted_by_d28"] == 0:
                continue
            out.append((f, c[j]))
        return out

    explained, examples, seen_sets, counter, n_counter_students = [], [], set(), [], []
    n_sentences = []
    for mp, g in te.groupby("mp"):
        top = g.nlargest(200, "p")
        for idx in top.index:
            reasons = pick(idx)
            explained.append(len(reasons)); n_sentences.append(len(reasons))
            for f, _ in reasons:
                v = raw.loc[idx, f]
                if f.startswith(("clicks_", "log_clicks", "active_days", "n_sites")) and f != "clicks_change_w1_w4" and pd.notna(v) and v > med[f]:
                    counter.append(f)
            n_counter_students.append(any(f.startswith(("clicks_", "log_clicks", "active_days", "n_sites")) and f != "clicks_change_w1_w4"
                                          and raw.loc[idx, f] > med[f] for f, _ in reasons))
            key = tuple(f for f, _ in reasons)
            want_fp = te.loc[idx, "target_noncompletion"] == 0 and not any(e["actual"] == 0 for e in examples)
            if (len(examples) < 3 and key not in seen_sets) or (want_fp and len(examples) < 5):
                seen_sets.add(key)
                vals = {f: (float(np.expm1(raw.loc[idx, f])) if f == "log_clicks_0_28" else raw.loc[idx, f]) for f, _ in reasons}
                meds = {f: (float(np.expm1(med[f])) if f == "log_clicks_0_28" else med[f]) for f, _ in reasons}
                examples.append({"student": f"{mp}-{int(te.loc[idx, 'id_student'])}", "p": round(float(te.loc[idx, "p"]), 3),
                                 "actual": int(te.loc[idx, "target_noncompletion"]),
                                 "reasons": [{"feature": f, "contribution_logit": round(float(cc), 3)} for f, cc in reasons],
                                 "text": " ".join(sentence(f.replace("log_clicks_0_28", "log_clicks_0_28"), vals[f], meds[f]) for f, _ in reasons)})
    # lưu dự đoán + 3 lý do cho mọi SV test (phục vụ dashboard wireframe ở P5)
    rows = []
    for idx in te.index:
        rs = [f for f, _ in pick(idx)]
        txt = []
        for f in rs:
            v = float(np.expm1(raw.loc[idx, f])) if f == "log_clicks_0_28" else raw.loc[idx, f]
            m = float(np.expm1(med[f])) if f == "log_clicks_0_28" else med[f]
            txt.append(sentence(f, v, m))
        rows.append({"reason_1": txt[0] if len(txt) > 0 else "", "reason_2": txt[1] if len(txt) > 1 else "",
                     "reason_3": txt[2] if len(txt) > 2 else "",
                     "reason_f1": rs[0] if len(rs) > 0 else "", "reason_f2": rs[1] if len(rs) > 1 else "", "reason_f3": rs[2] if len(rs) > 2 else ""})
    pred = pd.concat([te[["code_module", "code_presentation", "id_student", "target_noncompletion", "p"] +
                         [f"clicks_w{i}" for i in range(1, 5)] + ["active_days_0_28", "tma1_submitted_by_d28", "tma1_on_time", "days_since_last_active_d28"]].reset_index(drop=True),
                      pd.DataFrame(rows)], axis=1)
    pred["p_M1"] = P[("M1", "test")]; pred["p_M3"] = P[("M3", "test")]
    pred = pred.rename(columns={"p": "p_M2"})
    pred["rank_in_module"] = pred.groupby(["code_module"])["p_M2"].rank(ascending=False, method="first").astype(int)
    pred["in_top200"] = (pred.rank_in_module <= 200).astype(int)
    pred.to_csv(RES / "test_predictions.csv", index=False)
    coef = pd.Series(clf.coef_[0], index=names).sort_values()
    va_df = pd.read_csv(PROC / "aq3_model_table.csv").query("split == 'validation'")
    pi = permutation_importance(lr, va_df[feats], va_df.target_noncompletion, scoring="average_precision", n_repeats=10, random_state=SEED, n_jobs=4)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    res = {"faithfulness_max_abs_error": faithful_err, "n_explained": len(explained),
           "share_with_ge2_reasons": round(float(np.mean(np.array(explained) >= 2)), 4),
           "sensitive_in_features": [f for f in feats if f in sens], "examples": examples,
           "coef_top_positive": coef.tail(6).round(3).to_dict(), "coef_top_negative": coef.head(6).round(3).to_dict(),
           "permutation_importance_top10": imp.head(10).round(4).to_dict()}
    # pre-registered check: sign consistency with AQ2 for top global features
    a2 = json.loads((RES / "aq2_results.json").read_text())
    ref = {r["indicator"]: np.sign(r["effect"]) for r in a2["unadjusted"] if r["significant"]}
    ref.update({r["predictor"]: np.sign(np.log(r["OR_per_SD"])) for r in a2["adjusted"] if r["p"] < 0.05 and r["predictor"] not in ref})
    sign = {}
    for f in imp.head(10).index:
        if f in ref:
            sign[f] = {"aq2_sign": int(ref[f]), "lr_sign": int(np.sign(coef[f])), "consistent": bool(ref[f] == np.sign(coef[f]))}
    res["sign_consistency_top10"] = sign
    eng = [f for f in names if f.startswith(("clicks_w", "active_days_w", "clicks_") ) and f not in ("clicks_change_w1_w4",)] + \
          [f for f in ("log_clicks_0_28", "active_days_0_28", "n_sites_0_28", "clicks_pre_start", "active_days_pre_start") if f in names]
    eng = list(dict.fromkeys(eng))
    res["engagement_features_positive_coef"] = {f: round(float(coef[f]), 3) for f in eng if coef[f] > 0}
    res["max_sentences_per_student"] = int(max(n_sentences)) if n_sentences else 0
    res["background_excluded_from_reasons"] = sorted(BACKGROUND)
    res["counterintuitive_reasons"] = {"share_students": round(float(np.mean(n_counter_students)), 4),
                                       "by_feature": pd.Series(counter).value_counts().to_dict()}
    res["n_inconsistent"] = int(sum(not v["consistent"] for v in sign.values()))
    res["success_AQ5"] = bool(res["faithfulness_max_abs_error"] < 1e-6 and res["share_with_ge2_reasons"] == 1.0
                              and not res["sensitive_in_features"] and res["n_inconsistent"] == 0)
    fig, ax = plt.subplots(figsize=(7, 3))
    top = imp.head(8)[::-1]
    ax.barh([label_of(f) for f in top.index], top.values, color=C1, height=0.55)
    ax.set_xlabel("Mức giảm PR-AUC khi xáo trộn (validation 2014B)"); ax.set_ylabel("Biến đặc trưng"); ax.grid(axis="y", visible=False)
    ax.set_title("AQ5 – 8 biến quan trọng nhất (permutation importance)", loc="left", fontsize=11, fontweight="bold")
    save(fig, "p4_fig5_aq5_permutation_importance.png")
    log(f"AQ5: faithful err {faithful_err:.2e}; >=2 reasons {res['share_with_ge2_reasons']}")
    return res


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(c - h, 3), round(c + h, 3)]


def extras(tr, va, te, y, P, R, primary, r4):
    """Bổ sung sau rà soát: khoảng tin cậy, so sánh cùng mức recall, calibration, ngưỡng thay thế, drift TMA1."""
    rng = np.random.default_rng(SEED); yt = y["test"]; n = len(yt); out = {}
    bs = {m: [] for m in ["M1", "M2", "M3"]}
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if yt[i].sum() == 0:
            continue
        for m in bs:
            bs[m].append(average_precision_score(yt[i], P[(m, "test")][i]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    out["pr_auc_ci95"] = {m: q(v) for m, v in bs.items()}
    out["pr_auc_diff_ci95"] = {"M2-M1": q(np.array(bs["M2"]) - np.array(bs["M1"])), "M2-M3": q(np.array(bs["M2"]) - np.array(bs["M3"]))}
    t = R[primary]["threshold_from_validation"]; yh = P[(primary, "test")] >= t
    tp = int((yh & (yt == 1)).sum()); pos = int(yt.sum())
    out["recall_test_ci95"] = {"tp": tp, "pos": pos, "ci": wilson(tp, pos)}
    # cùng mức recall 0,70 trên test (mô tả, không dùng để chọn)
    def prec_at_recall(p, r=0.70):
        pr, rc, _ = precision_recall_curve(yt, p); ok = rc[:-1] >= r
        return round(float(pr[:-1][ok].max()), 4)
    out["precision_at_recall_0_70_test"] = {m: prec_at_recall(P[(m, "test")]) for m in ["M1", "M2", "M3"]}
    out["flag_share_at_t"] = {m: R[m]["test"]["flagged_share"] for m in ["M1", "M2", "M3"]}
    # recall của chính danh sách top-200
    te2 = te.assign(p=P[(primary, "test")], mp=te.code_module + "-" + te.code_presentation)
    out["top200_list_recall"] = {mp: round(float(g.nlargest(200, "p").target_noncompletion.sum() / g.target_noncompletion.sum()), 3) for mp, g in te2.groupby("mp")}
    out["precision_at_200_ci95"] = {mp: wilson(v["tp"], v["n_selected"]) for mp, v in r4["test"]["a_k200"]["per_module"].items()}
    # calibration
    cal = {}
    for s_, d_ in [("validation", va), ("test", te)]:
        p_ = np.clip(P[(primary, s_)], 1e-6, 1 - 1e-6); lg = np.log(p_ / (1 - p_))
        fit = sm.Logit(y[s_], sm.add_constant(lg)).fit(disp=0)
        cal[s_] = {"mean_pred": round(float(p_.mean()), 4), "base_rate": round(float(y[s_].mean()), 4),
                   "slope": round(float(fit.params[1]), 3), "intercept": round(float(fit.params[0]), 3)}
    out["calibration"] = cal
    out["brier_base_rate_test"] = round(float(yt.mean() * (1 - yt.mean())), 4)
    out["roc_auc_val_vs_test"] = {"validation": R[primary]["validation"]["roc_auc"], "test": R[primary]["test"]["roc_auc"]}
    # ngưỡng thay thế (chọn trên validation)
    alt = {}
    for rr in [0.6, 0.7, 0.8]:
        tt = thr_for_recall(y["validation"], P[(primary, "validation")], rr)
        alt[str(rr)] = {"threshold": round(tt, 4), **{k: v for k, v in metrics(yt, P[(primary, "test")], tt).items() if k in ("recall", "precision", "flagged_share")}}
    out["alternative_thresholds"] = alt
    # drift ý nghĩa TMA1 (bằng chứng cho DQ10)
    a = pd.read_csv(P4.parent / "P2/data/raw/oulad/assessments.csv", na_values="?")
    a = a[a.code_module.isin(["BBB", "FFF"]) & (a.assessment_type != "Exam")].sort_values("date").groupby(["code_module", "code_presentation"]).head(1)
    allm = pd.concat([tr, va, te])
    med = allm.groupby(["code_module", "code_presentation", "target_noncompletion"]).tma1_score.median().unstack()
    out["tma1_drift"] = {f"{r.code_module}-{r.code_presentation}": {"weight": float(r.weight), "deadline": int(r.date),
                          "median_score_completers": float(med.loc[(r.code_module, r.code_presentation), 0]),
                          "median_score_noncompleters": float(med.loc[(r.code_module, r.code_presentation), 1])} for r in a.itertuples()}
    log(f"Extras: PR-AUC CI {out['pr_auc_ci95']['M2']}, recall CI {out['recall_test_ci95']['ci']}, cal test {cal['test']}")
    return out


def exploratory_reduced(man, tr, va, te, y, R):
    """PD-02 (khám phá, ngoài plan đã chốt): LR trên bộ feature rút gọn của P3 v1.1 (VIF ≤ 5).
    Không dùng để thay primary; chỉ kiểm tra trade-off hiệu năng – khả năng giải thích."""
    feats = man["feature_sets"]["reduced_core"]
    counts = [f for f in feats if f.startswith(("clicks_", "active_days_pre")) and f != "clicks_change_w1_w4"]
    lr = make_lr(feats, counts).fit(tr[feats], y["train"])
    pv, pt = lr.predict_proba(va[feats])[:, 1], lr.predict_proba(te[feats])[:, 1]
    t = thr_for_recall(y["validation"], pv)
    pre, clf = lr.named_steps["pre"], lr.named_steps["lr"]
    names = [c.split("__", 1)[1] for c in pre.get_feature_names_out()]
    coef = pd.Series(clf.coef_[0], index=names)
    a2 = json.loads((RES / "aq2_results.json").read_text())
    ref = {r["indicator"]: np.sign(r["effect"]) for r in a2["unadjusted"] if r["significant"]}
    beh = [f for f in names if f in ref]
    sign = {f: bool(np.sign(coef[f]) == ref[f]) for f in beh}
    out = {"n_features": len(feats), "threshold_from_validation": round(t, 4),
           "validation": metrics(y["validation"], pv, t), "test": metrics(y["test"], pt, t),
           "primary_test_pr_auc": R["M2"]["test"]["pr_auc"], "sign_check_vs_aq2": sign,
           "n_sign_inconsistent": int(sum(not v for v in sign.values())),
           "note": "PD-02: phân tích khám phá sau khi P4 phát hiện đa cộng tuyến; không thay đổi primary approach đã chốt"}
    te2 = te.assign(p=pt, mp=te.code_module + "-" + te.code_presentation)
    out["precision_at_200"] = {mp: round(float(g.nlargest(200, "p").target_noncompletion.mean()), 4) for mp, g in te2.groupby("mp")}
    # lý do ngược trực giác ở mô hình rút gọn (cùng quy tắc chọn lý do như AQ5)
    Z = pre.transform(te[feats]); contrib = Z * clf.coef_[0]
    med = tr[feats].median(); bg = set(man["feature_sets"]["background"])
    eng_pref = ("clicks_", "log_clicks", "active_days", "n_sites")
    out["engagement_features_positive_coef"] = {f: round(float(coef[f]), 3) for f in names if f.startswith(eng_pref) and f != "clicks_change_w1_w4" and coef[f] > 0}
    flags = []
    for mp, g in te2.groupby("mp"):
        for idx in g.nlargest(200, "p").index:
            c = contrib[idx]; order = np.argsort(-c); rs = []
            for j in order:
                if c[j] <= 0 or len(rs) == 3:
                    break
                if names[j] in bg:
                    continue
                rs.append(names[j])
            flags.append(any(f.startswith(eng_pref) and f != "clicks_change_w1_w4" and te.loc[idx, f] > med[f] for f in rs))
    out["counterintuitive_share_top200"] = round(float(np.mean(flags)), 4)
    log(f"PD-02 reduced LR: test {out['test']['pr_auc']} vs primary {out['primary_test_pr_auc']}; sign inconsistent {out['n_sign_inconsistent']}")
    return out


# ============================================================== fairness + error analysis
def fairness_error(te, P, R, primary):
    aud = pd.read_csv(PROC / "aq3_audit_attributes.csv")
    d = te.merge(aud, on=["code_module", "code_presentation", "id_student"], how="left", validate="1:1")
    t = R[primary]["threshold_from_validation"]
    d = d.assign(p=P[(primary, "test")], yhat=lambda x: (x.p >= t).astype(int), mp=d.code_module + "-" + d.code_presentation)
    d["age_grp"] = np.where(d.age_band == "0-35", "0-35", "35+")
    def imdg(s):
        if s == "Unknown": return "Unknown"
        lo = int(s.split("-")[0]); return "0-30%" if lo < 30 else ("30-70%" if lo < 70 else "70-100%")
    d["imd_group"] = d.imd_band.map(imdg)
    top = d.groupby("mp", group_keys=False).apply(lambda g: g.nlargest(200, "p"))
    out = {}
    for attr, col in [("gender", "gender"), ("age_band", "age_grp"), ("disability", "disability"), ("imd_group", "imd_group")]:
        rows = {}
        for gname, g in d.groupby(col):
            pos = g[g.target_noncompletion == 1]
            neg = g[g.target_noncompletion == 0]
            rows[gname] = {"n": int(len(g)), "n_pos": int(len(pos)), "base_rate": round(float(g.target_noncompletion.mean()), 3),
                           "fpr": round(float(neg.yhat.mean()), 3) if len(neg) else None,
                           "fnr": round(float(1 - pos.yhat.mean()), 3) if len(pos) else None,
                           "recall_ci95": wilson(int(pos.yhat.sum()), len(pos)),
                           "recall": round(float(pos.yhat.mean()), 3) if len(pos) else None,
                           "precision": round(float(g[g.yhat == 1].target_noncompletion.mean()), 3) if g.yhat.sum() else None,
                           "share_population": round(float(len(g) / len(d)), 3),
                           "share_top200": round(float((top[col] == gname).mean()), 3)}
        rec = [v["recall"] for v in rows.values() if v["recall"] is not None]
        out[attr] = {"groups": rows, "recall_gap": round(max(rec) - min(rec), 3), "flag": bool(max(rec) - min(rec) > 0.10)}
    pos = d[d.target_noncompletion == 1]
    fn = pos[pos.yhat == 0]; tp = pos[pos.yhat == 1]
    tea = pd.read_csv(PROC / "aq2_diagnostic_table.csv")[["code_module", "code_presentation", "id_student", "final_result"]]
    dd = d.merge(tea, on=["code_module", "code_presentation", "id_student"])
    err = {"recall_by_final_result": dd[dd.target_noncompletion == 1].groupby("final_result").yhat.mean().round(3).to_dict(),
           "fn_profile_median": fn[["active_days_0_28", "clicks_w4", "tma1_submitted_by_d28", "days_since_last_active_d28"]].median().to_dict(),
           "tp_profile_median": tp[["active_days_0_28", "clicks_w4", "tma1_submitted_by_d28", "days_since_last_active_d28"]].median().to_dict(),
           "n_fn": int(len(fn)), "n_fp": int(((d.target_noncompletion == 0) & (d.yhat == 1)).sum()),
           "by_module": {mp: metrics(g.target_noncompletion.to_numpy(), g.p.to_numpy(), t) for mp, g in d.groupby("mp")},
           "excluding_seen_earlier": metrics(d[d.seen_in_earlier_presentation == 0].target_noncompletion.to_numpy(),
                                             d[d.seen_in_earlier_presentation == 0].p.to_numpy(), t)}
    log(f"Fairness flags: { {k: v['flag'] for k, v in out.items()} }")
    return out, err


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    lock, man = check_inputs()
    plan = json.loads((P4 / "evaluation/evaluation_plan.json").read_text())
    r1 = aq1()
    r2 = aq2(plan)
    (RES / "aq2_results.json").write_text(json.dumps(r2, indent=2, ensure_ascii=False, default=float))
    df, tr, va, te, y, P, R, sel, models = aq3(plan, man)
    r4 = aq4(plan, va, te, P, R, sel["primary"])
    r5 = aq5(man, te, P, models, None, man["feature_sets"]["audit_only_sensitive"])
    fair, err = fairness_error(te, P, R, sel["primary"])
    red = exploratory_reduced(man, tr, va, te, y, R)
    ext = extras(tr, va, te, y, P, R, sel["primary"], r4)
    plan_data = plan.get("data_input", "")
    pd03 = {"plan_data_input": plan_data, "actual_data_version": man["prepared_version"],
            "note": "PD-03: plan chốt với v1.0; chạy chính thức trên v1.2 sau khi sửa lỗi dữ liệu do rà soát phát hiện (tuần 4 dài 8 ngày; cờ thiếu ngày đăng ký làm predictor). Metric, mô hình, ngưỡng, quy tắc chọn giữ nguyên."}
    split_sizes = {s: {"n": int(len(d)), "rate": round(float(d.target_noncompletion.mean()), 4)} for s, d in [("train", tr), ("validation", va), ("test", te)]}
    for name, obj in [("aq1_results", r1), ("aq2_results", r2), ("aq3_results", {"models": R, "selection": sel, "split": split_sizes, "exploratory_reduced_PD02": red, "extras": ext, "PD03": pd03}),
                      ("aq4_results", r4), ("aq5_results", r5), ("fairness_error_results", {"fairness": fair, "error": err})]:
        (RES / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "statsmodels": __import__("statsmodels").__version__, "seed": SEED, "plan_sha256": lock["sha256"],
           "plan_frozen_utc": lock["frozen_utc"], "data_version": man["prepared_version"],
           "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "runtime_s": round(time.time() - t0, 1)}
    (RES / "experiment_log.json").write_text(json.dumps({"environment": env, "log": LOG}, indent=2, ensure_ascii=False))
    print("done", env["runtime_s"], "s")


if __name__ == "__main__":
    main()
