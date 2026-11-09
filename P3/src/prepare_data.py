"""Reproducible Data Preparation Pipeline – Milestone 3 (N05).

raw (P2/data/raw, bất biến) -> data/interim -> data/processed (+ metadata).
Chạy:  python3 src/prepare_data.py
Mỗi bước tương ứng 10 bước trong mục 10 của đề Milestone 3 và ghi vào transformation_log.json.
"""
import hashlib, json, platform, sys, time
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import *  # noqa: E402,F403

LOG = []          # transformation log
CHECKS = {}       # validation summary
KEY = ["code_module", "code_presentation", "id_student"]


def _vn(text):
    """Định dạng số kiểu Việt Nam (10.655.280) và mũi tên chuẩn trong log."""
    import re
    t = str(text).replace("->", "→")
    for _ in range(3):
        t = re.sub(r"(\d),(\d{3})", r"\1.\2", t)
    return t


def log(step, action, rows_in=None, rows_out=None, note=""):
    action, note = _vn(action), _vn(note)
    LOG.append({"step": step, "action": action, "rows_in": rows_in, "rows_out": rows_out, "note": note})
    print(f"[{step}] {action} {'' if rows_in is None else f'{rows_in:,} -> {rows_out:,}'} {note}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. load + validate
def load_raw():
    expected = {}
    for line in RAW_SHA.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip().lstrip("*").split("/")[-1]] = digest
    ok = {}
    for f in ["studentInfo.csv", "studentRegistration.csv", "assessments.csv",
              "studentAssessment.csv", "vle.csv", "courses.csv", "studentVle.csv"]:
        if f in expected:
            ok[f] = sha256(RAW / f) == expected[f]
            assert ok[f], f"checksum mismatch: {f} – raw đã bị thay đổi"
    CHECKS["raw_checksum_ok"] = ok
    schema = {
        "studentInfo.csv": ["code_module", "code_presentation", "id_student", "gender", "region",
                            "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
                            "studied_credits", "disability", "final_result"],
        "studentRegistration.csv": ["code_module", "code_presentation", "id_student",
                                    "date_registration", "date_unregistration"],
        "assessments.csv": ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
        "studentAssessment.csv": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
        "vle.csv": ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"],
        "courses.csv": ["code_module", "code_presentation", "module_presentation_length"],
    }
    T = {}
    for f, cols in schema.items():
        df = pd.read_csv(RAW / f, na_values=NA)
        assert list(df.columns) == cols, (f, list(df.columns))
        T[f.split(".")[0]] = df
    dt = {"code_module": "category", "code_presentation": "category", "id_student": "int32",
          "id_site": "int32", "date": "int16", "sum_click": "int32"}
    v = pd.read_csv(RAW / "studentVle.csv", dtype=dt)
    assert list(v.columns) == ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    T["studentVle"] = v
    log(1, "Load raw + kiểm tra SHA-256 và schema", note=f"{len(T)} bảng, studentVle {len(v):,} dòng")
    return T


# ---------------------------------------------------------------- 2. scope
def select_scope(T):
    out = {}
    for name, df in T.items():
        if "code_module" in df.columns:
            n0 = len(df)
            df = df[df["code_module"].isin(MODULES) & df["code_presentation"].isin(PRESENTATIONS)].copy()
            for c in ("code_module", "code_presentation"):
                df[c] = df[c].astype(str)
            log(2, f"Chọn phạm vi BBB+FFF, 4 kỳ: {name}", n0, len(df))
        out[name] = df
    sa = out["studentAssessment"]
    n0 = len(sa)
    out["studentAssessment"] = sa[sa["id_assessment"].isin(out["assessments"]["id_assessment"])].copy()
    log(2, "Chọn phạm vi: studentAssessment (theo id_assessment của BBB+FFF)", n0, len(out["studentAssessment"]))
    return out


# ---------------------------------------------------------------- 3-6. standardize, keys, missing, invalid
def build_registrations(T):
    si, sr = T["studentInfo"], T["studentRegistration"]
    for name, df in (("studentInfo", si), ("studentRegistration", sr)):
        dups = int(df.duplicated(KEY).sum())
        CHECKS[f"dup_key_{name}"] = dups
        assert dups == 0, f"duplicate key in {name}"
    n0 = len(si)
    reg = si.merge(sr, on=KEY, how="left", validate="1:1", indicator=True)
    CHECKS["registration_join_unmatched"] = int((reg["_merge"] != "both").sum())
    reg = reg.drop(columns="_merge")
    log(7, "Ghép studentInfo + studentRegistration (1:1, validate)", n0, len(reg))

    # 3. standardize labels (DQ-05) and types
    reg["imd_band"] = reg["imd_band"].replace({"10-20": "10-20%"})
    log(3, "Chuẩn hóa nhãn imd_band '10-20' -> '10-20%' (DQ-05)")
    # 5. missing values
    reg["imd_band_missing"] = reg["imd_band"].isna().astype(int)
    reg["imd_band"] = reg["imd_band"].fillna("Unknown")
    reg["date_registration_missing"] = reg["date_registration"].isna().astype(int)
    log(5, "imd_band thiếu -> nhãn 'Unknown' + cờ; date_registration thiếu -> giữ NaN + cờ (impute ở bước fit-on-train)",
        note=f"imd_band thiếu {int(reg['imd_band_missing'].sum())}, date_registration thiếu {int(reg['date_registration_missing'].sum())}")

    # target + labels
    reg["target_noncompletion"] = reg["final_result"].isin(NONCOMPLETION).astype(int)
    reg["label3"] = reg["final_result"].replace({"Distinction": "Pass"})
    # population flags (C04, C05)
    unreg = reg["date_unregistration"]
    reg["unregistered_by_d28"] = (unreg <= PREDICTION_DAY).fillna(False).astype(int)
    reg["registered_after_d28"] = (reg["date_registration"] > PREDICTION_DAY).fillna(False).astype(int)
    reg["active_d28"] = ((reg["unregistered_by_d28"] == 0) & (reg["registered_after_d28"] == 0)).astype(int)
    # 6. inconsistent label vs. unregistration date (DQ-04): keep final_result as official outcome, flag
    reg["flag_withdrawn_no_unreg_date"] = ((reg["final_result"] == "Withdrawn") & unreg.isna()).astype(int)
    reg["flag_unreg_date_not_withdrawn"] = ((reg["final_result"] != "Withdrawn") & unreg.notna()).astype(int)
    log(6, "Giữ final_result làm outcome chính thức; gắn cờ ca mâu thuẫn với date_unregistration (DQ-04)",
        note=f"Withdrawn không có ngày hủy {int(reg['flag_withdrawn_no_unreg_date'].sum())}, có ngày hủy nhưng không Withdrawn {int(reg['flag_unreg_date_not_withdrawn'].sum())}")
    reg["split"] = reg["code_presentation"].map(SPLIT)
    return reg


def vle_features(T, reg):
    v = T["studentVle"]
    n0 = len(v)
    # 4. duplicates: grain = một lượt truy cập, KHÔNG phải một ngày (xem DQ-06) -> giữ và cộng dồn
    k = ["code_module", "code_presentation", "id_student", "id_site", "date"]
    g = v.groupby(k, observed=True)["sum_click"].agg(["size", "nunique"])
    CHECKS["vle_exact_duplicate_extra_rows"] = int(v.duplicated().sum())
    CHECKS["vle_keys_multi_rows"] = int((g["size"] > 1).sum())
    CHECKS["vle_keys_multi_rows_diff_clicks"] = int(((g["size"] > 1) & (g["nunique"] > 1)).sum())
    log(4, "Giữ dòng trùng studentVle (nhiều lượt/ngày là hợp lệ), cộng dồn theo ngày", note=
        f"dòng trùng hoàn toàn {CHECKS['vle_exact_duplicate_extra_rows']:,}; khóa nhiều dòng có click khác nhau {CHECKS['vle_keys_multi_rows_diff_clicks']:,}")

    vle = T["vle"][["id_site", "activity_type"]]
    v = v.merge(vle, on="id_site", how="left", validate="m:1")
    CHECKS["vle_orphan_site_rows"] = int(v["activity_type"].isna().sum())

    # daily table (interim) – toàn kỳ, dùng cho profiling/descriptive; KHÔNG dùng làm feature sau ngày 28
    daily = (v.groupby(["code_module", "code_presentation", "id_student", "date"], observed=True)
               .agg(clicks=("sum_click", "sum"), n_sites=("id_site", "nunique")).reset_index())
    daily.to_parquet(INTERIM / "vle_student_day.parquet", index=False)
    log(9, "Interim: tổng click theo sinh viên × ngày (toàn kỳ)", n0, len(daily))

    # 8. features chỉ từ date <= 28  (cut-off cứng)
    w = v[v["date"] <= PREDICTION_DAY]
    CHECKS["feature_max_vle_date_used"] = int(w["date"].max())
    pre = w[w["date"] < 0].groupby(KEY).agg(clicks_pre_start=("sum_click", "sum"),
                                             active_days_pre_start=("date", "nunique"))
    obs = w[w["date"] >= 0].copy()
    obs_w = obs[obs["date"] <= WEEKS[-1][2]].copy()          # v1.2: tuần 1–4 đều 7 ngày (0–27)
    obs_w["week"] = (obs_w["date"] // 7 + 1).astype(int)
    wk = obs_w.groupby(KEY + ["week"]).agg(clicks=("sum_click", "sum"), active_days=("date", "nunique")).reset_index()
    tot = obs.groupby(KEY).agg(clicks_0_28=("sum_click", "sum"), active_days_0_28=("date", "nunique"),
                               n_sites_0_28=("id_site", "nunique"), last_active_day=("date", "max"))
    typ = (obs[obs["activity_type"].isin(ACTIVITY_TYPES)]
           .pivot_table(index=KEY, columns="activity_type", values="sum_click", aggfunc="sum", fill_value=0))
    typ.columns = [f"clicks_{c}_0_28" for c in typ.columns]
    wide = wk.pivot_table(index=KEY, columns="week", values=["clicks", "active_days"], fill_value=0)
    wide.columns = [f"{a}_w{b}" for a, b in wide.columns]
    has_any = v.groupby(KEY).size().rename("n_vle_rows_all")
    feat = pd.concat([wide, tot, typ, pre, has_any], axis=1).reset_index()

    # student-week view (AQ1)
    sw = reg[KEY].merge(pd.DataFrame({"week": [1, 2, 3, 4]}), how="cross")
    sw = sw.merge(wk, on=KEY + ["week"], how="left")
    sw[["clicks", "active_days"]] = sw[["clicks", "active_days"]].fillna(0).astype(int)
    return feat, sw


def assessment_features(T, reg):
    a, sa = T["assessments"], T["studentAssessment"]
    first = (a[a["assessment_type"] != "Exam"].sort_values("date")
             .groupby(["code_module", "code_presentation"]).head(1)
             [["code_module", "code_presentation", "id_assessment", "date", "weight"]]
             .rename(columns={"date": "tma1_deadline", "weight": "tma1_weight", "id_assessment": "tma1_id"}))
    CHECKS["tma1_deadline_max"] = int(first["tma1_deadline"].max())
    assert first["tma1_deadline"].max() <= PREDICTION_DAY
    s = sa.merge(first, left_on="id_assessment", right_on="tma1_id")
    CHECKS["tma1_banked_rows_excluded"] = int(s["is_banked"].sum())
    s = s[s["is_banked"] == 0]                         # bài bảo lưu từ lần học trước, không phản ánh kỳ này
    s = s[s["date_submitted"] <= PREDICTION_DAY]        # chỉ bài đã nộp trước mốc dự báo
    CHECKS["tma1_dup_student"] = int(s.duplicated(["code_module", "code_presentation", "id_student"]).sum())
    s = s[["code_module", "code_presentation", "id_student", "date_submitted", "score"]]
    out = reg[KEY].merge(first, on=["code_module", "code_presentation"], how="left")
    out = out.merge(s, on=KEY, how="left", validate="1:1")
    out["tma1_submitted_by_d28"] = out["date_submitted"].notna().astype(int)
    out["tma1_on_time"] = (out["date_submitted"] <= out["tma1_deadline"]).astype(int)
    out["tma1_days_before_deadline"] = out["tma1_deadline"] - out["date_submitted"]
    out = out.rename(columns={"score": "tma1_score"})
    log(8, "Đặc trưng bài đánh giá đầu tiên (TMA1, hạn ngày 12–24): nộp trước ngày 28, đúng hạn, điểm (có điều kiện)",
        note=f"loại {CHECKS['tma1_banked_rows_excluded']} bài bảo lưu")
    return out[KEY + ["tma1_deadline", "tma1_weight", "tma1_submitted_by_d28", "tma1_on_time",
                      "tma1_days_before_deadline", "tma1_score"]]


EDU_ORD = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
           "HE Qualification": 3, "Post Graduate Qualification": 4}


def main():
    t0 = time.time()
    for d in (INTERIM, PROCESSED, META):
        d.mkdir(parents=True, exist_ok=True)
    T = select_scope(load_raw())
    reg = build_registrations(T)
    reg.to_parquet(INTERIM / "registrations_scope.parquet", index=False)
    feat, sw = vle_features(T, reg)
    asm = assessment_features(T, reg)

    n0 = len(reg)
    df = reg.merge(feat, on=KEY, how="left", validate="1:1").merge(asm, on=KEY, how="left", validate="1:1")
    assert len(df) == n0, "join làm thay đổi số dòng"
    log(7, "Ghép đặc trưng VLE + đánh giá vào bảng đăng ký (left, 1:1)", n0, len(df))

    # missing VLE = không có hoạt động (0) – phân biệt bằng cờ (DQ-07)
    cnt = [c for c in df.columns if c.startswith(("clicks_", "active_days_", "n_sites_"))]
    df["no_vle_record_any"] = df["n_vle_rows_all"].isna().astype(int)
    df["no_vle_activity_0_28"] = df["clicks_0_28"].isna().astype(int)
    df[cnt] = df[cnt].fillna(0).astype(int)
    df["days_since_last_active_d28"] = (PREDICTION_DAY - df["last_active_day"]).fillna(PREDICTION_DAY + 1).astype(int)
    df = df.drop(columns=["last_active_day", "n_vle_rows_all"])
    log(5, "Không có bản ghi VLE trong 0–28 -> 0 click + cờ no_vle_activity_0_28; days_since_last_active = 29 nếu chưa từng hoạt động")

    # derived features
    df["clicks_change_w1_w4"] = np.log1p(df["clicks_w4"]) - np.log1p(df["clicks_w1"])
    df["log_clicks_0_28"] = np.log1p(df["clicks_0_28"])
    df["highest_education_ord"] = df["highest_education"].map(EDU_ORD)
    assert df["highest_education_ord"].notna().all()
    df["registration_lead_days"] = -df["date_registration"]
    log(8, "Đặc trưng dẫn xuất: clicks_change_w1_w4 = log1p(w4) − log1p(w1); log_clicks_0_28; highest_education_ord; registration_lead_days = −date_registration")

    # fit-on-train: impute registration_lead_days bằng median của TRAIN
    med = float(df.loc[df["split"] == "train", "registration_lead_days"].median())
    df["registration_lead_days"] = df["registration_lead_days"].fillna(med)
    fit_params = {"registration_lead_days_median_train": med}
    log(5, f"Impute registration_lead_days bằng median tập train = {med:g} (fit trên train, áp cho validation/test)")

    # seen-in-earlier-presentation (student leakage audit)
    order = {p: i for i, p in enumerate(PRESENTATIONS)}
    first_seen = df.assign(o=df["code_presentation"].map(order)).groupby("id_student")["o"].min()
    df["seen_in_earlier_presentation"] = (df["code_presentation"].map(order) > df["id_student"].map(first_seen)).astype(int)

    # v1.1 – cờ chất lượng theo bối cảnh (không xóa dòng nào)
    df["late_registration_1_28"] = ((df["date_registration"] > 0) & (df["date_registration"] <= PREDICTION_DAY)).astype(int)
    df["days_observable_0_28"] = (PREDICTION_DAY + 1 - df["date_registration"].clip(lower=0).fillna(0)).clip(1, PREDICTION_DAY + 1).astype(int)
    thr = float(df.loc[(df["split"] == "train") & (df["active_d28"] == 1), "clicks_0_28"].quantile(0.99))
    df["flag_high_activity"] = (df["clicks_0_28"] > thr).astype(int)
    fit_params["clicks_0_28_p99_train"] = thr
    log(6, f"Cờ chất lượng: late_registration_1_28 (cửa sổ quan sát bị rút ngắn), days_observable_0_28, flag_high_activity (> p99 tập train = {thr:g} click) – chỉ gắn cờ, không xóa/cắt",
        note=f"đăng ký muộn {int(df['late_registration_1_28'].sum())}, hoạt động cao {int(df['flag_high_activity'].sum())}")

    # ------------------------------------------------ 9. analytic-ready views
    df.to_parquet(INTERIM / "registration_features_all.parquet", index=False)
    pop = df[df["active_d28"] == 1].copy()
    log(9, "Population AQ1–AQ5: còn đăng ký ở cuối ngày 28 (C04, C06)", len(df), len(pop))

    ID = KEY + ["split"]
    BEHAV = ([f"clicks_w{i}" for i in range(1, 5)] + [f"active_days_w{i}" for i in range(1, 5)] +
             ["clicks_0_28", "log_clicks_0_28", "active_days_0_28", "n_sites_0_28", "clicks_change_w1_w4",
              "days_since_last_active_d28", "clicks_pre_start", "active_days_pre_start", "no_vle_activity_0_28"] +
             [f"clicks_{t}_0_28" for t in ACTIVITY_TYPES])
    ACAD = ["tma1_submitted_by_d28", "tma1_on_time", "tma1_days_before_deadline"]
    BACK = ["num_of_prev_attempts", "studied_credits", "highest_education_ord", "registration_lead_days"]
    # v1.2: date_registration_missing không làm predictor (2 ca trong population, đều Withdrawn → kênh rò rỉ nhãn)
    COND = ["tma1_score"]
    FLAGS = ["seen_in_earlier_presentation", "late_registration_1_28", "days_observable_0_28", "flag_high_activity"]
    QA = ["flag_withdrawn_no_unreg_date", "flag_unreg_date_not_withdrawn", "date_registration_missing"]  # v1.2: chỉ audit, tách khỏi bảng mô hình
    TARGET = ["target_noncompletion"]

    # AQ3/AQ4/AQ5 model table – KHÔNG chứa biến nhạy cảm, final_result, date_unregistration
    model = pop[ID + BEHAV + ACAD + BACK + COND + FLAGS + TARGET]
    model.to_csv(PROCESSED / "aq3_model_table.csv", index=False)
    audit = pop[KEY + SENSITIVE + ["imd_band_missing", "highest_education"] + QA]
    audit.to_csv(PROCESSED / "aq3_audit_attributes.csv", index=False)
    # AQ2 diagnostic table – có final_result/label để so sánh nhóm, có cả biến nền tảng
    diag = pop[ID + ["final_result", "label3"] + TARGET + BEHAV + ACAD + COND + BACK + SENSITIVE + ["highest_education"]]
    diag.to_csv(PROCESSED / "aq2_diagnostic_table.csv", index=False)
    # AQ1 student-week view
    sw = sw.merge(pop[KEY + ["label3", "final_result", "split"]], on=KEY, how="inner")
    sw.to_csv(PROCESSED / "aq1_student_week.csv", index=False)
    # context table: toàn bộ lượt đăng ký (bối cảnh 52,8% + rút sớm)
    ctx = df[KEY + ["final_result", "label3", "target_noncompletion", "active_d28", "unregistered_by_d28",
                    "registered_after_d28", "date_registration", "date_unregistration", "split"]]
    ctx.to_csv(PROCESSED / "context_all_registrations.csv", index=False)
    json.dump({"capacity_k_per_module_presentation": CAPACITY_K, "default_k": 200,
               "assumed_conversion_rate": 0.14, "source": "A5 (P1), C05 (P3)"},
              open(PROCESSED / "aq4_decision_params.json", "w"), indent=2)
    log(9, "Xuất 4 view: aq1_student_week, aq2_diagnostic_table, aq3_model_table (+ audit), context_all_registrations")

    # ------------------------------------------------ validation
    validate(df, pop, model, sw, BEHAV + ACAD + BACK + COND)
    CHECKS["fit_params"] = fit_params
    reduced = redundancy(pop, BEHAV, ACAD, BACK)
    feature_sets = {"behavioural": BEHAV, "academic_d28": ACAD, "background": BACK,
                    "conditional_score": COND, "audit_only_sensitive": SENSITIVE, "flags": FLAGS, "target": TARGET,
                    "reduced_core": reduced}
    write_metadata(feature_sets)
    print(f"done in {time.time()-t0:.1f}s")


def _vif(X):
    Z = (X - X.mean()) / X.std()
    inv = np.linalg.pinv(np.corrcoef(Z.values.T))
    return pd.Series(np.diag(inv), index=X.columns)


def redundancy(pop, BEHAV, ACAD, BACK):
    """Phân tích trùng lặp thông tin giữa feature (fit trên TRAIN) và đề xuất bộ feature rút gọn."""
    tr = pop[pop["split"] == "train"]
    feats = [f for f in BEHAV if f != "clicks_0_28"] + ACAD + BACK
    X = tr[feats].copy()
    cnt = [f for f in feats if f.startswith(("clicks_", "active_days_pre")) and f != "clicks_change_w1_w4"]
    for c in cnt:
        X[c] = np.log1p(X[c])
    X = X.fillna(X.median())
    C = X.corr()
    pairs = sorted([(a, b, round(float(C.loc[a, b]), 3)) for i, a in enumerate(feats) for b in feats[i + 1:]
                    if abs(C.loc[a, b]) > 0.8], key=lambda t: -abs(t[2]))
    exact = round(float((tr[[f"active_days_w{i}" for i in range(1, 5)]].sum(axis=1) == tr["active_days_0_28"]).mean()), 4)  # tỷ lệ dòng bằng đúng tổng 4 tuần
    vif_before = _vif(X).round(1)
    # Quy tắc: (1) bỏ biến là tổng chính xác của biến khác; (2) loại dần biến hành vi có VIF lớn nhất tới khi VIF ≤ 5;
    # biến học thuật/nền tảng luôn giữ (ít trùng, có ý nghĩa nghiệp vụ riêng).
    keep = [f for f in feats if f != "active_days_0_28"]
    dropped = [("active_days_0_28", "gần như tổng của active_days_w1..w4 (chỉ khác ngày 28)")]
    while True:
        v = _vif(X[keep])
        cand = v[[f for f in keep if f in BEHAV]]
        if cand.max() <= 5:
            break
        f = cand.idxmax(); keep.remove(f); dropped.append((f, f"VIF = {cand.max():.1f}"))
    CHECKS["feature_redundancy"] = {"pairs_abs_r_gt_0_8": pairs, "share_rows_active_days_total_equals_weeks": exact,
                                    "vif_before_top": vif_before.sort_values(ascending=False).head(10).to_dict(),
                                    "dropped_in_reduced": dropped, "vif_after_max": round(float(_vif(X[keep]).max()), 2),
                                    "reduced_core": keep}
    log(8, f"Phân tích trùng lặp feature (train): {len(pairs)} cặp |r| > 0,8; active_days_0_28 ≈ tổng 4 tuần ({exact:.1%} dòng bằng đúng); bộ rút gọn {len(keep)} feature, VIF lớn nhất {CHECKS['feature_redundancy']['vif_after_max']}")
    return keep


def validate(df, pop, model, sw, feats):
    C = CHECKS
    C["rows"] = {"registrations_scope": len(df), "population_active_d28": len(pop), "student_week": len(sw)}
    C["key_unique_model_table"] = bool(not model.duplicated(KEY).any())
    C["student_week_rows_equal_4x"] = bool(len(sw) == 4 * len(pop))
    C["missing_in_features"] = {c: int(model[c].isna().sum()) for c in feats if model[c].isna().any()}
    C["target_rate"] = {
        "all_registrations": round(float(df["target_noncompletion"].mean()), 4),
        "active_d28": round(float(pop["target_noncompletion"].mean()), 4),
        "by_split_active_d28": pop.groupby("split")["target_noncompletion"].agg(["size", "mean"]).round(4).to_dict("index"),
        "by_module_presentation_active_d28": {f"{m}-{p}": {"n": int(len(g)), "rate": round(float(g["target_noncompletion"].mean()), 4)}
                                              for (m, p), g in pop.groupby(["code_module", "code_presentation"])},
    }
    C["final_result_active_d28"] = pop["final_result"].value_counts().to_dict()
    C["leakage_columns_absent"] = {c: c not in model.columns for c in LABEL_DERIVED + SENSITIVE + ["date_registration_missing"]}
    assert all(C["leakage_columns_absent"].values())
    assert C["feature_max_vle_date_used"] <= PREDICTION_DAY
    # đối chiếu thủ công 3 sinh viên: tính lại từ raw
    raw = pd.read_csv(RAW / "studentVle.csv", usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"])
    sample = pd.concat([pop.sample(8, random_state=7), pop[pop["no_vle_activity_0_28"] == 1].sample(1, random_state=7),
                        pop[pop["late_registration_1_28"] == 1].sample(1, random_state=7)])
    manual = []
    for _, r in sample.iterrows():
        m = raw[(raw.code_module == r.code_module) & (raw.code_presentation == r.code_presentation) & (raw.id_student == r.id_student)]
        wk = [int(m[(m.date >= a) & (m.date <= b)].sum_click.sum()) for _, a, b in WEEKS]
        days = int(m[(m.date >= 0) & (m.date <= 28)].date.nunique())
        tot = int(m[(m.date >= 0) & (m.date <= 28)].sum_click.sum())
        ok = wk == [int(r[f"clicks_w{i}"]) for i in range(1, 5)] and days == int(r.active_days_0_28) and tot == int(r.clicks_0_28)
        manual.append({"key": f"{r.code_module}-{r.code_presentation}-{r.id_student}",
                       "clicks_w1_w4_pipeline": [int(r[f"clicks_w{i}"]) for i in range(1, 5)], "clicks_w1_w4_manual": wk,
                       "active_days_pipeline": int(r.active_days_0_28), "active_days_manual": days,
                       "clicks_0_28_pipeline": int(r.clicks_0_28), "clicks_0_28_manual": tot, "match": ok})
        assert ok
    C["manual_check"] = manual
    # v1.1 – kiểm tra ngữ nghĩa sau xử lý
    C["unknown_share_equals_missing"] = {"imd_unknown_after": int((df["imd_band"] == "Unknown").sum()),
                                         "imd_missing_flag": int(df["imd_band_missing"].sum())}
    assert C["unknown_share_equals_missing"]["imd_unknown_after"] == C["unknown_share_equals_missing"]["imd_missing_flag"]
    # v1.2: kiểm tra ngữ nghĩa CHỈ trên tập train (không nhìn outcome của validation/test khi ra quyết định xử lý)
    trp = pop[pop["split"] == "train"]
    C["semantic_checks_scope"] = f"train (2013B + 2013J), n = {len(trp)}"
    C["zero_activity_semantics"] = {"noncompletion_rate_no_vle_0_28": round(float(trp.loc[trp.no_vle_activity_0_28 == 1, "target_noncompletion"].mean()), 4),
                                    "noncompletion_rate_with_vle": round(float(trp.loc[trp.no_vle_activity_0_28 == 0, "target_noncompletion"].mean()), 4),
                                    "n_no_vle": int(trp.no_vle_activity_0_28.sum())}
    lr_ = trp[trp.late_registration_1_28 == 1]; on = trp[trp.late_registration_1_28 == 0]
    rate = float(lr_["target_noncompletion"].mean()); se = (rate * (1 - rate) / max(len(lr_), 1)) ** 0.5
    C["late_registration"] = {"n": int(len(lr_)), "n_population_all_splits": int(pop.late_registration_1_28.sum()),
                              "noncompletion_rate": round(rate, 4), "ci95": [round(rate - 1.96 * se, 3), round(rate + 1.96 * se, 3)],
                              "noncompletion_rate_ontime": round(float(on["target_noncompletion"].mean()), 4),
                              "median_clicks_late": float(lr_["clicks_0_28"].median()), "median_clicks_ontime": float(on["clicks_0_28"].median()),
                              "median_clicks_per_observable_day_late": round(float((lr_["clicks_0_28"] / lr_["days_observable_0_28"]).median()), 2),
                              "median_clicks_per_observable_day_ontime": round(float((on["clicks_0_28"] / on["days_observable_0_28"]).median()), 2)}
    C["high_activity"] = {"n": int(trp.flag_high_activity.sum()), "n_population_all_splits": int(pop.flag_high_activity.sum()),
                          "max_daily_clicks_in_group": int(pd.read_parquet(INTERIM / "vle_student_day.parquet").merge(
                              trp.loc[trp.flag_high_activity == 1, KEY], on=KEY).query("date >= 0 and date <= 28").clicks.max())}
    # v1.2: thống kê trước/sau biến đổi cho feature chính (train)
    ba = {}
    for c in ["clicks_0_28", "clicks_w1", "clicks_w4", "clicks_pre_start"]:
        x = trp[c]; lx = np.log1p(x)
        ba[c] = {"raw_median": float(x.median()), "raw_iqr": float(x.quantile(.75) - x.quantile(.25)), "raw_skew": round(float(x.skew()), 2),
                 "log1p_median": round(float(lx.median()), 2), "log1p_skew": round(float(lx.skew()), 2)}
    C["before_after_stats_train"] = ba
    reg = df[KEY + ["date_unregistration"]]
    daily = pd.read_parquet(INTERIM / "vle_student_day.parquet").merge(reg, on=KEY)
    after = daily[daily.date > daily.date_unregistration]
    C["activity_after_unregistration"] = {"student_days": int(len(after)), "registrations": int(after[KEY].drop_duplicates().shape[0]),
                                          "within_feature_window_for_population": int(((after.date <= PREDICTION_DAY) &
                                              (after.date_unregistration > PREDICTION_DAY)).sum())}
    assert C["activity_after_unregistration"]["within_feature_window_for_population"] == 0
    C["missing_before_after"] = {
        "imd_band": {"before": int(df["imd_band_missing"].sum()), "after": int((df["imd_band"].isna()).sum()), "how": "nhãn Unknown + cờ"},
        "registration_lead_days": {"before": int(df["date_registration_missing"].sum()), "in_population": int(pop["date_registration_missing"].sum()), "after": int(df["registration_lead_days"].isna().sum()), "how": "median train (kỹ thuật); cờ chỉ audit, không làm predictor"},
        "vle_counts_0_28": {"before": int(df["no_vle_activity_0_28"].sum()), "after": 0, "how": "0 + cờ no_vle_activity_0_28"},
        "tma1_score": {"before": int(model["tma1_score"].isna().sum()), "after": int(model["tma1_score"].isna().sum()), "how": "giữ NaN (chưa nộp / nộp nhưng chưa có điểm) – không impute"},
        "tma1_days_before_deadline": {"before": int(model["tma1_days_before_deadline"].isna().sum()), "after": int(model["tma1_days_before_deadline"].isna().sum()), "how": "giữ NaN = không áp dụng (chưa nộp); impute trong pipeline M4 + cờ tma1_submitted_by_d28"}}
    # độ nhạy: nếu (sai) xóa dòng trùng thì clicks_0_28 đổi bao nhiêu
    dd = raw[raw.code_module.isin(MODULES)]
    w = dd[(dd.date >= 0) & (dd.date <= 28)]
    keep = w.groupby(KEY).sum_click.sum()
    dedup = w.drop_duplicates().groupby(KEY).sum_click.sum()
    C["sensitivity_dedup"] = {"mean_clicks_keep": round(float(keep.mean()), 1), "mean_clicks_dedup": round(float(dedup.mean()), 1),
                              "spearman": round(float(keep.rank().corr(dedup.reindex(keep.index).rank())), 4)}


def write_metadata(fs):
    model = pd.read_csv(PROCESSED / "aq3_model_table.csv", nrows=5)
    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size,
                      "rows": int(sum(1 for _ in open(p)) - 1) if p.suffix == ".csv" else None}
             for p in sorted(PROCESSED.glob("*")) if p.is_file()}
    manifest = {"prepared_version": PREPARED_VERSION, "input_raw": DATASET_VERSION,
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "pipeline": "src/prepare_data.py", "pipeline_sha256": sha256(Path(__file__)), "config_sha256": sha256(Path(__file__).with_name("config.py")),
                "prediction_day": PREDICTION_DAY, "split": SPLIT,
                "feature_sets": fs, "files": files, "model_table_columns": list(model.columns)}
    json.dump(manifest, open(META / "prepared_manifest.json", "w"), indent=2, ensure_ascii=False)
    json.dump(LOG, open(META / "transformation_log.json", "w"), indent=2, ensure_ascii=False)
    json.dump(CHECKS, open(META / "validation_summary.json", "w"), indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
