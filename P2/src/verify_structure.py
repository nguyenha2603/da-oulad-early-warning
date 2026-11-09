"""Milestone P2 - Initial Structural Verification of OULAD raw files.
Read-only: không sửa, không làm sạch, không impute raw data.
Chạy: python3 src/verify_structure.py  (từ thư mục P2/)  -> data/metadata/structural_verification.json
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "oulad"
OUT = ROOT / "data" / "metadata" / "structural_verification.json"

TABLES = {
    "courses": ["code_module", "code_presentation"],
    "assessments": ["id_assessment"],
    "vle": ["id_site"],
    "studentInfo": ["code_module", "code_presentation", "id_student"],
    "studentRegistration": ["code_module", "code_presentation", "id_student"],
    "studentAssessment": ["id_assessment", "id_student"],
    "studentVle": None,  # log table: không có khóa duy nhất theo tài liệu
}
# Số liệu tài liệu (OULAD.names, mục 5) để đối chiếu
DOC_COUNTS = {"studentInfo": 32953, "courses": 22, "vle": 6364, "studentVle": 10655280,
              "studentRegistration": 32953, "assessments": 206, "studentAssessment": 173912}

res = {"tables": {}}
dfs = {}
for name, key in TABLES.items():
    # đọc mọi cột dạng chuỗi trước để đếm đúng ký hiệu thiếu ('?' hoặc rỗng) như trong file gốc
    raw = pd.read_csv(RAW / f"{name}.csv", dtype=str, keep_default_na=False)
    df = pd.read_csv(RAW / f"{name}.csv", na_values=["?"])
    dfs[name] = df
    t = {"rows": len(df), "cols": df.shape[1], "columns": {},
         "doc_rows": DOC_COUNTS.get(name)}
    for c in df.columns:
        t["columns"][c] = {"dtype": str(df[c].dtype),
                           "n_empty": int((raw[c] == "").sum()),
                           "n_qmark": int((raw[c] == "?").sum()),
                           "n_unique": int(df[c].nunique())}
    if key:
        t["key"] = key
        t["dup_key_rows"] = int(df.duplicated(subset=key).sum())
    t["dup_full_rows"] = int(df.duplicated().sum()) if name != "studentVle" else int(df.duplicated().sum())
    res["tables"][name] = t
    print(name, df.shape, file=sys.stderr)

si, sr, sa, a, sv, c, v = (dfs[k] for k in ["studentInfo", "studentRegistration", "studentAssessment",
                                             "assessments", "studentVle", "courses", "vle"])
# Categorical value sets (kiểm tra cấu trúc, không chuẩn hóa)
res["value_sets"] = {col: sorted(si[col].dropna().astype(str).unique().tolist())
                     for col in ["final_result", "imd_band", "age_band", "highest_education", "gender", "disability"]}
res["value_sets"]["activity_type"] = sorted(v["activity_type"].unique().tolist())
res["value_sets"]["assessment_type"] = sorted(a["assessment_type"].unique().tolist())
res["value_sets"]["code_presentation"] = sorted(c["code_presentation"].unique().tolist())
res["final_result_dist"] = si["final_result"].value_counts().to_dict()

# Referential integrity giữa các bảng (structural)
k3 = ["code_module", "code_presentation", "id_student"]
res["integrity"] = {
    "studentInfo_vs_registration_same_keys": int(len(si.merge(sr, on=k3))),
    "studentVle_keys_not_in_studentInfo": int(len(sv[k3].drop_duplicates().merge(si[k3], on=k3, how="left", indicator=True).query("_merge=='left_only'"))),
    "studentVle_site_not_in_vle": int((~sv["id_site"].isin(v["id_site"])).sum()),
    "studentAssessment_assess_not_in_assessments": int((~sa["id_assessment"].isin(a["id_assessment"])).sum()),
    "students_in_info_without_any_vle": int(len(si[k3].merge(sv[k3].drop_duplicates(), on=k3, how="left", indicator=True).query("_merge=='left_only'"))),
    "unique_id_student": int(si["id_student"].nunique()),
}
# Đối chiếu tài liệu: 'unregistered => Withdrawn'
m = si.merge(sr, on=k3)
res["doc_check_unreg"] = {
    "has_unreg_date": int(m["date_unregistration"].notna().sum()),
    "has_unreg_but_not_withdrawn": int((m["date_unregistration"].notna() & (m["final_result"] != "Withdrawn")).sum()),
    "withdrawn_without_unreg_date": int((m["date_unregistration"].isna() & (m["final_result"] == "Withdrawn")).sum()),
    "unreg_before_day0": int((m["date_unregistration"] <= 0).sum()),
    "unreg_day1_28": int(((m["date_unregistration"] > 0) & (m["date_unregistration"] <= 28)).sum()),
}
# Time fields
res["time_ranges"] = {
    "studentVle.date": [int(sv["date"].min()), int(sv["date"].max())],
    "studentAssessment.date_submitted": [int(sa["date_submitted"].min()), int(sa["date_submitted"].max())],
    "assessments.date": [float(a["date"].min()), float(a["date"].max())],
    "studentRegistration.date_registration": [float(sr["date_registration"].min()), float(sr["date_registration"].max())],
    "courses.length": [int(c["module_presentation_length"].min()), int(c["module_presentation_length"].max())],
}
res["score_range"] = [float(sa["score"].min()), float(sa["score"].max())]

# Tổng quan theo module (phục vụ đề xuất phạm vi module; chỉ đếm/tỷ lệ, không biến đổi dữ liệu)
mods = []
early_vle = sv[(sv["date"] >= 0) & (sv["date"] <= 28)]
ev = early_vle.groupby(k3)["sum_click"].sum().rename("clicks_0_28").reset_index()
for mod, g in si.groupby("code_module"):
    pres = sorted(g["code_presentation"].unique())
    nc = g["final_result"].isin(["Fail", "Withdrawn"]).mean()
    aa = a[a["code_module"] == mod]
    first = aa[aa["assessment_type"] != "Exam"].groupby("code_presentation")["date"].min()
    gm = g.merge(sr, on=k3)
    gv = g[k3].merge(ev, on=k3, how="left")
    vm = v[v["code_module"] == mod]
    mods.append({
        "module": mod, "n_presentations": len(pres), "presentations": pres,
        "n_enrolments": int(len(g)),
        "nonCompletion_rate": round(float(nc), 4),
        "withdrawn_rate": round(float((g["final_result"] == "Withdrawn").mean()), 4),
        "fail_rate": round(float((g["final_result"] == "Fail").mean()), 4),
        "first_nonexam_assessment_day_by_pres": {k: (None if pd.isna(x) else float(x)) for k, x in first.items()},
        "first_assessment_types": sorted(aa[aa["assessment_type"] != "Exam"].sort_values("date").groupby("code_presentation").head(1)["assessment_type"].unique().tolist()),
        "unreg_le_day0_share": round(float((gm["date_unregistration"] <= 0).mean()), 4),
        "unreg_day1_28_share": round(float(((gm["date_unregistration"] > 0) & (gm["date_unregistration"] <= 28)).mean()), 4),
        "share_with_vle_0_28": round(float(gv["clicks_0_28"].notna().mean()), 4),
        "median_clicks_0_28": float(gv["clicks_0_28"].median()),
        "imd_missing_share": round(float(g["imd_band"].isna().mean()), 4),
        "vle_week_from_missing_share": round(float(vm["week_from"].isna().mean()), 4),
        "length_days": c[c["code_module"] == mod].set_index("code_presentation")["module_presentation_length"].to_dict(),
    })
res["modules"] = mods

# Quần thể tại mốc quyết định (ngày 28) cho 2 module đề xuất: SV chưa hủy đăng ký trước/đúng ngày 28
mm = si.merge(sr, on=k3)
mm = mm[mm["code_module"].isin(["BBB", "FFF"])].copy()
mm["active_d28"] = ~(mm["date_unregistration"] <= 28)
mm["nc"] = mm["final_result"].isin(["Fail", "Withdrawn"])
res["proposed_scope"] = []
for (mod, pres), d in mm.groupby(["code_module", "code_presentation"]):
    act = d[d["active_d28"]]
    res["proposed_scope"].append({"module": mod, "presentation": pres, "n_enrolments": int(len(d)),
        "nonCompletion_all": round(float(d["nc"].mean()), 4), "active_at_day28": int(len(act)),
        "nonCompletion_active_d28": round(float(act["nc"].mean()), 4)})
act = mm[mm["active_d28"]]
res["proposed_scope_total"] = {"n_enrolments": int(len(mm)), "active_at_day28": int(len(act)),
    "nonCompletion_all": round(float(mm["nc"].mean()), 4), "nonCompletion_active_d28": round(float(act["nc"].mean()), 4)}
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=int))
print("written", OUT.relative_to(OUT.parents[2]), file=sys.stderr)
