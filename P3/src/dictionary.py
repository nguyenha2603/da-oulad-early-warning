"""Data dictionary + Data Lineage cho prepared dataset (Task 8). Chạy: python3 src/dictionary.py
Ghi data/metadata/data_dictionary.csv và data/metadata/data_lineage.csv (khớp cột thực tế của các view)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED, META, ACTIVITY_TYPES  # noqa: E402

W = "ngày 0–28"
# name: (source file/field, transformation, window/filter, purpose/AQ, dtype, role, sensitivity)
D = {
    "code_module": ("studentInfo.code_module", "giữ nguyên; lọc BBB, FFF", "C03", "Khóa; nhóm so sánh AQ1–AQ4", "str", "key", "Thấp"),
    "code_presentation": ("studentInfo.code_presentation", "giữ nguyên; lọc 4 kỳ", "C03", "Khóa; căn cứ split", "str", "key", "Thấp"),
    "id_student": ("studentInfo.id_student", "giữ nguyên (đã ẩn danh)", "—", "Khóa; không dùng làm predictor", "int", "key", "Trung bình (định danh giả)"),
    "split": ("studentInfo.code_presentation", "2013B, 2013J → train; 2014B → validation; 2014J → test", "—", "Chống temporal leakage (AQ3–AQ5)", "str", "split", "Thấp"),
    "target_noncompletion": ("studentInfo.final_result", "1 nếu ∈ {Fail, Withdrawn}, 0 nếu ∈ {Pass, Distinction}", "quan sát cuối module", "Target AQ3; outcome AQ1–AQ2; KPI", "int", "target", "Cao (kết quả học tập)"),
    "final_result": ("studentInfo.final_result", "giữ nguyên", "cuối module", "So sánh nhóm AQ1–AQ2 (chỉ ở view diagnostic)", "str", "outcome", "Cao"),
    "label3": ("studentInfo.final_result", "Distinction gộp vào Pass → Pass/Fail/Withdrawn", "cuối module", "Ba nhóm của AQ1", "str", "outcome", "Cao"),
    **{f"clicks_w{i}": ("studentVle.sum_click, date", f"tổng sum_click (giữ dòng trùng) với date trong tuần {i} (ngày {7*(i-1)}–{7*i-1})", f"tuần {i}", "AQ1 xu hướng; AQ2/AQ3 feature", "int", "feature", "Thấp") for i in range(1, 5)},
    **{f"active_days_w{i}": ("studentVle.date", f"số ngày phân biệt có ≥ 1 click trong tuần {i}", f"tuần {i}", "AQ1; AQ2/AQ3 feature", "int", "feature", "Thấp") for i in range(1, 5)},
    "clicks_0_28": ("studentVle.sum_click", "tổng sum_click, 0 nếu không có bản ghi", W, "AQ2/AQ3 feature", "int", "feature", "Thấp"),
    "log_clicks_0_28": ("studentVle.sum_click", "log(1 + clicks_0_28) – biến đổi không cần fit", W, "Giảm lệch phải (skew 3,1)", "float", "feature", "Thấp"),
    "active_days_0_28": ("studentVle.date", "số ngày phân biệt có hoạt động", W, "AQ2/AQ3 feature", "int", "feature", "Thấp"),
    "n_sites_0_28": ("studentVle.id_site", "số tài nguyên VLE phân biệt đã truy cập", W, "Độ rộng tương tác (AQ2/AQ3)", "int", "feature", "Thấp"),
    "clicks_change_w1_w4": ("clicks_w1, clicks_w4", "log1p(clicks_w4) − log1p(clicks_w1)", "tuần 1 và 4", "Mức thay đổi tương tác (AQ2, AQ4 quy tắc b)", "float", "feature", "Thấp"),
    "days_since_last_active_d28": ("studentVle.date", "28 − ngày hoạt động gần nhất; 29 nếu không có hoạt động 0–28", W, "Tín hiệu ngắt quãng (AQ3, AQ5)", "int", "feature", "Thấp"),
    "clicks_pre_start": ("studentVle.sum_click", "tổng click với date < 0", "trước khai giảng", "Mức chuẩn bị (AQ3)", "int", "feature", "Thấp"),
    "active_days_pre_start": ("studentVle.date", "số ngày có hoạt động với date < 0", "trước khai giảng", "AQ3", "int", "feature", "Thấp"),
    "no_vle_activity_0_28": ("studentVle", "1 nếu không có bản ghi nào trong 0–28 (P2: DQ-07)", W, "Phân biệt 'không có dữ liệu' với giá trị nhỏ", "int", "feature/flag", "Thấp"),
    **{f"clicks_{t}_0_28": ("studentVle.sum_click + vle.activity_type", f"tổng click vào tài nguyên loại {t}", W, "Cấu trúc tương tác (AQ2, AQ5)", "int", "feature", "Thấp") for t in ACTIVITY_TYPES},
    "tma1_submitted_by_d28": ("studentAssessment.date_submitted + assessments.date", "1 nếu nộp TMA đầu tiên (không bảo lưu) trước/đúng ngày 28", "hạn TMA1 ngày 12–24", "AQ2/AQ3 feature", "int", "feature", "Thấp"),
    "tma1_on_time": ("studentAssessment.date_submitted, assessments.date", "1 nếu date_submitted ≤ hạn nộp", "≤ ngày 28", "AQ2/AQ3 feature", "int", "feature", "Thấp"),
    "tma1_days_before_deadline": ("studentAssessment.date_submitted, assessments.date", "hạn − ngày nộp; NaN nếu chưa nộp (không áp dụng)", "≤ ngày 28", "AQ2/AQ3 feature", "float", "feature", "Thấp"),
    "tma1_score": ("studentAssessment.score", "giữ nguyên; NaN nếu chưa nộp", "nộp ≤ ngày 28", "AQ2; AQ3 chỉ ở bộ feature có điều kiện (thời điểm có điểm chưa xác minh)", "float", "conditional feature", "Trung bình"),
    "num_of_prev_attempts": ("studentInfo.num_of_prev_attempts", "giữ nguyên", "biết lúc đăng ký", "AQ2/AQ3 feature", "int", "feature", "Thấp"),
    "studied_credits": ("studentInfo.studied_credits", "giữ nguyên (không cắt outlier)", "biết lúc đăng ký", "Khối lượng học song song (AQ2/AQ3)", "int", "feature", "Thấp"),
    "highest_education_ord": ("studentInfo.highest_education", "mã thứ bậc 0–4 theo bậc học", "biết lúc đăng ký", "AQ2/AQ3 feature", "int", "feature", "Trung bình"),
    "highest_education": ("studentInfo.highest_education", "giữ nguyên", "biết lúc đăng ký", "AQ2; audit", "str", "background", "Trung bình"),
    "registration_lead_days": ("studentRegistration.date_registration", "−date_registration; NaN → median tập train (55)", "biết lúc đăng ký", "Đăng ký sớm/muộn (AQ3)", "float", "feature", "Thấp"),
    "date_registration_missing": ("studentRegistration.date_registration", "1 nếu thiếu", "—", "Cờ imputation – chỉ audit (v1.2: không làm predictor)", "int", "audit-only", "Thấp"),
    "seen_in_earlier_presentation": ("studentInfo.id_student", "1 nếu SV đã xuất hiện ở kỳ trước trong phạm vi", "—", "Audit student leakage (P4 sensitivity)", "int", "flag", "Thấp"),
    "flag_withdrawn_no_unreg_date": ("studentInfo.final_result, studentRegistration.date_unregistration", "1 nếu Withdrawn nhưng không có ngày hủy (P2: DQ-04)", "—", "Audit chất lượng nhãn", "int", "flag", "Thấp"),
    "flag_unreg_date_not_withdrawn": ("như trên", "1 nếu có ngày hủy nhưng nhãn khác Withdrawn (P2: DQ-04)", "—", "Audit chất lượng nhãn", "int", "flag", "Thấp"),
    "late_registration_1_28": ("studentRegistration.date_registration", "1 nếu đăng ký ở ngày 1–28 (cửa sổ quan sát bị rút ngắn)", "ngày 0–28", "Diễn giải đúng click thấp do mất ngày (v1.1)", "int", "flag", "Thấp"),
    "days_observable_0_28": ("studentRegistration.date_registration", "29 − max(date_registration, 0), giới hạn 1–29", "ngày 0–28", "Số ngày quan sát được; dùng chuẩn hóa nếu cần (v1.1)", "int", "flag", "Thấp"),
    "flag_high_activity": ("studentVle.sum_click", "1 nếu clicks_0_28 > p99 của tập train (1.911 click)", "ngày 0–28", "Đánh dấu outlier thay vì xóa (v1.1); ngưỡng fit trên train", "int", "flag", "Thấp"),
    "gender": ("studentInfo.gender", "giữ nguyên", "—", "Chỉ audit công bằng", "str", "audit-only", "Cao"),
    "age_band": ("studentInfo.age_band", "giữ nguyên", "—", "Chỉ audit công bằng", "str", "audit-only", "Cao"),
    "disability": ("studentInfo.disability", "giữ nguyên", "—", "Chỉ audit công bằng", "str", "audit-only", "Cao"),
    "imd_band": ("studentInfo.imd_band", "chuẩn hóa '10-20' → '10-20%'; thiếu → 'Unknown'", "—", "Chỉ audit công bằng (P1 mục 6)", "str", "audit-only", "Cao"),
    "imd_band_missing": ("studentInfo.imd_band", "1 nếu thiếu", "—", "Audit missingness", "int", "audit-only", "Cao"),
    "region": ("studentInfo.region", "giữ nguyên", "—", "Chỉ audit", "str", "audit-only", "Cao"),
    "week": ("studentVle.date", "date // 7 + 1 với 0 ≤ date ≤ 27 (mỗi tuần 7 ngày)", "tuần 1–4", "Trục thời gian AQ1", "int", "dimension", "Thấp"),
    "clicks": ("studentVle.sum_click", "tổng click của SV trong tuần", "tuần 1–4", "Measure AQ1", "int", "measure", "Thấp"),
    "active_days": ("studentVle.date", "số ngày có hoạt động trong tuần", "tuần 1–4", "Measure AQ1", "int", "measure", "Thấp"),
    "active_d28": ("studentRegistration.date_registration, date_unregistration", "1 nếu đăng ký ≤ 28 và chưa hủy ở ngày ≤ 28", "ngày 28", "Population AQ1–AQ5 (C04, C06); mẫu số KPI (C05)", "int", "filter", "Thấp"),
    "unregistered_by_d28": ("studentRegistration.date_unregistration", "1 nếu date_unregistration ≤ 28", "ngày 28", "Bối cảnh rút sớm", "int", "filter", "Trung bình"),
    "registered_after_d28": ("studentRegistration.date_registration", "1 nếu date_registration > 28", "ngày 28", "Loại khỏi population", "int", "filter", "Thấp"),
    "date_registration": ("studentRegistration.date_registration", "giữ nguyên", "—", "Bối cảnh", "float", "context", "Thấp"),
    "date_unregistration": ("studentRegistration.date_unregistration", "giữ nguyên – CHỈ ở view bối cảnh, cấm làm feature", "—", "Lọc population; bối cảnh", "float", "context (leakage)", "Trung bình"),
}


def main():
    rows = []
    for f in sorted(PROCESSED.glob("*.csv")):
        cols = pd.read_csv(f, nrows=1).columns
        for c in cols:
            s = D[c]  # KeyError nếu có cột chưa được mô tả -> buộc dictionary đầy đủ
            rows.append({"view": f.name, "variable": c, "source_field": s[0], "transformation": s[1],
                         "window_filter": s[2], "purpose_aq": s[3], "dtype": s[4], "role": s[5], "sensitivity": s[6]})
    dd = pd.DataFrame(rows)
    dd.to_csv(META / "data_dictionary.csv", index=False)
    lin = dd.drop_duplicates("variable")[["variable", "source_field", "transformation", "window_filter", "purpose_aq", "sensitivity"]]
    lin.columns = ["Prepared Variable", "Source File/Field", "Transformation", "Window/Filter", "Purpose/AQ", "Sensitivity"]
    lin.to_csv(META / "data_lineage.csv", index=False)
    print(len(dd), "dictionary rows;", len(lin), "lineage rows")


if __name__ == "__main__":
    main()
