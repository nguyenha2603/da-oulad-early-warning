"""Column-level Decision Log (v1.1): quyết định cho TỪNG cột của 7 bảng OULAD, kèm lý do theo bối cảnh business.
Nguyên tắc: không xóa cột/dòng nào khỏi raw; "không dùng" ≠ "xóa". Chạy sau prepare_data.py và profile_data.py:
python3 src/column_decisions.py -> data/metadata/column_decision_log.csv"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import META  # noqa: E402

S = json.loads((META / "profiling_summary.json").read_text())
V = json.loads((META / "validation_summary.json").read_text())
miss = {k.split(".")[1] if k.split(".")[0] != "vle" else "vle." + k.split(".")[1]: v for k, v in S["missing"].items()}
pc = lambda r: f"{r*100:.1f}%".replace(".", ",")
ZA = V["zero_activity_semantics"]; LR = V["late_registration"]; HA = V["high_activity"]

# (bảng, cột, thiếu, vấn đề, quyết định, lý do business/câu chuyện, dùng cho)
R = [
 ("studentInfo", "code_module", "0", "—", "Lọc BBB, FFF", "Hai module đủ 4 kỳ và có TMA1 trước ngày 28 (C03)", "Khóa, nhóm so sánh"),
 ("studentInfo", "code_presentation", "0", "—", "Lọc 4 kỳ; tạo cột split", "Kỳ là đơn vị thời gian của quyết định D1 → split theo thời gian", "Khóa, split"),
 ("studentInfo", "id_student", "0", "Một SV có thể học nhiều kỳ (569 SV)", "Giữ; không làm predictor; cờ seen_in_earlier_presentation", "Định danh giả; SV học lại là thực tế triển khai, không loại", "Khóa"),
 ("studentInfo", "gender", "0", "Nhạy cảm", "Chỉ audit, tách file riêng", "Danh sách liên hệ không được xếp theo giới; vẫn phải kiểm tra công bằng", "Audit"),
 ("studentInfo", "region", "0", "Nhạy cảm (vị trí)", "Chỉ audit", "Vùng không phải hành vi học tập có thể can thiệp", "Audit"),
 ("studentInfo", "highest_education", "0", "Thứ bậc ẩn trong nhãn chữ", "Mã thứ bậc 0–4 (highest_education_ord); giữ nhãn gốc", "Trình độ đầu vào liên hệ mạnh với kết quả (AQ2) và biết từ lúc đăng ký", "Feature AQ2/AQ3"),
 ("studentInfo", "imd_band", f"{S['missing']['studentInfo.imd_band']['n']} ({pc(S['missing']['studentInfo.imd_band']['rate'])})",
  f"Thiếu KHÔNG ngẫu nhiên: phụ thuộc vùng (χ² = {S['imd_missing_mechanism']['region']['chi2']}), kết quả (χ² = {S['imd_missing_mechanism']['final_result']['chi2']}), giới; nhãn “10-20” lệch định dạng",
  "Không xóa dòng, không impute; nhãn “Unknown” + cờ imd_band_missing; chuẩn hóa “10-20%”",
  "Xóa dòng sẽ làm mất 34% SV North Region → audit công bằng theo vùng bị lệch; impute mode sẽ bịa mức thiếu thốn cho SV", "Audit"),
 ("studentInfo", "age_band", "0", "Nhạy cảm; nhóm 55+ rất nhỏ (0,3%)", "Chỉ audit; gộp 35–55 và 55+ khi audit", "Gộp để nhóm đủ lớn cho thống kê, không dùng để xếp hạng", "Audit"),
 ("studentInfo", "num_of_prev_attempts", "0", "Lệch phải (0–6)", "Giữ nguyên", "Học lại là bối cảnh có thật, biết từ lúc đăng ký", "Feature"),
 ("studentInfo", "studied_credits", "0", f"Tối đa {S['studied_credits']['max']}; {S['studied_credits']['n_over_240']} SV > 240", "Giữ, không cắt", "Học nhiều module song song là hợp lệ và có thể là lý do quá tải", "Feature"),
 ("studentInfo", "disability", "0", "Nhạy cảm", "Chỉ audit", "Không xếp ưu tiên theo khuyết tật; kiểm tra mô hình không bỏ sót nhóm này", "Audit"),
 ("studentInfo", "final_result", "0", "46 ca mâu thuẫn với ngày hủy (0,3%)", "Giữ nguyên làm outcome chính thức; tạo target nhị phân và label3; gắn 2 cờ", "Nhãn do trường ghi nhận đáng tin hơn suy luận từ ngày hủy; không tự sửa nhãn", "Target"),
 ("studentRegistration", "date_registration", f"{S['missing']['studentRegistration.date_registration']['n']} (2 trong population)",
  f"Thiếu tập trung ở Withdrawn ({S['date_registration_missing_outcome'].get('Withdrawn', 0)}/20) → thiếu có thông tin (nghi MNAR); {LR['n_population_all_splits']} SV đăng ký ngày 1–28 có cửa sổ quan sát ngắn hơn",
  "Dẫn xuất registration_lead_days; thiếu → median train (chỉ để pipeline chạy); cờ thiếu KHÔNG làm predictor (v1.2, tránh rò rỉ nhãn); cờ late_registration_1_28 + days_observable_0_28; loại 7 lượt đăng ký sau ngày 28 khỏi population",
  f"Trên train, SV đăng ký muộn (n = {LR['n']}) click ít hơn cả khi chia theo ngày quan sát ({str(LR['median_clicks_per_observable_day_late']).replace('.', ',')} so với {str(LR['median_clicks_per_observable_day_ontime']).replace('.', ',')} click/ngày) nhưng tỷ lệ không hoàn thành tương đương ({pc(LR['noncompletion_rate'])} so với {pc(LR['noncompletion_rate_ontime'])}, KTC rộng) → mô hình có thể đánh giá rủi ro quá cao cho nhóm này; gắn cờ để M4 kiểm tra, không xóa", "Feature, lọc"),
 ("studentRegistration", "date_unregistration", f"{pc(S['missing']['studentRegistration.date_unregistration']['rate'])}", "Thiếu theo thiết kế (= chưa hủy); gần như mã hóa nhãn Withdrawn", "Chỉ dùng xác định population ngày 28; cấm làm feature (assert)", "Biến chỉ tồn tại sau khi sự việc xảy ra → leakage", "Lọc"),
 ("studentVle", "code_module / code_presentation / id_student", "0", "—", "Khóa nối với bảng đăng ký", "—", "Khóa"),
 ("studentVle", "id_site", "0", "0 orphan", "Nối vle để lấy loại tài nguyên; đếm số tài nguyên phân biệt", "Độ rộng tương tác là một khía cạnh của tham gia học", "Feature"),
 ("studentVle", "date", "0", "Từ −23 đến 269; 334.584 dòng trước khai giảng", "Feature chỉ lấy date ≤ 28; trước ngày 0 tách thành feature riêng", "Mốc dự báo cuối ngày 28; hoạt động trước khai giảng là tín hiệu chuẩn bị hợp lệ", "Feature, AQ1"),
 ("studentVle", "sum_click", "0", f"{V['vle_exact_duplicate_extra_rows']:,} dòng trùng hoàn toàn; skew {S['clicks_0_28']['skew']}; max {S['daily_clicks']['max']} click/SV-ngày".replace(",", "."),
  "Giữ dòng trùng và cộng dồn (aggregate); không cắt outlier; log1p trong pipeline mô hình; cờ flag_high_activity (> p99 train)",
  f"Tài liệu OULAD mô tả sum_click là số click trong một lượt ghi nhận; nhiều dòng cùng SV-tài nguyên-ngày với số click khác nhau là phổ biến → grain là lượt truy cập. Không có VLE trong 0–28 → 0 + cờ: trên train nhóm này {pc(ZA['noncompletion_rate_no_vle_0_28'])} không hoàn thành so với {pc(ZA['noncompletion_rate_with_vle'])} – thiếu bản ghi mang thông tin, xóa đi là mất đúng nhóm cần hỗ trợ. Outlier: không có trường nào cho phép kiểm chứng lỗi ghi nhận, nên giữ nguyên và dùng biến đổi bền vững (log1p) + cờ thay vì xóa",
  "Feature"),
 ("vle", "activity_type", "0", "19 loại; nhiều loại hiếm", "Dùng 7 loại phổ biến nhất làm feature riêng; không gộp tùy ý loại hiếm", "Loại tài nguyên giúp lời giải thích cụ thể hơn (AQ5)", "Feature"),
 ("vle", "week_from / week_to", f"{pc(S['missing']['vle.week_from']['rate'])}", "Thiếu có cấu trúc: 100% ở forumng, homepage (tài nguyên không gắn tuần)", "Không dùng (giữ trong raw)", "Không phải lỗi – loại tài nguyên đó không có lịch tuần; AQ không cần biến này", "—"),
 ("vle", "id_site / code_module / code_presentation", "0", "—", "Khóa", "—", "Khóa"),
 ("assessments", "id_assessment", "0", "—", "Khóa; xác định TMA1 = bài không phải Exam có hạn sớm nhất", "—", "Khóa"),
 ("assessments", "assessment_type", "0", "Exam có hạn sau ngày 28", "Loại Exam khỏi feature", "Thi cuối kỳ là thông tin sau mốc dự báo", "Lọc"),
 ("assessments", "date", "4 (Exam BBB)", "Thiếu chỉ ở bài thi cuối kỳ", "Không xử lý (Exam không dùng)", "Thiếu không ảnh hưởng AQ", "Feature TMA1"),
 ("assessments", "weight", "0", "TMA1 của BBB 2014J có trọng số 0%", "Ghi nhận; không dùng làm feature", "Ý nghĩa điểm TMA1 thay đổi giữa kỳ → lý do điểm chỉ là feature có điều kiện", "Bối cảnh"),
 ("studentAssessment", "id_student / id_assessment", "0", "0 bài của SV không đăng ký", "Khóa", "—", "Khóa"),
 ("studentAssessment", "date_submitted", "0", f"{S['submitted_before_start']['n']} bài nộp trước ngày 0 ({S['submitted_before_start']['banked']} là bảo lưu); 72 bài nộp sau kỳ", "Chỉ lấy bài nộp ≤ ngày 28; 67 bài TMA1 nộp sớm (không bảo lưu) giữ và tính là đúng hạn", "Nộp sớm trước khai giảng là hành vi thật khi tài liệu đã mở", "Feature TMA1"),
 ("studentAssessment", "is_banked", "0", f"{S['submitted_before_start']['banked_total']} bài bảo lưu, đều mang ngày âm", "Loại bài bảo lưu khỏi feature TMA1", "Điểm chuyển từ lần học trước không phản ánh hành vi kỳ này", "Lọc"),
 ("studentAssessment", "score", f"{S['score_missing_detail']['n']}", f"Thiếu dù đã nộp ({S['score_missing_detail']['has_date_submitted']}/{S['score_missing_detail']['n']}); {S['score_missing_detail']['score_zero_n']} bài điểm 0 thật", "Giữ NaN, không impute 0; chỉ dùng ở AQ2 và bộ feature có điều kiện", "Điền 0 sẽ trộn “chưa có điểm” với “0 điểm thật”; thời điểm có điểm chưa biết (DQ10)", "AQ2; conditional"),
 ("courses", "module_presentation_length", "0", "234–269 ngày; tên cột khác tài liệu (P2 DQ-02)", "Dùng kiểm tra hợp lệ thời gian", "—", "Kiểm tra"),
]
df = pd.DataFrame(R, columns=["Bảng", "Cột", "Thiếu (phạm vi)", "Vấn đề phát hiện", "Quyết định", "Lý do theo business / câu chuyện", "Dùng cho"])
df.to_csv(META / "column_decision_log.csv", index=False)
print(len(df), "dòng quyết định")
