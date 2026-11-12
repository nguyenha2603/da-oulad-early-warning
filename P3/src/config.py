"""Cấu hình dùng chung cho Milestone 3 (N05 – OULAD early warning)."""
from pathlib import Path

P3 = Path(__file__).resolve().parents[1]
FINAL = P3.parent
# Raw data được thu thập và khóa ở Milestone 2 – chỉ đọc, không sao chép/sửa.
RAW = FINAL / "P2" / "data" / "raw" / "oulad"
RAW_SHA = FINAL / "P2" / "data" / "metadata" / "SHA256SUMS.txt"
INTERIM = P3 / "data" / "interim"
PROCESSED = P3 / "data" / "processed"
META = P3 / "data" / "metadata"
FIG = P3 / "reports" / "milestone3" / "figures"

DATASET_VERSION = "OULAD (UCI id 349, DOI 10.24432/C5KK69), tải 09/11/2026"
PREPARED_VERSION = "N05-prepared-v1.2"

MODULES = ["BBB", "FFF"]                       # Change Log C03
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]
PREDICTION_DAY = 28                            # cuối ngày 28 = cuối tuần 4 (AQ3, C04)
WEEKS = [(1, 0, 6), (2, 7, 13), (3, 14, 20), (4, 21, 27)]   # 4 tuần đủ 7 ngày; ngày 28 chỉ vào các tổng 0–28 (v1.2)
NA = ["?"]                                     # P2 DQ-03: missing được mã hóa bằng "?"

# Split theo thời gian (Task 6): học trên quá khứ, kiểm thử trên kỳ mới nhất.
SPLIT = {"2013B": "train", "2013J": "train", "2014B": "validation", "2014J": "test"}

NONCOMPLETION = ["Fail", "Withdrawn"]
ACTIVITY_TYPES = ["homepage", "oucontent", "resource", "subpage", "forumng", "quiz", "url"]

# Biến nhạy cảm: giữ để audit công bằng, không đưa vào bộ predictor mặc định.
SENSITIVE = ["gender", "age_band", "disability", "imd_band", "region"]
CAPACITY_K = [100, 200, 300]                   # ca liên hệ / module-kỳ (A5, C05)

# Biến dẫn xuất từ nhãn / ngày hủy – cấm xuất hiện trong bảng mô hình (assert ở validate)
LABEL_DERIVED = ["final_result", "label3", "date_unregistration", "unregistered_by_d28", "flag_withdrawn_no_unreg_date", "flag_unreg_date_not_withdrawn"]
