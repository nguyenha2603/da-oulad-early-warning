# Early Warning System for Students Needing Support (OULAD)

**Nhóm N05 – MTH045 Khoa học Phân tích Dữ liệu, ĐH Khoa học Tự nhiên – ĐHQG TP.HCM**

Đề tài: *Hệ thống phát hiện sớm nhu cầu hỗ trợ của sinh viên dựa trên dữ liệu tương tác học tập.*
Dữ liệu: Open University Learning Analytics Dataset (OULAD), module BBB và FFF, 4 kỳ 2013B–2014J.

## Câu hỏi phân tích
| AQ | Mức | Nội dung |
|---|---|---|
| AQ1 | Descriptive | Tương tác 4 tuần đầu khác nhau thế nào giữa Pass / Fail / Withdrawn? |
| AQ2 | Diagnostic | Chỉ báo nào khác biệt có ý nghĩa, từ tuần nào? |
| AQ3 | Predictive | Dự báo Fail/Withdrawn chỉ từ dữ liệu ngày 0–28? |
| AQ4 | Prescriptive (decision support) | Quy tắc ưu tiên nào cho danh sách 200 ca mỗi module-kỳ? |
| AQ5 | Diagnostic (explanatory) | Với mỗi sinh viên trong danh sách ưu tiên, đặc trưng nào đóng góp nhiều nhất vào điểm rủi ro của mô hình, và có diễn đạt được thành 2–3 câu cho người không chuyên không? |

## Kết quả chính (test = kỳ 2014J)
- Khác biệt giữa sinh viên hoàn thành và không hoàn thành đã hiện diện từ tuần 1.
- Logistic regression: PR-AUC 0,662 (KTC 95% 0,637–0,686), Recall 0,693 – chưa đạt mốc 0,70 đã cam kết.
- Danh sách 200 ca/module-kỳ: Precision 0,760 (BBB) và 0,880 (FFF), khoảng 2 lần liên hệ ngẫu nhiên.
- Giới hạn: dữ liệu quan sát (không đo được hiệu quả can thiệp); một trường ở Anh Quốc, 2013–2014.

## Cấu trúc
```
P2/  Thu thập dữ liệu: data/raw (zip gốc OULAD – chỉ có trên GitHub, không kèm bản nộp), data/metadata (nguồn, license, SHA-256), notebook 01, src/verify_structure.py
P3/  Hiểu và chuẩn bị dữ liệu: src/prepare_data.py (pipeline), profile_data.py, dictionary.py, column_decisions.py,
     notebooks 02–03, data/processed (analytic-ready views), data/metadata (dictionary, lineage, manifest, logs)
P4/  Mô hình hóa và đánh giá: evaluation/ (plan chốt bằng SHA-256), src/run_prototype.py, notebook 04, results/
P5/  Dữ liệu biểu đồ (chart_data/), hình báo cáo P5 (figures/) và biểu đồ kể chuyện (story_charts/)
```

## Tái lập
Môi trường đã kiểm tra: **Python 3.11**, các phiên bản ghi trong `requirements.txt`. Toàn bộ pipeline chạy khoảng 1 phút.

**Bước 0 – Lấy dữ liệu gốc.** File zip OULAD (~45 MB) không kèm trong bản nộp Moodle. Notebook 01 tự tải từ UCI nếu `P2/data/raw/` chưa có file; có thể tải tay:
`https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip` (trang dataset: https://archive.ics.uci.edu/dataset/349) và đặt vào `P2/data/raw/`. SHA-256 của zip phải là
`f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4` (xem `P2/data/metadata/SHA256SUMS.txt`).

**Chạy theo đúng thứ tự** (P4 cần dữ liệu gốc để kiểm tra trọng số TMA1; P5 cần `P3/data/interim` do bước P3 sinh ra):
```bash
pip install -r requirements.txt
cd P2 && jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_collection.ipynb && cd ..   # tải/giải nén + kiểm tra SHA-256
cd P3 && python3 src/prepare_data.py && python3 src/profile_data.py && python3 src/dictionary.py && python3 src/column_decisions.py && cd ..
cd P4 && python3 src/run_prototype.py && cd ..
cd P5 && python3 src/export_chart_data.py && python3 src/story_charts.py
```
Notebook minh họa: chạy `P3/notebooks/03_data_preparation.ipynb` trước `02_data_understanding.ipynb` (notebook 02 đọc dữ liệu do 03/`prepare_data.py` tạo), rồi `P4/notebooks/04_modeling_prototype.ipynb`.
**Không chạy lại** `P4/src/freeze_eval_plan.py`: Evaluation Plan đã được chốt bằng SHA-256 trước khi chạy mô hình (`P4/evaluation/`).

Prepared dataset hiện tại: `N05-prepared-v1.2`. Mọi bước đều tất định (seed 42); kết quả ghi trong `P3/data/metadata` và `P4/results`.

## Dữ liệu và giấy phép
OULAD – Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset.* Scientific Data, 4, 170171. doi:10.1038/sdata.2017.171.
Bản sử dụng: UCI Machine Learning Repository, id 349, doi:10.24432/C5KK69. Giấy phép CC BY 4.0 – file zip gốc được phân phối lại nguyên vẹn kèm ghi công.
Dữ liệu đã được ẩn danh bởi nhà cung cấp; báo cáo và biểu đồ chỉ trình bày kết quả tổng hợp hoặc mã giả danh.

## Thành viên
Nguyễn Võ Thiên Hà (25C12020) · Nguyễn Gia Hân (25C12021) · Trần Nhật Trường (25C12014) · Trương Trung Vĩnh (25C12015)
Giảng viên hướng dẫn: TS. Nguyễn Trần Minh Thư
