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
| AQ5 | Diagnostic (explanatory) | Lý do nào khiến một sinh viên được đưa vào danh sách? |

## Kết quả chính (test = kỳ 2014J)
- Khác biệt giữa sinh viên hoàn thành và không hoàn thành đã hiện diện từ tuần 1.
- Logistic regression: PR-AUC 0,662 (KTC 95% 0,637–0,686), Recall 0,693 – chưa đạt mốc 0,70 đã cam kết.
- Danh sách 200 ca/module-kỳ: Precision 0,760 (BBB) và 0,880 (FFF), khoảng 2 lần liên hệ ngẫu nhiên.
- Giới hạn: dữ liệu quan sát (không đo được hiệu quả can thiệp); một trường ở Anh Quốc, 2013–2014.

## Cấu trúc
```
P2/  Thu thập dữ liệu: data/raw (zip gốc OULAD), data/metadata (nguồn, license, SHA-256), notebook 01, src/verify_structure.py
P3/  Hiểu và chuẩn bị dữ liệu: src/prepare_data.py (pipeline), profile_data.py, dictionary.py, column_decisions.py,
     notebooks 02–03, data/processed (analytic-ready views), data/metadata (dictionary, lineage, manifest, logs)
P4/  Mô hình hóa và đánh giá: evaluation/ (plan chốt bằng SHA-256), src/run_prototype.py, notebook 04, results/
P5/  Dữ liệu biểu đồ, biểu đồ kể chuyện (story_charts/) và hướng dẫn vẽ lại (chart_guide/N05_chart_guide.html)
```

## Tái lập
```bash
pip install -r requirements.txt
cd P2 && jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb && cd ..   # giải nén + kiểm tra SHA-256
cd P3 && python3 src/prepare_data.py && python3 src/profile_data.py && python3 src/dictionary.py && python3 src/column_decisions.py && cd ..
cd P4 && python3 src/run_prototype.py && cd ..
cd P5 && python3 src/export_chart_data.py && python3 src/story_charts.py && python3 src/build_chart_guide.py
```
Prepared dataset hiện tại: `N05-prepared-v1.2`. Mọi bước đều tất định (seed 42); kết quả ghi trong `P3/data/metadata` và `P4/results`.

## Dữ liệu và giấy phép
OULAD – Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset.* Scientific Data, 4, 170171. doi:10.1038/sdata.2017.171.
Bản sử dụng: UCI Machine Learning Repository, id 349, doi:10.24432/C5KK69. Giấy phép CC BY 4.0 – file zip gốc được phân phối lại nguyên vẹn kèm ghi công.
Dữ liệu đã được ẩn danh bởi nhà cung cấp; báo cáo và biểu đồ chỉ trình bày kết quả tổng hợp hoặc mã giả danh.

## Thành viên
Nguyễn Võ Thiên Hà (25C12020) · Nguyễn Gia Hân (25C12021) · Trần Nhật Trường (25C12014) · Trương Trung Vĩnh (25C12015)
Giảng viên hướng dẫn: TS. Nguyễn Trần Minh Thư
