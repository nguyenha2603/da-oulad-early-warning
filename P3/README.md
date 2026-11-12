# N05 – Milestone P3: From Understanding to Preparation (OULAD)

Đề tài: Hệ thống phát hiện sớm nhu cầu hỗ trợ của sinh viên dựa trên dữ liệu tương tác học tập (MTH045).

## Tái lập (raw → interim → processed)
```bash
cd P3
pip install pandas numpy pyarrow matplotlib
python3 src/prepare_data.py   # kiểm tra SHA-256 raw, chuẩn bị dữ liệu, validation
python3 src/profile_data.py   # profiling + hình reports/milestone3/figures
python3 src/dictionary.py     # data dictionary + lineage
python3 src/column_decisions.py  # Column Decision Log
```
Raw đọc trực tiếp từ `../P2/data/raw/oulad` (chỉ đọc; chạy notebook P2 để giải nén từ zip gốc). Pipeline dừng nếu SHA-256 không khớp.
Notebook: `notebooks/03_data_preparation.ipynb` (chạy trước), `notebooks/02_data_understanding.ipynb`.

## Phạm vi và quy tắc chính
- Module BBB + FFF, kỳ 2013B, 2013J, 2014B, 2014J (C03). Population = lượt đăng ký còn học ở cuối ngày 28 (C04, C06): 13.020.
- Mốc dự báo: cuối ngày 28; feature chỉ từ `date <= 28` / `date_submitted <= 28`. Tuần 1 = 0–6, …, tuần 4 = 21–27 (ngày 28 chỉ vào các tổng 0–28).
- Split theo kỳ: train 2013B+2013J, validation 2014B, test 2014J. Impute fit trên train.
- Biến nhạy cảm (gender, age_band, disability, imd_band, region) chỉ ở `aq3_audit_attributes.csv`.
- `tma1_score` là feature có điều kiện (chưa xác minh thời điểm có điểm) – M4 so sánh có/không.

## Prepared Data Package (`data/processed`)
| File | Dùng cho |
|---|---|
| aq1_student_week.csv | AQ1 – SV × tuần 1–4 |
| aq2_diagnostic_table.csv | AQ2 – so sánh nhóm |
| aq3_model_table.csv | AQ3/AQ4/AQ5 – feature ≤ ngày 28 + target + split |
| aq3_audit_attributes.csv | Audit công bằng |
| context_all_registrations.csv | Bối cảnh (gồm SV rút trước ngày 28) |
| aq4_decision_params.json | k = 100/200/300 ca/module-kỳ, tỷ lệ chuyển đổi giả định |

Metadata: `data/metadata/` (data_dictionary, data_lineage, prepared_manifest có SHA-256, transformation_log, validation_summary, profiling_summary).

## License
Dữ liệu: OULAD CC BY 4.0 – Kuzilek, Hlosta & Zdrahal (2017), doi:10.1038/sdata.2017.171.


## v1.1 (09/11/2026) – rà soát tiền xử lý
- `python3 src/column_decisions.py` → `data/metadata/column_decision_log.csv`: quyết định + lý do business cho từng cột của 7 bảng.
- Kiểm định cơ chế thiếu (chi-square) trong `profile_data.py`; kiểm tra ngữ nghĩa 0 click, đăng ký muộn, hoạt động sau ngày hủy trong `validation_summary.json`.
- Cờ mới: `late_registration_1_28`, `days_observable_0_28`, `flag_high_activity` (ngưỡng p99 fit trên train). Không xóa dòng nào vì thiếu dữ liệu.
- Phân tích VIF (train) → `feature_sets.reduced_core` trong `prepared_manifest.json` (21 biến, VIF ≤ 5).

## v1.2 (09/11/2026)
- Tuần 4 = ngày 21–27; hai cờ dẫn xuất từ nhãn và `date_registration_missing` chuyển sang `aq3_audit_attributes.csv` (assert vắng mặt trong bảng mô hình).
- Kiểm tra ngữ nghĩa chỉ tính trên tập train; thống kê trước/sau biến đổi; Cramér's V; SHA-256 của mã nguồn trong manifest.
