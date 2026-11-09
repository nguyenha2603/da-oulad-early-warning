# P4 – From Modeling to Evaluation (Level 1 + Level 2 prototype)

## Tái lập
```bash
cd P4
python3 src/run_prototype.py      # kiểm tra SHA-256 của evaluation plan và dữ liệu P3, chạy AQ1–AQ5 + fairness (~10 giây)
```
`src/freeze_eval_plan.py` chỉ dùng một lần để chốt Evaluation Plan (đã chốt: `evaluation/evaluation_plan.lock`). Notebook: `notebooks/04_modeling_prototype.ipynb`. Seed 42.

## Thiết kế
- AQ1: trung vị + bootstrap KTC 95%. AQ2: Mann-Whitney / Cliff's δ, chi-square, Holm + logistic hiệu chỉnh.
- AQ3: M0 base rate, M1 quy tắc, M2 logistic regression (primary), M3 gradient boosting; fit 2013B+2013J, chọn trên 2014B, test 2014J một lần.
- AQ4: top-k mỗi module-kỳ (k = 100/200/300). AQ5: linear contribution + permutation importance.
- Sai lệch so với plan: PD-01 (tính khả thi quy tắc AQ4), PD-02 (bộ feature rút gọn – khám phá), PD-03 (dữ liệu v1.0 → v1.2).

## Kết quả (`results/*.json`)
| AQ | Kết quả | Tiêu chí chốt trước |
|---|---|---|
| AQ3 | PR-AUC 0,662 (M1 0,615); Recall 0,693 (KTC 95% 0,668–0,717) | Chưa đạt |
| AQ4 | Precision@200: BBB 0,760, FFF 0,880 | Đạt (BBB sát biên) |
| AQ5 | Fidelity đạt; một số biến tương tác có hệ số dương → lý do khó hiểu | Đạt một phần |
