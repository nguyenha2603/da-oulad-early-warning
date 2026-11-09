"""Dựng file HTML hướng dẫn vẽ lại chart (Power BI / Superset). Chạy sau export_chart_data.py:
python3 P5/src/build_chart_guide.py  ->  P5/chart_guide/N05_chart_guide.html (một file, dữ liệu nhúng sẵn)"""
import json
from pathlib import Path

P5 = Path(__file__).resolve().parents[1]
B = json.loads((P5 / "chart_data/bundle.json").read_text())
A4 = json.loads((P5.parent / "P4/results/aq4_results.json").read_text())
A3 = json.loads((P5.parent / "P4/results/aq3_results.json").read_text())
A2 = json.loads((P5.parent / "P4/results/aq2_results.json").read_text())
FE = json.loads((P5.parent / "P4/results/fairness_error_results.json").read_text())
V3 = json.loads((P5.parent / "P3/data/metadata/validation_summary.json").read_text())
U = {r["indicator"]: r for r in A2["unadjusted"]}
f = lambda x, k=3: f"{x:.{k}f}".replace(".", ",").replace("-", "−")
pc = lambda x: f"{x*100:.1f}".replace(".", ",") + "%"
T = A3["models"]["M2"]["test"]; SPL = A3["split"]; G = FE["fairness"]["gender"]["groups"]


def S(id, title, where, data, goal, kind, x, y, legend, filt, pbi, sup, note):
    return {"id": id, "title": title, "where": where, "data": data, "goal": goal, "spec": [
        ["Câu hỏi / mục tiêu", goal], ["Loại chart", kind], ["Trục X / Category", x], ["Trục Y / Values", y],
        ["Legend / màu", legend], ["Bộ lọc", filt], ["Power BI", pbi], ["Superset", sup], ["Ghi chú", note]]}


SPECS = [
    S("c01", "1. Tỷ lệ dữ liệu thiếu theo biến", "P3 Hình 3.1 · chất lượng dữ liệu", "c01_missingness",
      "Biến nào thiếu dữ liệu và thiếu bao nhiêu – căn cứ cho quyết định xử lý missing (P3 mục 3).",
      "Bar ngang", "<code>variable</code>", "<code>missing_rate</code> (định dạng %)", "Một màu (xanh dương)", "—",
      "Clustered bar chart: Y-axis = variable, X-axis = missing_rate; Format → Data labels bật; sắp giảm dần.",
      "Bar Chart, orientation Horizontal; Dimension = variable; Metric = MAX(missing_rate).",
      "date_unregistration thiếu 69,6% là thiếu theo thiết kế (SV không hủy), ghi chú dưới chart."),
    S("c02", "2. Kết quả cuối khóa theo module-kỳ", "P3 Hình 3.2 · P5 Hình 6.2 · cân bằng nhãn", "c02_target_balance",
      "Tỷ lệ Pass/Fail/Withdrawn của SV còn học ở ngày 28 thay đổi thế nào giữa các kỳ (DQ12).",
      "Cột chồng 100%", "<code>code_module</code> + <code>code_presentation</code> (gộp thành cột Module-kỳ)", "<code>share</code> hoặc <code>n</code>",
      "<code>label3</code>: Pass xanh dương, Fail cam, Withdrawn xanh ngọc", "Module",
      "100% stacked column chart: X-axis = Module-kỳ (cột tính toán), Legend = label3, Values = SUM(n).",
      "Bar Chart, Stacked + Contribution mode = Row; Dimension = module-kỳ; Series = label3; Metric = SUM(n).",
      "Thứ tự legend cố định Pass → Fail → Withdrawn."),
    S("c03", "3. Số SV hoạt động mỗi ngày", "P3 Hình 3.3 · độ phủ thời gian", "c03_daily_active",
      "Dữ liệu tương tác có phủ đủ cửa sổ ngày 0–28 không; chu kỳ tuần quanh hạn nộp.",
      "Line", "<code>date</code> (số, ngày so với khai giảng)", "<code>active_students</code>", "<code>code_module</code>: BBB xanh dương, FFF cam", "Module",
      "Line chart: X-axis = date (Type = Continuous), Y-axis = SUM(active_students), Legend = code_module; Analytics → X-axis constant line = 28.",
      "Line Chart (Generic x-axis); X = date; Metric = SUM(active_students); Group by code_module; thêm Annotation layer tại 28.",
      "Đường đứt tại ngày 28 = mốc dự báo."),
    S("c04", "4. Click theo tuần theo kết quả (AQ1)", "P4 Hình 7.1 · P5 Hình 10.1 · trái tim data story", "c04_weekly_clicks",
      "AQ1: tương tác tuần 1–4 khác nhau thế nào giữa Pass / Fail / Withdrawn.",
      "Line + dải KTC 95%", "<code>week</code>", "<code>clicks_median</code>; dải <code>clicks_ci_low</code>–<code>clicks_ci_high</code>",
      "<code>label3</code> (3 màu cố định)", "—",
      "Line chart: X = week, Y = clicks_median, Legend = label3; Analytics → Error bars: Upper = clicks_ci_high, Lower = clicks_ci_low.",
      "Mixed Chart hoặc Line Chart; Metric = MAX(clicks_median); Series = label3; dải KTC vẽ bằng hai metric ci_low/ci_high (Area).",
      "Dùng trung vị vì phân phối lệch phải; có thể vẽ thêm active_days_mean ở chart thứ hai (không dùng trục kép)."),
    S("c05", "5. Phân bố tổng click 4 tuần đầu", "P3 Hình 3.5 · phân phối", "c05_clicks_per_student",
      "Phân phối click lệch phải và có nhiều SV 0 click → lý do dùng log và cờ không hoạt động.",
      "Histogram", "log(1 + <code>clicks_0_28</code>), bin 0,2", "Số sinh viên", "Một màu; có thể tách theo <code>label3</code> bằng small multiples", "Module, label3",
      "Thêm cột Power Query = Number.Log(1+[clicks_0_28]); New group → Bins (size 0,2); Clustered column: X = bin, Y = COUNT(id_student).",
      "Histogram chart; Column = log cột tính bằng SQL <code>LN(1+clicks_0_28)</code>; Number of bins ≈ 45.",
      "542 SV có 0 click trong 0–28."),
    S("c06", "6. Thời điểm hủy đăng ký", "P3 Hình 3.6 · P5 Hình 6.1", "c06_unregistration",
      "Bao nhiêu SV đã rút trước mốc quyết định ngày 28 – căn cứ C04, C05, C06.",
      "Histogram", "<code>date_unregistration</code>, bin 7 ngày", "Số lượt hủy", "Một màu (xanh ngọc)", "Module, final_result",
      "Bins (size 7) trên date_unregistration; Column chart: X = bin, Y = COUNT; constant line X = 28.",
      "Histogram chart; Column = date_unregistration; bins theo tuần; Annotation tại 28.",
      "2.644 lượt hủy ≤ ngày 28 (55,1% số Withdrawn)."),
    S("c07", "7. Effect size theo tuần (AQ2)", "P4 Hình 7.2 · P5 Hình 10.2", "c07_aq2_effect_by_week",
      "AQ2: khác biệt giữa nhóm hoàn thành và không hoàn thành bắt đầu từ tuần nào.",
      "Line", "<code>week</code>", "<code>cliffs_delta</code>", "<code>measure</code>: click xanh dương, ngày hoạt động cam", "—",
      "Line chart: X = week, Y = cliffs_delta, Legend = measure; Analytics → Y constant line = −0,147 (ngưỡng nhỏ).",
      "Line Chart; Metric = MAX(cliffs_delta); Series = measure; Annotation y = −0,147.",
      "δ âm = nhóm không hoàn thành thấp hơn; đã vượt ngưỡng ngay tuần 1. Bảng chỉ báo đầy đủ: c07b_aq2_indicators.csv."),
    S("c08", "8a. Đường Precision–Recall (AQ3)", "P4 Hình 7.3 trái · P5 Hình 10.3", "c08_pr_curve",
      "AQ3: mô hình có xếp hạng tốt hơn baseline không (test 2014J).",
      "Line (x-y)", "<code>recall</code>", "<code>precision</code>", "<code>model</code>: M1 vàng, M2 xanh dương, M3 cam", "Model",
      "Line chart với X = recall (Continuous, không tổng hợp – Don't summarize), Y = precision, Legend = model; constant line Y = 0,378 (tỷ lệ nền).",
      "Line Chart (Generic x-axis); X = recall; Metric = MAX(precision); Series = model.",
      f"PR-AUC test: M1 {f(A3['models']['M1']['test']['pr_auc'])} · M2 {f(T['pr_auc'])} · M3 {f(A3['models']['M3']['test']['pr_auc'])}."),
    S("c08b", "8b. Calibration (AQ3)", "P4 Hình 7.3 phải", "c08b_calibration",
      "Xác suất dự báo có khớp tỷ lệ thực tế không (kỳ test lệch phân phối).",
      "Scatter + đường chéo", "<code>predicted</code>", "<code>observed</code>", "<code>model</code>: M2 xanh dương, M3 cam", "Model",
      "Scatter chart: X = predicted, Y = observed, Legend = model; thêm đường chéo bằng Analytics → Trend line hoặc bảng phụ (0,0)-(1,1).",
      "Scatter Plot (hoặc Line Chart có markers); X = predicted; Metric = MAX(observed).",
      "Điểm dưới đường chéo = mô hình đánh giá rủi ro cao hơn thực tế."),
    S("c09", "9. Precision@k theo quy tắc (AQ4)", "P4 Hình 7.4 · P5 Hình 10.4", "c09_precision_at_k",
      "AQ4: với 100/200/300 ca mỗi module-kỳ, quy tắc nào đúng đối tượng nhiều nhất.",
      "Cột nhóm", "<code>k</code>", "AVERAGE(<code>precision</code>)", "<code>rule</code>: (a) xanh dương, (b) xanh ngọc, (d) vàng", "module_presentation",
      "Clustered column: X = k, Y = AVERAGE(precision), Legend = rule; constant line Y = 0,70; slicer module_presentation.",
      "Bar Chart; Dimension = k; Series = rule; Metric = AVG(precision); Annotation y = 0,70.",
      "(d) gộp hai module vượt trần 200 ca/module → không khả thi (PD-01)."),
    S("c10", "10. Độ quan trọng đặc trưng (AQ5)", "P4 Hình 7.5 · P5 Hình 10.5", "c10_permutation_importance",
      "AQ5: mô hình dựa vào yếu tố nào nhiều nhất.",
      "Bar ngang", "<code>feature</code>", "<code>pr_auc_drop</code>", "Một màu", "—",
      "Clustered bar chart: Y = feature (sắp giảm dần), X = pr_auc_drop.",
      "Bar Chart Horizontal; Dimension = feature; Metric = MAX(pr_auc_drop).",
      "Đổi tên feature sang tiếng Việt khi trình bày (vd. tma1_submitted_by_d28 = Đã nộp TMA1 trước ngày 28)."),
    S("c11", "11. Recall/Precision theo nhóm (công bằng)", "P4 Bảng 7.8", "c11_fairness",
      "Mô hình có bỏ sót nhóm sinh viên nào nhiều hơn không.",
      "Cột nhóm", "<code>attribute</code> + <code>group</code>", "<code>recall</code>, <code>precision</code>", "Recall xanh dương, Precision cam", "attribute",
      "Clustered column: X = group, Values = recall, precision; slicer attribute (hoặc small multiples theo attribute).",
      "Bar Chart; Dimension = group; Metrics = MAX(recall), MAX(precision); filter attribute.",
      "Flag duy nhất: nhóm IMD Unknown (2,8% SV, ít ca dương); nữ có FPR cao hơn nam. Tham khảo thêm cột fpr, fnr, recall_ci95 trong CSV."),
]

OIEL = [
    ["6. Thời điểm hủy đăng ký", "Đỉnh hủy quanh ngày 0–28", "Hơn nửa số rút môn xảy ra trước mốc quyết định", "55,1% Withdrawn rút ≤ ngày 28", "Không biết lý do rút"],
    ["2. Kết quả theo module-kỳ", "Tỷ lệ không hoàn thành khác nhau giữa các kỳ", "Ngưỡng cố định khó chuyển giữa các kỳ", f"train {pc(SPL['train']['rate'])} → test {pc(SPL['test']['rate'])}", "Chỉ 4 kỳ, 2 module"],
    ["4. Click theo tuần", "Pass luôn cao hơn Fail cả 4 tuần", "Khác biệt có từ đầu, không tụt dần", "KTC 95% của trung vị không chồng nhau", "Click là proxy"],
    ["7. Effect size theo tuần", "|δ| vượt ngưỡng ngay tuần 1", "Theo dõi được từ tuần 1–2", f"δ tuần 1 = {f(U['clicks_w1']['effect'],2)}; tuần 4 = {f(U['clicks_w4']['effect'],2)}", "Tuần 1 là tuần sớm nhất được đo; association"],
    ["Chỉ báo TMA1 (c07b)", "SV không hoàn thành nộp TMA1 ít hơn", "Tín hiệu sớm nhất quan sát được", f"{pc(U['tma1_submitted_by_d28']['median_noncompl'])} vs {pc(U['tma1_submitted_by_d28']['median_compl'])}", "Giá trị can thiệp chưa kiểm chứng"],
    ["8a. Precision–Recall", "LR nằm trên M1 ở hầu hết dải recall", "Mô hình xếp hạng tốt hơn quy tắc đơn giản", f"PR-AUC {f(T['pr_auc'])} vs {f(A3['models']['M1']['test']['pr_auc'])}", f"Recall {f(T['recall'])} < 0,70"],
    ["8b. Calibration", "Điểm dưới đường chéo", "Mô hình overestimate rủi ro ở kỳ mới", f"Xác suất TB {f(A3['extras']['calibration']['test']['mean_pred'])} vs base rate {f(A3['extras']['calibration']['test']['base_rate'])}", "Cần calibration và chọn lại ngưỡng"],
    ["9. Precision@k", "(a) vượt 0,70 ở k = 100, 200", "200 ca/module là mức hợp lý; diminishing returns",
     f"P@200 {f(A4['test']['a_k200']['per_module']['BBB-2014J']['precision'])} / {f(A4['test']['a_k200']['per_module']['FFF-2014J']['precision'])}", "Không đo hiệu quả liên hệ"],
    ["10. Độ quan trọng", "TMA1, số ngày hoạt động đứng đầu", "Lời giải thích xoay quanh nộp bài và đăng nhập", "Mức giảm PR-AUC khi xáo trộn", "Multicollinearity biến click"],
    ["11. Công bằng", "Recall gần nhau; FPR nữ cao hơn nam", "Không bỏ sót nhóm nào rõ rệt; nữ bị gắn cờ nhầm nhiều hơn", f"FPR {f(G['F']['fpr'],2)} vs {f(G['M']['fpr'],2)}", "Nhóm nhỏ (IMD Unknown) có KTC rộng"],
]

STORY = (f"""
<h2 style="margin-top:0">Sinh viên cần hỗ trợ không “biến mất dần” – họ khác ngay từ tuần đầu tiên</h2>
<h3>1. Bối cảnh</h3><p>Ở một trường đào tạo từ xa, cố vấn học tập chỉ có khoảng 200 ca liên hệ cho mỗi module mỗi kỳ, trong khi mỗi module có khoảng 1.600 sinh viên còn học sau tháng đầu và khoảng 2 trên 5 người sẽ không hoàn thành. Câu hỏi không phải “ai có nguy cơ”, mà là “200 cuộc gọi nên dành cho ai”.</p>
<h3>2. Điều nhóm từng nghĩ</h3><p>Proposal giả định sinh viên gặp khó sẽ giảm tương tác dần, nên “mức suy giảm tuần 1→4” được chọn làm tín hiệu chính.</p>
<h3>3. Điều dữ liệu cho thấy</h3><p>Không phải vậy. Ngay tuần 1, nhóm sau này không hoàn thành (Fail + Withdrawn) có trung vị {f(U["clicks_w1"]["median_noncompl"],0)} click so với {f(U["clicks_w1"]["median_compl"],0)} ở nhóm hoàn thành, và khoảng cách giữ nguyên cả bốn tuần (biểu đồ 4, 7). Mức suy giảm tuần 1→4 gần như không phân biệt hai nhóm. Tín hiệu rõ nhất là chưa nộp bài đánh giá đầu tiên: 1 trên 4 sinh viên không hoàn thành chưa nộp, so với chưa tới 1 trên 30 ở nhóm hoàn thành.</p>
<h3>4. Từ tín hiệu đến danh sách</h3><p>Một mô hình hồi quy logistic đơn giản, chỉ dùng dữ liệu đến ngày 28, xếp hạng sinh viên của kỳ 2014J – kỳ mô hình chưa từng thấy. Gọi 200 người đầu danh sách thì khoảng 3/4 (BBB) đến 7/8 (FFF) cuộc gọi đến đúng người cần hỗ trợ, gấp khoảng hai lần gọi ngẫu nhiên (biểu đồ 9). Mỗi tên kèm 2–3 lý do trung tính (tab Dashboard). Kết quả này cùng mức với nghiên cứu đã công bố trên OULAD (Top-10%: PPV 0,834 – Wu et al., 2026).</p>
<h3>5. Điều chưa trả lời được</h3><p>Mô hình bỏ sót sinh viên trông bình thường ở tuần 4 nhưng gặp khó về sau; Recall {f(T["recall"])} chưa đạt mốc 0,70; một số lý do giải thích còn khó hiểu. OULAD không ghi nhận việc liên hệ nên chưa biết một cuộc gọi có giúp sinh viên hoàn thành hay không.</p>
<h3>6. Đề xuất</h3><p>Theo dõi từ tuần 1–2; nhắc nhở quanh hạn bài đánh giá đầu tiên; dùng danh sách 200 ca kèm lý do ở cuối tuần 4, con người duyệt từng ca; chạy thử nghiệm có nhóm đối chứng trước khi mở rộng.</p>
<p class="note">Gợi ý trình bày 10 phút: Bối cảnh (1p) → Điều nhóm từng nghĩ (1p) → Biểu đồ 4 + 7 (2p) → Danh sách + dashboard demo (3p) → Giới hạn (2p) → Đề xuất (1p).</p>
""")


def js(o):
    return json.dumps(o, ensure_ascii=False, default=float).replace("</", "<\\/")


html = (P5 / "src/chart_guide_template.html").read_text()
for k, v in [("__DATA__", js(B)), ("__SPECS__", js(SPECS)), ("__OIEL__", js(OIEL)), ("__STORY__", js(STORY))]:
    html = html.replace(k, v, 1)
out = P5 / "chart_guide/N05_chart_guide.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")
