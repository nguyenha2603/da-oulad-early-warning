# P5 – Dữ liệu biểu đồ và hướng dẫn vẽ lại

- `src/export_chart_data.py`: xuất dữ liệu cho 11 biểu đồ + dashboard (`chart_data/*.csv`, `bundle.json`) và vẽ `figures/p5_dashboard_wireframe.png`, `figures/p5_fairness.png`.
- `src/build_chart_guide.py`: dựng `chart_guide/N05_chart_guide.html` – mở bằng trình duyệt; mỗi biểu đồ có dữ liệu CSV và hướng dẫn dựng lại trong Power BI / Apache Superset; có tab dashboard mẫu, bảng Observation–Insight–Evidence–Limitation và data story.

Mã sinh viên trên dashboard được giả danh (SV-001…); dữ liệu cấp sinh viên không chứa id gốc.

- `src/story_charts.py`: 7 biểu đồ kể chuyện khổ 16:9 (`story_charts/s1…s7.png`) + `storyboard_6.png` – tiêu đề là thông điệp, một màu nhấn, nhãn trực tiếp; dùng cho slide và mục Data Story của báo cáo.
