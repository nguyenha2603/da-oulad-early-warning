"""Dựng P5/chart_guide/N05_dashboard_guide.html: chỉ dashboard cố vấn, mỗi thành phần kèm file/cột/cách vẽ/số kiểm tra.
Chạy sau export_chart_data.py:  python3 P5/src/build_dashboard_guide.py"""
import json
from pathlib import Path
import pandas as pd

P5 = Path(__file__).resolve().parents[1]
CD = P5 / "chart_data"
rec = lambda df: df.to_dict(orient="records")
top = pd.read_csv(CD / "dashboard_top300.csv").drop(columns=["target_noncompletion"])  # đáp án thật: không đưa lên dashboard
D = {"cohort": rec(pd.read_csv(CD / "dashboard_cohort_scores.csv")), "top": rec(top.fillna("")),
     "week": rec(pd.read_csv(CD / "dashboard_weekly_list_vs_rest.csv")), "med": rec(pd.read_csv(CD / "dashboard_class_median_weekly.csv")),
     "fair": rec(pd.read_csv(CD / "c11_fairness.csv")[["attribute", "group", "share_population", "share_top200"]])}
html = (P5 / "src/dashboard_guide_template.html").read_text()
html = html.replace("__DATA__", json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"), 1)
out = P5 / "chart_guide/N05_dashboard_guide.html"
out.write_text(html)
print("wrote", out, round(out.stat().st_size / 1e3), "KB")
