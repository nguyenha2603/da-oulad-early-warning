# N05 – Milestone P2: Data Collection Package (OULAD)

Đề tài: Hệ thống phát hiện sớm nhu cầu hỗ trợ của sinh viên dựa trên dữ liệu tương tác học tập (MTH045).

## Cấu trúc
```
P2/
  README.md
  data/
    raw/                      # nguyên gốc, chỉ đọc (chmod a-w) – KHÔNG chỉnh sửa
      open+university+learning+analytics+dataset.zip
      oulad/                  # 7 CSV + OULAD.names giải nén từ zip
    metadata/
      SOURCE_LICENSE_CITATION.md   # provenance, license, citation
      SHA256SUMS.txt, MD5SUMS.txt  # bằng chứng toàn vẹn
      access_timestamp_utc.txt, download_headers.txt
      official_OULAD_documentation_page.html   # bản lưu trang tài liệu chính thức
      uci_api_*.json, uci_page_*.html          # metadata các candidate dataset (Search Log)
      structural_verification.json             # kết quả kiểm tra cấu trúc
  notebooks/01_data_collection.ipynb   # tải + checksum + structural verification
  src/verify_structure.py              # script kiểm tra cấu trúc (read-only)
```

## Tái lập
```bash
cd P2
pip install pandas jupyter
jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb
```
Notebook chỉ tải dữ liệu khi `data/raw` chưa có file, kiểm tra SHA-256 với `data/metadata/SHA256SUMS.txt`
(dừng nếu không khớp), rồi chạy `src/verify_structure.py`.

Tải thủ công: mở https://archive.ics.uci.edu/dataset/349 → Download, đặt zip vào `data/raw/`, giải nén vào `data/raw/oulad/`.
Link trên trang OU Analyse (`schools.stem.open.ac.uk/cdn/files/anonymisedData.zip`) trả về 404 tại thời điểm 09/11/2026.

## Không làm ở Milestone 2
Không xóa missing, không impute, không chuẩn hóa, không bỏ bản ghi trùng, không feature engineering, không train model.
Mọi bất thường cấu trúc chỉ được ghi nhận (xem báo cáo mục 6) và chuyển sang Milestone 3.

## License
OULAD: CC BY 4.0 — Kuzilek, Hlosta & Zdrahal (2017), *Scientific Data* 4:170171, doi:10.1038/sdata.2017.171.
