# Thư mục lưu dữ liệu

Thư mục này chứa cache dataset local và các split đã chuẩn bị. Raw data, processed data
và segmented data là runtime artifact nên bị Git ignore. Provenance của dataset được ghi
trong [`manifests/`](manifests/), nơi version source, revision, schema, preprocessing và
split dự kiến mà không commit chính dữ liệu.

Logic load dataset và tokenization nằm trong
[`src/data/loader.py`](../src/data/loader.py). Dùng các script chuẩn bị trong
[`scripts/`](../scripts/) để tạo dataset segmented local.
