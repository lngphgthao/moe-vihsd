# Dữ liệu ngoài Git

Logic dataset ở src/data/loader.py; chỉ README và manifests được lưu trong Git.
Hai DatasetDict local nằm tại data/processed/vianli_segmented và vihsd_segmented.
Chuyển vị trí ngày 2026-11-22 đã đối chiếu SHA-256 của từng file; không xóa dữ liệu local.

Clone mới cần tạo bằng scripts/segment_vianli.py hoặc preprocess_vihsd.py, hoặc khôi phục
bản ngoài Git vào đường dẫn manifest. PhoBERT dùng segmented; XLM-R/mBERT dùng text gốc
trừ khi experiment chủ động khảo sát preprocessing. Loader không tự tìm fallback.

Manifest ghi checksum bản local và provenance biết được; phiên bản segmentation của
bản lịch sử chưa được xác minh, không khẳng định tái tạo byte-identical. Không phân phối
lại dữ liệu nếu chưa kiểm tra điều kiện truy cập của nguồn.
