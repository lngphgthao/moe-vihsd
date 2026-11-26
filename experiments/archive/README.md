# Archive experiment tracker

`experiment_tracker_2026-11-26.csv` lưu nguyên byte 27 dòng dữ liệu của tracker
trước khi chuyển sang đợt frozen encoder/MoE head D015, gồm planned, failed và
completed. Giữ nguyên status, verification, metric, notes và W&B/artifact link;
archive không đồng nghĩa các run đã được xác minh hoặc cần chạy lại.

Tracker hiện hành `../experiment_tracker.csv` giữ MoE frozen exp-018..023,
run-046..051 và original frozen exp-024..029, run-052..057 (D016), seed 42,
full, planned. Các metric chưa chạy để trống. Run ID chỉ đăng ký local,
chưa kiểm tra trùng trên W&B/storage của session khác.

Không tái sử dụng ID trong archive. Allocator hiện chỉ đọc tracker hiện hành và
artifact directory, không đọc archive; các dòng mới giữ mốc ID cao hơn toàn bộ ID cũ,
nên auto allocation bắt đầu từ run-058 hoặc cao hơn. Khi cấp ID tường minh, phải
đối chiếu cả archive, tracker, W&B và storage.

Smoke dùng `--auto-run-ids` trên filesystem/tracker chung để tránh tiêu thụ ID full.
Full dùng group frozen_head_lr hoặc frozen_original_lr không có `--auto-run-ids` để chạy ID planned;
retry dùng ID mới. Không merge archive vào tracker bằng cách ghi đè toàn bộ file.
