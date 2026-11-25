# Cấu hình

Giai đoạn PhoBERT-base 0–1: original exp-010 (ViANLI)/exp-011 (ViHSD);
C1/C2/M0 lần lượt exp-012/013/014 và exp-015/016/017. M0 kế thừa original,
C1 kế thừa M0 rồi đổi E/k/coef; C2 kế thừa C1 rồi đổi H=2051 để khớp tổng head
parameters. Giữ revision, dataset segmented, pooling/dropout/budget giữa các control.
Project W&B kế thừa base người dùng: cmoe_vianli/cmoe_vihsd.

Mỗi dataset giữ hai group chính: `<dataset>_phobert_stage0_baseline.yaml` và
`<dataset>_phobert_stage1_screening.yaml`. Không có YAML dành riêng cho smoke.
Dùng cùng experiment/group với `--smoke-test` hoặc `--no-smoke-test`;
giới hạn smoke nằm trong training.smoke của base. Logging không tự thay đổi theo cờ.
Run ID dùng một lần: khi đổi profile để chạy lượt mới, cấp run ID mới.
Diagnostic tiny-overfit vẫn giữ vì có mục tiêu và sampling khác smoke.
Xem bảng experiment và các gate tại experiments/PLAN.md.

Base chứa mặc định; experiments chứa overlay có ID; groups khai báo từng config, seed,
run_id; diagnostics chứa mục đích và giới hạn mẫu. Dùng base_config tương đối theo file.
Khi cần chạy lại group ở nhiều profile trên cùng filesystem, truyền `--auto-run-ids`:
runner bỏ qua ID trong YAML và cấp ID mới từ tracker/artifact cho từng run; YAML gốc
không bị sửa. Cờ này cần CSV tracker bật. Với nhiều session/filesystem độc lập, giữ
ID tường minh đã đối chiếu W&B/storage vì allocator cục bộ không đồng bộ giữa máy.
Group chạy với ID tường minh trùng artifact sẽ dừng thay vì ghi đè.
Model chỉ nhận backbone/revision/freeze_backbone và head.type original hoặc moe.
MoE tách router.scoring=linear, router.selection=top_k, router.top_k khỏi expert settings.
Trainer nhận routing.balance_loss_coef; head trả balance_loss thô.

Chưa có finalist. Khi khóa, tạo configs/finalists/<dataset>/exp-NNN.yaml là config đầy đủ
của experiment, thêm finalization: {locked: true, locked_at: <ngày>, selection_reason: <lý do>,
seeds: [<tập seed đã khóa>]}. Không thêm CLI override nghiên cứu sau khi khóa.
Fingerprint dataset/model/training/routing phải khớp run; seed phải nằm trong tập khóa.
Không dùng lại config hay alias family lịch sử.


Logging hiện hành: logging.use_tracker=true mặc định theo D012; CSV có lock OS
chờ tối đa 30 giây. Có thể tắt rõ ràng bằng false; lúc đó train cần --run-id và không
đọc/ghi CSV. logging.use_wandb độc lập với CSV. Các notebook Kaggle/Colab cung cấp hai
cách chạy cùng cặp config (hai lệnh train riêng hoặc run_group), tuần tự trên một GPU;
smoke tắt W&B, full dùng thiết lập W&B của experiment.
