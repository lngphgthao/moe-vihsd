# Cấu hình

Base chứa mặc định; experiments chứa overlay có ID; groups khai báo từng config, seed,
run_id; diagnostics chứa mục đích và giới hạn mẫu. Dùng base_config tương đối theo file.
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
đọc/ghi CSV. logging.use_wandb độc lập với CSV. Hai diagnostic parallel dùng batch 8,
2 epoch, 256 train/64 validation và bật cả CSV/W&B; xem notebook parallel_kaggle_smoke.
