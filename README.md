# Nghiên cứu MoE classification head

So sánh encoder dense với **head chuẩn Hugging Face** và **MoE head** trên ViANLI/ViHSD.
MoE hiện dùng MLP expert hai lớp, linear scoring và top-k selection. LSTM/BiLSTM và
router mới chưa được triển khai. Phạm vi: [THESIS_SPECIFICATION](docs/THESIS_SPECIFICATION.md).
Implementation: [ARCHITECTURE](docs/ARCHITECTURE.md). Bước tiếp theo: [PLAN](experiments/PLAN.md).

## Môi trường

Python >=3.10, PyTorch >=2.6; cài dependency theo requirements. Java >=8 và VnCoreNLP
cần cho word segmentation. Giữ pin Transformers 5.5.0 trong requirements; xem phiên bản
thực tế đã kiểm tra trong ARCHITECTURE trước khi so sánh môi trường.

```bash
python -m pip install -r requirements.txt
hf auth login
wandb login
python -m unittest discover -s tests -v
```

Credentials lấy qua `HF_TOKEN`, `WANDB_API_KEY` hoặc secrets của nền tảng. Không commit `.env`.
Không nâng dependency cùng lúc với thay đổi phương pháp nếu chưa kiểm chứng lại.

## Dữ liệu

XLM-R/mBERT dùng text gốc; PhoBERT cần bản segmented tương ứng. Config ViANLI mẫu
chạy XLM-R từ Hub có revision; config ViHSD mẫu chạy PhoBERT với DatasetDict local.
Không dùng segmentation của PhoBERT ngầm định cho mọi backbone.

```bash
python scripts/segment_vianli.py --output-dir data/processed/vianli_segmented
python scripts/preprocess_vihsd.py --output-dir data/processed/vihsd_segmented
python scripts/count_labels.py --config configs/base/vihsd.yaml --json
```

Script từ chối ghi đè dataset đã tồn tại. Máy hiện tại đã giữ hai bản segmented tại
`data/processed/`; clone mới cần chạy preprocessing hoặc khôi phục từ storage bên ngoài.
[Manifest dữ liệu](data/manifests/) ghi checksum và giới hạn provenance của bản lịch sử.

## Chạy

```bash
python train.py --config configs/experiments/vianli/exp-003.yaml --smoke-test
python train.py --config configs/experiments/vianli/exp-004.yaml --no-smoke-test
python scripts/run_group.py --group-config configs/groups/vianli_screening.yaml --no-smoke-test
python scripts/aggregate_results.py --runs-dir runs --output-dir runs/analysis
```

CLI cấp ID kế tiếp nếu không truyền `--run-id`; group dùng ID đã đăng ký rõ ràng.
Run ID dùng một lần, retry cấp ID mới. Điều phối bằng tracker chung trước khi phân
phối nhiều máy. Screening dùng seed 42; seed cuối chưa khóa.
`--set training.head_learning_rate=0.0001` và các override khác chỉ nhận key có trong
resolved config. Đổi cấu hình nghiên cứu đã chạy cần experiment ID mới.

`runs/<run_id>/` giữ manifest, config, labels, native HF config, metrics JSONL, summary,
console, checkpoint và diagnostics. W&B và tracker cùng liên kết run ID. Đặt `RUNS_DIR`
để lưu ra ổ khác; khi W&B lỗi, local logging vẫn tiếp tục và `wandb_sync=pending`.
Các score dùng thang 0–1; thiếu metric giữ trống.

Test chỉ dành cho finalist có config đầy đủ và `finalization` đã khóa. Không có finalist
mặc định. Khi đã chốt quyết định và seed:

```bash
python evaluate.py --run-dir runs/<run_id> --finalist-config configs/finalists/<dataset>/<experiment_id>.yaml --final-evaluation
```

Evaluator kiểm tra config/seed/provenance, dựng model từ HF config đã lưu và nạp checkpoint;
không tìm “latest run” và không dùng config nền hiện tại để dựng lại model.

## Notebook và lịch sử

[Colab](notebooks/main_colab.ipynb) và [Kaggle](notebooks/main_kaggle.ipynb) chỉ setup,
chọn config, đặt output root và gọi entry point chung. Không chứa logic train/model riêng.
Lưu toàn bộ thư mục run khi tải output hoặc kết thúc phiên GPU. Notebook clone nhánh
`classifier-main`; cần push thay đổi hoặc upload checkout mới trước khi dùng trên cloud.

Lịch sử trước refactor: nhánh `refactor`, commit
`c04f25d05d8802c8a32d26eb1d6a6b53e2781adf`. Transformer-MoE, Hybrid, NLIMoE, bản thảo và
metric cũ nằm tại đó; repo hiện hành không nạp checkpoint/config cũ. Baseline Linear cũ
khác head chuẩn mới nên metric cũ không được nhập vào bảng so sánh mới.
