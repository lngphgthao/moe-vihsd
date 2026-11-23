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
python train.py --config configs/experiments/vianli/exp-003.yaml --run-id run-010 --smoke-test
python train.py --config configs/experiments/vianli/exp-004.yaml --run-id run-004 --no-smoke-test
python scripts/run_group.py --group-config configs/groups/vianli_screening.yaml --no-smoke-test
python scripts/aggregate_results.py --runs-dir runs --output-dir runs/analysis
```

Tracker CSV đã bật lại: `logging.use_tracker: true` là mặc định. Train có thể cấp
ID tiếp theo từ CSV, hoặc dùng --run-id; group có ID cụ thể. Run ID dùng một lần,
retry cấp ID mới; kiểm tra trùng trên W&B/ổ lưu trữ khi phân phối nhiều máy.
Screening dùng seed 42; seed cuối chưa khóa.
`--set training.head_learning_rate=0.0001` và các override khác chỉ nhận key có trong
resolved config. Đổi cấu hình nghiên cứu đã chạy cần experiment ID mới.

`runs/<run_id>/` giữ manifest, config, labels, native HF config, metrics JSONL, summary,
console, checkpoint và diagnostics. W&B và artifact local cùng liên kết run ID. Đặt `RUNS_DIR`
để lưu ra ổ khác; khi W&B lỗi, local logging vẫn tiếp tục và `wandb_sync=pending`.
Các score dùng thang 0–1; thiếu metric giữ trống.

Test chỉ dành cho finalist có config đầy đủ và `finalization` đã khóa. Không có finalist
mặc định. Khi đã chốt quyết định và seed:

```bash
python evaluate.py --run-dir runs/<run_id> --finalist-config configs/finalists/<dataset>/<experiment_id>.yaml --final-evaluation
```

Evaluator kiểm tra config/seed/provenance, dựng model từ HF config đã lưu và nạp checkpoint;
không tìm “latest run” và không dùng config nền hiện tại để dựng lại model.

## Notebook kiểm tra hai run song song trên Kaggle T4 ×2

Mở [parallel_kaggle_smoke.ipynb](notebooks/parallel_kaggle_smoke.ipynb). Đây là notebook
riêng để xác nhận hai process chia sẻ một checkout/cache/tracker nhưng dùng hai GPU,
model/optimizer và artifact riêng. Bật T4 x2, Internet và secrets HF_TOKEN/WANDB_API_KEY;
push/upload code mới trước khi chạy.

1. Setup repo/dependency, secrets và kiểm tra hai GPU.
2. Chạy test console/tracker offline, đăng ký run-016/017 rồi làm ấm cache model/dataset.
3. Launch XLM-R (exp-003) trên GPU 0, mDeBERTa (exp-007) trên GPU 1. Cả hai là
   diagnostic original-head, 2 epoch, 256 train/64 validation, batch 8, seed 42.
4. Xem GPU memory/utilization và log mỗi 10 giây; chờ cả hai kết thúc.
5. Cell xác nhận kiểm tra status, GPU mask, thời gian training chồng nhau, checkpoint,
   metric hữu hạn, hai dòng CSV đúng metadata và W&B sync nếu bật. Lưu output; ZIP tùy chọn.

PASS chỉ được xác nhận khi chạy notebook thực tế. Notebook không đánh giá test set.
Run-014/015 đã failed do lỗi Tee.isatty theo log Kaggle người dùng gửi; retry dùng
run-016/017 sau khi cập nhật code. Cell setup không tự pull checkout đã tồn tại.
Nếu ID đã dùng, đổi sang ID mới; không reset trạng thái để ghi đè. Nếu chưa có
training overlap thì chưa xác nhận chạy song song, kể cả khi hai run hoàn tất riêng lẻ.

CSV dùng OS-backed FileLock có chờ tối đa 30 giây; file .lock có thể còn sau khi nhả khóa,
không xóa nó trong khi worker đang chạy. Lock này dùng cho cùng filesystem, không phải
đồng bộ giữa hai session Kaggle độc lập. Group runner thường vẫn chạy tuần tự.
Có thể tắt CSV bằng logging.use_tracker=false khi thật sự không cần, vẫn giữ W&B/artifact.

## Screening XLM-R và mDeBERTa trên hai session Kaggle

Đợt hiện hành chạy **original head** trước: ViANLI có exp-003 (XLM-R), exp-007
(mDeBERTa); ViHSD có exp-008 (XLM-R), exp-009 (mDeBERTa). Tất cả seed 42, 7 epoch,
batch 16, lr 1e-5; chỉ validation. Chi tiết ID và protocol ở [PLAN](experiments/PLAN.md).

Dùng code/config mới đã push hoặc upload, bật GPU và Internet. Chạy setup/secrets
của notebook rồi `pip install -r requirements.txt`; dependency mới gồm SentencePiece
và Protobuf. `RUNS_DIR` nên trỏ `/kaggle/working/runs`. Mỗi session chỉ dùng một GPU;
group chạy hai backbone tuần tự. Hai GPU T4 không tự được gộp bộ nhớ.

ViHSD mới đọc text gốc từ Hub, không cần chạy segmentation. Tài khoản HF phải đã
chấp nhận điều kiện truy cập [uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd);
đặt HF_TOKEN và WANDB_API_KEY trong Kaggle Secrets. Không trỏ exp-008/009 vào bản
segmented của PhoBERT.

Chạy từ Git root; trong Python cell Kaggle thêm `!` trước các lệnh dưới đây.

**Session 1 — ViANLI:**

```bash
python scripts/run_group.py --group-config configs/groups/vianli_multilingual_smoke.yaml --smoke-test
# Chạy full khi cả hai smoke đã hoàn tất, loss hữu hạn và có artifact:
python scripts/run_group.py --group-config configs/groups/vianli_multilingual_screening.yaml --no-smoke-test
```

**Session 2 — ViHSD:**

```bash
python scripts/run_group.py --group-config configs/groups/vihsd_multilingual_smoke.yaml --smoke-test
python scripts/run_group.py --group-config configs/groups/vihsd_multilingual_screening.yaml --no-smoke-test
```

Smoke dùng run-010/011 ở session 1 và run-012/013 ở session 2; full dùng run-003/007
và run-008/009, nên không đụng ID giữa hai session hoặc giữa smoke/full. Không dùng
`--smoke-test` với group full, vì sẽ tiêu thụ ID đã dành cho full. ID chỉ dùng một lần;
retry phải cấp và đăng ký ID mới trước khi chạy.

Nếu run-003 đã hoàn thành đúng cấu hình, đối chiếu artifact để tái dùng nó và chỉ
chạy phần còn thiếu ở session 1:

```bash
python train.py --config configs/experiments/vianli/exp-007.yaml --run-id run-007 --no-smoke-test
```

Sau khi chạy, xem `summary.json`, `metrics.jsonl`, `manifest.json`, `console.log`.
Tải toàn bộ run directories và tracker CSV của **cả hai session**, giữ W&B URL.
Đối chiếu/gộp từng dòng theo run_id, không ghi đè toàn CSV bằng bản từ một session.
Không chạy evaluate test hoặc khóa finalist ở bước screening. Nếu smoke OOM, cần
điều chỉnh protocol/budget của cặp so sánh trước full, không tự hạ batch riêng một model.

## Notebook và lịch sử

[Colab](notebooks/main_colab.ipynb) và [Kaggle](notebooks/main_kaggle.ipynb) chỉ setup,
chọn config, đặt output root và gọi entry point chung. Không chứa logic train/model riêng.
Lưu toàn bộ thư mục run khi tải output hoặc kết thúc phiên GPU. Notebook clone nhánh được chọn ở biến `BRANCH`; cần push thay đổi hoặc upload
checkout mới trước khi dùng trên cloud.

Lịch sử trước refactor: nhánh `refactor`, commit
`c04f25d05d8802c8a32d26eb1d6a6b53e2781adf`. Transformer-MoE, Hybrid, NLIMoE, bản thảo và
metric cũ nằm tại đó; repo hiện hành không nạp checkpoint/config cũ. Baseline Linear cũ
khác head chuẩn mới nên metric cũ không được nhập vào bảng so sánh mới.
