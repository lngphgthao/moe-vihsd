# Nghiên cứu MoE classification head

So sánh encoder dense với **head chuẩn Hugging Face** và **MoE head** trên ViANLI/ViHSD.
MoE hiện dùng MLP expert hai lớp, linear scoring và top-k selection. LSTM/BiLSTM và
router mới chưa được triển khai. Phạm vi: [THESIS_SPECIFICATION](docs/THESIS_SPECIFICATION.md).
Implementation: [ARCHITECTURE](docs/ARCHITECTURE.md). Bước tiếp theo: [PLAN](experiments/PLAN.md).

Ưu tiên từ 2026-11-24 (D013): cố định **vinai/phobert-base**, kiểm chứng original trước
khi thử các thiết kế MoE classifier. Các hướng screening multilingual bên dưới là hồ sơ
trước quyết định này; không phải bước bắt buộc của kế hoạch hiện hành.

## Chạy PhoBERT-base original

Config mới: ViANLI `configs/experiments/vianli/exp-010.yaml`; ViHSD
`configs/experiments/vihsd/exp-011.yaml`. Cả hai pin model/tokenizer cùng revision,
fine-tune toàn bộ, 7 epoch, batch 16, lr 1e-5, seed 42, CE. Max length lần lượt
256 (cả cặp premise/hypothesis) và 128. Đây là baseline phát triển, chưa phải finalist.

Chuẩn bị/khôi phục `data/processed/vianli_segmented` và `data/processed/vihsd_segmented`
trên máy chạy; loader không tự segment text gốc. Với checkout mới, chạy hai lệnh
preprocessing trong mục Dữ liệu bên dưới (Java/VnCoreNLP và quyền truy cập HF cần thiết).
Không ghi đè bản có sẵn. Bản local lịch sử còn giới hạn provenance theo `data/manifests/`.

```bash
# Smoke riêng, W&B tắt; CSV tự cấp ID mới trong cùng filesystem.
python train.py --config configs/experiments/vianli/exp-010.yaml --smoke-test --set logging.use_wandb=false
python train.py --config configs/experiments/vihsd/exp-011.yaml --smoke-test --set logging.use_wandb=false
# Full chỉ sau smoke đạt và kiểm tra segmentation/truncation train/validation.
python train.py --config configs/experiments/vianli/exp-010.yaml --no-smoke-test
python train.py --config configs/experiments/vihsd/exp-011.yaml --no-smoke-test
```

Lệnh không có `--run-id` dùng allocator CSV đang bật; smoke/full nhận ID khác nhau.
Trước khi chạy trên nhiều session, phải đối chiếu W&B/storage và chỉ định `--run-id run-NNN`
còn trống cho từng run; allocator không đồng bộ giữa hai filesystem. Không sao chép ID
cũ hoặc dùng `--set` đổi thành MoE dưới ID original.

## Group PhoBERT giai đoạn 0–1

Đã có config C1/C2/M0: ViANLI exp-012/013/014, ViHSD exp-015/016/017.
C1 dùng một MLP expert H=512; C2 một expert H=2051 khớp tổng head parameters M0;
M0 dùng 4 expert H=512, top-2, balance coefficient 0.01. Cả ba kế thừa dữ liệu,
revision và budget original; không thay implementation model. Chi tiết ở [PLAN](experiments/PLAN.md).

Ví dụ chạy ViANLI, từng bước sau khi bước trước đạt (ViHSD thay `vianli` bằng `vihsd`):

```bash
python scripts/run_group.py --group-config configs/groups/vianli_phobert_stage0_baseline.yaml --smoke-test --auto-run-ids
python scripts/run_group.py --group-config configs/groups/vianli_phobert_stage0_baseline.yaml --no-smoke-test --auto-run-ids
python scripts/run_group.py --group-config configs/groups/vianli_phobert_stage1_screening.yaml --smoke-test --auto-run-ids
python scripts/run_group.py --group-config configs/groups/vianli_phobert_stage1_screening.yaml --no-smoke-test --auto-run-ids
```

Stage 0 baseline có seed 42/123/2026; stage 1 screening có C1/C2/M0 seed 42.
Smoke dùng 1 epoch, 64 train/32 validation. Logging giữ theo config. Full dùng project kế thừa
`cmoe_vianli`/`cmoe_vihsd`. `--auto-run-ids` cấp ID mới từ tracker/artifact hiện tại cho
mỗi lần gọi, nên smoke và full không dùng chung ID; YAML group vẫn giữ nguyên ID khai báo.
Chỉ dùng chế độ này khi các run dùng chung filesystem/tracker. Nếu chạy ở nhiều session
độc lập, hãy chọn ID riêng đã đối chiếu W&B/storage cho từng session. Các group chạy tuần tự
và chưa được chạy GPU trong tác vụ setup.

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

## CI

GitHub Actions chạy bộ unittest offline trên Ubuntu/Python 3.12 cho mỗi push vào `main`
và pull request. CI kiểm tra notebook compile, cấu trúc config, artifact/tracker,
forward/backward, checkpoint, runner và overlap fixture; không tải model/dataset,
không gọi W&B và không yêu cầu GPU.

Chạy GPU thủ công bằng các notebook Kaggle/Colab; CI không tải model/dữ liệu và không
thay thế smoke hoặc full run thực tế.

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
python scripts/run_group.py --group-config configs/groups/vianli_screening.yaml --no-smoke-test --auto-run-ids
python scripts/run_group.py --group-config configs/groups/vianli_screening.yaml --no-smoke-test --set logging.use_wandb=false --auto-run-ids
python scripts/aggregate_results.py --runs-dir runs --output-dir runs/analysis
```

Tracker CSV đã bật lại: `logging.use_tracker: true` là mặc định. Train có thể cấp
ID tiếp theo từ CSV, hoặc dùng --run-id; group có ID cụ thể. Run ID dùng một lần,
retry cấp ID mới; kiểm tra trùng trên W&B/ổ lưu trữ khi phân phối nhiều máy.
Screening dùng seed 42; seed cuối chưa khóa.
`--set` có thể lặp lại và nhận key có trong resolved config; trên group, giá trị CLI
ghi đè override YAML cho mọi run. Ví dụ trên tắt W&B nhưng vẫn giữ CSV tracker bật;
hai tracker được điều khiển độc lập qua `logging.use_wandb` và `logging.use_tracker`.
`--auto-run-ids` bỏ qua ID khai báo trong YAML và cấp ID mới từ tracker cùng thư mục
artifact; cần giữ CSV tracker bật và dùng chung filesystem. Với nhiều session độc lập,
dùng ID tường minh đã đối chiếu thay vì auto. Đổi cấu hình nghiên cứu đã chạy cần experiment ID mới.

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

## Notebook chạy baseline PhoBERT trên Kaggle/Colab

Các notebook [Kaggle](notebooks/main_kaggle.ipynb), [Colab](notebooks/main_colab.ipynb)
và [Kaggle song song](notebooks/parallel_kaggle_smoke.ipynb) đều chạy cặp D013:
ViANLI exp-010 và ViHSD exp-011. Hai notebook main chạy tuần tự trên **một GPU**;
notebook parallel chạy đồng thời hai process trên **hai GPU T4**, mỗi config một GPU.
Gắn hai DatasetDict segmented dưới `DATA_SOURCE_ROOT`, push/upload checkout mới,
cài dependencies và thêm `WANDB_API_KEY` cho full run.

Mỗi phase có hai option: hai `train.py` hoặc `run_group`. Bỏ comment đúng một option
kèm assert; giữ option còn lại comment. Trước full, smoke của cả hai config phải đạt.
Smoke dùng run-020/021, W&B tắt; full seed 42 dùng run-022/025 theo
`configs/groups/phobert_original_pair_baseline.yaml`. Cell chuẩn bị tạo group runtime
dưới output từ group chính. Nếu chọn option `run_group`, `--auto-run-ids` cấp ID khác
cho smoke và full trên cùng filesystem; không giữ YAML runtime trong Git.
Notebook parallel tách mỗi group runtime thành một entry, gọi hai runner đồng thời,
chờ cả hai và báo lỗi nếu một process thất bại. Runner gốc vẫn xử lý entry tuần tự.
Nếu ID đã dùng, sửa group chính/ID smoke và các lệnh trực tiếp tương ứng trước launch.
Các group stage0 theo dataset chỉ còn seed 123/2026 để không lặp seed 42.

Aggregate ghi `runs.csv`, `summary.csv` và `summary.md` dưới `RUNS_DIR/analysis`; tracker
được copy vào output của phiên. Khi retry hoặc đổi storage/session, đối chiếu tracker,
artifacts và W&B rồi dùng ID mới. Test set chỉ chạy qua cell `evaluate.py` sau khi finalist
và seed đã khóa.

CSV dùng OS-backed FileLock có chờ tối đa 30 giây; file .lock có thể còn sau khi nhả khóa,
không xóa nó trong khi worker đang chạy. Lock này dùng cho cùng filesystem, không phải
đồng bộ giữa hai session Kaggle độc lập. Group runner thường vẫn chạy tuần tự.
Có thể tắt CSV bằng logging.use_tracker=false khi thật sự không cần, vẫn giữ W&B/artifact.

## Hồ sơ D010 — screening XLM-R và mDeBERTa trên hai session Kaggle

Đợt trước D013 chạy **original head** trước: ViANLI có exp-003 (XLM-R), exp-007
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
python scripts/run_group.py --group-config configs/groups/vianli_multilingual_screening.yaml --smoke-test
# Chạy full khi cả hai smoke đã hoàn tất, loss hữu hạn và có artifact:
python scripts/run_group.py --group-config configs/groups/vianli_multilingual_screening.yaml --no-smoke-test
```

**Session 2 — ViHSD:**

```bash
python scripts/run_group.py --group-config configs/groups/vihsd_multilingual_screening.yaml --smoke-test
python scripts/run_group.py --group-config configs/groups/vihsd_multilingual_screening.yaml --no-smoke-test
```

Smoke dùng run-010/011 ở session 1 và run-012/013 ở session 2; full dùng run-003/007
và run-008/009, nên không đụng ID giữa hai session hoặc giữa smoke/full. Các lệnh
hai-session này dùng ID tường minh; không thêm `--auto-run-ids` vì tracker của mỗi
session không đồng bộ. Không dùng
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

[Colab](notebooks/main_colab.ipynb) và [Kaggle](notebooks/main_kaggle.ipynb) cấu hình dữ liệu/
output, cung cấp lựa chọn train trực tiếp hoặc group, aggregate và lệnh evaluate có điều kiện.
Không chứa logic model riêng. Notebook clone nhánh ở biến `BRANCH`; cần push thay đổi hoặc
upload checkout mới trước khi dùng trên cloud. Lưu toàn bộ run directories và tracker khi
tải output hoặc kết thúc phiên GPU.

Lịch sử trước refactor: nhánh `refactor`, commit
`c04f25d05d8802c8a32d26eb1d6a6b53e2781adf`. Transformer-MoE, Hybrid, NLIMoE, bản thảo và
metric cũ nằm tại đó; repo hiện hành không nạp checkpoint/config cũ. Baseline Linear cũ
khác head chuẩn mới nên metric cũ không được nhập vào bảng so sánh mới.

Quy ước từ 2026-11-25: không tạo YAML riêng cho smoke. Dùng cùng config/group với
`--smoke-test` hoặc `--no-smoke-test`; giới hạn lấy từ training.smoke trong base.
Run ID dùng một lần: nếu chạy cùng group ở cả hai chế độ, cấp ID mới cho lượt sau.
Smoke trực tiếp bằng train CLI có thể tự cấp ID khi CSV bật và không truyền --run-id.
Logging giữ theo config; muốn tắt W&B dùng --set logging.use_wandb=false trên train CLI.
Giữ danh sách seed hiện có trong group, không tự thêm seed đã chạy trở lại.
