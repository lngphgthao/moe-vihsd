# Architecture hiện hành

Cập nhật sau refactor ngày 2026-11-22. Chỉ có dense encoder và hai loại head.

## Frozen-head screening — 2026-11-26, D015

Theo D016, bổ sung baseline original frozen exp-024..029 và hai group
`<dataset>_phobert_frozen_original_lr.yaml`. Tái dùng OriginalClassifier: freeze
model.base_model, giữ native HF classification head trainable; không có router.
LR classifier 1e-4/3e-4/1e-3 với cùng 50 epoch/seed 42/batch 16 như MoE.
Không đổi implementation hoặc chính sách dropout. Tracker thêm planned run-052..057.
Kiểm chứng D016: resolve inheritance YAML của 6 baseline, đối chiếu head original,
freeze/revision/data/budget/LR/smoke với MoE và bảo toàn 6 dòng tracker cũ đều đạt.
Kiểm tra forward/backward riêng bị dừng do tiến trình không trả output; không ghi
nhận là đạt. Chưa chạy GPU, pretrained full run hoặc đối chiếu W&B từ xa.

Thêm exp-018..023 và hai group `<dataset>_phobert_frozen_head_lr.yaml`; tái dùng
freeze_backbone và optimizer groups hiện hành, không đổi source implementation.
Encoder pretrained không nhận gradient, optimizer chỉ chứa head/router. Freeze
không ép encoder eval: dropout encoder vẫn theo model.train()/eval(). Train 50
epoch, validation mỗi epoch và nạp lại best checkpoint bằng macro F1 sẵn có.

Kiểm chứng CPU/offline: cả 6 config resolve/validate đúng, giữ M0/data/revision,
LR và epoch đúng từng config, smoke còn 1 epoch/64 train/32 validation. Suite chạy
52 test; các test head frozen/backward, optimizer, group và best checkpoint đạt.
Toàn suite chưa đạt: failures=2, errors=6 trong test_parallel_notebook do thiếu
parallel_kaggle_smoke.ipynb, thiếu cell được test tìm bằng prefix và CUDA mask
không khớp notebook hiện tại. Không sửa notebook/test ngoài phạm vi trong đợt này.
Diff check đạt. Chưa chạy smoke GPU/full run hoặc đối chiếu W&B/storage từ xa.

## Smoke qua CLI — 2026-11-25

Theo yêu cầu người dùng, không giữ YAML dành riêng cho smoke. Experiment và group
chính dùng chung cho `--smoke-test`/`--no-smoke-test`; `training.smoke` trong base
giữ giới hạn diagnostic. Cờ chỉ chọn profile, không tự tắt W&B. Run ID vẫn dùng một
lần, kể cả khi đổi giữa smoke/full. Các mô tả file smoke ở mục kiểm chứng trước đây
là lịch sử setup; không phải config hiện hành. Diagnostic tiny-overfit vẫn giữ riêng.

Với group chạy nhiều profile trên cùng filesystem, `run_group.py --auto-run-ids`
bỏ qua các ID khai báo trong YAML và cấp ID mới theo CSV tracker/artifact. Đây là
opt-in; group YAML giữ nguyên ID. CSV tracker phải bật. Các session/filesystem độc lập
vẫn cần ID tường minh được đối chiếu chung trên W&B/storage.

Notebook baseline hiện tại dùng group `phobert_original_pair_baseline.yaml` cho cả
hai chế độ qua group runtime có ID riêng cho smoke/full. Hai notebook main chạy
tuần tự trên một GPU; parallel chạy hai process trên hai GPU, mỗi group một entry.
Runner gốc vẫn tuần tự. Test config kiểm tra cùng
overlay ở cả hai profile và giữ danh sách seed hiện có trong group baseline.

Kiểm chứng đợt dọn: không còn YAML smoke; cú pháp YAML/Python, đường dẫn kế thừa/
group và hai profile của 8 experiment PhoBERT đạt. Lượt full CPU/offline chạy ngoài
sandbox đạt 47/48 test, một test notebook có 3 subtest lỗi do helper bỏ dòng shell
IPython làm rỗng suite của if. Đã sửa helper giữ indentation bằng pass, kiểm tra
contract trên source gốc; sau sửa 3 test notebook chạy cô lập đạt (dùng function
config/profile gốc, không import encoder). Chưa chạy lại toàn bộ suite sau sửa helper.
Không chạy notebook trên GPU hoặc nghiên cứu mới trong tác vụ dọn config.

## Config PhoBERT-base và kiểm chứng — 2026-11-24

Theo D013, exp-010/011 dùng original PhoBERT-base cho ViANLI/ViHSD, cùng revision
model/tokenizer `01daacda68afe13d83023d16ec647239e344a1e6`, DatasetDict local segmented.
Không thay đổi implementation, công thức MoE/router hoặc các config đã đăng ký trước.
Pool/activation/selection mới trong PLAN chỉ là thiết kế đề xuất nếu ghi cần triển khai.

Kiểm chứng local CPU/offline bằng `.venv`:
- Resolve hai overlay, validate head, model/tokenizer revision, budget và smoke profile
  1 epoch/64 train/32 validation đều đạt; không gọi training pretrained.
- Schema và split sizes train/validation khớp config: ViANLI 8012/1000, ViHSD
  24048/2672. Không có text null ở các field đầu vào. ViHSD train có 2 chuỗi rỗng;
  loader hiện chỉ lọc null nên vẫn giữ 24048 example, khác bản Hub raw lọc còn 24046.
  Không tự xóa mẫu hoặc sửa bản dữ liệu lịch sử trong tác vụ cấu hình.
- Bộ `python -m unittest discover -s tests -v` với OMP_NUM_THREADS=MKL_NUM_THREADS=2
  chạy 46 test: 45 đạt, 1 fail ở `test_main_notebooks_leave_offline_suite_to_ci` vì
  `main_kaggle.ipynb` thiếu chuỗi `GitHub Actions`. Notebook/test liên quan không đổi
  so với HEAD trong tác vụ này; không ghi nhận toàn bộ suite là đạt.
- `git diff --check` đạt. Chưa đo truncation, tải weights/tokenizer thật cho config mới,
  smoke GPU, W&B online hoặc full run. Pin model không xác minh segmentation lịch sử.

## Cấu trúc

Setup giai đoạn 0–1 ngày 2026-11-25 (D014): thêm config C1/C2/M0, hai diagnostic
original và tám group theo dataset. Tái sử dụng factory/MoEHead hiện hành, không đổi
implementation. Config MoE kế thừa original cùng dataset; C1 kế thừa M0, C2 kế thừa
C1. W&B project giữ cmoe_vianli/cmoe_vihsd theo base người dùng đã sửa.

Test notebook đã bỏ yêu cầu markdown có chữ `GitHub Actions`; thay bằng kiểm tra
code notebook không tự chạy unittest và workflow CI thực sự chạy suite offline.
Test giai đoạn mới kiểm tra config/profile/seed/ID group và forward/backward/count
tham số head C1/C2/M0 trên CPU, không tải encoder hoặc dữ liệu từ mạng.

Kiểm chứng 2026-11-25: tái hiện lỗi notebook trước sửa; sau sửa 7 test tập trung đạt,
toàn bộ 49 test CPU/offline đạt bằng `.venv` (OMP_NUM_THREADS=MKL_NUM_THREADS=2;
HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1, WANDB_MODE=offline).
Forward/backward và count xác nhận C1=790788, C2=M0=3156231 head parameters.
Diff check đạt; src/notebooks/tracker CSV không đổi trong tác vụ giai đoạn 0–1.
Chưa chạy smoke GPU, pretrained full run hoặc đối chiếu run ID trên W&B/storage từ xa.

```text
train.py / evaluate.py           CLI điều phối
src/
  data/loader.py                DatasetDict/Hub, nhãn, tokenizer, DataLoader
  models/
    factory.py                  Kiểm tra head type và dựng model
    backbone.py                 Tải AutoModel hoặc dựng từ HF config đã lưu
    classifier.py               Dense encoder + pooling + MoE head
    heads/original.py           Wrapper AutoModelForSequenceClassification
    heads/moe.py                MLP experts và kết hợp output theo example
    heads/router.py             Linear scoring, top-k, raw balance loss
  training/
    engine.py                   Train/validation, optimizer và vòng đời run
    losses.py                   Cross entropy, weighted CE, focal
    diagnostics.py              Routing theo example và class
  evaluation/metrics.py         Accuracy/F1/report/confusion matrix
  analysis/aggregate.py         Bảng run và mean/sample SD theo config
  utils/config.py               YAML inheritance, override, seed
  utils/artifacts.py            Manifest, JSONL, snapshot source, tracker lock
configs/                        base, experiments, groups, diagnostics
scripts/                        Runner, preprocessing, labels, aggregate, W&B export
notebooks/                      Colab/Kaggle setup và CLI
experiments/                    PLAN, RESULTS, tracker và template ghi chú
data/manifests/                 Provenance và checksum; dữ liệu ngoài Git
runs/<run_id>/                  Artifact ngoài Git
```

Chỉ tạo notes và finalist khi có nội dung thật. Module package có __init__.py cần thiết.

## Model và loss

Batch gồm input_ids, attention_mask và token_type_ids tùy tokenizer. Model trả
(logits, aux_dict). Factory không nhận architecture/alias lịch sử.

Original giữ native head của backbone, bao gồm pooling/dropout theo implementation
Hugging Face. MoE dùng CLS hoặc masked mean của AutoModel; mỗi expert là
Linear(d,h) → activation → dropout → Linear(h,d). Weighted sum sau top-k có thể thêm
shared expert và residual, rồi classifier dropout và Linear(d,num_labels).
Không thêm LayerNorm trong bước refactor vì source MoE gốc không có phép tính đó.

Router hiện linear scoring + softmax, chọn top-k và chuẩn hóa trọng số trong tập được
chọn. Với E expert: importance là mean xác suất theo batch, load là assignment count
chia tổng B*k. Balance loss giữ công thức cũ:
E*sum(importance*load) + E*(MSE(importance,uniform)+MSE(load,uniform)).
Đây là công thức của implementation hiện hành, không phải tuyên bố tái lập NLIMoE.
Trainer tính classification_loss + balance_loss_coef*balance_loss, hệ số đúng một lần.
Original trả auxiliary rỗng và balance loss bằng 0 ở trainer.

Nhóm optimizer phân biệt encoder/head/router bằng ownership tham số; router được ưu tiên.
Thông số được chia một lần, bỏ tham số frozen. head/router learning rate null dùng lr chung.
Freeze backbone giữ nguyên chính sách dropout train/eval hiện hành; không tự ép encoder eval.

## Dữ liệu và checkpoint

Loader dùng nguồn chính xác trong config: local DatasetDict thiếu thì báo lỗi, không
fallback sang một bản ViHSD khác. Giữ riêng train/validation/test, kiểm tra split sizes
nếu khai báo; validation holdout chỉ dùng khi config không yêu cầu split chính thức.
Nhãn string map theo tên, nhãn integer map theo index với display names riêng.
Macro F1 tính trên toàn bộ label list cấu hình.

Config định nghĩa backbone/revision/freeze_backbone và head; tokenizer dùng backbone,
tokenizer_revision được khai báo riêng hoặc lấy revision của model. Preprocessing PhoBERT
phải được chọn chủ động. Revision null được giữ là giới hạn provenance nếu không xác định.

Checkpoint dựng lại từ resolved_config.yaml + hf_config.json và weights đã lưu; evaluator
không tải lại pretrained weights. Tokenizer/dataset vẫn cần cache hoặc quyền truy cập nguồn.
Evaluator kiểm tra mapping và fingerprint dữ liệu/tokenizer so với run huấn luyện.

## Vòng đời run và artifact

Theo D012, logging.use_tracker=true là mặc định. Train/group đăng ký run planned,
manifest chuyển running, engine cập nhật best validation và completed/failed; evaluate
chỉ bổ sung test metrics sau khi khóa finalist. update_tracker chuẩn hóa dataset dict
thành dataset ID và model thành backbone/architecture/variant trước khi ghi cột scalar.

Tracker đọc-sửa-ghi dưới FileLock dùng khóa OS, chờ tối đa 30 giây; CSV được flush/fsync
và thay nguyên tử. File .lock có thể tồn tại sau khi nhả khóa, không phải bằng chứng còn
writer; không xóa file trong lúc worker chạy. Scope là cùng filesystem local, không
đồng bộ hai máy hoặc hai session có tracker riêng. Opt-out use_tracker=false vẫn giữ
hành vi không đọc/ghi CSV, khi đó train cần --run-id rõ ràng.

Hai process có thể dùng cùng checkout/cache/RUNS_DIR nếu ID khác nhau và CUDA_VISIBLE_DEVICES
riêng. Phiên bản cũ của notebook parallel_kaggle_smoke dùng run-016/017 để retry run-014/015 failed,
với diagnostic profile riêng; group
runner thường vẫn tuần tự. Manifest hardware lưu GPU mask, số GPU nhìn thấy và PID.
training_started_at/training_finished_at bao gồm vòng train/validation, phục vụ kiểm tra
overlap; không dùng như benchmark GPU kernel time hoặc hiệu quả chạy riêng từng model.

Console dùng Tee để nhân bản text vào console.log và lọc credentials. Các thuộc tính/
phương thức stream khác (isatty, encoding, fileno) được chuyển sang stream gốc để
thư viện progress/download hoạt động. Context manager luôn khôi phục stdout/stderr
khi thoát, kể cả khi có exception. Notebook kiểm tra console và tracker trước launch.

```text
runs/run-NNN/
  manifest.json
  resolved_config.yaml
  labels.json
  hf_config.json
  metrics.jsonl
  summary.json
  console.log
  source/                       nếu working tree bẩn
  checkpoints/best.safetensors
  evaluation/                   test/predictions/confusion matrix sau khi khóa
  diagnostics/validation.json   khi bật diagnostics
```

Manifest lưu config nguồn, CLI override, seed/profile, commit/dirty state, source snapshot,
package/platform/hardware, model revision biết được, dataset/tokenizer fingerprints,
parameter count, optimizer groups, artifact URI và W&B linkage. Giá trị chưa biết là null.
Snapshot chỉ chứa source Python và requirements; resolved config là bản config phục hồi.
Secrets từ environment không được ghi vào config/log/source snapshot.

Validation theo epoch hoặc optimizer step toàn cục; mỗi lần validation ghi train/validation
metric JSONL và cập nhật checkpoint khi macro F1 tăng. Sau training nạp lại best checkpoint.
Summary ghi best epoch/step, metric chọn, checkpoint và chi phí đo với đơn vị/phạm vi.
Lỗi giữ log/artifact, cập nhật failed và kết thúc W&B phù hợp. status, verification và
wandb_sync tách riêng. W&B lỗi vẫn giữ local run; thư mục W&B offline cần sync riêng khi có mạng.

Evaluate yêu cầu run-dir, finalist-config và final-evaluation. Chỉ nhận completed/full,
config fingerprint khớp và seed nằm trong tập đã khóa cùng ngày/lý do khóa. Không có
latest-run fallback, không ghi đè test đã có. Finalist lưu config đầy đủ, không phụ thuộc
base_config có thể thay đổi. Kết quả test ghi local; W&B summary cập nhật khi có linkage online. CSV cập nhật khi resolved config của run bật logging.use_tracker (mặc định).

Aggregation chỉ đọc schema artifact mới, giữ validation/test riêng và missing là missing.
Mean/sample SD nhóm theo experiment và fingerprint config (không gồm seed), chỉ full
completed, từ chối hai completed run cùng seed trong một nhóm để tránh đếm retry hai lần.
Verification vẫn cần đối chiếu; summary thống kê không tự chứng nhận run đã verified.

## Bản đồ chuyển đổi và lịch sử

| Trước | Hiện hành |
|---|---|
| dense.py: AutoModel + Linear | heads/original.py: native classification head, cần run mới |
| classifier_moe.py | classifier.py + heads/moe.py + heads/router.py |
| MoE trong Transformer, Hybrid, NLIMoE | bỏ khỏi nhánh hiện hành, xem bản lưu |
| train.py chứa train/validation/diagnostics | training/engine.py và training/diagnostics.py |
| configs/vianli.yaml, vihsd.yaml và overlay cũ | configs/base, experiments theo exp-NNN |
| hai group dense trùng nhau | group screening có config/seed/run_id cụ thể |
| checkpoints + results | runs/<run_id> |
| data/*_segmented | data/processed/*_segmented, giữ byte checksum và gỡ theo dõi Git |
| EXPERIMENT_PLAN, nhiều báo cáo/task/bản thảo cũ | PLAN và tài liệu scope hiện hành |
| notebook theo dataset | hai notebook theo môi trường dùng CLI chung |

Nguồn khôi phục: refactor@c04f25d05d8802c8a32d26eb1d6a6b53e2781adf. Source, tài liệu,
tracker, ghi chú và bản thảo có giá trị trước dọn đã được lưu vào nhánh đó. Không rewrite
lịch sử Git; runtime results lịch sử còn trên máy và không được aggregate vào scope mới.

## Kiểm chứng

Ngày 2026-11-22, môi trường local Windows/Python 3.13.3, PyTorch 2.14.0+cpu,
Transformers 5.5.0, datasets 5.0.1, VnCoreNLP wrapper 1.0.3:

- Toàn bộ 29 test offline qua unittest discover đạt, gồm native-head parity trên
  BERT/RoBERTa/XLM-R, backward/padding/freeze, checkpoint, train/evaluate, group/tracker,
  loss coefficient, diagnostics, W&B lỗi, redaction và CLI help với code page Windows.
- So với head và router tại refactor@c04f25d, sáu cấu hình MoE (GELU/ReLU/SiLU,
  shared/residual bật hoặc tắt) có state dict, logits, raw balance loss và gradients
  bằng nhau chính xác khi dùng cùng seed/input, kể cả dropout trong chế độ train.
- pip check không có dependency xung đột; compileall và git diff --check đạt.
- Đã xác minh SHA-256 của 10 file ViANLI và 19 file ViHSD sau di chuyển; load lại
  DatasetDict kiểm tra split/label counts khớp. Dữ liệu không còn trong Git index.
- Code cell hai notebook biên dịch được. Log local tại runs/refactor_verification/,
  ngoài Git; script parity lưu tại đó có thể chạy từ Git root.

Chưa chạy notebook trên cloud, VnCoreNLP Java service, smoke GPU, full experiment hay
W&B online. Các test dùng dữ liệu tổng hợp; không phải kết quả luận văn. Recipe
segmentation lịch sử và revision PhoBERT/tokenizer chưa xác minh vẫn ghi giới hạn.


## Bổ sung screening multilingual backbone — 2026-11-23

exp-003/007 và exp-008/009 so sánh original head XLM-R/mDeBERTa trên ViANLI/ViHSD.
Các group multilingual_smoke và multilingual_screening dùng ID riêng cho từng session;
không thêm contract mới vào group runner, trainer hoặc model factory. mDeBERTa sử dụng
model_type deberta-v2 qua AutoModelForSequenceClassification và giữ native head.

Pin revision tokenizer độc lập cho từng model, vì base ViANLI đang pin tokenizer XLM-R.
ViHSD override cả dataset.name/revision sang Hub text gốc, tránh dùng segmentation
PhoBERT từ base. Chỉ bổ sung SentencePiece/Protobuf vào dependency cho tokenizer SPM.
Kiểm chứng CPU/tokenizer của đợt setup được ghi bên dưới; chưa có smoke GPU.


Kiểm chứng setup ngày 2026-11-23 trên CPU, Transformers 5.5.0:

- Toàn bộ 31 test offline đạt, gồm hai test config/group mới và subtest DeBERTa-v2
  native head: logits parity, backward với segment IDs, save/load từ HF config.
- Tokenizer mDeBERTa thật tại revision a0484667b22365f84929a935b5e50a51f71f159d
  đã tải và encode text đơn/cặp NLI có padding; keys gồm input_ids, attention_mask,
  token_type_ids. Không tải pretrained model weights. Vocabulary tokenizer 250101
  nhỏ hơn embedding vocabulary 251000 là hợp lệ; tất cả token ID kiểm tra nằm trong
  giới hạn embedding. Không resize embedding theo kích thước tokenizer.
- SentencePiece 0.2.3 và Protobuf 7.36.1 có trong môi trường kiểm tra; pip check
  không phát hiện dependency xung đột. Compileall và diff check đạt.
- Log và metadata tokenizer ở runs/backbone_setup_verification/, ngoài Git.
- Chưa tải ViHSD raw có kiểm soát truy cập trong tác vụ này; source revision đối chiếu
  từ download metadata lịch sử. Runtime sẽ kiểm tra schema/split sizes. Chưa kiểm tra
  tải weights thật, chạy GPU, W&B online hoặc đánh giá test/finalist.


## Kiểm chứng tắt CSV — 2026-11-23

34 test offline đạt. Regression mới khởi tạo hai artifact đồng thời cùng checkout,
giữ nguyên CSV giả lập và một lock đã có; default group không đăng ký/cập nhật CSV.
Luồng train -> checkpoint -> final evaluation với dữ liệu tổng hợp chạy được khi tắt
CSV và không tạo CSV/lock. Các test tính năng CSV cũ chỉ bật trên file tạm riêng.
Checksum SHA-256 của experiments/experiment_tracker.csv giữ nguyên trước/sau toàn bộ
tác vụ; notebook cells biên dịch được, compileall và diff check đạt. Log kiểm chứng
ở runs/tracker_disabled_verification/, ngoài Git. Chưa kiểm chứng parallel GPU hoặc
W&B online trong tác vụ này.

## Kiểm chứng bật lại tracker và notebook song song — 2026-11-23

Tracker CSV mặc định bật lại; FileLock tuần tự hóa cập nhật giữa các process trên
cùng filesystem. Notebook `parallel_kaggle_smoke.ipynb` dành cho hai GPU riêng,
run-014/015 là diagnostic đã đăng ký, chưa phải kết quả thực nghiệm.

Kiểm chứng CPU: bộ 37 test trước khi thêm test notebook đạt; 5 test notebook
riêng đạt; 3 test tracker riêng đạt. Lượt toàn bộ 42 test có một lỗi timeout chờ
worker khởi động. Đã chuyển readiness barrier ra ngoài khoảng giữ lock để thời
gian khởi động không tiêu thụ lock timeout; lượt chạy lại chưa có kết quả trong
môi trường Windows hiện tại. Không ghi nhận toàn bộ 42 test là đạt. Diff check đạt.
Log đã có được giữ ở `runs/parallel_setup_verification/`, ngoài Git.

Chưa chạy notebook trên Kaggle, chưa kiểm chứng GPU hoặc W&B online. Chỉ xác nhận
hai run song song sau khi cell cuối tạo `confirmation.json` với `passed: true`;
điều kiện gồm mask GPU riêng, checkpoint riêng, tracker khớp và thời gian train/
validation chồng nhau. Đây không phải kiểm chứng chất lượng mô hình hoặc test set.

## Sửa console stream sau lỗi Kaggle — 2026-11-23

Theo log người dùng cung cấp, run-014/015 thoát mã 1 với AttributeError vì Tee
thiếu isatty; chưa có bằng chứng training GPU thành công. Cell xác nhận báo false
là đúng. Tracker ghi failed/unverified theo nguồn log này, không nhập metric.
Retry đã đăng ký run-016/017, giữ nguyên experiment và diagnostic config.

Regression tái hiện đúng AttributeError trước sửa. Sau khi chuyển các truy vấn
stream sang original, cả 44 test CPU/offline đạt với interpreter .venv và
OMP_NUM_THREADS=MKL_NUM_THREADS=2. Test mới kiểm tra terminal/non-terminal,
encoding/errors/fileno, progress bar, lọc secret và khôi phục stdout/stderr khi lỗi.
Notebook biên dịch được và chạy preflight console/tracker trước launch. Diff check
đạt; log ở runs/parallel_setup_verification/tee_full_tests.log. Đây là kiểm chứng
CPU; chưa chạy retry trên Kaggle, pretrained download hoặc W&B online.
