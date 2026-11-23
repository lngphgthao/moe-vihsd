# Architecture hiện hành

Cập nhật sau refactor ngày 2026-11-22. Chỉ có dense encoder và hai loại head.

## Cấu trúc

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

Tracker khóa ghi bằng file lock độc quyền, thay file CSV nguyên tử. Run ID có dạng run-NNN;
run-001/002 dành cho lịch sử. Group liệt kê config/seed/run_id rõ ràng và đăng ký tất cả
trước khi chạy. Mỗi retry dùng ID mới; mkdir không cho ghi đè run directory.
Đây là khóa trên một filesystem, không thay thế tracker chung khi phân phối nhiều máy.
Nếu tiến trình bị kill và để lại file lock, kiểm tra không còn writer trước khi xóa lock.

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
base_config có thể thay đổi. Kết quả test ghi local và tracker; W&B summary cập nhật khi có linkage online.

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
