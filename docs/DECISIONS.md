# Quyết định nghiên cứu và architecture

Các quyết định quan trọng được ghi theo thứ tự thời gian.

Không âm thầm sửa quyết định lịch sử. Nếu quyết định thay đổi, thêm quyết định mới
và giải thích thay đổi.

---

## D001 — Dùng nhiều backbone

Date: 2026-09-29

Quyết định:

The project will not be restricted to PhoBERT.

Candidate backbones include:

- PhoBERT
- XLM-R
- mBERT

Lý do:

The thesis investigates MoE architectures for text classification
rather than specifically optimizing PhoBERT.

---

## D002 — Đánh giá nhiều routing strategy

Date: 2026-09-29

Quyết định:

Routing strategy is treated as an experimental variable.

Top-k routing is not assumed to be the only possible routing mechanism.

---

## D003 — Result từ experiment bên ngoài cần manual verification

Date: 2026-09-29

Quyết định:

Experiments executed on Kaggle/Colab are manually recorded in
`docs/RESULTS.md`.

Lý do:

The repository does not automatically receive the execution state
from those environments.

---

## D004 — Hiệu chỉnh reproducibility sau audit

Recorded on 2026-11-12.

Use Transformers 5.5.0 for the current tensor-returning MoE encoder-layer contract.
Require PyTorch 2.6 or newer for compatible pretrained weight loading.
Offline integration tests exercise BERT, RoBERTa, and XLM-R across all model families.
Retain optional sentence-segment IDs throughout the pipeline for BERT pair inputs.

Compute macro F1 across every configured class so absent classes receive zero F1,
matching the per-class report. This corrects checkpoint scoring on incomplete subsets.
Scores from affected historical subsets require recomputation from saved predictions.

Aggregation uses saved metadata and keeps missing values explicit. Test columns use
recorded test metrics. Segmentation failures stop processing before a dataset is saved.
Existing experiment records retain their original values and provenance.

## D005 — Chuẩn bị tái triển khai NLIMoE trên XLM-R-base

Ngày ghi nhận 2026-11-13.

### Bằng chứng và giới hạn của eval_frequency

Mục 6.2 của arXiv v3 ghi `eval_frequency=400` và `batch_size=16`.
Bài chưa xác định đơn vị của frequency. Dataset card cũng chưa cung cấp
quy trình huấn luyện hoặc mã tác giả để xác nhận đơn vị. Việc diễn giải 400
optimizer step được ghi nhận là giả định triển khai của repo.

- `training.eval_frequency` là số nguyên dương, đếm số lần `optimizer.step()`
  đã thực hiện trên toàn bộ quá trình train. Bộ đếm đi tiếp qua các epoch.
- Mỗi mốc đánh giá toàn bộ validation split đang cấu hình. Số 400 xác định
  khoảng cách giữa hai lần đánh giá.
- Đánh giá thêm ở bước huấn luyện cuối nếu bước đó nằm giữa hai mốc. Khi bước
  cuối trùng mốc, chỉ đánh giá một lần. Đây là quy tắc hoàn tất run của repo.
- Giá trị thiếu hoặc `null` tiếp tục đánh giá sau mỗi epoch cho cấu hình cũ.
- Trainer hiện thực hiện một optimizer step sau mỗi batch. Với 8.012 mẫu,
  batch 16 và 7 epoch, có 3.507 step. Các mốc là 400, 800, 1.200, 1.600,
  2.000, 2.400, 2.800, 3.200 và 3.507.
- Checkpoint được chọn theo validation macro F1, giữ checkpoint đầu tiên khi
  bằng điểm. Bài chưa nêu rõ tiêu chí chọn checkpoint. Quy tắc này theo AGENTS.
- Lịch sử, checkpoint metadata, summary và W&B ghi `global_step`. Train metric
  tại mỗi mốc là số liệu tích lũy từ đầu epoch đang chạy đến mốc đó.

Nguồn được đối chiếu là [mục 6.2 của bài](https://arxiv.org/html/2406.17716v3#S6.SS2)
và [dataset card](https://huggingface.co/datasets/uitnlp/ViANLI).

### Kiến trúc và giả định còn lại

- Người dùng chọn XLM-R-base vì tài nguyên Colab/Kaggle miễn phí. Bài mô tả
  vector 1.024 chiều. Kết quả mới được mô tả là tái triển khai NLIMoE với
  XLM-R-base và có đối chiếu điều kiện backbone khi so sánh với bài.
- Dùng family độc lập `nlimoe`. Dynamic có 7 expert, threshold 0.1 + 0.1 × C,
  alpha 0.001 và beta 0.01. Top-K có 5 expert và Top-4 theo mục 6.2.
- Expert giữ Linear → Dropout(0.4) → ReLU. Classifier dùng LayerNorm → Linear
  theo công thức 10 bằng `classifier_dropout=0.0`. Khi key này thiếu, lớp
  model giữ mặc định dropout trước đây để hỗ trợ cấu hình lịch sử.
- Giữ straight-through estimator với temperature 0.02 để complexity gate
  nhận gradient. Bài chỉ viết mask cứng, chưa mô tả estimator. Đây là giả định
  triển khai có ảnh hưởng đến huấn luyện.
- Giữ fallback chọn expert xác suất cao nhất khi mask rỗng để tổng hợp hữu hạn.
  Bài chưa xác định hành vi này. Phân phối đều trên 7 expert với C=0.5 tạo mask
  rỗng vì 1/7 nhỏ hơn ngưỡng 0.15.
- Top-K tiếp tục tắt dynamic loss và giữ balance loss. Bài chưa mô tả riêng
  đầy đủ loss của Top-K. Giữ lựa chọn hiện có và ghi nhận giới hạn này.
- Giữ AdamW, gradient clipping 1.0 và learning rate cố định của trainer.
  Những lựa chọn ngoài thông số được ghi trong bài tiếp tục là giả định.
- `auxiliary_loss` giữ tổng loss thô để tương thích artifact. Trường mới
  `weighted_auxiliary_loss` ghi đóng góp sau khi nhân hệ số vào total loss.

### Dữ liệu và khả năng tái lập

Hai cấu hình NLIMoE cố định dataset revision
`0fec8d6ecb043a61c609f9b51f80401fdf1e84d3` và model/tokenizer revision
`e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, đối chiếu từ Hugging Face API
ngày 2026-11-13. Đây là revision được chọn cho lần triển khai này.

Loader yêu cầu các split chính thức train/validation/test và kiểm tra số mẫu
8.012/1.000/1.000 sau khi lọc dữ liệu thiếu. Các giới hạn của smoke test được
áp dụng sau bước kiểm tra dữ liệu gốc. Đầu vào XLM-R giữ văn bản thô và native
pair tokenization. Cấu hình đã resolve lưu các revision và lịch đánh giá.

Các thay đổi này cần run mới để kiểm chứng thực nghiệm. Unit/integration test
offline xác minh luồng tính toán và artifact. Điểm benchmark cần huấn luyện
và đánh giá riêng sau khi khóa finalist.

## D006 — Version hóa tài liệu quản trị và manifest dữ liệu

Ngày ghi nhận 2026-11-13.

`AGENTS.md`, tài liệu trong `docs/`, kế hoạch và ghi chú trong `experiments/`, cùng
manifest trong `data/manifests/` là nguồn cần có trong một clone mới. `.gitignore`
chỉ loại dữ liệu, cache, checkpoint, run output và credentials; không loại quy tắc
đóng góp hoặc provenance. Dataset đã chuẩn bị vẫn ở ngoài Git, còn manifest ghi
nguồn, revision, schema, preprocessing và giới hạn kiểm chứng.

Quyết định này không di chuyển cấu trúc implementation hoặc đổi schema artifact hiện
tại. Những thay đổi đó vẫn phải đi qua một đợt refactor riêng như đã ghi trong
`AGENTS.md` và `docs/ARCHITECTURE.md`.
