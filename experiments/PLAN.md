# Kế hoạch thực nghiệm hiện hành

## Kế hoạch PhoBERT-base classifier — 2026-11-24, D013

Người dùng chốt `vinai/phobert-base` cho cả ViANLI/ViHSD. Các mục 2026-11-23 phía
dưới được giữ làm hồ sơ, đã được kế hoạch này thay thế về thứ tự ưu tiên. Không tiếp
tục tìm backbone tốt nhất. Không có run GPU mới, finalist hoặc seed xác nhận đã khóa.

### Câu hỏi và bằng chứng cần thu

| Câu hỏi | Bằng chứng cần có | Điều chưa được suy ra |
|---|---|---|
| C1: MoE classifier có cải thiện chất lượng/độ ổn định so với original và MLP control trên PhoBERT-base? | Validation để phát triển; sau khóa có test nhiều seed chung, mean/sample SD, paired differences/CI theo protocol | Không mặc định MoE thắng; không tổng quát qua backbone/ngôn ngữ |
| C2: Thành phần nào tạo ra thay đổi và đánh đổi chi phí? | Phép so sánh từng thành phần, ablation trên cấu hình kết hợp, routing theo class và checkpoint, thời gian/VRAM | Routing khác nhau chưa chứng minh expert học chuyên môn; thêm tham số chưa chứng minh routing hữu ích |

Phải loại trừ giải thích thay thế: tăng capacity, đổi pooling/dropout, khác ngân sách
tuning, hoặc chọn cấu hình thuận lợi từ một seed. Kết quả âm vẫn có giá trị.

### Thiết lập chung và baseline chạy đầu tiên

| Experiment | Dataset/input | Backbone/head | Dữ liệu | max_length | Trạng thái |
|---|---|---|---|---:|---|
| exp-010 | ViANLI/cặp câu | PhoBERT-base/original HF | data/processed/vianli_segmented | 256 tổng cặp | Config sẵn; chưa chạy |
| exp-011 | ViHSD/một văn bản | PhoBERT-base/original HF | data/processed/vihsd_segmented | 128 | Config sẵn; chưa chạy |

- Model/tokenizer revision: `01daacda68afe13d83023d16ec647239e344a1e6`.
- Budget khởi đầu: full fine-tuning encoder, 7 epoch, batch 16, lr chung 1e-5,
  weight decay 0, CE không class weights; head/router LR null dùng LR chung.
  Đây là budget phát triển, chưa chứng minh tối ưu. Giữ nguyên trong mỗi phép so sánh.
- Không freeze weights encoder trong thí nghiệm chính. "Cố định backbone" nghĩa là
  giữ model/revision/chính sách fine-tune, không phải đặt freeze_backbone=true.
- Screening seed 42; development confirmation đề xuất [42, 123, 2026]. Xác nhận cuối
  dự kiến 10–15 seed chung, danh sách/số lượng/statistical protocol chưa khóa.
- Original giữ native HF head; khác pooling/nonlinearity/capacity với MoE phải được
  báo cáo. Pin model không khôi phục recipe segmentation lịch sử; xem manifests.
- Config mới không tự đăng ký một run từ xa. Khi launch, đối chiếu CSV/W&B/storage,
  cấp ID mới và giữ resolved config/artifact. Không dùng lại exp-005/006 để gán kết quả.
- Kiểm tra local 2026-11-24: bản ViHSD segmented có 2 text train rỗng và không có
  null; loader giữ chúng (24048 train), khác raw Hub lọc null còn 24046. Giữ nguyên
  chính sách trong các phép so sánh; nếu đổi chính sách phải cấp config/ID và provenance mới.
- Giai đoạn 1 đã có config/ID cụ thể theo bảng bên dưới (2026-11-25). Các biến thể
  giai đoạn 2–4 vẫn là nhãn thiết kế; tạo ID mới trước launch. Không chạy override MoE
  dưới exp-010/011 hoặc tạo config chứa cơ chế chưa được implementation hỗ trợ.

### Giai đoạn 0 — kiểm chứng dữ liệu và original (bắt buộc)

1. Kiểm tra schema, label mapping/split sizes và nguồn segmented; không segment hai lần.
   Đo độ dài token và tỷ lệ truncation trên train/validation, kiểm tra ví dụ cặp câu bị cắt.
   Không dùng nội dung/metric test để chọn max_length, preprocessing hoặc thiết kế.
2. Smoke mỗi dataset: 64 train/32 validation, 1 epoch, W&B tắt, ID riêng. Smoke chỉ
   kiểm tra pipeline/loss/checkpoint; nếu lớp hiếm vắng trong subset, không diễn giải F1.
3. Chạy exp-010/011 full seed 42. Nếu pipeline hợp lệ, kiểm tra baseline bằng hai seed
   123/2026 (ID run mới), trước khi mở rộng tìm kiếm MoE tốn kém.
4. Kiểm tra loss/gradient hữu hạn, checkpoint load lại được, validation per-class F1,
   confusion matrix, majority prediction và biến động giữa seed. Không đặt ngưỡng F1
   theo cảm tính; nếu collapse thì điều tra data/head/LR trước khi tăng độ phức tạp.

Đầu ra: baseline table, báo cáo truncation, artifact và chi phí. Mục tiêu 2 smoke +
6 full baseline cho hai dataset. Chưa có bằng chứng đã hoàn thành bất kỳ lượt nào.

### Giai đoạn 1 — reference MoE và đối chứng capacity (bắt buộc)

Reference M0: CLS → MoE(E=4,k=2,H=512,GELU,dropout=0.1) → shared final Linear,
không shared expert/residual, classifier_dropout=0, linear scoring, normalized top-k,
balance_loss_coef=0.01. Routing theo example sau pooling; encoder vẫn dense.

| Nhãn | Thiết kế | Mục đích | Hỗ trợ hiện tại |
|---|---|---|---|
| O | Original HF, exp-010/011 | Mốc hệ thống chuẩn | Có, config mới sẵn |
| C1 | Cùng pooling/expert/classifier của M0, E=1,k=1,H=512, coef=0 | Single MLP không routing có lựa chọn; không thêm baseline Linear lịch sử | Config exp-012/015 |
| C2 | Như C1, H=2051 | Đối chứng khớp tổng capacity M0 ở d=768, 3 nhãn | Config exp-013/016 |
| M0 | Reference MoE ở trên | Cặp so sánh đầu tiên với original/C1/C2 | Config exp-014/017 |

Khởi đầu seed 42 trên cả hai dataset; O đã có ở giai đoạn 0 thì không chạy lại.
C1 so với M0 gần số tham số của một expert; C2 gần tổng capacity. Không có một control
đồng thời khớp total parameters, active parameters và runtime; phải báo cả ba khi đo được.
Đếm head/router tách encoder. Với E=1, coef=0 loại auxiliary hằng số khỏi loss tổng.

Khởi tạo 2026-11-25: C1 có 790788 tham số head, C2 và M0 cùng 3156231 tham số
(tính cả router và final classifier, không tính encoder). Công thức đếm với hidden
encoder d, expert hidden H, E expert, C nhãn:
`E*(2*d*H + H + d) + (d+1)*E + (d+1)*C`.
H=2051 cho C2 khớp tổng này thay vì H=2048 lệch 4611 tham số. Router E=1 còn
được đăng ký nhưng không học từ classification loss; vì vậy khớp tổng đăng ký không
có nghĩa khớp số tham số có gradient, active compute hoặc runtime.

### Config thực thi giai đoạn 0–1 — 2026-11-25

| Thiết kế | ViANLI config | ViHSD config | Seed full ban đầu |
|---|---|---|---|
| O | configs/experiments/vianli/exp-010.yaml | configs/experiments/vihsd/exp-011.yaml | 42,123,2026 |
| C1 | configs/experiments/vianli/exp-012.yaml | configs/experiments/vihsd/exp-015.yaml | 42 |
| C2 | configs/experiments/vianli/exp-013.yaml | configs/experiments/vihsd/exp-016.yaml | 42 |
| M0 | configs/experiments/vianli/exp-014.yaml | configs/experiments/vihsd/exp-017.yaml | 42 |

Mỗi dataset có hai group tại `configs/groups/<dataset>_phobert_<tên>.yaml`:

| Tên group | ViANLI run ID dự kiến | ViHSD run ID dự kiến | Profile |
|---|---|---|---|
| stage0_baseline | run-023..024 | run-026..027 | Original full, seed 123/2026 |
| stage1_screening | run-034..036 | run-037..039 | C1/C2/M0 full, seed 42 |

Hai notebook main Kaggle/Colab chạy O tuần tự trên một GPU; notebook parallel chạy
hai config đồng thời trên hai GPU riêng. Group chính `phobert_original_pair_baseline.yaml`
dùng seed 42, run-022/025 cho full. Cell chuẩn bị notebook tạo group runtime từ group
chính, cấp run-020/021 và tắt W&B cho smoke. Parallel tách thành hai group một entry
để gọi hai runner đồng thời; runner gốc vẫn tuần tự. Direct `train.py` dùng cùng ID.
Không chạy lại baseline seed 42 qua option còn lại hoặc dùng lại cùng ID.

Smoke/full dùng cùng config nghiên cứu, lần lượt `--smoke-test`/`--no-smoke-test`.
Không gọi smoke trực tiếp bằng group có ID dành cho full; chọn option smoke notebook.
Kế thừa project người dùng đã đặt:
`cmoe_vianli`/`cmoe_vihsd`; không di chuyển run cũ, không thay budget khoa học.
ID trên mới đối chiếu checkout local, chưa đặt chỗ CSV hoặc kiểm tra W&B/storage
từ xa. Phải đối chiếu trước launch; nếu ID đã dùng thì sửa group sang ID mới.
Retry cấp ID mới. Chưa chạy GPU và không ghi metric hoặc claim completed.
Nếu đã có baseline full đúng fingerprint/seed/provenance, chỉ chạy các entry còn thiếu.

Lần lượt chạy cặp O rồi các seed xác nhận bổ sung (ví dụ):

```bash
# Smoke: chọn đúng một option trong notebook (ID riêng, W&B tắt).
python scripts/run_group.py --group-config configs/groups/phobert_original_pair_baseline.yaml --no-smoke-test
python scripts/run_group.py --group-config configs/groups/vianli_phobert_stage0_baseline.yaml --no-smoke-test
python scripts/run_group.py --group-config configs/groups/vianli_phobert_stage1_screening.yaml --smoke-test
python scripts/run_group.py --group-config configs/groups/vianli_phobert_stage1_screening.yaml --no-smoke-test
```

Mỗi lệnh chỉ chạy sau khi gate bước trước đạt; không Run All để bỏ qua kiểm tra dữ liệu,
truncation hoặc smoke. Thêm smoke giai đoạn 1 để kiểm chứng từng control/M0 trước full;
không tính các metric smoke vào bảng nghiên cứu. Seed ở group là development, chưa phải
tập seed xác nhận cuối được khóa.

Chỉ chuyển bước khi M0 có forward/backward/checkpoint/routing hợp lệ. Nếu không cải
thiện, vẫn ghi nhận và chỉ làm các phép thử có giả thuyết; không mặc định mở rộng grid.

### Giai đoạn 2 — số expert, selection và balance loss (bắt buộc, có giới hạn)

Mỗi hàng đổi đúng một yếu tố so với M0; O/C1/C2 vẫn là mốc đối chứng. Không chạy
tích Descartes. Các giá trị dưới đây là đề xuất screening, không phải lựa chọn tốt nhất.

| Thành phần | Giá trị reference → biến thể | Câu hỏi | Hỗ trợ |
|---|---|---|---|
| Số expert E | 4 → 2, 8; giữ k=2,H=512 | Thêm expert hữu ích hay chỉ tăng capacity? | Có |
| Selection k | 2 → 1, 4; giữ E=4 | Sparse so với dense learned mixture; k=4 kích hoạt toàn bộ expert | Có |
| Balance coefficient | 0.01 → 0, 0.001, 0.1 | Cân bằng assignment có giúp F1 hay gây ràng buộc quá mạnh? | Có |

Tối đa 7 cấu hình mới/dataset/seed. So E với H cố định làm thay đổi total capacity;
không kết luận số expert độc lập capacity nếu chưa có control khớp tham số ở giai đoạn 5.
Đo histogram assignment, probability mass và entropy theo class tại best validation checkpoint.
Diagnostic tổng hợp chưa đủ chứng minh ổn định routing theo từng epoch; nếu cần phải
bổ sung logging có scope/checkpoint rõ trước khi chạy.

**Giới hạn top-1 hiện tại:** weight được chuẩn hóa trong tập expert được chọn. Khi k=1,
weight=1 nên classification loss không truyền gradient hữu ích tới router qua weight;
hard index không khả vi. Router chỉ được gradient trực tiếp từ auxiliary nếu coef>0.
Không gọi kết quả top-1 là so sánh thuần mức sparsity; khi coef=0 không dùng nó làm
reference router học từ task loss. Đổi gating/gradient estimator cần quyết định và ID mới.

Gate: chọn ứng viên bằng validation F1, chi phí và routing; xem cả hai dataset, không
chỉ hàng tốt nhất. Nếu chênh lệch nhỏ hoặc bất nhất, giữ thiết kế đơn giản hơn để xác nhận
ba seed. Mất cân bằng tải không tự động đồng nghĩa mô hình sai.

### Giai đoạn 3 — representation, expert và đường kết hợp (bắt buộc có chọn lọc)

Chọn một anchor routing từ giai đoạn 2, ghi config/ID và lý do trước launch. Các hàng
được so riêng với cùng anchor; không đổi đồng thời pooling, expert và router.

| Thành phần | Reference → ứng viên | Giả thuyết cần kiểm tra | Hỗ trợ |
|---|---|---|---|
| Pooling | CLS → masked mean | Representation đầu vào có ảnh hưởng hơn routing? | Có |
| Residual | false → true | Đường giữ representation gốc có cải thiện train/F1? | Có |
| Shared expert | false → true | Nhánh chung có bổ sung thông tin hay chỉ tăng capacity? | Có; cộng output, không có gate riêng |
| Expert hidden H | 512 → 256, 1024 | Capacity nhỏ/lớn thay đổi chất lượng và chi phí thế nào? | Có |
| Activation | GELU → ReLU, SiLU | Nonlinearity có ảnh hưởng nhất quán? | Có |
| Expert dropout | 0.1 → 0, 0.2 | Regularization có hạn chế overfit? | Có |

Tối đa 9 cấu hình mới/dataset/seed; pooling/residual/H ưu tiên trước activation/dropout.
Shared expert thêm một MLP luôn hoạt động; phải tính vào total/active capacity và runtime.
Masked mean hiện bao gồm special tokens còn attention_mask=1; không âm thầm đổi công thức.
Screening từng yếu tố không xác định tương tác: cấu hình ghép các lựa chọn tốt phải
được chạy riêng, không cộng các mức tăng F1 để dự báo kết quả.

### Giai đoạn 4 — cơ chế mới, tùy chọn sau bằng chứng ban đầu

Chỉ chọn một hướng có vấn đề được quan sát và ngân sách rõ; không triển khai tất cả.
Các cơ chế dưới đây chưa có config hợp lệ trong pipeline hiện hành.

| Thành phần | Hướng thiết kế | Đối chứng/gate trước khi chạy | Trạng thái |
|---|---|---|---|
| Scoring network | Linear → MLP một hidden layer | Chỉ đổi scoring; giữ expert, selection và budget; kiểm tra gradient | Cần triển khai |
| Selection/weighting | Learned dense (k=E hiện có) → uniform mixture cùng expert | Tách tác dụng routing có học với phép trung bình; giữ capacity | Uniform cần triển khai |
| Expert output | Trộn representation + shared Linear → trộn logits từng expert | Định nghĩa trộn logits hay xác suất, kiểm soát capacity; không đổi cả hai cùng lúc | Cần triển khai |
| Regularization router | Temperature hoặc router noise | Chỉ xét khi có dấu hiệu routing bão hòa; chốt công thức/train-eval trước | Cần triển khai |
| Representation trước MoE | LSTM/BiLSTM trên chuỗi hidden trước pooling | So cùng LSTM/BiLSTM + một MLP để tách lợi ích sequence module với routing; mask padding đúng | Hoãn, cần triển khai |

LSTM/BiLSTM nếu xét vẫn nằm phía classifier sau dense encoder. Không thêm MoE trong
Transformer, không phục hồi code/checkpoint lịch sử. Quyết định toán học phải có ngày,
config/ID mới và test forward/backward/padding/checkpoint trước smoke GPU.

### Giai đoạn 5 — xác nhận, ablation và test cuối (bắt buộc)

1. Trên validation, xác nhận original + reference M0 + tối đa một cấu hình ghép tốt
   bằng [42,123,2026] trên cả hai dataset; tái dùng seed 42 chỉ khi fingerprint/provenance
   khớp. Đối chứng C1/C2 của cấu hình được chọn phải được xác nhận tương ứng trước claim
   routing/capacity. Không chỉ xác nhận biến thể có delta dương ở một seed.
2. Ablation cấu hình ghép: bỏ từng thành phần đã thêm (residual/shared/scoring mới...);
   thêm capacity-matched control khi claim liên quan E/shared. H dùng để khớp tham số
   phải được tính và chốt trước run, không dò theo F1. Khi ngân sách thiếu, thu hẹp claim.
3. Trước test, khóa cấu hình finalist và baseline đối chứng, danh sách 10–15 seed chung,
   metric chính, paired CI/kiểm định, correction nếu nhiều so sánh và xử lý run failed.
   Không tăng seed theo p-value. Backbones đã chốt không đồng nghĩa finalist đã khóa.
4. Train full các config khóa, evaluate test qua CLI có finalization. Báo mean,
   sample SD, n thực tế, paired differences/CI theo protocol, per-class F1, routing và
   chi phí cùng đơn vị/phạm vi. Nếu hiệu quả không nhất quán giữa dataset/seed, nêu rõ.

### Ngân sách và bảng kết quả dự kiến

- Giai đoạn 0: 2 smoke + 6 full baseline. Giai đoạn 1–3 screening seed 42 có trần
  20 thiết kế/dataset tính cả O/C1/C2/M0 (4 + 7 + 9), tối đa 40 full trên hai dataset;
  O seed 42 đã chạy thì tái dùng, không cộng lại. Đây là trần để cắt giảm, không phải
  danh sách bắt buộc chạy hết. Giai đoạn 4 không nằm trong trần.
- Development ba seed và ablation bổ sung tùy số ứng viên thật; chốt danh sách trước
  launch. Xác nhận cuối original + một MoE trên hai dataset với 10–15 seed cần 40–60
  lượt train; thêm một control thành 60–90. Chưa tính retry hoặc ablation.
- Chưa có GPU-hour dự toán có bằng chứng. Dùng thời gian/VRAM giai đoạn 0–1 để dự toán
  từng loại head: tổng chi phí = tổng (số run còn thiếu × thời gian thực đo tương ứng),
  cộng download/preprocessing/evaluation và dự phòng được ghi riêng. Không dùng thời gian
  smoke ít mẫu để tuyên bố full run rẻ; hai T4 không tự gộp VRAM.
- Engine hiện ghi elapsed_seconds bao gồm setup/train/validation theo scope trong summary,
  cuda_peak_allocated_bytes và parameter counts; không gọi đó là latency inference/FLOPs.
  Benchmark throughput/latency độc lập là việc bổ sung nếu cần claim hiệu quả.
- Bảng chính: dataset × {original, capacity control, MoE reference, MoE chọn}, F1
  mean/sample SD/n/delta, chi phí. Bảng ablation: component/reference/variant/config ID/
  seeds/validation F1/cost. Routing figure: expert load/probability theo class tại best
  checkpoint, không gán expert=class từ một histogram.

Mọi kết quả cần liên kết run ID, W&B, resolved config và artifact; CSV giữ metric thiếu
là trống. Chưa nhập score từ tracker hiện có nếu chưa đối chiếu W&B/artifact. Không tạo
notes chỉ để chứa kế hoạch; chỉ tạo notes khi có điều tra thực tế.

## Hồ sơ trước D013 — theo dõi run và kiểm tra parallel, 2026-11-23

Theo D012, người dùng yêu cầu bật lại CSV và dùng notebook xác nhận hai run đồng thời.
W&B, tracker và local artifact cùng liên kết run_id. Base config mặc định
logging.use_tracker=true; CSV dùng lock OS có chờ trên cùng filesystem.

Run-014/015 failed với AttributeError: Tee.isatty theo log Kaggle người dùng gửi
ngày 2026-11-23; chưa đối chiếu artifact đầy đủ, không có metric được nhập.
Đã sửa giao diện console stream và thêm regression/preflight test.
Notebook notebooks/parallel_kaggle_smoke.ipynb retry bằng run-016 (exp-003/XLM-R) và run-017
(exp-007/mDeBERTa), original head trên ViANLI, 2 epoch, 256 train/64 validation,
batch 8, seed 42. Đây là diagnostic chưa chạy trên GPU; không nhập score vào bảng
screening full. Cell cuối phải xác nhận GPU mask 0/1, hai training window chồng nhau,
CSV/manifest/summary nhất quán và W&B đồng bộ nếu bật. Không gọi evaluate test.

## Hồ sơ trước D013 — screening multilingual backbone, 2026-11-23

Mục tiêu dài hạn là kiểm tra cải thiện của MoE head trên nhiều ngôn ngữ và backbone.
Bước này chỉ khảo sát original head trên hai dataset tiếng Việt, chưa chứng minh MoE
cải thiện hoặc tổng quát đa ngôn ngữ. Tham chiếu quyết định D010.

| Experiment | Dataset | Backbone | Head | Full run | Smoke run |
|---|---|---|---|---|---|
| exp-003 (tái dùng) | ViANLI | XLM-R-base | original | run-003 | run-010 |
| exp-007 | ViANLI | mDeBERTa-v3-base | original | run-007 | run-011 |
| exp-008 | ViHSD | XLM-R-base | original | run-008 | run-012 |
| exp-009 | ViHSD | mDeBERTa-v3-base | original | run-009 | run-013 |

- Tất cả seed 42, fine-tune toàn bộ, 7 epoch, batch 16, learning rate 1e-5,
  cross entropy, checkpoint theo validation macro F1. ViANLI max_length=256;
  ViHSD max_length=128. Budget giống nhau giữa hai backbone trong từng dataset.
- XLM-R model/tokenizer revision: e73636d4f797dec63c3081bb6ed5c7b0bb3f2089.
  mDeBERTa model/tokenizer revision: a0484667b22365f84929a935b5e50a51f71f159d.
- Cả hai dùng text gốc. ViHSD đọc uitnlp/vihsd revision
  88e81b36ca376867640dad9df295e0f7d499ab80, không kế thừa segmentation PhoBERT.
  Cần quyền truy cập dataset và HF_TOKEN từ đúng tài khoản trên Kaggle.
- Cặp model/head native có pooling/dropout khác nhau; đây là screening hệ thống
  backbone + native head, không phải ablation cô lập riêng encoder.
- Smoke giới hạn 64 mẫu train/32 validation, 1 epoch, tắt W&B; ID riêng bảo đảm
  không tiêu thụ ID full run. Hai session phải dùng group khác nhau và kiểm tra ID
  trên W&B/storage và tracker. Group chạy tuần tự trên một GPU mỗi session.
- Khởi chạy group smoke trước rồi group screening cùng dataset. Lệnh ở README.
  Nếu run-003 đã hoàn thành với cùng resolved config/provenance, đối chiếu artifact
  để tái dùng; chạy riêng exp-007 thay vì chạy lại toàn group ViANLI.
- Group từ chối ghi đè run đã dùng. Retry phải đăng ký ID mới. Không tự đổi batch
  chỉ cho một backbone khi OOM; cần thống nhất lại budget của cặp so sánh và ID liên quan.
- Lưu runs, console và W&B URL của từng session trước khi kết thúc Kaggle.
  Với hai session độc lập, giữ từng bản CSV và đối chiếu theo run_id khi nhập về, không ghi đè cả file.

Chọn hướng tiếp theo bằng validation và chi phí thực đo. Một seed là screening;
giữ cả hai backbone để xác nhận MoE nếu ngân sách cho phép. Chưa khóa finalist,
chưa thêm ngôn ngữ/dataset mới và không chạy evaluate test trong đợt này.

## Các cấu hình original/MoE đã đăng ký trước đợt screening

Pipeline hướng tới original head chuẩn backbone và MoE top-k trên encoder dense.
Các cấu hình sau chỉ được đăng ký planned, chưa có full run được xác nhận:

| Experiment | Dataset | Backbone | Head | Seed screening |
|---|---|---|---|---:|
| exp-003 | ViANLI | XLM-R-base | original | 42 |
| exp-004 | ViANLI | XLM-R-base | moe | 42 |
| exp-005 | ViHSD | PhoBERT-base | original | 42 |
| exp-006 | ViHSD | PhoBERT-base | moe | 42 |

Run-003 đến run-006 được giữ chỗ cho group screening tương ứng. Diagnostic dùng run ID
riêng do CLI cấp; artifact diagnostic không dùng để kết luận chất lượng full run.

## Trình tự trước D013 (giữ để đối chiếu lịch sử)

1. Kiểm chứng offline train/checkpoint/evaluate, label mapping và artifact.
   Đợt hiện hành ưu tiên ma trận backbone original ở trên trước khi mở rộng MoE.
2. Smoke GPU trên môi trường miễn phí thực tế; lưu source/library/data provenance.
3. Screening original/MoE cùng backbone, preprocessing và budget theo mỗi dataset;
   chọn bằng validation macro F1. Đo thời gian/bộ nhớ để dự toán ngân sách.
4. Cân nhắc LSTM/BiLSTM và khảo sát scoring/selection riêng, mỗi thay đổi có câu hỏi và ID.
5. Khóa finalist, baseline đối chứng và khoảng 10–15 seed chung cùng statistical protocol.
6. Đánh giá test qua config khóa; báo cáo mean/sample SD, paired differences/CI và kiểm định
   theo protocol đã chốt, đồng thời phân tích lỗi và routing.

Chưa khóa seed, chưa có finalist. Không chạy test để quyết định cấu hình. Không đưa
metric Linear/Transformer-MoE/Hybrid/NLIMoE lịch sử vào bảng mới. Nguồn cũ: refactor@c04f25d.

Quy ước từ 2026-11-25: không tạo YAML riêng cho smoke. Dùng cùng config/group với
`--smoke-test` hoặc `--no-smoke-test`; giới hạn lấy từ training.smoke trong base.
Run ID dùng một lần: nếu chạy cùng group ở cả hai chế độ, cấp ID mới cho lượt sau.
Smoke trực tiếp bằng train CLI có thể tự cấp ID khi CSV bật và không truyền --run-id.
Logging giữ theo config; muốn tắt W&B dùng --set logging.use_wandb=false trên train CLI.
Giữ danh sách seed hiện có trong group, không tự thêm seed đã chạy trở lại.
