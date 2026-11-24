# Quyết định hiện hành

## D014 — 2026-11-25: khởi tạo config PhoBERT giai đoạn 0–1

Theo yêu cầu người dùng, cụ thể hóa C1/C2/M0 trong PLAN thành exp-012/013/014 cho
ViANLI và exp-015/016/017 cho ViHSD. Giữ original exp-010/011, revision và budget
đã chọn; không thay toán học model/router/loss. Mỗi experiment mới kế thừa cùng dữ liệu
segmented và diagnostics validation của original. Không chạy nghiên cứu GPU ở bước này.

C1 dùng E=1,k=1,H=512, coef=0. C2 dùng cùng single-expert control với H=2051 để
khớp tổng head parameters M0 tại d=768, 3 nhãn. M0 E=4,k=2,H=512,GELU,dropout=0.1,
linear scoring/top-k, coef=0.01, CLS, không residual/shared expert. Tổng tham số head
C1=790788, C2=M0=3156231, gồm router còn đăng ký và classifier. Router E=1 không
học từ task loss; không khẳng định khớp active compute/runtime hoặc số tham số có gradient.

Thêm hai diagnostic original và tám group theo dataset/stage, IDs dự kiến run-020..039:
stage0 smoke, baseline [42,123,2026], stage1 smoke C1/C2/M0 và screening seed 42.
ID chỉ đối chiếu local, chưa kiểm tra/đặt chỗ W&B/storage từ xa; không cập nhật CSV hoặc
metric trong tác vụ setup. Trước launch phải đối chiếu ID, tái dùng baseline cũ chỉ khi
fingerprint/seed/provenance khớp. Seed development không thay thế tập seed cuối chưa khóa.

Giữ tên project người dùng đã sửa trong base config là cmoe_vianli/cmoe_vihsd.
Không di chuyển run W&B cũ. Đổi project logging không thay đổi phương pháp nghiên cứu.

Xem xét lỗi test notebook: bỏ assertion buộc có chữ `GitHub Actions` vì phụ thuộc câu
chữ markdown, giữ kiểm tra code notebook không tự chạy unittest; kiểm tra trực tiếp
workflow CI chạy suite với biến offline. Không sửa notebook chỉ để làm test đạt.

## D013 — 2026-11-24: cố định PhoBERT-base, tập trung thiết kế MoE classifier

Theo người dùng chốt, chọn `vinai/phobert-base` làm backbone cho cả ViANLI/ViHSD;
không chọn base-v2. Thay thế ưu tiên screening backbone của D010, giữ nguyên hồ sơ cũ.
Mục tiêu là so sánh các thiết kế classifier trên một dense encoder tiếng Việt cố định.
Không cần chứng minh backbone tốt nhất; không suy rộng hiệu quả qua backbone/ngôn ngữ.

Cơ sở: PhoBERT là encoder đơn ngữ có bằng chứng thực nghiệm trên nhiều tác vụ tiếng
Việt, gồm NLI; xem [paper](https://aclanthology.org/2020.findings-emnlp.92/).
Độ ổn định trong pipeline vẫn cần kiểm tra nhiều seed và chi phí thực đo.
Metadata công khai [HF API](https://huggingface.co/api/models/vinai/phobert-base)
được đối chiếu ngày 2026-11-24: revision `01daacda68afe13d83023d16ec647239e344a1e6`;
pin cùng revision cho tokenizer, không kế thừa tokenizer revision XLM-R.

Thêm exp-010 (ViANLI original) và exp-011 (ViHSD original), dữ liệu local segmented,
fine-tune encoder, budget khởi đầu 7 epoch/batch 16/lr 1e-5/seed 42, CE, validation
macro F1 chọn checkpoint. Dùng ID mới thay vì sửa exp-005 chưa pin revision. Chưa
chạy GPU, chưa đăng ký run ID trên W&B/storage, chưa khóa finalist hoặc seed cuối.

`experiments/PLAN.md` sở hữu bảng giai đoạn: baseline, reference MoE, số expert/routing/
balance loss, representation/expert/đường kết hợp, cơ chế mới tùy chọn, xác nhận/ablation.
Các cơ chế mới chỉ là đề xuất, chưa được factory hỗ trợ. Khi triển khai cần quyết định
toán học cụ thể và experiment ID mới; không dùng config giả với key chưa hỗ trợ.
So sánh capacity bằng single-expert/dense controls, không mặc định nhiều tham số là MoE tốt hơn.
Giữ nguyên công thức expert/router/loss trong tác vụ này.

Giới hạn dữ liệu segmented lịch sử theo `data/manifests/` vẫn còn: pin model hiện tại
không xác minh lại recipe segmentation hoặc revision nguồn của từng file local.
Phải kiểm tra truncation train/validation trước full, đặc biệt cặp câu ViANLI ở 256 token.

## D007 — 2026-11-21: ưu tiên MoE tại classifier/head

Theo người dùng đã thống nhất với giảng viên: dùng encoder dense, khảo sát head/router,
khởi đầu ViANLI và ViHSD; finalist và baseline đối chứng dự kiến 10–15 seed chung.
Bản ghi nguyên gốc và D001–D006: refactor@c04f25d05d8802c8a32d26eb1d6a6b53e2781adf.

## D008 — 2026-11-22: làm sạch repo theo scope mới

Người dùng chọn nhánh hiện hành sạch, không giữ tương thích checkpoint/config lịch sử;
bỏ hoàn toàn NLIMoE, Transformer-MoE và Hybrid. Lưu bổ sung hồ sơ chưa commit/chưa
được theo dõi vào nhánh refactor trước khi dọn; commit lưu cố định là c04f25d.

Original baseline dùng head chuẩn Hugging Face; bỏ baseline AutoModel + Linear cũ
khỏi pipeline mới. Đây là thay đổi mô hình đối chứng, cần run mới; không gán metric cũ.
MoE giữ expert/top-k, pooling, dropout, residual và loss thô hiện có trong bước tách code.
Không còn trả auxiliary đã nhân hệ số từ head; trainer sở hữu hệ số duy nhất.

Đợt triển khai gồm original và MoE; hoãn LSTM/BiLSTM và router mới. Hai notebook theo
môi trường dùng CLI chung. Runtime chuyển sang runs; dataset segmented chuyển sang
processed, đối chiếu checksum, gỡ theo dõi Git và giữ local. Không rewrite lịch sử Git.

## D009 — 2026-11-22: cấu hình mới là cấu hình screening

exp-003/004 là original/MoE ViANLI với XLM-R; exp-005/006 là original/MoE ViHSD với
PhoBERT. Seed mẫu 42; budget khởi đầu 7 epoch, batch 16, lr 1e-5, không phải finalist.
Hai head dùng cùng budget trong mỗi dataset. Chưa quyết định backbone/seed/protocol
xác nhận cuối và chưa có metric mới. Thay đổi config đã chạy cần ID mới.


## D010 — 2026-11-23: screening backbone đa ngôn ngữ trước nghiên cứu head

Theo trao đổi và yêu cầu setup của người dùng, ưu tiên XLM-R-base và
mDeBERTa-v3-base cho mục tiêu dài hạn khảo sát MoE head qua nhiều ngôn ngữ.
Bước hiện hành chỉ dùng original head trên ViANLI/ViHSD, seed 42, không mở rộng
ngôn ngữ và không tạo claim cải thiện MoE. PhoBERT/exp-005/006 vẫn giữ nguyên hồ sơ.

Tái dùng exp-003; thêm exp-007 (mDeBERTa/ViANLI), exp-008 (XLM-R/ViHSD) và
exp-009 (mDeBERTa/ViHSD). Pin revision model và tokenizer tương ứng để tránh vô tình
kế thừa revision XLM-R khi đổi backbone. ViHSD dùng text gốc từ Hub, không dùng
DatasetDict đã segment cho PhoBERT. Không thay đổi hai base config hay notebook
người dùng đang sửa. Các experiment kế thừa budget chung hiện tại, không phải finalist.

Bổ sung SentencePiece/Protobuf để đọc checkpoint tokenizer chỉ có spm.model. mDeBERTa
được nhận diện bởi AutoModelForSequenceClassification dưới model_type deberta-v2;
không cần thêm family hoặc thay contract model/trainer.

Nguồn metadata (đối chiếu 2026-11-23):
- [mDeBERTa model card](https://huggingface.co/microsoft/mdeberta-v3-base)
- [mDeBERTa revision metadata](https://huggingface.co/api/models/microsoft/mdeberta-v3-base)
- [ViHSD nguồn có kiểm soát truy cập](https://huggingface.co/datasets/uitnlp/vihsd)

Full run dùng run-003/007/008/009; diagnostic dùng run-010..013. Khóa ID trước
khi phân phối sang hai session Kaggle; chưa có metric mới hoặc full run được xác nhận.
One-seed screening không kết luận significance hoặc khả năng tổng quát đa ngôn ngữ.


## D011 — 2026-11-23: tạm ngừng cập nhật CSV, theo dõi bằng W&B

Theo chỉ đạo người dùng, không cập nhật tracker CSV trong repo/code ở workflow
hiện hành. Giữ CSV làm hồ sơ tĩnh; W&B và artifact local tiếp tục liên kết run ID.
Không xóa dữ liệu, CSV hoặc utility cũ.

Thêm logging.use_tracker=false mặc định. Train/group/evaluate không đọc/ghi CSV
hoặc tạo lock khi tắt. Train yêu cầu --run-id rõ ràng; group dùng ID entry. Hai process
có thể dùng chung checkout và cache, mỗi process một GPU, ID và thư mục artifact riêng.
Group runner vẫn tuần tự, không tự tạo process song song. Notebook bổ sung RUN_ID.

CSV chỉ bật lại khi chủ động đặt logging.use_tracker=true. Khi bật, cơ chế lock cũ
không chờ tranh chấp, nên không dùng cho hai writer đồng thời trong cùng checkout.
Đây là thay đổi vận hành/logging, không đổi kiến trúc, toán học hoặc protocol lựa chọn
checkpoint; không cần cấp lại experiment ID chỉ vì tắt CSV. Chưa kiểm chứng GPU.


## D012 — 2026-11-23: bật lại CSV và xác nhận hai GPU trong một checkout

Thay thế mặc định tạm tắt CSV ở D011 theo yêu cầu mới của người dùng. Base config
và giá trị mặc định runtime bật tracker; opt-out rõ ràng use_tracker=false vẫn được giữ.
Sửa lock sang FileLock có chờ 30 giây và khóa OS tự nhả khi process đóng; không xóa
.lock sau mỗi lần ghi để tránh tách inode/khóa giữa các writer. CSV thay nguyên tử
sau flush/fsync. Sửa mapping manifest dataset/model sang các cột scalar của tracker.
Nguồn thư viện: https://py-filelock.readthedocs.io/en/latest/ ; khai báo dependency trực tiếp.

Thêm notebook parallel_kaggle_smoke.ipynb, hai process trong cùng checkout, shared cache,
mỗi process một GPU và run ID riêng. run-014/015 tham chiếu exp-003/007 nhưng dùng profile
diagnostic (2 epoch, 256 train/64 validation, batch 8), không phải full research run.
Lưu GPU mask, PID, visible GPU count và mốc thời gian train/validation vào manifest.
Notebook chỉ xác nhận parallel nếu các training window chồng nhau và artifact/tracker
đúng; kết quả CPU/fixture không được trình bày thành bằng chứng đã chạy GPU.

Chưa chạy notebook trên Kaggle trong tác vụ chuẩn bị. Không khóa finalist, không đánh
giá test và không thay đổi toán học của model/loss. Mốc kiểm chứng CPU được cập nhật
trong ARCHITECTURE sau khi hoàn thành test.

## 2026-11-25 — dùng CLI chọn smoke/full, không nhân bản config

Theo yêu cầu người dùng, bỏ YAML chỉ dành cho smoke; cùng experiment/group chọn
profile bằng --smoke-test hoặc --no-smoke-test. Giữ training.smoke trong base và
diagnostic tiny-overfit có mục đích riêng. Không đổi phương pháp hoặc run lịch sử.
Run ID vẫn dùng một lần cho mỗi lượt thực thi; không dùng lại ID smoke cho full.
Các ghi nhận config smoke ở quyết định trước là lịch sử setup, không phải file hiện hành.
