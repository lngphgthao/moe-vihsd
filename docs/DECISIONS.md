# Quyết định hiện hành

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
