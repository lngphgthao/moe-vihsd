# Phạm vi luận văn hiện hành

Cập nhật phạm vi triển khai 2026-11-23. Nghiên cứu áp dụng Mixture of Experts cho phân lớp văn bản tiếng Việt,
với MoE tại classifier/head trên encoder dense. ViANLI (premise/hypothesis) và ViHSD
(single text) là điểm xuất phát. XLM-R, PhoBERT và mBERT là backbone có thể khảo sát;
config chạy cụ thể cần lý do lựa chọn và ngân sách.

Ưu tiên hiện hành (D010): screening XLM-R-base và mDeBERTa-v3-base với original
head trên cả hai dataset. Hướng dài hạn là kiểm tra cải thiện MoE qua nhiều ngôn ngữ;
chưa thêm ngôn ngữ mới và dữ liệu hiện tại chưa đủ để xác nhận khả năng đó.

## Câu hỏi nghiên cứu

1. MoE head so với head chuẩn backbone và head thay thế có chất lượng/độ ổn định ra sao?
2. Scoring network, selection rule và số expert ảnh hưởng kết quả, chi phí và hành vi expert thế nào?
3. Kết luận nhất quán giữa dataset và seed đến đâu, trong tài nguyên miễn phí thực có?

Implementation đợt đầu chỉ gồm original và MoE linear/top-k. LSTM/BiLSTM, router khác
và dataset mở rộng được xem xét sau; chưa có kết quả hoặc lựa chọn cuối cho chúng.
Transformer-MoE, Hybrid và NLIMoE không thuộc pipeline hiện hành.

## Protocol

- Original dùng AutoModelForSequenceClassification, giữ head chuẩn theo backbone.
  MoE dùng AutoModel, CLS hoặc masked mean pooling và expert MLP hai lớp.
- Cùng backbone, revision, preprocessing, split, seed và training budget khi so sánh.
  Công khai khác biệt pooling, dropout, capacity và tham số trainable giữa hai head.
- Validation macro F1 dùng chọn checkpoint và phát triển cấu hình. Test chỉ dùng sau
  khóa config finalist, baseline đối chứng, seed và statistical protocol.
- Screening có thể dùng seed 42. Cấu hình cuối dự kiến khoảng 10–15 seed chung;
  số seed/danh sách cụ thể chưa khóa. Không tăng seed dựa trên p-value.
- Báo cáo mean, sample SD, số run thực tế và phân tích ghép cặp/CI/kiểm định thích hợp
  khi có đủ dữ liệu. Kết quả âm hoặc thiếu bằng chứng vẫn được báo cáo.
- Chỉ báo cáo chi phí với phép đo, đơn vị và phạm vi. Parameter count không thay thế
  phép đo runtime hoặc FLOPs. Không giả định MoE tốt hơn hay rẻ hơn baseline.

Baseline Linear lịch sử và metric của nó không phải kết quả original head mới.
Source/provenance lịch sử được lưu tại commit c04f25d trên nhánh refactor.
