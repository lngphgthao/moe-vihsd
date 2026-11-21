# Kế hoạch thực nghiệm hiện hành

## Trạng thái

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

## Trình tự

1. Kiểm chứng offline train/checkpoint/evaluate, label mapping và artifact.
2. Smoke GPU trên môi trường miễn phí thực tế; lưu source/library/data provenance.
3. Screening original/MoE cùng backbone, preprocessing và budget theo mỗi dataset;
   chọn bằng validation macro F1. Đo thời gian/bộ nhớ để dự toán ngân sách.
4. Cân nhắc LSTM/BiLSTM và khảo sát scoring/selection riêng, mỗi thay đổi có câu hỏi và ID.
5. Khóa finalist, baseline đối chứng và khoảng 10–15 seed chung cùng statistical protocol.
6. Đánh giá test qua config khóa; báo cáo mean/sample SD, paired differences/CI và kiểm định
   theo protocol đã chốt, đồng thời phân tích lỗi và routing.

Chưa khóa seed, chưa có finalist. Không chạy test để quyết định cấu hình. Không đưa
metric Linear/Transformer-MoE/Hybrid/NLIMoE lịch sử vào bảng mới. Nguồn cũ: refactor@c04f25d.
