# Kế hoạch thực nghiệm hiện hành

## Ưu tiên hiện hành — screening multilingual backbone, 2026-11-23

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
  không tiêu thụ ID full run. Hai session phải dùng group khác nhau, không auto-cấp
  ID độc lập từ hai bản tracker. Group chạy tuần tự trên một GPU mỗi session.
- Khởi chạy group smoke trước rồi group screening cùng dataset. Lệnh ở README.
  Nếu run-003 đã hoàn thành với cùng resolved config/provenance, đối chiếu artifact
  để tái dùng; chạy riêng exp-007 thay vì chạy lại toàn group ViANLI.
- Group từ chối ghi đè run đã dùng. Retry phải đăng ký ID mới. Không tự đổi batch
  chỉ cho một backbone khi OOM; cần thống nhất lại budget của cặp so sánh và ID liên quan.
- Lưu runs, tracker và console của từng session riêng trước khi kết thúc Kaggle.
  Khi nhập về, đối chiếu từng dòng bằng run_id; không ghi đè toàn CSV của session kia.

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

## Trình tự

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
