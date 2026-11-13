# Định nghĩa dự án

Xem [THESIS_SPECIFICATION.md](THESIS_SPECIFICATION.md) để đọc phạm vi luận văn hiện tại,
câu hỏi nghiên cứu, protocol experiment, quyết định còn mở và workflow bàn giao. File này
ghi định nghĩa cấp cao ổn định; specification đối chiếu thêm proposal, implementation và
record experiment.

## Tên đề tài

Applying Mixture of Experts Architectures to Text Classification

## Mục tiêu

Khảo sát việc áp dụng và hiệu quả của architecture Mixture-of-Experts cho bài toán phân
loại văn bản tiếng Việt.

## Dataset

### ViANLI

Dataset Vietnamese Natural Language Inference.

### ViHSD

Dataset Vietnamese Hate Speech Detection.

Hai dataset này là task đánh giá cho nghiên cứu phân loại văn bản dựa trên MoE, không
giới hạn luận văn vào riêng adversarial NLI.

## Backbone ứng viên

- PhoBERT
- XLM-R
- mBERT
- Other justified Vietnamese or multilingual encoder models

## Hướng nghiên cứu

Nghiên cứu có thể khảo sát:

- Dense vs MoE architectures
- Backbone selection
- Number of experts
- Expert architecture
- Routing strategies
- Top-k routing
- Load balancing
- Auxiliary losses
- MoE placement
- Ablation studies
- Computational cost

## Đánh giá

Đánh giá cần dùng metric classification phù hợp, gồm accuracy và macro F1 khi áp dụng được.

Experiment quan trọng cần dùng nhiều random seed khi tài nguyên cho phép.

## Ranh giới phạm vi

Dự án tập trung áp dụng và đánh giá architecture MoE cho bài toán phân loại văn bản.

ViANLI và ViHSD là dataset đánh giá, không phải toàn bộ định nghĩa của bài toán nghiên cứu.
