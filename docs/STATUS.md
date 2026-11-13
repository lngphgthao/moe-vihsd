# Trạng thái dự án hiện tại

Last updated: 2026-09-29

## Giai đoạn

Phát triển và đánh giá experiment.

## Mục tiêu hiện tại

Xây dựng framework experiment reproducible để so sánh architecture dense và MoE trên
ViANLI và ViHSD.

## Đã hoàn thành

- [x] Initial repository established
- [x] Dataset scope defined
- [x] MoE research direction defined
- [x] Multiple backbone strategy established
- [ ] Verify all current implementation details
- [x] Classifier-MoE architecture implemented
- [x] Hybrid-MoE composition implemented

## Đang thực hiện

- Experimental framework
- Backbone comparison
- Dense baselines
- MoE baseline
- Routing experiments
- Hybrid-MoE first experiment configuration prepared

## Bước tiếp theo

1. Kiểm tra architecture hiện tại của repository.
2. Thiết lập dense baseline.
3. Thiết lập MoE baseline.
4. So sánh các backbone.
5. Đánh giá routing strategy.
6. Thực hiện ablation experiment.
7. Lặp lại experiment quan trọng với nhiều seed.

## Chạy bên ngoài repository

Experiment có thể chạy trên:

- Kaggle
- Google Colab
- Other GPU environments

Vì vậy, kết quả có thể cần đồng bộ thủ công vào `experiments/RESULTS.md` và
`experiments/experiment_tracker.csv`.

## Cảnh báo quan trọng

Code hoặc config tồn tại không có nghĩa experiment đã chạy thành công.

Chỉ xem result được ghi nhận rõ ràng là đã hoàn thành.
