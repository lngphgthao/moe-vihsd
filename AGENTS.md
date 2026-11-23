# Hướng dẫn làm việc trong repo

Phạm vi cập nhật ngày 2026-11-22 theo lựa chọn của người dùng khi duyệt refactor.
Git root là thư mục `code/`; chạy lệnh từ đây. Tài liệu giải thích dùng tiếng Việt;
định danh code/config/metric dùng tiếng Anh.

## 1. Phạm vi hiện hành

- Nghiên cứu MoE ở classification head trên encoder dense; ViANLI và ViHSD là điểm xuất phát.
- Implementation hiện hành gồm `model.head.type: original | moe`. Original là head
  chuẩn Hugging Face của backbone; MoE dùng expert MLP hai lớp và linear/top-k routing.
- LSTM/BiLSTM, router mới, dataset mới là việc tiếp theo, chưa được triển khai.
- Không giữ Transformer-MoE, Hybrid, NLIMoE hay alias/checkpoint compatibility lịch sử.
- Toàn bộ source và hồ sơ trước refactor có tại nhánh `refactor`, commit
  `c04f25d05d8802c8a32d26eb1d6a6b53e2781adf`. Không copy lại code lịch sử vào pipeline hiện hành.

## 2. Trước khi sửa

Đọc README, source, caller và test liên quan. Sửa implementation cần đối chiếu
`docs/ARCHITECTURE.md`; đổi phương pháp cần đối chiếu `docs/THESIS_SPECIFICATION.md`
và `docs/DECISIONS.md`; đề xuất run cần xem `experiments/PLAN.md`, W&B và artifact.
Theo D012, tracker CSV đã bật lại mặc định; đối chiếu cùng W&B và artifact.
Tôn trọng working tree. Chỉ tạo module/thư mục khi có trách nhiệm và nội dung thực tế.

## 3. Cấu trúc và contract

- `src/models/factory.py` kiểm tra config và dựng model. `backbone.py` tải dense encoder.
- `heads/original.py` giữ model phân lớp chuẩn; `classifier.py` ghép encoder với `heads/moe.py`.
- Router nằm trong `heads/router.py`. Tách scoring và selection trong config, hiện chỉ linear/top-k.
- Model nhận `input_ids`, `attention_mask`, `token_type_ids` tùy tokenizer; trả `(logits, aux_dict)`.
- Auxiliary loss trả dạng thô. Trainer áp `routing.balance_loss_coef` đúng một lần.
- Engine sở hữu train/validation và optimizer; CLI chỉ resolve config và điều phối.
- Diagnostics routing theo example và class, có split/checkpoint rõ ràng; không gọi số example là token_count.
- Không đổi expert, pooling, activation, dropout, residual hay khởi tạo khi chỉ di chuyển code.
  Thay đổi toán học cần quyết định mới và provenance của run.

## 4. Config và định danh

- `configs/base/`: mặc định dataset/training, không phải finalist.
- `configs/experiments/<dataset>/exp-NNN.yaml`: overlay có ID; dataset/backbone/head resolve rõ ràng.
- `configs/groups/`: danh sách config, seed và run ID cụ thể; kiểm tra trùng trên W&B/storage trước khi chạy.
- `configs/diagnostics/`: giới hạn dữ liệu và mục tiêu kiểm tra rõ ràng.
- Chỉ tạo `configs/finalists/<dataset>/` khi khóa finalist; lưu config đầy đủ cùng
  `finalization.locked`, `locked_at`, `selection_reason`, `seeds`.
- `exp-001/002` và `run-001/002` đã dùng trong lịch sử, không tái sử dụng.
- `logging.use_tracker: true` là mặc định; CSV giữ một dòng mỗi run. Group có ID entry;
  train có thể cấp ID kế tiếp từ CSV nhưng nên chỉ định rõ khi phân phối nhiều máy.
  FileLock chờ writer khác tối đa 30 giây và CSV được thay nguyên tử; mkdir chống ghi đè.
  Retry nhận ID mới. Khóa chỉ điều phối các process cùng filesystem, không đồng bộ hai máy.
- Cấu hình nghiên cứu đã chạy mà thay đổi phải cấp experiment ID mới. CLI override được lưu;
  seed/profile diagnostic không biến metric diagnostic thành bằng chứng full run.

## 5. Artifact, tracker và W&B

Mỗi run lưu dưới `runs/<run_id>/`: manifest, resolved config, metrics JSONL, summary,
console log, labels, HF config, checkpoint, evaluation và diagnostics. `RUNS_DIR` có thể
đặt ra Drive hoặc output Kaggle. Source snapshot của working tree bẩn chỉ gồm source đã
kiểm tra, không có `.env`, dữ liệu hay output runtime.

W&B là kênh theo dõi chính, đồng thời phải giữ artifact local. Tách `status`,
`verification`, `wandb_sync`; mất kết nối W&B không làm mất run local. Lỗi thực thi
phải giữ log và trạng thái failed. Lọc credentials khỏi config, log và snapshot.

Tracker CSV bật mặc định theo yêu cầu mới, liên kết run_id với W&B và local artifact.
Có thể chủ động tắt bằng logging.use_tracker=false; khi đó phải chỉ định run ID và
không đọc/ghi CSV. CSV mỗi dòng một run, giữ validation/test riêng. Các trạng thái nghiên cứu:
planned, running, completed, failed, invalid, anomaly, reproduced, final. Completed
chưa đồng nghĩa verified. Thiếu số liệu thì để trống, không thay bằng zero.

## 6. Kỷ luật thực nghiệm

- Validation macro F1 dùng chọn checkpoint và phát triển cấu hình.
- Test chỉ chạy sau khi khóa finalist và tập seed, qua evaluate CLI. Không chọn bằng test.
- Baseline original mới khác baseline Linear lịch sử; không gán lại metric cũ.
- Dùng seed chung và kiểm soát dữ liệu, backbone, budget; nêu khác biệt head/pooling/dropout.
- Finalist và baseline đối chứng dự kiến khoảng 10–15 seed chung; chốt danh sách và
  statistical protocol trước xác nhận, không tăng seed theo p-value.
- Báo cáo mean, sample SD và số run thực tế; phân tích ghép cặp/CI/kiểm định chỉ khi
  có thiết kế và bằng chứng phù hợp. Không suy ra significance từ số seed.
- Không tự khóa finalist, chạy nghiên cứu GPU hay nhập metric chưa đối chiếu trong tác vụ refactor.

## 7. Dữ liệu và Git

Git giữ source, tests, config, tài liệu, tracker và `data/manifests/`. Raw/processed/cache,
weights, credentials và `runs/` nằm ngoài Git. Không rewrite lịch sử Git để dọn dữ liệu.
Manifest ghi source/revision, schema/labels, preprocessing, vị trí và checksum có thực.
Không trình bày revision lịch sử hoặc checksum file như bằng chứng đã tái lập segmentation.
Đổi vị trí dữ liệu phải so checksum và giữ khả năng truy cập; không xóa bản local ngầm định.

## 8. Nơi sở hữu tài liệu và kiểm chứng

README: setup/lệnh/notebook. THESIS_SPECIFICATION: scope/phương pháp.
ARCHITECTURE: implementation/contract/bản đồ chuyển đổi. DECISIONS: quyết định có ngày.
experiments/PLAN: câu hỏi và bước tiếp theo; tracker: từng run; RESULTS: nhận định có nguồn.
Chỉ tạo `experiments/notes/<experiment_id>.md` khi có điều tra thực tế, gồm Quan sát,
Giả thuyết, Kiểm chứng, Kết luận, Bước tiếp theo.

Test mặc định offline với encoder/dữ liệu nhỏ. Kiểm tra shape, forward/backward,
padding, input tùy chọn, loss, checkpoint và entry point chịu ảnh hưởng. Báo cáo rõ
unit/integration CPU, smoke GPU và thí nghiệm hoàn thành; không suy rộng giữa chúng.

```bash
python -m unittest discover -s tests -v
python train.py --config configs/experiments/vianli/exp-003.yaml --run-id run-010 --smoke-test
python scripts/run_group.py --group-config configs/groups/vianli_screening.yaml --no-smoke-test
python scripts/aggregate_results.py --runs-dir runs
```
