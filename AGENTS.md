# Hướng dẫn làm việc trong repo

Quy ước được chốt với người dùng ngày 2026-11-12.

## 1. Phạm vi và cách bắt đầu

Repo phục vụ luận văn **Áp dụng kiến trúc Mixture of Experts cho bài toán phân lớp**,
nghiên cứu trên ViANLI và ViHSD với XLM-R, PhoBERT, mBERT và backbone có lý do lựa chọn.
Mục tiêu gồm chất lượng phân lớp, độ ổn định, chi phí tính toán và hành vi chuyên gia.

Thư mục `code/` trong workspace luận văn là Git root và thư mục chạy các lệnh bên dưới.
Đọc `AGENTS.md`, phần README liên quan và source trực tiếp trước khi sửa.
Đọc thêm tài liệu theo công việc thay vì tải toàn bộ tài liệu vào mọi tác vụ.

| Công việc | Tài liệu cần đối chiếu |
| --- | --- |
| Sửa implementation | `docs/ARCHITECTURE.md`, source, caller và test liên quan |
| Đề xuất thí nghiệm | Kế hoạch, tracker, ghi chú experiment và kết quả đã ghi nhận |
| Diễn giải kết quả | Tracker, artifact thực tế và ghi chú các run liên quan |
| Thay đổi phạm vi hoặc phương pháp nghiên cứu | `docs/THESIS_SPECIFICATION.md`, `docs/DECISIONS.md` |

Các mục **cấu trúc đích** bên dưới là thiết kế quản lý đã chốt.
Việc di chuyển file, đổi schema, đổi cách ghi log và chuyển dữ liệu sẽ được triển khai
trong tác vụ refactor riêng. Trong giai đoạn chuyển đổi, dùng các đường dẫn hiện tại
ở mục 3 và các lệnh hiện hành ở mục 11. Xác minh tính năng qua source trước khi sử dụng.
Một quy ước đích chỉ được mô tả là đã triển khai khi có code và kiểm chứng tương ứng.

## 2. Cấu trúc thư mục đích

Giữ một pipeline dùng chung và file riêng cho từng family kiến trúc.
Chỉ tạo file/thư mục khi có nội dung hoặc trách nhiệm thực tế.

```text
code/
  AGENTS.md
  README.md
  requirements.txt
  train.py
  evaluate.py
  src/
    data/
      loader.py
    models/
      factory.py
      backbone.py
      dense.py
      transformer_moe.py
      classifier_moe.py
      nlimoe.py
      hybrid_moe.py
      moe/
        layer.py
        ffn.py
        router.py
        dynamic_head.py
    training/
      engine.py
      losses.py
      diagnostics.py
    evaluation/
      metrics.py
    analysis/
      aggregate.py
    utils/
      config.py
  scripts/
    run_group.py
    aggregate_results.py
    preprocess_vihsd.py
    segment_vianli.py
  configs/
    base/
      vianli.yaml
      vihsd.yaml
    experiments/
      vianli/exp-001.yaml
      vihsd/exp-002.yaml
    groups/
    diagnostics/
    finalists/
      vianli/
      vihsd/
  experiments/
    PLAN.md
    experiment_tracker.csv
    RESULTS.md
    notes/
      exp-001.md
  docs/
    THESIS_SPECIFICATION.md
    ARCHITECTURE.md
    DECISIONS.md
  data/
    manifests/
      vianli.yaml
      vihsd.yaml
    raw/
    processed/
    cache/
  notebooks/
  tests/
  runs/
    run-001/
```

Các file `__init__.py` được giữ khi cần cho package và đường import hiện hành.
`runs/`, dữ liệu lớn và cache là nơi lưu runtime bên ngoài Git. `runs/` có thể được
ánh xạ tới Drive, output Kaggle hoặc ổ lưu trữ khác. Tracker ghi vị trí thực tế.

## 3. Bản đồ chuyển đổi từ cấu trúc hiện tại

| Hiện tại | Cấu trúc đích hoặc trách nhiệm sau chuyển đổi |
| --- | --- |
| `src/models/moe/layer.py` chứa layer, head và classifier hoàn chỉnh | Layer ở `models/moe/layer.py`; model và head của family ở `models/transformer_moe.py` |
| Vòng lặp train và validation trong `train.py` | `src/training/engine.py`; `train.py` điều phối CLI và lần chạy |
| Chẩn đoán routing trong `train.py` | `src/training/diagnostics.py` |
| `configs/vianli.yaml`, `configs/vihsd.yaml` | `configs/base/` |
| `configs/architectures/`, `baselines/`, `stages/` | Cấu hình chạy có ID tại `configs/experiments/<dataset>/` |
| `configs/workflows/` mô tả giai đoạn nghiên cứu | `experiments/PLAN.md` |
| `checkpoints/<run-id>/` và `results/<run-id>/` | `runs/<run-id>/` với manifest liên kết đầy đủ artifact |
| `experiments/EXPERIMENT_PLAN.md` | `experiments/PLAN.md` |
| `experiments/analysis_notes.md` | `experiments/notes/<experiment_id>.md` |
| `docs/RESULTS.md` và `experiments/RESULTS.md` | Đối chiếu lịch sử, dùng `experiments/RESULTS.md` để tổng hợp nhận định |
| `docs/PROJECT.md`, `STATUS.md`, `EXPERIMENTS.md`, `.agent/context.yaml` | Chuyển nội dung có giá trị về đúng tài liệu sở hữu ở mục 9 |
| `tasks/`, `idea-stage/` | Giữ liên kết nguồn; chuyển quyết định và kế hoạch đã chốt về tài liệu sở hữu |
| Dữ liệu đã xử lý và cache đang được Git theo dõi | Lưu bên ngoài Git, kèm manifest dữ liệu và kiểm tra khả năng truy cập |

Mỗi đợt chuyển đổi cần cập nhật caller, import, notebook, cấu hình, test và tài liệu
liên quan. Ghi ánh xạ đường dẫn cũ sang mới trong `docs/ARCHITECTURE.md`.
Bảo toàn ID, kết quả, provenance và quyết định lịch sử. Việc xóa dữ liệu hoặc gỡ dữ
liệu đang được Git theo dõi cần nằm trong phạm vi tác vụ được người dùng giao.
Đối chiếu các file gần giống nhau trước khi hợp nhất, kể cả hai group cấu hình dense.

## 4. Triển khai và mở rộng kiến trúc

Năm family có đầu mối riêng với tên registry chuẩn hiện tại.

| Registry | File đích | Trách nhiệm |
| --- | --- | --- |
| `dense_transformer` | `models/dense.py` | Encoder dense, pooling và classifier |
| `transformer_moe` | `models/transformer_moe.py` | Encoder có FFN được thay bằng MoE và classifier |
| `classifier_moe` | `models/classifier_moe.py` | Encoder dense và head MoE |
| `nlimoe` | `models/nlimoe.py` | Kiến trúc NLIMoE với các lựa chọn routing của nó |
| `hybrid_moe` | `models/hybrid_moe.py` | Kết hợp encoder Transformer-MoE với head được chọn |

- Dùng `factory.py` làm nơi đăng ký và dựng model; `backbone.py` sở hữu việc tải và
  kiểm tra backbone. Các alias lịch sử được duy trì khi caller hoặc artifact còn dùng.
- Chia sẻ router, expert, head, pooling và helper khi cùng phép tính và ngữ nghĩa.
  Đối chiếu activation, số lớp, dropout, residual, khởi tạo và chuẩn hóa trọng số trước khi gộp.
- Expert của `NLIMoEClassifier` hiện có một lớp Linear; `DynamicMoEHead` của Hybrid
  có hai lớp. Ghi rõ khác biệt khi so sánh hoặc tái sử dụng hai implementation này.
- Hybrid tái sử dụng phần encoder và head theo trách nhiệm thành phần. Khi refactor,
  kiểm tra tác động của việc dựng rồi bỏ classifier trung gian tới khởi tạo theo seed.
- Phân biệt rõ backbone, vị trí MoE, loại expert, router và head trong cấu hình.
  Với Hybrid, cấu hình encoder và head có namespace riêng.
- Giữ giao diện batch gồm `input_ids`, `attention_mask` và `token_type_ids` tùy tokenizer.
  Dùng helper input/output hiện có để các family cùng chạy được qua train, evaluate và diagnostics.
- Model trả logits cùng thông tin auxiliary theo contract dùng chung. Trainer áp dụng
  hệ số loss đúng một lần. Chẩn đoán ghi rõ phạm vi token/example, lớp và split.
- Một lựa chọn về số expert, top-k hoặc placement thường được biểu diễn bằng cấu hình.
  Thêm family hoặc helper khi có khác biệt tính toán và trách nhiệm cụ thể.
- Thay đổi toán học, loss, pooling hoặc khởi tạo phải được mô tả trong quyết định và
  provenance của lần chạy. Refactor cần kiểm tra state dict và checkpoint cũ liên quan.

## 5. Cấu hình, ID và tên file

### Cấu hình

- `configs/base/` chứa mặc định dataset, nhãn, preprocessing và training dùng chung.
- `configs/experiments/<dataset>/exp-001.yaml` xác định một cấu hình nghiên cứu bằng
  `base_config` và các giá trị thay đổi. Khai báo family và backbone rõ ràng.
- `configs/groups/` chọn các experiment và seed để chạy. `configs/diagnostics/`
  xác định kiểm tra nhỏ, overfit hoặc điều tra cụ thể, kèm mục đích và giới hạn dữ liệu.
- `configs/finalists/<dataset>/exp-001.yaml` lưu cấu hình đầy đủ đã khóa, có liên kết
  các run và tiêu chí chọn. Nội dung này giữ độc lập với các mặc định có thể sửa sau đó.
- Lưu cấu hình đã resolve và CLI override cho từng run. Dùng cấu hình đã resolve khi
  dựng lại checkpoint. Các file mô tả kế hoạch có nơi quản lý tại `experiments/PLAN.md`.

### Định danh

- Dùng mã ngắn `exp-001`, `exp-002` cho experiment và `run-001`, `run-002` cho lần thực thi.
  Dataset, backbone, family và seed là các trường có tên trong tracker và manifest.
- Một experiment biểu diễn cấu hình nghiên cứu; các seed và lần chạy lại có run ID riêng.
  Cấu hình định nghĩa experiment thay đổi sau khi đã chạy sẽ có experiment ID mới.
- Tra tracker trước khi cấp ID, chọn mã chưa dùng và ghi nhận run dự kiến trước khi
  phân phối sang Colab/Kaggle. Giữ ID duy nhất khi nhiều người hoặc môi trường cùng chạy.
  Tạo thư mục run bằng thao tác phát hiện trùng; một lần retry mới nhận run ID mới.
- ID lịch sử tiếp tục được tham chiếu bằng giá trị gốc. Bổ sung ánh xạ khi chuyển đổi.
  Phiên bản code và lý do chạy lại giúp phân biệt reproduction với lần sửa lỗi.
- Python module, hàm, biến và YAML key dùng `snake_case`; class dùng `PascalCase`.
  File cấu hình experiment dùng đúng ID, ví dụ `exp-001.yaml`. File group và diagnostic
  dùng tên tiếng Anh mô tả mục đích bằng `snake_case`.
- AGENTS và tài liệu giải thích dùng tiếng Việt. Định danh code, field, metric và tên
  kiến trúc dùng tiếng Anh. Comment giải thích lý do, giả định, kích thước hoặc đơn vị cần thiết.

## 6. Artifact và logging của một run

W&B là nơi theo dõi chính. Mỗi run đồng thời xuất đầy đủ artifact cục bộ để lưu trữ
và chuyển giữa các môi trường. W&B run, tracker và thư mục artifact cùng liên kết một `run_id`.

```text
runs/run-001/
  manifest.json
  resolved_config.yaml
  metrics.jsonl
  summary.json
  console.log
  checkpoints/
    best.safetensors
  evaluation/
  diagnostics/
```

- `manifest.json` ghi `experiment_id`, `run_id`, trạng thái, profile, seed, thời gian,
  config nguồn, commit, trạng thái working tree, dataset/model revision hoặc fingerprint,
  preprocessing, phiên bản thư viện, platform, phần cứng, artifact URI và W&B URL khi có.
  Giá trị chưa xác định được ghi `null` cùng lý do nếu cần.
- Khi code có sửa đổi chưa commit, lưu diff hoặc snapshot source liên quan để phục hồi
  đúng phiên bản. Snapshot gồm source cần thiết và loại trừ credentials, dữ liệu riêng tư.
  Nếu thiếu bằng chứng tái lập, ghi rõ giới hạn trong verification và ghi chú run.
- `metrics.jsonl` ghi nối sau mỗi epoch hoặc mốc đã cấu hình. Mỗi bản ghi có step/epoch,
  split, thời gian và tên metric rõ ràng. Metrics được ghi bền vững trong quá trình chạy.
- Ghi task loss, auxiliary loss thô, từng hệ số và total loss để truy được cách tối ưu.
  Chỉ báo cáo chi phí tính toán, thời gian và bộ nhớ khi có phép đo, đơn vị và phạm vi cụ thể.
- `summary.json` ghi checkpoint được chọn, best epoch, tiêu chí chọn và các metric tóm tắt.
  Prediction, confusion matrix và kết quả test nằm trong `evaluation/`, có split và checkpoint.
  Tên artifact dùng chung cho cả ViANLI và ViHSD.
- `console.log` lưu thông tin tiến trình và lỗi có thể chẩn đoán. Lỗi thực thi cập nhật
  trạng thái `failed`, giữ log và artifact đã có, rồi kết thúc W&B run phù hợp.
- Theo dõi trạng thái đồng bộ W&B riêng với trạng thái thực thi và verification.
  Khi mất kết nối, giữ log cục bộ, ghi phần chưa đồng bộ và đồng bộ lại khi có điều kiện.
  Thông báo rõ các lần chạy cục bộ hoặc diagnostic đã chọn tắt W&B.
- Lọc token, mật khẩu và cấu hình nhạy cảm khỏi log, manifest, snapshot và W&B config.
  Credentials được lấy từ biến môi trường hoặc secrets của nền tảng.

## 7. Tracker và kỷ luật nghiên cứu

`experiments/experiment_tracker.csv` là sổ tra cứu, mỗi dòng tương ứng một run.
Cấu hình đã resolve và artifact là bằng chứng để đối chiếu dữ liệu trên tracker và W&B.

- Giữ các cột hiện có khi chuyển đổi. Bổ sung các liên kết còn thiếu, tối thiểu gồm
  `experiment_id`, `run_id`, `status`, `verification`, `profile`, `dataset`, `backbone`,
  `architecture`, `variant`, `seed`, `config_path`, `git_commit`, `artifact_uri`, `wandb_url`.
  Các metric validation và test có cột riêng. Giá trị thiếu trong CSV để trống.
- Duy trì bộ trạng thái hiện hành `planned`, `running`, `completed`, `failed`, `invalid`,
  `anomaly`, `reproduced`, `final`; ghi verification và đồng bộ W&B bằng trường riêng.
  Trạng thái và điểm số cần có bằng chứng tương ứng với run cụ thể.
- Trước khi đề xuất run, tra các cấu hình và kết quả đã có. Nêu câu hỏi nghiên cứu,
  yếu tố thay đổi, seed, artifact dự kiến và tiêu chí chọn bằng validation.
- Dùng validation macro F1 để chọn checkpoint và phát triển cấu hình. Giữ nhất quán
  nhãn, preprocessing và các điều kiện kiểm soát của phép so sánh.
- Đánh giá test sau khi khóa finalist, cấu hình và tập seed; ghi lại quyết định khóa.
  Dữ liệu test phục vụ xác nhận cuối cùng. Các điểm test lịch sử cần ghi rõ provenance.
- Khởi đầu thăm dò có thể dùng seed 42; các cấu hình so sánh dùng seed chung.
  Báo cáo mean và standard deviation khi có nhiều seed, đồng thời nêu số lần chạy thực tế.
- Kết quả từ môi trường ngoài được nhập cùng nguồn, ID và artifact nhận được.
  Dùng `unverified` khi chưa đối chiếu đủ; dữ liệu thiếu giữ nguyên trạng thái thiếu.
- Khi W&B, tracker hoặc artifact mâu thuẫn, ghi nhận mâu thuẫn và bằng chứng cần kiểm tra.
  Bảo toàn lịch sử khi hiệu chỉnh số liệu, kèm lý do và liên kết nguồn.
- Run bất thường được ghi `anomaly` cùng hướng điều tra nhãn, logits, loss, gradient,
  routing hoặc aggregation. Kết luận nêu đúng phạm vi bằng chứng và dataset đã kiểm chứng.

## 8. Git, dữ liệu và khả năng tái lập

- Git lưu source, test, cấu hình, `AGENTS.md`, README, tài liệu, tracker, ghi chú và manifest dữ liệu.
- Dữ liệu raw/processed, cache Arrow, checkpoint, output run và credentials có nơi lưu bên ngoài Git.
  `data/manifests/<dataset>.yaml` ghi nguồn, revision/fingerprint hoặc checksum khả dụng,
  split, schema/nhãn, cách preprocessing, công cụ/phiên bản và vị trí truy cập.
- Khi điều chỉnh `.gitignore` trong bước refactor, kiểm tra cả file đã được theo dõi và
  quy tắc ignore. Xác minh bản lưu bên ngoài và manifest trước khi chuyển dữ liệu.
- Bản clone mới cần có tài liệu và lệnh chuẩn bị môi trường/dữ liệu đủ để tiếp tục công việc.
  Ghi rõ yêu cầu Python, Java, dependency và phiên bản đã kiểm chứng.
- Mọi thay đổi cần tôn trọng working tree có sẵn. Review diff theo phạm vi tác vụ và
  bảo toàn công việc của người dùng. Tác vụ Git, xóa dữ liệu hoặc thay đổi nghiên cứu
  được thực hiện theo phạm vi người dùng đã giao.

## 9. Tài liệu và ghi chú có nơi sở hữu rõ ràng

| File | Nội dung sở hữu |
| --- | --- |
| `README.md` | Setup, dữ liệu đầu vào, các lệnh chạy, notebook và đường dẫn bắt đầu |
| `AGENTS.md` | Quy tắc làm việc và quản lý repo đã chốt |
| `docs/THESIS_SPECIFICATION.md` | Phạm vi luận văn, câu hỏi nghiên cứu và phương pháp đã thống nhất |
| `docs/ARCHITECTURE.md` | Cấu trúc implementation thực tế, contract và bản đồ chuyển đổi |
| `docs/DECISIONS.md` | Quyết định có ngày, lý do, ảnh hưởng và lựa chọn được thay thế |
| `experiments/PLAN.md` | Câu hỏi, giai đoạn, ID experiment, điều kiện so sánh và bước tiếp theo |
| `experiments/experiment_tracker.csv` | Tra cứu từng run và liên kết bằng chứng |
| `experiments/RESULTS.md` | Tổng hợp nhận định qua nhiều run, dẫn tới tracker và artifact |
| `experiments/notes/<experiment_id>.md` | Điều tra và diễn biến lập luận của một experiment |

Mỗi ghi chú experiment có mục tiêu, config, các run liên quan, ngày cập nhật và các phần
**Quan sát**, **Giả thuyết**, **Kiểm chứng**, **Kết luận**, **Bước tiếp theo**.
Ghi rõ phần chưa biết, kiểm tra đã thực hiện và bằng chứng ủng hộ hoặc bác bỏ giả thuyết.
Tổng hợp kết quả bằng liên kết và nhận định; bảng metric chi tiết được sinh từ dữ liệu run.
Cập nhật quyết định theo lịch sử và dẫn tới quyết định mới khi thay đổi hướng.
Các bản ghi tác vụ tạm thời dẫn về tài liệu sở hữu khi nội dung đã trở thành quyết định chính thức.

## 10. Kiểm thử và hoàn tất thay đổi

- Truy caller, cấu hình, notebook và checkpoint liên quan trước khi sửa contract.
- Sửa lỗi tại nơi sở hữu invariant. Tái sử dụng implementation phù hợp; chỉ chia sẻ
  thành phần sau khi đối chiếu hành vi và khác biệt nghiên cứu.
- Lỗi có regression test nhỏ qua entry point liên quan. Test model kiểm tra shape,
  forward/backward, padding, input tùy chọn, loss và nạp checkpoint theo thay đổi.
- Test mặc định chạy offline bằng dữ liệu và encoder nhỏ. Tách rõ bằng chứng unit test,
  integration test, smoke GPU và thí nghiệm đã hoàn thành.
- Notebook điều phối setup, chọn cấu hình và gọi entry point dùng chung. Logic model,
  training, metric và ghi artifact có implementation trong source.
- Sau thay đổi, báo cáo file đã sửa, hành vi được sửa, kiểm tra đã chạy và giới hạn kiểm chứng.
  Cập nhật đúng tài liệu sở hữu. Chỉ đưa metric thực nghiệm vào báo cáo khi có dữ liệu thực tế.

## 11. Các lệnh hiện hành trong giai đoạn chuyển đổi

Các lệnh này dùng cấu trúc đang có tại thời điểm chốt quy ước. Chạy từ Git root `code/`.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --config configs/vihsd.yaml --smoke-test --run-id run-001
python train.py --config configs/vianli.yaml --no-smoke-test --run-id run-002
python scripts/run_group.py --group-config configs/groups/vianli_stage1_dense_backbones.yaml --no-smoke-test
python evaluate.py --config configs/vianli.yaml --run-id run-002 --final-evaluation
python scripts/aggregate_results.py --results-dir results
```

Thay các ID ví dụ bằng ID đã được ghi nhận cho lần chạy thực tế. Lệnh test evaluation
được dùng sau bước khóa finalist. Group runner hiện vẫn dựng ID từ tên group entry và seed;
ID dạng mã ngắn cho group, schema tracker mở rộng, thư mục `runs/` và xuất đầy đủ artifact
là các hạng mục của bước refactor.

`.gitignore` phải giữ `AGENTS.md`, tài liệu quản trị, kế hoạch, ghi chú và manifest dữ liệu
trong diện có thể version control; chỉ dữ liệu, cache, checkpoint, run output và credentials
được loại khỏi Git. Khi refactor tiếp theo, kiểm tra cả file đã được theo dõi và quy tắc ignore
bằng `git check-ignore -v <path>` và `git ls-files`. Dữ liệu segmented lịch sử đang được theo
dõi cần được xử lý trong một đợt di chuyển dữ liệu riêng, không tự động xóa trong các tác vụ khác.
