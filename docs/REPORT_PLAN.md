# Kế hoạch thuyết trình và báo cáo đề tài với giảng viên

Ngày soạn: 26/11/2026. Trạng thái: dàn ý làm việc cho buổi giới thiệu đề tài và hướng nghiên cứu.

Mục tiêu buổi báo cáo là giúp thầy nắm rõ bài toán, vị trí áp dụng Mixture of Experts
(MoE), lý do khảo sát và thiết kế thực nghiệm để kiểm chứng. Phần thử nghiệm có khung
riêng để bổ sung sau khi đối chiếu kết quả. Tạm thiết kế cho **20 phút trình bày và
5–10 phút trao đổi**; thời lượng này là giả định lập kế hoạch, chưa phải lịch đã chốt.

## 1. Thông điệp và phạm vi

Tên đề tài làm việc: **Nghiên cứu Mixture of Experts tại classification head cho
phân lớp văn bản tiếng Việt trên PhoBERT-base**. Tên chính thức sẽ theo thống nhất
với giảng viên.

Thông điệp xuyên suốt: giữ cùng encoder PhoBERT-base và khảo sát liệu một head
gồm nhiều expert được lựa chọn theo đầu vào có cải thiện chất lượng, độ ổn định và
đánh đổi chi phí so với head chuẩn và các đối chứng MLP hay không.

Sau buổi báo cáo, thầy cần có đủ thông tin để góp ý ba việc:

1. Phạm vi và câu hỏi nghiên cứu có phù hợp với luận văn không?
2. Các đối chứng đã đủ để phân biệt lợi ích routing với lợi ích tăng capacity chưa?
3. Thứ tự thực nghiệm và mức ngân sách xác nhận cuối có khả thi không?

Phạm vi hiện hành là dense encoder, MoE ở head, hai tác vụ ViANLI và ViHSD.
“Cố định backbone” nghĩa là giữ model/revision và chính sách fine-tune giữa các
phép so sánh; encoder vẫn được fine-tune toàn bộ. Chưa đưa LSTM/BiLSTM hoặc router
mới vào phương pháp đã triển khai. Đóng góp dự kiến là thiết kế và đánh giá có kiểm
soát trong phạm vi này; chưa khẳng định có thuật toán routing mới hoặc MoE tốt hơn.

## 2. Mạch trình bày

Đi từ nhu cầu phân lớp tiếng Việt → hai bài toán cụ thể → hệ thống baseline →
ý tưởng MoE → phương pháp hiện hành → câu hỏi và đối chứng → lộ trình kiểm chứng →
thử nghiệm đã chạy → nội dung xin góp ý.

Phân bổ dự kiến: 6 phút bài toán và nền tảng, 6 phút phương pháp, 4 phút thiết kế
thực nghiệm, 3 phút tiến độ/kết quả và 1 phút xin góp ý. Khi chưa điền kết quả,
phần tương ứng được giữ trong dàn ý, chưa coi là slide kết quả hoàn chỉnh.

## 3. Dàn ý thuyết trình gồm 15 slide

| Slide | Ý chính và nội dung cần có | Hình hoặc minh họa | Thời gian |
|---|---|---|---:|
| 1 | **Đề tài khảo sát MoE tại head của PhoBERT-base.** Tên đề tài, sinh viên, giảng viên, mục tiêu buổi báo cáo. | Sơ đồ một dòng encoder → head → nhãn | 0:30 |
| 2 | **Phân lớp tiếng Việt đặt ra các kiểu quyết định khác nhau.** Đặt cạnh một bài toán quan hệ giữa hai câu và một bài toán phân loại một văn bản; dẫn tới nhu cầu khảo sát cùng thiết kế head trên hai tác vụ. | Hai ví dụ minh họa tự soạn, ghi rõ không phải mẫu dataset | 1:00 |
| 3 | **ViANLI và ViHSD là hai điểm kiểm chứng ban đầu.** ViANLI nhận premise/hypothesis; ViHSD nhận một văn bản. Nêu nhãn theo mapping đã kiểm tra, split và yêu cầu segmentation của PhoBERT. | Bảng input → nhãn; thống kê dữ liệu lấy từ manifest/config | 1:30 |
| 4 | **Baseline dùng PhoBERT-base cùng head chuẩn Hugging Face.** Giải thích encoder tạo biểu diễn ngữ cảnh, head đưa ra logits và nhãn; nêu full fine-tuning. | Đường đi dữ liệu của baseline | 1:00 |
| 5 | **MoE mở ra khả năng xử lý đầu vào bằng các expert khác nhau.** Giải thích expert, router và top-k bằng trực giác; nêu đây là giả thuyết cần kiểm chứng với đối chứng. | Một đầu vào, bốn expert, hai nhánh được chọn | 1:00 |
| 6 | **Routing diễn ra sau pooling, theo từng example.** Dense encoder → CLS → router/expert → tổng có trọng số → linear classifier. Phân biệt vị trí này với MoE bên trong Transformer. | Sơ đồ kiến trúc với kích thước tensor | 1:30 |
| 7 | **Router chọn expert và chuẩn hóa trọng số trong top-k.** Linear scoring, softmax, top-k; expert MLP hai lớp; trộn representation rồi mới phân lớp. | Công thức ngắn và ví dụ trọng số giả định | 1:30 |
| 8 | **Huấn luyện kết hợp classification loss và balance loss.** CE học nhãn; auxiliary khuyến khích cân bằng sử dụng expert. M0: E=4, k=2, H=512, GELU, dropout=0.1, λ=0.01, không shared expert/residual. | Loss tổng và bảng cấu hình M0 | 1:00 |
| 9 | **Nghiên cứu đo chất lượng, độ ổn định và chi phí.** Ba câu hỏi: MoE có giúp không; thành phần nào tạo thay đổi; hiệu ứng nhất quán giữa dataset/seed đến đâu. | Câu hỏi ↔ bằng chứng cần thu | 1:00 |
| 10 | **Original và hai đối chứng MLP giúp diễn giải hiệu quả của MoE.** O, C1, C2, M0; giải thích C2 khớp tổng tham số head nhưng không khớp runtime/active compute. | Bảng đối chứng ở mục 5 | 1:30 |
| 11 | **Validation phục vụ phát triển, test dành cho xác nhận cuối.** Giữ cùng dữ liệu, revision, seed và budget; checkpoint theo validation macro F1; test sau khóa finalist và protocol. | Luồng train → validation → khóa thiết kế → test | 1:30 |
| 12 | **Thực nghiệm tiến từ baseline tới khảo sát từng thành phần.** Kiểm chứng dữ liệu/baseline → C1/C2/M0 → E/k/λ → pooling/expert/đường kết hợp → xác nhận và ablation. | Lộ trình có điều kiện chuyển bước | 1:00 |
| 13 | **Tiến độ thực nghiệm được phân loại theo mức kiểm chứng.** Điền sau: kiểm thử CPU, smoke GPU, full training; số run và trạng thái đối chiếu. | Bảng tiến độ, không dùng điểm smoke làm kết quả nghiên cứu | 1:00 |
| 14 | **Kết quả ban đầu sẽ được trình bày cùng điều kiện so sánh.** Điền sau: validation macro F1, per-class F1, đường học, routing, chi phí và giới hạn diễn giải. | Bảng kết quả và tối đa hai hình có nguồn run | 2:00 |
| 15 | **Bước tiếp theo cần thống nhất phạm vi và mức xác nhận.** Nhắc lại câu hỏi nghiên cứu; xin góp ý về đối chứng, ưu tiên khảo sát và ngân sách nhiều seed. | Ba nội dung xin góp ý | 1:00 |

Tổng: **20 phút**. Nếu thời lượng được rút còn 15 phút, gộp slide 2–3, 6–7,
9–10 và rút phần kết quả còn 2 phút. Công thức chi tiết và provenance để ở phụ lục.

### Nội dung kỹ thuật cần giải thích ở slide 6–8

Với một example, encoder trả chuỗi hidden states; pooling tạo vector h có chiều d.
Ở reference M0, d=768 và pooling là CLS. Router tính:

```text
p = softmax(W_r h + b_r)
S = top_k(p)
w_e = p_e / sum(p_j for j in S), với e thuộc S
f_e(h) = Linear_2(Dropout(GELU(Linear_1(h))))
z = sum(w_e * f_e(h) for e in S)
logits = W_c z + b_c
L = L_CE + λ * L_balance
```

Expert ánh xạ d → H → d. Các expert được trộn ở mức representation, dùng chung
linear classifier cuối; không mô tả thành mỗi expert tự dự đoán một lớp.
Trọng số router phụ thuộc example, không gán cứng expert theo nhãn.

Đưa công thức auxiliary hiện hành vào phụ lục nếu cần giải thích:

```text
importance = mean xác suất router trên batch
load = số assignment tới từng expert / (batch_size * k)
uniform = vector có mỗi phần tử 1/E
L_balance = E * sum(importance * load)
            + E * (MSE(importance, uniform) + MSE(load, uniform))
```

Hệ số λ chỉ áp dụng một lần trong loss tổng. Nếu minh họa router bằng các số tự
chọn, ghi “ví dụ minh họa”, không dùng chúng như routing thực đo.

## 4. Dàn ý báo cáo viết

Bản báo cáo dự kiến 8–12 trang nội dung chính, chưa tính tài liệu tham khảo và
phụ lục. Đây là mức độ dài đề xuất; điều chỉnh theo mẫu của thầy/khoa khi có.

| Mục | Nội dung cần viết | Độ dài dự kiến | Liên hệ slide |
|---|---|---:|---|
| 1. Giới thiệu đề tài | Bối cảnh, bài toán phân lớp, động cơ khảo sát head, mục tiêu và phạm vi. Kết thúc bằng câu hỏi nghiên cứu. | 1–1,5 trang | 1–3, 9 |
| 2. Cơ sở và nghiên cứu liên quan | Encoder pretrained/PhoBERT, classification head, MLP, MoE, scoring/selection, cân bằng expert. Phân biệt MoE ở head với MoE trong encoder; xác định điều cần kiểm chứng trong phạm vi đề tài. | 1,5–2 trang | 4–5 |
| 3. Phương pháp và kiến trúc | Đường đi input/output; baseline original; MoE reference; công thức routing, expert, tổng hợp và loss; cấu hình đã có và hướng chưa triển khai. | 2–2,5 trang | 6–8 |
| 4. Thiết kế nghiên cứu | Dataset/preprocessing, O/C1/C2/M0, yếu tố kiểm soát, metric, validation/test, seed, diagnostics và phép đo chi phí. | 1,5–2 trang | 10–11 |
| 5. Thử nghiệm đã thực hiện | Khung ở mục 6; điền tiến độ, điều kiện chạy, kết quả, phân tích và vấn đề tồn tại sau khi đối chiếu. | 1–2 trang | 13–14 |
| 6. Hướng triển khai tiếp theo | Thứ tự khảo sát, điều kiện chuyển bước, rủi ro/ngân sách và nội dung cần thầy góp ý. | 1 trang | 12, 15 |

Phần nghiên cứu liên quan cần bổ sung và kiểm tra tài liệu về MoE/router, ViANLI,
ViHSD trước khi hoàn thiện bản gửi. Tài liệu PhoBERT đã được dẫn trong đặc tả:
[Nguyen và Nguyen, 2020](https://aclanthology.org/2020.findings-emnlp.92/).
Chưa dùng tên đề tài làm bằng chứng cho tính mới; cần đối chiếu nghiên cứu liên quan
trước khi phát biểu khoảng trống nghiên cứu cụ thể.

## 5. Đối chứng và hướng thực hiện

| Thiết kế | Cấu hình head | Vai trò | Config ViANLI / ViHSD |
|---|---|---|---|
| O | Native classification head của PhoBERT | Baseline hệ thống chuẩn | exp-010 / exp-011 |
| C1 | Một expert MLP, H=512, E=k=1, λ=0 | So với head cùng kiểu expert/pooling, không lựa chọn giữa nhiều expert | exp-012 / exp-015 |
| C2 | Một expert MLP, H=2051, E=k=1, λ=0 | Khớp tổng tham số head đã đăng ký của M0 | exp-013 / exp-016 |
| M0 | Bốn expert H=512, top-2, λ=0.01 | Reference MoE ban đầu | exp-014 / exp-017 |

C1 có 790.788 tham số head; C2 và M0 có 3.156.231, gồm router và classifier,
không gồm encoder. Router E=1 vẫn được đăng ký nhưng không học từ classification
loss; khớp tổng tham số không đồng nghĩa khớp số tham số có gradient hoặc chi phí.

Thiết lập phát triển hiện tại: PhoBERT-base đúng revision đã pin; full fine-tuning,
7 epoch, batch 16, learning rate 1e-5, CE; max length ViANLI 256 cho cả cặp và ViHSD
128. Seed 42 phục vụ screening; [42, 123, 2026] cho xác nhận phát triển theo kế hoạch.

Thứ tự công việc bám theo [kế hoạch thực nghiệm](../experiments/PLAN.md):

1. Kiểm tra dữ liệu segmented, nhãn, truncation và pipeline; kiểm chứng original.
2. So sánh O/C1/C2/M0 dưới cùng điều kiện; đo chi phí thực tế để dự toán ngân sách.
3. Khảo sát từng yếu tố E, k, balance coefficient; sau đó chọn lọc pooling, expert
   và đường kết hợp. Chỉ mở rộng khi có câu hỏi rõ từ quan sát.
4. Xác nhận ứng viên và đối chứng bằng các seed phát triển chung; ablation thành phần.
5. Khóa finalist, baseline, seed và statistical protocol trước test. Dự kiến khoảng
   10–15 seed chung cho xác nhận cuối, chưa chốt số lượng/danh sách và chưa khóa finalist.

Top-1 cần được giải thích riêng nếu khảo sát: chuẩn hóa trong tập được chọn khiến
weight bằng 1 khi k=1, nên task loss không truyền gradient hữu ích qua weight tới
router. Không diễn giải phép thử đó như chỉ thay đổi độ thưa.

## 6. Khung báo cáo thử nghiệm sẽ điền sau

### 6.1. Thông tin của đợt báo cáo

- Ngày chốt dữ liệu báo cáo: **[điền sau]**.
- Môi trường, GPU, phiên bản thư viện và source commit: **[điền sau]**.
- Dataset/revision, segmentation, split và truncation: **[điền sau]**.
- Config đầy đủ, CLI override và cách chọn checkpoint: **[điền sau]**.

### 6.2. Tiến độ và mức kiểm chứng

| Hạng mục | Run/config liên quan | Trạng thái thực thi | Mức kiểm chứng và nguồn |
|---|---|---|---|
| Unit/integration CPU offline | [điền sau] | [điền sau] | [log test, ngày kiểm tra] |
| Smoke GPU | [điền sau] | [điền sau] | [manifest, console, checkpoint] |
| Full training original | [điền sau] | [điền sau] | [W&B + CSV + artifact] |
| Full training C1/C2/M0 hoặc biến thể | [điền sau] | [điền sau] | [W&B + CSV + artifact] |

Không suy từ config có sẵn ra run đã chạy. Completed và verified là hai thông tin
riêng; test CPU/smoke chỉ chứng minh phần pipeline đã kiểm tra.

### 6.3. Bảng kết quả validation

| Dataset | Thiết kế / experiment | Seed | Run ID và nguồn | Best validation macro F1 | Accuracy | Chi phí và phạm vi đo |
|---|---|---|---|---|---|---|
| ViANLI | [điền sau] | [điền sau] | [điền sau] | [điền sau] | [điền sau] | [điền sau] |
| ViHSD | [điền sau] | [điền sau] | [điền sau] | [điền sau] | [điền sau] | [điền sau] |

Bảng tổng hợp nhiều seed bổ sung mean, sample SD, n và danh sách seed thực tế;
không dùng một seed để kết luận độ ổn định. Dùng thang điểm 0–1 nhất quán.
Trong bảng trình bày, “[điền sau]” là chỗ trống soạn thảo; trong tracker, metric
thiếu vẫn để trống, không thay bằng zero. Test chỉ bổ sung khi đủ điều kiện khóa.

### 6.4. Phân tích và kết luận của đợt chạy

Mỗi nhận định cần đi kèm run và hình/bảng làm căn cứ:

- Quan sát: **[đường học; per-class F1/confusion matrix; chênh lệch đối chứng]**.
- Routing: **[assignment/probability theo example và class, split/checkpoint]**.
- Giải thích có thể có: **[routing, capacity, pooling, dropout hoặc biến động seed]**.
- Kiểm chứng bổ sung cần làm: **[phép so sánh có thể phân biệt các giải thích]**.
- Kết luận hiện được hỗ trợ và bước tiếp theo: **[điền sau]**.

Histogram routing khác nhau chưa chứng minh expert chuyên môn hóa. Chi phí cần
ghi rõ phần cứng, đơn vị và phạm vi: elapsed time toàn run không phải inference
latency; parameter count không thay cho runtime/FLOPs. Giữ cả kết quả âm và run lỗi
có ảnh hưởng tới cách diễn giải, không chỉ trình bày hàng tốt nhất.

## 7. Chuẩn bị tài liệu và tiêu chí hoàn thành

| Bước | Công việc | Điều kiện hoàn thành |
|---|---|---|
| 1 | Hoàn thiện câu chuyện và ví dụ hai tác vụ | Input/output rõ; nhãn khớp mapping; ví dụ tự soạn có chú thích |
| 2 | Vẽ baseline, MoE head và luồng validation/test | Hình khớp implementation; routing theo example; encoder dense |
| 3 | Viết phần giới thiệu, phương pháp và thiết kế nghiên cứu | Phân biệt đã triển khai/đề xuất; công thức và đối chứng khớp config |
| 4 | Điền phần thử nghiệm khi có nguồn | Mỗi metric khớp W&B, tracker và artifact; ghi seed/profile/split |
| 5 | Dựng slide từ dàn ý và chuẩn bị ghi chú nói | Mỗi slide một ý chính; công thức chi tiết chuyển phụ lục |
| 6 | Rà nội dung và tập trình bày | Đủ 20 phút; không claim vượt bằng chứng; ba câu hỏi xin góp ý rõ |

Bước 1–3 có thể thực hiện trước khi điền kết quả. Bước 4 bổ sung vào cùng khung
báo cáo và slide 13–14. Bước 5–6 hoàn thiện sau khi đã có nội dung phù hợp buổi gặp.

Phụ lục dự kiến: công thức auxiliary; config/revision và thống kê dữ liệu; bảng run
chi tiết; giới hạn gradient top-1; giải thích capacity/active compute; tài liệu tham khảo.

Câu hỏi cần chuẩn bị trả lời: Vì sao MoE ở head? Vì sao dùng PhoBERT-base? Nếu
MLP lớn đạt tương đương thì kết luận gì? Routing khác nhau chứng minh được gì?
Vì sao không dùng test để chọn cấu hình? Nếu ngân sách thiếu cho nhiều seed thì
thu hẹp khảo sát và claim như thế nào?

## 8. Nguồn nội bộ để cập nhật

- [Đặc tả đề tài](THESIS_SPECIFICATION.md): phạm vi, câu hỏi và protocol.
- [Kiến trúc](ARCHITECTURE.md): implementation, công thức và mức kiểm chứng kỹ thuật.
- [Quyết định](DECISIONS.md): D013 chọn backbone và D014 cấu hình đối chứng/reference.
- [Kế hoạch thực nghiệm](../experiments/PLAN.md): thứ tự, gate và ngân sách.
- [Kết quả](../experiments/RESULTS.md), [tracker](../experiments/experiment_tracker.csv),
  W&B và artifact từng run: nguồn điền phần thử nghiệm sau đối chiếu.

Tài liệu này sở hữu dàn ý buổi báo cáo; thay đổi phương pháp hoặc kế hoạch chạy vẫn
được ghi tại các tài liệu nghiên cứu tương ứng.
