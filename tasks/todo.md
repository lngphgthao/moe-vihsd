# Công việc từng ngày từ 11/11 đến 26/12/2026

Hạn người học xác nhận là 26/12/2026, có thể dành cả ngày. Xem giả định, quy tắc thực nghiệm và phương án khi trễ tại [plan.md](plan.md).

Mỗi mục tương ứng một ngày và chỉ đánh dấu khi đầu ra đã được kiểm tra. Ngày thường dành ba khối 2 giờ, ngày nhẹ dành 2–3 giờ. Chạy GPU có thể kéo dài qua đêm nếu nền tảng hỗ trợ và artifact được giữ an toàn. Công việc ngày sau dùng đầu ra ngày trước, nhưng viết phần nền tảng vẫn tiếp tục khi GPU chưa xong.

## Tuần 1 · Chốt phạm vi và đường chạy · 11–17/11

- [x] **11/11 — Lập bản đồ phần còn thiếu.** Đã kiểm kê 182 W&B JSON export tại `results/wandb/lngphthao-university-of-information-technology/`: 179 run `finished`, 2 `crashed`, 1 `failed`; cả 182 log đọc được metadata/config/summary/history, nhưng vẫn `unverified` cho đến khi đối chiếu checkpoint, source revision và dataset provenance. Danh mục máy đọc ở [`experiments/wandb_inventory.csv`](../experiments/wandb_inventory.csv), bản tóm tắt phân loại ở [`experiments/wandb_inventory.md`](../experiments/wandb_inventory.md), và khung báo cáo có mục lục/trạng thái phần ở [`experiments/THESIS_REPORT_OUTLINE.md`](../experiments/THESIS_REPORT_OUTLINE.md). Chưa tự động map run vào tracker vì thiếu bằng chứng định danh duy nhất; 3 run lỗi cần tái chạy nếu dùng cho claim.
- [ ] **12/11 — Chốt vị trí đóng góp.** Đọc kỹ phương pháp và protocol của bài ViANLI/NLIMoE cùng bài ViHSD. Viết khoảng một trang về câu hỏi, đóng góp và khác biệt so với công trình gần nhất. **Đầu ra** là ghi chú có nguồn, tránh tuyên bố mới vượt bằng chứng, và nháp phần mở đầu.
- [ ] **13/11 — Trao đổi phạm vi với giảng viên.** Trình phương án hai bộ dữ liệu, Dense/Transformer-MoE, seeds và routing. Xác nhận các họ kiến trúc bắt buộc, demo, mẫu báo cáo, hạn duyệt/nộp và yêu cầu bài báo. **Đầu ra** là biên bản điều đã thống nhất hoặc danh sách còn chờ phản hồi, kèm ma trận lượt chính 12 vị trí.
- [ ] **14/11 — Xác minh môi trường và dữ liệu.** Dùng GPU dự kiến, kiểm tra dependency/Java/segmentation, split và label mapping, rồi chạy smoke test qua entry point bình thường. Khởi chạy Dense đại diện nếu kiểm tra đạt. **Đầu ra** là log môi trường, kiểm tra dữ liệu và artifact smoke, cùng thời điểm bắt đầu full run nếu có.
- [ ] **15/11 — Ngày nhẹ, dựng chương dữ liệu.** Viết mô tả hai tác vụ, phân phối nhãn và tiền xử lý từ số liệu đã kiểm tra. Kiểm tra Dense đang chạy và sao lưu. **Đầu ra** là bảng split/nhãn có nguồn và nháp phần dữ liệu. Phần thời gian còn lại dành nghỉ.
- [ ] **16/11 — Kiểm tra MoE đại diện và ngân sách.** Chạy kiểm thử liên quan, forward/backward, kiểm tra loss hữu hạn và artifact, sau đó chạy MoE đại diện nếu đạt. Dùng thời lượng thực tế để dự toán GPU-hours. **Đầu ra** là log kiểm tra và dự toán số lượt còn thiếu so với quota. Thiếu đường chạy thì trao đổi giảng viên ngay.
- [ ] **17/11 — Khóa protocol phát triển.** Ghi backbone, preprocessing, seed, training budget, metric chọn checkpoint và điều kiện so sánh. Kiểm chứng các run cũ có thể tái sử dụng. **Đầu ra** là protocol có ngày, bảng coverage 12 vị trí và hàng đợi ưu tiên không lặp lượt đã đủ bằng chứng.

**Checkpoint 17/11.** Phạm vi đang chờ duyệt phải hiển thị rõ. Chưa mở rộng thí nghiệm nếu pipeline đại diện hoặc lưu artifact chưa đạt.

## Tuần 2 · Đối chứng chính và viết phương pháp · 18–24/11

- [ ] **18/11 — Hoàn thiện đối chứng ViANLI seed đầu.** Thu và kiểm tra cặp Dense/MoE cùng điều kiện, bổ sung lượt thiếu nếu cần. Viết mô tả kiến trúc Dense. **Đầu ra** là cặp run có config/metrics/provenance và nháp Dense, hoặc lỗi chặn có hướng xử lý cụ thể.
- [ ] **19/11 — Tăng coverage ViANLI.** Chạy seed chung còn thiếu theo hàng đợi. Viết MoE, router và loss đúng với mã thực tế. **Đầu ra** là coverage cập nhật, artifacts đã tải về và công thức được đối chiếu implementation.
- [ ] **20/11 — Đóng hoặc tiếp tục seeds ViANLI.** Rà learning curve, per-class metrics và dấu hiệu mất ổn định. Mọi lượt lỗi có trạng thái/lý do rõ. **Đầu ra** là bảng tiến độ ViANLI theo seed và nháp protocol huấn luyện, không kết luận vượt số lượt đã có.
- [ ] **21/11 — Đối chứng ViHSD seed đầu.** Chạy hoặc kiểm chứng Dense/MoE cùng backbone, preprocessing và criterion. Kiểm tra CLEAN/OFFENSIVE/HATE đúng ID. **Đầu ra** là cặp đối chứng có bằng chứng hoặc log lỗi, cùng mô tả xử lý mất cân bằng nhãn.
- [ ] **22/11 — Ngày nhẹ, rà nguồn và phục hồi backup.** Hoàn thiện tài liệu liên quan, gắn nguồn cho định nghĩa/so sánh. Thử mở một bản backup và một artifact đã lưu. **Đầu ra** là danh mục tài liệu đã đọc, phần related work có nội dung và bản sao khôi phục được.
- [ ] **23/11 — Kiểm tra mốc hai bộ dữ liệu.** Tổng hợp cặp Dense/MoE hợp lệ trên từng bộ, dự toán seeds còn thiếu. Trao đổi tiến độ với giảng viên, nêu rõ yếu tố kiểm soát và thiếu sót. **Đầu ra** là báo cáo một trang và lịch hàng đợi cập nhật theo GPU thực tế.
- [ ] **24/11 — Tiếp tục seeds ViHSD và đóng phương pháp.** Chạy lượt còn thiếu ưu tiên bảng chính. Rà sơ đồ mô hình, metric, checkpoint selection và quy trình dữ liệu. **Đầu ra** là chương phương pháp đọc được liên tục, bảng coverage và nguồn cho từng chi tiết quan trọng.

## Tuần 3 · Đóng bằng chứng chính và khảo sát có giới hạn · 25/11–01/12

- [ ] **25/11 — Chẩn đoán có mục tiêu, buổi 1.** Với Classifier/Hybrid bắt buộc hoặc cần giải thích, kiểm tra labels, tiny-overfit, confusion matrix và weighted losses bằng cấu hình kiểm tra phù hợp. **Đầu ra** là báo cáo triệu chứng có log và giả thuyết kiểm tra được. Lượt chính vẫn được ưu tiên GPU.
- [ ] **26/11 — Chẩn đoán buổi 2 và quyết định dừng.** Kiểm tra một giả thuyết có khả năng phân biệt nguyên nhân. Chỉ sửa khi tìm được lỗi, giữ hồ sơ lịch sử. **Đầu ra** là bằng chứng sửa lỗi hoặc ghi nhận giới hạn và đề xuất cách trình bày với giảng viên. Kết thúc ngân sách hai ngày chẩn đoán.
- [ ] **27/11 — Mục tiêu đóng bộ chính.** Kiểm tra 12 vị trí Dense/MoE × hai bộ × ba seed. Tính mean/sample SD từ lượt hợp lệ. **Đầu ra** là bảng chính validation truy được tới run, hoặc danh sách thiếu và quyết định dùng 28–30/11 để hoàn tất trước ablation.
- [ ] **28/11 — Routing thay thế khi đủ điều kiện.** Nếu bộ chính đạt, chạy/kiểm chứng routing đã chốt, giữ expert count và placement cố định. Nếu chưa, hoàn thiện đối chứng chính. **Đầu ra** là run artifact và bảng so sánh đúng một yếu tố, ghi rõ khảo sát một seed nếu chưa đủ seeds.
- [ ] **29/11 — Ngày nhẹ, hình và chương kết quả.** Vẽ bảng/hình từ số liệu đã kiểm chứng, kiểm tra đơn vị phần trăm so với tỷ lệ 0–1. **Đầu ra** là hình có chú thích và nháp kết quả phân biệt quan sát với giải thích.
- [ ] **30/11 — Đóng phát triển cấu hình.** Hoàn tất lượt routing đủ ngân sách hoặc ghi giới hạn. Chốt finalist theo validation, toàn bộ config và seeds cho final test. **Đầu ra** là biên bản lựa chọn có thời điểm, run IDs và phạm vi kết luận, đã trao đổi giảng viên nếu giảm phần dự kiến.
- [ ] **01/12 — Đánh giá test cuối.** Dùng đúng checkpoint và config đã khóa để đánh giá theo protocol. Không dùng test để quay lại chọn hyperparameter. **Đầu ra** là predictions/metrics/per-class/confusion matrix gắn đúng run và ghi chú việc test lịch sử đã được xem nếu liên quan.

**Checkpoint 01/12.** Lượt chạy kéo dài có thể dùng 02–03/12 để hoàn tất. Không ghép bảng validation và test với seed/config khác nhau mà không giải thích.

## Tuần 4 · Phân tích và bản thảo đầy đủ · 02–08/12

- [ ] **02/12 — Đo chi phí thực tế.** Đo latency và VRAM của Dense/MoE trên cùng thiết bị, batch/độ dài/warm-up. Thống kê tham số từ mô hình thực tế. **Đầu ra** là bảng đo kèm protocol, log phần cứng và giới hạn của cách đo.
- [ ] **03/12 — Đóng bảng chính và thống kê.** Kiểm tra đủ số seed và provenance, đối chiếu mean/sample SD. Dùng phép kiểm định phù hợp nếu đã thống nhất và đủ điều kiện, giới hạn kết luận nếu chưa đủ. **Đầu ra** là bảng kết quả cuối có thể tái tạo từ artifact và nhật ký phần còn thiếu.
- [ ] **04/12 — Phân tích lỗi và expert.** Đọc các trường hợp sai đại diện trên validation, kiểm tra per-class và routing/utilization nếu đã lưu. **Đầu ra** là ví dụ có mã mẫu, hình/bảng hành vi expert có dữ liệu hỗ trợ và phần phân tích lỗi.
- [ ] **05/12 — Ghép luận điểm với bằng chứng.** Với từng kết luận, chỉ ra bảng/hình/run hỗ trợ, phân biệt chi phí thêm tham số với lợi ích kiến trúc. **Đầu ra** là chương thảo luận và giới hạn, bỏ hoặc thu hẹp phát biểu thiếu bằng chứng.
- [ ] **06/12 — Ngày nhẹ, mở bản thảo từ đầu đến cuối.** Rà thứ tự chương, thuật ngữ, ký hiệu, nguồn và chỗ trống. **Đầu ra** là danh sách lỗi theo mức ưu tiên và bản xuất PDF/Word đọc được, dùng mẫu trường.
- [ ] **07/12 — Viết phần đầu và kết luận.** Hoàn chỉnh mở đầu, mục tiêu, phạm vi, tóm tắt và kết luận theo kết quả thực tế. **Đầu ra** là mọi chương đều có nội dung, abstract/tóm tắt nhất quán với bảng chính.
- [ ] **08/12 — Gửi bản thảo đầy đủ.** Rà tên bảng/hình, tài liệu tham khảo và số liệu quan trọng, xuất bản gửi giảng viên. Đề nghị phản hồi trước 12/12. **Đầu ra** là bản thảo có phiên bản/ngày gửi và danh sách câu hỏi cần giảng viên góp ý.

## Tuần 5 · Demo và sửa nội dung · 09–15/12

- [ ] **09/12 — Làm demo tối thiểu.** Tải checkpoint được chọn, nhận đầu vào đúng tác vụ và hiển thị nhãn. Dùng lại preprocessing/tokenizer của đánh giá. **Đầu ra** là demo chạy qua entry point bình thường với đầu vào mẫu của từng tác vụ trong phạm vi đã duyệt.
- [ ] **10/12 — Kiểm tra demo và cách giao.** Kiểm tra văn bản rỗng, tiếng Việt có dấu, độ dài đầu vào, thiết bị inference và nhãn. So prediction với pipeline đánh giá trên cùng mẫu. **Đầu ra** là log đối chiếu, hướng dẫn chạy và cách tải checkpoint có thể sử dụng được.
- [ ] **11/12 — Kiểm tra gói tái lập.** Thử môi trường sạch hoặc môi trường đích, chạy inference và một kiểm tra train nhỏ, không cần chạy lại toàn bộ. Rà đường dẫn và phiên bản. **Đầu ra** là lệnh chạy đã kiểm tra, manifest môi trường và vị trí artifact lưu bền.
- [ ] **12/12 — Xử lý phản hồi nội dung.** Khi có góp ý, chia bắt buộc/quan trọng/trình bày và sửa nhóm quan trọng trước. Nếu chưa có phản hồi, rà lại claim–bằng chứng và nhắc lịch phản hồi. **Đầu ra** là danh sách góp ý có trạng thái và bản thảo sửa vòng 1.
- [ ] **13/12 — Ngày nhẹ, kiểm tra trích dẫn.** Đối chiếu tác giả, tên bài, năm, DOI và nội dung được viện dẫn với nguồn đã đọc. **Đầu ra** là tài liệu tham khảo khớp trích dẫn và danh sách nguồn còn cần bổ sung.
- [ ] **14/12 — Đóng góp ý quan trọng.** Sửa protocol, phân tích, giới hạn và hình/bảng theo góp ý. Gửi câu hỏi còn bất đồng cho giảng viên. **Đầu ra** là bản sửa có changelog ngắn và mọi góp ý quan trọng đã xử lý hoặc nêu cách giải quyết.
- [ ] **15/12 — Khóa nội dung chính.** Gửi bản gần cuối, thống nhất các việc còn lại. Đối chiếu báo cáo, demo và tracker. **Đầu ra** là bản nội dung ổn định, số liệu thống nhất và checklist công việc trình bày/thủ tục.

## Tuần 6 · Bản sẵn nộp và bảo vệ · 16–21/12

- [ ] **16/12 — Chuẩn hóa hình thức.** Áp dụng quy định trường về bìa, mục lục, font, đánh số, caption, phụ lục và tài liệu tham khảo. **Đầu ra** là PDF xuất đúng mẫu, đã kiểm tra thủ công các trang dễ lỗi.
- [ ] **17/12 — Kiểm tra toàn bộ tài liệu bàn giao.** Rà PDF, hyperlink, hình, công thức, dấu tiếng Việt, gói code, checkpoint và hướng dẫn. **Đầu ra** là bản đọc được trên một môi trường xem khác và manifest tệp bàn giao.
- [ ] **18/12 — Làm slide bảo vệ.** Theo thời lượng trường quy định, trình bày vấn đề, phương pháp, bằng chứng chính, chi phí và giới hạn. **Đầu ra** là slide dùng số liệu đúng luận văn, có chú thích và thông điệp rõ cho mỗi phần.
- [ ] **19/12 — Chốt bản sẵn nộp.** Kiểm tra yêu cầu xác nhận của giảng viên, biểu mẫu, tên file và cách nộp. **Đầu ra** là gói sẵn nộp với phiên bản cố định và checklist yêu cầu trường đã đối chiếu.
- [ ] **20/12 — Ngày nhẹ, tập bảo vệ lần 1.** Bấm giờ toàn bài, chạy demo trong đúng môi trường dự kiến. **Đầu ra** là thời lượng thực tế, danh sách câu trả lời yếu và ghi chú điều chỉnh slide. Nghỉ phần thời gian còn lại.
- [ ] **21/12 — Tập bảo vệ lần 2.** Trả lời câu hỏi về novelty, lựa chọn backbone, fairness, ba seed, test leakage, routing và kết quả âm. **Đầu ra** là bài trình bày trong thời lượng, demo kiểm tra lại và bản backup slide/PDF/checkpoint.

## Dự phòng và nộp · 22–26/12

- [ ] **22/12 — Xử lý việc chậm quan trọng nhất.** Ưu tiên góp ý giảng viên hoặc lỗi ảnh hưởng tính đúng. Nếu mọi thứ đạt, kiểm tra điều kiện nộp sớm. **Đầu ra** là lỗi ưu tiên đã đóng hoặc gói nộp sớm được xác nhận đủ yêu cầu.
- [ ] **23/12 — Kiểm tra lại sau chỉnh sửa.** Đối chiếu bảng, hình, abstract, kết luận và slide sau mọi sửa đổi. **Đầu ra** là bản nhất quán và danh sách thay đổi cuối.
- [ ] **24/12 — Nộp khi đủ điều kiện.** Nếu cổng/quy trình cho phép, nộp trước hạn và lưu biên nhận. Nếu cần chờ thủ tục, kiểm tra rõ người phụ trách và thời điểm hoàn tất. **Đầu ra** là biên nhận hoặc checklist điều kiện còn chờ có thời hạn cụ thể.
- [ ] **25/12 — Kiểm tra tình trạng tiếp nhận.** Đảm bảo hệ thống/người nhận mở được đúng tệp, đủ biểu mẫu và bản in nếu trường yêu cầu. **Đầu ra** là xác nhận tiếp nhận hoặc phần thiếu đã bổ sung, cùng backup cuối.
- [ ] **26/12 — Đóng hạn.** Hoàn tất phần nộp còn lại trước giờ đóng thực tế, kiểm tra phiên bản và lưu xác nhận. Nếu đã nộp đủ, chỉ kiểm tra trạng thái và chuẩn bị theo lịch bảo vệ. **Đầu ra** là hồ sơ nộp đủ và bằng chứng tiếp nhận.

## Cách cập nhật tiến độ

Cuối ngày ghi dưới mục tương ứng tên artifact/đường dẫn, điều đã kiểm tra và việc còn mắc. Không tự đánh dấu ngày đã trôi qua là hoàn thành. Khi có yêu cầu bắt buộc mới, thay đổi thứ tự và phạm vi được duyệt trong plan, đồng thời giữ mốc bản thảo và dự phòng nộp.
