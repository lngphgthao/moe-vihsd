# Kế hoạch hoàn thành luận văn đến 26/12/2026

Cập nhật ngày 11/11/2026, múi giờ Asia/Saigon. Người học xác nhận hạn 26/12/2026 và có thể dành cả ngày. Lịch chi tiết và ô đánh dấu nằm trong [todo.md](todo.md).

## Đánh giá khả năng hoàn thành

**Có thể kịp, nhưng cần thu gọn số câu hỏi thực nghiệm và viết ngay từ tuần đầu.** Đây là đánh giá có điều kiện dựa trên hồ sơ cục bộ, chưa phải cam kết về tiến độ chạy GPU hay thời gian giảng viên phản hồi.

Từ 11/11 đến hết 26/12 có 46 ngày lịch. Mục tiêu nội bộ là hoàn tất nội dung chính ngày 19/12, tập bảo vệ ngày 20–21/12 và giữ 22–26/12 làm dự phòng. Nếu trường yêu cầu nộp trước ngày bảo vệ, hạn nộp sớm hơn phải trở thành mốc điều phối.

### Những gì đang có

- Đã có pipeline huấn luyện/đánh giá, cấu hình Dense và các họ MoE, dữ liệu cục bộ ViANLI/ViHSD và các tệp kiểm thử. Sự tồn tại của chúng chưa xác nhận toàn bộ pipeline chạy đúng.
- Tracker hiện có 51 dòng, gồm 48 `completed`, 2 `anomaly`, 1 `planned`. Cả 51 vẫn là `unverified`. Dense ViANLI có ghi nhận ba backbone với seeds 42, 43, 44. Phần lớn so sánh MoE là một seed.
- Có tệp metrics của Hybrid-MoE, nhưng hồ sơ ghi thiếu resolved config/checkpoint/history để kiểm chứng nguồn gốc. Có bảng tổng hợp ViHSD lịch sử thiếu metadata.
- Hồ sơ ngày 04/11 đã rà hướng nghiên cứu và phát hiện công trình ViANLI/NLIMoE liên quan trực tiếp. Đóng góp cần được định vị bằng so sánh có kiểm soát, độ ổn định, chi phí và hành vi expert.

### Những gì còn quyết định tiến độ

- Khôi phục bằng chứng cho các lượt sẽ đưa vào bảng chính. Chỉ tái chạy những lượt cần thiết mà bằng chứng không khôi phục được.
- Hoàn tất các đối chứng Dense/MoE trên cả hai bộ dữ liệu với seeds chung, cùng quy trình lựa chọn theo validation.
- Chốt câu hỏi, số ablation và cách so sánh backbone với giảng viên. Phạm vi thực nghiệm hiện tại bao phủ nhiều biến hơn quỹ thời gian an toàn.
- Viết báo cáo, làm demo, nhận phản hồi và hoàn thiện hồ sơ. Trong phạm vi tệp đã kiểm tra chưa thấy bản thảo luận văn hoàn chỉnh, chỉ thấy mẫu LaTeX và tài liệu đề cương.
- Bản kiểm tra môi trường ngày 04/11 ghi PyTorch CPU và CUDA không khả dụng. Khả năng sử dụng GPU hiện tại, quota Kaggle/Colab và thời lượng mỗi lượt chưa được xác minh.

Không thể suy ra phần trăm hoàn thành đáng tin cậy chỉ từ số cấu hình hay số dòng tracker. Theo lịch đề cương, giai đoạn hiện nay đáng lẽ đã chuyển sang phân tích. Khối bằng chứng thực nghiệm còn cần đóng trước khi xem mốc đó là hoàn thành.

## Phạm vi đề xuất để hoàn thành đúng hạn

Đây là phương án ưu tiên để trao đổi với giảng viên, chưa thay thế phạm vi đã được duyệt hoặc các quyết định trong `code/docs/DECISIONS.md`.

1. Giữ cả ViANLI và ViHSD. Câu hỏi chính là Dense/MoE khác nhau thế nào về chất lượng, độ ổn định và chi phí trong điều kiện được kiểm soát.
2. Giữ khảo sát backbone đã có nếu kiểm chứng được. Với thí nghiệm bổ sung, chốt backbone bằng validation và ngân sách. Mỗi cặp Dense/MoE phải cùng backbone và tiền xử lý. Nếu hai bộ dữ liệu dùng backbone khác nhau, kết luận chuyển giao phải nêu yếu tố gây nhiễu này.
3. Bộ đối chứng chính gồm Dense và một Transformer-MoE ổn định trên hai bộ dữ liệu, seeds 42/43/44. Đó là **12 vị trí cấu hình–seed cần có bằng chứng**. Lượt lịch sử kiểm chứng được có thể lấp các vị trí này.
4. Thêm một routing thay thế trên mỗi bộ dữ liệu với cùng seeds nếu tài nguyên cho phép, tương đương tối đa 6 vị trí bổ sung. Dùng cùng placement/expert count để cô lập routing. Chốt trước lựa chọn và cách xử lý cấu hình không ổn định.
5. Placement, số expert và shared expert dùng làm khảo sát phụ khi hồ sơ đủ tin cậy. Chỉ đưa thêm vào lịch chạy khi cần trả lời câu hỏi cụ thể đã chốt. Ghi rõ khảo sát một seed và xác nhận nhiều seed.
6. Classifier-MoE/Hybrid-MoE được giữ trong hồ sơ. Dành tối đa hai ngày làm việc cho chẩn đoán có mục tiêu. Nếu chưa xử lý xong, ghi giới hạn và xin giảng viên thống nhất mức trình bày. Số liệu thấp tự nó chưa xác nhận collapse hay nguyên nhân.
7. Demo nhỏ tải checkpoint đã chọn, nhận đầu vào đúng định dạng từng tác vụ và trả nhãn. ViANLI cần cặp premise/hypothesis. ViHSD cần văn bản bình luận. Đầu ra phải khớp pipeline đánh giá.
8. Báo cáo phải có phương pháp, bảng chính có nguồn gốc, phân tích lỗi/per-class, hiệu quả tài nguyên đo được, giới hạn và hướng dẫn tái lập. Kết quả MoE ngang hoặc kém Dense vẫn có thể trả lời câu hỏi nếu thực nghiệm hợp lệ.

Nếu giảng viên yêu cầu toàn bộ họ kiến trúc hoặc thêm backbone/ablation, cần thay một phần việc trong lịch và tính lại GPU-hours ngay khi chốt ngày 13/11.

## Nhịp làm việc mỗi ngày

Ngày làm chính dành khoảng **6 giờ tập trung**, chia ba khối 2 giờ. Nghỉ giữa khối và để thêm tối đa một giờ xử lý tải dữ liệu, sao lưu, trao đổi. Khi GPU chạy, chuyển sang viết hoặc đọc tài liệu.

- Khối 1 làm việc khó nhất của ngày, như kiểm chứng dữ liệu, sửa lỗi chặn hoặc phân tích kết quả.
- Khối 2 chuẩn bị/thu kết quả thực nghiệm, kiểm tra artifact, cập nhật tracker và bảng số liệu.
- Khối 3 viết báo cáo, bổ sung nguồn và rà tính nhất quán. Từ 11/11 đã bắt đầu viết, mỗi ngày làm chính phải có một đầu ra văn bản hoặc hình/bảng dùng được.
- Cuối ngày dành 15 phút ghi việc hoàn thành, bằng chứng và bước đầu tiên ngày mai. Đánh dấu hoàn thành khi đầu ra tồn tại và đã kiểm tra.
- Các ngày 15/11, 22/11, 29/11, 06/12, 13/12, 20/12 là ngày nhẹ khoảng 2–3 giờ. Phần thời gian còn lại dành nghỉ và phục hồi.

Các khối là ngân sách thời gian. Nếu một lỗi chiếm hết hai khối liên tiếp, ghi lại lỗi, chuyển sang phần viết độc lập và xử lý tại buổi kỹ thuật kế tiếp. Lỗi ảnh hưởng tính đúng của bảng chính phải đóng trước khi dùng kết quả đó.

## Ngân sách thực nghiệm và điều kiện tiếp tục

Trong ngày 14–16/11, đo thời lượng một lượt Dense và một lượt MoE đại diện trên GPU thực tế. Dùng các lượt hợp lệ này cho nghiên cứu nếu chúng đáp ứng protocol.

GPU-hours cần dự kiến bằng tổng thời lượng các lượt còn thiếu, cộng ít nhất 25% cho lỗi và tái chạy. Với các lượt có độ dài khác nhau, cộng riêng từng loại. So sánh với quota thực sự có thể dùng trước 30/11 và lịch truy cập GPU. Không lấy thời gian CPU hoặc smoke test để ngoại suy trực tiếp thời gian full run.

- Đến **16/11**, phải có đường chạy GPU dùng được, artifact lưu bền và dự toán hoàn thành bộ chính. Nếu thiếu, trao đổi giảng viên trong ngày, ưu tiên bộ đối chứng chính và bằng chứng cũ có thể kiểm chứng.
- Đến **23/11**, phải có ít nhất một cặp Dense/MoE có bằng chứng hợp lệ trên mỗi bộ dữ liệu và lịch hoàn thành các seed còn lại.
- Đến **27/11**, ưu tiên đóng 12 vị trí chính. Nếu thiếu, dành 28–30/11 cho đối chứng chính, giảm số ablation bổ sung sau khi thống nhất. Không giảm yêu cầu kiểm chứng hay đổi tiêu chí chọn theo test.
- Đến **30/11**, đóng lựa chọn kiến trúc/hyperparameter. Sau mốc này chỉ tái chạy để sửa lỗi hoặc hoàn thiện lượt xác nhận đã định trước, có ghi lý do.
- Đến **03/12**, mục tiêu đóng bảng chính và đánh giá test cuối cho tập cấu hình/seed đã khóa. Nếu còn thiếu bằng chứng, để trống có giải thích hoặc thu hẹp kết luận với giảng viên.
- Đến **08/12**, có bản thảo đầy đủ. Chương còn thiếu số liệu phải nêu đúng trạng thái, không điền số dự kiến.

Các mốc phụ thuộc lẫn nhau. Khi trượt một mốc, cập nhật các ngày bị ảnh hưởng trong checklist, giữ thời gian sửa bản thảo và hạn nộp. Phần mở rộng được giảm trước.

## Quy tắc bảo toàn độ tin cậy

- Mỗi lượt dùng trong bảng chính cần run ID, config thực tế, seed, phiên bản mã/môi trường và dữ liệu, metrics cùng checkpoint hoặc artifact phù hợp với kết luận. Ghi đường dẫn lưu bền trong tracker.
- Phân biệt kiểm chứng artifact của lượt cũ với tái lập bằng lượt mới. Lượt mới có ID mới và lý do tái chạy, giữ nguyên lịch sử.
- Dùng validation macro F1 để chọn. Ghi tập finalist, quy tắc lựa chọn và seeds trước khi đánh giá test. Lưu test lịch sử làm hồ sơ, không dùng để hướng dẫn lựa chọn mới. Báo cáo việc test đã từng được xem nếu có liên quan.
- Báo mean và sample standard deviation trên các seed chung. Chỉ kết luận có ý nghĩa thống kê khi đã chọn phép kiểm định phù hợp và đáp ứng giả định. Ba seed cho độ chắc chắn còn hạn chế.
- Không so raw auxiliary loss trực tiếp với task loss để kết luận auxiliary chi phối tối ưu. Hồ sơ 04/11 đã chỉ ra cần xét hệ số trọng số và tổng loss thực tế.
- Đo latency/VRAM trên cùng phần cứng, batch size, độ dài đầu vào và quy trình warm-up. Nếu thiếu phép đo FLOPs/active parameters đáng tin cậy, trình bày metric đo được và giới hạn.
- Sao lưu bản thảo, cấu hình, bảng số liệu và artifact quan trọng mỗi ngày. Kiểm tra khả năng khôi phục ít nhất mỗi tuần. Thư mục results/checkpoints có thể bị Git bỏ qua.

## Mốc bàn giao

- **13/11** có gói câu hỏi, phạm vi tối thiểu và yêu cầu nộp để giảng viên duyệt.
- **17/11** có protocol và bảng kiểm chứng nguồn gốc, pipeline đại diện đã chạy.
- **23/11** có đối chứng ban đầu trên hai bộ dữ liệu, chương dữ liệu/phương pháp đang thành hình.
- **30/11** có bảng validation chính và phạm vi thực nghiệm đã khóa.
- **03/12** có bảng kết quả cuối cùng theo protocol và bản phân tích đầu tiên.
- **08/12** gửi bản thảo đầy đủ cho giảng viên, đề nghị phản hồi trước 12/12.
- **15/12** đóng vòng sửa nội dung chính. Thời gian phản hồi là giả định lập lịch cần xác nhận với giảng viên.
- **19/12** bản sẵn nộp, demo và gói tái lập đã kiểm tra.
- **21/12** slide và buổi tập bảo vệ hoàn tất.
- **22–26/12** dự phòng góp ý muộn, thủ tục và nộp. Nếu đã đủ điều kiện, nộp sớm và giữ biên nhận.

## Nguồn và giới hạn của đánh giá

Đã đọc tracker hiện tại, `code/AGENTS.md`, các tài liệu `STATUS.md`, `PROJECT.md`, `THESIS_SPECIFICATION.md`, `DECISIONS.md`, kế hoạch/kết quả/ghi chú thực nghiệm và audit `code/idea-stage/IDEA_REPORT_2026-11-04.md`. Đã xem requirements và các điểm lưu cấu hình/đánh giá trong `evaluate.py`. `STATUS.md` cập nhật 29/09 nên được đối chiếu với hồ sơ 04/11 và tracker hiện tại.

Chưa chạy huấn luyện, kiểm thử hay đo GPU trong lần lập lịch này. Audit 04/11 ghi lượt unit test bị ngắt, nên chưa có bằng chứng toàn bộ test đã pass. Các vấn đề môi trường trong tài liệu cũ cần kiểm tra lại trên phiên bản hiện tại, vì requirements đã thay đổi. Chưa xác minh artifact bên ngoài máy, bản thảo ngoài workspace, quy định nộp/bảo vệ hay yêu cầu bài báo. Nếu có yêu cầu bài báo được chấp nhận trước 26/12, cần đánh giá riêng thời gian phản biện bên ngoài.
