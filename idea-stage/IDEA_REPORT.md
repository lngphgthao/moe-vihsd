# Quy trình nghiên cứu và rà soát bằng chứng ban đầu

Ngày rà soát gốc 2026-11-04, múi giờ Asia/Saigon  
Mã đợt nghiên cứu `vietnamese-moe-2026-11-04`  
Trạng thái **BLOCKED tại bước chấp thuận Giai đoạn 1. Quy trình nghiên cứu chưa hoàn tất.**

## Phạm vi và tình hình thực hiện

Thư mục thực thi là `code/` trong không gian làm việc của luận văn. Hướng nghiên cứu đã xác lập là so sánh thực nghiệm có kiểm soát giữa mô hình dense và MoE trên ViANLI và ViHSD, với các backbone XLM-R, PhoBERT và mBERT. Việc đề xuất một kiến trúc mới nằm ngoài phạm vi hiện tại. Các tệp `docs/THESIS_SPECIFICATION.md`, `experiments/EXPERIMENT_PLAN.md`, bảng theo dõi thí nghiệm, kết quả, ghi chú phân tích và mã nguồn cung cấp bối cảnh nghiên cứu. Đợt rà soát giữ nguyên câu hỏi nghiên cứu, bộ dữ liệu, chính sách lựa chọn mô hình, kết quả lịch sử và phần triển khai huấn luyện.

Các giá trị mặc định được áp dụng gồm `AUTO_PROCEED=true`, `CODE_REVIEW=true`, `ARXIV_DOWNLOAD=false` và `AUTO_WRITE=false`. Hội nghị hoặc tạp chí mục tiêu chưa được xác định. Số giờ GPU sử dụng trong đợt này là **0**. Việc mua tài nguyên đám mây, chạy thí nghiệm GPU, gửi thông báo và đánh giá tập test đều chưa được thực hiện. Chi phí tính toán của các thí nghiệm lịch sử chưa rõ.

Cây làm việc đã có các thay đổi của người dùng trong phần huấn luyện, đánh giá, nạp dữ liệu, notebook, thư viện phụ thuộc, bảng theo dõi và tài liệu. Các thay đổi đó được giữ nguyên. Báo cáo phản ánh cây làm việc tại thời điểm rà soát. Phiên bản mã nguồn thực sự tạo ra các kết quả lịch sử vẫn cần được xác minh.

## Tổng quan tài liệu

Đợt tìm kiếm web có trọng tâm ngày 2026-11-04 đã kiểm tra công trình trực tiếp liên quan đến benchmark và tìm thêm các đầu mối gần đây về phân lớp tiếng Việt bằng MoE. Mức độ bao phủ hiện tại là sơ bộ. Khảo sát toàn diện và đánh giá tính mới vẫn cần được hoàn thành.

- Huynh, Nguyen và Nguyen công bố bài *A new benchmark dataset and mixture-of-experts language models for adversarial natural language inference in Vietnamese*, trên Expert Systems with Applications, tập 306, bài 130109, năm 2026. Phần tóm tắt giới thiệu ViANLI và NLIMoE, sử dụng cơ chế định tuyến động được học phía trên một bộ mã hóa dùng chung. Vì vậy, riêng việc áp dụng MoE cho ViANLI hoặc so sánh định tuyến động với top-k chưa đủ để tuyên bố đóng góp đầu tiên. Cần đọc phương pháp và giao thức đánh giá đầy đủ trước khi xác nhận mức độ tái lập của mã nguồn hiện tại. [DOI của nhà xuất bản](https://doi.org/10.1016/j.eswa.2025.130109)
- Thẻ dữ liệu ViHSD của nhóm tác giả cung cấp nguồn benchmark và trích dẫn Luu, Nguyen và Nguyen, năm 2021. Cần xác minh tiền xử lý, thứ tự nhãn, cách chia tập và chỉ số đánh giá trước khi so sánh kết quả giữa các công trình. [Thẻ dữ liệu ViHSD](https://huggingface.co/datasets/uitnlp/vihsd)
- Kết quả tìm kiếm có mô hình phân tích cảm xúc BamiBERT MoE. Thẻ mô hình dẫn đến DOI `10.18653/v1/2026.semeval-1.25`. Cần kiểm tra bài báo gốc trước khi sử dụng trích dẫn này hoặc xác nhận tính tương đồng về tác vụ và giao thức. Đầu mối này cũng đòi hỏi sự thận trọng với tuyên bố rộng rằng đây là mô hình phân lớp MoE đầu tiên cho tiếng Việt. [Thẻ mô hình BamiBERT MoE](https://huggingface.co/TheSon2202/bamibert-moe-sentiment?library=transformers)
- Thư mục `../papers/` chứa bốn tệp PDF. Thông tin thư mục của chúng đã có trong `docs/THESIS_SPECIFICATION.md`. Đợt này chỉ kiểm kê các tệp, việc đọc lại độc lập và tải thêm PDF chưa được thực hiện. Khả năng tìm kiếm bằng Gemini chưa khả dụng trong danh sách lệnh và công cụ đã kiểm tra.

Đóng góp dự kiến là một nghiên cứu có kiểm soát và có thể tái lập về chất lượng phân lớp, độ ổn định, chi phí và hành vi chuyên gia trên hai tác vụ đã xác lập. Tính mới đang ở trạng thái **UNCONFIRMED**, chờ đối chiếu bài báo gốc và phản biện bên ngoài.

## Xếp hạng hướng nghiên cứu

Các hướng dưới đây được xếp hạng sơ bộ trong phạm vi hiện tại. Việc sinh ý tưởng qua mô hình bên ngoài và chấp thuận ý tưởng vẫn chưa hoàn thành. Số thí nghiệm thăm dò đã chạy là 0. Việc xếp hạng chỉ dựa trên khoảng trống bằng chứng và các bước kiểm tra cần thiết, độc lập với chỉ số test lịch sử.

| Hạng | Hướng điều tra | Bằng chứng cần bổ sung và bước kiểm tra tiếp theo | Trạng thái |
| --- | --- | --- | --- |
| 1 | Thiết lập so sánh dense và MoE đáng tin cậy | Khôi phục cấu hình đã phân giải đầy đủ cùng các tệp kết quả của lần chạy hiện có. Chạy tái lập những lần thiếu bằng chứng theo giao thức đã công bố. So sánh trên cùng tập seed và báo cáo chênh lệch số tham số, chi phí tính toán | Ưu tiên tạm thời, chưa có thí nghiệm thăm dò được kiểm chứng |
| 2 | Chẩn đoán lỗi ở đầu phân lớp mức chuỗi và mô hình hybrid | Khôi phục cấu hình và lịch sử của các lần chạy bất thường. Kiểm tra nhãn, dự đoán theo lớp, định tuyến, tính hữu hạn của gradient và các thành phần loss sau khi nhân trọng số. Tái sử dụng cấu hình kiểm tra khả năng học thuộc tập rất nhỏ | Hướng dự phòng, nguyên nhân chưa rõ |
| 3 | Kiểm tra ảnh hưởng của định tuyến và vị trí đặt MoE trên hai tác vụ | Trước hết thiết lập đối chứng ổn định với backbone và kiến trúc tương ứng. Thay đổi từng yếu tố và xác nhận trên nhiều seed trước khi kết luận khả năng khái quát sang tác vụ khác | Chờ kiểm chứng các đối chứng |

Ưu tiên xử lý các khoảng trống bằng chứng trước khi bổ sung backbone, kiến trúc, bộ dữ liệu hoặc quét toàn bộ tổ hợp cấu hình. Chưa có ứng viên nào bị loại dựa trên thực nghiệm trong đợt rà soát này.

## Rà soát bằng chứng

Các nguồn chuẩn gồm `experiments/experiment_tracker.csv`, `experiments/RESULTS.md` và `experiments/analysis_notes.md`. Cần phân biệt chỉ số dạng phần trăm trong bảng theo dõi với chỉ số dạng tỷ lệ trong tệp đã lưu.

- Bảng theo dõi ghi nhận các baseline dense và các đợt thử cấu hình MoE trước đây. Các trường xác minh vẫn mang trạng thái `unverified`. Mỗi lần chạy cần bằng chứng riêng trước khi được nâng trạng thái kiểm chứng. Tên tệp, cấu hình và trạng thái hoàn thành do người thực hiện báo cáo chỉ hỗ trợ việc truy vết.
- Tệp `results/vianli-hybrid-moe-xlmr-last6-e6-dynamic-head-e4-top2-s42/run_metrics.json` ghi nhận macro F1 trên validation là **0.3751085481**, F1 lớp contradiction trên tập huấn luyện là **0.0793882010**, classification loss là **1.0072895288**, balance loss là **21.4069826285**, dynamic loss là **1.7176198889** và tổng loss huấn luyện là **1.2230769708**. Đây là các giá trị đọc từ một tệp đã lưu. Nguồn gốc lần chạy và nguyên nhân chất lượng thấp vẫn cần xác minh.
- Cách diễn giải loss hiện tại cần được hiệu chỉnh. Auxiliary loss thô **23.1246025174** là tổng chưa nhân trọng số. `train.py` tối ưu biểu thức `classification_loss + balance_factor * balance_loss + dynamic_factor * dynamic_loss`. Cấu hình hybrid hiện tại đặt hai hệ số là 0.01 và 0.001. Áp dụng vào các giá trị thô đã lưu cho phần đóng góp phụ khoảng **0.2157874421**, phù hợp với hiệu giữa tổng loss và classification loss. Sự phù hợp này là một kiểm tra nhất quán số học, còn cấu hình đã phân giải của lần chạy lịch sử vẫn đang thiếu. Vì vậy, phép so sánh auxiliary loss thô 23.12 với task loss 1.01 chưa đủ để kết luận thành phần phụ chi phối giá trị loss được tối ưu. Ảnh hưởng lên gradient cần được đo riêng.
- Giá trị macro F1 tham chiếu do dự đoán ngẫu nhiên trong bài toán ba lớp phụ thuộc vào phân bố nhãn và cách dự đoán. Để chẩn đoán hiện tượng suy sụp mô hình, cần kiểm tra phân bố lớp, ma trận nhầm lẫn, dự đoán và các baseline phù hợp bên cạnh điểm số thấp. Trạng thái bất thường của các lần chạy lịch sử được duy trì trong khi điều tra.
- `.agent/context.yaml` ghi rằng kết quả đã được xác minh thủ công, trong khi bảng theo dõi ghi rõ `unverified`. Ưu tiên trạng thái trong bảng theo dõi cho đến khi bằng chứng của từng lần chạy giải quyết được mâu thuẫn này.
- `configs/groups/vianli_stage1_dense_backbones.yaml` hiện yêu cầu seed **43 và 44**, trong khi kế hoạch xác nhận dùng chung các seed 42/43/44. Cần kiểm tra bảng theo dõi và tệp kết quả trước khi chạy. Các lần chạy lại phải có lý do tái lập hoặc bổ sung nguồn gốc rõ ràng và mã lần chạy mới.
- Tài liệu đặc tả đã ghi nhận sự thiếu nhất quán giữa chú thích, cấu hình ứng viên cuối cùng và thông tin nguồn gốc. Việc lựa chọn mô hình cần có hồ sơ xác nhận riêng, kể cả khi tên tệp chứa `best`.

## Kiểm chứng tính mới

**NOT ACCEPTED — chưa được chấp thuận.** Đã có công trình trực tiếp về MoE trên ViANLI. Việc tìm kiếm đầy đủ công trình gần đây, đối chiếu bài báo gốc và lấy kết luận tính mới từ phản biện bên ngoài kèm danh tính xác thực vẫn chưa hoàn thành. Hồ sơ hiện chưa có biên nhận kiểm chứng tính mới hoặc danh tính người phản biện.

## Phản biện độc lập

**REVIEW_UNAVAILABLE — kênh phản biện quy định chưa khả dụng.** Skill đã cài đặt tham chiếu đến `../shared-references/reviewer-backends.md`, nhưng tệp này đang thiếu. Danh sách công cụ khả dụng cũng chưa có công cụ MCP phản biện Codex. Việc phân giải và gọi backend phản biện theo quy định bị gián đoạn với bộ cài hiện tại. Bước chấp thuận qua mô hình phản biện độc lập vẫn đang chờ thực hiện. Hồ sơ chưa ghi nhận điểm phản biện, mã luồng, kết luận tích cực hoặc giai đoạn được chấp thuận.

## Bước kiểm duyệt đang dở và cách khôi phục

`idea-discovery/SKILL.md` yêu cầu `run_state.py` và `idea_discovery_gate.py`. Theo quy định của skill, báo cáo cuối phải mang trạng thái `BLOCKED` nếu thiếu một trong hai công cụ. Các vị trí chuẩn đã kiểm tra gồm `.aris/tools` và `tools` trong repo, `$ARIS_REPO` và con trỏ `~/.aris/repo`. Các thư mục công cụ trong repo và thư mục `.aris` của người dùng đều vắng mặt, còn biến `ARIS_REPO` chưa được đặt. Các tài liệu dùng chung về đầu ra và phản biện cũng thiếu trong cây skill đã cài. Phạm vi kiểm tra này chỉ xác nhận tình trạng tại các vị trí đã nêu.

Tài liệu hiện tại lưu lại kết quả rà soát. Tệp trạng thái do công cụ quản lý và việc chấp thuận Giai đoạn 1 vẫn đang thiếu. Thiếu bước kiểm duyệt bắt buộc đang chặn triển khai Giai đoạn 2 và phản biện Giai đoạn 3. `AUTO_PROCEED` tiếp tục giữ giá trị `true`.

Thứ tự khôi phục được đề xuất như sau.

1. Khôi phục các tệp dùng chung mà gói skill tham chiếu, cùng công cụ quản lý trạng thái, kiểm tra bằng chứng và định tuyến phản biện. Thiết lập backend được hỗ trợ có thể hoạt động, lưu danh tính thực tế trả về và mã tham chiếu bền vững đến kết luận phản biện.
2. Khởi tạo đợt chạy có khả năng tiếp tục với mã `vietnamese-moe-2026-11-04` bằng công cụ chuẩn và nạp báo cáo này làm bối cảnh một phần. Hoàn thành bằng chứng tính mới và phản biện bắt buộc trước khi chấp thuận Giai đoạn 1.
3. Khôi phục cấu hình đã phân giải, phiên bản mã nguồn, phiên bản dữ liệu và mô hình, checkpoint, lịch sử huấn luyện, dự đoán và siêu dữ liệu thực thi của các lần chạy hiện có. Bảo toàn các dòng lịch sử và ghi rõ mức độ xác minh tệp kết quả, cũng như mức độ tái lập thực nghiệm.
4. Với đối chứng cần chạy lại vì thiếu bằng chứng có thể khôi phục, xác định trước yếu tố so sánh duy nhất, tiền xử lý, ngân sách huấn luyện và lựa chọn, seed, cách tính số tham số và chi phí, mã lần chạy duy nhất, đầu ra dự kiến và lý do tái lập. Tái sử dụng `scripts/run_group.py`, bộ huấn luyện và cơ chế quản lý cấu hình hiện có.
5. Với bất thường ở đầu phân lớp, tái sử dụng `configs/diagnostics/vianli_classifier_moe_tiny_overfit.yaml` sau khi phản biện. Cấu hình hiện trỏ đến dữ liệu trên Hub và chạy 50 epoch chẩn đoán. Cần kiểm tra chi phí và bảo đảm tiền xử lý khớp với đối chứng trước khi chạy.
6. Trên môi trường GPU đã được cho phép sử dụng, chạy các kiểm tra cơ bản đã qua phản biện và thí nghiệm nhỏ nhất có thể phân biệt các giả thuyết. Chỉ mở rộng các so sánh ổn định sang seed 42/43/44, sau đó đến các ablation được lựa chọn và bộ dữ liệu thứ hai. Ngân sách tính toán trả phí mới cần được xác định riêng.
7. Lựa chọn mô hình theo macro F1 trên validation, báo cáo đầy đủ các seed cùng độ lệch chuẩn mẫu, lưu bằng chứng chi phí và định tuyến. Ghi lại quyết định khóa cấu hình ứng viên cuối cùng và tập seed trước khi đánh giá test cuối cùng.

Kiểm tra môi trường cục bộ cho kết quả PyTorch **2.14.0+cpu**, CUDA chưa khả dụng và số GPU nhìn thấy là 0. Môi trường ảo đã kiểm tra chỉ hỗ trợ huấn luyện bằng CPU. Khả năng sử dụng GPU hoặc môi trường khác trên máy cần kiểm tra riêng. Việc truy cập phiên Kaggle hoặc Colab bên ngoài chưa được thực hiện trong đợt này.

## Kiểm tra và trạng thái từng giai đoạn

Kết quả kiểm tra được ghi trong `validation_2026-11-04.txt`. Bộ kiểm thử đơn vị đã bị ngắt khi chưa trả về kết quả, nên trạng thái kiểm tra là chưa hoàn tất. Kiểm thử đơn vị kiểm tra các bất biến của phần triển khai. Bằng chứng cho thí nghiệm thăm dò, xác minh lần chạy lịch sử và phản biện độc lập được quản lý ở các bước tương ứng.

| Giai đoạn | Trạng thái |
| --- | --- |
| Khám phá ý tưởng | Đã ghi nhận rà soát một phần. Chấp thuận ở trạng thái BLOCKED do thiếu công cụ bắt buộc và bằng chứng phản biện qua kênh quy định |
| Triển khai thí nghiệm | Chưa bắt đầu huấn luyện hoặc triển khai |
| Vòng lặp phản biện tự động | Chưa bắt đầu, chưa có điểm hoặc kết luận tích cực |
| Tổng kết và bàn giao viết bài | Chưa bắt đầu, báo cáo tổng hợp cuối cùng và xác nhận hoàn tất nghiên cứu đang chờ các giai đoạn trước |
| Viết bài | Tắt theo mặc định, hội nghị hoặc tạp chí mục tiêu chưa được xác định |

Các đầu ra `NARRATIVE_REPORT.md`, PDF bài báo, bản cam kết nghiên cứu cho ý tưởng được chấp thuận và trạng thái đợt chạy đã được chấp thuận còn chờ các bước kiểm duyệt. Bước xuất HTML chưa được thực hiện. Kế hoạch hiện có tiếp tục là nguồn kế hoạch chuẩn của dự án.
