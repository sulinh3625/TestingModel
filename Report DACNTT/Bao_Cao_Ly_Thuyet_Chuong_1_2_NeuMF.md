# TÓM TẮT BÁO CÁO VÀ LÝ THUYẾT CHƯƠNG 1, CHƯƠNG 2

---

## TÓM TẮT (ABSTRACT)

### XÂY DỰNG MÔ HÌNH LAI MATRIX FACTORIZATION VÀ DEEP NEURAL NETWORK (NEUMF) ĐỂ GỢI Ý SẢN PHẨM

**TÓM TẮT**  
Trong kỷ nguyên thương mại điện tử và kinh tế số, hệ thống gợi ý (Recommendation System) đóng vai trò then chốt trong việc tối ưu hóa trải nghiệm người dùng và nâng cao doanh thu doanh nghiệp. Các phương pháp Lọc cộng tác (Collaborative Filtering - CF) truyền thống như Nhân tử hóa ma trận (Matrix Factorization - MF) chủ yếu sử dụng tích vô hướng (inner product) để mô hình hóa tương tác giữa người dùng (User) và sản phẩm (Item). Tuy nhiên, việc phụ thuộc vào tuyến tính hóa khiến MF khó nắm bắt được các mối quan hệ phức tạp và phi tuyến trong không gian đặc trưng ẩn.

Khóa luận đề xuất giải pháp xây dựng mô hình khuyến nghị lai Neural Matrix Factorization (NeuMF) kết hợp giữa Generalized Matrix Factorization (GMF) và Multi-Layer Perceptron (MLP). Mô hình tận dụng ưu thế tuyến tính của GMF cùng khả năng học các mối liên hệ phi tuyến sâu của MLP để nâng cao độ chính xác trong việc dự đoán hành vi tương tác dựa trên tín hiệu phản hồi ẩn (Implicit Feedback).

Nghiên cứu được triển khai thực nghiệm trên bộ dữ liệu chuỗi cung ứng và thương mại điện tử DataCo Supply Chain. Quy trình xử lý dữ liệu xây dựng ma trận tương tác User–Item, áp dụng kỹ thuật lấy mẫu tiêu cực (Negative Sampling) và phương pháp đánh giá Leave-One-Out. Kết quả kiểm chứng định lượng qua các chỉ số HR@K, NDCG@K, Precision@K và Recall@K chứng minh mô hình NeuMF đạt hiệu năng vượt trội so với các mô hình thành phần độc lập (GMF, MLP) và các phương pháp cơ sở cổ điển. Đồng thời, nghiên cứu thực hiện phân tích Ablation Study, đánh giá hiện tượng Cold-start và đóng gói mô hình thành dịch vụ API (FastAPI) tích hợp giao diện demo (Streamlit) bằng Docker, khẳng định tính hiệu quả học thuật và khả năng ứng dụng thực tiễn.

---

### ABSTRACT

**BUILDING A HYBRID MATRIX FACTORIZATION AND DEEP NEURAL NETWORK MODEL (NEUMF) FOR PRODUCT RECOMMENDATION**

**ABSTRACT**  
In the era of e-commerce and the digital economy, recommendation systems play a pivotal role in optimizing user experience and driving business revenue. Traditional Collaborative Filtering (CF) approaches, such as Matrix Factorization (MF), primarily rely on the inner product to model interactions between users and items. However, this linear reliance restricts MF from capturing complex, non-linear relationships within latent feature spaces.

This thesis proposes a hybrid recommendation framework based on Neural Matrix Factorization (NeuMF), which integrates Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). The architecture leverages the linear capability of GMF alongside the deep non-linear learning power of MLP to enhance prediction accuracy based on implicit feedback signals.

The empirical evaluation is conducted on the DataCo Supply Chain dataset. The data pipeline constructs a User-Item interaction matrix, implements Negative Sampling techniques, and applies a Leave-One-Out evaluation protocol. Quantitative results evaluated via HR@K, NDCG@K, Precision@K, and Recall@K demonstrate that the proposed NeuMF model outperforms standalone components (GMF, MLP) and baseline methods. Furthermore, the study presents an ablation analysis, evaluates cold-start scenarios, and deploys the model as a FastAPI service with a Streamlit interface packaged via Docker, proving both theoretical depth and real-world applicability.

---

## CHƯƠNG 1. TỔNG QUAN VỀ ĐỀ TÀI

### 1.1 Bối cảnh nghiên cứu và Tính cấp thiết của đề tài

#### 1.1.1 Thực trạng hệ thống gợi ý trong thương mại điện tử và chuỗi cung ứng
Sự bùng nổ của các nền tảng thương mại điện tử và giao dịch trực tuyến tạo ra lượng dữ liệu sản phẩm và người dùng khổng lồ. Đối mặt với tình trạng quá tải thông tin (information overload), hệ thống gợi ý trở thành công cụ cốt lõi giúp cá nhân hóa trải nghiệm khách hàng, giảm chi phí tìm kiếm và gia tăng tỷ lệ chuyển đổi đơn hàng. Trong môi trường chuỗi cung ứng và bán lẻ, việc gợi ý chính xác sản phẩm phù hợp không chỉ nâng cao mức độ hài lòng của người dùng mà còn hỗ trợ doanh nghiệp dự báo nhu cầu tiêu thụ và tối ưu hóa luồng hàng hóa.

#### 1.1.2 Hạn chế của các phương pháp Lọc cộng tác truyền thống
Lọc cộng tác (Collaborative Filtering - CF), đặc biệt là các phương pháp Nhân tử hóa ma trận (Matrix Factorization - MF) như SVD hay ALS, được sử dụng phổ biến nhờ tính đơn giản và hiệu quả. MF biểu diễn User và Item dưới dạng các vector ẩn (latent vectors) trong không gian có số chiều thấp và dự đoán mức độ tương tác thông qua tích vô hướng.

Tuy nhiên, tích vô hướng là một kết hợp tuyến tính đơn giản của các đặc trưng ẩn. Phương pháp này không thể học được các tương tác phức tạp và phi tuyến giữa các chiều đặc trưng của User và Item. Sự hạn chế này làm giảm độ chính xác của mô hình khi xử lý ma trận tương tác có độ thưa cao (data sparsity) hoặc chứa các biểu hiện hành vi ẩn (implicit feedback) phức tạp.

#### 1.1.3 Sự cần thiết của việc kết hợp Matrix Factorization và Deep Learning (NeuMF)
Để khắc phục rào cản tuyến tính của MF, các nghiên cứu Học sâu (Deep Learning) được ứng dụng nhằm khai thác khả năng học biểu diễn đặc trưng phi tuyến. Mạng nơ-ron truyền thẳng đa tầng (Multi-Layer Perceptron - MLP) có khả năng học các mối quan hệ phi tuyến bậc cao giữa User và Item.

Mô hình Neural Collaborative Filtering (NCF/NeuMF) do He et al. đề xuất kết hợp ưu điểm của cả hai tiếp cận: sử dụng Generalized Matrix Factorization (GMF) để duy trì khả năng học tương tác tuyến tính cố định và kết hợp mạng MLP để mở rộng khả năng học quan hệ phi tuyến sâu. Việc ứng dụng kiến trúc NeuMF vào bài toán gợi ý sản phẩm giúp nâng cao năng lực dự đoán, tạo nền tảng cho việc triển khai các hệ thống gợi ý cá nhân hóa có độ chính xác cao.

### 1.2 Mục tiêu và Phạm vi nghiên cứu

#### 1.2.1 Mục tiêu nghiên cứu
- Nghiên cứu cơ sở lý thuyết về Lọc cộng tác, Nhân tử hóa ma trận, Mạng nơ-ron sâu và kiến trúc mô hình lai NeuMF.
- Xây dựng quy trình xử lý dữ liệu (Data Pipeline) hoàn chỉnh trên bộ dữ liệu DataCo Supply Chain, biến đổi dữ liệu giao dịch thành ma trận tương tác User–Item dạng tín hiệu ẩn (Implicit Feedback), thực hiện lấy mẫu tiêu cực (Negative Sampling) và chia tập dữ liệu huấn luyện/kiểm thử.
- Cài đặt, huấn luyện và tinh chỉnh các mô hình GMF, MLP độc lập và mô hình lai NeuMF.
- Đánh giá thực nghiệm định lượng hiệu năng của mô hình NeuMF so với các nhánh thành phần và mô hình baseline cổ điển thông qua các chỉ số HR@K, NDCG@K, Precision@K, Recall@K; thực hiện Ablation Study và đánh giá trường hợp Cold-start.
- Đóng gói mô hình thành dịch vụ API (FastAPI) và giao diện ứng dụng demo (Streamlit) bằng Docker để chứng minh khả năng triển khai thực tế.

#### 1.2.2 Phạm vi nghiên cứu
- **Đối tượng nghiên cứu:** Kiến trúc mô hình Neural Collaborative Filtering (GMF, MLP, NeuMF), kỹ thuật biểu diễn Embedding và các thuật toán đánh giá hệ thống gợi ý Top-K.
- **Dữ liệu thực nghiệm:** Bộ dữ liệu DataCo Smart Supply Chain Dataset, tập trung khai thác thông tin khách hàng (User), thông tin sản phẩm (Item) và lịch sử giao dịch (Interaction).
- **Tín hiệu tương tác:** Tín hiệu phản hồi ẩn (Implicit Feedback) được khởi tạo dưới dạng nhị phân dựa trên lịch sử mua hàng, kết hợp mở rộng thử nghiệm biến thể tương tác có trọng số theo giá trị đơn hàng.

### 1.3 Phương pháp nghiên cứu
Nghiên cứu áp dụng kết hợp phương pháp lý thuyết và phương pháp thực nghiệm:
- **Phương pháp lý thuyết:** Phân tích, tổng hợp tài liệu học thuật về hệ thống khuyến nghị, học sâu, lọc cộng tác và các bài báo khoa học nền tảng về NCF/NeuMF.
- **Phương pháp thực nghiệm:** Xây dựng mã nguồn huấn luyện mô hình bằng PyTorch, thực hiện tiền xử lý dữ liệu bằng Pandas/NumPy, theo dõi thực nghiệm qua TensorBoard/W&B và tiến hành đánh giá định lượng trên các kịch bản thực nghiệm cố định.

### 1.4 Ý nghĩa khoa học và Thực tiễn của đề tài
- **Ý nghĩa khoa học:** Đóng góp kết quả thực nghiệm định lượng chi tiết và phân tích bóc tách (Ablation Study) về vai trò của từng thành phần tuyến tính (GMF) và phi tuyến (MLP) trong kiến trúc NeuMF khi ứng dụng trên dữ liệu chuỗi cung ứng thực tế.
- **Ý nghĩa thực tiễn:** Cung cấp giải pháp đóng gói mô hình gợi ý hoàn chỉnh có khả năng tích hợp vào các hệ thống thương mại điện tử qua API, giúp doanh nghiệp cải thiện cá nhân hóa dịch vụ và tối ưu hóa hoạt động bán hàng.

### 1.5 Bố cục của khóa luận
Khóa luận được trình bày trong 6 chương chính:
- **Chương 1: Tổng quan về đề tài** – Giới thiệu bối cảnh, tính cấp thiết, mục tiêu, phạm vi, phương pháp nghiên cứu và ý nghĩa của đề tài.
- **Chương 2: Cơ sở lý thuyết và Các công trình liên quan** – Trình bày nền tảng lý thuyết về hệ thống gợi ý, Matrix Factorization, GMF, MLP, NeuMF và tổng quan các nghiên cứu tiền nhiệm.
- **Chương 3: Phân tích và Thiết kế hệ thống** – Mô tả chi tiết kiến trúc mô hình NeuMF, thiết kế pipeline xử lý dữ liệu DataCo, kỹ thuật Negative Sampling và chiến lược đánh giá Top-K.
- **Chương 4: Triển khai và Huấn luyện mô hình** – Chi tiết cài đặt mã nguồn bằng PyTorch, quy trình huấn luyện GMF, MLP, quá trình tích hợp NeuMF và tinh chỉnh siêu tham số.
- **Chương 5: Thực nghiệm và Đánh giá kết quả** – Trình bày kết quả thực nghiệm định lượng, so sánh NeuMF với các baseline, phân tích Ablation Study, đánh giá Cold-start và mô tả hệ thống API/Demo.
- **Chương 6: Kết luận và Hướng phát triển** – Tổng kết các kết quả đạt được, chỉ ra các hạn chế và đề xuất các hướng mở rộng trong tương lai.

---

## CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN

### 2.1 Tổng quan về Hệ thống gợi ý (Recommendation System)

#### 2.1.1 Khái niệm và Phân loại hệ thống gợi ý
Hệ thống gợi ý là một thuật toán lớp thông tin nhằm dự đoán mức độ quan tâm hoặc đánh giá của người dùng đối với một sản phẩm/dịch vụ cụ thể. Các hệ thống gợi ý chính bao gồm:
- **Lọc dựa trên nội dung (Content-Based Filtering):** Dựa trên đặc tính của sản phẩm và hồ sơ sở thích của người dùng trong quá khứ để gợi ý các sản phẩm tương tự.
- **Lọc cộng tác (Collaborative Filtering):** Dựa trên lịch sử tương tác chung giữa nhiều người dùng và nhiều sản phẩm mà không cần thông tin mô tả chi tiết của đối tượng.
- **Hệ thống gợi ý lai (Hybrid Recommendation System):** Kết hợp nhiều tiếp cận khác nhau nhằm khắc phục hạn chế của từng phương pháp đơn lẻ.

#### 2.1.2 Phản hồi hiện (Explicit Feedback) và Phản hồi ẩn (Implicit Feedback)
- **Explicit Feedback:** Dữ liệu thể hiện trực tiếp mức độ hài lòng của người dùng thông qua điểm số đánh giá (rating 1-5 sao) hoặc nút thích/chê. Dạng dữ liệu này có độ chính xác cao nhưng rất thưa thớt trong thực tế.
- **Implicit Feedback:** Dữ liệu ghi nhận gián tiếp qua hành vi người dùng như lịch sử mua hàng, lượt click, thời gian xem hoặc lịch sử tìm kiếm. Tín hiệu được biểu diễn dưới dạng nhị phân:

$$y_{ui} = \begin{cases} 1, & \text{nếu tương tác giữa User } u \text{ và Item } i \text{ được ghi nhận} \\ 0, & \text{ngược lại} \end{cases}$$

Mặc dù $y_{ui} = 1$ không đồng nghĩa với việc người dùng thích sản phẩm và $y_{ui} = 0$ không khẳng định người dùng ghét sản phẩm, dữ liệu phản hồi ẩn thu thập dễ dàng với quy mô lớn hơn rất nhiều.

#### 2.1.3 Thách thức độ thưa dữ liệu và Vấn đề Cold-Start
- **Độ thưa dữ liệu (Data Sparsity):** Số lượng tương tác thực tế giữa User và Item chỉ chiếm một tỷ lệ rất nhỏ trong ma trận tương tác toàn cục, gây khó khăn cho việc học biểu diễn.
- **Vấn đề Cold-Start:** Xảy ra khi một User mới hoặc Item mới xuất hiện trong hệ thống mà chưa có lịch sử tương tác, dẫn đến việc mô hình không thể tạo ra các vector biểu diễn chính xác.

### 2.2 Cơ sở lý thuyết về Lọc cộng tác và Nhân tử hóa ma trận

#### 2.2.1 Mô hình Nhân tử hóa ma trận (Matrix Factorization - MF)
Matrix Factorization chiếu cả User và Item vào một không gian đặc trưng ẩn chung (shared latent space) có số chiều $d \ll \min(M, N)$. Mỗi User $u$ được biểu diễn bởi vector $p_u \in \mathbb{R}^d$ và mỗi Item $i$ bởi vector $q_i \in \mathbb{R}^d$. Mức độ tương tác dự đoán $\hat{y}_{ui}$ được tính bằng tích vô hướng:

$$\hat{y}_{ui} = p_u^T q_i = \sum_{f=1}^{d} p_{uf} q_{if}$$

#### 2.2.2 Hạn chế của tích vô hướng trong mô hình hóa tương tác
Phép tích vô hướng kết hợp các chiều đặc trưng ẩn độc lập theo dạng tuyến tính. Điều này tạo ra rào cản lớn khi học các quan hệ phức tạp. Nếu vị trí tương quan của các User thay đổi trong không gian biểu diễn, tích vô hướng không đủ linh hoạt để mô hình hóa chính xác sự tương đồng không gian mà không làm ảnh hưởng đến mối quan hệ giữa các đối tượng khác.

### 2.3 Kiến trúc Mạng khuyến nghị thần kinh (Neural Collaborative Filtering - NCF)

#### 2.3.1 Generalized Matrix Factorization (GMF)
GMF là sự mở rộng của MF truyền thống dưới dạng biểu diễn mạng nơ-ron. Thay vì tính tổng trực tiếp tích vô hướng, GMF thực hiện phép nhân Element-wise (Hadamard product) giữa vector Embedding của User $p_u^G$ và Item $q_i^G$, sau đó đưa qua một lớp kết nối đầy đủ có hàm kích hoạt sigmoid:

$$\phi^{GMF} = p_u^G \odot q_i^G$$

$$\hat{y}_{ui} = \sigma \left( h^T \phi^{GMF} \right)$$

Trong đó $\odot$ là phép nhân từng phần tử, $h$ là trọng số của lớp đầu ra, và $\sigma(x) = \frac{1}{1 + e^{-x}}$.

#### 2.3.2 Multi-Layer Perceptron (MLP)
Đoạn nhánh MLP truyền các vector Embedding $p_u^M$ và $q_i^M$ qua phép nối vector (Concatenation) để tạo thành đầu vào cho chuỗi các lớp mạng liên tiếp:

$$z_1 = \phi^{MLP}_1 = \begin{bmatrix} p_u^M \\ q_i^M \end{bmatrix}$$

$$\phi^{MLP}_L = a_L \left( W_L^T \phi^{MLP}_{L-1} + b_L \right)$$

Trong đó $W_L, b_L, a_L$ lần lượt là trọng số, độ lệch (bias) và hàm kích hoạt (như ReLU) tại lớp thứ $L$. Kiến trúc này giúp mô hình học các tương tác phi tuyến tính phức tạp giữa đặc trưng của User và Item.

#### 2.3.3 Kiến trúc mô hình lai NeuMF (Neural Matrix Factorization)
NeuMF tích hợp hai nhánh GMF và MLP bằng cách nối vector đầu ra của cả hai nhánh trước khi đưa vào lớp dự đoán cuối cùng:

$$\phi^{NeuMF} = \begin{bmatrix} \phi^{GMF} \\ \phi^{MLP} \end{bmatrix}$$

$$\hat{y}_{ui} = \sigma \left( h^T \begin{bmatrix} p_u^G \odot q_i^G \\ \phi^{MLP} \end{bmatrix} \right)$$

```
User ID ----> Embedding GMF (p_u^G) ----\
                                        ├--> ( Hadamard Product ) --> GMF Vector --\
Item ID ----> Embedding GMF (q_i^G) ----/                                           \
                                                                                     ├--> [ Concatenation ] --> Output Layer (Sigmoid) --> Prediction Score
User ID ----> Embedding MLP (p_u^M) ----\                                           /
                                        ├--> [ Concatenation ] ----> MLP Layers ----/
Item ID ----> Embedding MLP (q_i^M) ----/
```

Mô hình cho phép không gian Embedding của GMF và MLP học độc lập các biểu diễn đặc trưng riêng biệt phù hợp cho từng cấu trúc.

#### 2.3.4 Kỹ thuật lấy mẫu tiêu cực (Negative Sampling) và Hàm tổn thất Binary Cross-Entropy
Do dữ liệu implicit feedback chỉ cung cấp các tương tác tích cực ($y_{ui} = 1$), kỹ thuật lấy mẫu tiêu cực (Negative Sampling) thực hiện chọn ngẫu nhiên các Item mà User chưa từng tương tác để gán nhãn tiêu cực ($y_{ui} = 0$).

Hàm tổn thất Binary Cross-Entropy (BCE) được áp dụng để tối ưu hóa mô hình:

$$\mathcal{L} = -\sum_{(u, i) \in \mathcal{Y} \cup \mathcal{Y}^-} \left[ y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log (1 - \hat{y}_{ui}) \right]$$

Trong đó $\mathcal{Y}$ là tập tương tác tích cực và $\mathcal{Y}^-$ là tập tương tác tiêu cực được lấy mẫu.

### 2.4 Các chỉ số đánh giá hệ thống gợi ý Top-K

- **Hit Ratio at K (HR@K):** Đo lường tỷ lệ mẫu kiểm thử tích cực xuất hiện trong danh sách Top-K gợi ý:

$$\text{HR@K} = \frac{\text{Số lượt đánh trúng Top-K}}{\text{Tổng số mẫu kiểm thử}}$$

- **Normalized Discounted Cumulative Gain at K (NDCG@K):** Đánh giá vị trí thứ tự của sản phẩm gợi ý, gán trọng số cao hơn cho các sản phẩm đúng nằm ở vị trí đầu:

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad \text{với } \text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

- **Precision@K & Recall@K:** Precision@K đo tỷ lệ gợi ý đúng trong K sản phẩm xuất ra; Recall@K đo tỷ lệ tìm lại được các sản phẩm tích cực của người dùng.

### 2.5 Tổng quan các công trình nghiên cứu liên quan

| Tác giả & Năm | Mô hình / Phương pháp | Dữ liệu sử dụng | Đóng góp & Hạn chế |
| :--- | :--- | :--- | :--- |
| **Rendle et al. (2009)** | BPR (Bayesian Personalized Ranking) | MovieLens, Netflix | Đề xuất khung tối ưu dựa trên so sánh cặp (pairwise optimization) cho implicit feedback. **Hạn chế:** Dựa trên mô hình tuyến tính. |
| **He et al. (2017)** | NCF / NeuMF (Neural Collaborative Filtering) | MovieLens, Pinterest | Đề xuất framework kết hợp GMF và MLP, chứng minh ưu thế của học sâu trên ma trận tương tác. **Hạn chế:** Chưa tích hợp đặc trưng nội dung (content features). |
| **Cheng et al. (2016)** | Wide & Deep Learning | Google Play Store | Tích hợp Wide (học thuộc ghi nhớ) và Deep (khái quát hóa) cho hệ thống gợi ý app. **Hạn chế:** Yêu cầu kỹ nghệ đặc trưng thủ công cho phần Wide. |
| **Guo et al. (2017)** | DeepFM | Criteo, Company datasets | Thay thế phần Wide bằng Factorization Machine để tự động hóa việc học tương tác bậc thấp. **Hạn chế:** Chi phí tính toán tăng khi số lượng đặc trưng lớn. |
| **Khóa luận đề xuất** | **NeuMF + FastAPI/Streamlit (DataCo)** | **DataCo Supply Chain** | **Ứng dụng NeuMF trên dữ liệu chuỗi cung ứng, thực nghiệm Ablation Study, đánh giá Cold-start và đóng gói Docker.** |
