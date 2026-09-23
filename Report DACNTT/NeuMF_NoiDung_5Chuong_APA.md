# XÂY DỰNG MÔ HÌNH LAI MATRIX FACTORIZATION VÀ DEEP NEURAL NETWORK (NEUMF) ĐỂ GỢI Ý SẢN PHẨM

---

## TÓM TẮT

Trong kỷ nguyên thương mại điện tử và kinh tế số, hệ thống gợi ý (Recommendation System) đóng vai trò then chốt trong việc tối ưu hóa trải nghiệm người dùng và nâng cao doanh thu doanh nghiệp. Các phương pháp Lọc cộng tác (Collaborative Filtering – CF) truyền thống như Nhân tử hóa ma trận (Matrix Factorization – MF) chủ yếu sử dụng tích vô hướng để mô hình hóa tương tác giữa người dùng và sản phẩm (Koren, Bell, & Volinsky, 2009). Tuy nhiên, việc phụ thuộc vào tuyến tính hóa khiến MF khó nắm bắt được các mối quan hệ phức tạp và phi tuyến trong không gian đặc trưng ẩn (He, Liao, Zhang, Nie, Hu, & Chua, 2017).

Khóa luận đề xuất xây dựng mô hình khuyến nghị lai Neural Matrix Factorization (NeuMF) kết hợp Generalized Matrix Factorization (GMF) và Multi-Layer Perceptron (MLP) (He et al., 2017), thực nghiệm trên bộ dữ liệu DataCo Supply Chain (Kaggle, 2019), sử dụng tín hiệu phản hồi ẩn có trọng số theo giá trị đơn hàng (Sales/Order Item Total), lấy mẫu tiêu cực (Negative Sampling) và đánh giá Leave-One-Out. Hiệu năng được kiểm chứng qua HR@K, NDCG@K (Järvelin & Kekäläinen, 2002), Precision@K, Recall@K, kèm phân tích Ablation Study và đánh giá Cold-start.

## ABSTRACT

In the era of e-commerce and the digital economy, recommendation systems play a pivotal role in optimizing user experience and driving business revenue. Traditional Collaborative Filtering approaches such as Matrix Factorization primarily rely on the inner product to model user–item interactions (Koren, Bell, & Volinsky, 2009), which restricts them from capturing complex, non-linear relationships (He, Liao, Zhang, Nie, Hu, & Chua, 2017).

This thesis proposes a hybrid Neural Matrix Factorization (NeuMF) framework combining Generalized Matrix Factorization and a Multi-Layer Perceptron (He et al., 2017), evaluated on the DataCo Supply Chain dataset (Kaggle, 2019) with weighted implicit feedback, negative sampling, and a Leave-One-Out protocol. Performance is measured via HR@K, NDCG@K (Järvelin & Kekäläinen, 2002), Precision@K, and Recall@K, together with an ablation study and a cold-start evaluation.

---

# CHƯƠNG 1. TỔNG QUAN VỀ ĐỀ TÀI

## 1.1 Bối cảnh nghiên cứu và tính cấp thiết của đề tài

### 1.1.1 Thực trạng hệ thống gợi ý trong thương mại điện tử và chuỗi cung ứng

Sự bùng nổ của các nền tảng thương mại điện tử và giao dịch trực tuyến tạo ra lượng dữ liệu sản phẩm và người dùng khổng lồ. Đối mặt với tình trạng quá tải thông tin (information overload), hệ thống gợi ý trở thành công cụ cốt lõi giúp cá nhân hóa trải nghiệm khách hàng, giảm chi phí tìm kiếm và gia tăng tỷ lệ chuyển đổi đơn hàng. Trong môi trường chuỗi cung ứng và bán lẻ, việc gợi ý chính xác sản phẩm phù hợp không chỉ nâng cao mức độ hài lòng của người dùng mà còn hỗ trợ doanh nghiệp dự báo nhu cầu tiêu thụ và tối ưu hóa luồng hàng hóa.

### 1.1.2 Hạn chế của các phương pháp Lọc cộng tác truyền thống

Lọc cộng tác, đặc biệt là các phương pháp Nhân tử hóa ma trận như SVD hay ALS, được sử dụng phổ biến nhờ tính đơn giản và hiệu quả (Koren, Bell, & Volinsky, 2009). MF biểu diễn User và Item dưới dạng các vector ẩn trong không gian có số chiều thấp và dự đoán mức độ tương tác thông qua tích vô hướng. Tuy nhiên, tích vô hướng là một kết hợp tuyến tính đơn giản của các đặc trưng ẩn, không thể học được các tương tác phức tạp và phi tuyến giữa các chiều đặc trưng của User và Item (He et al., 2017), làm giảm độ chính xác khi xử lý ma trận tương tác có độ thưa cao (data sparsity) hoặc tín hiệu phản hồi ẩn (implicit feedback) phức tạp.

### 1.1.3 Sự cần thiết của việc kết hợp Matrix Factorization và Deep Learning (NeuMF)

Mô hình Neural Collaborative Filtering (NCF/NeuMF) do He và cộng sự (2017) đề xuất kết hợp ưu điểm của cả hai tiếp cận: sử dụng Generalized Matrix Factorization (GMF) để duy trì khả năng học tương tác tuyến tính cố định, kết hợp mạng MLP để mở rộng khả năng học quan hệ phi tuyến sâu. Việc ứng dụng kiến trúc NeuMF vào bài toán gợi ý sản phẩm giúp nâng cao năng lực dự đoán, tạo nền tảng cho việc triển khai các hệ thống gợi ý cá nhân hóa có độ chính xác cao.

## 1.2 Mục tiêu và phạm vi nghiên cứu

### 1.2.1 Mục tiêu nghiên cứu

- Nghiên cứu cơ sở lý thuyết về Lọc cộng tác, Nhân tử hóa ma trận, Mạng nơ-ron sâu và kiến trúc mô hình lai NeuMF (He et al., 2017; Koren, Bell, & Volinsky, 2009).
- Xây dựng quy trình xử lý dữ liệu hoàn chỉnh trên bộ dữ liệu DataCo Supply Chain, biến đổi dữ liệu giao dịch thành ma trận tương tác User–Item dạng tín hiệu ẩn, thực hiện lấy mẫu tiêu cực và chia tập huấn luyện/kiểm thử.
- Cài đặt, huấn luyện và tinh chỉnh các mô hình GMF, MLP độc lập và mô hình lai NeuMF.
- Đánh giá thực nghiệm định lượng hiệu năng của NeuMF so với các nhánh thành phần và baseline cổ điển qua HR@K, NDCG@K (Järvelin & Kekäläinen, 2002), Precision@K, Recall@K; thực hiện Ablation Study và đánh giá Cold-start.
- Đóng gói mô hình thành dịch vụ API (FastAPI) và giao diện demo (Streamlit) bằng Docker.

### 1.2.2 Phạm vi nghiên cứu

- **Đối tượng nghiên cứu:** kiến trúc Neural Collaborative Filtering (GMF, MLP, NeuMF) (He et al., 2017), kỹ thuật biểu diễn Embedding và các thuật toán đánh giá hệ thống gợi ý Top-K.
- **Dữ liệu thực nghiệm:** bộ dữ liệu DataCo Smart Supply Chain (Kaggle, 2019), khai thác thông tin khách hàng, sản phẩm và lịch sử giao dịch.
- **Tín hiệu tương tác:** phản hồi ẩn dạng nhị phân dựa trên lịch sử mua hàng, kết hợp thử nghiệm biến thể tương tác có trọng số theo giá trị đơn hàng.

## 1.3 Phương pháp nghiên cứu

Nghiên cứu áp dụng kết hợp phương pháp lý thuyết (phân tích, tổng hợp tài liệu học thuật nền tảng về NCF/NeuMF) và phương pháp thực nghiệm (xây dựng mã nguồn bằng PyTorch, tiền xử lý dữ liệu bằng Pandas/NumPy, theo dõi thực nghiệm và đánh giá định lượng trên các kịch bản cố định).

## 1.4 Ý nghĩa khoa học và thực tiễn của đề tài

Về khoa học, khóa luận đóng góp kết quả thực nghiệm và phân tích Ablation Study về vai trò của thành phần tuyến tính (GMF) và phi tuyến (MLP) trong kiến trúc NeuMF (He et al., 2017) trên dữ liệu chuỗi cung ứng thực tế. Về thực tiễn, khóa luận cung cấp giải pháp đóng gói mô hình gợi ý hoàn chỉnh, có khả năng tích hợp vào hệ thống thương mại điện tử qua API.

## 1.5 Bố cục của khóa luận

Khóa luận được trình bày trong 5 chương:

- **Chương 1 – Tổng quan về đề tài:** giới thiệu bối cảnh, tính cấp thiết, mục tiêu, phạm vi, phương pháp nghiên cứu và ý nghĩa của đề tài.
- **Chương 2 – Cơ sở lý thuyết và các công trình liên quan:** trình bày nền tảng lý thuyết về hệ thống gợi ý, Matrix Factorization, GMF, MLP, NeuMF và tổng quan các nghiên cứu tiền nhiệm.
- **Chương 3 – Phân tích, thiết kế và triển khai hệ thống:** mô tả pipeline xử lý dữ liệu DataCo, kiến trúc mô hình, kỹ thuật Negative Sampling, quy trình huấn luyện và hệ thống API/Demo.
- **Chương 4 – Thực nghiệm và đánh giá kết quả:** trình bày kết quả thực nghiệm định lượng, so sánh NeuMF với các baseline, Ablation Study và đánh giá Cold-start.
- **Chương 5 – Kết luận và hướng phát triển:** tổng kết kết quả đạt được, hạn chế và hướng mở rộng.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN

## 2.1 Tổng quan về hệ thống gợi ý (Recommendation System)

### 2.1.1 Khái niệm và phân loại hệ thống gợi ý

Hệ thống gợi ý là một lớp thuật toán nhằm dự đoán mức độ quan tâm của người dùng đối với một sản phẩm/dịch vụ cụ thể, gồm ba nhóm chính:

- **Lọc dựa trên nội dung (Content-Based Filtering):** dựa trên đặc tính sản phẩm và hồ sơ sở thích của người dùng trong quá khứ.
- **Lọc cộng tác (Collaborative Filtering):** dựa trên lịch sử tương tác chung giữa nhiều người dùng và nhiều sản phẩm (Koren, Bell, & Volinsky, 2009).
- **Hệ thống gợi ý lai (Hybrid Recommendation System):** kết hợp nhiều tiếp cận nhằm khắc phục hạn chế của từng phương pháp đơn lẻ (Cheng et al., 2016; Guo, Tang, Ye, Li, & He, 2017).

### 2.1.2 Phản hồi hiện (Explicit Feedback) và phản hồi ẩn (Implicit Feedback)

- **Explicit Feedback:** thể hiện trực tiếp mức độ hài lòng của người dùng (rating, thích/chê); độ chính xác cao nhưng rất thưa thớt trong thực tế.
- **Implicit Feedback:** ghi nhận gián tiếp qua hành vi người dùng như lịch sử mua hàng, lượt click (He et al., 2017), biểu diễn dưới dạng nhị phân:

$$y_{ui} = \begin{cases} 1, & \text{nếu tương tác giữa User } u \text{ và Item } i \text{ được ghi nhận} \\ 0, & \text{ngược lại} \end{cases}$$

Mặc dù $y_{ui}=1$ không đồng nghĩa với việc người dùng thích sản phẩm (He et al., 2017), dữ liệu phản hồi ẩn thu thập dễ dàng với quy mô lớn hơn nhiều.

### 2.1.3 Thách thức độ thưa dữ liệu và vấn đề Cold-Start

Số lượng tương tác thực tế giữa User và Item chỉ chiếm một tỷ lệ rất nhỏ trong ma trận tương tác toàn cục (Data Sparsity) (Koren, Bell, & Volinsky, 2009). Vấn đề Cold-Start xảy ra khi một User hoặc Item mới chưa có lịch sử tương tác, khiến mô hình không thể tạo ra vector biểu diễn chính xác.

## 2.2 Cơ sở lý thuyết về Lọc cộng tác và Nhân tử hóa ma trận

### 2.2.1 Mô hình Nhân tử hóa ma trận (Matrix Factorization – MF)

MF chiếu User và Item vào một không gian đặc trưng ẩn chung có số chiều $d \ll \min(M,N)$ (Koren, Bell, & Volinsky, 2009). Mỗi User $u$ được biểu diễn bởi $p_u \in \mathbb{R}^d$ và mỗi Item $i$ bởi $q_i \in \mathbb{R}^d$:

$$\hat{y}_{ui} = p_u^{T} q_i = \sum_{f=1}^{d} p_{uf} \, q_{if}$$

### 2.2.2 Hạn chế của tích vô hướng trong mô hình hóa tương tác

Tích vô hướng kết hợp các chiều đặc trưng ẩn độc lập theo dạng tuyến tính, không đủ linh hoạt để mô hình hóa chính xác sự tương đồng không gian khi vị trí tương quan của các User thay đổi (He et al., 2017).

## 2.3 Kiến trúc Mạng khuyến nghị thần kinh (Neural Collaborative Filtering – NCF)

### 2.3.1 Generalized Matrix Factorization (GMF)

GMF là sự mở rộng của MF dưới dạng biểu diễn mạng nơ-ron (He et al., 2017), thực hiện phép nhân Element-wise (Hadamard product) giữa vector Embedding của User $p_u^{G}$ và Item $q_i^{G}$:

$$\varphi^{GMF} = p_u^{G} \odot q_i^{G}, \qquad \hat{y}_{ui} = \sigma\left(h^{T} \varphi^{GMF}\right)$$

trong đó $\odot$ là phép nhân từng phần tử, $h$ là trọng số lớp đầu ra, $\sigma(x) = 1/(1+e^{-x})$.

### 2.3.2 Multi-Layer Perceptron (MLP)

Nhánh MLP nối vector Embedding $p_u^{M}$ và $q_i^{M}$ (Concatenation) làm đầu vào cho chuỗi lớp mạng liên tiếp (He et al., 2017):

$$z_1 = \varphi^{MLP}_1 = \begin{bmatrix} p_u^{M} \\ q_i^{M} \end{bmatrix}, \qquad \varphi^{MLP}_L = a_L\left(W_L^{T} \varphi^{MLP}_{L-1} + b_L\right)$$

### 2.3.3 Kiến trúc mô hình lai NeuMF

NeuMF nối vector đầu ra của GMF và MLP trước lớp dự đoán cuối cùng (He et al., 2017):

$$\hat{y}_{ui} = \sigma\left(h^{T} \begin{bmatrix} p_u^{G} \odot q_i^{G} \\ \varphi^{MLP} \end{bmatrix}\right)$$

### 2.3.4 Lấy mẫu tiêu cực (Negative Sampling) và hàm tổn thất Binary Cross-Entropy

Do dữ liệu implicit feedback chỉ có tương tác tích cực ($y_{ui}=1$), kỹ thuật Negative Sampling chọn ngẫu nhiên các Item chưa từng tương tác để gán nhãn tiêu cực, theo giao thức của He và cộng sự (2017):

$$\mathcal{L} = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^{-}} \Big[ y_{ui}\log \hat{y}_{ui} + (1-y_{ui})\log(1-\hat{y}_{ui}) \Big]$$

## 2.4 Các chỉ số đánh giá hệ thống gợi ý Top-K

- **Hit Ratio at K (HR@K):** đo tỷ lệ mẫu kiểm thử tích cực xuất hiện trong Top-K, theo giao thức Leave-One-Out của He và cộng sự (2017).
- **Normalized Discounted Cumulative Gain at K (NDCG@K):** đánh giá vị trí thứ tự, theo định nghĩa gốc của Järvelin và Kekäläinen (2002):

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \qquad \text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_2(i+1)}$$

- **Precision@K & Recall@K:** Precision@K đo tỷ lệ gợi ý đúng trong K sản phẩm; Recall@K đo tỷ lệ tìm lại được các sản phẩm tích cực.

## 2.5 Tổng quan các công trình nghiên cứu liên quan

Bảng 2.1 tổng hợp các công trình nền tảng liên quan trực tiếp đến hướng tiếp cận của khóa luận.

**Bảng 2.1: Tổng quan các công trình nghiên cứu liên quan**

| Tác giả & Năm | Mô hình / Phương pháp | Dữ liệu | Đóng góp & Hạn chế |
|---|---|---|---|
| Rendle et al. (2009) | BPR (Bayesian Personalized Ranking) | MovieLens, Netflix | Khung tối ưu dựa trên so sánh cặp cho implicit feedback. Hạn chế: mô hình tuyến tính. |
| He et al. (2017) | NCF / NeuMF | MovieLens, Pinterest | Kết hợp GMF và MLP, chứng minh ưu thế của học sâu. Hạn chế: chưa tích hợp đặc trưng nội dung. |
| Cheng et al. (2016) | Wide & Deep Learning | Google Play Store | Kết hợp ghi nhớ (Wide) và khái quát hóa (Deep). Hạn chế: cần kỹ nghệ đặc trưng thủ công. |
| Guo et al. (2017) | DeepFM | Criteo, dữ liệu doanh nghiệp | Thay Wide bằng Factorization Machine. Hạn chế: chi phí tính toán tăng khi số đặc trưng lớn. |
| Khóa luận đề xuất | NeuMF + FastAPI/Streamlit | DataCo Supply Chain | Ứng dụng NeuMF trên dữ liệu chuỗi cung ứng, Ablation Study, đánh giá Cold-start, đóng gói Docker. |

## 2.6 Khoảng trống nghiên cứu

Các nghiên cứu nền tảng về NCF/NeuMF (He et al., 2017) chủ yếu được đánh giá trên dữ liệu điện ảnh hoặc mạng xã hội hình ảnh, với tín hiệu tương tác nhị phân đơn giản. Việc ứng dụng NeuMF trên dữ liệu chuỗi cung ứng thực tế (DataCo), kết hợp thử nghiệm tín hiệu tương tác có trọng số theo giá trị đơn hàng và đóng gói thành dịch vụ triển khai thực tế, là khoảng trống mà khóa luận hướng tới.

---

# CHƯƠNG 3. PHÂN TÍCH, THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG

*(Nội dung chương này cần được nhóm bổ sung dựa trên mã nguồn và quyết định kỹ thuật thực tế của dự án `neumf_project/`; các đề mục dưới đây là khung sườn theo đúng cấu trúc đã thống nhất, không suy diễn số liệu khi chưa chạy thực nghiệm.)*

## 3.1 Phân tích và thiết kế quy trình xử lý dữ liệu DataCo

### 3.1.1 Mô tả bộ dữ liệu DataCo Supply Chain

[Trình bày cấu trúc bảng dữ liệu, số dòng giao dịch, các trường User/Item/Order được sử dụng để xây dựng ma trận tương tác.]

### 3.1.2 Xây dựng ma trận tương tác User–Item

[Mô tả quy trình chuyển đổi lịch sử giao dịch thành tín hiệu implicit feedback nhị phân và biến thể có trọng số theo Sales/Order Item Total.]

### 3.1.3 Lấy mẫu tiêu cực và chia tập huấn luyện/kiểm thử

[Mô tả tỷ lệ Negative Sampling, phương pháp chia tập theo Leave-One-Out.]

## 3.2 Thiết kế kiến trúc mô hình

### 3.2.1 Cấu hình GMF và MLP

[Trình bày số chiều embedding, số lớp ẩn MLP, hàm kích hoạt được lựa chọn cho thực nghiệm.]

### 3.2.2 Tích hợp NeuMF (pretraining GMF + MLP, fine-tuning NeuMF)

[Mô tả quy trình huấn luyện trước từng nhánh và hợp nhất trọng số.]

## 3.3 Triển khai huấn luyện mô hình

### 3.3.1 Môi trường và công cụ cài đặt

[PyTorch, cấu hình phần cứng, thư viện xử lý dữ liệu.]

### 3.3.2 Siêu tham số huấn luyện

[Bảng siêu tham số: learning rate, batch size, số epoch, hệ số regularization.]

## 3.4 Thiết kế hệ thống triển khai (API/Demo)

### 3.4.1 Dịch vụ suy luận FastAPI

[Mô tả endpoint, định dạng input/output.]

### 3.4.2 Giao diện demo Streamlit và đóng gói Docker

[Mô tả kiến trúc container hóa.]

---

# CHƯƠNG 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

*(Khung sườn — chờ số liệu thực nghiệm thật từ nhóm; không điền số liệu giả định.)*

## 4.1 Cấu hình thực nghiệm

[Môi trường phần cứng/phần mềm, phiên bản thư viện.]

## 4.2 Phương pháp đối chứng (Baselines)

So sánh NeuMF với các nhánh thành phần độc lập (GMF, MLP) và phương pháp cổ điển, ví dụ BPR (Rendle, Freudenthaler, Gantner, & Schmidt-Thieme, 2009).

## 4.3 Kết quả thực nghiệm

### 4.3.1 So sánh HR@K và NDCG@K

[Bảng kết quả HR@K, NDCG@K theo Järvelin và Kekäläinen (2002) — **[CẦN ĐIỀN sau khi chạy thực nghiệm]**.]

### 4.3.2 Phân tích Ablation Study

[So sánh đóng góp riêng của GMF và MLP.]

### 4.3.3 Đánh giá hiện tượng Cold-start

[Phân tích hiệu năng trên nhóm User/Item có ít tương tác.]

## 4.4 Thảo luận

[Tổng hợp và diễn giải kết quả, đối chiếu với giả thuyết ban đầu.]

---

# CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1 Kết luận

[Tổng kết các kết quả đạt được, đối chiếu với mục tiêu nghiên cứu đã đề ra ở Chương 1.]

## 5.2 Hạn chế của đề tài

[Nêu các hạn chế còn tồn tại.]

## 5.3 Hướng phát triển

[Đề xuất hướng mở rộng: dữ liệu quy mô lớn hơn, kiến trúc attention, v.v.]

---

# DANH MỤC TÀI LIỆU THAM KHẢO

**B. Tài liệu tham khảo**

*Tiếng nước ngoài*

Cheng, H.-T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., Anderson, G., Corrado, G., Chai, W., Ispir, M., Anil, R., Haque, Z., Hong, L., Jain, V., Liu, X., & Shah, H. (2016). Wide & deep learning for recommender systems. In *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems (DLRS 2016)* (pp. 7–10). Boston, MA, USA. https://doi.org/10.1145/2988450.2988454

Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A factorization-machine based neural network for CTR prediction. In *Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI 2017)* (pp. 1725–1731). Melbourne, Australia.

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural collaborative filtering. In *Proceedings of the 26th International Conference on World Wide Web (WWW '17)* (pp. 173–182). Perth, Australia. https://doi.org/10.1145/3038912.3052569

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems, 20*(4), 422–446. https://doi.org/10.1145/582415.582418

Kaggle. (2019). *DataCo smart supply chain for big data analysis*. Truy cập ngày 29/08/2026, từ https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8), 30–37. https://doi.org/10.1109/MC.2009.263

Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. In *Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI '09)* (pp. 452–461). Montreal, QC, Canada.
