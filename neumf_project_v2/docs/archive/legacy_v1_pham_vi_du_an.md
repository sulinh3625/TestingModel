# Hướng Dẫn Chạy Dự Án: NeuMF Cho Gợi Ý Sản Phẩm Trên DataCo Supply Chain

## Tên Đề Tài

> **"Xây Dựng Mô Hình Lai Matrix Factorization Và Deep Neural Network (NeuMF)
> Để Gợi Ý Sản Phẩm"**

GVHD: TS. Hồ Thị Linh — Nhóm: Lê Minh Lý, Sử Thị Yến Linh

---

## 1. Mục Tiêu Bộ Mã Nguồn

Chạy pipeline NeuMF (GMF + MLP + NeuMF) trên **dữ liệu DataCo Supply Chain
thật** (không mô phỏng) để sinh số liệu thực nghiệm cho Bảng 4.1 và mục
4.3.3 (Cold-start/Long-tail) của báo cáo, với khả năng chạy lặp lại nhiều
seed để có số liệu đủ tin cậy (mean ± std) thay vì chỉ 1 lần chạy đơn lẻ.

---

## 2. Phạm Vi

### 2.1. Phạm Vi TRONG bộ mã nguồn

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHẠM VI THỰC NGHIỆM                        │
│                                                                 │
│  ✅ Xử lý dữ liệu DataCo THẬT (lọc k-core k=5, ánh xạ ID,       │
│     implicit feedback nhị phân hoặc có trọng số theo Sales)    │
│  ✅ Chia Leave-One-Out theo thời gian giao dịch thực tế         │
│  ✅ Kiến trúc GMF + MLP + NeuMF (He et al., 2017), pre-training │
│     GMF/MLP bằng Adam, fine-tuning NeuMF bằng SGD               │
│  ✅ 4 baseline đối chứng: Random, Most Popular, Item-CF, BPR-MF │
│  ✅ Đánh giá Top-K: HR@K, NDCG@K, Precision@K, Recall@K         │
│     với K ∈ {1, 3, 5, 10, 20}                                  │
│  ✅ Đánh giá phân tầng Popular / Long-tail                      │
│  ✅ Chạy đa-seed (scripts/run_multi_seed.py) — mean ± std       │
│  ✅ Unit test cho toàn bộ pipeline (data, model, metrics)       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2. Phạm Vi NGOÀI bộ mã nguồn

- ❌ Đặc trưng đa phương thức (hình ảnh, mô tả văn bản sản phẩm)
- ❌ Mô hình hóa chuỗi thời gian / hành vi phiên (Session-based, SASRec)
- ❌ Triển khai production (FastAPI/Streamlit/Docker — thuộc thiết kế hệ
  thống ở Chương 3 báo cáo, không nằm trong bộ mã thực nghiệm này)
- ❌ Tinh chỉnh siêu tham số tự động (Hyperparameter search / AutoML)
- ❌ So sánh với các kiến trúc hậu-NCF (LightGCN, SASRec)

---

## 3. Đặc Điểm Dữ Liệu Cần Biết Trước Khi Chạy

| Đặc điểm | Giá trị |
|---|---|
| Số dòng CSV gốc | 180.519 dòng, 53 cột, ~91MB |
| Sau lọc k-core (k=5) | 11.494 users, 100 items, 90.705 tương tác |
| Mật độ ma trận tương tác | 7,9% |
| Độ lệch phổ biến | 10/100 sản phẩm chiếm >83% tổng tương tác |

> [!IMPORTANT]
> Catalog sau lọc chỉ còn **100 sản phẩm** và lệch phổ biến rất mạnh. Vì
> vậy HR@K/NDCG@K trên toàn bộ tập test cao đồng loạt ở mọi phương pháp
> (kể cả baseline yếu) — không phản ánh đúng chất lượng mô hình. Dùng bảng
> đánh giá **Long-tail** (mục 8.2) làm luận điểm chính khi so sánh NeuMF
> với baseline.

> [!NOTE]
> File CSV phải là bản DataCo Kaggle đầy đủ. Nếu số dòng đọc được dưới
> 50.000, `main.py` sẽ tự động cảnh báo — kiểm tra lại đường dẫn file.
> File có thể đặt tại `neumf_project/DataCoSupplyChainDataset.csv` **hoặc**
> `neumf_project/data/DataCoSupplyChainDataset.csv` — code tự tìm ở cả 2 nơi.

> [!IMPORTANT]
> File CSV (~91MB) **không đính kèm trong zip** vì quá nặng. Tải tại Google
> Drive của nhóm: <https://drive.google.com/drive/folders/1xsFYuFEKZWkd5FWZo_Dguc7au36FqO1l>
> (hoặc bản gốc trên Kaggle), rồi đặt vào một trong 2 vị trí ở trên.
>
> **Khi nén project để gửi/nộp, KHÔNG bao giờ nén thư mục `venv/`** — đây là
> nguyên nhân chính khiến file zip nặng cả GB (venv chứa toàn bộ thư viện đã
> cài, ~35.000 file). `.gitignore` đã loại `venv/` khi dùng Git, nhưng khi
> tự nén tay bằng "Send to > Compressed folder" trên Windows thì KHÔNG có
> gì tự động loại trừ — phải tự bỏ chọn thư mục `venv/` trước khi nén.

---

## 4. Cấu Trúc Mã Nguồn

```
neumf_project/
├── DataCoSupplyChainDataset.csv  ← Đặt file dữ liệu gốc tại đây
├── config.py                     ← Cấu hình tập trung (đọc config.yaml nếu có)
├── config.yaml                   ← Ghi đè siêu tham số không cần sửa code
├── data/
│   ├── data_loader.py            ← Đọc CSV, lọc k-core, xây interaction table
│   └── dataset.py                ← Leave-One-Out split, Negative Sampling
├── models/
│   └── neumf.py                  ← GMF, MLP, NeuMF (+ load_pretrained)
├── baselines/
│   ├── __init__.py
│   └── classical_baselines.py    ← Random, Most Popular, Item-CF, BPR-MF
├── scripts/
│   └── run_multi_seed.py         ← Chạy nhiều seed, gộp mean ± std
├── train.py                      ← Vòng lặp huấn luyện + early stopping
├── evaluate.py                   ← HR@K, NDCG@K, Precision@K, Recall@K
├── main.py                       ← Chạy 1 lần pipeline đầy đủ
├── plot_results.py               ← Vẽ 4 biểu đồ từ results/ (results/figures/)
├── requirements.txt              ← Thư viện cần cài
├── tests/
│   └── test_project.py           ← 14 unit test
├── checkpoints/                  ← gmf.pt, mlp.pt, neumf.pt (sinh khi chạy)
├── results/                      ← Bảng kết quả chính thức (sinh khi chạy)
├── runs/                         ← Log + config từng lần chạy riêng lẻ
├── README.md                     ← Giới thiệu ngắn, trỏ vào file này
└── pham_vi_du_an.md              ← File này
```

---

## 5. Các Thông Số Cấu Hình Chính (khớp Bảng 3.1 báo cáo)

| Nhóm | Tham số | Giá trị |
|---|---|---|
| **Dữ liệu** | k-core | 5 |
| | Negative ratio (train) | 4 |
| | Negative samples (eval) | 99 (hoặc toàn bộ item khả dụng nếu catalog nhỏ) |
| **Mô hình** | Embedding dim (d) | 32 |
| | MLP layers | [64, 32, 16, 8] |
| | Dropout | 0,2 |
| **Huấn luyện** | Batch size | 256 |
| | Learning rate | 1e-3 (Adam) / 1e-2 (SGD) |
| | Weight decay (L2) | 1e-6 |
| | Max epochs | 50 |
| | Patience (early stopping) | 10 |
| **Đánh giá** | K values | 1, 3, 5, 10, 20 |

Có thể ghi đè toàn bộ bảng trên qua `config.yaml` (không cần sửa `config.py`).

---

## 6. Hướng Dẫn Chạy

### 6.1. Cài đặt môi trường ảo (venv)

> [!TIP]
> Dùng venv để thư viện chỉ nằm trong thư mục `venv/` của project, xoá thư
> mục này là gỡ sạch, không ảnh hưởng Python hệ thống.

```bash
cd neumf_project
python -m venv venv

# Kích hoạt (mỗi lần mở terminal mới)
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

# Cài torch bản CPU-only trước (tránh tải kèm ~5GB gói nvidia-* không cần thiết)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện còn lại (bao gồm pyyaml cho config.yaml)
pip install -r requirements.txt
```

Khi làm xong: `deactivate`. Không nộp thư mục `venv/` kèm báo cáo/Git — chỉ
cần nộp `requirements.txt`.

### 6.2. Chạy 1 lần (nhanh, để kiểm tra)

```bash
# Đặt DataCoSupplyChainDataset.csv (bản đầy đủ) tại thư mục neumf_project/

python main.py                                          # Cấu hình đầy đủ Bảng 3.1
python main.py --max-epochs-pretrain 20 --max-epochs-finetune 20 --patience 5   # Rút gọn
python -m pytest tests/ -v                                # Unit test
python plot_results.py                                    # Vẽ biểu đồ từ results/ hiện có
```

### 6.3. Chạy nhiều seed (khuyến nghị cho số liệu chính thức trong báo cáo)

```bash
python scripts/run_multi_seed.py --seeds 42 43 44
```

Mỗi seed chạy full pipeline (8 phương pháp) và lưu riêng vào
`runs/seed_<n>/`; kết quả gộp mean ± std nằm ở
`results/multi_seed_summary.csv`. Xem mục 8 để biết thời gian ước tính.

### 6.4. Kết quả sinh ra

| File | Nội dung |
|---|---|
| `results/results.json` | Số liệu đầy đủ của lần chạy `main.py` gần nhất |
| `results/results_table.csv` | Bảng 4.1 (toàn bộ tập test) |
| `results/results_table_long_tail.csv` | Bảng bổ sung cho mục 4.3.3 |
| `results/multi_seed_summary.csv` | Gộp mean ± std nhiều seed |
| `checkpoints/gmf.pt`, `mlp.pt`, `neumf.pt` | Checkpoint mô hình (lần chạy gần nhất) |
| `runs/<tag>/results.json` | Số liệu riêng của từng lần chạy (`main.py` hoặc từng seed) |
| `runs/<tag>/config_snapshot.json` | Snapshot toàn bộ cấu hình dùng cho lần chạy đó |
| `runs/<tag>/run.log` | Log chi tiết từng epoch của lần chạy đó |

> [!NOTE]
> Chạy lại trên máy khác có thể lệch vài phần nghìn ở HR@K/NDCG@K (thứ tự
> rút gọn phép cộng khi nhân ma trận đa luồng CPU khác nhau giữa các máy dù
> đã cố định seed). Đây là hành vi bình thường của PyTorch trên CPU, không
> ảnh hưởng kết luận — chạy nhiều seed (mục 6.3) giúp thấy rõ mức dao động
> này thay vì chỉ đoán.

---

## 7. Kết Quả 1 Lần Chạy Đầy Đủ (cấu hình Bảng 3.1, máy nhóm, 14,4 phút)

### 7.1. Bảng 4.1 — Toàn bộ tập test (n = 11.481 user)

| Mô hình | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| Random | 0,0566 | 0,1130 | 0,0336 | 0,0515 |
| Most Popular | 0,9080 | 0,9423 | 0,6147 | 0,6268 |
| Item-CF | 0,9060 | 0,9439 | 0,5984 | 0,6116 |
| BPR-MF | 0,9145 | 0,9434 | 0,6318 | 0,6419 |
| GMF | 0,9309 | 0,9443 | 0,7217 | 0,7263 |
| MLP | 0,9423 | 0,9469 | 0,9413 | 0,9427 |
| NeuMF (from scratch) | 0,9236 | 0,9467 | 0,6415 | 0,6495 |
| **NeuMF (pre-trained)** | **0,9333** | **0,9462** | **0,7220** | **0,7264** |

Bảng đầy đủ (K=1,3,5,10,20, kèm Precision/Recall): `results/results_table.csv`.

### 7.2. Đánh giá Long-tail — 90 sản phẩm ít phổ biến (n = 683 user)

| Mô hình | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| Most Popular | 0,0059 | 0,0293 | 0,0025 | 0,0099 |
| Item-CF | 0,0073 | 0,0586 | 0,0031 | 0,0193 |
| BPR-MF | 0,0161 | 0,0586 | 0,0082 | 0,0216 |
| GMF | 0,0059 | 0,0717 | 0,0025 | 0,0237 |
| Random | 0,0337 | 0,0776 | 0,0226 | 0,0367 |
| NeuMF (pre-trained) | 0,0381 | 0,0981 | 0,0175 | 0,0364 |
| NeuMF (from scratch) | 0,0220 | 0,1054 | 0,0107 | 0,0377 |
| **MLP** | **0,0322** | **0,1083** | **0,0145** | **0,0386** |

---

## 8. Kết Quả Đa-Seed (3 seed: 42, 43, 44 — kiểm tra độ ổn định)

Chạy bằng `scripts/run_multi_seed.py`, cấu hình rút gọn (15 epoch,
patience=5) để kiểm tra nhanh độ ổn định — khuyến nghị chạy lại với cấu
hình đầy đủ (Bảng 3.1) trước khi nộp báo cáo cuối cùng.

| Mô hình | HR@10 (mean ± std) | NDCG@10 (mean ± std) |
|---|---|---|
| Random | 0,1130 ± 0,0000 | 0,0515 ± 0,0000 |
| Most Popular | 0,9423 ± 0,0000 | 0,6268 ± 0,0000 |
| Item-CF | 0,9439 ± 0,0000 | 0,6116 ± 0,0000 |
| BPR-MF | 0,9440 ± 0,0006 | 0,6493 ± 0,0198 |
| GMF | 0,9435 ± 0,0004 | 0,7010 ± 0,0097 |
| MLP | 0,9450 ± 0,0010 | 0,7800 ± 0,1156 |
| NeuMF (from scratch) | 0,9456 ± 0,0004 | 0,6530 ± 0,0095 |
| NeuMF (pre-trained) | 0,9445 ± 0,0010 | 0,7039 ± 0,0078 |

> [!IMPORTANT]
> HR@10 rất ổn định qua các seed (std < 0,001 ở hầu hết phương pháp) — kết
> luận về xếp hạng HR@10 giữ vững. Nhưng **NDCG@10 của MLP dao động rất lớn**
> (std = 0,1156, gần bằng độ lớn của chính giá trị mean) — nghĩa là chất
> lượng xếp hạng chi tiết (không chỉ "có trúng top-10 hay không") của MLP
> **không ổn định giữa các lần huấn luyện**, có thể do MLP đôi khi hội tụ về
> nghiệm gần như đoán đúng thứ hạng 1 tuyệt đối (như mô tả ở mục 7.1) và đôi
> khi không. Đây là điểm nên nêu trong mục 5.2 (Hạn chế) của báo cáo, và là
> lý do nên ưu tiên NeuMF (pre-trained) — có NDCG@10 thấp hơn một chút
> nhưng ổn định hơn nhiều (std = 0,0078) — làm mô hình đề xuất chính thay
> vì chỉ chọn theo con số cao nhất của một lần chạy đơn lẻ.

Bảng đầy đủ (K=1,3,5,10,20, mọi metric): `results/multi_seed_summary.csv`.
Chi tiết từng seed: `runs/seed_42/`, `runs/seed_43/`, `runs/seed_44/`.

---

## 9. Hướng Mở Rộng Tiềm Năng

> [!TIP]
> Những hướng có thể bổ sung nếu còn thời gian:

- Chạy `scripts/run_multi_seed.py` với cấu hình đầy đủ Bảng 3.1 (5+ seed)
  để có số liệu chính thức cuối cùng cho báo cáo, thay bảng mục 7 bằng
  bảng mean ± std
- Thử `feedback_mode: binary` trong `config.yaml` để so sánh với biến thể
  weighted hiện tại
- Điều tra nguyên nhân MLP kém ổn định (mục 8) — có thể do dropout kết hợp
  learning rate Adam khiến MLP dễ hội tụ về các nghiệm cục bộ khác nhau
