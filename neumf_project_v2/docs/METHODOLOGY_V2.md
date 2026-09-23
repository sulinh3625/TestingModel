# Methodology V2 — DACNTT NeuMF

## Tên đề tài

**Xây dựng mô hình khuyến nghị lai kết hợp nhân tử hoá ma trận và mạng lưới thần kinh sâu**

## Trục nghiên cứu

1. So sánh fair giữa classical MF/CF và neural interaction.
2. Ablation: GMF, MLP, NeuMF.
3. Ablation: NeuMF scratch vs pretrained, giữ cùng optimizer/LR/budget ở giai đoạn fine-tune.
4. Phân tích Head/Long-tail dựa trên popularity của **train set**.
5. Full-ranking là đánh giá chính; sampled 99 negatives chỉ là protocol đối chiếu NCF.
6. Binary implicit feedback là thí nghiệm chính; Sales/value chỉ là confidence-weighted ablation.

## Pipeline dữ liệu chính thức

```text
Raw transactions
  -> standard event schema
  -> aggregate unique User-Item
  -> iterative k-core
  -> re-index
  -> temporal Leave-One-Out
  -> train / validation / test
  -> feedback transform FIT ON TRAIN ONLY
  -> dynamic training negative sampling
```

### Vì sao aggregate trước k-core?

K-core phải đếm degree trên số **item khác nhau/user** và **user khác nhau/item**. Mua lại cùng một sản phẩm nhiều lần vẫn chỉ là một cạnh User-Item.

### Split

- Test: unique item cuối cùng của user.
- Validation: unique item áp chót.
- Train: phần trước đó.
- Assert `train∩val = train∩test = val∩test = 0` ở cấp `(user,item)`.

### Negative sampling

- Training: mặc định 4 negatives/positive; negative pool chỉ loại **train positives** để không dùng future labels trong training.
- Sampled evaluation: loại toàn bộ known positives để không tạo false negative.

## Evaluation

### Primary

- HR@5, HR@10
- NDCG@5, NDCG@10
- Early stopping: **NDCG@10**
- **Full ranking** trên toàn bộ candidate chưa thấy.

### Secondary

- 1 positive + 99 sampled negatives để đối chiếu protocol He et al. (2017).
- Precision/Recall không phải luận điểm riêng trong Leave-One-Out vì `Recall@K = HR@K` và `Precision@K = HR@K/K`.

## Long-tail

- Popularity chỉ tính từ **train interactions**.
- Head mặc định = top 10% item theo interaction count.
- Long-tail = phần còn lại.
- Báo cáo số user có **test item thuộc Long-tail**, không gọi họ là “long-tail users” theo nghĩa bản thân user.

## Dataset thứ hai

- DataCo = dataset chính.
- H&M = external/scalability validation sau khi DataCo V2 ổn định.
- Không gộp DataCo + H&M thành một interaction matrix.
