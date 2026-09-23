"""SASRec (Kang & McAuley, ICDM 2018) — "Self-Attentive Sequential
Recommendation", đại diện nhóm Temporal/Sequential Recommendation.

Khác biệt cốt lõi so với GMF/MLP/NeuMF/LightGCN: các mô hình đó coi lịch sử
tương tác của một User là một TẬP HỢP KHÔNG THỨ TỰ (chỉ cần biết "user u đã
mua item i chưa", không quan tâm mua item nào TRƯỚC/SAU item nào). SASRec mã
hoá THỨ TỰ THỜI GIAN của chuỗi tương tác bằng Self-Attention một chiều
(causal — vị trí sau chỉ được nhìn vị trí trước, không được nhìn tương lai),
sinh ra một vector đại diện "trạng thái sở thích hiện tại" của user từ TOÀN
BỘ chuỗi Train, dùng vector đó để chấm điểm ứng viên tiếp theo.

Quyết định thiết kế (nhất quán với LightGCN, mục 3.4.6 báo cáo): huấn luyện
bằng đúng BCE + Negative Sampling + interface forward(users, items) dùng
chung toàn bộ hạ tầng train_one_model/evaluate_torch_model, thay vì mục tiêu
"dự đoán token tiếp theo tại MỌI vị trí" của bài báo gốc — đánh đổi lấy một
phép so sánh công bằng hơn giữa các kiến trúc (chỉ khác nhau ở CÓ/KHÔNG dùng
thứ tự thời gian), không lẫn với khác biệt về hàm mục tiêu huấn luyện.

Chuỗi mỗi user CHỈ xây từ TRAIN (build_user_sequences nhận train_df), đúng
nguyên tắc chống rò rỉ dữ liệu áp dụng xuyên suốt dự án.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_user_sequences(train_df, n_users: int, n_items: int, seq_len: int) -> torch.Tensor:
    """Xây sequence Item mỗi user từ TRAIN, sắp theo (last_timestamp,
    last_source_order) tăng dần -- LEFT-PAD bằng pad_id=n_items nếu chuỗi
    ngắn hơn seq_len, cắt giữ đúng `seq_len` tương tác GẦN NHẤT nếu dài hơn.
    Left-pad giúp vị trí cuối cùng LUÔN là tương tác thật gần nhất, không
    cần theo dõi độ dài thật riêng khi suy diễn.
    """
    pad_id = n_items
    seqs = np.full((n_users, seq_len), pad_id, dtype=np.int64)
    sorted_df = train_df.sort_values(["user", "last_timestamp", "last_source_order"], kind="mergesort")
    for u, g in sorted_df.groupby("user", sort=False):
        items = g["item"].to_numpy(dtype=np.int64)[-seq_len:]
        seqs[int(u), seq_len - len(items):] = items
    return torch.from_numpy(seqs)


class SASRec(nn.Module):
    def __init__(
        self, n_users: int, n_items: int, embedding_dim: int, user_seq: torch.Tensor,
        seq_len: int = 20, n_heads: int = 2, n_blocks: int = 2, dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = int(n_items)
        self.seq_len = int(seq_len)
        self.pad_id = self.n_items

        self.item_emb = nn.Embedding(n_items + 1, embedding_dim, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(seq_len, embedding_dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=embedding_dim * 4,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.dropout = nn.Dropout(dropout)

        # buffer -- không phải tham số học được, di chuyển cùng .to(device)
        self.register_buffer("user_seq", user_seq, persistent=False)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def encode_user(self, users: torch.Tensor) -> torch.Tensor:
        seqs = self.user_seq[users]  # (B, L)
        positions = torch.arange(self.seq_len, device=seqs.device).unsqueeze(0).expand_as(seqs)
        x = self.dropout(self.item_emb(seqs) + self.pos_emb(positions))
        pad_mask = seqs == self.pad_id  # (B, L) True tại vị trí đệm
        out = self.encoder(x, mask=self.causal_mask, src_key_padding_mask=pad_mask)
        return out[:, -1, :]  # vị trí cuối luôn là tương tác thật gần nhất (left-pad)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u_repr = self.encode_user(users)
        i_emb = self.item_emb(items)
        return (u_repr * i_emb).sum(dim=-1)
