"""LightGCN (He et al., SIGIR 2020) — "Simplifying and Powering Graph
Convolution Network for Recommendation".

Khác biệt cốt lõi so với GMF/MLP/NeuMF: các mô hình đó tính điểm trực tiếp
từ vector Embedding thô (lookup theo ID), không dùng thông tin cấu trúc đồ
thị User--Item. LightGCN lan truyền (propagate) Embedding qua đồ thị hai
phía User--Item chuẩn hoá đối xứng $k$ lớp trước khi tính điểm, cho phép mã
hoá tương tác bậc cao (\emph{high-order connectivity}, VD "user tương tự
mua item gì") mà MF/NeuMF thuần ID không nắm bắt được trực tiếp. LightGCN
bỏ hẳn phép biến đổi tuyến tính (weight matrix) và hàm kích hoạt phi tuyến ở
mỗi lớp lan truyền so với NGCF gốc -- theo paper gốc, hai thành phần này dư
thừa và gây khó huấn luyện cho bài toán Collaborative Filtering thuần tuý.

Đồ thị chuẩn hoá CHỈ xây từ TRAIN (build_norm_adj nhận train_df), khớp đúng
nguyên tắc chống rò rỉ dữ liệu áp dụng xuyên suốt dự án (không dùng Validation
hay Test để xây cấu trúc đồ thị).

Interface forward(users, items) -> logits giống hệt GMF/MLP/NeuMF nên cắm
thẳng được vào toàn bộ hạ tầng train_one_model/evaluate_torch_model sẵn có,
không cần sửa gì ở tầng huấn luyện/đánh giá chung.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def build_norm_adj(train_df, n_users: int, n_items: int) -> torch.Tensor:
    """Xây ma trận kề đồ thị hai phía User--Item, chuẩn hoá đối xứng
    $D^{-1/2} A D^{-1/2}$ (đúng công thức lan truyền LightGCN), CHỈ dùng
    cạnh từ TRAIN. Trả về sparse COO tensor kích thước (M+N) x (M+N), CPU
    (chuyển device ở nơi gọi).
    """
    users = train_df["user"].to_numpy(dtype=np.int64)
    items = train_df["item"].to_numpy(dtype=np.int64) + n_users  # offset item vào nửa sau đồ thị

    row = np.concatenate([users, items])
    col = np.concatenate([items, users])
    data = np.ones(len(row), dtype=np.float32)

    n_nodes = n_users + n_items
    adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

    degree = np.asarray(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(degree, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_mat = sp.diags(deg_inv_sqrt)

    norm_adj = (d_mat @ adj @ d_mat).tocoo()
    indices = torch.from_numpy(np.vstack([norm_adj.row, norm_adj.col])).long()
    values = torch.from_numpy(norm_adj.data).float()
    return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, norm_adj: torch.Tensor, n_layers: int = 3):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_layers = int(n_layers)
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        # buffer -- di chuyển cùng .to(device) với model, không phải tham số học được
        self.register_buffer("norm_adj", norm_adj, persistent=False)
        self._cached: tuple[torch.Tensor, torch.Tensor] | None = None

    def train(self, mode: bool = True):
        if mode:
            self._cached = None  # weight sắp đổi -> huỷ cache lan truyền cũ
        return super().train(mode)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Lan truyền Embedding qua $K$ lớp đồ thị, biểu diễn cuối = trung
        bình cộng biểu diễn ở mọi lớp (kể cả lớp 0, đúng công thức gốc).
        Cache lại khi ở chế độ eval (trọng số không đổi giữa các lần gọi
        trong 1 lượt đánh giá) để tránh lan truyền lại toàn đồ thị mỗi batch.
        """
        if not self.training and self._cached is not None:
            return self._cached

        emb0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layer_embs = [emb0]
        emb = emb0
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            layer_embs.append(emb)
        final = torch.stack(layer_embs, dim=0).mean(dim=0)
        users_final, items_final = torch.split(final, [self.n_users, self.n_items], dim=0)

        if not self.training:
            self._cached = (users_final, items_final)
        return users_final, items_final

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        users_final, items_final = self.propagate()
        u = users_final[users]
        i = items_final[items]
        return (u * i).sum(dim=-1)
