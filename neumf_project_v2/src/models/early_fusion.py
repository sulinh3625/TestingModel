from __future__ import annotations

import torch
import torch.nn as nn


class EarlyFusionModel(nn.Module):
    """Baseline Early Fusion — đối chứng trực tiếp với NeuMF (Late Fusion).

    Khác NeuMF (4 embedding riêng cho GMF/MLP, chỉ nối ở lớp output cuối),
    EarlyFusionModel dùng MỘT bộ embedding user/item duy nhất, nối
    (concatenate) ngay từ đầu vào rồi đưa qua một mạng MLP chung.
    Không có nhánh tương tác nhân element-wise (GMF) riêng biệt.

    Dùng để trả lời trực tiếp câu hỏi giảng viên: "Early fusion hay late
    fusion hiệu quả hơn trong bài toán này?" — cùng embedding_dim,
    mlp_layers, dropout, optimizer, LR, epoch budget với NeuMF để so sánh
    công bằng (fair ablation).
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int, layers: list[int], dropout: float = 0.2):
        super().__init__()
        input_size = 2 * embedding_dim
        if not layers or layers[0] != input_size:
            raise ValueError(f"mlp_layers[0] phải bằng 2*embedding_dim = {input_size}")

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        modules = []
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            modules.extend([nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(dropout)])
        self.mlp_layers = nn.Sequential(*modules)
        self.output_layer = nn.Linear(layers[-1], 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def vector(self, users, items):
        # Kết hợp NGAY từ đầu vào (early fusion) — khác GMF.vector() (nhân
        # element-wise) và khác NeuMF (2 nhánh riêng rồi mới nối ở cuối).
        return torch.cat([self.user_emb(users), self.item_emb(items)], dim=-1)

    def forward(self, users, items):
        x = self.mlp_layers(self.vector(users, items))
        return self.output_layer(x).squeeze(-1)