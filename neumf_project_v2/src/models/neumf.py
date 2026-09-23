from __future__ import annotations

import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def vector(self, users, items):
        return self.user_emb(users) * self.item_emb(items)

    def forward(self, users, items):
        return self.output_layer(self.vector(users, items)).squeeze(-1)


class MLP(nn.Module):
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
        x = torch.cat([self.user_emb(users), self.item_emb(items)], dim=-1)
        return self.mlp_layers(x)

    def forward(self, users, items):
        return self.output_layer(self.vector(users, items)).squeeze(-1)


class NeuMF(nn.Module):
    """NeuMF với 4 bảng embedding độc lập cho GMF/MLP."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int, layers: list[int], dropout: float = 0.2):
        super().__init__()
        input_size = 2 * embedding_dim
        if not layers or layers[0] != input_size:
            raise ValueError(f"mlp_layers[0] phải bằng 2*embedding_dim = {input_size}")

        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)

        modules = []
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            modules.extend([nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(dropout)])
        self.mlp_layers = nn.Sequential(*modules)
        self.output_layer = nn.Linear(embedding_dim + layers[-1], 1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for emb in (self.gmf_user_emb, self.gmf_item_emb, self.mlp_user_emb, self.mlp_item_emb):
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, users, items):
        gmf = self.gmf_user_emb(users) * self.gmf_item_emb(items)
        mlp_in = torch.cat([self.mlp_user_emb(users), self.mlp_item_emb(items)], dim=-1)
        mlp = self.mlp_layers(mlp_in)
        fused = torch.cat([gmf, mlp], dim=-1)
        return self.output_layer(fused).squeeze(-1)

    @torch.no_grad()
    def load_pretrained(self, gmf: GMF, mlp: MLP, alpha: float = 0.5):
        self.gmf_user_emb.weight.copy_(gmf.user_emb.weight)
        self.gmf_item_emb.weight.copy_(gmf.item_emb.weight)
        self.mlp_user_emb.weight.copy_(mlp.user_emb.weight)
        self.mlp_item_emb.weight.copy_(mlp.item_emb.weight)

        target_linear = [m for m in self.mlp_layers if isinstance(m, nn.Linear)]
        source_linear = [m for m in mlp.mlp_layers if isinstance(m, nn.Linear)]
        for tgt, src in zip(target_linear, source_linear):
            tgt.weight.copy_(src.weight)
            tgt.bias.copy_(src.bias)

        gmf_h = gmf.output_layer.weight.squeeze(0)
        mlp_h = mlp.output_layer.weight.squeeze(0)
        self.output_layer.weight.copy_(torch.cat([alpha * gmf_h, (1 - alpha) * mlp_h]).unsqueeze(0))
