from __future__ import annotations

import torch
import pytest

from src.models.neumf import GMF, MLP, NeuMF


def test_forward_shapes_and_logits():
    gmf = GMF(10, 20, 8)
    mlp = MLP(10, 20, 8, [16, 8, 4], dropout=0.1)
    neumf = NeuMF(10, 20, 8, [16, 8, 4], dropout=0.1)
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([3, 4, 5])
    assert gmf(u, i).shape == (3,)
    assert mlp(u, i).shape == (3,)
    assert neumf(u, i).shape == (3,)


def test_neumf_uses_four_independent_embeddings():
    m = NeuMF(10, 20, 8, [16, 8, 4])
    assert m.gmf_user_emb is not m.mlp_user_emb
    assert m.gmf_item_emb is not m.mlp_item_emb


def test_pretrained_copy():
    gmf = GMF(10, 20, 8)
    mlp = MLP(10, 20, 8, [16, 8, 4])
    n = NeuMF(10, 20, 8, [16, 8, 4])
    n.load_pretrained(gmf, mlp, alpha=0.5)
    assert torch.allclose(n.gmf_user_emb.weight, gmf.user_emb.weight)
    assert torch.allclose(n.mlp_user_emb.weight, mlp.user_emb.weight)


def test_mlp_dropout_is_configurable():
    m = MLP(10, 20, 8, [16, 8, 4], dropout=0.35)
    drops = [x for x in m.mlp_layers if isinstance(x, torch.nn.Dropout)]
    assert drops and all(d.p == pytest.approx(0.35) for d in drops)
