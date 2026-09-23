"""Demo API — nạp checkpoint NeuMF/GMF/MLP/EarlyFusion đã huấn luyện (Chương 4
của báo cáo) để phục vụ suy diễn (inference-only) qua REST API, đúng thiết kế
3 tầng ở mục 3.7 của báo cáo (Report DACNTT/content/C3.tex).

Chạy:
    uvicorn demo.backend.main:app --reload --port 8000
    (chạy từ thư mục neumf_project_v2/)

Mở trình duyệt tại http://localhost:8000
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import build_adapter  # noqa: E402
from src.data_pipeline.preprocessing import build_interactions  # noqa: E402
from src.data_pipeline.negative_sampling import build_user_positive_sets  # noqa: E402
from src.evaluation.long_tail import define_head_items  # noqa: E402
from src.models.neumf import GMF, MLP, NeuMF  # noqa: E402
from src.models.early_fusion import EarlyFusionModel  # noqa: E402

DATASETS = {
    "dataco": {"config": "configs/dataco.yaml", "prefix": "dataco_", "label": "DataCo Supply Chain"},
    "hm": {"config": "configs/hm_subset.yaml", "prefix": "hm_", "label": "H&M Fashion (lát cắt 100k)"},
}


def resolve_latest_run_tag(prefix: str) -> str:
    """Tìm run_tag MỚI NHẤT khớp tiền tố, ưu tiên run đã huấn luyện xong hoàn
    chỉnh (có results.json + ít nhất 1 checkpoint .pt) -- tránh chọn nhầm một
    run bị ngắt giữa chừng (vd. do bấm dừng khi đang train)."""
    exp_dir = PROJECT_ROOT / "outputs" / "experiments"
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    candidates = []
    if exp_dir.exists():
        for d in exp_dir.iterdir():
            if not d.is_dir() or not d.name.startswith(prefix):
                continue
            results_path = d / "results.json"
            has_ckpt = (ckpt_dir / d.name).exists() and any((ckpt_dir / d.name).glob("*.pt"))
            if results_path.exists() and has_ckpt:
                candidates.append((results_path.stat().st_mtime, d.name))
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy run nào đã train xong hoàn chỉnh với tiền tố '{prefix}' "
            f"trong outputs/experiments/. Hãy chạy scripts/run_all.py trước."
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]

MODEL_CHECKPOINTS = {
    "GMF": "gmf.pt",
    "MLP": "mlp.pt",
    "EarlyFusion": "early_fusion.pt",
    "NeuMF-Scratch": "neumf_scratch.pt",
    "NeuMF-Pretrained": "neumf_pretrained.pt",
}
MODEL_ORDER = ["GMF", "MLP", "EarlyFusion", "NeuMF-Scratch", "NeuMF-Pretrained"]


def _build_model(name: str, n_users: int, n_items: int, cfg) -> torch.nn.Module:
    d = cfg.model.embedding_dim
    layers = list(cfg.model.mlp_layers)
    dropout = cfg.model.dropout
    if name == "GMF":
        return GMF(n_users, n_items, d)
    if name == "MLP":
        return MLP(n_users, n_items, d, layers, dropout)
    if name == "EarlyFusion":
        return EarlyFusionModel(n_users, n_items, d, layers, dropout)
    if name in ("NeuMF-Scratch", "NeuMF-Pretrained"):
        return NeuMF(n_users, n_items, d, layers, dropout)
    raise ValueError(f"Model không hỗ trợ: {name}")


class DatasetContext:
    """Tái tạo đúng pipeline tiền xử lý (Chương 3 báo cáo) để suy ra lại
    user2idx/item2idx khớp chính xác với checkpoint đã huấn luyện, cộng thêm
    bảng nhãn hiển thị sản phẩm (tra cứu từ dữ liệu thô/articles.csv)."""

    def __init__(self, key: str):
        info = DATASETS[key]
        self.key = key
        self.label = info["label"]
        self.run_tag = resolve_latest_run_tag(info["prefix"])

        print(f"[demo] Đang nạp bối cảnh dữ liệu cho '{key}' ({info['config']}), "
              f"run_tag mới nhất = '{self.run_tag}'...")
        cfg, adapter = build_adapter(info["config"])
        self.cfg = cfg
        events = adapter.load_events()
        data = build_interactions(events, cfg.dataset.k_core)

        self.n_users = data.n_users
        self.n_items = data.n_items
        self.user2idx = data.user2idx
        self.item2idx = data.item2idx
        self.idx2user = {v: k for k, v in data.user2idx.items()}
        self.idx2item = {v: k for k, v in data.item2idx.items()}
        self.seen = build_user_positive_sets(data.df, data.n_users)
        self.head_items = define_head_items(data.df, data.n_items, cfg.evaluation.head_fraction)

        ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints" / self.run_tag
        table_dir = PROJECT_ROOT / "outputs" / "tables" / self.run_tag
        self.beyond_path = table_dir / "beyond_accuracy.csv"

        self.models: dict[str, torch.nn.Module] = {}
        for name, fname in MODEL_CHECKPOINTS.items():
            path = ckpt_dir / fname
            if not path.exists():
                continue
            model = _build_model(name, self.n_users, self.n_items, cfg)
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.models[name] = model
        print(f"[demo]   -> {self.n_users:,} users, {self.n_items:,} items, "
              f"{len(self.models)} model(s) nạp thành công: {list(self.models)}")

        primary_path = table_dir / "primary_results.csv"
        self.primary_results = pd.read_csv(primary_path, index_col=0) if primary_path.exists() else None

        self._external_user_id = {}   # model_idx -> chuỗi ID gốc hiển thị
        self._resolve_external_user = {}  # chuỗi ID gốc -> model_idx
        self.item_label: dict[int, str] = {}
        self._build_labels(key)

    def _build_labels(self, key: str):
        if key == "dataco":
            for user_raw, idx in self.user2idx.items():
                ext = str(int(user_raw))
                self._external_user_id[idx] = ext
                self._resolve_external_user[ext] = idx
            raw_path = PROJECT_ROOT / "data" / "raw" / "dataco" / "DataCoSupplyChainDataset.csv"
            try:
                raw = pd.read_csv(
                    raw_path, encoding="latin1",
                    usecols=["Product Card Id", "Product Name", "Category Name"],
                ).drop_duplicates("Product Card Id")
                name_map = dict(zip(raw["Product Card Id"], raw["Product Name"]))
                cat_map = dict(zip(raw["Product Card Id"], raw["Category Name"]))
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"[demo] Cảnh báo: không đọc được nhãn sản phẩm DataCo ({exc})")
                name_map, cat_map = {}, {}
            for item_raw, idx in self.item2idx.items():
                nm = name_map.get(item_raw)
                cat = cat_map.get(item_raw)
                self.item_label[idx] = f"{nm} ({cat})" if nm else f"Sản phẩm #{item_raw}"

        elif key == "hm":
            # configs/hm_subset.yaml đọc trực tiếp transactions_train.csv (không
            # qua cache Parquet), nên user2idx/item2idx ở đây được khoá bằng
            # đúng customer_id (hash 64 ký tự) / article_id (10 chữ số) GỐC —
            # không cần đi qua user_id_map/item_id_map (chỉ áp dụng khi dùng
            # configs/hm.yaml trỏ tới data/processed/hm/transactions_clean.parquet).
            articles_path = PROJECT_ROOT / "data" / "raw" / "hm" / "articles.csv"

            for customer_id, idx in self.user2idx.items():
                ext = str(customer_id)
                short_ext = ext[:12] + "…" if len(ext) > 12 else ext
                self._external_user_id[idx] = short_ext
                self._resolve_external_user[short_ext] = idx
                self._resolve_external_user[ext] = idx  # cho phép nhập ID đầy đủ

            prod_map: dict[str, str] = {}
            if articles_path.exists():
                try:
                    arts = pd.read_csv(
                        articles_path, dtype={"article_id": "string"},
                        usecols=["article_id", "prod_name", "product_type_name", "colour_group_name"],
                    )
                    for row in arts.itertuples(index=False):
                        prod_map[row.article_id] = f"{row.prod_name} — {row.product_type_name} ({row.colour_group_name})"
                except Exception as exc:  # pragma: no cover
                    print(f"[demo] Cảnh báo: không đọc được articles.csv ({exc})")
            for article_id, idx in self.item2idx.items():
                label = prod_map.get(str(article_id))
                self.item_label[idx] = label or f"Sản phẩm #{article_id}"

    def sample_users(self, limit: int = 30) -> list[str]:
        return list(self._external_user_id.values())[:limit]

    def resolve_user(self, external_id: str) -> int:
        if external_id in self._resolve_external_user:
            return self._resolve_external_user[external_id]
        raise KeyError(external_id)

    def recommend(self, model_name: str, external_user_id: str, k: int = 10) -> list[dict]:
        u_idx = self.resolve_user(external_user_id)
        model = self.models[model_name]
        seen = self.seen[u_idx]
        candidates = np.array([i for i in range(self.n_items) if i not in seen], dtype=np.int64)
        if len(candidates) == 0:
            return []
        with torch.no_grad():
            users_t = torch.full((len(candidates),), u_idx, dtype=torch.long)
            items_t = torch.from_numpy(candidates).long()
            scores = torch.sigmoid(model(users_t, items_t)).numpy()
        top_k = min(k, len(candidates))
        order = np.argpartition(-scores, top_k - 1)[:top_k]
        order = order[np.argsort(-scores[order])]
        results = []
        for rank, pos in enumerate(order, start=1):
            item_idx = int(candidates[pos])
            results.append({
                "rank": rank,
                "item_idx": item_idx,
                "item_id": str(self.idx2item[item_idx]),
                "label": self.item_label.get(item_idx, f"Item #{item_idx}"),
                "score": round(float(scores[pos]), 4),
                "is_head": item_idx in self.head_items,
            })
        return results

    def history(self, external_user_id: str, limit: int = 20) -> list[dict]:
        u_idx = self.resolve_user(external_user_id)
        items = sorted(self.seen[u_idx])[:limit]
        return [
            {
                "item_idx": i,
                "item_id": str(self.idx2item[i]),
                "label": self.item_label.get(i, f"Item #{i}"),
                "is_head": i in self.head_items,
            }
            for i in items
        ]

    def _item_embedding(self) -> tuple[str, torch.Tensor] | None:
        for name in ("GMF", "NeuMF-Pretrained", "NeuMF-Scratch", "MLP"):
            model = self.models.get(name)
            if model is None:
                continue
            if hasattr(model, "item_emb"):
                return name, model.item_emb.weight.detach()
            if hasattr(model, "gmf_item_emb"):
                return name, model.gmf_item_emb.weight.detach()
        return None

    def similar_items(self, item_idx: int, k: int = 8) -> dict:
        found = self._item_embedding()
        if found is None:
            return {"source_model": None, "items": []}
        source_model, emb = found
        if not (0 <= item_idx < emb.shape[0]):
            raise KeyError(item_idx)
        vecs = torch.nn.functional.normalize(emb, dim=1)
        sims = (vecs @ vecs[item_idx]).numpy()
        sims[item_idx] = -1.0  # loại chính nó
        top_k = min(k, len(sims) - 1)
        order = np.argsort(-sims)[:top_k]
        items = [
            {
                "item_idx": int(i),
                "item_id": str(self.idx2item[int(i)]),
                "label": self.item_label.get(int(i), f"Item #{i}"),
                "similarity": round(float(sims[i]), 4),
                "is_head": int(i) in self.head_items,
            }
            for i in order
        ]
        return {"source_model": source_model, "items": items}

    def beyond_accuracy(self) -> list[dict]:
        if not self.beyond_path.exists():
            return []
        df = pd.read_csv(self.beyond_path).rename(columns={"Model": "model"})
        df = df[df["model"].isin(MODEL_ORDER)]
        return df.to_dict(orient="records")


_contexts: dict[str, DatasetContext] = {}


def get_context(key: str) -> DatasetContext:
    if key not in DATASETS:
        raise HTTPException(404, f"Dataset '{key}' không tồn tại. Chọn 1 trong: {list(DATASETS)}")
    if key not in _contexts:
        _contexts[key] = DatasetContext(key)
    return _contexts[key]


app = FastAPI(title="NeuMF Demo API", description="Suy diễn (inference-only) trên checkpoint đã huấn luyện — xem Chương 3 (mục 3.7) và Chương 4 của báo cáo.")


@app.middleware("http")
async def no_cache(request, call_next):
    """Demo đang trong giai đoạn chỉnh sửa liên tục — tắt hẳn cache của trình
    duyệt để tránh hiển thị bản HTML/JS cũ sau khi server đã cập nhật."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/api/health")
def health():
    return {"status": "ok", "datasets_loaded": list(_contexts.keys())}


@app.post("/api/reload/{dataset}")
def reload_dataset(dataset: str):
    """Xoá cache trong bộ nhớ để lần gọi API tiếp theo nạp lại run_tag MỚI
    NHẤT hiện có trên đĩa -- dùng sau khi train xong một lần chạy mới, không
    cần khởi động lại server."""
    if dataset not in DATASETS:
        raise HTTPException(404, f"Dataset '{dataset}' không tồn tại")
    old_tag = _contexts.pop(dataset, None)
    new_ctx = get_context(dataset)
    return {
        "dataset": dataset,
        "previous_run_tag": old_tag.run_tag if old_tag else None,
        "current_run_tag": new_ctx.run_tag,
    }


@app.get("/api/datasets")
def list_datasets():
    out = []
    for k, v in DATASETS.items():
        try:
            tag = resolve_latest_run_tag(v["prefix"])
        except FileNotFoundError:
            tag = None
        out.append({"key": k, "label": v["label"], "run_tag": tag})
    return out


@app.get("/api/models/{dataset}")
def list_models(dataset: str):
    ctx = get_context(dataset)
    out = []
    for name in MODEL_ORDER:
        if name not in ctx.models:
            continue
        metrics = {}
        if ctx.primary_results is not None and name in ctx.primary_results.index:
            metrics = {k: round(float(v), 4) for k, v in ctx.primary_results.loc[name].to_dict().items()}
        out.append({"name": name, "metrics": metrics})
    return out


@app.get("/api/users/{dataset}")
def list_users(dataset: str, limit: int = 30):
    ctx = get_context(dataset)
    return ctx.sample_users(limit=limit)


@app.get("/api/metrics/{dataset}")
def metrics(dataset: str):
    ctx = get_context(dataset)
    if ctx.primary_results is None:
        return []
    df = ctx.primary_results.rename_axis("model").reset_index()
    df = df[df["model"].isin(MODEL_ORDER)]
    return df.to_dict(orient="records")


@app.get("/api/recommend/{dataset}/{model_name}/{user_id}")
def recommend(dataset: str, model_name: str, user_id: str, k: int = 10):
    ctx = get_context(dataset)
    if model_name not in ctx.models:
        raise HTTPException(404, f"Model '{model_name}' chưa có checkpoint cho dataset '{dataset}'")
    try:
        recs = ctx.recommend(model_name, user_id, k=k)
    except KeyError:
        raise HTTPException(
            404,
            f"User '{user_id}' không có trong dữ liệu huấn luyện — đây là tình huống Cold-start "
            f"nằm ngoài phạm vi mô hình (xem mục 3.7.2 / 4.4 của báo cáo).",
        )
    return {"dataset": dataset, "model": model_name, "user_id": user_id, "k": k, "recommendations": recs}


@app.get("/api/history/{dataset}/{user_id}")
def history(dataset: str, user_id: str, limit: int = 20):
    ctx = get_context(dataset)
    try:
        items = ctx.history(user_id, limit=limit)
    except KeyError:
        raise HTTPException(404, f"User '{user_id}' không có trong dữ liệu huấn luyện")
    return {"dataset": dataset, "user_id": user_id, "total_purchases": len(ctx.seen[ctx.resolve_user(user_id)]), "items": items}


@app.get("/api/similar-items/{dataset}/{item_idx}")
def similar_items(dataset: str, item_idx: int, k: int = 8):
    ctx = get_context(dataset)
    try:
        result = ctx.similar_items(item_idx, k=k)
    except KeyError:
        raise HTTPException(404, f"Sản phẩm #{item_idx} không tồn tại trong dataset '{dataset}'")
    return {"dataset": dataset, "item_idx": item_idx, **result}


@app.get("/api/beyond/{dataset}")
def beyond_accuracy(dataset: str):
    ctx = get_context(dataset)
    return ctx.beyond_accuracy()


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
