from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .base import DatasetAdapter


class HMAdapter(DatasetAdapter):
    """Adapter H&M Personalized Fashion Recommendations."""

    def __init__(
        self,
        path: Path,
        encoding: str = "utf-8",
        nrows: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.path = Path(path)
        self.encoding = encoding
        self.nrows = nrows
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.end_date = pd.Timestamp(end_date) if end_date else None

    def load_events(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Không tìm thấy H&M transactions: {self.path}")

        if self.path.suffix.lower() == ".parquet":
            return self._load_from_cache()

        raw = pd.read_csv(
            self.path,
            encoding=self.encoding,
            usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
            dtype={"customer_id": "string", "article_id": "string"},
            nrows=self.nrows,
        )
        raw["source_order"] = np.arange(len(raw), dtype=np.int64)
        raw["t_dat"] = pd.to_datetime(raw["t_dat"], errors="coerce")
        raw["article_id"] = raw["article_id"].astype("string").str.zfill(10)
        raw["price"] = pd.to_numeric(raw["price"], errors="coerce").fillna(0.0)

        if self.start_date is not None:
            raw = raw[raw["t_dat"] >= self.start_date]
        if self.end_date is not None:
            raw = raw[raw["t_dat"] <= self.end_date]

        raw = raw.dropna(subset=["customer_id", "article_id", "t_dat"]).copy()
        raw = raw.rename(columns={
            "customer_id": "user_raw",
            "article_id": "item_raw",
            "t_dat": "timestamp",
            "price": "value_raw",
        })
        return raw[["user_raw", "item_raw", "timestamp", "value_raw", "source_order"]]

    def _load_from_cache(self) -> pd.DataFrame:
        """Đọc file Parquet đã tiền xử lý sẵn (xem scripts/00_prepare_hm_cache.py).

        user_raw/item_raw ở đây đã là mã int32 (factorize từ customer_id/article_id
        gốc) thay vì chuỗi hash -- không ảnh hưởng đến kết quả vì build_interactions
        chỉ cần giá trị duy nhất, tự re-index lại từ đầu.
        """
        raw = pd.read_parquet(self.path, columns=[
            "user_raw", "item_raw", "timestamp", "value_raw", "source_order",
        ])
        if self.nrows is not None:
            raw = raw.sort_values("source_order", kind="mergesort").head(self.nrows)
        if self.start_date is not None:
            raw = raw[raw["timestamp"] >= self.start_date]
        if self.end_date is not None:
            raw = raw[raw["timestamp"] <= self.end_date]
        return raw.reset_index(drop=True)
