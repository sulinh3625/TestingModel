from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .base import DatasetAdapter


class DataCoAdapter(DatasetAdapter):
    USER_COL = "Customer Id"
    ITEM_COL = "Product Card Id"
    VALUE_COL = "Sales"
    DATE_COL = "order date (DateOrders)"
    CATEGORY_COL = "Category Name"

    def __init__(self, path: Path, encoding: str = "latin1", nrows: int | None = None):
        self.path = Path(path)
        self.encoding = encoding
        self.nrows = nrows

    def load_events(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Không tìm thấy DataCo CSV: {self.path}")

        usecols = [self.USER_COL, self.ITEM_COL, self.VALUE_COL, self.DATE_COL]
        raw = pd.read_csv(
            self.path,
            encoding=self.encoding,
            usecols=usecols,
            nrows=self.nrows,
        )
        raw = raw.rename(columns={
            self.USER_COL: "user_raw",
            self.ITEM_COL: "item_raw",
            self.VALUE_COL: "value_raw",
            self.DATE_COL: "timestamp",
        })
        raw["source_order"] = np.arange(len(raw), dtype=np.int64)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw["value_raw"] = pd.to_numeric(raw["value_raw"], errors="coerce").fillna(0.0)
        raw = raw.dropna(subset=["user_raw", "item_raw", "timestamp"]).copy()
        return raw[["user_raw", "item_raw", "timestamp", "value_raw", "source_order"]]

    def load_item_categories(self) -> dict:
        """Đọc Category Name theo Product Card Id -- đặc trưng nội dung phụ
        trợ (side information), KHÔNG thuộc schema sự kiện chung dùng cho
        pipeline chính (user_raw/item_raw/timestamp/value_raw) nên tách
        thành hàm riêng, chỉ dùng cho thực nghiệm content-based cold-start
        (src/baselines/content_based.py).
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Không tìm thấy DataCo CSV: {self.path}")
        raw = pd.read_csv(
            self.path, encoding=self.encoding,
            usecols=[self.ITEM_COL, self.CATEGORY_COL],
        ).drop_duplicates(subset=[self.ITEM_COL])
        return dict(zip(raw[self.ITEM_COL], raw[self.CATEGORY_COL]))
