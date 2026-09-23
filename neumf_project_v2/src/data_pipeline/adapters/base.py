from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class DatasetAdapter(ABC):
    """Chuẩn hóa dataset nguồn về schema sự kiện chung.

    Output bắt buộc:
      user_raw, item_raw, timestamp, value_raw, source_order
    """

    @abstractmethod
    def load_events(self) -> pd.DataFrame:
        raise NotImplementedError
