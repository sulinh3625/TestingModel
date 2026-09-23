from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .negative_sampling import sample_train_negatives, build_positive_codes, sample_train_negatives_vectorized


class TrainDataset(Dataset):
    """Dynamic negative sampling cho implicit feedback.

    Quan trọng: exclude_set phải lấy từ TRAIN positives, không dùng validation/test
    để tránh future-information leakage vào quá trình huấn luyện.

    fast_negative_sampling=False (mặc định): giữ NGUYÊN hành vi gốc — lặp
    Python theo từng dòng, dựng mask O(n_items) mỗi lần gọi qua
    sample_train_negatives. Đủ nhanh cho DataCo/H&M subset (catalog nhỏ,
    đã kiểm thử kỹ — xem tests/test_data_pipeline.py), KHÔNG đổi để tránh
    rủi ro phá vỡ hành vi đã tested.

    fast_negative_sampling=True: dùng sample_train_negatives_vectorized
    (kỹ thuật sample-and-reject, vector hoá numpy) — bắt buộc cho quy mô
    lớn (VD H&M full scale, ~90.690 sản phẩm x ~26,2 triệu tương tác/epoch)
    vì cách dựng mask O(n_items) x số dòng không khả thi về thời gian ở quy
    mô này. Đúng cùng bất biến chính xác (không gán nhãn âm cho positive
    thật), chỉ khác cơ chế lấy mẫu bên trong.
    """

    def __init__(
        self, train_df, n_items, train_positive_sets, neg_ratio: int, seed: int = 42,
        fast_negative_sampling: bool = False,
    ):
        self.n_items = int(n_items)
        self.neg_ratio = int(neg_ratio)
        self.train_positive_sets = train_positive_sets
        self.pos_users = train_df["user"].to_numpy(dtype=np.int64)
        self.pos_items = train_df["item"].to_numpy(dtype=np.int64)
        self.pos_weights = train_df["sample_weight"].to_numpy(dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.fast_negative_sampling = bool(fast_negative_sampling)
        self.positive_codes = build_positive_codes(train_df, self.n_items) if self.fast_negative_sampling else None
        self.users = self.items = self.labels = self.sample_weights = None
        self.resample()

    def resample(self):
        """Sinh lại negative cho mỗi epoch."""
        if self.fast_negative_sampling:
            self._resample_vectorized()
        else:
            self._resample_loop()

    def _resample_vectorized(self):
        n_pos = len(self.pos_users)
        neg_items = sample_train_negatives_vectorized(
            self.pos_users, self.n_items, self.neg_ratio, self.positive_codes, self.rng,
        )  # (n_pos, neg_ratio)
        neg_users = np.repeat(self.pos_users, self.neg_ratio)

        self.users = np.concatenate([self.pos_users, neg_users])
        self.items = np.concatenate([self.pos_items, neg_items.reshape(-1)])
        self.labels = np.concatenate([
            np.ones(n_pos, dtype=np.float32), np.zeros(len(neg_users), dtype=np.float32),
        ])
        self.sample_weights = np.concatenate([
            self.pos_weights, np.ones(len(neg_users), dtype=np.float32),
        ])

    def _resample_loop(self):
        """Hành vi gốc — lặp Python theo từng dòng (xem docstring lớp).

        Gom kết quả bằng mảng NumPy có sẵn kích thước thay vì list.append()
        + tuple-của-Python cho từng phần tử — giảm overhead đáng kể so với
        cách viết ngây thơ, dù vẫn còn vòng lặp Python cấp-dòng.
        """
        n_pos = len(self.pos_users)
        max_total = n_pos * (1 + self.neg_ratio)  # cận trên; catalog nhỏ có thể sinh ít negative hơn

        users = np.empty(max_total, dtype=np.int64)
        items = np.empty(max_total, dtype=np.int64)
        labels = np.zeros(max_total, dtype=np.float32)
        weights = np.ones(max_total, dtype=np.float32)

        cursor = 0  # con trỏ ghi — chỉ tăng, không bao giờ resize mảng giữa chừng
        for u, i, w in zip(self.pos_users, self.pos_items, self.pos_weights):
            users[cursor] = u
            items[cursor] = i
            labels[cursor] = 1.0
            weights[cursor] = w
            cursor += 1

            negs = sample_train_negatives(
                self.train_positive_sets[int(u)], self.n_items, self.neg_ratio, self.rng
            )
            n_neg = len(negs)
            users[cursor : cursor + n_neg] = u
            items[cursor : cursor + n_neg] = negs
            # labels/weights của negative đã đúng mặc định (0.0 / 1.0)
            cursor += n_neg

        self.users = users[:cursor]
        self.items = items[:cursor]
        self.labels = labels[:cursor]
        self.sample_weights = weights[:cursor]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.sample_weights[idx], dtype=torch.float32),
        )