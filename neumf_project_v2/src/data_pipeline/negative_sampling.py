from __future__ import annotations

import numpy as np


def build_user_positive_sets(df, n_users: int):
    positives = [set() for _ in range(n_users)]
    for u, i in zip(df["user"].values, df["item"].values):
        positives[int(u)].add(int(i))
    return positives


def available_negatives(exclude_set: set[int], n_items: int) -> np.ndarray:
    if not exclude_set:
        return np.arange(n_items, dtype=np.int64)
    mask = np.ones(n_items, dtype=bool)
    valid = [i for i in exclude_set if 0 <= i < n_items]
    if valid:
        mask[np.fromiter(valid, dtype=np.int64)] = False
    return np.flatnonzero(mask).astype(np.int64)


def sample_train_negatives(
    exclude_set: set[int], n_items: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    available = available_negatives(exclude_set, n_items)
    if len(available) == 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(available, size=n_samples, replace=len(available) < n_samples).astype(np.int64)


def sample_eval_negatives(
    exclude_set: set[int], n_items: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    available = available_negatives(exclude_set, n_items)
    if len(available) == 0:
        return np.empty(0, dtype=np.int64)
    n_take = min(n_samples, len(available))
    return rng.choice(available, size=n_take, replace=False).astype(np.int64)


def build_positive_codes(train_df, n_items: int) -> np.ndarray:
    """Mã hoá mọi cặp (user,item) positive thành 1 số nguyên user*n_items+item,
    sắp xếp tăng dần -- dùng cho tra cứu vector hoá bằng searchsorted thay vì
    dựng mask O(n_items) cho từng dòng. CHỈ dùng cho catalog lớn (VD H&M full
    scale, xem sample_train_negatives_vectorized) nơi cách dựng mask cũ
    (available_negatives) không khả thi về thời gian ở quy mô hàng chục
    triệu dòng train.
    """
    codes = train_df["user"].to_numpy(dtype=np.int64) * np.int64(n_items) + train_df["item"].to_numpy(dtype=np.int64)
    return np.unique(codes)  # np.unique đã trả về mảng sắp xếp tăng dần


def sample_train_negatives_vectorized(
    pos_users: np.ndarray, n_items: int, neg_ratio: int,
    positive_codes_sorted: np.ndarray, rng: np.random.Generator,
    max_vectorized_retries: int = 5,
) -> np.ndarray:
    """Lấy mẫu âm vector hoá cho TOÀN BỘ tập positive cùng lúc bằng kỹ thuật
    'sample rồi loại trừ' (sample-and-reject) thay vì vòng lặp Python +
    dựng mask O(n_items) cho từng dòng (sample_train_negatives) -- cần
    thiết khi n_items và số dòng train đều lớn (VD H&M full scale:
    ~90.690 sản phẩm x ~26,2 triệu tương tác/epoch).

    Đúng bất biến CHÍNH XÁC như sample_train_negatives (không bao giờ gán
    nhãn âm cho 1 cặp positive thật): sau `max_vectorized_retries` lượt thử
    lại vector hoá, các vị trí HIẾM còn va chạm (thực tế gần như luôn bằng 0
    khi catalog đủ lớn so với số positive/user) được xử lý dứt điểm bằng
    sample_train_negatives (chính xác tuyệt đối, chấp nhận chậm hơn vì số
    lượng còn lại rất nhỏ).

    Trả về mảng (n_pos, neg_ratio).
    """
    n_pos = len(pos_users)
    total = n_pos * neg_ratio
    rep_users = np.repeat(pos_users, neg_ratio)
    neg_items = rng.integers(0, n_items, size=total)

    for _ in range(max_vectorized_retries):
        codes = rep_users.astype(np.int64) * np.int64(n_items) + neg_items.astype(np.int64)
        idx = np.searchsorted(positive_codes_sorted, codes)
        idx = np.clip(idx, 0, len(positive_codes_sorted) - 1)
        collide = positive_codes_sorted[idx] == codes
        n_collide = int(collide.sum())
        if n_collide == 0:
            break
        neg_items[collide] = rng.integers(0, n_items, size=n_collide)
    else:
        # Fallback chính xác tuyệt đối cho phần dư hiếm gặp còn lại sau khi
        # hết lượt thử vector hoá -- gộp theo user để dựng mask 1 lần/user
        # thay vì 1 lần/dòng.
        codes = rep_users.astype(np.int64) * np.int64(n_items) + neg_items.astype(np.int64)
        idx = np.clip(np.searchsorted(positive_codes_sorted, codes), 0, len(positive_codes_sorted) - 1)
        collide = positive_codes_sorted[idx] == codes
        if collide.any():
            residual_users = np.unique(rep_users[collide])
            for u in residual_users:
                u_mask = collide & (rep_users == u)
                n_needed = int(u_mask.sum())
                user_positive_items = set(
                    (positive_codes_sorted[
                        (positive_codes_sorted >= u * n_items) & (positive_codes_sorted < (u + 1) * n_items)
                    ] - u * n_items).tolist()
                )
                neg_items[u_mask] = sample_train_negatives(user_positive_items, n_items, n_needed, rng)

    return neg_items.reshape(n_pos, neg_ratio)
