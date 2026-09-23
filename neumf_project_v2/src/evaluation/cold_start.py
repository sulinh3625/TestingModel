from __future__ import annotations

import math

import pandas as pd


def define_cold_users(train_df: pd.DataFrame, n_users: int, cold_fraction: float = 0.20) -> set[int]:
    """User 'cold' = nhóm cold_fraction% có ÍT tương tác nhất trong TRAIN.

    Đối xứng với define_head_items (long_tail.py) nhưng áp dụng cho USER thay
    vì ITEM — trả lời câu hỏi "mô hình gợi ý kém hơn bao nhiêu cho người dùng
    ít dữ liệu?" (mục 5.4 đề cương chi tiết).

    Lưu ý phạm vi: đây là cold-start THEO NGƯỠNG TƯƠNG ĐỐI trong tập user đã
    qua lọc k-core (mọi user còn lại đều có >= k_core tương tác TỔNG, tức
    >= k_core-2 tương tác TRAIN sau khi trừ val/test) — KHÔNG phải cold-start
    tuyệt đối (user hoàn toàn chưa có tương tác nào, chưa từng vào ma trận
    User-Item). Cold-start tuyệt đối nằm ngoài khả năng của mọi mô hình
    collaborative filtering thuần ID, kể cả NeuMF — đây là giới hạn lý
    thuyết cố hữu đã nêu ở "Đề cương chi tiết" mục 3.5, không phải thứ có
    thể "đo" bằng thực nghiệm trên chính tập dữ liệu đó.

    Item cold-start (item ít tương tác) đã được bao phủ bởi phân tích
    Long-tail sẵn có (định nghĩa theo popularity trong TRAIN, xem
    long_tail.py) — không lặp lại metric tương đương ở đây để tránh 2 bảng
    số liệu đo cùng một hiện tượng dưới 2 tên khác nhau.
    """
    counts = train_df["user"].value_counts()
    all_counts = [(u, int(counts.get(u, 0))) for u in range(n_users)]
    all_counts.sort(key=lambda pair: pair[1])  # tăng dần theo số tương tác train
    n_cold = max(1, int(math.ceil(n_users * cold_fraction)))
    return set(u for u, _ in all_counts[:n_cold])


def split_records_by_coldness(records, cold_users: set[int]):
    cold, warm = [], []
    for record in records:
        (cold if record.user in cold_users else warm).append(record)
    return cold, warm
