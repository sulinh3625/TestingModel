from __future__ import annotations

import numpy as np


class CategoryPopularityBaseline:
    """Baseline content-based dùng Category (danh mục sản phẩm) làm đặc
    trưng nội dung phụ trợ, thử nghiệm như một giải pháp giảm nhẹ Cold-start
    (mục 5.4 đề cương chi tiết + phản hồi GVHD): thay vì gợi ý mù quáng theo
    độ phổ biến toàn cục (MostPopularBaseline thuần ID), với mỗi user, ưu
    tiên sản phẩm phổ biến TRONG CÙNG danh mục user đã từng mua trong train.

    Trực giác: một user ít lịch sử tương tác (cold) vẫn để lộ TÍN HIỆU về
    sở thích qua (các) danh mục đã mua, dù số lượng giao dịch quá ít để mô
    hình Collaborative Filtering thuần ID (GMF/MLP/NeuMF) học được vector
    Embedding tốt cho user đó.

    off_category_penalty: hệ số nhân cho sản phẩm KHÁC danh mục user đã mua
    (không phải 0 tuyệt đối -- vẫn cần một điểm số fallback có ý nghĩa để
    xếp hạng phần candidate còn lại, tránh hoà điểm hàng loạt khi đánh giá
    Full Ranking).
    """

    def __init__(self, train_df, item_category: dict[int, str], n_items: int, off_category_penalty: float = 0.1):
        self.item_category = item_category
        self.off_category_penalty = float(off_category_penalty)

        self.pop_score = np.zeros(n_items, dtype=np.float64)
        counts = train_df["item"].value_counts()
        for item, count in counts.items():
            self.pop_score[int(item)] = float(count)

        self.user_categories: dict[int, set[str]] = {}
        for u, i in zip(train_df["user"].to_numpy(), train_df["item"].to_numpy()):
            cat = item_category.get(int(i))
            if cat is not None:
                self.user_categories.setdefault(int(u), set()).add(cat)

    def score(self, user: int, item: int) -> float:
        base = float(self.pop_score[int(item)])
        cat = self.item_category.get(int(item))
        known_cats = self.user_categories.get(int(user))
        if known_cats and cat is not None and cat in known_cats:
            return base
        return base * self.off_category_penalty
