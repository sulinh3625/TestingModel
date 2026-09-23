# Migration V1 -> V2

## Giữ

- Kiến trúc GMF / MLP / NeuMF với 4 embedding độc lập.
- `BCEWithLogitsLoss` và forward trả logits.
- Pretraining GMF + MLP và `load_pretrained(alpha=0.5)`.
- Random, MostPopular, ItemCF, BPR-MF.
- Dynamic negative sampling, multi-seed, checkpoints.

## Sửa bắt buộc

1. **Aggregate User-Item trước k-core**.
2. Binary implicit feedback là default.
3. Weighted confidence fit statistics trên train only.
4. Training negative sampler không được trả positive item.
5. Training negative pool không dùng val/test labels.
6. Full-ranking thành primary evaluation.
7. Tie ranking deterministic, không dùng `scores > positive_score` đơn giản.
8. Early stopping theo NDCG@10.
9. NeuMF scratch/pretrained dùng cùng optimizer/LR/budget trong pretraining ablation.
10. MLP dropout lấy từ config, không hard-code 0.2.
11. Long-tail dùng head fraction trên train, không hard-code top-10.
12. BPR negative sampler an toàn và hyperparameter đưa vào config.
13. Bỏ `venv/` khỏi project/repository.

## Cấu trúc V1

V1 trộn source code và raw data trong `data/`, có `venv/`, evaluator sampled-only và main.py ôm toàn bộ pipeline.

## Cấu trúc V2

V2 tách rõ:

- `configs/`: dataset/experiment config.
- `data/`: raw, processed, splits.
- `src/data_pipeline/`: adapter + preprocessing.
- `src/models/`, `src/baselines/`.
- `src/training/`.
- `src/evaluation/`.
- `scripts/`: entry points.
- `outputs/`: artifacts.
- `tests/`: methodology guards.
