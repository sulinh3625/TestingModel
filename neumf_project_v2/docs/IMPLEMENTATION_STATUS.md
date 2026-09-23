# Implementation status — V2 refactor

## Đã triển khai

- Cây thư mục V2 tách data / source / configs / outputs / tests.
- Dataset adapters cho DataCo và H&M.
- Aggregate unique User-Item **trước** iterative k-core.
- K-core sensitivity audit.
- Temporal Leave-One-Out trên unique User-Item với deterministic tie-break.
- Assert không có overlap `(user,item)` giữa train/validation/test.
- Binary implicit feedback làm mặc định.
- Weighted confidence transform fit trên **train only**.
- Dynamic negative sampling không trả positive item.
- Training negative pool chỉ dùng train positives.
- Full-ranking evaluator chính + sampled-99 evaluator phụ.
- Tie handling deterministic.
- Early stopping theo configurable metric; default `NDCG@10`.
- MLP dropout lấy từ config.
- NeuMF scratch/pretrained dùng cùng optimizer/LR/budget ở controlled ablation.
- Long-tail định nghĩa theo head fraction từ **train set**.
- BPR-MF negative sampling an toàn, hyperparameter qua config.
- Unit tests bảo vệ methodology.

## Audit DataCo đã chạy trên file gốc 180,519 dòng

Sau khi sửa đúng thứ tự **aggregate -> k-core**:

| k | Users | Items | Unique interactions | Density |
|---:|---:|---:|---:|---:|
| 3 | 11,943 | 100 | 92,230 | 7.72% |
| 5 | 10,799 | 100 | 88,088 | 8.16% |
| 10 | 2,736 | 97 | 29,749 | 11.21% |

Với `k=5`:

- Raw rows: 180,519
- Raw users: 20,652
- Raw items: 118
- Unique User-Item pairs trước k-core: 101,196
- Repeat transaction rows: 79,323 (~43.94% raw rows)
- Final: 10,799 users, 100 items, 88,088 unique interactions
- Train: 66,490
- Validation: 10,799
- Test: 10,799
- Train/Val/Test user-item overlap: **0 / 0 / 0**

Kết quả này khác V1 (ví dụ số test users 11,481) vì V1 chạy k-core trên transaction rows trước khi aggregate.

## Kiểm thử

`pytest -q` -> **15 tests passed**.

Smoke test end-to-end trên DataCo subset đã chạy thành công. Full DataCo training chưa được chạy đến hội tụ trong môi trường CPU của phiên này; đó là bước tiếp theo sau khi người dùng đặt raw CSV vào đúng thư mục và chọn CPU/GPU.

## Chưa triển khai trong phase này

- Optuna/fair hyperparameter tuning budget.
- Multi-seed aggregation + Wilcoxon report table tự động.
- Coverage/ARP integration vào bảng kết quả chính (module metric đã có).
- H&M S/M/L subset builder.
- Plot V2 mới.

Các mục này nên làm sau khi DataCo core experiment chạy ổn định.
