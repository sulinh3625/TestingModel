# DACNTT — NeuMF Recommendation Project V2

**Đề tài:** Xây dựng mô hình khuyến nghị lai kết hợp nhân tử hoá ma trận và mạng lưới thần kinh sâu.

Phiên bản V2 refactor pipeline để tránh data leakage, chuẩn hóa đánh giá full-ranking. Dự án thực nghiệm trên **hai bộ dữ liệu độc lập**: DataCo Supply Chain (chính, catalog nhỏ/dày đặc) và H&M Personalized Fashion Recommendations (đối chứng, catalog lớn/cực thưa). Chi tiết phương pháp luận và số liệu đầy đủ nằm trong `Report DACNTT/main.pdf` (Chương 3, 4).

## Cây thư mục chính thức

```text
neumf_project_v2/
├── configs/
│   ├── dataco.yaml       # Config DataCo (dùng cho run.py / scripts/run_all.py)
│   ├── hm.yaml           # Config H&M quy mô ĐẦY ĐỦ (đọc data/processed/hm/*.parquet)
│   └── hm_subset.yaml    # Config H&M lát cắt thực nghiệm (đọc raw CSV trực tiếp) — dùng để có số liệu trong báo cáo
├── data/
│   ├── raw/{dataco,hm}/          # Đặt file CSV gốc tại đây (xem README riêng từng thư mục)
│   ├── processed/hm/             # Cache Parquet của H&M, sinh bởi scripts/00_prepare_hm_cache.py
│   └── splits/{dataco,hm}/
├── docs/                          # METHODOLOGY_V2, IMPLEMENTATION_STATUS (đang dùng)
│   └── archive/                   # MIGRATION_V1_TO_V2, legacy_v1_pham_vi_du_an (tài liệu lịch sử V1)
├── src/
│   ├── config.py
│   ├── data_pipeline/            # Xử lý dữ liệu: adapters, preprocessing, k-core, splitting, negative sampling, dataset, audit
│   │                              # (đặt tên khác với data/ ở trên — data/ chỉ chứa FILE dữ liệu thô/đã xử lý, không phải code)
│   ├── models/                   # GMF / MLP / NeuMF / EarlyFusion
│   ├── baselines/                # ItemKNN, BPR-MF
│   ├── training/                 # Vòng lặp huấn luyện
│   ├── evaluation/                # Metrics, full/sampled ranking, long-tail, cold-start, beyond-accuracy, kiểm định thống kê
│   └── utils/                    # seed, io
├── scripts/
│   ├── 00_prepare_hm_cache.py    # Tiền xử lý 1 lần: CSV H&M gốc (31,8 triệu dòng) -> Parquet gọn nhẹ
│   ├── 01_data_audit.py          # Audit k-core, mật độ, kiểm tra Disjoint — chạy trước khi train
│   ├── 02_preprocess.py          # Sinh splits (dùng khi cần splits độc lập với run_all)
│   ├── 03_run_experiment.py      # Huấn luyện toàn bộ mô hình (GMF/MLP/EarlyFusion/LightGCN/SASRec/NeuMF) + baselines (kể cả CategoryPopularity)
│   ├── 04_multi_seed.py          # Lặp lại 03 trên nhiều seed (mặc định 42, 2024, 2025, 2026, 3407)
│   ├── 05_evaluate.py            # Đánh giá + xuất 11 biểu đồ + bảng kết quả từ 1 run_tag
│   ├── 06_aggregate_seeds.py     # Tổng hợp mean±std và kiểm định Wilcoxon từ nhiều seed
│   ├── 07_compare_datasets.py    # Biểu đồ so sánh DataCo vs H&M (dùng trong báo cáo, Hình 4.1)
│   ├── 08_hyperparam_sweep.py    # Sweep embedding_dim / số lớp MLP / negative_ratio (mục 4.5/5.3)
│   ├── 09_full_report.py         # Pipeline 1 lần: audit+sweep+train+multi-seed+weighted+evaluate (mục 5a)
│   └── run_all.py                # Chạy trọn gói: 03 rồi 05 cho một config, một lệnh duy nhất
├── run.py                         # CLI tổng hợp: all/audit/train/evaluate/multi-seed/aggregate/compare/sweep/report-full/demo (python run.py -h)
├── demo/                          # Giao diện thực nghiệm (xem mục "Chạy demo" bên dưới)
│   ├── backend/main.py           # FastAPI — suy diễn trên checkpoint đã huấn luyện
│   └── frontend/index.html       # Giao diện web tĩnh
├── Dockerfile, docker-compose.yml, .dockerignore   # Đóng gói demo (mục "Chạy demo bằng Docker")
├── outputs/{data_audit,checkpoints,experiments,tables,figures}/
│   └── archive/legacy_v1/         # Kết quả bản V1 cũ, giữ lại để đối chiếu
├── notebooks/
│   └── data_pipeline_walkthrough.ipynb   # Xem code từng bước pipeline + chạy trên Colab GPU lưu kết quả vào Drive (mục 7c)
├── tests/
├── requirements.txt
└── pytest.ini
```

## 1. Cài đặt

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu cần CUDA, cài PyTorch theo đúng bản CUDA của máy trước khi cài phần còn lại. Thực nghiệm gốc của báo cáo chạy hoàn toàn trên CPU (không có GPU), xem cấu hình máy ở mục 4.1 báo cáo.

## 2. Đặt dữ liệu thô

```text
data/raw/dataco/DataCoSupplyChainDataset.csv
data/raw/hm/transactions_train.csv   (+ articles.csv, customers.csv — xem data/raw/hm/README.md)
```

## 3. Huấn luyện DataCo (một lệnh)

```bash
python run.py all --dataset dataco --run-tag dataco_<tag_của_bạn>
```

`run.py` ở gốc dự án gom tất cả lệnh trong `scripts/` lại một chỗ — `all` tự động: audit → huấn luyện GMF → MLP → EarlyFusion → LightGCN → SASRec → NeuMF-Scratch → NeuMF-Pretrained → baselines (ItemKNN, BPR-MF, CategoryPopularity) → gọi `05_evaluate.py` xuất bảng (`outputs/tables/<run_tag>/`) và 12 biểu đồ (`outputs/figures/<run_tag>/`). Trên DataCo mất khoảng **~60-90 phút** trên CPU phổ thông — LightGCN chiếm phần lớn thời gian này (lan truyền lại toàn đồ thị mỗi batch huấn luyện, ~50 phút riêng mô hình này, xem mục 4.3.3 báo cáo); tắt qua `training.train_lightgcn: false` trong config nếu cần chạy nhanh.

`--dataset` nhận `dataco` / `hm` / `hm_subset`. Xem toàn bộ lệnh sẵn có:

```bash
python run.py -h              # danh sách lệnh: all, audit, preprocess, train, evaluate,
                               # multi-seed, aggregate, compare, prepare-hm-cache, demo
python run.py <lệnh> -h       # chi tiết tham số từng lệnh
```

Muốn chạy tay từng bước hoặc gọi thẳng script gốc thì dùng `scripts/01_data_audit.py`, `scripts/run_all.py`, ... — `run.py` chỉ gọi lại các script này, không có logic riêng.

## 4. Huấn luyện H&M

H&M gốc có **31,8 triệu dòng** — quá nặng để đọc lại mỗi lần chạy. Quy trình 2 bước:

```bash
# Bước 1 (chạy 1 lần duy nhất, ~2 phút): nén CSV gốc thành cache Parquet nhẹ
python scripts/00_prepare_hm_cache.py

# Bước 2: audit + train trên lát cắt thực nghiệm (nhanh, ~1-2 phút — đúng cấu hình đã dùng trong báo cáo)
python run.py all --dataset hm_subset --run-tag hm_<tag_của_bạn>
```

> `configs/hm_subset.yaml` đọc trực tiếp `transactions_train.csv` (100.000 dòng đầu) — đây là quy mô thực tế dùng để tạo số liệu trong báo cáo (mục 3.3.3).
>
> `configs/hm.yaml` trỏ tới cache Parquet ở quy mô **toàn bộ** (889.062 người dùng, 90.690 sản phẩm sau lọc). Đã cấu hình sẵn để khả thi ở quy mô này (khác `hm_subset.yaml`): `evaluation.primary: sampled` (thay Full Ranking — tránh chi phí ~8×10¹⁰ cặp/mô hình), `training.fast_negative_sampling: true` (lấy mẫu âm vector hoá NumPy thay vì vòng lặp Python — bắt buộc ở quy mô ~26,2 triệu tương tác/epoch), và tắt sẵn `LightGCN`/`BPR-MF` (cả hai đều có chi phí per-batch/per-sample chưa tối ưu cho quy mô này — xem `training.train_lightgcn`, `baselines.enabled` trong file). **Đã kiểm thử đúng logic ở quy mô nhỏ (subset), nhưng CHƯA chạy thực sự ở quy mô đầy đủ** — dự kiến vẫn mất nhiều giờ trên CPU phổ thông (huấn luyện GMF/MLP/NeuMF/EarlyFusion/SASRec trên ~26,2 triệu tương tác/epoch), khuyến nghị chạy trên GPU (mục 7c):
>
> ```bash
> python run.py train --dataset hm --run-tag hm_full_<tag_của_bạn>
> ```

## 5. Multi-seed & kiểm định thống kê (tuỳ chọn)

```bash
python run.py multi-seed --dataset dataco --seeds 42 2024 2025 2026 3407
python run.py aggregate --dataset dataco --seeds 42 2024 2025 2026 3407
```

Mỗi seed chạy lại toàn bộ bước 3 (~10-15 phút/seed trên DataCo) — hạ tầng đã sẵn sàng nhưng chưa được chạy đủ 5 seed trong báo cáo hiện tại (ghi rõ là hạn chế ở mục 5.2).

## 5a. Pipeline đầy đủ 1 lần — số liệu chính thức cho báo cáo (khuyến nghị)

```bash
python run.py report-full --dataset dataco
python run.py report-full --dataset hm_subset
```

Chạy nối tiếp, tự động, đúng 1 trình tự khoa học duy nhất — **không cherry-pick** số liệu từ nhiều lần chạy rời rạc: audit → sweep siêu tham số đầy đủ (chọn theo validation) → train "final" (mọi model + baseline, Precision/Recall, long-tail, cold-start, beyond-accuracy) → multi-seed (5 seed mặc định) + Wilcoxon trên ĐÚNG cấu hình đã chọn → biến thể weighted-feedback + so sánh trực tiếp với binary → xuất bảng/biểu đồ chính thức. Chi tiết từng bước và vị trí file kết quả xem docstring đầu `scripts/09_full_report.py`.

Đã smoke-test toàn chuỗi (subset nhỏ) chạy thành công không lỗi — bản đầy đủ trên toàn bộ dữ liệu nên chạy trên GPU (xem mục "Chạy trên Colab" bên dưới), vì bao gồm sweep 15 trial + train 7 model/dataset + 5 lần multi-seed + 1 lần weighted-feedback, tổng cộng tốn nhiều thời gian hơn hẳn 1 lần `run.py all` thông thường.

Có thể resume nếu bị ngắt giữa chừng (mất kết nối Colab):

```bash
python run.py report-full --dataset dataco --skip-audit --skip-sweep
```

## 5b. Sweep siêu tham số (tuỳ chọn, mục 4.5/5.3 báo cáo)

```bash
python run.py sweep --dataset dataco
python run.py sweep --dataset hm_subset
```

Quét độc lập từng siêu tham số quanh cấu hình mặc định trong `configs/*.yaml`: `embedding_dim` ∈ {8,16,32,64,128}, số lớp MLP ∈ {2..6} (tower dạng halving bắt đầu từ `2×embedding_dim`, bắt buộc theo kiến trúc NeuMF), `negative_ratio` ∈ {1,2,4,8,16} — mỗi trial chỉ huấn luyện NeuMF-Scratch (bỏ qua pretrain/baseline) để giữ chi phí hợp lý, pipeline dữ liệu chỉ dựng 1 lần dùng chung. Cấu hình tốt nhất được **chọn theo NDCG@10 trên validation** (không dùng test để tránh rò rỉ dữ liệu khi tinh chỉnh siêu tham số) — cột test trong bảng kết quả chỉ để minh hoạ xu hướng.

Kết quả: `outputs/tables/sweep_<dataset>/` (3 bảng theo từng trục + `sweep_all.csv` + `best_config.json`) và `outputs/figures/sweep_<dataset>/hyperparam_sensitivity.png`.

Sau khi có cấu hình tốt nhất, cập nhật `model.embedding_dim` / `model.mlp_layers` / `training.negative_ratio` trong config rồi chạy lại `python run.py all --dataset <tên>` để lấy **số liệu test chính thức** (đầy đủ pretrain GMF/MLP + mọi baseline) cho báo cáo — script sweep không thay thế `03_run_experiment.py`.

## 6. So sánh trực quan DataCo vs H&M

```bash
python scripts/07_compare_datasets.py
```

Sinh `outputs/figures/comparison/dataco_vs_hm_comparison.png` (dùng trong báo cáo, Hình 4.1) từ `results_primary.csv` của hai run tag `dataco_20260918_verify` / `hm_20260918_verify` — sửa hằng số `dataco_dir`/`hm_dir` trong script nếu bạn dùng run tag khác.

## 7. Chạy demo (giao diện thực nghiệm)

Demo nạp checkpoint đã huấn luyện ở bước 3/4 để suy diễn (inference-only, không huấn luyện lại), thiết kế theo mục 3.7 báo cáo.

```bash
# fastapi/uvicorn/pyarrow đã có trong requirements.txt
python -m uvicorn demo.backend.main:app --port 8000 --host 127.0.0.1
```

Mở trình duyệt tại **http://localhost:8000**. Lần đầu chọn mỗi bộ dữ liệu sẽ mất vài giây (server tái tạo pipeline tiền xử lý để suy ra đúng ánh xạ ID ↔ checkpoint); các lần sau tức thời vì đã cache trong bộ nhớ tiến trình.

Yêu cầu trước khi chạy: đã train xong ít nhất 1 lần cho mỗi bộ dữ liệu (mục 3-4). **Demo tự động dùng run_tag mới nhất** khớp tiền tố `dataco_`/`hm_` trong `outputs/experiments/` (ưu tiên run đã hoàn chỉnh, có checkpoint đầy đủ) — train ra run_tag mới rồi gọi `POST /api/reload/{dataset}` (hoặc khởi động lại server) để demo dùng ngay dữ liệu mới nhất, không cần sửa code. Chi tiết kiến trúc/API xem `demo/README.md` và mục 3.7 báo cáo.

## 7b. Chạy demo bằng Docker

Đóng gói môi trường (không đóng gói dữ liệu/checkpoint — quá lớn và thay đổi liên tục) để chạy được trên máy khác không lỗi môi trường:

```bash
docker compose up --build
```

Mở trình duyệt tại **http://localhost:8000**, giống hệt chạy không-Docker. Yêu cầu trước khi chạy: vẫn cần `data/raw/{dataco,hm}/...` và ít nhất 1 lần huấn luyện xong cho mỗi bộ dữ liệu ở máy host (mục 3-4) — `docker-compose.yml` mount `data/` (read-only) và `outputs/` từ host vào container, container chỉ suy diễn trên checkpoint đã có sẵn, không tự huấn luyện.

Chạy tay không qua compose:

```bash
docker build -t neumf-demo .
docker run -p 8000:8000 -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/outputs:/app/outputs" neumf-demo
```

## 7c. Chạy trên Colab (GPU) — lưu kết quả trực tiếp vào Google Drive

Dùng sẵn **`notebooks/data_pipeline_walkthrough.ipynb`** — cùng 1 notebook dùng để (a) đọc/xem code pipeline có giải thích, VÀ (b) chạy thực nghiệm trên Colab GPU lưu kết quả vào Drive. Upload file này vào Colab (hoặc mở qua `File > Open notebook > GitHub` sau khi đã push lên repo) — notebook tự nhận biết đang chạy trên Colab hay local (biến `IN_COLAB`) và tự bỏ qua phần thiết lập Colab nếu chạy local. Phần đầu notebook (Phần 0) tự động:

1. Mount Google Drive, tạo thư mục `DACNTT/` trên Drive để chứa dữ liệu thô + toàn bộ kết quả.
2. `git clone`/`git pull` code mới nhất từ GitHub (**bắt buộc `git push` từ máy bạn trước khi chạy** — notebook lấy đúng bản đã push, không tự đồng bộ code local).
3. Copy dữ liệu thô từ Drive vào đĩa Colab (đọc/ghi nhanh hơn Drive mount trực tiếp), tái sử dụng cache H&M nếu đã upload sẵn.
4. **Symlink `outputs/` thẳng vào Drive** — checkpoint/bảng/biểu đồ ghi ra ngay lập tức nằm trên Drive, không cần đợi chạy xong mới copy. Nếu Colab ngắt kết nối giữa chừng, kết quả đã sinh ra đến lúc đó vẫn còn nguyên trên Drive.

Phần cuối notebook (Phần 3) có các ô chạy thực nghiệm độc lập: `report-full --dataset dataco`, `report-full --dataset hm_subset`, và **`train --dataset hm`** (H&M quy mô đầy đủ 889.062 user — phần chưa từng chạy được trên CPU, xem mục 4 và `configs/hm.yaml`). Phần giữa (Phần 1-2) là nội dung walkthrough gốc — xem code từng module + chạy sống pipeline trên DataCo, dùng để hiểu/đọc, không bắt buộc phải chạy trước khi tới Phần 3.

Lần đầu dùng cần upload thủ công (1 lần duy nhất, các phiên Colab sau tái sử dụng): `DataCoSupplyChainDataset.csv` và 3 file H&M vào đúng thư mục Drive mà notebook tạo sẵn (chi tiết trong chính notebook, Phần 0.3).

## 8. Test

```bash
pytest -q
```
