# Demo NeuMF — hướng dẫn chạy

Giao diện thực nghiệm suy diễn (inference-only), đúng thiết kế 3 tầng ở mục 3.7
báo cáo (`Report DACNTT/content/C3.tex`). Không huấn luyện lại mô hình — chỉ
nạp checkpoint đã có trong `outputs/checkpoints/`.

## Chạy

Từ thư mục `neumf_project_v2/`:

```bash
pip install -r requirements.txt
python -m uvicorn demo.backend.main:app --port 8000 --host 127.0.0.1
```

Mở trình duyệt tại **http://localhost:8000**.

Lần đầu chọn mỗi bộ dữ liệu sẽ mất vài giây (server tái tạo pipeline tiền xử
lý để suy ra đúng ánh xạ ID -> checkpoint), các lần sau tức thời vì đã cache
trong bộ nhớ tiến trình.

## Yêu cầu trước khi chạy

Cần đã chạy `scripts/run_all.py` xong ít nhất một lần cho mỗi bộ dữ liệu (xem README chính, mục "Huấn luyện").

**Demo tự động dùng run_tag MỚI NHẤT** khớp tiền tố `dataco_`/`hm_` trong `outputs/experiments/` — chọn theo run nào có `results.json` mới nhất VÀ có checkpoint `.pt` đầy đủ (tự bỏ qua run bị ngắt giữa chừng, thiếu file). Không cần sửa gì trong code khi train ra run_tag mới.

Server chỉ dò lại run mới nhất khi:
- Khởi động lại server, hoặc
- Gọi `POST /api/reload/{dataset}` (không cần khởi động lại) — vd. `curl -X POST http://localhost:8000/api/reload/dataco` sau khi vừa train xong một lần chạy mới.

Muốn đổi sang một prefix run_tag khác hẳn (không phải `dataco_`/`hm_`), sửa `DATASETS` trong `demo/backend/main.py`.

## Cấu trúc

- `backend/main.py` — FastAPI, expose các endpoint bên dưới (xem mục 3.7.2 báo cáo) + phục vụ luôn `frontend/` tại `/`.
- `frontend/index.html` — trang tĩnh (Inter + Chart.js qua CDN), gọi API bằng `fetch`, không cần build step.

### API endpoints

| Endpoint | Chức năng |
|---|---|
| `GET /api/health` | Kiểm tra server + danh sách dataset đã nạp |
| `GET /api/datasets` | Danh sách bộ dữ liệu khả dụng |
| `GET /api/models/{dataset}` | Danh sách model đã có checkpoint + số liệu Full Ranking |
| `GET /api/users/{dataset}?limit=` | Mẫu ID khách hàng hợp lệ |
| `GET /api/history/{dataset}/{user_id}?limit=` | Lịch sử mua hàng của khách hàng |
| `GET /api/recommend/{dataset}/{model}/{user_id}?k=` | Top-K gợi ý (kèm nhãn Head/Long-tail) |
| `GET /api/similar-items/{dataset}/{item_idx}?k=` | Sản phẩm tương tự theo cosine similarity trên embedding GMF |
| `GET /api/beyond/{dataset}` | Coverage/Novelty/Head Rec Rate theo model (`beyond_accuracy.csv`) |
| `GET /api/metrics/{dataset}` | Toàn bộ `primary_results.csv` dạng JSON, dùng vẽ biểu đồ |
| `POST /api/reload/{dataset}` | Xoá cache, dò lại run_tag mới nhất trên đĩa ngay lập tức |

## Dừng server

Nhấn `Ctrl+C` trong terminal đang chạy uvicorn, hoặc trên Windows tìm PID đang
lắng nghe cổng 8000 (`netstat -ano | findstr :8000`) rồi `taskkill /PID <pid> /F`.
