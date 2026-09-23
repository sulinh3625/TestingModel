# Dockerfile — Demo NeuMF (suy diễn, inference-only)
#
# Ảnh chỉ đóng gói mã nguồn + dependency Python. Dữ liệu thô (data/) và
# checkpoint đã huấn luyện (outputs/) KHÔNG được copy vào ảnh — quá lớn để
# đóng gói hợp lý (H&M transactions_train.csv ~3,5GB) và thường xuyên đổi
# mới sau mỗi lần huấn luyện. Hai thư mục này PHẢI được mount qua volume khi
# chạy container (xem docker-compose.yml, hoặc README mục "Chạy demo bằng
# Docker") — đúng yêu cầu "đóng gói môi trường, không đóng gói dữ liệu".
FROM python:3.11-slim

WORKDIR /app

# Cài torch bản CPU-only trước, tránh pip kéo wheel CUDA mặc định (>2GB,
# không cần thiết cho suy diễn inference-only trên máy chấm không có GPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY demo/ demo/
COPY run.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "demo.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
