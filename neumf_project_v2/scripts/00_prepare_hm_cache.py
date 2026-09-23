"""00_prepare_hm_cache.py — Tiền xử lý một lần bộ H&M transactions_train.csv (gốc)
thành file Parquet gọn nhẹ, chỉ giữ các thuộc tính thực sự cần cho pipeline
(user_raw, item_raw, timestamp, value_raw, source_order).

Vấn đề gốc: transactions_train.csv có ~31.8 triệu dòng, cột customer_id/article_id
là chuỗi (hash 64 ký tự / mã 10 chữ số) khiến việc đọc trực tiếp bằng pandas tốn
hàng chục giây và nhiều GB RAM mỗi lần chạy pipeline, không khả thi để chạy lặp lại
nhiều lần trên CPU.

Giải pháp: đọc CSV theo từng chunk, mã hoá customer_id/article_id thành số nguyên
int32 (factorize toàn cục, ổn định theo thứ tự xuất hiện), hạ kiểu dữ liệu các cột
số (float32/int8) và ghi thẳng ra Parquet theo dạng streaming (không giữ toàn bộ
31.8 triệu dòng trong RAM cùng lúc). Kết quả: file .parquet nhỏ hơn nhiều lần so với
CSV gốc và tải lại chỉ mất vài giây cho các lần chạy pipeline sau.

Output:
  data/processed/hm/transactions_clean.parquet   (dữ liệu đã mã hoá, dùng cho adapter)
  data/processed/hm/user_id_map.parquet          (customer_id gốc -> user_raw int32)
  data/processed/hm/item_id_map.parquet          (article_id gốc -> item_raw int32)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH_DEFAULT = PROJECT_ROOT / "data" / "raw" / "hm" / "transactions_train.csv"
OUT_DIR_DEFAULT = PROJECT_ROOT / "data" / "processed" / "hm"


def build_codes(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Ánh xạ chuỗi -> mã int32 toàn cục, ổn định qua nhiều chunk."""
    codes = series.map(mapping)
    missing_mask = codes.isna()
    if missing_mask.any():
        start = len(mapping)
        missing_vals = pd.unique(series[missing_mask])
        for offset, val in enumerate(missing_vals):
            mapping[val] = start + offset
        codes = series.map(mapping)
    return codes.astype("int32")


def main():
    ap = argparse.ArgumentParser(description="Tiền xử lý H&M transactions_train.csv -> Parquet gọn nhẹ")
    ap.add_argument("--raw-path", default=str(RAW_PATH_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    raw_path = Path(args.raw_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "transactions_clean.parquet"
    tmp_file = out_dir / "transactions_clean.parquet.tmp"

    print(f"Đọc raw: {raw_path}")
    t0 = time.time()

    user_map: dict[str, int] = {}
    item_map: dict[str, int] = {}

    schema = pa.schema([
        ("user_raw", pa.int32()),
        ("item_raw", pa.int32()),
        ("timestamp", pa.timestamp("ns")),
        ("value_raw", pa.float32()),
        ("source_order", pa.int64()),
    ])

    reader = pd.read_csv(
        raw_path,
        usecols=["t_dat", "customer_id", "article_id", "price"],
        dtype={"customer_id": "string", "article_id": "string"},
        chunksize=args.chunksize,
    )

    n_rows = 0
    writer = pq.ParquetWriter(tmp_file, schema, compression="snappy")
    try:
        for chunk_idx, chunk in enumerate(reader):
            chunk = chunk.dropna(subset=["customer_id", "article_id", "t_dat"])
            user_codes = build_codes(chunk["customer_id"], user_map)
            item_codes = build_codes(chunk["article_id"], item_map)
            timestamp = pd.to_datetime(chunk["t_dat"], errors="coerce")
            value_raw = pd.to_numeric(chunk["price"], errors="coerce").fillna(0.0).astype("float32")
            source_order = pd.RangeIndex(n_rows, n_rows + len(chunk)).to_numpy(dtype="int64")

            table = pa.table({
                "user_raw": pa.array(user_codes, type=pa.int32()),
                "item_raw": pa.array(item_codes, type=pa.int32()),
                "timestamp": pa.array(timestamp, type=pa.timestamp("ns")),
                "value_raw": pa.array(value_raw, type=pa.float32()),
                "source_order": pa.array(source_order, type=pa.int64()),
            }, schema=schema)
            writer.write_table(table)

            n_rows += len(chunk)
            print(f"  Chunk {chunk_idx + 1}: +{len(chunk):,} dòng (tổng {n_rows:,}) "
                  f"| users={len(user_map):,} items={len(item_map):,} "
                  f"| {time.time() - t0:.1f}s")
    finally:
        writer.close()

    tmp_file.replace(out_file)

    # Lưu bảng ánh xạ ID gốc <-> mã số để tra cứu/giải thích khi cần (vd. Cold-start, demo).
    user_map_df = pd.DataFrame({"customer_id": list(user_map.keys()), "user_raw": list(user_map.values())})
    item_map_df = pd.DataFrame({"article_id": list(item_map.keys()), "item_raw": list(item_map.values())})
    user_map_df.to_parquet(out_dir / "user_id_map.parquet", index=False)
    item_map_df.to_parquet(out_dir / "item_id_map.parquet", index=False)

    elapsed = time.time() - t0
    raw_size_mb = raw_path.stat().st_size / 1e6
    out_size_mb = out_file.stat().st_size / 1e6
    print("\nHoàn tất tiền xử lý H&M.")
    print(f"  Tổng dòng hợp lệ : {n_rows:,}")
    print(f"  Số khách hàng    : {len(user_map):,}")
    print(f"  Số sản phẩm      : {len(item_map):,}")
    print(f"  Thời gian xử lý  : {elapsed:.1f}s")
    print(f"  Kích thước gốc   : {raw_size_mb:,.1f} MB  ({raw_path})")
    print(f"  Kích thước cache : {out_size_mb:,.1f} MB  ({out_file})")


if __name__ == "__main__":
    main()
