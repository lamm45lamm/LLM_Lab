# Lab — Data Pipeline cho LLM (Tokenization → Embedding)

Chuyển thể từ `ch02.ipynb` và `dataloader.ipynb` — "Build a Large Language
Model From Scratch" (Sebastian Raschka).
Nguồn tham khảo gốc: <https://github.com/rasbt/LLMs-from-scratch>

Xem hướng dẫn chi tiết từng bước trong [LAB_GUIDE.md](./LAB_GUIDE.md) —
mỗi block có checkpoint để bạn tự biết mình đang đúng tiến độ.

## Mục Tiêu

Sau buổi lab này, bạn sẽ tự cài đặt lại được toàn bộ pipeline xử lý dữ liệu
văn bản đầu vào cho một LLM kiểu GPT:

- Xây dựng vocab và một tokenizer đơn giản, tự viết (`SimpleTokenizerV1`)
- Xử lý từ chưa biết (unknown word) bằng special token (`SimpleTokenizerV2`)
- Dùng Byte Pair Encoding (BPE) qua thư viện `tiktoken` (giống GPT-2)
- Cài đặt `GPTDatasetV1` — cắt văn bản thành cặp (input, target) kiểu
  sliding window để huấn luyện next-token prediction
- Cài đặt `create_dataloader_v1` — nạp dữ liệu theo batch bằng PyTorch
  `DataLoader`
- Ghép Token Embedding + Positional Embedding thành Input Embedding hoàn
  chỉnh đưa vào mô hình

## Cài Đặt

### Yêu cầu

- Python 3.10+
- Internet ở lần chạy đầu tiên (để `requests` tải văn bản mẫu và `tiktoken`
  tải file vocab GPT-2 — sau đó được cache lại trên máy, không cần internet
  nữa)

### Tạo môi trường ảo & cài thư viện

**macOS / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cấu Trúc Thư Mục

```
.
├── README.md            # File này — tổng quan, cài đặt, chấm điểm
├── LAB_GUIDE.md          # Hướng dẫn chi tiết từng bước + checkpoint
├── exercises.md          # Phiếu bài tập & phản ánh (6 câu)
├── template.py           # Nơi bạn viết code — điền các TODO
├── grade.py              # Chấm điểm tự động
├── requirements.txt
└── tests/
    ├── _target.py         # (nội bộ) chọn solution/ hoặc template.py để test
    ├── test_part1.py      # Checkpoint 1 — Vocab & Tokenizer
    ├── test_part2.py      # Checkpoint 2 — BPE & sliding window
    ├── test_part3.py      # Checkpoint 3 — GPTDatasetV1 & DataLoader
    └── test_part4.py      # Checkpoint 4 — Embedding & pipeline hoàn chỉnh
```

## Checkpoint

| Block | Nội dung | Checkpoint |
|---|---|---|
| Block 1 | Vocab & Tokenizer (`SimpleTokenizerV1`/`V2`) — Phần 1.1–2.3 | `pytest tests/test_part1.py -v` |
| Block 2 | BPE (`tiktoken`) & trực giác sliding window — Phần 3.1, 4.1 | `pytest tests/test_part2.py -v` |
| Block 3 | `GPTDatasetV1` & `create_dataloader_v1` — Phần 5.1–6.1 | `pytest tests/test_part3.py -v` |
| Block 4 | Embedding & pipeline hoàn chỉnh — Phần 7.1, 8.1 | `pytest tests/test_part4.py -v` |

Chi tiết từng bước của mỗi block: xem [LAB_GUIDE.md](./LAB_GUIDE.md).

## Chạy Kiểm Thử

```
# Từng checkpoint
pytest tests/test_part1.py -v

# Toàn bộ
pytest tests/ -v
```

Phần lớn test dùng dữ liệu/tokenizer giả lập (`DummyTokenizer`) để kết quả
xác định (deterministic) và không cần internet — trừ 2 test ở
`test_part2.py` gọi `tiktoken` thật (`test_3_1_*`), lần đầu chạy cần
internet để tải vocab GPT-2.

## Chấm Điểm Tự Động (100 điểm)

```
python grade.py
```

| Tiêu chí | Cách chấm | Điểm |
|---|---|---|
| Block 1 — Vocab & Tokenizer | `tests/test_part1.py` | 20 |
| Block 2 — BPE & sliding window | `tests/test_part2.py` | 15 |
| Block 3 — GPTDatasetV1 & DataLoader | `tests/test_part3.py` | 30 |
| Block 4 — Embedding & pipeline hoàn chỉnh | `tests/test_part4.py` | 20 |
| `exercises.md` — 6 câu phản ánh | Đếm số câu đã trả lời | 15 |
| **Tổng** | | **100** |

Điểm mỗi block tỷ lệ với số test pass, nên **làm được đến đâu có điểm đến
đó**. Điểm `exercises.md` là điểm hoàn thành; chất lượng nội dung giảng
viên có thể điều chỉnh sau.

## Hướng Dẫn Nộp Bài

Điều chỉnh phần này theo quy trình nộp bài thực tế của lớp bạn (nộp file
zip qua LMS, hoặc nộp link GitHub như mẫu bên dưới).

**Nếu nộp qua GitHub:**
```
# 1. Tạo folder solution và copy bài làm
mkdir -p solution
cp template.py solution/solution.py
cp exercises.md solution/exercises.md

# 2. Chấm thử lần cuối (grade.py sẽ ưu tiên chấm folder solution)
python grade.py

# 3. Tạo repo MỚI trên github.com, tên theo quy ước <MãMônHọc>-<MSSV>-<HoVaTen>
git init                          # bỏ qua nếu folder đã là git repo
git add .
git commit -m "Nộp bài lab Data Pipeline"
git branch -M main
git remote add origin https://github.com/<tài khoản của bạn>/<tên-repo>.git
git push -u origin main

# 4. Nộp LINK repo vào hệ thống nộp bài của lớp
```

⚠️ **Kiểm tra sau khi push:** mở repo trên GitHub, xác nhận `.venv/` và các
file cache không bị commit (đã có `.gitignore` chặn — tự kiểm tra lại).

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `pytest tests/ -v` — các checkpoint đều pass
- [ ] `python grade.py` — xem điểm
- [ ] `solution/exercises.md` — cả 6 câu đã trả lời
- [ ] `solution/solution.py` — bản code cuối cùng
- [ ] Đã nộp đúng hạn theo quy định của lớp
