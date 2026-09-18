# LAB GUIDE — Data Pipeline cho LLM

Hướng dẫn chi tiết từng bước. Mở song song `template.py` để vừa đọc vừa
làm. Ở mỗi bước, chạy checkpoint tương ứng để biết mình đã làm đúng chưa
trước khi qua bước tiếp theo.

---

## Trước khi bắt đầu

```
pip install -r requirements.txt
pytest tests/ -v
```

Chạy `pytest tests/ -v` ngay từ đầu — **tất cả test FAIL là bình thường**,
vì bạn chưa viết code gì cả (mọi hàm đang `raise NotImplementedError`). Đây
chính là điểm xuất phát của bạn.

---

## Block 1 — Vocab & Tokenizer (`template.py`, Phần 1.1 → 2.3)

### Ý tưởng

Một LLM không đọc trực tiếp chữ cái — nó cần văn bản được quy về dãy số
nguyên (token id). `tokenize_text()` (đã cho sẵn) tách văn bản thành các
"từ" và dấu câu riêng biệt; bạn sẽ dùng danh sách từ đó để xây một **vocab**
(bảng ánh xạ từ ↔ số) và một **tokenizer** biết encode/decode qua lại.

### Việc cần làm

1. **Phần 1.1 — `build_vocab(tokens)`**: lấy tập từ duy nhất, sắp xếp
   alphabet, đánh số từ 0.
   - Gợi ý: `sorted(set(tokens))` rồi `enumerate(...)`.
2. **Phần 1.2 — `SimpleTokenizerV1.encode`**: tách text bằng
   `tokenize_text()`, tra từng từ sang id qua `self.str_to_int`.
3. **Phần 1.3 — `SimpleTokenizerV1.decode`**: tra ngược id → từ, nối bằng
   dấu cách, rồi dùng `re.sub` xoá khoảng trắng thừa trước dấu câu.
4. **Phần 2.1 — `build_vocab_with_special_tokens(tokens)`**: giống 1.1
   nhưng thêm `"<|endoftext|>"` và `"<|unk|>"` vào cuối trước khi đánh số.
5. **Phần 2.2 — `SimpleTokenizerV2.encode`**: giống 1.2, nhưng từ nào
   không có trong vocab thì thay bằng `"<|unk|>"` trước khi tra id.
6. **Phần 2.3 — `SimpleTokenizerV2.decode`**: giống hệt 1.3.

### Vì sao cần Tokenizer V2?

`SimpleTokenizerV1` sẽ crash (`KeyError`) ngay khi gặp một từ nằm ngoài
vocab đã xây — điều này chắc chắn xảy ra khi bạn đưa vào một câu mới bất kỳ.
`SimpleTokenizerV2` xử lý việc này bằng cách "gộp" mọi từ lạ vào chung một
token `<|unk|>`, đổi lại là bạn mất thông tin về từ đó là từ gì.

### ✅ Checkpoint 1
```
pytest tests/test_part1.py -v
```
Kỳ vọng: 6/6 test pass.

---

## Block 2 — BPE & Trực Giác Sliding Window (Phần 3.1, 4.1)

### Ý tưởng

`SimpleTokenizerV2` mất thông tin khi gặp từ lạ. **Byte Pair Encoding
(BPE)** — chính là tokenizer thật của GPT-2 — giải quyết việc này bằng cách
tách một từ chưa từng thấy thành các đơn vị con (subword) nhỏ hơn thay vì
vứt bỏ thông tin. Thư viện `tiktoken` cho bạn dùng lại đúng bộ BPE này.

Phần thứ hai của block giới thiệu trực giác quan trọng nhất của việc huấn
luyện LLM: với một dãy token, mô hình học cách nhìn `i` token đầu để đoán
token thứ `i+1`.

### Việc cần làm

1. **Phần 3.1 — `bpe_roundtrip_demo(text)`**: dùng
   `tiktoken.get_encoding("gpt2")` để encode rồi decode lại `text`. Nhớ
   truyền `allowed_special={"<|endoftext|>"}` khi encode.
2. **Phần 4.1 — `get_next_token_pairs(token_ids, context_size)`**: với
   `i` chạy từ 1 đến `context_size`, tạo cặp
   `(token_ids[:i], token_ids[i])`.

### ✅ Checkpoint 2
```
pytest tests/test_part2.py -v
```
Kỳ vọng: 4/4 test pass (2 test đầu cần internet ở lần chạy đầu để tải vocab
GPT-2 qua `tiktoken`).

---

## Block 3 — `GPTDatasetV1` & `DataLoader` (Phần 5.1 → 6.1)

**Đây là phần quan trọng nhất của bài lab** — cũng là nội dung mà
`dataloader.ipynb` gốc tóm tắt lại làm "cốt lõi" của cả chương.

### Ý tưởng

Để huấn luyện next-token prediction trên một cuốn sách dài, ta không thể
đưa cả cuốn sách vào mô hình một lần. Ta cắt văn bản (đã mã hoá thành token
id) thành nhiều đoạn ngắn có độ dài cố định `max_length`, bằng kỹ thuật
**sliding window**: cửa sổ trượt qua token id với bước nhảy `stride`. Mỗi
đoạn `input` đi kèm một đoạn `target` — chính là `input` dịch phải 1 vị trí
— vì nhiệm vụ của mô hình là đoán token kế tiếp tại mọi vị trí trong đoạn.

`stride` càng nhỏ, các đoạn càng chồng lấn (overlap) nhiều — nhiều dữ liệu
huấn luyện hơn nhưng cũng dư thừa hơn; `stride == max_length` thì các đoạn
hoàn toàn không chồng lấn.

### Việc cần làm

1. **Phần 5.1 — `GPTDatasetV1.__init__`**:
   - Encode `txt` bằng `tokenizer.encode(txt,
     allowed_special={"<|endoftext|>"})`.
   - `for i in range(0, len(token_ids) - max_length, stride):` cắt
     `input_chunk = token_ids[i : i + max_length]` và
     `target_chunk = token_ids[i + 1 : i + max_length + 1]`.
   - `self.input_ids.append(torch.tensor(input_chunk))` (tương tự cho
     target).
2. **Phần 5.2 — `__len__`**: `return len(self.input_ids)`.
3. **Phần 5.3 — `__getitem__`**: `return self.input_ids[idx],
   self.target_ids[idx]`.
4. **Phần 6.1 — `create_dataloader_v1`**: tạo `tokenizer =
   tiktoken.get_encoding("gpt2")`, tạo `dataset = GPTDatasetV1(...)`, bọc
   trong `DataLoader(dataset, batch_size=..., shuffle=..., drop_last=...,
   num_workers=...)`.

### ✅ Checkpoint 3
```
pytest tests/test_part3.py -v
```
Kỳ vọng: 5/5 test pass. Các test này dùng một tokenizer **giả lập**
(`DummyTokenizer`, luôn trả về 20 token id cố định) để kiểm tra đúng logic
cắt cửa sổ trượt của bạn mà không phụ thuộc internet — hãy đọc code trong
`tests/test_part3.py` để hiểu rõ nó đang kiểm tra điều gì.

---

## Block 4 — Embedding & Pipeline Hoàn Chỉnh (Phần 7.1, 8.1)

### Ý tưởng

Token id vẫn chỉ là số nguyên rời rạc — ta cần biến chúng thành vector số
thực (embedding) để mạng neural có thể xử lý. `nn.Embedding(vocab_size,
output_dim)` về bản chất là một **bảng tra cứu**: ma trận trọng số
`(vocab_size, output_dim)`, trong đó hàng thứ `k` chính là vector embedding
của token id `k`.

Nhưng embedding riêng của token không mang thông tin **vị trí** của nó
trong câu — hai token giống nhau ở hai vị trí khác nhau sẽ có embedding
giống hệt nhau nếu chỉ dùng token embedding. Vì vậy ta cộng thêm
**positional embedding** (cũng là một bảng tra cứu, nhưng tra theo vị trí
0, 1, 2, ... thay vì theo token id) để tạo ra input embedding cuối cùng.

### Việc cần làm

1. **Phần 7.1 — `demo_token_embedding`**: `torch.manual_seed(seed)`, tạo
   `embedding_layer = torch.nn.Embedding(vocab_size, output_dim)`, trả về
   `(embedding_layer.weight, embedding_layer(input_ids))`.
2. **Phần 8.1 — `build_input_embeddings`**: ghép Block 3 (dataloader) +
   token embedding + positional embedding:
   - Lấy 1 batch từ `create_dataloader_v1(..., stride=max_length,
     shuffle=False)`.
   - `token_embeddings = token_embedding_layer(inputs)`
   - `pos_embeddings = pos_embedding_layer(torch.arange(max_length))`
   - `input_embeddings = token_embeddings + pos_embeddings` (broadcasting
     tự lặp `pos_embeddings` cho từng mẫu trong batch)

### ✅ Checkpoint 4
```
pytest tests/test_part4.py -v
```
Kỳ vọng: 4/4 test pass. Với tham số mặc định, `build_input_embeddings` phải
trả về tensor shape `(8, 4, 256)` — giống hệt kết quả in ra trong
`dataloader.ipynb` gốc.

---

## Sau khi hoàn thành

```
pytest tests/ -v        # tất cả checkpoint đều pass
python template.py      # chạy thử pipeline hoàn chỉnh trên dữ liệu thật
python grade.py          # xem điểm tổng
```

Đừng quên trả lời 6 câu hỏi trong [exercises.md](./exercises.md) trước khi
nộp bài — xem [README.md](./README.md) để biết quy trình nộp bài.
