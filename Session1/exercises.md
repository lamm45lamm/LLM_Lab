# Phiếu Bài Tập & Phản Ánh

Trả lời ngắn gọn (2-4 câu mỗi câu) ngay bên dưới mỗi câu hỏi, thay cho dòng
placeholder. File này được tính điểm dựa trên **số câu đã trả lời** (xem
`grade.py`), nội dung có thể được giảng viên xem lại và điều chỉnh điểm sau.

---

## Câu 1: Vì sao SimpleTokenizerV1 (Block 1) sẽ báo lỗi KeyError khi gặp một
từ không có trong vocab, còn SimpleTokenizerV2 thì không?

**Trả lời:**
(Viết câu trả lời của bạn vào đây)

---

## Câu 2: BPE (Byte Pair Encoding) giải quyết vấn đề "từ chưa từng thấy"
(unknown word) khác với cách dùng token `<|unk|>` như thế nào?

**Trả lời:**
(Viết câu trả lời của bạn vào đây)

---

## Câu 3: Trong `GPTDatasetV1`, nếu bạn đặt `stride = max_length` thay vì
`stride = 1`, dữ liệu huấn luyện sẽ thay đổi như thế nào? Đánh đổi
(trade-off) giữa 2 lựa chọn này là gì (số lượng mẫu, độ trùng lặp thông tin)?

**Trả lời:**
(Viết câu trả lời của bạn vào đây)

---

## Câu 4: Tại sao `target_chunk` trong `GPTDatasetV1` luôn là `input_chunk`
dịch phải đúng 1 vị trí? Điều này liên quan gì đến việc mô hình LLM được
huấn luyện để làm gì?

**Trả lời:**
(Viết câu trả lời của bạn vào đây)

---

## Câu 5: Token embedding và positional embedding có vai trò khác nhau như
thế nào? Điều gì sẽ xảy ra nếu mô hình chỉ dùng token embedding mà bỏ qua
positional embedding?

**Trả lời:**
(Viết câu trả lời của bạn vào đây)

---

## Câu 6 (mở rộng): Nếu `max_length` (độ dài ngữ cảnh) tăng từ 4 lên 1024
như GPT-2 thật, những phần nào trong bài lab này (vocab, tokenizer, dataset,
embedding) cần thay đổi và phần nào giữ nguyên?

**Trả lời:**
(Viết câu trả lời của bạn vào đây)
