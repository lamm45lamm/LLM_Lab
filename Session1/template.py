"""
================================================================================
 TEMPLATE — DATA PIPELINE CHO LLM (TOKENIZATION -> EMBEDDING)
================================================================================
Đây là file DUY NHẤT bạn cần chỉnh sửa trong bài lab này.
Tìm các khối được đánh dấu:

        # TODO (x.y): <mô tả việc cần làm>
        # Hướng dẫn: ...
        raise NotImplementedError("Phần x.y chưa được cài đặt")

Xoá dòng `raise NotImplementedError(...)` rồi viết code của bạn ngay bên
dưới, dựa theo phần "Hướng dẫn" phía trên. KHÔNG sửa phần code không có TODO.

Xem hướng dẫn chi tiết từng bước trong LAB_GUIDE.md.
Nguồn tham khảo gốc: "Build a Large Language Model From Scratch" -
Sebastian Raschka — https://github.com/rasbt/LLMs-from-scratch
================================================================================
"""

import os
import re

import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# HÀM TIỆN ÍCH — ĐÃ HOÀN CHỈNH, KHÔNG CẦN SỬA
# ==============================================================================
def download_text(file_path: str = "the-verdict.txt") -> str:
    """Tải truyện ngắn 'The Verdict' (Edith Wharton) nếu chưa có, đọc nội dung."""
    if not os.path.exists(file_path):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_text(text: str) -> list:
    """Tách `text` thành list các token (từ + dấu câu riêng biệt).

    Ví dụ: tokenize_text("Hello, world!") -> ['Hello', ',', 'world', '!']
    Dùng lại hàm này bên trong encode() của các tokenizer ở BLOCK 1.
    """
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    return result


# ==============================================================================
# BLOCK 1 — VOCAB & TOKENIZER (Phần 1.x, 2.x)  ->  tests/test_part1.py
# ==============================================================================
def build_vocab(tokens: list) -> dict:
    """
    TODO (1.1): Xây dựng bảng từ vựng (vocab) từ danh sách token.

    Yêu cầu:
      - Lấy các token DUY NHẤT (không trùng lặp), sắp xếp theo alphabet.
      - Gán cho mỗi token một id nguyên, bắt đầu từ 0, tăng dần theo thứ tự
        đã sắp xếp.

    Hướng dẫn: set(...) để loại trùng, sorted(...) để sắp xếp, enumerate(...)
    kết hợp dict comprehension để tạo mapping token -> id.

    Trả về: dict[str, int]
    """
    # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
    raise NotImplementedError("Phần 1.1 chưa được cài đặt")


class SimpleTokenizerV1:
    """Tokenizer đơn giản: text <-> id dựa trên vocab cho trước.
    KHÔNG xử lý được từ chưa có trong vocab (xem SimpleTokenizerV2)."""

    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list:
        """
        TODO (1.2): Chuyển `text` thành danh sách token id.

        Hướng dẫn:
          - Dùng tokenize_text() để tách `text` thành token.
          - Tra id tương ứng của từng token trong self.str_to_int.
          - Trả về list[int] theo đúng thứ tự token xuất hiện.
        """
        # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
        raise NotImplementedError("Phần 1.2 chưa được cài đặt")

    def decode(self, ids: list) -> str:
        """
        TODO (1.3): Chuyển danh sách id ngược lại thành văn bản.

        Hướng dẫn:
          - Tra ngược mỗi id sang token qua self.int_to_str, nối bằng " ".
          - Nối bằng dấu cách sẽ tạo khoảng trắng thừa trước dấu câu
            (vd: "Hello , world ."). Dùng re.sub để xoá khoảng trắng đứng
            ngay trước các ký tự , . ? ! " ( ) '
            Gợi ý pattern:  r'\\s+([,.?!"()\\'])'  ->  thay bằng r'\\1'
        """
        # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
        raise NotImplementedError("Phần 1.3 chưa được cài đặt")


def build_vocab_with_special_tokens(tokens: list) -> dict:
    """
    TODO (2.1): Giống build_vocab(), nhưng bổ sung thêm 2 token đặc biệt vào
    CUỐI vocab (theo đúng thứ tự này):
        "<|endoftext|>"  -> đánh dấu ranh giới giữa 2 văn bản khác nhau
        "<|unk|>"        -> thay thế cho từ không có trong vocab

    Trả về: dict[str, int], trong đó "<|unk|>" phải có id LỚN NHẤT.
    """
    # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
    raise NotImplementedError("Phần 2.1 chưa được cài đặt")


class SimpleTokenizerV2:
    """Giống SimpleTokenizerV1, nhưng thay mọi từ KHÔNG có trong vocab
    bằng token đặc biệt "<|unk|>" thay vì báo lỗi KeyError."""

    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list:
        """
        TODO (2.2): Giống encode() của SimpleTokenizerV1, NHƯNG trước khi
        tra id, kiểm tra từng token có nằm trong self.str_to_int không —
        nếu không, thay token đó bằng chuỗi "<|unk|>" trước khi tra id.
        """
        # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
        raise NotImplementedError("Phần 2.2 chưa được cài đặt")

    def decode(self, ids: list) -> str:
        """
        TODO (2.3): Cài đặt giống hệt decode() của SimpleTokenizerV1 (1.3).
        """
        # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
        raise NotImplementedError("Phần 2.3 chưa được cài đặt")


# ==============================================================================
# BLOCK 2 — BPE & TRỰC GIÁC SLIDING WINDOW (Phần 3.x, 4.x) -> tests/test_part2.py
# ==============================================================================
def bpe_roundtrip_demo(text: str) -> tuple:
    """
    TODO (3.1): Dùng bộ mã hoá BPE của GPT-2 (thư viện tiktoken) để encode
    rồi decode lại `text`.

    Vì sao cần BPE thay vì tokenizer tự viết ở Block 1?
      - BPE tách được cả từ chưa từng thấy thành các đơn vị con (subword),
        nên không cần token "<|unk|>" nữa. Đây cũng là tokenizer thật của GPT-2.

    Hướng dẫn:
      - tokenizer = tiktoken.get_encoding("gpt2")
      - ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        (allowed_special cho phép "<|endoftext|>" được coi là 1 token đặc
        biệt thay vì bị tách nhỏ).
      - decoded_text = tokenizer.decode(ids)

    Trả về: tuple (ids, decoded_text)
    """
    # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
    raise NotImplementedError("Phần 3.1 chưa được cài đặt")


def get_next_token_pairs(token_ids: list, context_size: int) -> list:
    """
    TODO (4.1): Tạo các cặp (context, target) minh hoạ bài toán "dự đoán từ
    kế tiếp", với độ dài context tăng dần từ 1 đến context_size.

    Ví dụ: token_ids = [10, 20, 30, 40, 50], context_size = 4
        -> [([10], 20), ([10, 20], 30), ([10, 20, 30], 40), ([10, 20, 30, 40], 50)]

    Hướng dẫn: for i in range(1, context_size + 1):
                   context = token_ids[:i]; target = token_ids[i]

    Trả về: list[tuple(list[int], int)]
    """
    # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
    raise NotImplementedError("Phần 4.1 chưa được cài đặt")


# ==============================================================================
# BLOCK 3 — GPTDatasetV1 & DATALOADER (Phần 5.x, 6.x) -> tests/test_part3.py
# ==============================================================================
# Đây là phần QUAN TRỌNG NHẤT của bài lab: mã hoá toàn bộ văn bản, sau đó cắt
# thành nhiều đoạn (chunk) độ dài max_length bằng "sliding window", mỗi đoạn
# input đi kèm 1 đoạn target là chính đoạn input dịch phải 1 vị trí.
# ==============================================================================
class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """
        TODO (5.1): Cài đặt constructor:
          1. Mã hoá toàn bộ `txt` bằng `tokenizer.encode(txt,
             allowed_special={"<|endoftext|>"})` -> token_ids.
          2. Trượt cửa sổ để cắt token_ids thành nhiều đoạn:
                 for i in range(0, len(token_ids) - max_length, stride):
                     input_chunk  = token_ids[i : i + max_length]
                     target_chunk = token_ids[i + 1 : i + max_length + 1]
             `stride` là bước nhảy giữa 2 cửa sổ liên tiếp.
          3. Chuyển mỗi input_chunk, target_chunk thành torch.tensor(...) và
             append vào self.input_ids, self.target_ids tương ứng.
        """
        self.input_ids = []
        self.target_ids = []

        # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
        raise NotImplementedError("Phần 5.1 chưa được cài đặt")

    def __len__(self):
        """TODO (5.2): Trả về số mẫu trong dataset (độ dài self.input_ids)."""
        # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
        raise NotImplementedError("Phần 5.2 chưa được cài đặt")

    def __getitem__(self, idx):
        """TODO (5.3): Trả về tuple (self.input_ids[idx], self.target_ids[idx])."""
        # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
        raise NotImplementedError("Phần 5.3 chưa được cài đặt")


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                          shuffle=True, drop_last=True, num_workers=0):
    """
    TODO (6.1): Ghép mọi thứ thành 1 DataLoader hoàn chỉnh:
      1. tokenizer = tiktoken.get_encoding("gpt2")
      2. dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
      3. dataloader = DataLoader(dataset, batch_size=batch_size,
             shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
      4. return dataloader
    """
    # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
    raise NotImplementedError("Phần 6.1 chưa được cài đặt")


# ==============================================================================
# BLOCK 4 — EMBEDDING & PIPELINE HOÀN CHỈNH (Phần 7.x, 8.x) -> tests/test_part4.py
# ==============================================================================
def demo_token_embedding(vocab_size: int, output_dim: int,
                          input_ids: torch.Tensor, seed: int = 123):
    """
    TODO (7.1): Tạo embedding layer và tra embedding cho `input_ids`.

    Hướng dẫn:
      - torch.manual_seed(seed)
      - embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        (bản chất là ma trận trọng số (vocab_size, output_dim); mỗi HÀNG là
        vector embedding của 1 token id — id chính là chỉ số hàng)
      - embedded = embedding_layer(input_ids)

    Trả về: tuple (embedding_layer.weight, embedded)
    """
    # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
    raise NotImplementedError("Phần 7.1 chưa được cài đặt")


def build_input_embeddings(raw_text: str, vocab_size: int = 50257,
                            output_dim: int = 256, max_length: int = 4,
                            batch_size: int = 8) -> torch.Tensor:
    """
    TODO (8.1): Xây pipeline hoàn chỉnh: văn bản thô -> input embedding.

    Các bước:
      1. dataloader = create_dataloader_v1(raw_text, batch_size=batch_size,
             max_length=max_length, stride=max_length, shuffle=False)
      2. inputs, targets = next(iter(dataloader))   (không dùng targets)
      3. token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
         token_embeddings = token_embedding_layer(inputs)
      4. pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
         pos_embeddings = pos_embedding_layer(torch.arange(max_length))
      5. input_embeddings = token_embeddings + pos_embeddings
      6. return input_embeddings

    Kết quả mong đợi: shape == (batch_size, max_length, output_dim)
    """
    # TODO: xoá dòng raise bên dưới và viết code của bạn tại đây
    raise NotImplementedError("Phần 8.1 chưa được cài đặt")


# ==============================================================================
# DEMO THỦ CÔNG (không bắt buộc) — chạy: python template.py
# ==============================================================================
if __name__ == "__main__":
    print("Đang tải dữ liệu mẫu (the-verdict.txt)...")
    raw_text = download_text()
    print(f"Đã tải xong, tổng số ký tự: {len(raw_text)}\n")

    print("Chạy pipeline hoàn chỉnh (Block 4)...")
    input_embeddings = build_input_embeddings(raw_text)
    print("Shape của input_embeddings:", input_embeddings.shape)
