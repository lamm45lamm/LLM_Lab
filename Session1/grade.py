"""
Chấm điểm tự động cho bài lab Data Pipeline (Tokenization -> Embedding).

Chạy:   python grade.py

Ưu tiên chấm bài trong solution/solution.py (dùng khi nộp bài); nếu thư mục
solution/ chưa tồn tại thì chấm trực tiếp template.py (dùng khi làm bài tại
chỗ để tự kiểm tra tiến độ).
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

WEIGHTS = [
    ("tests/test_part1.py", "Block 1 - Vocab & Tokenizer (V1/V2)", 20),
    ("tests/test_part2.py", "Block 2 - BPE (tiktoken) & sliding window", 15),
    ("tests/test_part3.py", "Block 3 - GPTDatasetV1 & DataLoader", 30),
    ("tests/test_part4.py", "Block 4 - Embedding & pipeline hoàn chỉnh", 20),
]
EXERCISES_FILE = "exercises.md"
EXERCISES_MAX_POINTS = 15
EXERCISES_PLACEHOLDER = "(Viết câu trả lời của bạn vào đây)"


def run_pytest(rel_path: str):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", rel_path, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    output = result.stdout + "\n" + result.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", output)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", output)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", output)) else 0
    total = passed + failed + errors
    return passed, total, output


def score_tests():
    rows = []
    total_points = 0.0
    for rel_path, label, weight in WEIGHTS:
        passed, total, output = run_pytest(rel_path)
        if total == 0:
            points = 0.0
            status = "KHÔNG CHẠY ĐƯỢC (xem lỗi bên dưới)"
        else:
            points = weight * passed / total
            status = f"{passed}/{total} test pass"
        total_points += points
        rows.append((label, status, points, weight, output if total == 0 else None))
    return rows, total_points


def score_exercises():
    path = os.path.join(ROOT, EXERCISES_FILE)
    if not os.path.exists(path):
        return 0, 0
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    sections = re.split(r"\n(?=## Câu \d+)", content)
    sections = [s for s in sections if s.strip().startswith("## Câu")]
    n_total = len(sections)
    n_answered = 0
    for s in sections:
        after = s.split("**Trả lời:**", 1)
        if len(after) < 2:
            continue
        answer = after[1].strip()
        if answer and EXERCISES_PLACEHOLDER not in answer:
            n_answered += 1
    return n_answered, n_total


def main():
    solution_path = os.path.join(ROOT, "solution", "solution.py")
    mode = "solution/solution.py" if os.path.exists(solution_path) else "template.py"
    print("=" * 72)
    print(f"Đang chấm bài dựa trên: {mode}")
    print("=" * 72)

    rows, tests_points = score_tests()
    print(f"{'Tiêu chí':45s} {'Kết quả':22s} {'Điểm':>8s}")
    print("-" * 72)
    for label, status, points, weight, error_output in rows:
        print(f"{label:45s} {status:22s} {points:6.1f}/{weight}")
        if error_output:
            print("  --- log lỗi (rút gọn) ---")
            print("  " + "\n  ".join(error_output.strip().splitlines()[-8:]))

    n_answered, n_total = score_exercises()
    if n_total > 0:
        ex_points = EXERCISES_MAX_POINTS * n_answered / n_total
    else:
        ex_points = 0.0
    print(f"{'exercises.md - câu hỏi phản ánh':45s} "
          f"{f'{n_answered}/{n_total} câu':22s} {ex_points:6.1f}/{EXERCISES_MAX_POINTS}")

    total = tests_points + ex_points
    max_total = sum(w for _, _, w in WEIGHTS) + EXERCISES_MAX_POINTS
    print("-" * 72)
    print(f"{'TỔNG ĐIỂM':45s} {'':22s} {total:6.1f}/{max_total}")
    print("=" * 72)


if __name__ == "__main__":
    main()
