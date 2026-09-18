"""
Helper dùng nội bộ bởi các file test — KHÔNG PHẢI nơi bạn viết code.

Ưu tiên chấm bài trong solution/solution.py (dùng khi nộp bài); nếu không
tồn tại thì rơi về template.py (dùng khi làm bài tại chỗ / CP checkpoint).
"""
import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_solution_path = os.path.join(ROOT, "solution", "solution.py")

if os.path.exists(_solution_path):
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location("target_module", _solution_path)
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
else:
    import template as target  # noqa: F401
