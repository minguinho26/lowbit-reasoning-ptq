"""일부 문항 평가 결과가 전체 평가 파일과 분리되는지 검사한다."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_config
from evaluate import result_path


class ScriptTest(unittest.TestCase):
    def test_limit_results_stay_out_of_full_evaluation(self):
        cfg = load_config()
        full = result_path(cfg, "qwen3-8b__bf16", "aime120", 0)
        limited = result_path(cfg, "qwen3-8b__bf16", "aime120", 0, 30)
        self.assertEqual(limited.parent, full.parent)
        self.assertEqual(limited.name, "aime120__s0.limit30.jsonl")
        self.assertEqual(full.name, "aime120__s0.jsonl")


if __name__ == "__main__":
    unittest.main()
