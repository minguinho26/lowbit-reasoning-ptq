"""미완료 seed와 잘못된 문제 집합이 결과 표에 들어가지 않는지 검사한다."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze import load_eval
from common import load_config


class AnalysisInputTest(unittest.TestCase):
    def write_seed(self, root, seed, uids):
        folder = root / "eval/qwen3-8b__bf16"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"aime120__s{seed}.jsonl"
        f.write_text("".join(json.dumps({"uid": u, "correct": True, "n_tokens": 20}) + "\n" for u in uids))
        f.with_suffix(".done.json").write_text(json.dumps({"n": len(uids), "acc": 1.0}))

    def test_only_all_seeds_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config()
            cfg["_dirs"]["results"] = root
            self.write_seed(root, 0, ["a", "b"])
            self.assertFalse(load_eval(cfg))
            for seed in (1, 2):
                self.write_seed(root, seed, ["a", "b"])
            data = load_eval(cfg)[("qwen3-8b__bf16", "aime120")]
            self.assertEqual(set(data), {"a", "b"})
            self.assertTrue(all(len(records) == 3 for records in data.values()))

    def test_duplicate_and_mismatched_ids_are_rejected(self):
        for broken in (["a", "a"], ["a", "c"]):
            with self.subTest(uids=broken), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = load_config()
                cfg["_dirs"]["results"] = root
                for seed in (0, 1):
                    self.write_seed(root, seed, ["a", "b"])
                self.write_seed(root, 2, broken)
                with self.assertRaises(ValueError):
                    load_eval(cfg)


if __name__ == "__main__":
    unittest.main()
