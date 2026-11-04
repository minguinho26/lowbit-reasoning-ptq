"""AWQ의 세 포맷을 작은 Qwen3로 양자화·저장·재로드하는 통합 검사.

quant 환경에서 실행한다. CUDA_VISIBLE_DEVICES=''로 실행하면 실제 평가 GPU를
사용하지 않는다. 이 검사는 recipe/파일 호환성을 확인하며 8B 정확도 검사는 아니다.
"""

import json
from contextlib import nullcontext
from unittest.mock import patch
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import Dataset
from llmcompressor import oneshot
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    CompressedTensorsConfig,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import enumerate_runs, load_config
from quantize import build_recipe, patch_awq_output_cache


class AWQIntegrationTest(unittest.TestCase):
    def test_active_matrix(self):
        """AutoRound를 제외하고 18개 AWQ 셀을 갖는 52-run 구성을 확인한다."""
        runs = enumerate_runs(load_config())
        self.assertEqual(len(runs), 52)
        self.assertFalse(any(r["algo"] == "autoround" for r in runs))
        awq = [r for r in runs if r["algo"] == "awq"]
        self.assertEqual(len(awq), 18)
        for model in ("qwen3-8b", "qwen3-4b", "qwen3-1.7b"):
            self.assertEqual(sum(r["model"] == model for r in awq), 6)

    def test_awq_output_cache(self):
        """출력값은 동일하고 큰 보조 tensor는 batch 사이에 누적되지 않는지 검사한다."""
        from llmcompressor.modifiers.transform.awq import AWQModifier

        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.auxiliary = []
                self.peak_live = 0

            def forward(self, x):
                auxiliary = torch.ones(256)
                self.auxiliary.append(weakref.ref(auxiliary))
                self.peak_live = max(self.peak_live, sum(r() is not None for r in self.auxiliary))
                return x * 2, auxiliary

        original = AWQModifier._run_samples
        session = SimpleNamespace(state=SimpleNamespace(sequential_prefetch=False))
        with patch("llmcompressor.modifiers.transform.awq.base.active_session", return_value=session), patch(
            "llmcompressor.core.session_functions.active_session", return_value=session
        ):
            baseline = Probe()
            cache = [{"x": torch.tensor([i], dtype=torch.float32)} for i in range(3)]
            expected = original(SimpleNamespace(_parent_args_cache={baseline: cache}), baseline)
            patch_awq_output_cache(AWQModifier)
            candidate = Probe()
            actual = AWQModifier._run_samples(SimpleNamespace(_parent_args_cache={candidate: cache}), candidate)
        self.assertEqual(baseline.peak_live, 3)
        self.assertEqual(candidate.peak_live, 1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        self.assertTrue(all(ref() is None for ref in candidate.auxiliary))

    def test_quantize_save_reload(self):
        """양자화 포맷을 유지하고 lm_head를 보존한 채 실제 압축 파일을 재로드한다."""
        torch.set_num_threads(2)
        backend = Tokenizer(WordLevel({str(i): i for i in range(256)}, unk_token="0"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="0", pad_token="0", bos_token="1", eos_token="2"
        )
        dataset = Dataset.from_dict({
            "input_ids": [[3 + (i + j) % 200 for j in range(16)] for i in range(4)],
            "attention_mask": [[1] * 16 for _ in range(4)],
        })
        cfg = load_config()
        for fmt in ("nvfp4a16", "nvfp4", "w2a16"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(0)
                model = Qwen3ForCausalLM(Qwen3Config(
                    vocab_size=256, hidden_size=128, intermediate_size=256,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                    head_dim=32, max_position_embeddings=128, tie_word_embeddings=False,
                    bos_token_id=1, eos_token_id=2, pad_token_id=0,
                    attn_implementation="eager", dtype="bfloat16",
                )).to(dtype=torch.bfloat16).eval()
                # oneshot은 원본 config를 디스크에서도 검사하므로 테스트용 원본 경로를 준다.
                original = Path(tmp) / "original"
                model.config._name_or_path = str(original)
                model.config.save_pretrained(original)
                head = model.lm_head.weight.detach().clone()
                recipe = build_recipe(cfg, "awq", cfg["formats"][fmt]["scheme"], ["lm_head"])
                # CPU 검사에서는 CUDA 전용 pinned-memory 최적화만 건너뛴다.
                # tensor 값과 AWQ 계산은 그대로이며 실제 GPU 실행에는 패치를 적용하지 않는다.
                pinning = nullcontext() if torch.cuda.is_available() else patch(
                    "llmcompressor.pipelines.cache.IntermediatesCache.pin_memory", return_value=None
                )
                with pinning:
                    oneshot(
                        model=model, tokenizer=tokenizer, recipe=recipe,
                        dataset=dataset, max_seq_length=16, num_calibration_samples=4,
                        shuffle_calibration_samples=False, clear_sparse_session=True,
                    )
                self.assertTrue(torch.equal(head.cpu(), model.lm_head.weight.detach().cpu()))
                model.save_pretrained(tmp, save_compressed=True)
                saved = json.loads((Path(tmp) / "config.json").read_text())
                groups = saved["quantization_config"]["config_groups"].values()
                for group in groups:
                    self.assertEqual(group["weights"]["num_bits"], 2 if fmt == "w2a16" else 4)
                    self.assertEqual(group["weights"]["group_size"], 128 if fmt == "w2a16" else 16)
                    self.assertEqual(group["weights"]["symmetric"], True)
                    self.assertEqual(group["input_activations"] is not None, fmt == "nvfp4")
                self.assertTrue(list(Path(tmp).glob("*.safetensors")))
                restored = AutoModelForCausalLM.from_pretrained(
                    tmp, quantization_config=CompressedTensorsConfig(run_compressed=False),
                    device_map="cpu", dtype=torch.bfloat16,
                ).eval()
                with torch.no_grad():
                    logits = restored(torch.tensor([dataset[0]["input_ids"]])).logits
                self.assertEqual(tuple(logits.shape), (1, 16, 256))
                self.assertTrue(torch.isfinite(logits).all())
                print(f"AWQ {fmt}: compressed save/reload and forward passed", flush=True)


if __name__ == "__main__":
    unittest.main()
