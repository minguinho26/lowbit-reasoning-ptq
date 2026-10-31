"""평가 벤치마크: 데이터 로딩, 프롬프트, 채점.

데이터셋과 프롬프트는 2504.04823 코드(github.com/ruikangliu/Quantized-Reasoning-Models,
lighteval_custom/tasks/reasoning.py)와 같다.
  AIME-120 = xiaoyuanliu/AIME90 (AIME 2022~2024) + yentinglin/aime_2025
  GPQA-Diamond = Idavidrein/gpqa, gpqa_diamond (gated: HF 토큰과 이용 동의 필요)
GPQA 선택지 순서는 고정 seed로 한 번만 섞어서 모든 실행 단위에서 같은 프롬프트를 쓴다(문제 단위 짝짓기를 위해).
"""

import random
import re

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
GPQA_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step "
    "before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
GPQA_SHUFFLE_SEED = 1234
THINK_END_MARKERS = ("</think>", "[/THINK]")


def load_aime120():
    from datasets import load_dataset

    items, seen = [], set()
    for repo in ("xiaoyuanliu/AIME90", "yentinglin/aime_2025"):
        for i, row in enumerate(load_dataset(repo, split="train")):
            key = " ".join(row["problem"].split())
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "uid": f"{repo.split('/')[-1]}-{row.get('id', i)}",
                "kind": "math",
                "question": row["problem"],
                "gold": str(row["answer"]).strip(),
                "prompt": row["problem"] + MATH_SUFFIX,
            })
    assert len(items) == 120, f"AIME-120 문제 수가 {len(items)}개임. 데이터셋 버전을 확인할 것"
    return items


def load_gpqa_diamond():
    from datasets import load_dataset

    rng = random.Random(GPQA_SHUFFLE_SEED)
    items = []
    for i, row in enumerate(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")):
        gold_index = rng.randint(0, 3)
        choices = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        choices.insert(gold_index, row["Correct Answer"])
        items.append({
            "uid": str(row.get("Record ID") or f"gpqa-{i}"),
            "kind": "mcq",
            "question": row["Question"],
            "gold": "ABCD"[gold_index],
            "prompt": GPQA_TEMPLATE.format(Question=row["Question"], A=choices[0], B=choices[1], C=choices[2], D=choices[3]),
        })
    assert len(items) == 198, f"GPQA-Diamond 문제 수가 {len(items)}개임"
    return items


LOADERS = {"aime120": load_aime120, "gpqa_diamond": load_gpqa_diamond}


def load(bench):
    return LOADERS[bench]()


def answer_part(text):
    """thinking 구간 뒤의 최종 답 부분. Qwen3.5는 </think>, Ministral은 [/THINK]."""
    for marker in THINK_END_MARKERS:
        if marker in text:
            return text.rsplit(marker, 1)[1]
    return text


def _grade_math(ans, gold):
    from math_verify import parse, verify

    try:
        return bool(verify(parse(f"${gold}$"), parse(ans)))
    except Exception:
        return False


_MCQ_PATTERNS = [
    re.compile(r"answer\s*[:：]\s*\**\s*\(?\$?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}"),
]


def _grade_mcq(ans, gold):
    for pat in _MCQ_PATTERNS:
        found = pat.findall(ans)
        if found:
            return found[-1].upper() == gold
    return False


def grade(item, text, truncated):
    """max_tokens에 걸려 끝나지 못한 샘플은 오답."""
    if truncated:
        return False
    ans = answer_part(text)
    if item["kind"] == "math":
        return _grade_math(ans, item["gold"])
    return _grade_mcq(ans, item["gold"])
