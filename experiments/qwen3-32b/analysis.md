# Qwen3-32B Baseline

Target prompt template: `src/minisweagent/config/benchmarks/swebench_qwen3_32b_modal.yaml`

Model: Qwen3-32B on Together AI dedicated endpoint (2x H100 80GB SXM).

Patch:
- `patch.diff` is intentionally empty — same original prompt as Qwen3-8B baseline, just a larger model.

Artifacts:
- Hugging Face upload: [`AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-32b-eval`](https://huggingface.co/datasets/AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-32b-eval)

## Results

- `resolved`: 6 / 100 (6.00%)
- `non_empty_patches`: 63 / 100
- `empty_patches`: 37 / 100

## Resolved Instances

- `django__django-15368`: bulk_update() with F() expressions
- `django__django-16485`
- `scikit-learn__scikit-learn-15100`
- `sympy__sympy-13480`
- `sympy__sympy-16450`
- `sympy__sympy-22456`

## Comparison with Qwen3-8B

| Metric | Qwen3-8B (original) | Qwen3-32B (original) |
|---|---|---|
| Resolved | 0/97 (0.00%) | 6/100 (6.00%) |
| Non-empty patches | 50 (51.5%) | 63 (63.0%) |
| Empty patches | 47 (48.5%) | 37 (37.0%) |

## Notes

- Qwen3-32B resolves 6 instances out of the box with the original prompt — 6x the best 8B prompt-engineered result.
- The non-empty patch rate improved from 51.5% to 63.0%, suggesting the larger model produces more successful edits.
- Only 1 overlap with 8B-v2 resolved set (`django__django-15368`).
- This serves as a sanity check that the evaluation pipeline works and that model scale is the dominant factor for SWE-bench performance.
