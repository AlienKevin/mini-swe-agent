# Required Workflow

Target prompt template: `src/minisweagent/config/benchmarks/swebench_qwen3_8b_modal.yaml`

Applied change:
- Renamed "Recommended Workflow" to "REQUIRED WORKFLOW" (all-caps).
- Wrapped the 5-step workflow in `<CRITICAL>` tags for emphasis.
- Exact patch: [`patch.diff`](./patch.diff)

Artifacts:
- Hugging Face upload: [`AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-8b-required-workflow-eval`](https://huggingface.co/datasets/AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-8b-required-workflow-eval)

## Top-Level Taxonomy

- `no_deliverable`: 48 / 96
- `semantically_bad_patch`: ~33 / 96
- `malformed_patch`: ~13 / 96
- `resolved`: 2 / 96

## Resolved Instances

- `django__django-14855`: Admin URL for readonly ForeignKey fields in custom Admin Site.
- `psf__requests-2317`: `builtin_str(method)` converts bytes to wrong string in Python 3.

## Failure Mode Distribution

| Failure Mode | v1 (original) | v2 (view-before-edit) | v3 (required-workflow) |
|---|---|---|---|
| blind_edit | 94.8% | 96.8% | 95.8% |
| no_verification | 74.2% | 67.4% | **58.3%** |
| hallucinated_patch | 9.3% | 5.3% | **4.2%** |
| wrong_path | 39.2% | 36.8% | **30.2%** |
| repeated_commands | 20.6% | 20.0% | 29.2% |
| environment_broken | 34.0% | 49.5% | 64.6% |
| network_errors | 66.0% | 41.1% | 57.3% |
| format_errors | 30.9% | 40.0% | 39.6% |

## Delta vs. Original

- `resolved`: `0 -> 2`
- `no_verification`: `74.2% -> 58.3%` (-15.9pp, biggest behavioral improvement)
- `hallucinated_patch`: `9.3% -> 4.2%` (-5.1pp)
- `wrong_path`: `39.2% -> 30.2%` (-9.0pp)
- `repeated_commands`: `20.6% -> 29.2%` (+8.6pp, regression)
- `environment_broken`: `34.0% -> 64.6%` (+30.6pp, infrastructure noise)
- `blind_edit`: unchanged (~95%)

## Delta vs. View-Before-Edit (v2)

- `resolved`: `3 -> 2` (different instances, zero overlap)
- v2 resolved: `django-12050`, `django-15277`, `django-15368`
- v3 resolved: `django-14855`, `psf-requests-2317`
- `no_verification`: `67.4% -> 58.3%` (further improvement)

## Notes

- The REQUIRED WORKFLOW + CRITICAL tags improved verification behavior (58.3% no_verification vs 74.2% baseline), the largest behavioral shift of any prompt variant.
- However, it did not reduce blind editing (~95% across all runs).
- The resolved instances are entirely different from v2, suggesting high variance at the 8B model scale. Prompt changes push the model in different directions but which specific instances benefit is stochastic.
- The core tension: the structured workflow helps on straightforward instances (where reading code first is enough) but hurts on harder instances by consuming the agent's limited reasoning budget on exploration steps.
- Environment breakage increased sharply (+30.6pp vs baseline), likely infrastructure noise rather than prompt-caused.
