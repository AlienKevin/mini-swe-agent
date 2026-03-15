# View-Before-Edit System Prompt

Target prompt template: `src/minisweagent/config/benchmarks/swebench_qwen3_8b_modal.yaml`

Applied change:
- Added `Never edit a file you haven't already viewed.` to the system prompt.
- Exact patch: [`patch.diff`](./patch.diff)

Artifacts:
- Local taxonomy: [`results/qwen3-8b-verified-100-v2/trajectory_failure_taxonomy.md`](../../results/qwen3-8b-verified-100-v2/trajectory_failure_taxonomy.md)
- Local taxonomy summary JSON: [`results/qwen3-8b-verified-100-v2/trajectory_failure_taxonomy_summary.json`](../../results/qwen3-8b-verified-100-v2/trajectory_failure_taxonomy_summary.json)
- Hugging Face upload: [`AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-8b-view-before-edit-system-prompt-eval`](https://huggingface.co/datasets/AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-8b-view-before-edit-system-prompt-eval)

## Top-Level Taxonomy

- `no_deliverable`: 48 / 95
- `semantically_bad_patch`: 31 / 95
- `malformed_patch`: 13 / 95
- `resolved`: 3 / 95

## Leaf Breakdown

- `empty_submission_or_missing_patch`: 48
- `api_logic_or_target_mismatch`: 27
- `syntactically_invalid_or_broken_patch`: 10
- `fabricated_or_placeholder_diff`: 3
- `benchmark_resolved`: 3
- `no_op_or_irrelevant_change`: 2
- `ambiguous_unverified_patch`: 1
- `overbroad_or_destructive_change`: 1

## Cross-Cutting Contributors

- `blind_text_editing`: 61
- `environment_or_dependency_breakage`: 56
- `no_meaningful_verification`: 22
- `syntax_or_indentation_issue`: 12
- `no_op_or_irrelevant_change`: 7
- `fabricated_or_placeholder_diff`: 5
- `overbroad_or_destructive_change`: 2
- `wrong_file_or_path`: 1

## Delta vs. Original

- `resolved`: `0 -> 3`
- `no_deliverable`: `47 -> 48`
- `semantically_bad_patch`: `35 -> 31`
- `malformed_patch`: `15 -> 13`
- `blind_text_editing`: `60 -> 61`
- `environment_or_dependency_breakage`: `32 -> 56`

## Notes

- This experiment did improve benchmark resolution from 0 to 3 solved instances.
- It did not materially reduce blind text editing: the count stayed essentially flat.
- Empty or missing submissions remained the dominant failure class.
- Environment and dependency breakage increased sharply in this run and likely masked any modest prompt-only gains.
