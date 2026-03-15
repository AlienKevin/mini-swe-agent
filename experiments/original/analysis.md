# Original

Target prompt template: `src/minisweagent/config/benchmarks/swebench_qwen3_8b_modal.yaml`

Artifacts:
- Local taxonomy: [`results/qwen3-8b-verified-100/trajectory_failure_taxonomy.md`](../../results/qwen3-8b-verified-100/trajectory_failure_taxonomy.md)
- Local taxonomy summary JSON: [`results/qwen3-8b-verified-100/trajectory_failure_taxonomy_summary.json`](../../results/qwen3-8b-verified-100/trajectory_failure_taxonomy_summary.json)
- Hugging Face upload: [`AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-8b-eval`](https://huggingface.co/datasets/AlienKevin/sweb-verified-rand-100-mini-swe-v2.2.7-regex-parser-qwen3-8b-eval)

Patch:
- `patch.diff` is intentionally empty for the baseline run.

## Top-Level Taxonomy

- `no_deliverable`: 47 / 97
- `semantically_bad_patch`: 35 / 97
- `malformed_patch`: 15 / 97
- `resolved`: 0 / 97

## Leaf Breakdown

- `empty_submission_or_missing_patch`: 46
- `api_logic_or_target_mismatch`: 21
- `fabricated_or_placeholder_diff`: 9
- `no_op_or_irrelevant_change`: 7
- `overbroad_or_destructive_change`: 6
- `syntactically_invalid_or_broken_patch`: 6
- `plausible_but_unverified_fix`: 1
- `context_overflow_before_submission`: 1

## Cross-Cutting Contributors

- `blind_text_editing`: 60
- `environment_or_dependency_breakage`: 32
- `no_meaningful_verification`: 21
- `syntax_or_indentation_issue`: 13
- `fabricated_or_placeholder_diff`: 11
- `no_op_or_irrelevant_change`: 8
- `overbroad_or_destructive_change`: 7
- `wrong_file_or_path`: 4

## Notes

- This run produced no benchmark-resolved instances in the analyzed 97 trajectories.
- The dominant failure mode was still empty or missing submissions, followed by patch logic mismatches.
- Blind text editing remained a major contributor even in the baseline, appearing in 60 trajectories.
