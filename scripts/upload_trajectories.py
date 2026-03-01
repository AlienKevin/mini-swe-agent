#!/usr/bin/env python3
"""Convert mini-swe-agent trajectories to HuggingFace dataset format and upload.

Produces a dataset matching the schema of AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories:
  messages, instance_id, resolved, model, traj_id, patch
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset


def load_eval_results(eval_path: str) -> set[str]:
    """Load evaluation results and return set of resolved instance IDs."""
    path = Path(eval_path)
    resolved = set()

    if path.is_dir():
        # Look for results JSON files in the directory
        for f in sorted(path.glob("**/*.json")):
            data = json.loads(f.read_text())
            # Handle swebench harness format: {instance_id: {"resolved": bool}}
            if isinstance(data, dict):
                for iid, result in data.items():
                    if isinstance(result, dict) and result.get("resolved"):
                        resolved.add(iid)
                    elif isinstance(result, bool) and result:
                        resolved.add(iid)
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for iid, result in data.items():
                if isinstance(result, dict) and result.get("resolved"):
                    resolved.add(iid)
                elif isinstance(result, bool) and result:
                    resolved.add(iid)
        elif isinstance(data, list):
            # List of resolved instance IDs
            resolved = set(data)
    elif path.suffix == ".txt":
        resolved = set(path.read_text().strip().splitlines())

    return resolved


def convert_trajectory(traj_data: dict) -> list[dict]:
    """Convert a raw trajectory's messages to the HF dataset chat format.

    Each message gets: role, content, name, tool_call_id, tool_calls
    """
    raw_messages = traj_data.get("messages", [])
    converted = []

    for msg in raw_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        entry = {
            "role": role,
            "content": content,
            "name": msg.get("name"),
            "tool_call_id": msg.get("tool_call_id"),
            "tool_calls": msg.get("tool_calls", []),
        }

        # If this is an assistant message with a bash command block but no tool_calls,
        # synthesize tool_calls in the expected format
        if role == "assistant" and not entry["tool_calls"] and "```mswea_bash_command" in (content or ""):
            import re

            match = re.search(r"```mswea_bash_command\n(.*?)```", content, re.DOTALL)
            if match:
                command = match.group(1).strip()
                call_id = f"call_{hash(command) & 0xFFFFFFFFFFFF:012x}"
                entry["tool_calls"] = [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": command}),
                    },
                }]

        # If this is a user message that looks like tool output (returncode format),
        # convert to tool role
        if role == "user" and content and "<returncode>" in content and converted:
            prev = converted[-1]
            if prev.get("tool_calls"):
                entry["role"] = "tool"
                entry["tool_call_id"] = prev["tool_calls"][-1]["id"]

        converted.append(entry)

    return converted


def main():
    parser = argparse.ArgumentParser(description="Upload trajectories to HuggingFace")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with trajectory results")
    parser.add_argument("--eval_results", type=str, required=True, help="Path to evaluation results (JSON or directory)")
    parser.add_argument("--repo_id", type=str, default="AlienKevin/SWE-smith-rs-glm-4.6-trajectories")
    parser.add_argument("--model_name", type=str, default="glm-4.6")
    parser.add_argument("--dry_run", action="store_true", help="Don't push, just print stats")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    resolved_ids = load_eval_results(args.eval_results)
    print(f"Loaded {len(resolved_ids)} resolved instance IDs")

    # Find all trajectory files
    traj_files = sorted(results_dir.glob("**/*.traj.json"))
    print(f"Found {len(traj_files)} trajectory files")

    rows = []
    for traj_path in traj_files:
        traj_data = json.loads(traj_path.read_text())

        instance_id = traj_data.get("instance_id", traj_path.stem.replace(".traj", ""))
        info = traj_data.get("info", {})
        patch = info.get("submission", "")

        messages = convert_trajectory(traj_data)

        rows.append({
            "messages": messages,
            "instance_id": instance_id,
            "resolved": instance_id in resolved_ids,
            "model": args.model_name,
            "traj_id": instance_id,
            "patch": patch or "",
        })

    print(f"Converted {len(rows)} trajectories")
    n_resolved = sum(1 for r in rows if r["resolved"])
    print(f"  Resolved: {n_resolved}/{len(rows)} ({100*n_resolved/len(rows):.1f}%)")

    ds = Dataset.from_list(rows)
    print(f"Dataset: {ds}")
    print(f"Columns: {ds.column_names}")

    if args.dry_run:
        print("Dry run — not pushing.")
        return

    print(f"Pushing to {args.repo_id}...")
    ds.push_to_hub(args.repo_id)
    print("Done!")


if __name__ == "__main__":
    main()
