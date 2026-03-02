#!/usr/bin/env python3
"""Download teacher trajectories from Docent and push to HuggingFace.

Usage:
    # Download from docent and push to HF
    python scripts/download_test_trajs.py

    # Custom collection URL and output dir
    python scripts/download_test_trajs.py \
        --collection-url https://docent.transluce.org/dashboard/7d28aece-48d0-4e07-bb03-936d982614bf/ \
        --output-dir results/sweb_multilingual_mini-v2.0.0a0_glm-5

    # Dry run (download only, don't push)
    python scripts/download_test_trajs.py --dry-run
"""

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset
from docent import Docent


DEFAULT_COLLECTION_URL = "https://docent.transluce.org/dashboard/718e213c-83f3-4265-bc7b-e71243b5f1f9/"
DEFAULT_OUTPUT_DIR = "results/sweb_multilingual_mini-v2.0.0a0_minimax-m2.5"
DEFAULT_REPO_ID = "AlienKevin/SWE-bench-multilingual-minimax-m2.5-trajectories"
DEFAULT_MODEL_NAME = "minimax-m2.5"


def extract_collection_id(url: str) -> str:
    """Extract collection ID from a docent dashboard URL."""
    match = re.search(r"dashboard/([0-9a-f-]+)", url)
    if not match:
        raise ValueError(f"Could not extract collection ID from URL: {url}")
    return match.group(1)


def download_trajectories(client: Docent, collection_id: str, output_dir: Path) -> list[Path]:
    """Download all agent runs from a docent collection and save as .traj.json files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_run_ids = client.list_agent_run_ids(collection_id)
    print(f"Found {len(agent_run_ids)} agent runs in collection {collection_id}")

    saved_paths = []
    for i, run_id in enumerate(agent_run_ids):
        agent_run = client.get_agent_run(collection_id, run_id)
        if agent_run is None:
            print(f"  [{i+1}/{len(agent_run_ids)}] Skipping {run_id} (not found)")
            continue

        # Use agent run name as instance_id, fall back to run id
        instance_id = agent_run.name or run_id

        # Convert transcripts to messages
        messages = []
        for transcript in agent_run.transcripts:
            for msg in transcript.messages:
                messages.append(msg.model_dump(mode="json"))

        # Extract resolved status from metadata.scores.resolved
        metadata = agent_run.metadata or {}
        scores = metadata.get("scores", {})
        resolved = bool(scores.get("resolved", False)) if isinstance(scores, dict) else False

        # Build trajectory data
        traj_data = {
            "instance_id": instance_id,
            "messages": messages,
            "resolved": resolved,
            "info": {
                "submission": metadata.get("submission", ""),
                **metadata,
            },
        }

        # Save to file
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        traj_path = instance_dir / f"{instance_id}.traj.json"
        traj_path.write_text(json.dumps(traj_data, indent=2))
        saved_paths.append(traj_path)

        if (i + 1) % 50 == 0 or i + 1 == len(agent_run_ids):
            print(f"  [{i+1}/{len(agent_run_ids)}] Downloaded {instance_id}")

    print(f"Saved {len(saved_paths)} trajectories to {output_dir}")
    return saved_paths


def convert_trajectory(traj_data: dict) -> list[dict]:
    """Convert a raw trajectory's messages to the HF dataset chat format."""
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

        if role == "assistant" and not entry["tool_calls"] and "```mswea_bash_command" in (content or ""):
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

        if role == "user" and content and "<returncode>" in content and converted:
            prev = converted[-1]
            if prev.get("tool_calls"):
                entry["role"] = "tool"
                entry["tool_call_id"] = prev["tool_calls"][-1]["id"]

        converted.append(entry)

    return converted


def push_to_hf(output_dir: Path, repo_id: str, model_name: str, eval_results: str | None = None):
    """Convert downloaded trajectories to HF dataset format and push."""
    # Load eval results if provided
    resolved_ids: set[str] = set()
    if eval_results:
        eval_path = Path(eval_results)
        if eval_path.suffix == ".json":
            data = json.loads(eval_path.read_text())
            if isinstance(data, dict):
                for iid, result in data.items():
                    if isinstance(result, dict) and result.get("resolved"):
                        resolved_ids.add(iid)
                    elif isinstance(result, bool) and result:
                        resolved_ids.add(iid)
            elif isinstance(data, list):
                resolved_ids = set(data)
        elif eval_path.suffix == ".txt":
            resolved_ids = set(eval_path.read_text().strip().splitlines())
        if resolved_ids:
            print(f"Loaded {len(resolved_ids)} resolved instance IDs")

    # Find all trajectory files
    traj_files = sorted(output_dir.glob("**/*.traj.json"))
    print(f"Found {len(traj_files)} trajectory files in {output_dir}")

    rows = []
    for traj_path in traj_files:
        traj_data = json.loads(traj_path.read_text())
        instance_id = traj_data.get("instance_id", traj_path.stem.replace(".traj", ""))
        info = traj_data.get("info", {})
        patch = info.get("submission", "")
        messages = convert_trajectory(traj_data)

        # Use resolved from traj data (from docent metadata.scores.resolved),
        # with eval_results file as override if provided
        if resolved_ids:
            resolved = instance_id in resolved_ids
        else:
            resolved = bool(traj_data.get("resolved", False))

        rows.append({
            "messages": messages,
            "instance_id": instance_id,
            "resolved": resolved,
            "model": model_name,
            "traj_id": instance_id,
            "patch": patch or "",
        })

    print(f"Converted {len(rows)} trajectories")
    n_resolved = sum(1 for r in rows if r["resolved"])
    print(f"  Resolved: {n_resolved}/{len(rows)} ({100*n_resolved/len(rows):.1f}%)")

    ds = Dataset.from_list(rows)
    print(f"Dataset: {ds}")

    print(f"Pushing to {repo_id}...")
    ds.push_to_hub(repo_id)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Download teacher trajectories from Docent and push to HuggingFace")
    parser.add_argument("--collection-url", type=str, default=DEFAULT_COLLECTION_URL,
                        help="Docent dashboard URL for the collection")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save downloaded trajectories")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID,
                        help="HuggingFace dataset repo ID to push to")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Model name for the dataset")
    parser.add_argument("--eval-results", type=str, default=None,
                        help="Path to evaluation results (JSON or text file)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, just push existing trajectories from output-dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download only, don't push to HuggingFace")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    collection_id = extract_collection_id(args.collection_url)

    if not args.skip_download:
        client = Docent()
        download_trajectories(client, collection_id, output_dir)

    if not args.dry_run:
        push_to_hf(output_dir, args.repo_id, args.model_name, args.eval_results)


if __name__ == "__main__":
    main()
