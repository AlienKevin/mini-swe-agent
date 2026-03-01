#!/usr/bin/env python3
"""Sample 500 SWE-smith-rs tasks, excluding repos covered by SWE-bench Multilingual."""

import argparse
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare filtered SWE-smith-rs tasks")
    parser.add_argument("--output", type=str, default="data/smith_rs_filtered", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of instances to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading SWE-smith-rs dataset...")
    smith_rs = load_dataset("SWE-bench/SWE-smith-rs", split="train")
    print(f"  Loaded {len(smith_rs)} instances")

    print("Loading SWE-bench Multilingual dataset...")
    multilingual = load_dataset("SWE-bench/SWE-bench_Multilingual", split="test")
    print(f"  Loaded {len(multilingual)} instances")

    # Extract unique repo names from Multilingual, normalized to lowercase owner/repo
    multilingual_repos = set(inst["repo"].lower() for inst in multilingual)
    print(f"  {len(multilingual_repos)} unique repos in Multilingual")

    # SWE-smith-rs repos use format "swesmith/Owner__Repo.hash"
    # Multilingual repos use format "owner/repo"
    # Normalize smith repo names: "swesmith/BurntSushi__ripgrep.3b7fd442" -> "burntsushi/ripgrep"
    def smith_repo_to_canonical(repo: str) -> str:
        """Convert SWE-smith repo name to canonical owner/repo format."""
        # Remove "swesmith/" prefix
        name = repo.removeprefix("swesmith/")
        # Remove trailing hash: "BurntSushi__ripgrep.3b7fd442" -> "BurntSushi__ripgrep"
        name = name.rsplit(".", 1)[0]
        # Convert double underscore to slash: "BurntSushi__ripgrep" -> "BurntSushi/ripgrep"
        name = name.replace("__", "/", 1)
        return name.lower()

    # Filter SWE-smith-rs: exclude repos that appear in Multilingual
    filtered = smith_rs.filter(lambda x: smith_repo_to_canonical(x["repo"]) not in multilingual_repos)
    print(f"  After filtering: {len(filtered)} instances ({len(smith_rs) - len(filtered)} excluded)")

    # Show excluded repos for verification
    smith_rs_canonical = {smith_repo_to_canonical(inst["repo"]) for inst in smith_rs}
    overlap = smith_rs_canonical & multilingual_repos
    if overlap:
        print(f"  Excluded repos (overlap): {sorted(overlap)}")
    else:
        print("  No repo overlap found between SWE-smith-rs and Multilingual")

    # Sample
    if len(filtered) < args.n_samples:
        print(f"  WARNING: Only {len(filtered)} instances available, requested {args.n_samples}")
        sampled = filtered
    else:
        random.seed(args.seed)
        indices = random.sample(range(len(filtered)), args.n_samples)
        sampled = filtered.select(indices)
    print(f"  Sampled {len(sampled)} instances")

    # Verify no overlap
    sampled_canonical = {smith_repo_to_canonical(inst["repo"]) for inst in sampled}
    remaining_overlap = sampled_canonical & multilingual_repos
    assert not remaining_overlap, f"Overlap still exists: {remaining_overlap}"

    # Save as JSON so load_dataset() can load it (save_to_disk requires load_from_disk)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "train.json"
    sampled.to_json(str(json_path))
    print(f"  Saved to {json_path}")
    print(f"\nTo load: load_dataset('{output_path}', split='train')")


if __name__ == "__main__":
    main()
