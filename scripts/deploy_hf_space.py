"""Upload the app to a Hugging Face Space in one commit (the Space then rebuilds itself).

Usage:  uv run python scripts/deploy_hf_space.py --space sawcheet/biomed-rag
Needs a Hugging Face token with WRITE access (uv run hf auth login).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

# Exactly what the Dockerfile needs. The Space gets its own README (with Space settings).
FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock"]
FOLDERS = ["src", "data/indexes/dense_bge-small_abstract"]
SPACE_README = "deployment/huggingface/README.md"


def collect(root: Path) -> list[CommitOperationAdd]:
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(root / SPACE_README))]
    paths = [root / f for f in FILES]
    for folder in FOLDERS:
        paths += [p for p in (root / folder).rglob("*") if p.is_file()]
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        repo_path = path.relative_to(root).as_posix()
        ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(path)))
    return ops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True, help="e.g. sawcheet/biomed-rag")
    parser.add_argument("--message", default="Deploy from GitHub repo")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    missing = [f for f in [*FILES, *FOLDERS, SPACE_README] if not (root / f).exists()]
    if missing:
        raise SystemExit(f"missing: {missing} (build the dense index first?)")

    ops = collect(root)
    for op in ops:
        print("  +", op.path_in_repo)
    api = HfApi()
    print(f"logged in as: {api.whoami()['name']}")
    info = api.create_commit(
        repo_id=args.space, repo_type="space", operations=ops, commit_message=args.message
    )
    print(f"uploaded {len(ops)} files: {info.commit_url}")
    print(f"build logs: https://huggingface.co/spaces/{args.space}?logs=build")


if __name__ == "__main__":
    main()
