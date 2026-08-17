"""Create/update a Hugging Face Docker Space running modelmap and watch it come up.

  uv run python scripts/deploy_space.py [--repo USER/modelmap] [--public] [--no-wait]

Needs a *write* token (`hf auth login`). Uploads the working tree (minus build
artifacts), sets the Space variables the container expects, then polls the
runtime until it is RUNNING and prints the URL. Idempotent: rerun to redeploy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

IGNORE = [
    ".git/*", ".venv/*", "web/node_modules/*", "web/dist/*", "src/modelmap/web/*", "dist/*",
    "tests/e2e/shots/*", "**/__pycache__/*", "*.graph.json", ".modelmap-cache/*", "docs/*",
    ".github/*", "compose.yaml", "README.md",  # the Space gets its own README (front-matter)
]

VARIABLES = {
    "MODELMAP_WARM": "1",          # pre-extract the gallery on every (re)start
    "MODELMAP_TRUST_PROXY": "1",   # HF's proxy sets X-Forwarded-For
    "MODELMAP_WORKERS": "2",       # free tier: 2 vCPU / 16 GB
    "MODELMAP_MAX_INFLIGHT": "6",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="Space id, default <you>/modelmap")
    ap.add_argument("--public", action="store_true", help="create as public (default: private)")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    me = api.whoami()
    role = (me.get("auth") or {}).get("accessToken", {}).get("role")
    if role == "read":
        sys.exit("your HF token is read-only; run `hf auth login` with a write token first")
    repo = a.repo or f"{me['name']}/modelmap"

    created = not api.repo_exists(repo, repo_type="space")
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=not a.public, exist_ok=True)
    print(("created" if created else "updating"), f"https://huggingface.co/spaces/{repo}")

    for k, v in VARIABLES.items():
        api.add_space_variable(repo, k, v)

    # Space README (front-matter) first, then the tree
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy/hf-space/README.md"), path_in_repo="README.md",
        repo_id=repo, repo_type="space", commit_message="space metadata",
    )
    api.upload_folder(
        folder_path=str(ROOT), repo_id=repo, repo_type="space", ignore_patterns=IGNORE,
        commit_message="deploy modelmap",
    )
    print("uploaded; the Space is building the Docker image (node + python, a few minutes)")
    if a.no_wait:
        return

    url = f"https://{repo.replace('/', '-').replace('_', '-').lower()}.hf.space"
    last = None
    for _ in range(240):  # up to ~20 min
        rt = api.get_space_runtime(repo)
        if rt.stage != last:
            print(f"  {time.strftime('%H:%M:%S')}  {rt.stage}" + (f"  ({rt.hardware})" if rt.hardware else ""))
            last = rt.stage
        if rt.stage == "RUNNING":
            print(f"\nrunning: {url}\n(private Space: open it while logged in to huggingface.co)")
            return
        if rt.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            sys.exit(f"space failed: {rt.stage} — check the Logs tab on the Space page")
        time.sleep(5)
    print("still building; check the Space page")


if __name__ == "__main__":
    main()
