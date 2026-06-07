#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


VERSION_FILE = Path("relay_audit/version.py")
VERSION_PATTERN = re.compile(r"^__version__\s*=\s*[\"'](\d+)\.(\d+)\.(\d+)[\"']", re.MULTILINE)
ZERO_SHA = "0" * 40


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], check=check, text=True, capture_output=True)


def repo_root() -> Path:
    result = run_git(["rev-parse", "--show-toplevel"])
    return Path(result.stdout.strip())


def current_branch() -> str | None:
    result = run_git(["symbolic-ref", "--quiet", "--short", "HEAD"], check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def parse_push_updates(stdin_text: str) -> list[tuple[str, str, str, str]]:
    updates = []
    for line in stdin_text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            updates.append((parts[0], parts[1], parts[2], parts[3]))
    return updates


def current_branch_update(updates: list[tuple[str, str, str, str]]) -> tuple[str, str, str, str] | None:
    branch = current_branch()
    if not branch:
        return None
    local_ref = f"refs/heads/{branch}"
    head = run_git(["rev-parse", "HEAD"]).stdout.strip()
    for update in updates:
        update_local_ref, update_local_sha, remote_ref, _remote_sha = update
        if update_local_ref == local_ref and update_local_sha == head and remote_ref.startswith("refs/heads/"):
            return update
    return None


def version_file_is_clean() -> bool:
    unstaged = run_git(["diff", "--quiet", "--", str(VERSION_FILE)], check=False).returncode == 0
    staged = run_git(["diff", "--cached", "--quiet", "--", str(VERSION_FILE)], check=False).returncode == 0
    return unstaged and staged


def bump_patch_version(root: Path) -> str:
    path = root / VERSION_FILE
    text = path.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(text)
    if not match:
        raise RuntimeError(f"Cannot find semantic __version__ in {VERSION_FILE}")
    major, minor, patch = (int(part) for part in match.groups())
    next_version = f"{major}.{minor}.{patch + 1}"
    updated = VERSION_PATTERN.sub(f'__version__ = "{next_version}"', text, count=1)
    path.write_text(updated, encoding="utf-8")
    return next_version


def create_version_commit(version: str) -> None:
    run_git(["add", str(VERSION_FILE)])
    run_git(["commit", "--only", str(VERSION_FILE), "-m", f"chore: bump version to {version}"])


def pre_push(remote: str, remote_url: str, stdin_text: str) -> int:
    if os.getenv("AI_RELAY_AUDIT_SKIP_VERSION_BUMP") == "1":
        return 0

    updates = parse_push_updates(stdin_text)
    update = current_branch_update(updates)
    if update is None:
        return 0

    _local_ref, _local_sha, remote_ref, remote_sha = update
    if not version_file_is_clean():
        print(f"Version file has uncommitted changes: {VERSION_FILE}", file=sys.stderr)
        print("Commit or discard those changes before pushing.", file=sys.stderr)
        return 1

    root = repo_root()
    version = bump_patch_version(root)
    create_version_commit(version)

    target = f"HEAD:{remote_ref}"
    print(f"Auto bumped {VERSION_FILE} to v{version}.", file=sys.stderr)
    print(f"Pushing version commit to {remote} ({remote_url})...", file=sys.stderr)
    push_args = ["push", "--no-verify", remote, target]
    if remote_sha == ZERO_SHA:
        push_args.insert(1, "--set-upstream")
    push = run_git(push_args, check=False)
    if push.stdout:
        print(push.stdout, end="", file=sys.stderr)
    if push.stderr:
        print(push.stderr, end="", file=sys.stderr)
    if push.returncode != 0:
        return push.returncode

    print("Version bump commit was pushed. Stopping the original push because Git computed refs before this hook.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Bump ai-relay-audit patch version.")
    parser.add_argument("--pre-push", action="store_true", help="Run as a git pre-push hook.")
    parser.add_argument("remote", nargs="?", default="origin")
    parser.add_argument("remote_url", nargs="?", default="")
    args = parser.parse_args()

    if args.pre_push:
        return pre_push(args.remote, args.remote_url, sys.stdin.read())

    root = repo_root()
    if not version_file_is_clean():
        print(f"Version file has uncommitted changes: {VERSION_FILE}", file=sys.stderr)
        return 1
    version = bump_patch_version(root)
    create_version_commit(version)
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
