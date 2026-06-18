from __future__ import annotations

import os
import re
import urllib.error
import urllib.request


PROJECT_NAME = "ai-relay-audit"
__version__ = "0.1.2"

DEFAULT_REMOTE_VERSION_URL = (
    "https://raw.githubusercontent.com/maya1900/ai-relay-audit/master/relay_audit/version.py"
)
VERSION_PATTERN = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)


def parse_version_text(text: str) -> str | None:
    match = VERSION_PATTERN.search(text)
    return match.group(1) if match else None


def version_tuple(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Unsupported version format: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def is_newer_version(candidate: str, current: str = __version__) -> bool:
    return version_tuple(candidate) > version_tuple(current)


def fetch_remote_version(timeout: float = 2.0, url: str | None = None) -> str | None:
    version_url = url or os.getenv("AI_RELAY_AUDIT_VERSION_URL") or DEFAULT_REMOTE_VERSION_URL
    request = urllib.request.Request(version_url, headers={"User-Agent": f"{PROJECT_NAME}/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return None
    return parse_version_text(body)
