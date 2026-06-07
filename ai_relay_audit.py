#!/usr/bin/env python3
"""Compatibility entrypoint for AI Relay Audit."""

from relay_audit import *  # noqa: F403
from relay_audit.runner import main
from relay_audit.version import __version__ as __version__


if __name__ == "__main__":
    raise SystemExit(main())
