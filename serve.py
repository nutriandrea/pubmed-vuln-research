#!/usr/bin/env python3
"""
Launch the FastAPI web server.

Usage:
    python serve.py                 # default: http://localhost:8000
    python serve.py --port 8080
    python serve.py --reload        # auto-reload on code changes (dev)

If you see import errors, make sure to run inside the project venv:
    source .venv/bin/activate && python serve.py
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# ── Venv guard ──────────────────────────────────────────────────────────────
# If we are NOT running from the project .venv, re-exec with the venv python.
_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON  = _PROJECT_ROOT / ".venv" / "bin" / "python"

def _inside_venv() -> bool:
    """
    Return True when the active interpreter is running inside THIS project's .venv.
    Uses sys.prefix comparison — the only reliable cross-platform method.
    """
    if sys.prefix == sys.base_prefix:
        return False  # not in any venv at all
    # Also verify it's specifically our project venv, not some other one
    return Path(sys.prefix).resolve() == (_PROJECT_ROOT / ".venv").resolve()

if not _inside_venv():
    if _VENV_PYTHON.exists():
        print(
            f"[serve.py] Detected system Python ({sys.executable}).\n"
            f"           Re-launching with project venv: {_VENV_PYTHON}\n"
        )
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
    else:
        print(
            "ERROR: .venv not found. Create it first:\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

# ── Normal startup (always inside .venv beyond this point) ──────────────────
import argparse

def _check_env() -> None:
    """Warn if OPENAI_API_KEY is not set."""
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "WARNING: OPENAI_API_KEY is not set.\n"
            "         Copy .env.example → .env and add your key before ingesting.\n",
            file=sys.stderr,
        )

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        description="PubMed Limitation Analyzer — Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host",   default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true",   help="Enable auto-reload (dev)")
    args = parser.parse_args()

    _check_env()

    print(
        f"\n  PubMed Limitation Analyzer\n"
        f"  {'─' * 41}\n"
        f"  Python   →  {sys.executable}\n"
        f"  Web UI   →  http://localhost:{args.port}\n"
        f"  API docs →  http://localhost:{args.port}/docs\n"
        f"  Press Ctrl+C to stop\n"
    )

    uvicorn.run(
        "web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
