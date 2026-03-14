#!/usr/bin/env python3
"""
Launch the FastAPI web server for PubMed Limitation Analyzer.

Usage:
    python serve.py                 # default: http://localhost:8000
    python serve.py --port 8080
"""

import os
import sys
from pathlib import Path

# ── Venv guard ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"

def _inside_venv() -> bool:
    """Check if running inside project venv."""
    if sys.prefix == sys.base_prefix:
        return False
    return Path(sys.prefix).resolve() == (_PROJECT_ROOT / ".venv").resolve()

if not _inside_venv():
    if _VENV_PYTHON.exists():
        print(f"[serve.py] Using project venv: {_VENV_PYTHON}")
        os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)
    else:
        print("ERROR: .venv not found. Create it first:\n"
              "  python3 -m venv .venv && source .venv/bin/activate\n"
              "  pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

# ── Normal startup ──────────────────────────────────────────────────────────
import argparse
from dotenv import load_dotenv
import uvicorn

def _check_env() -> None:
    """Check environment variables."""
    load_dotenv(_PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set in .env", file=sys.stderr)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PubMed Limitation Analyzer - Web Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload")
    args = parser.parse_args()

    _check_env()

    print(f"\n  📚 PubMed Limitation Analyzer")
    print(f"  {'─' * 40}")
    print(f"  Web UI   →  http://localhost:{args.port}")
    print(f"  API docs →  http://localhost:{args.port}/docs")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(
        "web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()
