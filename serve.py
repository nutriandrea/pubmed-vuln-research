#!/usr/bin/env python3
"""
Launch the FastAPI web server.

Usage:
    python serve.py                 # default: http://localhost:8000
    python serve.py --port 8080
    python serve.py --reload        # auto-reload on code changes (dev)
"""
import argparse
import sys
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="PubMed Analyzer Web Server")
    parser.add_argument("--host",   default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true",   help="Enable auto-reload")
    args = parser.parse_args()

    print(f"\n  PubMed Limitation Analyzer")
    print(f"  ─────────────────────────────────────────")
    print(f"  Web UI  →  http://localhost:{args.port}")
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
