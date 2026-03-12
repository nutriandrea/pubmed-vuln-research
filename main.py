#!/usr/bin/env python3
"""
PubMed Research Limitation Analyzer — CLI Entry Point

Usage examples:

  # Ingest then get full report
  python main.py ingest --topic "breast cancer detection" \
                        --date-from 2020 --date-to 2025 \
                        --paper-type review --max-papers 10

  # Ask a specific question (requires prior ingest)
  python main.py ask --topic "breast cancer detection" \
                     --question "What are the main dataset limitations?"

  # Full pipeline: ingest + synthesize report
  python main.py report --topic "deep learning MRI" \
                        --date-from 2021 --date-to 2025 \
                        --max-papers 15

All output is printed to stdout. Use shell redirection to save to a file.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track

from config.settings import settings
from src.orchestrator import ResearchLimitationAnalyzer
from src.logger import logger

console = Console()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _print_md(text: str, title: str = "") -> None:
    """Render Markdown text via Rich."""
    if title:
        console.print(Panel(Markdown(text), title=title, border_style="cyan"))
    else:
        console.print(Markdown(text))


def _build_analyzer() -> ResearchLimitationAnalyzer:
    return ResearchLimitationAnalyzer(
        model_name=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        top_k=settings.retrieval_top_k,
    )


# ------------------------------------------------------------------ #
# Sub-commands
# ------------------------------------------------------------------ #

def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest papers and display a short preview of extracted limitations."""
    analyzer = _build_analyzer()
    console.rule(f"[bold cyan]Ingesting: {args.topic}")
    n = analyzer.ingest(
        topic=args.topic,
        date_from=args.date_from,
        date_to=args.date_to,
        paper_type=args.paper_type,
        max_papers=args.max_papers,
    )
    console.print(f"\n[green]Done.[/green] {n} chunks indexed in vector store.")
    if args.preview:
        console.rule("Preview: synthesized limitations")
        report = analyzer.synthesize()
        _print_md(report, title=f"Limitations — {args.topic}")


def cmd_ask(args: argparse.Namespace) -> None:
    """Ingest (if needed) then answer a specific question."""
    analyzer = _build_analyzer()
    console.rule(f"[bold cyan]Ingesting: {args.topic}")
    analyzer.ingest(
        topic=args.topic,
        date_from=args.date_from,
        date_to=args.date_to,
        paper_type=args.paper_type,
        max_papers=args.max_papers,
    )
    console.rule(f"[bold yellow]Question: {args.question}")
    result = analyzer.ask_with_sources(args.question)

    _print_md(result["answer"], title="Answer")

    if result["sources"]:
        console.rule("Sources")
        for i, src in enumerate(result["sources"], 1):
            console.print(
                f"  [dim]{i}.[/dim] [bold]{src['paper_title']}[/bold] "
                f"({src['year']}) — PMID:{src['pmid']} "
                f"[{src.get('category', '')}]"
            )
            if src.get("pubmed_url"):
                console.print(f"     {src['pubmed_url']}")


def cmd_report(args: argparse.Namespace) -> None:
    """Full pipeline: ingest + generate structured limitations report."""
    analyzer = _build_analyzer()
    console.rule(f"[bold cyan]Full pipeline: {args.topic}")
    n = analyzer.ingest(
        topic=args.topic,
        date_from=args.date_from,
        date_to=args.date_to,
        paper_type=args.paper_type,
        max_papers=args.max_papers,
    )
    console.print(f"\n{n} chunks indexed. Generating report...\n")
    report = analyzer.synthesize()
    _print_md(report, title=f"Research Limitations Report — {args.topic}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"# Research Limitations Report\n\n**Topic:** {args.topic}\n\n")
            f.write(report)
        console.print(f"\n[green]Report saved to:[/green] {args.output}")


# ------------------------------------------------------------------ #
# Argument parser
# ------------------------------------------------------------------ #

def _common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across sub-commands."""
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--date-from", type=int, default=2020, metavar="YEAR")
    parser.add_argument("--date-to", type=int, default=2025, metavar="YEAR")
    parser.add_argument(
        "--paper-type",
        default=None,
        choices=[
            "review", "clinical_trial", "meta_analysis",
            "systematic_review", "case_report", "randomized_controlled_trial",
        ],
        help="Filter by PubMed publication type",
    )
    parser.add_argument(
        "--max-papers", type=int, default=10, metavar="N",
        help="Maximum papers to retrieve (default: 10)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pubmed-rag",
        description="Research Limitation Analyzer powered by PubMed + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest papers into vector store")
    _common_args(p_ingest)
    p_ingest.add_argument(
        "--preview", action="store_true",
        help="Print a quick synthesis report after ingestion"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # --- ask ---
    p_ask = sub.add_parser("ask", help="Ingest + answer a specific question")
    _common_args(p_ask)
    p_ask.add_argument("--question", required=True, help="Question to answer")
    p_ask.set_defaults(func=cmd_ask)

    # --- report ---
    p_report = sub.add_parser("report", help="Ingest + generate full limitations report")
    _common_args(p_report)
    p_report.add_argument(
        "--output", default=None, metavar="FILE",
        help="Save Markdown report to this file"
    )
    p_report.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
