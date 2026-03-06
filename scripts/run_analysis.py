#!/usr/bin/env python3
"""
scripts/run_analysis.py - CLI entry point for the paper-to-code pipeline.
Used by GitHub Actions workflow for automated paper analysis.

Usage:
    python scripts/run_analysis.py --paper 1706.03762 --output_dir outputs/
    python scripts/run_analysis.py --paper paper.pdf --output_dir outputs/ --no_llm
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_analysis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paper-to-code pipeline on a research paper"
    )
    parser.add_argument("--paper", required=True,
        help="arXiv ID (e.g. 1706.03762), PDF path, or URL")
    parser.add_argument("--output_dir", default="outputs",
        help="Directory for generated code and reports")
    parser.add_argument("--cache_dir", default=".paper_cache",
        help="Cache directory for downloaded PDFs")
    parser.add_argument("--no_llm", action="store_true",
        help="Disable LLM, use heuristic mode only")
    parser.add_argument("--llm_model", default="mistral",
        help="Ollama model name")
    parser.add_argument("--ollama_url", default="http://localhost:11434",
        help="Ollama API URL")
    parser.add_argument("--report_name", default=None,
        help="Custom name for the report file")
    parser.add_argument("--batch", nargs="+", metavar="PAPER",
        help="Run on multiple papers (space-separated arXiv IDs)")
    return parser.parse_args()


def main():
    args = parse_args()

    from paper2code import ExperimentRunner

    runner = ExperimentRunner(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
    )

    if args.batch:
        # Batch mode: run on multiple papers
        papers = [args.paper] + args.batch
        logger.info(f"Batch mode: {len(papers)} papers")
        reports = runner.run_batch(papers)
        summary = runner.generate_summary(reports)
        summary_path = Path(args.output_dir) / "reports" / "batch_summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary)
        logger.info(f"Batch summary: {summary_path}")
        failed = [r for r in reports if not r.success]
        if failed:
            logger.warning(f"{len(failed)}/{len(reports)} papers failed")
            sys.exit(1)
    else:
        # Single paper mode
        logger.info(f"Analysing: {args.paper}")
        report = runner.run(args.paper)

        if report.success:
            logger.info(f"SUCCESS - Output: {report.output_dir}")
            logger.info(f"Architecture: {report.architecture_type}")
            logger.info(f"Model: {report.model_name}")
            if report.estimated_params:
                logger.info(f"Est. params: {report.estimated_params:,}")
            logger.info(f"Files: {len(report.files_generated)}")
            logger.info(f"Code lines: {report.total_code_lines:,}")
            print("\n" + report.to_markdown())
        else:
            logger.error(f"FAILED: {report.error_message}")
            print(report.to_markdown())
            sys.exit(1)


if __name__ == "__main__":
    main()
