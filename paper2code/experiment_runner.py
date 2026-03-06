"""
experiment_runner.py - Orchestrates the full paper-to-code pipeline.
Runs sample paper analyses and generates implementation reports.
"""

import os
import json
import time
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from .paper_loader import PaperLoader, PaperMetadata
from .pdf_text_extractor import PDFTextExtractor, PaperSections
from .methodology_parser import MethodologyParser, MethodologyResult
from .architecture_interpreter import ArchitectureInterpreter, ArchitectureSpec
from .code_generator import CodeGenerator, GeneratedCode

logger = logging.getLogger(__name__)


@dataclass
class ExperimentReport:
    """Full report of a paper-to-code experiment run."""
    paper_source: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str = ""
    paper_title: str = ""
    arxiv_id: str = ""
    num_pages: int = 0
    sections_found: List[str] = field(default_factory=list)
    methodology_summary: str = ""
    architecture_type: str = ""
    model_name: str = ""
    estimated_params: Optional[int] = None
    optimizer: str = ""
    learning_rate: float = 0.0
    batch_size: int = 0
    num_epochs: int = 0
    dataset_name: str = ""
    task_type: str = ""
    output_dir: str = ""
    files_generated: List[str] = field(default_factory=list)
    total_code_lines: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"# Paper-to-Code Implementation Report",
            f"",
            f"**Status**: {status}",
            f"**Paper**: {self.paper_source}",
            f"**Generated**: {self.timestamp}",
            f"**Duration**: {self.duration_seconds:.1f}s",
            f"",
            f"## Architecture Analysis",
            f"- **Architecture Type**: {self.architecture_type}",
            f"- **Model Name**: {self.model_name or chr(39)Unknown{chr(39)}}",
            f"",
            f"## Training Configuration",
            f"- Optimizer: {self.optimizer}",
            f"- Learning Rate: {self.learning_rate}",
            f"- Batch Size: {self.batch_size}",
            f"- Epochs: {self.num_epochs}",
        ]
        if self.error_message:
            lines.append(f"## Error")
            lines.append(self.error_message)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class ExperimentRunner:
    """
    Orchestrates the complete paper-to-code pipeline.

    Pipeline stages:
    1. paper_loader          - Load PDF from local file, arXiv, or URL
    2. pdf_text_extractor    - Extract and segment text into sections
    3. methodology_parser    - Parse training procedure and dataset requirements
    4. architecture_interpreter - Build structured ArchitectureSpec
    5. code_generator        - Generate runnable PyTorch implementation
    """

    def __init__(self, output_dir="outputs", cache_dir=".paper_cache",
                 use_llm=True, llm_model="mistral",
                 ollama_url="http://localhost:11434"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = PaperLoader(cache_dir=cache_dir)
        self.extractor = PDFTextExtractor()
        self.methodology_parser = MethodologyParser(
            use_llm=use_llm, llm_model=llm_model, ollama_url=ollama_url)
        self.arch_interpreter = ArchitectureInterpreter(
            use_llm=use_llm, llm_model=llm_model, ollama_url=ollama_url)
        self.code_gen = CodeGenerator(
            use_llm=use_llm, llm_model=llm_model, ollama_url=ollama_url)
        logger.info(f"ExperimentRunner initialized (LLM={use_llm})")

    def run(self, paper_source: str) -> ExperimentReport:
        """Run the full paper-to-code pipeline on a paper."""
        report = ExperimentReport(
            paper_source=paper_source,
            timestamp=datetime.datetime.now().isoformat(),
        )
        start_time = time.time()
        try:
            report = self._run_pipeline(paper_source, report)
            report.success = True
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            report.error_message = str(e)
            report.success = False
        finally:
            report.duration_seconds = time.time() - start_time
        self._save_report(report)
        return report

    def _run_pipeline(self, paper_source, report):
        # Stage 1: Load
        logger.info(f"[1/5] Loading: {paper_source}")
        pdf_bytes, meta = self.loader.load(paper_source)
        report.arxiv_id = meta.arxiv_id or ""

        # Stage 2: Extract text
        logger.info("[2/5] Extracting text")
        sections = self.extractor.extract(pdf_bytes)
        report.sections_found = sections.non_empty_sections()
        if not sections.methodology and not sections.architecture:
            report.warnings.append("No methodology/architecture sections found")

        # Stage 3: Parse methodology
        logger.info("[3/5] Parsing methodology")
        methodology = self.methodology_parser.parse(
            methodology_text=sections.methodology or sections.raw_text[:3000],
            experiments_text=sections.experiments,
            abstract_text=sections.abstract,
        )
        report.methodology_summary = methodology.task_description
        report.optimizer = methodology.training_config.optimizer
        report.learning_rate = methodology.training_config.learning_rate
        report.batch_size = methodology.training_config.batch_size
        report.num_epochs = methodology.training_config.num_epochs
        report.dataset_name = methodology.dataset_config.name
        report.task_type = methodology.dataset_config.task_type

        # Stage 4: Interpret architecture
        logger.info("[4/5] Interpreting architecture")
        arch_spec = self.arch_interpreter.interpret(
            arch_text=sections.architecture or sections.methodology,
            methodology_text=sections.methodology,
            abstract_text=sections.abstract,
        )
        report.architecture_type = arch_spec.arch_type.value
        report.model_name = arch_spec.model_name
        report.estimated_params = arch_spec.num_parameters_estimated

        # Save arch spec
        arch_dir = self.output_dir / "architecture"
        arch_dir.mkdir(exist_ok=True)
        safe = paper_source.replace("/","_").replace(":","_")[:50]
        (arch_dir / f"{safe}_arch.json").write_text(
            json.dumps(arch_spec.to_dict(), indent=2))

        # Stage 5: Generate code
        logger.info("[5/5] Generating code")
        code_dir = self.output_dir / "implementations" / safe
        generated = self.code_gen.generate(arch_spec, methodology)
        generated.save(str(code_dir))
        report.output_dir = str(code_dir)
        if generated.output_dir:
            p = Path(generated.output_dir)
            report.files_generated = [f.name for f in p.glob("*.*")]
            report.total_code_lines = sum(
                len(f.read_text().splitlines()) for f in p.glob("*.py"))
        return report

    def _save_report(self, report):
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        safe = report.paper_source.replace("/","_").replace(":","_")[:50]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        (reports_dir / f"{safe}_{ts}.md").write_text(report.to_markdown())
        (reports_dir / f"{safe}_{ts}.json").write_text(report.to_json())
        logger.info(f"Report saved to {reports_dir}")

    def run_batch(self, paper_sources):
        """Run pipeline on multiple papers."""
        return [self.run(src) for src in paper_sources]

    def generate_summary(self, reports):
        """Generate summary markdown for multiple reports."""
        ok = [r for r in reports if r.success]
        lines = [
            "# Paper-to-Code Batch Summary",
            f"**Total**: {len(reports)} | **OK**: {len(ok)} | **Failed**: {len(reports)-len(ok)}",
            "",
            "| Paper | Status | Arch | Params |",
            "|-------|--------|------|--------|",
        ]
        for r in reports:
            s = "OK" if r.success else "FAIL"
            p = f"{r.estimated_params:,}" if r.estimated_params else "N/A"
            lines.append(f"| {r.paper_source} | {s} | {r.architecture_type} | {p} |")
        return "\n".join(lines)
