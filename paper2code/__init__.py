"""
paper2code - Research Paper to PyTorch Code Generator

A modular pipeline for converting research papers (PDF) into
working PyTorch implementations using local open-source LLMs.

Modules:
    paper_loader           - Load PDFs from local files, arXiv, or URLs
    pdf_text_extractor     - Extract and segment PDF text into sections
    methodology_parser     - Parse training procedure and dataset requirements
    architecture_interpreter - Interpret neural network architecture
    code_generator         - Generate runnable PyTorch code
    experiment_runner      - Orchestrate the full pipeline

Usage:
    from paper2code import ExperimentRunner
    runner = ExperimentRunner(output_dir="outputs")
    report = runner.run("2106.09685")  # arXiv ID
    print(report.to_markdown())
"""

__version__ = "1.0.0"
__author__ = "paper2code"
__license__ = "MIT"

from .paper_loader import PaperLoader, PaperMetadata
from .pdf_text_extractor import PDFTextExtractor, PaperSections
from .methodology_parser import MethodologyParser, MethodologyResult
from .architecture_interpreter import ArchitectureInterpreter, ArchitectureSpec, ArchType
from .code_generator import CodeGenerator, GeneratedCode
from .experiment_runner import ExperimentRunner, ExperimentReport

__all__ = [
    "PaperLoader",
    "PaperMetadata",
    "PDFTextExtractor",
    "PaperSections",
    "MethodologyParser",
    "MethodologyResult",
    "ArchitectureInterpreter",
    "ArchitectureSpec",
    "ArchType",
    "CodeGenerator",
    "GeneratedCode",
    "ExperimentRunner",
    "ExperimentReport",
]
