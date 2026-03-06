"""
pdf_text_extractor.py - Extracts structured text from research paper PDFs.
Parses sections: Abstract, Introduction, Methodology, Architecture, Experiments.
Uses pdfplumber (preferred) with PyPDF2 fallback.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    pdfplumber = None
    PDF_BACKEND = None

try:
    import PyPDF2
    if PDF_BACKEND is None:
        PDF_BACKEND = "PyPDF2"
except ImportError:
    PyPDF2 = None

if PDF_BACKEND is None:
    raise ImportError("Install pdfplumber or PyPDF2: pip install pdfplumber PyPDF2")

logger.info(f"PDF backend: {PDF_BACKEND}")


@dataclass
class PaperSections:
    """Structured sections extracted from a research paper."""
    raw_text: str = ""
    abstract: str = ""
    introduction: str = ""
    related_work: str = ""
    methodology: str = ""
    architecture: str = ""
    experiments: str = ""
    results: str = ""
    conclusion: str = ""
    references: str = ""
    other: Dict[str, str] = field(default_factory=dict)

    def get_primary_sections(self) -> Dict[str, str]:
        """Return only the most important sections for code generation."""
        return {
            "abstract": self.abstract,
            "methodology": self.methodology,
            "architecture": self.architecture,
            "experiments": self.experiments,
        }

    def non_empty_sections(self) -> List[str]:
        return [k for k, v in vars(self).items()
                if isinstance(v, str) and v.strip() and k != "raw_text"]


SECTION_PATTERNS = {
    "abstract": re.compile(r"^abstract\b", re.I),
    "introduction": re.compile(r"^(1\.?\s+)?introduction\b", re.I),
    "related_work": re.compile(r"^(\d\.?\s+)?(related\s+work|background|prior\s+work)\b", re.I),
    "methodology": re.compile(r"^(\d\.?\s+)?(method(ology)?|approach|proposed\s+method|our\s+method)\b", re.I),
    "architecture": re.compile(r"^(\d\.?\s+)?(model\s+architecture|network\s+architecture|architecture|model\s+design)\b", re.I),
    "experiments": re.compile(r"^(\d\.?\s+)?(experiment(s|al\s+setup)?|evaluation|empirical)\b", re.I),
    "results": re.compile(r"^(\d\.?\s+)?(results?|performance|analysis|ablation)\b", re.I),
    "conclusion": re.compile(r"^(\d\.?\s+)?(conclusion|summary|future\s+work)\b", re.I),
    "references": re.compile(r"^references?\b", re.I),
}


class PDFTextExtractor:
    """
    Extracts and segments text from a research paper PDF.

    Usage:
        extractor = PDFTextExtractor()
        sections = extractor.extract(pdf_bytes)
        print(sections.methodology)
    """

    def __init__(self, min_section_length: int = 100):
        self.min_section_length = min_section_length

    def extract(self, pdf_bytes: bytes) -> PaperSections:
        """Extract text from PDF bytes and parse into sections."""
        raw_text = self._extract_raw_text(pdf_bytes)
        sections = self._parse_sections(raw_text)
        sections.raw_text = raw_text
        return sections

    def _extract_raw_text(self, pdf_bytes: bytes) -> str:
        if PDF_BACKEND == "pdfplumber":
            return self._extract_pdfplumber(pdf_bytes)
        return self._extract_pypdf2(pdf_bytes)

    def _extract_pdfplumber(self, pdf_bytes: bytes) -> str:
        import io
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    def _extract_pypdf2(self, pdf_bytes: bytes) -> str:
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    def _parse_sections(self, text: str) -> PaperSections:
        sections = PaperSections()
        lines = text.split("\n")
        current_section = "other"
        current_buffer: List[str] = []
        section_map: Dict[str, List[str]] = {k: [] for k in SECTION_PATTERNS}
        section_map["other"] = []

        for line in lines:
            stripped = line.strip()
            matched_section = self._match_section(stripped)
            if matched_section:
                # Save previous section
                self._flush_buffer(section_map, current_section, current_buffer)
                current_buffer = []
                current_section = matched_section
            else:
                current_buffer.append(line)

        self._flush_buffer(section_map, current_section, current_buffer)

        # Assign to dataclass
        for sec_name, lines_list in section_map.items():
            content = "\n".join(lines_list).strip()
            if sec_name in vars(sections) and sec_name != "raw_text":
                setattr(sections, sec_name, content)

        return sections

    def _match_section(self, line: str) -> Optional[str]:
        if not line or len(line) > 120:
            return None
        for sec_name, pattern in SECTION_PATTERNS.items():
            if pattern.match(line):
                return sec_name
        return None

    @staticmethod
    def _flush_buffer(section_map: Dict, section: str, buffer: List[str]) -> None:
        content = "\n".join(buffer).strip()
        if section in section_map:
            section_map[section].extend(buffer)
        else:
            section_map.setdefault(section, []).extend(buffer)
