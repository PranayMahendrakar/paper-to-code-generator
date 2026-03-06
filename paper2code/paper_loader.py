"""
paper_loader.py - Module for loading research papers from various sources.
Handles PDF files, arXiv URLs, and local file paths.
"""

import os
import re
import logging
import hashlib
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    abstract: str = ""
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    source_url: Optional[str] = None
    num_pages: int = 0


class PaperLoader:
    ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
    ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
    ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

    def __init__(self, cache_dir: str = ".paper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, source: str):
        source = source.strip()
        if os.path.exists(source):
            return self._load_local(source)
        elif self.ARXIV_ID_PATTERN.fullmatch(source.strip()):
            return self._load_arxiv(source)
        elif source.startswith(("http://", "https://")):
            return self._load_url(source)
        else:
            raise ValueError(f"Cannot determine source type for: {source!r}")

    def _load_local(self, path: str):
        pdf_path = Path(path)
        with open(pdf_path, "rb") as f:
            data = f.read()
        meta = PaperMetadata(file_path=str(pdf_path.resolve()),
                             file_hash=hashlib.sha256(data).hexdigest())
        logger.info(f"Loaded local PDF: {pdf_path} ({len(data)} bytes)")
        return data, meta

    def _load_arxiv(self, arxiv_id: str):
        clean_id = self.ARXIV_ID_PATTERN.search(arxiv_id).group(1)
        pdf_url = self.ARXIV_PDF_URL.format(arxiv_id=clean_id)
        cache_file = self.cache_dir / f"{clean_id}.pdf"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = f.read()
        else:
            data = self._download(pdf_url)
            with open(cache_file, "wb") as f:
                f.write(data)
        meta = PaperMetadata(arxiv_id=clean_id,
                             source_url=self.ARXIV_ABS_URL.format(arxiv_id=clean_id),
                             file_path=str(cache_file),
                             file_hash=hashlib.sha256(data).hexdigest())
        return data, meta

    def _load_url(self, url: str):
        match = self.ARXIV_ID_PATTERN.search(url)
        if "arxiv.org" in url and match:
            return self._load_arxiv(match.group(1))
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"url_{url_hash}.pdf"
        if not cache_file.exists():
            data = self._download(url)
            with open(cache_file, "wb") as f:
                f.write(data)
        with open(cache_file, "rb") as f:
            data = f.read()
        return data, PaperMetadata(source_url=url, file_path=str(cache_file),
                                   file_hash=hashlib.sha256(data).hexdigest())

    @staticmethod
    def _download(url: str, timeout: int = 60) -> bytes:
        req = urllib.request.Request(url,
              headers={"User-Agent": "paper2code/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

