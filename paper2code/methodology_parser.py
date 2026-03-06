"""
methodology_parser.py - Parses methodology and experiment details from extracted paper sections.
Uses a local LLM (via llama-cpp-python or Ollama HTTP API) for semantic understanding.
Falls back to rule-based heuristics when no LLM is available.
"""

import re
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters and procedure."""
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    scheduler: str = ""
    loss_function: str = "CrossEntropyLoss"
    regularization: str = ""
    gradient_clipping: Optional[float] = None
    warmup_steps: int = 0
    weight_decay: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset requirements identified from the paper."""
    name: str = ""
    task_type: str = ""          # classification, detection, generation, etc.
    input_modality: str = ""     # image, text, audio, etc.
    output_modality: str = ""
    num_classes: Optional[int] = None
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    test_size: Optional[int] = None
    preprocessing: List[str] = field(default_factory=list)
    augmentations: List[str] = field(default_factory=list)


@dataclass
class MethodologyResult:
    """Complete parsed methodology information."""
    task_description: str = ""
    key_contributions: List[str] = field(default_factory=list)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation_metrics: List[str] = field(default_factory=list)
    baseline_methods: List[str] = field(default_factory=list)
    implementation_notes: List[str] = field(default_factory=list)
    raw_llm_output: str = ""


# ── Rule-based extraction helpers ──────────────────────────────────────────────

_LR_PATTERN = re.compile(
    r"learning[\s-]+rate[\s:=]+([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", re.I)
_BS_PATTERN = re.compile(
    r"batch[\s-]+size[\s:=]+([0-9]+)", re.I)
_EPOCH_PATTERN = re.compile(
    r"(\d+)\s+epochs?", re.I)
_OPT_PATTERN = re.compile(
    r"\b(adam(?:w)?|sgd|adagrad|rmsprop|adamax|nadam|lion)\b", re.I)
_LOSS_PATTERN = re.compile(
    r"\b(cross[\s-]?entropy|mse|mae|bce|focal[\s-]?loss|nll|ctc|dice)\b", re.I)
_METRIC_PATTERN = re.compile(
    r"\b(accuracy|f1|precision|recall|map|bleu|rouge|psnr|ssim|fid|iou|ap@)\b", re.I)
_DATASET_PATTERN = re.compile(
    r"\b(imagenet|cifar[\-\s]?(?:10|100)|mnist|coco|voc|ade20k|"
    r"squad|glue|superglue|wikitext|cc(?:12m|3m)|laion|openimages|"
    r"kinetics|ucf[\-\s]?101|hmdb)\b", re.I)

OPTIMIZER_DEFAULTS = {
    "adam": {"lr": 1e-3}, "adamw": {"lr": 1e-4, "weight_decay": 0.01},
    "sgd": {"lr": 0.1, "momentum": 0.9}, "rmsprop": {"lr": 1e-3},
}


class MethodologyParser:
    """
    Parses methodology, training procedure, and dataset requirements from paper sections.

    LLM Mode (preferred): Uses Ollama or llama-cpp-python for semantic extraction.
    Fallback Mode: Uses regex/heuristic extraction.
    """

    def __init__(self,
                 use_llm: bool = True,
                 llm_model: str = "mistral",
                 ollama_url: str = "http://localhost:11434"):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self._llm_available = self._check_llm()

    def _check_llm(self) -> bool:
        if not self.use_llm:
            return False
        try:
            import urllib.request
            url = f"{self.ollama_url}/api/tags"
            with urllib.request.urlopen(url, timeout=3):
                logger.info(f"Ollama available at {self.ollama_url}")
                return True
        except Exception:
            logger.warning("Ollama not available, using heuristic parser")
            return False

    def parse(self, methodology_text: str, experiments_text: str = "",
              abstract_text: str = "") -> MethodologyResult:
        """
        Parse methodology and experiment sections into structured data.

        Args:
            methodology_text: Text of the methodology section.
            experiments_text: Text of the experiments section.
            abstract_text: Text of the abstract (for task description).

        Returns:
            MethodologyResult with all extracted information.
        """
        combined = f"{methodology_text}\n\n{experiments_text}"

        if self._llm_available:
            result = self._parse_with_llm(combined, abstract_text)
        else:
            result = self._parse_heuristic(combined, abstract_text)

        return result

    def _parse_with_llm(self, text: str, abstract: str) -> MethodologyResult:
        """Use local LLM via Ollama to extract structured information."""
        import json
        import urllib.request
        import urllib.error

        prompt = f"""You are a research paper analyzer. Extract structured information from this paper text.

Abstract: {abstract[:500]}

Methodology and Experiments:
{text[:3000]}

Return a JSON object with these exact fields:
{{
  "task_description": "one-sentence description of the ML task",
  "key_contributions": ["contribution 1", "contribution 2"],
  "optimizer": "Adam|SGD|AdamW|etc",
  "learning_rate": 0.001,
  "batch_size": 32,
  "num_epochs": 100,
  "scheduler": "cosine|step|linear|none",
  "loss_function": "CrossEntropyLoss|MSELoss|etc",
  "weight_decay": 0.0,
  "dataset_name": "dataset name",
  "task_type": "classification|detection|segmentation|generation|etc",
  "input_modality": "image|text|audio|etc",
  "num_classes": null,
  "evaluation_metrics": ["accuracy", "F1"],
  "implementation_notes": ["note 1", "note 2"]
}}

Respond ONLY with the JSON object, no other text."""

        payload = json.dumps({
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512}
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                response_data = json.loads(resp.read())
                raw_output = response_data.get("response", "")
                return self._parse_llm_json(raw_output)
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}, falling back to heuristics")
            return self._parse_heuristic(text, abstract)

    def _parse_llm_json(self, raw_output: str) -> MethodologyResult:
        """Parse JSON output from LLM into MethodologyResult."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in LLM response")

            tc = TrainingConfig(
                optimizer=data.get("optimizer", "Adam"),
                learning_rate=float(data.get("learning_rate", 1e-3)),
                batch_size=int(data.get("batch_size", 32)),
                num_epochs=int(data.get("num_epochs", 100)),
                scheduler=data.get("scheduler", ""),
                loss_function=data.get("loss_function", "CrossEntropyLoss"),
                weight_decay=float(data.get("weight_decay", 0.0)),
            )
            dc = DatasetConfig(
                name=data.get("dataset_name", ""),
                task_type=data.get("task_type", ""),
                input_modality=data.get("input_modality", ""),
                num_classes=data.get("num_classes"),
            )
            return MethodologyResult(
                task_description=data.get("task_description", ""),
                key_contributions=data.get("key_contributions", []),
                training_config=tc,
                dataset_config=dc,
                evaluation_metrics=data.get("evaluation_metrics", []),
                implementation_notes=data.get("implementation_notes", []),
                raw_llm_output=raw_output,
            )
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return MethodologyResult(raw_llm_output=raw_output)

    def _parse_heuristic(self, text: str, abstract: str) -> MethodologyResult:
        """Rule-based fallback extraction."""
        tc = TrainingConfig()
        dc = DatasetConfig()

        # Learning rate
        lr_match = _LR_PATTERN.search(text)
        if lr_match:
            try:
                tc.learning_rate = float(lr_match.group(1))
            except ValueError:
                pass

        # Batch size
        bs_match = _BS_PATTERN.search(text)
        if bs_match:
            tc.batch_size = int(bs_match.group(1))

        # Epochs
        epoch_matches = _EPOCH_PATTERN.findall(text)
        if epoch_matches:
            tc.num_epochs = int(epoch_matches[-1])

        # Optimizer
        opt_match = _OPT_PATTERN.search(text)
        if opt_match:
            tc.optimizer = opt_match.group(1).capitalize()

        # Loss
        loss_match = _LOSS_PATTERN.search(text)
        if loss_match:
            tc.loss_function = loss_match.group(1)

        # Dataset
        ds_match = _DATASET_PATTERN.search(text)
        if ds_match:
            dc.name = ds_match.group(1).upper()

        # Metrics
        metrics = list(set(m.lower() for m in _METRIC_PATTERN.findall(text)))

        return MethodologyResult(
            task_description=abstract[:200] if abstract else "",
            training_config=tc,
            dataset_config=dc,
            evaluation_metrics=metrics,
        )
