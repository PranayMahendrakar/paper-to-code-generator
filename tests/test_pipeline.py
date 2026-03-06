"""
tests/test_pipeline.py - Unit tests for the paper2code pipeline modules.
Tests run in heuristic mode (no LLM required) using synthetic text inputs.
"""

import io
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Test data ─────────────────────────────────────────────────────────────

TRANSFORMER_TEXT = """
Abstract
We propose the Transformer, a model architecture based solely on attention mechanisms.

Methodology
We use multi-head self-attention with 8 heads and a model dimension of 512.
The feed-forward network has an inner dimension of 2048.
We train with the Adam optimizer with beta1=0.9, beta2=0.98 and epsilon=10^-9.
We used a batch size of 25000 tokens. We trained for 100000 steps.

Model Architecture
The encoder is composed of a stack of N=6 identical layers.
Each layer has two sub-layers: multi-head self-attention and feed-forward network.
We employ residual connections and layer normalization.

Experiments
We trained on the WMT 2014 English-German dataset.
We evaluate using BLEU score.
"""

CNN_TEXT = """
Abstract
We present a deep residual learning framework for image recognition.

Methodology
We use SGD with momentum 0.9, learning rate 0.1, batch size 256, and 90 epochs.
We apply batch normalization after each convolutional layer.

Model Architecture
Our network has 50 convolutional layers with residual connections.
We use 3x3 convolution kernels throughout.
Input channels are 3 (RGB images), initial output channels 64.

Experiments
We train on ImageNet LSVRC-2012 with 1.28 million images and 1000 classes.
We report top-1 and top-5 accuracy.
"""


# ── PDF Text Extractor ────────────────────────────────────────────────────

class TestPDFTextExtractor:
    def setup_method(self):
        from paper2code.pdf_text_extractor import PDFTextExtractor
        self.extractor = PDFTextExtractor()

    def _make_fake_sections(self, text):
        """Use internal parser directly to avoid needing a real PDF."""
        return self.extractor._parse_sections(text)

    def test_parse_transformer_sections(self):
        sections = self._make_fake_sections(TRANSFORMER_TEXT)
        assert sections.abstract != "", "Abstract should be extracted"
        assert sections.methodology != "", "Methodology should be extracted"
        assert sections.architecture != "", "Architecture should be extracted"
        assert sections.experiments != "", "Experiments should be extracted"

    def test_parse_cnn_sections(self):
        sections = self._make_fake_sections(CNN_TEXT)
        assert "residual" in sections.abstract.lower() or sections.abstract != ""

    def test_non_empty_sections(self):
        sections = self._make_fake_sections(TRANSFORMER_TEXT)
        non_empty = sections.non_empty_sections()
        assert len(non_empty) >= 3


# ── Methodology Parser ───────────────────────────────────────────────────

class TestMethodologyParser:
    def setup_method(self):
        from paper2code.methodology_parser import MethodologyParser
        self.parser = MethodologyParser(use_llm=False)

    def test_transformer_training_config(self):
        result = self.parser.parse(TRANSFORMER_TEXT)
        tc = result.training_config
        assert tc.optimizer.lower() in ("adam", "adamw"), f"Expected Adam, got {tc.optimizer}"

    def test_cnn_training_config(self):
        result = self.parser.parse(CNN_TEXT)
        tc = result.training_config
        assert tc.optimizer.lower() in ("sgd",), f"Expected SGD, got {tc.optimizer}"
        assert tc.batch_size == 256, f"Expected batch_size=256, got {tc.batch_size}"
        assert tc.num_epochs == 90, f"Expected 90 epochs, got {tc.num_epochs}"

    def test_dataset_detection(self):
        result = self.parser.parse(CNN_TEXT)
        assert "imagenet" in result.dataset_config.name.lower()

    def test_metrics_extraction(self):
        result = self.parser.parse(CNN_TEXT)
        assert len(result.evaluation_metrics) >= 1


# ── Architecture Interpreter ──────────────────────────────────────────────

class TestArchitectureInterpreter:
    def setup_method(self):
        from paper2code.architecture_interpreter import ArchitectureInterpreter, ArchType
        self.interpreter = ArchitectureInterpreter(use_llm=False)
        self.ArchType = ArchType

    def test_transformer_detection(self):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        assert spec.arch_type == self.ArchType.TRANSFORMER, \
            f"Expected TRANSFORMER, got {spec.arch_type}"

    def test_cnn_detection(self):
        spec = self.interpreter.interpret(CNN_TEXT)
        assert spec.arch_type == self.ArchType.CNN, \
            f"Expected CNN, got {spec.arch_type}"

    def test_transformer_dimensions(self):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        assert spec.hidden_dim == 512, f"Expected 512, got {spec.hidden_dim}"
        assert spec.num_heads == 8, f"Expected 8, got {spec.num_heads}"
        assert spec.num_layers == 6, f"Expected 6, got {spec.num_layers}"
        assert spec.ff_dim == 2048, f"Expected 2048, got {spec.ff_dim}"

    def test_spec_to_dict(self):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        d = spec.to_dict()
        assert "arch_type" in d
        assert "hidden_dim" in d
        assert d["arch_type"] == "transformer"

    def test_param_estimation(self):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        assert spec.num_parameters_estimated is not None
        assert spec.num_parameters_estimated > 0


# ── Code Generator ───────────────────────────────────────────────────────

class TestCodeGenerator:
    def setup_method(self):
        from paper2code.architecture_interpreter import ArchitectureInterpreter
        from paper2code.methodology_parser import MethodologyParser
        from paper2code.code_generator import CodeGenerator
        self.interpreter = ArchitectureInterpreter(use_llm=False)
        self.parser = MethodologyParser(use_llm=False)
        self.gen = CodeGenerator(use_llm=False)

    def test_transformer_code_generation(self):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        methodology = self.parser.parse(TRANSFORMER_TEXT)
        code = self.gen.generate(spec, methodology)
        assert "import torch" in code.model_code
        assert "class" in code.model_code
        assert "def forward" in code.model_code

    def test_cnn_code_generation(self):
        spec = self.interpreter.interpret(CNN_TEXT)
        methodology = self.parser.parse(CNN_TEXT)
        code = self.gen.generate(spec, methodology)
        assert "import torch" in code.model_code
        assert "def forward" in code.model_code

    def test_training_code_generation(self):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        methodology = self.parser.parse(TRANSFORMER_TEXT)
        code = self.gen.generate(spec, methodology)
        assert "def train" in code.training_code
        assert "optimizer" in code.training_code.lower()

    def test_config_generation(self):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        methodology = self.parser.parse(TRANSFORMER_TEXT)
        code = self.gen.generate(spec, methodology)
        assert "hidden_dim" in code.config_yaml
        assert "learning_rate" in code.config_yaml

    def test_save_to_disk(self, tmp_path):
        spec = self.interpreter.interpret(TRANSFORMER_TEXT)
        methodology = self.parser.parse(TRANSFORMER_TEXT)
        code = self.gen.generate(spec, methodology)
        code.save(str(tmp_path / "output"))
        assert (tmp_path / "output" / "model.py").exists()
        assert (tmp_path / "output" / "train.py").exists()
        assert (tmp_path / "output" / "config.yaml").exists()


# ── Integration Test ─────────────────────────────────────────────────────

class TestEndToEnd:
    """End-to-end test using synthetic text (no actual PDF needed)."""

    def test_full_pipeline_heuristic(self, tmp_path):
        """Test the complete pipeline in heuristic mode."""
        from paper2code.pdf_text_extractor import PDFTextExtractor, PaperSections
        from paper2code.methodology_parser import MethodologyParser
        from paper2code.architecture_interpreter import ArchitectureInterpreter
        from paper2code.code_generator import CodeGenerator

        extractor = PDFTextExtractor()
        sections = extractor._parse_sections(TRANSFORMER_TEXT)
        sections.raw_text = TRANSFORMER_TEXT

        parser = MethodologyParser(use_llm=False)
        methodology = parser.parse(sections.methodology, sections.experiments, sections.abstract)

        interpreter = ArchitectureInterpreter(use_llm=False)
        spec = interpreter.interpret(sections.architecture, sections.methodology, sections.abstract)

        gen = CodeGenerator(use_llm=False)
        code = gen.generate(spec, methodology)
        code.save(str(tmp_path / "transformer_impl"))

        assert code.model_code
        assert "import torch" in code.model_code
        assert spec.arch_type.value == "transformer"
        assert methodology.training_config.optimizer.lower() in ("adam", "adamw")
