"""
architecture_interpreter.py - Interprets neural network architecture from paper text.
Produces a structured ArchitectureSpec before code generation.
Supports: Transformer, CNN, RNN/LSTM, GNN, Diffusion, VAE, GAN, MLP.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ArchType(str, Enum):
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GNN = "gnn"
    DIFFUSION = "diffusion"
    VAE = "vae"
    GAN = "gan"
    MLP = "mlp"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class LayerSpec:
    """Specification for a single layer in the network."""
    layer_type: str        # Linear, Conv2d, MultiheadAttention, etc.
    params: Dict[str, Any] = field(default_factory=dict)
    activation: str = ""
    normalization: str = ""
    dropout: float = 0.0
    skip_connection: bool = False
    description: str = ""


@dataclass
class BlockSpec:
    """A reusable block (e.g., Transformer block, ResNet block)."""
    name: str
    layers: List[LayerSpec] = field(default_factory=list)
    repeat: int = 1
    description: str = ""


@dataclass
class ArchitectureSpec:
    """
    Complete structured representation of a neural network architecture.
    This intermediate representation drives code generation.
    """
    arch_type: ArchType = ArchType.UNKNOWN
    model_name: str = ""
    description: str = ""

    # Dimensions
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    sequence_length: Optional[int] = None

    # CNN specific
    in_channels: int = 3
    out_channels: int = 64
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1

    # Transformer specific
    ff_dim: int = 2048
    use_pos_encoding: bool = True
    use_cross_attention: bool = False

    # GAN/VAE specific
    latent_dim: int = 128
    has_encoder: bool = False
    has_decoder: bool = False
    has_discriminator: bool = False

    # Components
    encoder_blocks: List[BlockSpec] = field(default_factory=list)
    decoder_blocks: List[BlockSpec] = field(default_factory=list)
    head_layers: List[LayerSpec] = field(default_factory=list)

    # Metadata
    activation: str = "relu"
    normalization: str = "layernorm"
    num_parameters_estimated: Optional[int] = None
    paper_snippets: List[str] = field(default_factory=list)
    raw_llm_output: str = ""

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON export."""
        return {
            "arch_type": self.arch_type.value,
            "model_name": self.model_name,
            "description": self.description,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "ff_dim": self.ff_dim,
            "latent_dim": self.latent_dim,
            "has_encoder": self.has_encoder,
            "has_decoder": self.has_decoder,
            "has_discriminator": self.has_discriminator,
            "activation": self.activation,
            "normalization": self.normalization,
            "in_channels": self.in_channels,
            "kernel_size": self.kernel_size,
        }


# ── Detection patterns ───────────────────────────────────────────────────────

ARCH_KEYWORDS = {
    ArchType.TRANSFORMER: re.compile(
        r"\b(transformer|self[\s-]attention|multi[\s-]head|bert|gpt|vit|"
        r"attention[\s-]mechanism|encoder[\s-]decoder|t5|llama|mistral)\b", re.I),
    ArchType.CNN: re.compile(
        r"\b(conv(olutional)?[\s-]neural|resnet|vgg|efficientnet|"
        r"inception|densenet|mobilenet|unet|feature[\s-]pyramid|fpn)\b", re.I),
    ArchType.RNN: re.compile(
        r"\b(lstm|gru|recurrent|bidirectional|seq2seq|rnn\b)\b", re.I),
    ArchType.GNN: re.compile(
        r"\b(graph[\s-]neural|gnn|gcn|gat|message[\s-]passing|"
        r"graph[\s-]convolutional|graph[\s-]attention)\b", re.I),
    ArchType.DIFFUSION: re.compile(
        r"\b(diffusion|denoising|score[\s-]matching|ddpm|ddim|"
        r"noise[\s-]prediction|unet[\s-]diffusion)\b", re.I),
    ArchType.VAE: re.compile(
        r"\b(variational[\s-]auto|vae|reparameter|kl[\s-]divergence|"
        r"latent[\s-]space|evidence[\s-]lower[\s-]bound|elbo)\b", re.I),
    ArchType.GAN: re.compile(
        r"\b(generative[\s-]adversarial|gan|discriminator|generator[\s-]network|"
        r"adversarial[\s-]training|wasserstein)\b", re.I),
}

DIM_PATTERNS = {
    "hidden_dim": re.compile(r"hidden[\s-]+(?:dim(?:ension)?|size)[\s:=]+([0-9]+)", re.I),
    "num_heads": re.compile(r"(?:attention[\s-]+)?heads?[\s:=]+([0-9]+)", re.I),
    "num_layers": re.compile(r"(?:num(?:ber)?[\s-]+of[\s-]+)?(?:layers?|blocks?|depth)[\s:=]+([0-9]+)", re.I),
    "ff_dim": re.compile(r"(?:feed[\s-]?forward|ffn|mlp)[\s-]+(?:dim|size|hidden)[\s:=]+([0-9]+)", re.I),
    "dropout": re.compile(r"dropout[\s:=]+([0-9]+(?:\.[0-9]+)?)", re.I),
    "latent_dim": re.compile(r"latent[\s-]+(?:dim|size)[\s:=]+([0-9]+)", re.I),
    "in_channels": re.compile(r"input[\s-]+channel[s]?[\s:=]+([0-9]+)", re.I),
    "kernel_size": re.compile(r"kernel[\s-]+size[\s:=]+([0-9]+)", re.I),
}


class ArchitectureInterpreter:
    """
    Interprets architecture sections from a research paper and produces
    a structured ArchitectureSpec.

    Usage:
        interpreter = ArchitectureInterpreter()
        spec = interpreter.interpret(arch_text, methodology_text)
        print(spec.to_dict())
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
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=3):
                return True
        except Exception:
            return False

    def interpret(self, arch_text: str, methodology_text: str = "",
                  abstract_text: str = "") -> ArchitectureSpec:
        """
        Interpret architecture from paper text sections.

        Args:
            arch_text: Architecture section text.
            methodology_text: Methodology section text.
            abstract_text: Abstract for context.

        Returns:
            ArchitectureSpec - structured architecture representation.
        """
        combined = f"{arch_text}\n\n{methodology_text}"
        spec = ArchitectureSpec()

        # Step 1: Detect architecture type
        spec.arch_type = self._detect_arch_type(combined)
        logger.info(f"Detected architecture type: {spec.arch_type}")

        # Step 2: Extract dimensions via regex
        self._extract_dimensions(combined, spec)

        # Step 3: LLM-enhanced interpretation
        if self._llm_available:
            self._enhance_with_llm(combined, abstract_text, spec)

        # Step 4: Set component flags
        self._set_component_flags(spec)

        # Step 5: Estimate parameter count
        spec.num_parameters_estimated = self._estimate_params(spec)

        return spec

    def _detect_arch_type(self, text: str) -> ArchType:
        scores: Dict[ArchType, int] = {}
        for arch_type, pattern in ARCH_KEYWORDS.items():
            matches = pattern.findall(text)
            if matches:
                scores[arch_type] = len(matches)
        if not scores:
            return ArchType.UNKNOWN
        if len(scores) > 2:
            return ArchType.HYBRID
        return max(scores, key=lambda k: scores[k])

    def _extract_dimensions(self, text: str, spec: ArchitectureSpec) -> None:
        for attr, pattern in DIM_PATTERNS.items():
            match = pattern.search(text)
            if match:
                try:
                    val = float(match.group(1))
                    if attr == "dropout":
                        spec.dropout = min(val, 1.0) if val <= 1.0 else val / 100
                    else:
                        setattr(spec, attr, int(val))
                except (ValueError, AttributeError):
                    pass

        # Extract activation function
        act_match = re.search(r"\b(relu|gelu|silu|swish|tanh|sigmoid|leaky[\s-]?relu|mish)\b",
                               text, re.I)
        if act_match:
            spec.activation = act_match.group(1).lower()

        # Extract normalization
        norm_match = re.search(r"\b(layer[\s-]?norm|batch[\s-]?norm|group[\s-]?norm|"
                                r"instance[\s-]?norm|rms[\s-]?norm)\b", text, re.I)
        if norm_match:
            spec.normalization = norm_match.group(1).lower().replace(" ", "")

    def _enhance_with_llm(self, text: str, abstract: str, spec: ArchitectureSpec) -> None:
        import urllib.request, urllib.error

        prompt = f"""Analyze this neural network architecture description and extract key parameters.

Abstract: {abstract[:300]}

Architecture text:
{text[:2000]}

Return JSON with these fields:
{{
  "model_name": "e.g. ResNet-50, BERT-base, ViT-B/16",
  "description": "one sentence description",
  "hidden_dim": 256,
  "num_heads": 8,
  "num_layers": 6,
  "ff_dim": 2048,
  "dropout": 0.1,
  "activation": "relu|gelu|silu",
  "normalization": "layernorm|batchnorm",
  "has_encoder": true,
  "has_decoder": false,
  "has_discriminator": false,
  "latent_dim": 128,
  "key_design_choices": ["choice 1", "choice 2"]
}}

Respond ONLY with the JSON."""

        payload = json.dumps({
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 400}
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
                raw = response_data.get("response", "")
                spec.raw_llm_output = raw
                self._merge_llm_data(raw, spec)
        except Exception as e:
            logger.warning(f"LLM architecture interpretation failed: {e}")

    def _merge_llm_data(self, raw: str, spec: ArchitectureSpec) -> None:
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                return
            data = json.loads(json_match.group())
            if data.get("model_name"):
                spec.model_name = data["model_name"]
            if data.get("description"):
                spec.description = data["description"]
            for attr in ["hidden_dim", "num_heads", "num_layers", "ff_dim", "latent_dim"]:
                if data.get(attr):
                    setattr(spec, attr, int(data[attr]))
            if data.get("dropout"):
                spec.dropout = float(data["dropout"])
            if data.get("activation"):
                spec.activation = data["activation"]
            if data.get("normalization"):
                spec.normalization = data["normalization"]
            for flag in ["has_encoder", "has_decoder", "has_discriminator"]:
                if flag in data:
                    setattr(spec, flag, bool(data[flag]))
        except Exception as e:
            logger.warning(f"Failed to merge LLM data: {e}")

    def _set_component_flags(self, spec: ArchitectureSpec) -> None:
        if spec.arch_type in (ArchType.VAE, ArchType.GAN, ArchType.DIFFUSION):
            spec.has_encoder = True
            spec.has_decoder = True
        if spec.arch_type == ArchType.GAN:
            spec.has_discriminator = True
        if spec.arch_type == ArchType.TRANSFORMER:
            spec.use_pos_encoding = True

    def _estimate_params(self, spec: ArchitectureSpec) -> int:
        """Rough parameter count estimate based on architecture type."""
        d = spec.hidden_dim
        L = spec.num_layers
        if spec.arch_type == ArchType.TRANSFORMER:
            # Each transformer layer: 4*d^2 (attention) + 8*d^2 (FFN) ~ 12*d^2
            return L * 12 * d * d
        elif spec.arch_type == ArchType.CNN:
            k = spec.kernel_size
            c = spec.out_channels
            return L * k * k * c * c
        elif spec.arch_type == ArchType.RNN:
            return L * 4 * d * d  # LSTM gates
        else:
            return L * d * d
