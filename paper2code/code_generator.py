"""
code_generator.py - Generates runnable PyTorch implementations from ArchitectureSpec.
Supports: Transformer, CNN, RNN, GNN, Diffusion, VAE, GAN, MLP.
Uses template-based generation + optional LLM refinement.
"""

import os
import re
import json
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

from .architecture_interpreter import ArchitectureSpec, ArchType
from .methodology_parser import MethodologyResult, TrainingConfig, DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Result of code generation for a paper."""
    model_code: str = ""
    training_code: str = ""
    dataset_code: str = ""
    inference_code: str = ""
    requirements_txt: str = ""
    config_yaml: str = ""
    full_script: str = ""
    output_dir: Optional[str] = None

    def save(self, output_dir: str) -> None:
        """Save all generated files to output directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {
            "model.py": self.model_code,
            "train.py": self.training_code,
            "dataset.py": self.dataset_code,
            "inference.py": self.inference_code,
            "requirements.txt": self.requirements_txt,
            "config.yaml": self.config_yaml,
            "main.py": self.full_script,
        }
        for filename, content in files.items():
            if content.strip():
                (out / filename).write_text(content, encoding="utf-8")
                logger.info(f"Wrote {out / filename}")
        self.output_dir = str(out)


# ── Code Templates ────────────────────────────────────────────────────────────

TRANSFORMER_TEMPLATE = '''import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = {dropout}):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model - {model_name}
    Architecture: {description}
    Estimated parameters: {num_params:,}
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = {hidden_dim},
        nhead: int = {num_heads},
        num_encoder_layers: int = {num_layers},
        num_decoder_layers: int = {num_layers},
        dim_feedforward: int = {ff_dim},
        dropout: float = {dropout},
        max_seq_len: int = 512,
        num_classes: int = 2,
        task: str = "classification",  # classification | generation | seq2seq
    ):
        super().__init__()
        self.d_model = d_model
        self.task = task

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="{activation}", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        if task == "seq2seq":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                activation="{activation}", batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes if task == "classification" else vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        src = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        if self.task == "seq2seq" and tgt is not None:
            tgt = self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model))
            out = self.decoder(tgt, memory)
        else:
            out = memory
        out = self.norm(out)
        if self.task == "classification":
            out = out.mean(dim=1)
        return self.output_proj(out)
'''

CNN_TEMPLATE = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = {kernel_size}, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.{activation_cls}()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CNNModel(nn.Module):
    """
    CNN model - {model_name}
    Architecture: {description}
    Estimated parameters: {num_params:,}
    """

    def __init__(
        self,
        in_channels: int = {in_channels},
        num_classes: int = 10,
        base_channels: int = {out_channels},
        num_blocks: int = {num_layers},
        dropout: float = {dropout},
    ):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        layers = []
        channels = base_channels
        for i in range(num_blocks):
            layers.append(ResidualBlock(channels))
            if i < num_blocks - 1:
                next_channels = min(channels * 2, 512)
                layers.append(ConvBlock(channels, next_channels, stride=2))
                channels = next_channels
        self.features = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x))
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)
'''

VAE_TEMPLATE = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.{activation_cls}(),
            nn.Linear(hidden_dim, hidden_dim), nn.{activation_cls}(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.{activation_cls}(),
            nn.Linear(hidden_dim, hidden_dim), nn.{activation_cls}(),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAEModel(nn.Module):
    """
    Variational Autoencoder - {model_name}
    Architecture: {description}
    Estimated parameters: {num_params:,}
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = {hidden_dim},
        latent_dim: int = {latent_dim},
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        mu, log_var = self.encoder(x_flat)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon.view_as(x), mu, log_var

    @staticmethod
    def loss(recon_x, x, mu, log_var, beta: float = 1.0):
        recon_loss = F.binary_cross_entropy(recon_x.view(x.size(0), -1),
                                             x.view(x.size(0), -1), reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kld
'''

RNN_TEMPLATE = '''import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    RNN/LSTM model - {model_name}
    Architecture: {description}
    Estimated parameters: {num_params:,}
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = {hidden_dim},
        hidden_dim: int = {hidden_dim},
        num_layers: int = {num_layers},
        num_classes: int = 2,
        dropout: float = {dropout},
        bidirectional: bool = False,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional, batch_first=True,
        )
        factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_dim * factor)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * factor, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.rnn(emb)
        out = self.norm(out[:, -1])
        return self.classifier(self.dropout(out))
'''

MLP_TEMPLATE = '''import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    MLP model - {model_name}
    Architecture: {description}
    Estimated parameters: {num_params:,}
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = {hidden_dim},
        num_layers: int = {num_layers},
        num_classes: int = 10,
        dropout: float = {dropout},
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.{activation_cls}(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.{activation_cls}(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(x.size(0), -1))
'''

TRAINING_TEMPLATE = '''"""
Training script - generated by paper2code
Model: {model_name}
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_optimizer(model, config):
    opt_name = config.get("optimizer", "adam").lower()
    lr = config.get("learning_rate", {learning_rate})
    wd = config.get("weight_decay", {weight_decay})
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {{opt_name}}")


def get_scheduler(optimizer, config):
    sched = config.get("scheduler", "").lower()
    epochs = config.get("num_epochs", {num_epochs})
    if sched == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif sched == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    return None


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        if outputs.dim() == 2:
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total > 0 else 0
    return total_loss / len(loader), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        if outputs.dim() == 2:
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total > 0 else 0
    return total_loss / len(loader), acc


def train(model, train_loader, val_loader, config: dict, save_dir: str = "checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {{device}}")
    model = model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()

    num_epochs = config.get("num_epochs", {num_epochs})
    best_val_acc = 0.0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config.get("gradient_clipping")
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        logger.info(
            f"Epoch {{epoch:3d}}/{{num_epochs}} | "
            f"Train Loss: {{train_loss:.4f}} Acc: {{train_acc:.4f}} | "
            f"Val Loss: {{val_loss:.4f}} Acc: {{val_acc:.4f}}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({{
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config,
            }}, save_path / "best_model.pt")

    logger.info(f"Training complete. Best val accuracy: {{best_val_acc:.4f}}")
    return model
'''

REQUIREMENTS_TEMPLATE = '''torch>=2.0.0
torchvision>=0.15.0
pdfplumber>=0.9.0
PyPDF2>=3.0.0
PyYAML>=6.0
requests>=2.28.0
tqdm>=4.64.0
numpy>=1.24.0
Pillow>=9.0.0
'''

CONFIG_TEMPLATE = '''# Configuration for {model_name}
# Generated by paper2code

model:
  name: "{model_name}"
  arch_type: "{arch_type}"
  hidden_dim: {hidden_dim}
  num_heads: {num_heads}
  num_layers: {num_layers}
  dropout: {dropout}
  activation: "{activation}"

training:
  optimizer: "{optimizer}"
  learning_rate: {learning_rate}
  batch_size: {batch_size}
  num_epochs: {num_epochs}
  scheduler: "{scheduler}"
  loss_function: "{loss_function}"
  weight_decay: {weight_decay}
  gradient_clipping: null

dataset:
  name: "{dataset_name}"
  task_type: "{task_type}"
  input_modality: "{input_modality}"

paths:
  output_dir: "outputs/{model_name_slug}"
  checkpoints: "checkpoints"
  logs: "logs"
'''


ACTIVATION_MAP = {
    "relu": "ReLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "swish": "SiLU",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "mish": "Mish",
}


class CodeGenerator:
    """
    Generates runnable PyTorch code from ArchitectureSpec and MethodologyResult.

    Usage:
        gen = CodeGenerator()
        code = gen.generate(spec, methodology)
        code.save("output/my_model")
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

    def generate(self, spec: ArchitectureSpec, methodology: MethodologyResult) -> GeneratedCode:
        """
        Generate complete PyTorch implementation from architecture spec and methodology.

        Args:
            spec: Structured architecture specification.
            methodology: Parsed methodology including training config and dataset info.

        Returns:
            GeneratedCode with all generated files.
        """
        code = GeneratedCode()
        tc = methodology.training_config
        dc = methodology.dataset_config

        # Select and fill model template
        code.model_code = self._generate_model_code(spec)

        # Generate training code
        code.training_code = self._generate_training_code(spec, tc)

        # Generate dataset code
        code.dataset_code = self._generate_dataset_code(dc)

        # Generate inference code
        code.inference_code = self._generate_inference_code(spec)

        # Generate configuration
        code.config_yaml = self._generate_config(spec, tc, dc)

        # Generate requirements
        code.requirements_txt = REQUIREMENTS_TEMPLATE.strip()

        # Optionally refine with LLM
        if self._llm_available:
            code.model_code = self._refine_with_llm(code.model_code, spec) or code.model_code

        # Generate main entry point
        code.full_script = self._generate_main(spec, tc, dc)

        return code

    def _get_template_vars(self, spec: ArchitectureSpec) -> Dict[str, Any]:
        act_cls = ACTIVATION_MAP.get(spec.activation.lower(), "ReLU")
        return {
            "model_name": spec.model_name or f"{spec.arch_type.value.title()}Model",
            "description": spec.description or "Generated from research paper",
            "hidden_dim": spec.hidden_dim,
            "num_heads": spec.num_heads,
            "num_layers": spec.num_layers,
            "ff_dim": spec.ff_dim,
            "dropout": spec.dropout,
            "activation": spec.activation,
            "activation_cls": act_cls,
            "in_channels": spec.in_channels,
            "out_channels": spec.out_channels,
            "kernel_size": spec.kernel_size,
            "latent_dim": spec.latent_dim,
            "num_params": spec.num_parameters_estimated or 0,
        }

    def _generate_model_code(self, spec: ArchitectureSpec) -> str:
        vars = self._get_template_vars(spec)
        template_map = {
            ArchType.TRANSFORMER: TRANSFORMER_TEMPLATE,
            ArchType.CNN: CNN_TEMPLATE,
            ArchType.VAE: VAE_TEMPLATE,
            ArchType.RNN: RNN_TEMPLATE,
            ArchType.MLP: MLP_TEMPLATE,
        }
        template = template_map.get(spec.arch_type, MLP_TEMPLATE)
        try:
            return template.format(**vars)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            return template

    def _generate_training_code(self, spec: ArchitectureSpec, tc: TrainingConfig) -> str:
        vars = self._get_template_vars(spec)
        vars.update({
            "learning_rate": tc.learning_rate,
            "weight_decay": tc.weight_decay,
            "num_epochs": tc.num_epochs,
            "batch_size": tc.batch_size,
        })
        try:
            return TRAINING_TEMPLATE.format(**vars)
        except KeyError as e:
            logger.warning(f"Training template error: {e}")
            return TRAINING_TEMPLATE

    def _generate_dataset_code(self, dc: DatasetConfig) -> str:
        dataset_name = dc.name or "CustomDataset"
        task_type = dc.task_type or "classification"
        return f'''"""
dataset.py - Dataset loading for {dataset_name}
Task: {task_type}
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


class {dataset_name.replace("-","_").replace(" ","_")}Dataset(Dataset):
    """
    Dataset class for {dataset_name}.
    Task type: {task_type}
    Input modality: {dc.input_modality or "unknown"}
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        # TODO: Implement actual data loading for {dataset_name}
        # This is a placeholder - replace with actual data loading logic
        raise NotImplementedError(
            f"Implement _load_samples() for {dataset_name}. "
            "Load file paths and labels from self.root / self.split"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample, label = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def get_dataloaders(root: str, batch_size: int = 32, num_workers: int = 4):
    """Create train, val, test dataloaders."""
    from torchvision import transforms

    # Default transforms - customize for your dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = {dataset_name.replace("-","_").replace(" ","_")}Dataset(root, "train", train_transform)
    val_ds   = {dataset_name.replace("-","_").replace(" ","_")}Dataset(root, "val",   val_transform)
    test_ds  = {dataset_name.replace("-","_").replace(" ","_")}Dataset(root, "test",  val_transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
'''

    def _generate_inference_code(self, spec: ArchitectureSpec) -> str:
        model_name = spec.model_name or f"{spec.arch_type.value.title()}Model"
        return f'''"""
inference.py - Inference utilities for {model_name}
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Union, List


def load_model(checkpoint_path: str, model_class, **model_kwargs):
    """Load a trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


@torch.no_grad()
def predict(model, inputs: torch.Tensor, top_k: int = 5):
    """Run inference on inputs."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    logits = model(inputs)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(top_k, dim=-1)
    return top_indices, top_probs


@torch.no_grad()
def predict_batch(model, dataloader, device=None):
    """Run inference on entire dataloader."""
    if device is None:
        device = next(model.parameters()).device
    all_preds, all_probs = [], []
    model.eval()
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return all_preds, all_probs
'''

    def _generate_config(self, spec: ArchitectureSpec, tc: TrainingConfig, dc: DatasetConfig) -> str:
        model_name = spec.model_name or f"{spec.arch_type.value.title()}Model"
        slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        return CONFIG_TEMPLATE.format(
            model_name=model_name,
            arch_type=spec.arch_type.value,
            hidden_dim=spec.hidden_dim,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            dropout=spec.dropout,
            activation=spec.activation,
            optimizer=tc.optimizer,
            learning_rate=tc.learning_rate,
            batch_size=tc.batch_size,
            num_epochs=tc.num_epochs,
            scheduler=tc.scheduler,
            loss_function=tc.loss_function,
            weight_decay=tc.weight_decay,
            dataset_name=dc.name,
            task_type=dc.task_type,
            input_modality=dc.input_modality,
            model_name_slug=slug,
        )

    def _generate_main(self, spec: ArchitectureSpec, tc: TrainingConfig, dc: DatasetConfig) -> str:
        model_name = spec.model_name or f"{spec.arch_type.value.title()}Model"
        return f'''"""
main.py - Entry point for {model_name} training
Generated by paper2code
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train {model_name}")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data_root", required=True, help="Dataset root directory")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    training_cfg = config.get("training", {{}})

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {{device}}")

    # Import model (update import based on architecture)
    from model import {model_name.replace(" ", "").replace("-", "")}

    # Build model
    model_cfg = config.get("model", {{}})
    model = {model_name.replace(" ", "").replace("-", "")}(**model_cfg)
    logger.info(f"Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")

    # Load data
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root,
        batch_size=training_cfg.get("batch_size", {tc.batch_size})
    )

    # Train
    from train import train
    trained_model = train(model, train_loader, val_loader, training_cfg, args.output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
'''

    def _refine_with_llm(self, code: str, spec: ArchitectureSpec) -> Optional[str]:
        """Use LLM to refine/improve generated code."""
        import urllib.request, json

        prompt = f"""Review and improve this PyTorch model code for {spec.model_name}.
Fix any issues, add missing components, and ensure it matches the architecture.

Current code:
{code[:2000]}

Return only the improved Python code, no explanations."""

        payload = json.dumps({
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024}
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                response_data = json.loads(resp.read())
                refined = response_data.get("response", "")
                if "import torch" in refined:
                    return refined
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
        return None
