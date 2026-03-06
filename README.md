# paper-to-code-generator

> **AI-powered system that converts research papers (PDF) into working PyTorch implementations using local open-source LLMs.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/PranayMahendrakar/paper-to-code-generator/paper_analysis.yml?label=CI)](https://github.com/PranayMahendrakar/paper-to-code-generator/actions)

---

## Overview

`paper-to-code-generator` is a modular pipeline that reads a research paper PDF, extracts key sections, interprets the described neural network architecture, and generates a complete runnable PyTorch implementation — all using **local, open-source LLMs** (via Ollama) with **no external API calls**.

```
PDF / arXiv ID / URL
        |
        v
  paper_loader          ← Load from local file, arXiv, or URL
        |
        v
  pdf_text_extractor    ← Parse sections: Abstract, Methodology, Architecture, Experiments
        |
        v
  methodology_parser    ← Extract: optimizer, LR, batch size, dataset, metrics
        |
        v
  architecture_interpreter ← Build ArchitectureSpec (type, dims, layers, activations)
        |
        v
  code_generator        ← Generate model.py, train.py, dataset.py, config.yaml
        |
        v
  ExperimentReport      ← Markdown report + JSON summary
```

---

## Features

- **PDF Loading**: local files, arXiv IDs (`1706.03762`), direct URLs, with caching
- **Section Extraction**: Abstract, Introduction, Methodology, Architecture, Experiments
- **Architecture Detection**: Transformer, CNN, RNN/LSTM, GNN, Diffusion, VAE, GAN, MLP
- **Dimension Extraction**: hidden_dim, num_heads, num_layers, dropout, ff_dim, latent_dim
- **Training Config**: optimizer, learning rate, batch size, epochs, scheduler, loss
- **LLM Integration**: Uses Ollama (local) with regex heuristic fallback
- **Code Templates**: Full PyTorch model + training + dataset + inference + config
- **GitHub Actions**: Automated weekly analysis, artifact upload, GitHub Pages docs
- **Zero External APIs**: Everything runs locally

---

## Installation

```bash
git clone https://github.com/PranayMahendrakar/paper-to-code-generator.git
cd paper-to-code-generator
pip install -e .
# or: pip install -r requirements.txt
```

**Optional (for LLM mode):** Install [Ollama](https://ollama.ai) and pull a model:
```bash
ollama pull mistral
# or: ollama pull llama3.2, ollama pull phi3
```

---

## Quick Start

### Python API
```python
from paper2code import ExperimentRunner

# Initialize runner (use_llm=False for heuristic-only mode)
runner = ExperimentRunner(
    output_dir="outputs",
    use_llm=True,          # requires Ollama
    llm_model="mistral",
)

# Run on an arXiv paper
report = runner.run("1706.03762")  # "Attention Is All You Need"
print(report.to_markdown())
print(f"Generated {len(report.files_generated)} files in {report.output_dir}")
```

### CLI
```bash
# Analyse a paper (heuristic mode, no LLM required)
python scripts/run_analysis.py --paper 1706.03762 --output_dir outputs/ --no_llm

# With local LLM via Ollama
python scripts/run_analysis.py --paper 1706.03762 --output_dir outputs/ --llm_model mistral

# Batch mode
python scripts/run_analysis.py --paper 1706.03762 --batch 1512.03385 1406.2661 --output_dir batch_out/

# Local PDF
python scripts/run_analysis.py --paper /path/to/paper.pdf --output_dir outputs/
```

---

## System Modules

| Module | File | Responsibility |
|--------|------|----------------|
| `paper_loader` | `paper2code/paper_loader.py` | Load PDFs from local, arXiv, or URL with caching |
| `pdf_text_extractor` | `paper2code/pdf_text_extractor.py` | Extract and segment PDF text into named sections |
| `methodology_parser` | `paper2code/methodology_parser.py` | Parse training hyperparams and dataset requirements |
| `architecture_interpreter` | `paper2code/architecture_interpreter.py` | Detect arch type and build `ArchitectureSpec` |
| `code_generator` | `paper2code/code_generator.py` | Generate PyTorch model, training, dataset, and inference code |
| `experiment_runner` | `paper2code/experiment_runner.py` | Orchestrate full pipeline and produce `ExperimentReport` |

---

## Generated Output

For each paper, the system generates:

```
outputs/implementations/<paper-id>/
    model.py        ← PyTorch model class
    train.py        ← Training loop with optimizer/scheduler/logging
    dataset.py      ← Dataset class skeleton
    inference.py    ← Prediction utilities
    main.py         ← CLI entry point
    config.yaml     ← All hyperparameters
    requirements.txt

outputs/architecture/
    <paper-id>_arch.json   ← Structured ArchitectureSpec

outputs/reports/
    <paper-id>_<timestamp>.md    ← Human-readable report
    <paper-id>_<timestamp>.json  ← Machine-readable report
```

### Architecture Types Supported

| Type | Key Components Generated |
|------|--------------------------|
| Transformer | Encoder, Decoder, MultiheadAttention, PositionalEncoding |
| CNN | ConvBlock, ResidualBlock, GlobalAvgPool, Classifier |
| RNN/LSTM | Embedding, LSTM/GRU, Bidirectional, Classifier |
| VAE | Encoder, Decoder, Reparameterization, ELBO loss |
| GAN | Generator, Discriminator (template) |
| MLP | Configurable depth/width with activations |

---

## GitHub Actions Workflow

The workflow `.github/workflows/paper_analysis.yml` runs automatically:

| Job | Trigger | Description |
|-----|---------|-------------|
| `lint-and-test` | every push/PR | Runs flake8 linting and pytest unit tests |
| `sample-analysis` | push to main + weekly | Analyses 3 landmark papers (Transformer, ResNet, GAN) |
| `custom-analysis` | manual trigger | Run any paper via `workflow_dispatch` input |
| `build-docs` | push to main | Builds MkDocs documentation site |
| `deploy-pages` | push to main | Publishes docs to GitHub Pages |

**Manual trigger:** Go to Actions → "Paper Analysis & Documentation" → Run workflow → enter an arXiv ID.

---

## System Design

### Architecture

The system follows a **linear pipeline** with each module producing structured output consumed by the next:

1. **Separation of concerns**: Each module has a single responsibility and can be used independently.
2. **Dual-mode operation**: LLM mode (Ollama) for semantic understanding; heuristic mode (regex) as reliable fallback.
3. **Structured intermediate representation**: `ArchitectureSpec` is the central data structure decoupling interpretation from generation.
4. **Template-based generation**: Proven architecture templates guarantee runnable code even without an LLM.
5. **Offline-first**: No internet required after PDF download; LLM runs locally via Ollama.

### Data Flow
```
PDF bytes → PaperSections → MethodologyResult + ArchitectureSpec → GeneratedCode → ExperimentReport
```

---

## Limitations of Automated Paper-to-Code Translation

Automated paper-to-code translation is an open and challenging problem. The following limitations apply to this system:

**1. Ambiguous Descriptions**
Research papers are written for human readers, not compilers. Authors often describe architectures informally ("we use a standard transformer encoder") without specifying every hyperparameter. The system uses sensible defaults where values are missing, but these may not match the paper exactly.

**2. Novel Architectures**
Papers introducing entirely new operations (e.g., custom CUDA kernels, Flash Attention variants, mixture-of-experts routing) cannot be captured by template-based generation. The system will fall back to the closest known architecture type.

**3. PDF Extraction Quality**
PDF parsing is imperfect. Mathematical notation, figures, tables, and multi-column layouts can confuse text extraction. Critical architecture details described in figures or equations may be missed.

**4. LLM Accuracy**
Local LLMs (Mistral 7B, LLaMA 3.2, Phi-3) may misinterpret technical terminology or hallucinate hyperparameter values. Always validate generated configurations against the original paper.

**5. Dataset Code**
The `dataset.py` skeleton requires manual implementation. The system identifies the dataset name and task type, but cannot automatically download or preprocess proprietary or custom datasets.

**6. Multi-Paper Dependencies**
Papers that build on prior work (e.g., "we follow the architecture of [5]") require the referenced papers to also be analysed. This is not currently automated.

**7. Reproducibility Gap**
Even with perfect architecture extraction, reproducing state-of-the-art results often requires implementation details not described in the paper: specific random seeds, learning rate warm-up schedules, data augmentation tricks, or hardware-specific optimizations.

**8. Scope**
Currently supports: supervised classification, generation, seq2seq, and generative models. Reinforcement learning, meta-learning, and multi-modal architectures have limited support.

---

## Project Structure

```
paper-to-code-generator/
├── paper2code/
│   ├── __init__.py
│   ├── paper_loader.py
│   ├── pdf_text_extractor.py
│   ├── methodology_parser.py
│   ├── architecture_interpreter.py
│   ├── code_generator.py
│   └── experiment_runner.py
├── scripts/
│   ├── run_analysis.py
│   └── generate_docs.py
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── index.md
├── .github/
│   └── workflows/
│       └── paper_analysis.yml
├── setup.py
└── README.md
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Ollama](https://ollama.ai) for local LLM serving
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [PyTorch](https://pytorch.org) for the deep learning framework
- Landmark papers used in tests: "Attention Is All You Need" (Vaswani et al.), "Deep Residual Learning" (He et al.), "Generative Adversarial Nets" (Goodfellow et al.)
