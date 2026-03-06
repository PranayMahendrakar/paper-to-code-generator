# paper-to-code-generator

**Convert research papers to working PyTorch implementations automatically.**

## What is this?

`paper-to-code-generator` is an AI-powered pipeline that reads a research paper PDF and outputs a complete, runnable PyTorch implementation — using only **local open-source LLMs** (no external API required).

## Pipeline

```
PDF / arXiv ID
    |
    v  paper_loader
    |  pdf_text_extractor
    |  methodology_parser
    |  architecture_interpreter
    |  code_generator
    v
model.py + train.py + dataset.py + config.yaml
```

## Quick Start

```bash
pip install -e .
python scripts/run_analysis.py --paper 1706.03762 --output_dir outputs/ --no_llm
```

Or with Python:
```python
from paper2code import ExperimentRunner
runner = ExperimentRunner(output_dir="outputs", use_llm=False)
report = runner.run("1706.03762")
print(report.to_markdown())
```

## Modules

- [`paper_loader`](api/paper_loader.md) — Load PDFs from local, arXiv, or URL
- [`pdf_text_extractor`](api/pdf_text_extractor.md) — Segment text into sections
- [`methodology_parser`](api/methodology_parser.md) — Extract training hyperparameters
- [`architecture_interpreter`](api/architecture_interpreter.md) — Build ArchitectureSpec
- [`code_generator`](api/code_generator.md) — Generate PyTorch code
- [`experiment_runner`](api/experiment_runner.md) — Full pipeline orchestration

## Architecture Types Supported

Transformer, CNN, RNN/LSTM, VAE, GAN, MLP, GNN, Diffusion

## Links

- [GitHub Repository](https://github.com/PranayMahendrakar/paper-to-code-generator)
- [Generated Implementations](implementations/index.md)
- [Limitations](limitations.md)
