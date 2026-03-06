#!/usr/bin/env python3
"""
scripts/generate_docs.py - Generates MkDocs documentation index.
Collects implementation reports and creates a unified docs site.
"""

import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DOCS_DIR = Path("docs")
IMPLEMENTATIONS_DIR = DOCS_DIR / "implementations"
MKDOCS_YML = Path("mkdocs.yml")

MKDOCS_CONFIG = """site_name: paper-to-code-generator
site_description: AI-powered research paper to PyTorch code generator
site_url: https://PranayMahendrakar.github.io/paper-to-code-generator/
repo_url: https://github.com/PranayMahendrakar/paper-to-code-generator

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  - admonition
  - pymdownx.details
  - tables

nav:
  - Home: index.md
  - API Reference:
    - paper_loader: api/paper_loader.md
    - pdf_text_extractor: api/pdf_text_extractor.md
    - methodology_parser: api/methodology_parser.md
    - architecture_interpreter: api/architecture_interpreter.md
    - code_generator: api/code_generator.md
    - experiment_runner: api/experiment_runner.md
  - Implementations: implementations/index.md
  - Limitations: limitations.md
"""


def generate_implementations_index(impl_dir: Path) -> str:
    """Scan for reports and generate implementations index."""
    lines = [
        "# Generated Implementations",
        "",
        f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Paper | Architecture | Parameters | Files | Report |",
        "|-------|-------------|------------|-------|--------|",
    ]

    json_files = sorted(impl_dir.rglob("*.json"))
    if not json_files:
        lines.append("| No implementations yet | - | - | - | - |")
    else:
        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                # Report JSON
                paper = data.get("paper_source", jf.stem)
                arch = data.get("architecture_type", "unknown")
                params = data.get("estimated_params", 0)
                files = len(data.get("files_generated", []))
                params_str = f"{params:,}" if params else "N/A"
                lines.append(f"| `{paper}` | {arch} | {params_str} | {files} | [{jf.stem}]({jf.name}) |")
            except Exception:
                pass

    return "\n".join(lines)


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    IMPLEMENTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Write mkdocs.yml
    MKDOCS_YML.write_text(MKDOCS_CONFIG)
    print(f"Written: {MKDOCS_YML}")

    # Generate implementations index
    impl_index = generate_implementations_index(IMPLEMENTATIONS_DIR)
    impl_index_path = IMPLEMENTATIONS_DIR / "index.md"
    impl_index_path.write_text(impl_index)
    print(f"Written: {impl_index_path}")

    # API reference stubs
    api_dir = DOCS_DIR / "api"
    api_dir.mkdir(exist_ok=True)
    modules = [
        "paper_loader", "pdf_text_extractor", "methodology_parser",
        "architecture_interpreter", "code_generator", "experiment_runner"
    ]
    for mod in modules:
        api_file = api_dir / f"{mod}.md"
        if not api_file.exists():
            api_file.write_text(
                f"# {mod}\n\n::: paper2code.{mod}\n"
            )
    print(f"Written API stubs to {api_dir}")

    # Limitations page
    lim_path = DOCS_DIR / "limitations.md"
    if not lim_path.exists():
        lim_path.write_text("""# Limitations

See the [README](../README.md#limitations-of-automated-paper-to-code-translation)
for a full discussion of system limitations.
""")
    print(f"Written: {lim_path}")
    print("Documentation generation complete.")


if __name__ == "__main__":
    main()
