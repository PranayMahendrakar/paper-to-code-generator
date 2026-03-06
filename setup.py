from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="paper2code",
    version="1.0.0",
    author="PranayMahendrakar",
    description="Convert research papers to PyTorch implementations using local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PranayMahendrakar/paper-to-code-generator",
    packages=find_packages(exclude=["tests*", "docs*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pdfplumber>=0.9.0",
        "PyPDF2>=3.0.0",
        "PyYAML>=6.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "paper2code=scripts.run_analysis:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="research papers pytorch code generation LLM NLP",
)
