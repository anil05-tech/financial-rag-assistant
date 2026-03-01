# financial-rag-assistant

A Retrieval-Augmented Generation (RAG) pipeline for financial documents, with an evaluation framework for measuring retrieval and generation quality.

## Project Structure

```
financial-rag-assistant/
├── src/
│   └── financial_rag/      # Core library
├── tests/                  # Unit and integration tests
├── notebooks/              # Exploratory notebooks
├── docker/                 # Dockerfiles and compose configs
├── ci/                     # CI/CD pipeline configs
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Create and activate virtualenv (Python 3.11)
py -3.11 -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install project with dev dependencies
pip install -e ".[dev]"
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src tests

# Type check
mypy src
```
