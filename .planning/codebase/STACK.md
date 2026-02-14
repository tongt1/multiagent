# Technology Stack

**Analysis Date:** 2026-02-14

## Languages

**Primary:**
- Python 3.11+ - All application code, training, inference, and tooling

**Secondary:**
- YAML - Configuration files (`config/*.yaml`, `configs/*.yaml`)
- Shell - Deployment and execution scripts (`scripts/*.sh`)

## Runtime

**Environment:**
- Python 3.11.2 (requires >=3.11)

**Package Manager:**
- Poetry 2.3.2
- Lockfile: `poetry.lock` (present, 497KB)
- Package manifest: `pyproject.toml`

## Frameworks

**Core:**
- LiteLLM 1.81.6+ - Unified LLM API interface (supports Cohere, OpenAI, Anthropic)
- Instructor 1.14.5+ - Structured outputs from LLMs with Pydantic
- Pydantic 2.12.5+ - Data validation and settings management
- Cohere 5.20.2+ - Primary LLM provider SDK

**Testing:**
- pytest 8.2+ - Test framework
- pytest-asyncio 0.25+ - Async test support

**Build/Dev:**
- Ruff 0.8.5+ - Linter and formatter
- MyPy 1.13+ - Static type checker
- Hatchling - Build backend

**Training:**
- PyTorch 2.0+ - Deep learning framework (optional dependency group)
- Transformers 4.40+ - HuggingFace model library (optional dependency group)
- Accelerate 0.30+ - Distributed training utilities (optional dependency group)

**Distributed Execution:**
- Kubernetes 28.0+ - Container orchestration (optional dependency group)
- Ray (with client) - Distributed training execution

**Data Visualization:**
- Streamlit 1.32.0+ - Interactive training viewer UI (optional dependency group)
- Plotly 5.19.0+ - Interactive plotting (optional dependency group)
- Pandas 2.2.0+ - Data manipulation (optional dependency group)

## Key Dependencies

**Critical:**
- `litellm>=1.81.6` - Abstraction layer for all LLM calls, enables multi-provider support
- `pydantic>=2.12.5` - Configuration management, structured outputs, data validation
- `cohere>=5.20.2` - Primary LLM provider (command-r, command-r-plus models)
- `instructor>=1.14.5` - Structured generation with Pydantic models

**Infrastructure:**
- `tenacity>=9.0` - Retry logic for LLM calls
- `loguru>=0.7` - Structured logging throughout application
- `rich>=14.1.0` - Rich terminal output
- `python-dotenv>=1.0` - Environment variable management
- `pydantic-settings>=2.0` - Settings from environment variables

**Data Processing:**
- `datasets>=3.2.0` - HuggingFace datasets (MATH, HumanEval)
- `jsonlines>=4.0.0` - JSONL trajectory file handling
- `pyyaml>=6.0` - YAML config parsing
- `pydantic-yaml>=1.6.0` - Pydantic models from YAML
- `sympy>=1.13` - Mathematical expression evaluation

**Machine Learning:**
- `wandb>=0.16` - Experiment tracking and logging (training dependency)
- `torch>=2.0` - PyTorch for model training (training dependency)
- `transformers>=4.40` - Model loading and training (training dependency)
- `numpy` - Numerical computations (via training dependencies)
- `scipy` - Statistical analysis (for comparison metrics)

**Evaluation:**
- `unidiff>=0.7.5` - Code diff parsing (cooperbench-eval subproject)
- `httpx>=0.27.0` - Async HTTP client (cooperbench-eval subproject)
- `matplotlib>=3.8.0` - Plotting (cooperbench-eval subproject)

## Configuration

**Environment:**
- `.env.example` template provided (actual `.env` not in repo)
- Required: `COHERE_API_KEY` for LLM inference
- Optional: `OPENAI_API_KEY` for alternative LLM provider
- Optional model overrides: `SOLVER_MODEL`, `VERIFIER_MODEL`, `JUDGE_MODEL`
- Wandb integration: `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_BASE_URL`

**Build:**
- `pyproject.toml` - Poetry project config, tool configs (ruff, mypy)
- `poetry.lock` - Pinned dependency versions
- Ruff config: target Python 3.11, 100 char line length
- MyPy config: strict mode, Pydantic plugin

**Runtime:**
- YAML configs in `config/` (pipeline, batch execution)
- YAML configs in `configs/` (cooperbench defaults)
- Experiment configs in `configs/experiments/`
- Run configs in `run_configs/` (SWEEP training configs)

## Platform Requirements

**Development:**
- Python >=3.11
- Poetry 2.0+
- Optional: Docker (for distributed training)
- Optional: Kubernetes cluster (for distributed execution)
- Optional: Ray cluster (for distributed training)

**Production:**
- Kubernetes cluster with kjobs support
- PyTorch container base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- GPU support: CUDA 12.4, cuDNN 9 (for training workloads)
- CPU-only mode supported for inference
- Wandb instance (Cohere internal: `https://cohere.wandb.io`)

---

*Stack analysis: 2026-02-14*
