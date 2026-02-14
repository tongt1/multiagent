# External Integrations

**Analysis Date:** 2026-02-14

## APIs & External Services

**LLM Providers:**
- Cohere - Primary inference provider
  - SDK/Client: `cohere>=5.20.2`
  - Auth: `COHERE_API_KEY` (environment variable)
  - Models: command-r, command-r-plus (configurable per agent role)
  - Accessed via: LiteLLM abstraction layer (`src/infrastructure/llm_client.py`)

- LiteLLM - Multi-provider LLM gateway
  - SDK/Client: `litellm>=1.81.6`
  - Purpose: Unified interface for Cohere, OpenAI, Anthropic
  - Features: Retry logic, structured outputs via Instructor
  - Configuration: Model names in YAML configs (`config/pipeline.yaml`)

**HuggingFace:**
- Datasets - Dataset loading
  - SDK/Client: `datasets>=3.2.0`
  - Datasets: `hendrycks/competition_math`, `openai/openai_humaneval`
  - Usage: Training data sourcing (`src/data/dataset_loader.py`)

- Transformers - Model training
  - SDK/Client: `transformers>=4.40`
  - Purpose: Model loading, tokenization, training
  - Usage: MARTI training (`src/training/train_marti.py`)

- Hub - Artifact uploads
  - SDK/Client: `huggingface_hub` (optional)
  - Auth: HuggingFace token (when uploading artifacts)
  - Usage: Experiment artifact bundling (`src/orchestration/artifact_bundler.py`)

## Data Storage

**Databases:**
- None (filesystem-based persistence)

**File Storage:**
- Local filesystem - Primary storage
  - Trajectories: JSONL files in `trajectories/` or custom paths
  - Checkpoints: Training model checkpoints
  - Artifacts: Tar bundles with experiment results

- GCS/S3 - Cloud storage (optional)
  - Purpose: Training data upload, checkpoint storage
  - Access: Via training scripts (`scripts/launch_training.py`)
  - Configuration: Paths passed as CLI arguments

**Caching:**
- None (no explicit caching layer)

## Authentication & Identity

**Auth Provider:**
- API key-based authentication
  - Cohere: `COHERE_API_KEY` environment variable
  - OpenAI: `OPENAI_API_KEY` environment variable (optional)
  - Wandb: `WANDB_API_KEY` environment variable
  - HuggingFace: Token for Hub uploads (optional)

**Implementation:**
- Environment variables loaded via `python-dotenv`
- Settings managed via `pydantic-settings` (`src/models/config.py`)
- API keys passed to SDK clients automatically

## Monitoring & Observability

**Error Tracking:**
- None (no centralized error tracking service)

**Logs:**
- Loguru - Structured logging to console/files
  - Library: `loguru>=0.7`
  - Configuration: Application-level in source code
  - Output: Console (development), files (production)

**Experiment Tracking:**
- Wandb - Training metrics and artifacts
  - SDK/Client: `wandb>=0.16`
  - Auth: `WANDB_API_KEY`, `WANDB_BASE_URL`
  - Instance: Cohere internal (`https://cohere.wandb.io`)
  - Usage: Training runs, eval results, debate metrics, rollout tables
  - Integration: `src/training/wandb_enrichment/`, `src/evaluation/bee_eval.py`

**Metrics:**
- Rich - Console output formatting
  - Library: `rich>=14.1.0`
  - Usage: Progress bars, formatted tables, status displays

## CI/CD & Deployment

**Hosting:**
- Kubernetes - Production execution platform
  - SDK/Client: `kubernetes>=28.0` (optional)
  - Purpose: Distributed training job submission
  - Implementation: `src/infrastructure/distributed_executor.py`
  - Job specs: `kjobs-compute.yaml` (kjobs format)

- Ray - Distributed training cluster
  - SDK/Client: `ray[client]` (optional)
  - Dashboard: `http://localhost:8265` (default)
  - Purpose: Multi-node training orchestration
  - Implementation: `src/infrastructure/ray_training_executor.py`

**CI Pipeline:**
- None (no GitHub Actions or CI workflows detected)

**Container Registry:**
- Docker Hub - Base images
  - Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
  - Usage: Kubernetes training jobs

## Environment Configuration

**Required env vars:**
- `COHERE_API_KEY` - Cohere API authentication (required for inference)

**Optional env vars:**
- `OPENAI_API_KEY` - OpenAI API authentication (alternative provider)
- `SOLVER_MODEL` - Override default solver model (default: command-r-plus)
- `VERIFIER_MODEL` - Override default verifier model (default: command-r)
- `JUDGE_MODEL` - Override default judge model (default: command-r-plus)
- `WANDB_API_KEY` - Wandb authentication for training runs
- `WANDB_PROJECT` - Wandb project name
- `WANDB_RUN_NAME` - Wandb run name
- `WANDB_BASE_URL` - Wandb instance URL (Cohere: https://cohere.wandb.io)
- `WANDB_ENTITY` - Wandb entity/organization name

**Secrets location:**
- `.env` file (local development, gitignored)
- Kubernetes secrets (production deployments)
- Environment variables (CI/CD, container runtimes)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Development Tools

**Code Quality:**
- Ruff - Linting and formatting
  - Version: `>=0.8.5`
  - Config: `pyproject.toml` (target py311, 100 char lines)
  - Rules: E, F, I, N, W, UP

- MyPy - Type checking
  - Version: `>=1.13`
  - Config: `pyproject.toml` (strict mode, Pydantic plugin)

**Interactive Development:**
- Streamlit - Training visualization UI
  - Version: `>=1.32.0`
  - Entry point: `streamlit_viewer/app.py`
  - Features: Trajectory browsing, comparison, metrics visualization
  - Dependencies: Plotly, Pandas, PyArrow

## Distributed Computing

**Ray Cluster:**
- Job submission via Ray client API
- Runtime environment: Pip packages injected per job
- Environment variables: Wandb credentials passed to workers
- Resource management: GPU/CPU allocation per task
- Status tracking: Job status polling via Ray API

**Kubernetes (kjobs):**
- Job submission via Kubernetes API
- Workload type: Dynamic batch jobs
- Namespace: Configurable (default: `default`)
- Queue: CPU or GPU queues
- Resource specs: CPU, memory, GPU count in YAML

---

*Integration audit: 2026-02-14*
