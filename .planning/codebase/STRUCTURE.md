# Codebase Structure

**Analysis Date:** 2026-02-14

## Directory Layout

```
reward-training/
├── src/                          # Main source code (multi-agent RL pipeline)
│   ├── agents/                   # Solver, Verifier, Judge agents
│   ├── cli/                      # Command-line interface
│   ├── data/                     # Dataset loaders and converters
│   ├── evaluation/               # Reward computation and benchmarks
│   │   └── cooperbench/          # CooperBench cooperation evaluation
│   ├── infrastructure/           # LLM client, logging, cost tracking
│   ├── models/                   # Pydantic schemas
│   ├── orchestration/            # Pipeline runners and batch executors
│   └── training/                 # RL training (Comb env, reward shaping)
│       ├── multi_model/          # Multi-model training (experimental)
│       ├── reward_shaping/       # Reward transformation strategies
│       ├── rollout_strategy/     # Inference-time sampling strategies
│       └── wandb_enrichment/     # WandB logging extensions
├── cooperbench-eval/             # Standalone failure mode evaluation pipeline
│   ├── src/
│   │   ├── classifiers/          # Heuristic failure detectors
│   │   ├── data_loading/         # Trajectory loaders
│   │   ├── llm_judge/            # LLM-based failure classifiers
│   │   └── report/               # Report generation
│   └── tests/                    # CooperBench tests
├── scripts/                      # Utility scripts
├── configs/                      # SWEEP training configurations
│   └── experiments/              # Experiment configs
├── streamlit_viewer/             # Interactive training metrics viewer
│   ├── pages/                    # Streamlit multi-page app
│   └── lib/                      # Viewer utilities
├── config/                       # Pipeline YAML configs
│   └── problems/                 # Example problem files
├── data/                         # Data storage (local)
├── tests/                        # Unit and integration tests
├── docs/                         # Documentation
├── run_configs/                  # Run-specific configs
├── pyproject.toml                # Python project config (Poetry/Hatch)
└── .planning/                    # GSD planning documents
```

## Directory Purposes

**src/**
- Purpose: Core multi-agent RL pipeline implementation
- Contains: All production Python modules organized by layer
- Key files: Entry points, agent implementations, pipeline orchestration

**src/agents/**
- Purpose: LLM-based reasoning agents
- Contains: `solver.py`, `verifier.py`, `judge.py`, `base.py`
- Key files: `base.py` (abstract interface), role-specific implementations

**src/cli/**
- Purpose: Command-line interface
- Contains: `main.py` (entry point), `runner.py` (subcommand implementations)
- Key files: `main.py` (parse args, route commands)

**src/orchestration/**
- Purpose: Pipeline execution logic
- Contains: `pipeline.py` (debate pipeline), `baseline_runner.py`, `batch_executor.py`, `iteration.py`
- Key files: `pipeline.py` (SolverVerifierJudgePipeline), `baseline_runner.py` (BaselineRunner)

**src/training/**
- Purpose: RL training components
- Contains: `comb_math_debate_env.py` (Comb environment), reward/rollout strategies, WandB enrichment
- Key files: `comb_math_debate_env.py` (@register_builder for Comb discovery)

**src/training/reward_shaping/**
- Purpose: Reward transformation strategies
- Contains: `base.py`, `registry.py`, strategy implementations (difference rewards, COMA, potential-based)
- Key files: `registry.py` (strategy discovery), `base.py` (RewardShaper interface)

**src/training/wandb_enrichment/**
- Purpose: Enhanced WandB logging for debate metrics
- Contains: Debate metric computers, rollout tables, workspace templates
- Key files: `debate_streamer.py` (custom WandB callback)

**src/evaluation/**
- Purpose: Reward computation and answer verification
- Contains: `math_verifier.py` (symbolic math verification), `code_executor.py`, `reward_calculator.py`
- Key files: `math_verifier.py` (sympy-based equivalence checking)

**src/evaluation/cooperbench/**
- Purpose: CooperBench cooperation benchmark integration
- Contains: Pipeline, evaluator, reward computation for multi-agent code cooperation tasks
- Key files: `pipeline.py` (CooperBenchPipeline), `evaluator.py`

**src/infrastructure/**
- Purpose: Cross-cutting infrastructure services
- Contains: `llm_client.py` (LiteLLM wrapper), `trajectory_logger.py`, `cost_tracker.py`
- Key files: `llm_client.py` (unified LLM interface), `trajectory_logger.py` (JSONL logging)

**src/data/**
- Purpose: Dataset loading and trajectory conversion
- Contains: `math500.py` (MATH-500 loader), `comb_converter.py`, `training_export.py`
- Key files: `comb_converter.py` (transforms to Comb format), `math500.py` (dataset caching)

**cooperbench-eval/**
- Purpose: Standalone failure mode evaluation pipeline
- Contains: CLI, classifiers (heuristic + LLM), data loaders, report generation
- Key files: `src/cli.py` (entry point), `src/classifiers/` (failure detectors)

**scripts/**
- Purpose: Utility scripts for data generation, training, evaluation
- Contains: `launch_training.py`, `generate_trajectories.py`, `evaluate_checkpoints.py`, `compare_experiments.py`
- Key files: `launch_training.py` (data conversion + SWEEP submission)

**configs/**
- Purpose: SWEEP training configurations
- Contains: `sweep_math_debate_grpo.py`, `sweep_math_baseline_grpo.py`, `model_profiles.py`
- Key files: Sweep configs (define hyperparameter grids, resources)

**streamlit_viewer/**
- Purpose: Interactive training metrics dashboard
- Contains: Streamlit app with pages for metrics, trajectories, comparisons
- Key files: Multi-page Streamlit app structure

**config/**
- Purpose: Pipeline runtime configuration (YAML)
- Contains: `pipeline.yaml` examples, problem definitions
- Key files: Pipeline configs for debate/baseline modes

**.planning/**
- Purpose: GSD project planning documents
- Contains: PROJECT.md, ROADMAP.md, STATE.md, phases/, codebase/
- Key files: Milestone tracking, phase plans, codebase analysis

## Key File Locations

**Entry Points:**
- `src/cli/main.py`: Main CLI entry point
- `scripts/launch_training.py`: Training pipeline script
- `configs/sweep_math_debate_grpo.py`: SWEEP config for debate training
- `cooperbench-eval/src/main.py`: CooperBench eval CLI

**Configuration:**
- `pyproject.toml`: Python project metadata, dependencies
- `config/pipeline.yaml`: Pipeline runtime config (inferred)
- `.env.example`: Environment variable template
- `kjobs-compute.yaml`: Kubernetes job config

**Core Logic:**
- `src/orchestration/pipeline.py`: Main debate pipeline orchestration
- `src/training/comb_math_debate_env.py`: Comb environment for RL training
- `src/agents/solver.py`: Solver agent implementation
- `src/evaluation/math_verifier.py`: Math answer verification

**Testing:**
- `tests/`: Unit and integration tests for main pipeline
- `cooperbench-eval/tests/`: CooperBench-specific tests

## Naming Conventions

**Files:**
- `snake_case.py`: All Python modules
- `UPPERCASE.md`: Documentation (README.md, planning docs)
- `*.yaml` / `*.yml`: Configuration files
- `*.jsonl`: Trajectory and data files (JSONL format)

**Directories:**
- `snake_case/`: All directories
- No dashes in src directories (use underscores)

**Classes:**
- `PascalCase`: All classes (SolverAgent, BaseAgent, TrajectoryEntry)
- Suffix patterns: `*Agent`, `*Shaper`, `*Strategy`, `*Config`, `*Result`

**Functions:**
- `snake_case`: All functions and methods
- Prefix patterns: `compute_*`, `create_*`, `load_*`, `export_*`

**Modules:**
- `snake_case.py`: Descriptive names matching primary class/function
- `base.py`: Abstract base classes
- `__init__.py`: Package exports

## Where to Add New Code

**New Agent Type:**
- Primary code: `src/agents/{agent_name}.py` (inherit from BaseAgent)
- Tests: `tests/agents/test_{agent_name}.py`
- Integration: Update `src/agents/__init__.py` exports

**New Reward Shaping Strategy:**
- Implementation: `src/training/reward_shaping/{strategy_name}.py`
- Decorator: Use `@register_strategy("strategy_name")`
- Tests: `tests/training/reward_shaping/test_{strategy_name}.py`

**New Rollout Strategy:**
- Implementation: `src/training/rollout_strategy/{strategy_name}.py`
- Decorator: Use `@register_strategy("strategy_name")`
- Tests: `tests/training/rollout_strategy/test_{strategy_name}.py`

**New Evaluation Benchmark:**
- Pipeline: `src/evaluation/{benchmark_name}/pipeline.py`
- Models: `src/evaluation/{benchmark_name}/models.py`
- Evaluator: `src/evaluation/{benchmark_name}/evaluator.py`

**New CLI Subcommand:**
- Parser: Add subparser in `src/cli/main.py` (parse_args)
- Handler: Add async handler in `src/cli/main.py` (async_main)
- Runner: Implement logic in `src/cli/runner.py` or new module

**New Script:**
- Location: `scripts/{script_name}.py`
- Pattern: Argparse for CLI, import from src modules, main() entry point

**New SWEEP Config:**
- Location: `configs/sweep_{experiment_name}.py`
- Pattern: Import from Comb, define config class, register with @sweep decorator (inferred)

**Utilities:**
- Shared helpers: Add to existing module in `src/` or create `src/utils/{category}.py`
- Infrastructure services: `src/infrastructure/{service_name}.py`

## Special Directories

**cooperbench-eval/**
- Purpose: Self-contained failure mode evaluation pipeline
- Generated: No
- Committed: Yes
- Note: Can be run independently from main pipeline

**data/**
- Purpose: Local data storage (trajectories, datasets, caches)
- Generated: Yes (by pipeline runs)
- Committed: No (in .gitignore)

**streamlit_viewer/**
- Purpose: Interactive metrics viewer
- Generated: No
- Committed: Yes
- Note: Requires streamlit extras: `pip install -e ".[viewer]"`

**.planning/**
- Purpose: GSD project planning and tracking
- Generated: Yes (by GSD commands)
- Committed: Yes
- Note: Managed by GSD workflow

**experiments/**
- Purpose: Experiment metadata and results (inferred from launch_training.py)
- Generated: Yes (by training scripts)
- Committed: No (likely in .gitignore)

**trajectories/**
- Purpose: Default trajectory output directory
- Generated: Yes (by pipeline runs)
- Committed: No (likely in .gitignore)

---

*Structure analysis: 2026-02-14*
