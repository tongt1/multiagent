# Architecture

**Analysis Date:** 2026-02-14

## Pattern Overview

**Overall:** Multi-Agent Reinforcement Learning (MARL) Pipeline with Debate-Based Training

**Key Characteristics:**
- Three-agent debate loop (Solver → Verifier → Judge) for multi-turn reasoning
- Dual-mode execution: debate (multi-agent) and baseline (single-agent)
- RL training via custom Comb environment integration with SWEEP infrastructure
- Modular reward shaping and rollout strategies via registry pattern
- Trajectory logging and export pipeline for training data generation

## Layers

**CLI Layer:**
- Purpose: User-facing command interface and orchestration entry points
- Location: `src/cli/`
- Contains: Argument parsing, subcommand routing, display formatting
- Depends on: Orchestration layer, Infrastructure layer
- Used by: Scripts, direct user invocation

**Agent Layer:**
- Purpose: LLM-based reasoning agents with role-specific prompts
- Location: `src/agents/`
- Contains: SolverAgent, VerifierAgent, JudgeAgent with structured output parsing
- Depends on: Infrastructure (LLMClient), Models (config, trajectory)
- Used by: Orchestration pipeline

**Orchestration Layer:**
- Purpose: Pipeline execution coordination and iteration control
- Location: `src/orchestration/`
- Contains: Pipeline runners, batch executors, iteration controllers, experiment state management
- Depends on: Agents, Infrastructure, Evaluation
- Used by: CLI, Scripts

**Evaluation Layer:**
- Purpose: Reward computation, answer verification, and benchmark evaluation
- Location: `src/evaluation/` and `src/evaluation/cooperbench/`
- Contains: Math verifiers (symbolic/LLM), code executors, reward calculators, CooperBench pipeline
- Depends on: Infrastructure, Models
- Used by: Orchestration, Training

**Training Layer:**
- Purpose: RL training configuration, reward shaping, and model management
- Location: `src/training/`
- Contains: Comb environment, reward shaping strategies, rollout strategies, multi-model training, WandB enrichment
- Depends on: Evaluation, Models
- Used by: SWEEP jobs, training scripts

**Infrastructure Layer:**
- Purpose: Cross-cutting services for LLM calls, logging, cost tracking
- Location: `src/infrastructure/`
- Contains: LLMClient (LiteLLM wrapper), TrajectoryLogger, CostTracker, DistributedExecutor, RayTrainingExecutor
- Depends on: Models (minimal)
- Used by: All layers

**Models Layer:**
- Purpose: Type definitions and data schemas
- Location: `src/models/`
- Contains: Pydantic models for config, messages, trajectories, evaluation results
- Depends on: Nothing (leaf layer)
- Used by: All layers

**Data Layer:**
- Purpose: Dataset loading, trajectory conversion, and export
- Location: `src/data/`
- Contains: MATH-500 loader, Comb converter, MARTI exporter, dataset utilities
- Depends on: Models, Evaluation
- Used by: Scripts, CLI

## Data Flow

**Inference Pipeline (Debate Mode):**

1. User provides problem → CLI → SolverVerifierJudgePipeline
2. Pipeline initializes agents (Solver, Verifier, Judge) with LLMClients
3. **Iteration loop** (max iterations):
   - Solver generates solution (with optional feedback from previous iteration)
   - Verifier validates solution, provides critique
   - If passed: break. If circular critique: break. Else: continue with critique as feedback
4. Judge scores final solution
5. If ground truth available: compute reward via MathVerifier
6. TrajectoryLogger writes JSONL entries for each step
7. Return PipelineResult with solution, scores, costs

**Inference Pipeline (Baseline Mode):**

1. User provides problem → CLI → BaselineRunner
2. BaselineRunner creates single Solver agent
3. Solver generates solution (one-shot, no iteration)
4. If ground truth available: compute reward
5. Log trajectory, return PipelineResult

**Training Data Generation:**

1. Load MATH-500 dataset → BatchPipelineExecutor
2. Execute debate/baseline pipelines on all problems concurrently
3. TrajectoryLogger writes raw JSONL trajectories
4. CombConverter transforms trajectories to Comb format
5. Split into train/eval sets
6. Upload to GCS (optional)

**RL Training Flow:**

1. SWEEP config loaded → submit training job
2. Comb preprocessor discovers `math_debate` environment via `@register_builder`
3. MathDebateScenarioBuilder creates multi-turn conversation setup
4. Environment executes rollout: problem → solver turn → verifier turn → final turn
5. MathDebateScenario.compute_reward() extracts final answer, calls SmartAnswerValidator
6. Reward computed (correctness + format) → gradients flow back through chatbot turns
7. WandB enrichment logs per-role metrics, debate dynamics, reward decomposition

**Multi-Model Training (Experimental):**

1. MultiModelManager loads separate models for solver/verifier roles
2. RoleRouter assigns model keys based on turn role labels
3. GradientRouter isolates gradients per model during backprop
4. DualLearner coordinates model updates with role-specific losses

**State Management:**
- Pipeline state: Iteration count, critique history, circular detection via IterationController
- Training state: Managed by Comb environment's ConversationState with debate_round tracking
- Experiment state: ExperimentState tracks run metadata, checkpoints, WandB run IDs

## Key Abstractions

**BaseAgent:**
- Purpose: Abstract agent interface with LLM client integration
- Examples: `src/agents/solver.py`, `src/agents/verifier.py`, `src/agents/judge.py`
- Pattern: Template method with `_build_messages()` hook

**TrajectoryEntry:**
- Purpose: Structured log entry for multi-agent interactions
- Examples: Used in `src/infrastructure/trajectory_logger.py`
- Pattern: Pydantic model with timestamp, agent, action, input/output, metadata, reward fields

**RewardShaper:**
- Purpose: Pluggable reward transformation strategies
- Examples: `src/training/reward_shaping/difference_rewards.py`, `src/training/reward_shaping/coma_advantage.py`
- Pattern: Strategy pattern with registry for discovery (`@register_strategy`)

**RolloutStrategy:**
- Purpose: Inference-time sampling strategies (best-of-N, self-consistency)
- Examples: `src/training/rollout_strategy/best_of_n.py`, `src/training/rollout_strategy/self_consistency.py`
- Pattern: Strategy pattern with registry

**Scenario (Comb):**
- Purpose: Defines RL environment structure for training
- Examples: `src/training/comb_math_debate_env.py` (MathDebateScenario)
- Pattern: Builder pattern with environment registration

**Pipeline:**
- Purpose: End-to-end orchestration of agent interactions
- Examples: `src/orchestration/pipeline.py` (SolverVerifierJudgePipeline), `src/evaluation/cooperbench/pipeline.py` (CooperBenchPipeline)
- Pattern: Facade pattern coordinating agents, logging, cost tracking

## Entry Points

**CLI Entry Point:**
- Location: `src/cli/main.py`
- Triggers: `python -m src.cli.main [subcommand]`
- Responsibilities: Parse arguments, route to subcommands (run, batch, train, generate, etc.), display results

**Training Script Entry Point:**
- Location: `scripts/launch_training.py`
- Triggers: Manual invocation for training job submission
- Responsibilities: Convert trajectories to Comb format, upload to GCS, submit SWEEP jobs

**SWEEP Config Entry Points:**
- Location: `configs/sweep_math_debate_grpo.py`, `configs/sweep_math_baseline_grpo.py`
- Triggers: SWEEP job submission via `uv run python config.py --submit start`
- Responsibilities: Define hyperparameter sweeps, training configs, resource allocation

**Comb Environment Entry Point:**
- Location: `src/training/comb_math_debate_env.py` (@register_builder)
- Triggers: Comb preprocessor discovery during training
- Responsibilities: Build debate scenarios, compute rewards, manage turn-taking

**CooperBench Evaluation Entry Point:**
- Location: `cooperbench-eval/src/main.py`
- Triggers: `python -m cooperbench-eval.src.main [args]`
- Responsibilities: Run failure mode evaluation pipeline on trajectories

**Streamlit Viewer Entry Point:**
- Location: `streamlit_viewer/app.py` (inferred)
- Triggers: `streamlit run streamlit_viewer/app.py`
- Responsibilities: Interactive visualization of training metrics and trajectories

## Error Handling

**Strategy:** Graceful degradation with logging

**Patterns:**
- Try-except blocks around agent calls with fallback values (0.0 scores, empty solutions)
- TrajectoryLogger.log_error() captures exceptions with context
- Tenacity retry decorators on LLM calls (in LLMClient)
- Validation errors logged via loguru with rich tracebacks
- Training errors surfaced via WandB run failures

## Cross-Cutting Concerns

**Logging:** Loguru with Rich handler for structured console output, TrajectoryLogger for JSONL step recording

**Validation:** Pydantic models enforce type safety, SmartAnswerValidator verifies math answers, code executors validate code solutions

**Authentication:** API keys loaded from `.env` via python-dotenv, LiteLLM handles provider authentication

---

*Architecture analysis: 2026-02-14*
