# MAS Bias TFG - Orchestration Infrastructure

## Architecture Overview

```
mas-bias-tfg/
├── config.py              # Experiment configurations & model endpoints
├── models.py              # Unified interface for vLLM + API models
├── agents.py              # Agent definitions (roles, prompts, behavior)
├── protocols.py           # Communication protocols (debate, cooperative)
├── orchestrator.py        # Main debate loop orchestrating agents
├── experiment_runner.py   # Iterates over experimental conditions
├── metrics.py             # Allocation Gap, Recommendation Parity, etc.
├── data/
│   └── baskets/           # JSON files with basket definitions
├── results/               # Output CSVs with per-turn per-agent data
└── analysis/
    └── analyze.py         # Post-hoc analysis & visualization
```

## How it works

1. `experiment_runner.py` reads the experiment matrix from `config.py`
2. For each condition, it initializes agents via `agents.py` with the right model + role
3. `orchestrator.py` runs the debate: genesis → N rounds of communication
4. Each turn's JSON output is appended to a CSV (checkpointing)
5. After all experiments, `metrics.py` computes Allocation Gap, etc.

## Running

```bash
# Single experiment (for testing)
python orchestrator.py --basket data/baskets/B001_privileged.json --config baseline

# Full experiment suite
python experiment_runner.py --config ablation_plan.yaml

# Analysis
python analysis/analyze.py --results results/
```
