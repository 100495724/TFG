# MAS Bias TFG - Orchestration Infrastructure

## Architecture Overview

```
mas-bias-tfg/
├── config.py              # Experiment configurations, model endpoints & ablation plan
├── models.py              # Unified interface for vLLM + API models
├── agents.py              # Agent definitions (roles, prompts, debate behavior)
├── orchestrator.py        # Main debate loop orchestrating agents
├── experiment_runner.py   # Iterates over experimental conditions (stages)
├── generate_baskets.py    # Builds the counterfactual basket dataset
├── data/
│   └── baskets/           # JSON files with basket definitions (+ placebo/)
├── results/               # Output CSVs with per-turn per-agent data
├── analysis/
│   ├── result_explorer.py # Per-basket merge/gap primitives + plots
│   └── aggregate.py       # Cross-basket aggregation & genesis statistics
└── app.py                 # Streamlit explorer over the analysis library
```

## How it works

1. `generate_baskets.py` builds the counterfactual baskets (control / gender /
   country variants + a per-composition placebo of control-vs-control twins).
2. `experiment_runner.py` reads the experiment matrix from `config.py`.
3. For each condition it initializes agents via `agents.py` with the right
   model + role, then `orchestrator.py` runs the debate: genesis (t=0) → 4
   communication turns.
4. Each turn's JSON output is appended to a stage-prefixed CSV (checkpointing):
   `detection_*`, `mitigation_*`, `placebo_*`, `single_*`.
5. `analysis/aggregate.py` computes the genesis (t=0) gap statistics with the
   placebo as the empirical null; `app.py` renders them.

The **bias endpoint is the genesis turn (t=0)**. Later turns document the
dissolution of the gap into generic debate noise (indistinguishable from the
placebo band).

## Running

```bash
# Generate the basket dataset first
python generate_baskets.py

# Detection stage (bias measurement) for one condition, 3 seeds
python experiment_runner.py --stage detection --condition baseline --seeds 42,123,456

# Placebo noise floor (one run per composition)
python experiment_runner.py --stage placebo --condition baseline --seeds 42,123,456

# Mitigation (vaccines) / single-agent baseline
python experiment_runner.py --stage mitigation --condition baseline --seeds 42,123,456
python experiment_runner.py --stage single --seeds 42,123,456

# Explore results
streamlit run app.py
```

> **Placebo note:** the placebo is a per-*composition* noise floor (the
> instruction level does not affect twin-vs-twin baseline noise), so it is run
> once per composition, not once per ablation cell. The `placebo_*` CSVs
> currently on disk were generated from only **5 archetypes**; after
> regenerating the baskets with the full **21 archetypes**
> (`generate_baskets.py::PLACEBO_ARCHETYPE_COUNT = 21`) the placebo stage must
> be re-run (1 run per composition × 3 seeds).
