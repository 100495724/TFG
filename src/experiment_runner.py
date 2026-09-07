"""
experiment_runner.py - Main entry point for running the experiment suite.

Features:
- Iterates over the ablation plan
- Checkpoints results after every single basket (crash-safe)
- Supports resuming from where it left off
- Logs progress and errors

Stages (``--stage``):
- ``detection``  : bias detection runs (endpoint is the genesis turn t=0).
- ``mitigation`` : same baskets with passive/active vaccines applied.
- ``placebo``    : control-vs-control noise floor, one run per composition.
- ``single``     : single-agent genesis baseline.

NOTE on placebo: the placebo is a per-COMPOSITION noise floor (instruction
level does not affect twin-vs-twin baseline noise), so it is run once per
composition. The ``placebo_*`` CSVs currently on disk were generated from only
5 archetypes; after regenerating baskets with the full 21 archetypes (see
``generate_baskets.py::PLACEBO_ARCHETYPE_COUNT``) the placebo stage must be
re-run (1 run per composition x 3 seeds).
"""

import os
import csv
import json
import glob
import logging
import argparse
from datetime import datetime
from pathlib import Path

from config import (
    MODEL_ENDPOINTS, INFERENCE_PARAMS, COMPOSITIONS, INSTRUCTION_LEVELS,
    ABLATION_PLAN, VACCINES, RESULTS_DIR, BASKETS_DIR, REPETITION_SEEDS,
)
from models import create_model
from agents import Agent
from orchestrator import run_debate, run_single_agent

LOGS_DIR = Path("logs")
EXPERIMENT_LOG_PATH = LOGS_DIR / "experiment.log"


def _configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if any(getattr(handler, "_tfg_experiment_runner", False)
           for handler in root_logger.handlers):
        return

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in [
        logging.StreamHandler(),
        logging.FileHandler(EXPERIMENT_LOG_PATH, encoding="utf-8"),
    ]:
        handler.setFormatter(formatter)
        handler._tfg_experiment_runner = True
        root_logger.addHandler(handler)


_configure_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# CSV CHECKPOINTING
# =============================================================================
CSV_FIELDNAMES = [
    "timestamp", "basket_id", "pair_id", "variant", "sensitive_attr",
    "sensitive_value", "experiment_label", "composition", "instruction_level",
    "protocol", "vaccine", "seed", "agent_id", "agent_model", "agent_role",
    "role_key", "is_blind", "turn", "phase", "company_name",
    "real_company_name", "ticker", "action", "allocation", "reasoning",
    "is_subject", "subject_position", "parse_error",
    "pre_normalize_total", "was_normalized", "validation_error",
    "missing_companies", "duplicate_companies",
]

SINGLE_AGENT_MODELS = ["llama-3.1-8b", "qwen-2.5-7b", "mistral-7b"]


def get_results_path(experiment_label: str, stage: str, max_turns: int | None = None) -> str:
    """Get the CSV path for a given experiment condition.

    ``stage`` is the output prefix: ``detection`` (bias detection at genesis)
    or ``mitigation`` (vaccine runs). The placebo stage has its own helper.

    ``max_turns`` is the effective renaissance-turn count for this run (Tarea
    C.1). When it is 0 (genesis only), the filename carries a ``_genesisonly``
    suffix so a scope change is visible in the filename itself and two runs of
    different scope never collide/append into the same CSV.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "_genesisonly" if max_turns == 0 else ""
    return os.path.join(RESULTS_DIR, f"{stage}_{experiment_label}{suffix}.csv")


def get_single_results_path(model_id: str, seed: int) -> str:
    """Get the CSV path for a single-agent baseline."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"single_{model_id}_seed{seed}.csv")


def get_placebo_results_path(label: str, seed: int) -> str:
    """Get the CSV path for placebo committee runs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"placebo_{label}_seed{seed}.csv")


def append_records_to_csv(filepath: str, records: list[dict]):
    """Append records to CSV, creating headers if file is new."""
    file_exists = os.path.exists(filepath)
    fieldnames = CSV_FIELDNAMES
    write_header = not file_exists or os.path.getsize(filepath) == 0
    if file_exists and not write_header:
        with open(filepath, "r", encoding="utf-8", newline="") as existing:
            reader = csv.reader(existing)
            existing_header = next(reader, [])
        if existing_header and existing_header != CSV_FIELDNAMES:
            missing_new_fields = [
                field for field in CSV_FIELDNAMES
                if field not in existing_header
            ]
            if missing_new_fields:
                logger.warning(
                    "Existing CSV %s lacks new columns %s; appending with its "
                    "current header to avoid rewriting prior results.",
                    filepath,
                    missing_new_fields,
                )
                fieldnames = existing_header

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def get_completed_baskets(filepath: str) -> set:
    """Read which basket_ids have already been completed (for resume)."""
    if not os.path.exists(filepath):
        return set()
    completed = set()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(row["basket_id"])
    return completed


# =============================================================================
# BASKET LOADING
# =============================================================================
def load_baskets(baskets_dir: str) -> list[dict]:
    """Load all basket JSON files from the data directory."""
    basket_files = sorted(glob.glob(os.path.join(baskets_dir, "*.json")))
    baskets = []
    for f in basket_files:
        with open(f, "r", encoding="utf-8") as fp:
            baskets.append(json.load(fp))
    logger.info(f"Loaded {len(baskets)} baskets from {baskets_dir}")
    return baskets


# =============================================================================
# AGENT INITIALIZATION
# =============================================================================
def initialize_agents(
    composition_key: str,
    instruction_key: str,
    protocol: str,
    vaccine_key: str = "none",
    seed: int = 42,
) -> list[Agent]:
    """Create 3 agents based on the experimental condition."""

    model_ids = COMPOSITIONS[composition_key]
    instructions = INSTRUCTION_LEVELS[instruction_key]
    vaccine_text = VACCINES.get(vaccine_key, "")

    # Override seed in params
    params = {**INFERENCE_PARAMS, "seed": seed}

    agents = []
    agent_roles = ["agent_1", "agent_2", "agent_3"]

    for i, (model_id, role_key) in enumerate(zip(model_ids, agent_roles)):
        model = create_model(model_id, MODEL_ENDPOINTS, params)
        instr = instructions[role_key]
        # Support both old string format and new dict format {"prompt": ..., "blind": ...}
        if isinstance(instr, dict):
            role_prompt = instr["prompt"]
            blind = instr.get("blind", False)
        else:
            role_prompt = instr
            blind = False
        agent = Agent(
            agent_id=f"{role_key}_{model_id}",
            model=model,
            role_prompt=role_prompt,
            protocol=protocol,
            vaccine=vaccine_text,
            blind=blind,
        )
        agents.append(agent)

    return agents


def initialize_single_agent(
    model_id: str,
    instruction_key: str = "level_0_neutral",
    seed: int = 42,
) -> Agent:
    """Create one neutral baseline agent for model-only bias checks."""
    params = {**INFERENCE_PARAMS, "seed": seed}
    model = create_model(model_id, MODEL_ENDPOINTS, params)
    instr = INSTRUCTION_LEVELS[instruction_key]["agent_1"]
    if isinstance(instr, dict):
        role_prompt = instr["prompt"]
        blind = instr.get("blind", False)
    else:
        role_prompt = instr
        blind = False

    return Agent(
        agent_id=f"single_{model_id}",
        model=model,
        role_prompt=role_prompt,
        protocol="single",
        vaccine="",
        blind=blind,
    )


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================
def run_detection(baskets: list[dict], condition: dict, seed: int, max_turns: int | None = None):
    """
    Detection - Run parallel baskets to measure the Allocation Gap.

    The bias endpoint is the genesis turn (t=0); later turns document the
    dissolution of the gap into generic debate noise. ``max_turns`` (Tarea
    C.1) overrides the renaissance-turn count without mutating
    ``config.MAX_DEBATE_TURNS``; default (None) keeps the full-length run.
    """
    label = condition["label"]
    csv_path = get_results_path(f"{label}_seed{seed}", "detection", max_turns=max_turns)
    completed = get_completed_baskets(csv_path)

    agents = initialize_agents(
        composition_key=condition["composition"],
        instruction_key=condition["instruction"],
        protocol=condition["protocol"],
        seed=seed,
    )

    total = len(baskets)
    for i, basket in enumerate(baskets):
        bid = basket["basket_id"]
        if bid in completed:
            logger.info(f"  [{i+1}/{total}] Skipping {bid} (already completed)")
            continue

        logger.info(f"  [{i+1}/{total}] Running {bid}...")

        metadata = {
            "basket_id": bid,
            "pair_id": basket.get("pair_id", ""),
            "variant": basket.get("variant", ""),
            "sensitive_attr": basket.get("sensitive_attr", ""),
            "sensitive_value": basket.get("sensitive_value", ""),
            "subject_company": basket.get("subject_company", ""),
            "experiment_label": label,
            "composition": condition["composition"],
            "instruction_level": condition["instruction"],
            "protocol": condition["protocol"],
            "vaccine": "none",
            "seed": seed,
        }

        try:
            records = run_debate(agents, basket, metadata, max_turns=max_turns)
            append_records_to_csv(csv_path, records)
            logger.info(f"  [{i+1}/{total}] {bid} complete: {len(records)} records saved")
        except Exception as e:
            logger.error(f"  [{i+1}/{total}] {bid} FAILED: {e}")
            continue  # Skip and continue with next basket


def run_mitigation(
    baskets: list[dict],
    condition: dict,
    seed: int,
    vaccine_keys: list[str] | None = None,
    max_turns: int | None = None,
):
    """
    Mitigation - Run the detection baskets but with vaccines applied.

    Tarea C: ``vaccine_keys`` restricts which vaccines to run (default both
    passive and active), and ``max_turns`` overrides the renaissance-turn
    count for this run (genesis-only mitigation passes 0) WITHOUT mutating
    ``config.MAX_DEBATE_TURNS`` - it flows explicitly into ``run_debate`` and
    into the CSV filename (``get_results_path``, ``_genesisonly`` suffix).
    """
    label = condition["label"]
    vaccine_keys = vaccine_keys or ["passive", "active"]

    for vaccine_key in vaccine_keys:
        csv_path = get_results_path(f"{label}_{vaccine_key}_seed{seed}", "mitigation", max_turns=max_turns)
        completed = get_completed_baskets(csv_path)

        agents = initialize_agents(
            composition_key=condition["composition"],
            instruction_key=condition["instruction"],
            protocol=condition["protocol"],
            vaccine_key=vaccine_key,
            seed=seed,
        )

        for i, basket in enumerate(baskets):
            bid = basket["basket_id"]
            if bid in completed:
                continue

            logger.info(f"  [Mitigation-{vaccine_key} {i+1}/{len(baskets)}] Running {bid}...")
            metadata = {
                "basket_id": bid,
                "pair_id": basket.get("pair_id", ""),
                "variant": basket.get("variant", ""),
                "sensitive_attr": basket.get("sensitive_attr", ""),
                "sensitive_value": basket.get("sensitive_value", ""),
                "subject_company": basket.get("subject_company", ""),
                "experiment_label": f"{label}_vaccine_{vaccine_key}",
                "composition": condition["composition"],
                "instruction_level": condition["instruction"],
                "protocol": condition["protocol"],
                "vaccine": vaccine_key,
                "seed": seed,
            }

            try:
                records = run_debate(agents, basket, metadata, max_turns=max_turns)
                append_records_to_csv(csv_path, records)
            except Exception as e:
                logger.error(f"  [Mitigation-{vaccine_key}] {bid} FAILED: {e}")


def run_single_agent_baseline(
    baskets: list[dict],
    model_id: str,
    seed: int,
    instruction_key: str = "level_0_neutral",
):
    """
    Single-agent baseline: one neutral analyst, genesis turn only.

    This separates model-level bias from multi-agent committee emergence.
    """
    csv_path = get_single_results_path(model_id, seed)
    completed = get_completed_baskets(csv_path)
    agent = initialize_single_agent(model_id, instruction_key=instruction_key, seed=seed)

    for i, basket in enumerate(baskets):
        bid = basket["basket_id"]
        if bid in completed:
            logger.info(f"  [Single {model_id} {i+1}/{len(baskets)}] Skipping {bid}")
            continue

        logger.info(f"  [Single {model_id} {i+1}/{len(baskets)}] Running {bid}...")
        metadata = {
            "basket_id": bid,
            "pair_id": basket.get("pair_id", ""),
            "variant": basket.get("variant", ""),
            "sensitive_attr": basket.get("sensitive_attr", ""),
            "sensitive_value": basket.get("sensitive_value", ""),
            "subject_company": basket.get("subject_company", ""),
            "experiment_label": f"single_{model_id}",
            "composition": f"single_{model_id}",
            "instruction_level": instruction_key,
            "protocol": "single",
            "vaccine": "none",
            "seed": seed,
        }

        try:
            records = run_single_agent(agent, basket, metadata)
            append_records_to_csv(csv_path, records)
            logger.info(f"  [Single] {bid} complete: {len(records)} records saved")
        except Exception as e:
            logger.error(f"  [Single] {bid} FAILED: {e}")


def run_placebo(baskets_placebo: list[dict], condition: dict, seed: int):
    """Run placebo control-vs-control baskets with a committee condition."""
    label = condition["label"]
    csv_path = get_placebo_results_path(label, seed)
    completed = get_completed_baskets(csv_path)

    agents = initialize_agents(
        composition_key=condition["composition"],
        instruction_key=condition["instruction"],
        protocol=condition["protocol"],
        seed=seed,
    )

    for i, basket in enumerate(baskets_placebo):
        bid = basket["basket_id"]
        if bid in completed:
            logger.info(f"  [Placebo {i+1}/{len(baskets_placebo)}] Skipping {bid}")
            continue

        logger.info(f"  [Placebo {i+1}/{len(baskets_placebo)}] Running {bid}...")
        metadata = {
            "basket_id": bid,
            "pair_id": basket.get("pair_id", ""),
            "variant": basket.get("variant", ""),
            "sensitive_attr": basket.get("sensitive_attr", ""),
            "sensitive_value": basket.get("sensitive_value", ""),
            "subject_company": basket.get("subject_company", ""),
            "experiment_label": f"placebo_{label}",
            "composition": condition["composition"],
            "instruction_level": condition["instruction"],
            "protocol": condition["protocol"],
            "vaccine": "none",
            "seed": seed,
        }

        try:
            records = run_debate(agents, basket, metadata)
            append_records_to_csv(csv_path, records)
            logger.info(f"  [Placebo] {bid} complete: {len(records)} records saved")
        except Exception as e:
            logger.error(f"  [Placebo] {bid} FAILED: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MAS Bias TFG Experiment Runner")
    parser.add_argument(
        "--stage", type=str, default="detection",
        choices=["detection", "mitigation", "placebo", "single", "all"],
        help=(
            "Which stage to run "
            "(detection=bias detection at genesis, mitigation=vaccine runs, "
            "placebo=control-vs-control noise floor, "
            "single=single-agent genesis baseline, all=detection+mitigation)"
        )
    )
    parser.add_argument(
        "--condition", type=str, default="all",
        help="Specific condition label to run, or 'all' for full ablation plan"
    )
    parser.add_argument(
        "--seeds", type=str, default="42",
        help="Comma-separated seeds (e.g., '42' for draft, '42,123,456' for final)"
    )
    parser.add_argument(
        "--baskets-dir", type=str, default=BASKETS_DIR,
        help="Path to basket JSON files"
    )
    parser.add_argument(
        "--single-instruction", type=str, default="level_0_neutral",
        help="Instruction level for --stage single (default: level_0_neutral)"
    )
    parser.add_argument(
        "--composition", type=str, default=None,
        help=(
            "Restrict to one composition key (e.g. homo_llama), applied on top "
            "of --condition. Combine with --instruction-levels for --stage "
            "mitigation (Tarea C.2)."
        )
    )
    parser.add_argument(
        "--instruction-levels", type=str, default=None,
        help=(
            "Comma-separated instruction levels to restrict to "
            "(e.g. level_0_neutral,level_1_professional). Tarea C.2."
        )
    )
    parser.add_argument(
        "--vaccines", type=str, default=None,
        help="Comma-separated vaccine keys for --stage mitigation (default: passive,active). Tarea C.2."
    )
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help=(
            "Override MAX_DEBATE_TURNS for this run without mutating the "
            "global constant (0 = genesis only). Passed explicitly down the "
            "call chain; the effective scope is recorded in the CSV filename "
            "(_genesisonly suffix when 0) and should also be noted in the "
            "MANIFEST when building memoria outputs from it. Tarea C.1."
        )
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    start_time = datetime.now()
    logger.info(f"Stage to run: {args.stage}")

    if args.stage == "single":
        baskets = load_baskets(args.baskets_dir)
        if not baskets:
            logger.error(f"No baskets found in {args.baskets_dir}. Generate dataset first.")
            return
        logger.info(
            f"Starting single-agent baselines: {len(SINGLE_AGENT_MODELS)} models x {len(seeds)} seeds"
        )
        for seed in seeds:
            for model_id in SINGLE_AGENT_MODELS:
                logger.info(f"\n{'='*60}")
                logger.info(f"SINGLE MODEL: {model_id} | SEED: {seed}")
                logger.info(f"{'='*60}")
                run_single_agent_baseline(
                    baskets,
                    model_id,
                    seed,
                    instruction_key=args.single_instruction,
                )
        elapsed = datetime.now() - start_time
        logger.info(f"\nAll single-agent baselines completed in {elapsed}")
        return

    if args.stage == "placebo":
        if args.condition == "all":
            conditions = [c for c in ABLATION_PLAN if c["label"] == "baseline"]
        else:
            conditions = [c for c in ABLATION_PLAN if c["label"] == args.condition]
            if not conditions:
                logger.error(f"Condition '{args.condition}' not found in ablation plan.")
                return

        placebo_dir = os.path.join(args.baskets_dir, "placebo")
        baskets_placebo = load_baskets(placebo_dir)
        if not baskets_placebo:
            logger.error(f"No placebo baskets found in {placebo_dir}. Run generate_baskets.py first.")
            return
        logger.info(f"Starting placebo runs: {len(conditions)} conditions x {len(seeds)} seeds")
        for condition in conditions:
            for seed in seeds:
                label = condition["label"]
                logger.info(f"\n{'='*60}")
                logger.info(f"PLACEBO CONDITION: {label} | SEED: {seed}")
                logger.info(f"{'='*60}")
                run_placebo(baskets_placebo, condition, seed)
        elapsed = datetime.now() - start_time
        logger.info(f"\nAll placebo runs completed in {elapsed}")
        return

    # Load baskets
    baskets = load_baskets(args.baskets_dir)
    if not baskets:
        logger.error(f"No baskets found in {args.baskets_dir}. Generate dataset first.")
        return

    # Determine which conditions to run
    if args.condition == "all":
        conditions = ABLATION_PLAN
    else:
        conditions = [c for c in ABLATION_PLAN if c["label"] == args.condition]
        if not conditions:
            logger.error(f"Condition '{args.condition}' not found in ablation plan.")
            return

    # Tarea C.2: restrict composition / instruction levels on top of --condition,
    # so a mitigation plan (e.g. homo_llama x {level_0, level_1}) can be run
    # without listing every condition label by hand.
    if args.composition:
        conditions = [c for c in conditions if c["composition"] == args.composition]
    if args.instruction_levels:
        wanted_levels = {level.strip() for level in args.instruction_levels.split(",") if level.strip()}
        conditions = [c for c in conditions if c["instruction"] in wanted_levels]
    if not conditions:
        logger.error("No conditions match --condition/--composition/--instruction-levels filters.")
        return

    vaccine_keys = (
        [v.strip() for v in args.vaccines.split(",") if v.strip()] if args.vaccines else None
    )

    # Run experiments
    logger.info(f"Starting experiments: {len(conditions)} conditions x {len(seeds)} seeds")

    for condition in conditions:
        for seed in seeds:
            label = condition["label"]
            logger.info(f"\n{'='*60}")
            logger.info(f"CONDITION: {label} | SEED: {seed}")
            logger.info(f"{'='*60}")

            if args.stage in ("detection", "all"):
                logger.info(f"--- Detection ---")
                run_detection(baskets, condition, seed, max_turns=args.max_turns)

            if args.stage in ("mitigation", "all"):
                logger.info(f"--- Mitigation ---")
                run_mitigation(
                    baskets, condition, seed,
                    vaccine_keys=vaccine_keys, max_turns=args.max_turns,
                )

    elapsed = datetime.now() - start_time
    logger.info(f"\nAll experiments completed in {elapsed}")


if __name__ == "__main__":
    main()
