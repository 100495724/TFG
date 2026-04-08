"""
experiment_runner.py - Main entry point for running the experiment suite.

Features:
- Iterates over the ablation plan
- Checkpoints results after every single basket (crash-safe)
- Supports resuming from where it left off
- Logs progress and errors
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
from orchestrator import run_debate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# CSV CHECKPOINTING
# =============================================================================
CSV_FIELDNAMES = [
    "timestamp", "basket_id", "pair_id", "variant", "sensitive_attr",
    "sensitive_value", "experiment_label", "composition", "instruction_level",
    "protocol", "vaccine", "seed", "agent_id", "agent_model", "agent_role",
    "turn", "phase", "company_name", "action", "allocation", "reasoning",
    "is_subject", "subject_position", "parse_error",
]


def get_results_path(experiment_label: str, act: str) -> str:
    """Get the CSV path for a given experiment condition."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{act}_{experiment_label}.csv")


def append_records_to_csv(filepath: str, records: list[dict]):
    """Append records to CSV, creating headers if file is new."""
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
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


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================
def run_act1(baskets: list[dict], condition: dict, seed: int):
    """
    Act 1: Detection - Run parallel baskets to measure Allocation Gap.
    """
    label = condition["label"]
    csv_path = get_results_path(f"{label}_seed{seed}", "act1")
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
            records = run_debate(agents, basket, metadata)
            append_records_to_csv(csv_path, records)
            logger.info(f"  [{i+1}/{total}] {bid} complete: {len(records)} records saved")
        except Exception as e:
            logger.error(f"  [{i+1}/{total}] {bid} FAILED: {e}")
            continue  # Skip and continue with next basket


def run_act2(baskets_mixed: list[dict], condition: dict, seed: int):
    """
    Act 2: Performative Fairness - Run baskets with twins in same prompt.
    Uses the same agents as baseline but with mixed baskets.
    """
    label = condition["label"]
    csv_path = get_results_path(f"{label}_seed{seed}", "act2")
    completed = get_completed_baskets(csv_path)

    agents = initialize_agents(
        composition_key=condition["composition"],
        instruction_key=condition["instruction"],
        protocol=condition["protocol"],
        seed=seed,
    )

    for i, basket in enumerate(baskets_mixed):
        bid = basket["basket_id"]
        if bid in completed:
            continue

        logger.info(f"  [Act2 {i+1}/{len(baskets_mixed)}] Running {bid}...")
        metadata = {
            "basket_id": bid,
            "pair_id": basket.get("pair_id", ""),
            "variant": "mixed",
            "sensitive_attr": basket.get("sensitive_attr", ""),
            "sensitive_value": "both_visible",
            "subject_company": basket.get("subject_company", ""),
            "experiment_label": f"{label}_performative",
            "composition": condition["composition"],
            "instruction_level": condition["instruction"],
            "protocol": condition["protocol"],
            "vaccine": "none",
            "seed": seed,
        }

        try:
            records = run_debate(agents, basket, metadata)
            append_records_to_csv(csv_path, records)
        except Exception as e:
            logger.error(f"  [Act2] {bid} FAILED: {e}")


def run_act3(baskets: list[dict], condition: dict, seed: int):
    """
    Act 3: Mitigation - Run Act1 baskets but with vaccines applied.
    """
    label = condition["label"]

    for vaccine_key in ["passive", "active"]:
        csv_path = get_results_path(f"{label}_{vaccine_key}_seed{seed}", "act3")
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

            logger.info(f"  [Act3-{vaccine_key} {i+1}/{len(baskets)}] Running {bid}...")
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
                records = run_debate(agents, basket, metadata)
                append_records_to_csv(csv_path, records)
            except Exception as e:
                logger.error(f"  [Act3-{vaccine_key}] {bid} FAILED: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MAS Bias TFG Experiment Runner")
    parser.add_argument(
        "--act", type=str, default="1",
        choices=["1", "2", "3", "all"],
        help="Which experimental act to run (1=detection, 2=performative, 3=mitigation, all)"
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
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

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

    # Run experiments
    start_time = datetime.now()
    logger.info(f"Starting experiments: {len(conditions)} conditions × {len(seeds)} seeds")
    logger.info(f"Acts to run: {args.act}")

    for condition in conditions:
        for seed in seeds:
            label = condition["label"]
            logger.info(f"\n{'='*60}")
            logger.info(f"CONDITION: {label} | SEED: {seed}")
            logger.info(f"{'='*60}")

            if args.act in ("1", "all"):
                logger.info(f"--- Act 1: Detection ---")
                run_act1(baskets, condition, seed)

            if args.act in ("2", "all"):
                # For Act 2, you need mixed baskets (twins in same prompt)
                # These would be in a separate directory or flagged in the JSON
                baskets_mixed_dir = os.path.join(args.baskets_dir, "mixed")
                if os.path.exists(baskets_mixed_dir):
                    baskets_mixed = load_baskets(baskets_mixed_dir)
                    logger.info(f"--- Act 2: Performative Fairness ---")
                    run_act2(baskets_mixed, condition, seed)
                else:
                    logger.warning(f"No mixed baskets found at {baskets_mixed_dir}, skipping Act 2")

            if args.act in ("3", "all"):
                logger.info(f"--- Act 3: Mitigation ---")
                run_act3(baskets, condition, seed)

    elapsed = datetime.now() - start_time
    logger.info(f"\nAll experiments completed in {elapsed}")


if __name__ == "__main__":
    main()
