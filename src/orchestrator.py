"""
orchestrator.py - Runs a single investment committee debate.

Flow:
1. Genesis: Each agent independently evaluates the basket
2. Renaissance: For MAX_DEBATE_TURNS rounds, agents see others' responses and update
3. Termination: Final allocations are collected

Each turn is recorded for dynamic bias analysis (emergence, propagation, amplification).
"""

import json
import logging
import random
import hashlib
from datetime import datetime

from agents import Agent
from config import MAX_DEBATE_TURNS

logger = logging.getLogger(__name__)


def _stable_hash(s: str) -> int:
    """Stable hash for deterministic shuffles across Python processes."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _normalize_company_name(name: str) -> str:
    """Normalize a company name for exact expected-company validation."""
    return " ".join(str(name).strip().casefold().split())


def _subject_name_set(subject_field: str) -> set[str]:
    """Return exact normalized subject names; mixed baskets use ' & '."""
    return {
        _normalize_company_name(part)
        for part in str(subject_field).split(" & ")
        if part.strip()
    }


def _agent_role_key(agent_id: str) -> str:
    """Derive stable role key such as agent_1 from IDs like agent_1_llama."""
    parts = str(agent_id).split("_")
    if len(parts) >= 2 and parts[0] == "agent" and parts[1].isdigit():
        return "_".join(parts[:2])
    return parts[0] if parts else ""


def _canonicalize_companies_for_shuffle(
    companies: list[dict],
    subject_field: str,
) -> list[dict]:
    """
    Canonicalize before deterministic shuffle.

    Subjects are placed before fillers so pair_id-level shuffling puts the
    subject in the same final prompt position across Act 1 variants even when
    the country variant has a different subject name.
    """
    subject_names = _subject_name_set(subject_field)
    subjects = []
    fillers = []
    for company in companies:
        company_copy = dict(company)
        if _normalize_company_name(company_copy["name"]) in subject_names:
            subjects.append(company_copy)
        else:
            fillers.append(company_copy)

    subjects = sorted(subjects, key=lambda c: _normalize_company_name(c["name"]))
    fillers = sorted(fillers, key=lambda c: _normalize_company_name(c["name"]))
    return subjects + fillers


def run_debate(
    agents: list[Agent],
    basket: dict,
    basket_metadata: dict,
) -> list[dict]:
    """
    Run a full committee debate and return per-turn per-agent records.

    Args:
        agents: List of initialized Agent objects
        basket: Raw basket dict (companies will be shuffled here)
        basket_metadata: Dict with basket_id, pair_id, variant, sensitive_attr, etc.

    Returns:
        List of flat dicts (one per agent per turn), ready for DataFrame.
    """
    records = []
    turn_responses = {}  # {agent_id: latest response dict}

    # Pair-aligned deterministic shuffle: Act 1 control/gender/country variants
    # share pair_id, so prompt position cannot become a counterfactual confound.
    shuffle_key = basket_metadata.get("pair_id") or basket_metadata.get("basket_id")
    shuffle_seed = int(basket_metadata.get("seed", 42)) + _stable_hash(str(shuffle_key))
    canonical_companies = _canonicalize_companies_for_shuffle(
        basket["companies"],
        basket_metadata.get("subject_company", ""),
    )
    rng = random.Random(shuffle_seed)
    shuffled_companies = list(canonical_companies)
    rng.shuffle(shuffled_companies)
    shuffled_basket = {**basket, "companies": shuffled_companies}
    expected_companies = [c["name"] for c in shuffled_companies]

    # Track first subject position after shuffle. Mixed baskets declare two
    # subjects with " & ", but the legacy CSV keeps one integer position.
    subject_names = _subject_name_set(basket_metadata.get("subject_company", ""))
    subject_position = next(
        (i for i, c in enumerate(shuffled_companies)
         if _normalize_company_name(c["name"]) in subject_names),
        -1,
    )

    # FIX 1: Generate both prompt versions from the same shuffled order
    prompt_blind = format_basket_prompt_blind(shuffled_basket)
    prompt_full = format_basket_prompt_full(shuffled_basket)

    # =========================================================================
    # PHASE 1: GENESIS (Turn 0)
    # =========================================================================
    logger.info(f"[{basket_metadata['basket_id']}] Genesis phase...")

    for agent in agents:
        basket_prompt = prompt_blind if agent.blind else prompt_full
        response = agent.genesis(basket_prompt)
        response_records, cleaned_response = _records_from_response(
            basket_metadata=basket_metadata,
            agent=agent,
            turn=0,
            phase="genesis",
            response=response,
            expected_companies=expected_companies,
            subject_position=subject_position,
        )
        turn_responses[agent.agent_id] = cleaned_response
        records.extend(response_records)

    # =========================================================================
    # PHASE 2: RENAISSANCE (Turns 1..N)
    # =========================================================================
    for turn in range(1, MAX_DEBATE_TURNS + 1):
        logger.info(f"[{basket_metadata['basket_id']}] Turn {turn}...")

        new_responses = {}
        for agent in agents:
            # Collect other agents' latest responses
            other_responses = [
                {"agent_id": aid, **resp}
                for aid, resp in turn_responses.items()
                if aid != agent.agent_id
            ]

            basket_prompt = prompt_blind if agent.blind else prompt_full
            response = agent.respond(basket_prompt, other_responses, turn)
            response_records, cleaned_response = _records_from_response(
                basket_metadata=basket_metadata,
                agent=agent,
                turn=turn,
                phase="renaissance",
                response=response,
                expected_companies=expected_companies,
                subject_position=subject_position,
            )
            new_responses[agent.agent_id] = cleaned_response
            records.extend(response_records)

        # Update turn_responses for next round
        turn_responses = new_responses

    return records


def run_single_agent(
    agent: Agent,
    basket: dict,
    basket_metadata: dict,
) -> list[dict]:
    """Run one genesis-only baseline with the same validation as debates."""
    shuffle_key = basket_metadata.get("pair_id") or basket_metadata.get("basket_id")
    shuffle_seed = int(basket_metadata.get("seed", 42)) + _stable_hash(str(shuffle_key))
    canonical_companies = _canonicalize_companies_for_shuffle(
        basket["companies"],
        basket_metadata.get("subject_company", ""),
    )
    rng = random.Random(shuffle_seed)
    shuffled_companies = list(canonical_companies)
    rng.shuffle(shuffled_companies)
    shuffled_basket = {**basket, "companies": shuffled_companies}
    expected_companies = [c["name"] for c in shuffled_companies]

    subject_names = _subject_name_set(basket_metadata.get("subject_company", ""))
    subject_position = next(
        (i for i, c in enumerate(shuffled_companies)
         if _normalize_company_name(c["name"]) in subject_names),
        -1,
    )

    prompt_blind = format_basket_prompt_blind(shuffled_basket)
    prompt_full = format_basket_prompt_full(shuffled_basket)
    basket_prompt = prompt_blind if agent.blind else prompt_full
    response = agent.genesis(basket_prompt)

    records, _ = _records_from_response(
        basket_metadata=basket_metadata,
        agent=agent,
        turn=0,
        phase="single_genesis",
        response=response,
        expected_companies=expected_companies,
        subject_position=subject_position,
    )
    return records


def _records_from_response(
    basket_metadata: dict,
    agent: Agent,
    turn: int,
    phase: str,
    response: dict,
    expected_companies: list[str],
    subject_position: int,
) -> tuple[list[dict], dict]:
    """
    Convert one model response into exactly one row per expected company.

    Parse errors, missing decisions, and duplicates are recorded as quality flags
    instead of disappearing from the CSV, preserving internal-validity audits.
    """
    response_parse_error = bool(response.get("parse_error", False))
    error_message = response.get("error_message", "parse error or missing decision")
    expected_lookup = {_normalize_company_name(name): name for name in expected_companies}
    first_valid_by_name = {}
    duplicate_companies = []
    unexpected_companies = []

    if not response_parse_error:
        for decision in response.get("decisions", []):
            normalized = _normalize_company_name(decision.get("company_name", ""))
            canonical_name = expected_lookup.get(normalized)
            if not canonical_name:
                unexpected_companies.append(str(decision.get("company_name", "")))
                continue
            if canonical_name in first_valid_by_name:
                duplicate_companies.append(canonical_name)
                continue

            cleaned = dict(decision)
            cleaned["company_name"] = canonical_name
            first_valid_by_name[canonical_name] = cleaned

    missing_companies = [
        name for name in expected_companies
        if name not in first_valid_by_name
    ]

    validation_messages = []
    if response_parse_error:
        validation_messages.append(error_message)
    if missing_companies:
        validation_messages.append(f"Missing companies: {', '.join(missing_companies)}")
    if duplicate_companies:
        validation_messages.append(f"Duplicate companies: {', '.join(duplicate_companies)}")
    if unexpected_companies:
        validation_messages.append(f"Unexpected companies: {', '.join(unexpected_companies)}")
    validation_error = " | ".join(validation_messages)

    common_quality = {
        "pre_normalize_total": response.get("pre_normalize_total"),
        "was_normalized": response.get("was_normalized", False),
        "validation_error": validation_error,
        "missing_companies": ";".join(missing_companies),
        "duplicate_companies": ";".join(dict.fromkeys(duplicate_companies)),
    }

    records = []
    for company_name in expected_companies:
        if company_name in first_valid_by_name:
            decision = first_valid_by_name[company_name]
            row_parse_error = response_parse_error
        else:
            decision = {
                "company_name": company_name,
                "action": "ERROR",
                "allocation": 0,
                "reasoning": error_message,
            }
            row_parse_error = True

        records.append(_build_record(
            basket_metadata=basket_metadata,
            agent=agent,
            turn=turn,
            phase=phase,
            decision=decision,
            parse_error=row_parse_error,
            subject_position=subject_position,
            quality=common_quality,
        ))

    cleaned_response = dict(response)
    cleaned_response["decisions"] = [
        first_valid_by_name[name]
        for name in expected_companies
        if name in first_valid_by_name
    ]
    return records, cleaned_response


def _build_record(
    basket_metadata: dict,
    agent: Agent,
    turn: int,
    phase: str,
    decision: dict,
    parse_error: bool,
    subject_position: int = -1,
    quality: dict | None = None,
) -> dict:
    """Build a flat dict record for one agent's decision on one company at one turn."""
    quality = quality or {}
    subject_names = _subject_name_set(basket_metadata.get("subject_company", ""))
    company_norm = _normalize_company_name(decision.get("company_name", ""))
    is_subject = company_norm in subject_names

    return {
        # Experiment identifiers
        "timestamp": datetime.now().isoformat(),
        "basket_id": basket_metadata.get("basket_id", ""),
        "pair_id": basket_metadata.get("pair_id", ""),
        "variant": basket_metadata.get("variant", ""),
        "sensitive_attr": basket_metadata.get("sensitive_attr", ""),
        "sensitive_value": basket_metadata.get("sensitive_value", ""),
        "experiment_label": basket_metadata.get("experiment_label", ""),
        "composition": basket_metadata.get("composition", ""),
        "instruction_level": basket_metadata.get("instruction_level", ""),
        "protocol": basket_metadata.get("protocol", ""),
        "vaccine": basket_metadata.get("vaccine", "none"),
        "seed": basket_metadata.get("seed", 42),

        # Agent info
        "agent_id": agent.agent_id,
        "agent_model": agent.model.model_name,
        "agent_role": agent.role_prompt[:80] + "...",  # Truncated for readability
        "role_key": _agent_role_key(agent.agent_id),
        "is_blind": agent.blind,

        # Turn info
        "turn": turn,
        "phase": phase,

        # Decision
        "company_name": decision.get("company_name", ""),
        "action": decision.get("action", ""),
        "allocation": decision.get("allocation", 0),
        "reasoning": decision.get("reasoning", ""),

        # Subject tracking (FIX 6)
        "is_subject": is_subject,
        "subject_position": subject_position,

        # Quality flags
        "parse_error": parse_error,
        "pre_normalize_total": quality.get("pre_normalize_total"),
        "was_normalized": quality.get("was_normalized", False),
        "validation_error": quality.get("validation_error", ""),
        "missing_companies": quality.get("missing_companies", ""),
        "duplicate_companies": quality.get("duplicate_companies", ""),
    }


def format_basket_prompt_full(basket: dict) -> str:
    """
    Convert a basket dict into a FULL text prompt (all fields visible).
    Used by Sentiment Analyst and Risk Manager.
    """
    lines = [
        f"INVESTMENT BASKET (Budget: {BUDGET:,}€)",
        f"{'='*50}",
        "",
    ]

    for i, company in enumerate(basket["companies"], 1):
        lines.append(f"Company {i}: {company['name']}")
        lines.append(f"  Sector: {company.get('sector', 'N/A')}")
        lines.append(f"  Industry: {company.get('industry', 'N/A')}")
        lines.append(f"  Headquarters: {company.get('headquarters', 'N/A')}")
        lines.append(f"  CEO: {company.get('ceo', 'N/A')}")
        lines.append(f"  Revenue: {company.get('revenue', 'N/A')} ({company.get('revenue_growth', 'N/A')})")
        lines.append(f"  P/E Ratio: {company.get('pe_ratio', 'N/A')}")
        lines.append(f"  Debt-to-Equity: {company.get('debt_to_equity', 'N/A')}")
        lines.append(f"  Trailing EPS: {company.get('trailing_eps', 'N/A')}")
        lines.append(f"  News Sentiment: {company.get('news_sentiment', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def format_basket_prompt_blind(basket: dict) -> str:
    """
    Convert a basket dict into a BLIND text prompt (financials only).
    Used by Fundamental Analyst — excludes headquarters, CEO, and news_sentiment.
    """
    lines = [
        f"INVESTMENT BASKET (Budget: {BUDGET:,}€)",
        f"{'='*50}",
        "",
    ]

    for i, company in enumerate(basket["companies"], 1):
        lines.append(f"Company {i}: {company['name']}")
        lines.append(f"  Sector: {company.get('sector', 'N/A')}")
        lines.append(f"  Industry: {company.get('industry', 'N/A')}")
        lines.append(f"  Revenue: {company.get('revenue', 'N/A')} ({company.get('revenue_growth', 'N/A')})")
        lines.append(f"  P/E Ratio: {company.get('pe_ratio', 'N/A')}")
        lines.append(f"  Debt-to-Equity: {company.get('debt_to_equity', 'N/A')}")
        lines.append(f"  Trailing EPS: {company.get('trailing_eps', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


# Import here to avoid circular imports
from config import BUDGET
