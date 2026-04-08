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
from datetime import datetime

from agents import Agent
from config import MAX_DEBATE_TURNS

logger = logging.getLogger(__name__)


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

    # FIX 2: Shuffle companies ONCE so all agents see the same order
    shuffled_companies = list(basket["companies"])
    random.shuffle(shuffled_companies)
    shuffled_basket = {**basket, "companies": shuffled_companies}

    # FIX 2: Track subject position after shuffle
    subject_name = basket_metadata.get("subject_company", "")
    subject_position = next(
        (i for i, c in enumerate(shuffled_companies)
         if c["name"] == subject_name),
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
        turn_responses[agent.agent_id] = response

        # Record each decision as a flat row
        for decision in response.get("decisions", []):
            records.append(_build_record(
                basket_metadata=basket_metadata,
                agent=agent,
                turn=0,
                phase="genesis",
                decision=decision,
                parse_error=response.get("parse_error", False),
                subject_position=subject_position,
            ))

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
            new_responses[agent.agent_id] = response

            for decision in response.get("decisions", []):
                records.append(_build_record(
                    basket_metadata=basket_metadata,
                    agent=agent,
                    turn=turn,
                    phase="renaissance",
                    decision=decision,
                    parse_error=response.get("parse_error", False),
                    subject_position=subject_position,
                ))

        # Update turn_responses for next round
        turn_responses = new_responses

    return records


def _build_record(
    basket_metadata: dict,
    agent: Agent,
    turn: int,
    phase: str,
    decision: dict,
    parse_error: bool,
    subject_position: int = -1,
) -> dict:
    """Build a flat dict record for one agent's decision on one company at one turn."""
    # FIX 6: Fuzzy match for is_subject (models may abbreviate company names)
    subject_name = basket_metadata.get("subject_company", "").lower()
    company_name = decision.get("company_name", "").lower()
    is_subject = (
        (subject_name in company_name or company_name in subject_name)
        if subject_name and company_name
        else False
    )

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
