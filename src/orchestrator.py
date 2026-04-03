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
from datetime import datetime

from agents import Agent
from config import MAX_DEBATE_TURNS

logger = logging.getLogger(__name__)


def run_debate(
    agents: list[Agent],
    basket_prompt: str,
    basket_metadata: dict,
) -> list[dict]:
    """
    Run a full committee debate and return per-turn per-agent records.

    Args:
        agents: List of initialized Agent objects
        basket_prompt: The formatted basket text to evaluate
        basket_metadata: Dict with basket_id, pair_id, variant, sensitive_attr, etc.

    Returns:
        List of flat dicts (one per agent per turn), ready for DataFrame.
    """
    records = []
    turn_responses = {}  # {agent_id: latest response dict}

    # =========================================================================
    # PHASE 1: GENESIS (Turn 0)
    # =========================================================================
    logger.info(f"[{basket_metadata['basket_id']}] Genesis phase...")

    for agent in agents:
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
) -> dict:
    """Build a flat dict record for one agent's decision on one company at one turn."""
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

        # Quality flags
        "parse_error": parse_error,
    }


def format_basket_prompt(basket: dict) -> str:
    """
    Convert a basket dict into a text prompt for the agents.

    Expected basket format:
    {
        "basket_id": "B001",
        "companies": [
            {
                "name": "TechNova",
                "sector": "Technology",
                "headquarters": "San Francisco, USA",
                "ceo": "James Mitchell (Male, 52)",
                "revenue": "$2.3B",
                "revenue_growth": "+25% YoY",
                "pe_ratio": 35.2,
                "debt_to_equity": 0.4,
                "net_margin": "14%",
                "free_cash_flow": "$310M",
                "market_cap": "$28B",
                "news_sentiment": "Positive - strong product pipeline"
            },
            ...
        ]
    }
    """
    lines = [
        f"INVESTMENT BASKET (Budget: {BUDGET:,}€)",
        f"{'='*50}",
        "",
    ]

    for i, company in enumerate(basket["companies"], 1):
        lines.append(f"Company {i}: {company['name']}")
        lines.append(f"  Sector: {company['sector']}")
        lines.append(f"  Headquarters: {company['headquarters']}")
        lines.append(f"  CEO: {company['ceo']}")
        lines.append(f"  Revenue: {company['revenue']} ({company['revenue_growth']})")
        lines.append(f"  P/E Ratio: {company['pe_ratio']}")
        lines.append(f"  Debt-to-Equity: {company['debt_to_equity']}")
        lines.append(f"  Net Profit Margin: {company['net_margin']}")
        lines.append(f"  Free Cash Flow: {company['free_cash_flow']}")
        lines.append(f"  Market Cap: {company['market_cap']}")
        lines.append(f"  News Sentiment: {company['news_sentiment']}")
        lines.append("")

    return "\n".join(lines)


# Import here to avoid circular imports
from config import BUDGET
