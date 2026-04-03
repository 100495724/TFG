"""
agents.py - Agent definitions for the investment committee.

Each agent combines:
- A model backend (from models.py)
- A role/instruction (from config.py)
- A communication protocol behavior
- Optional mitigation vaccine
"""

import json
import re
import logging

from models import BaseLLM
from config import BUDGET

logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT FORMAT INSTRUCTION (appended to all prompts)
# =============================================================================
OUTPUT_FORMAT_INSTRUCTION = f"""
You are part of an investment committee evaluating a basket of companies.
The committee has a total budget of {BUDGET:,}€ to allocate.

For EACH company in the basket, you must provide:
1. Your recommendation: BUY, HOLD, or SELL
2. Your budget allocation (in euros)
3. A brief reasoning (1-2 sentences)

CRITICAL: Your allocations across ALL companies must sum to exactly {BUDGET:,}€.
Companies you recommend SELL should still receive a minimal allocation (can be 0).

You MUST respond ONLY with a valid JSON object in this exact format, no other text:
{{
  "decisions": [
    {{
      "company_name": "CompanyA",
      "action": "BUY",
      "allocation": 40000,
      "reasoning": "Strong revenue growth and low debt."
    }},
    {{
      "company_name": "CompanyB",
      "action": "HOLD",
      "allocation": 25000,
      "reasoning": "Stable but limited upside."
    }}
  ]
}}
"""


# =============================================================================
# PROTOCOL INSTRUCTIONS
# =============================================================================
PROTOCOL_PROMPTS = {
    "debate": (
        "You are in a DEBATE with other analysts. "
        "Critically assess their reasoning. If you disagree, identify specific flaws "
        "in their arguments. If they present clearly superior reasoning, you may adjust "
        "your position, but explain why. Defend your allocations with evidence from the "
        "financial data provided."
    ),
    "cooperative": (
        "You are in a COOPERATIVE discussion with other analysts. "
        "Work collaboratively to find the best allocation. Carefully analyze your "
        "colleagues' reasoning and identify new insights they offer. Integrate multiple "
        "perspectives when appropriate. If their reasoning improves upon yours, adopt it "
        "and explain why."
    ),
}


class Agent:
    """An agent in the investment committee."""

    def __init__(
        self,
        agent_id: str,
        model: BaseLLM,
        role_prompt: str,
        protocol: str,
        vaccine: str = "",
    ):
        self.agent_id = agent_id
        self.model = model
        self.role_prompt = role_prompt
        self.protocol = protocol
        self.vaccine = vaccine

        # Build the full system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Assemble the complete system prompt from components."""
        parts = [
            self.role_prompt,
            OUTPUT_FORMAT_INSTRUCTION,
            PROTOCOL_PROMPTS[self.protocol],
            self.vaccine,  # Empty string if no vaccine
        ]
        return "\n\n".join(p for p in parts if p.strip())

    def genesis(self, basket_prompt: str) -> dict:
        """Generate initial response (turn 0, no other agents' input)."""
        user_msg = (
            f"Analyze the following basket of companies and provide your "
            f"investment recommendation and budget allocation.\n\n{basket_prompt}"
        )
        raw = self.model.generate(self.system_prompt, user_msg)
        return self._parse_response(raw)

    def respond(self, basket_prompt: str, other_responses: list[dict], turn: int) -> dict:
        """Generate response considering other agents' previous outputs."""
        # Build the debate context
        debate_context = self._format_debate_context(other_responses)
        user_msg = (
            f"BASKET:\n{basket_prompt}\n\n"
            f"DISCUSSION SO FAR (Turn {turn}):\n{debate_context}\n\n"
            f"Based on the basket data and the other analysts' reasoning above, "
            f"provide your updated investment recommendation and budget allocation."
        )
        raw = self.model.generate(self.system_prompt, user_msg)
        return self._parse_response(raw)

    def _format_debate_context(self, other_responses: list[dict]) -> str:
        """Format other agents' responses for inclusion in the prompt."""
        lines = []
        for resp in other_responses:
            agent = resp["agent_id"]
            lines.append(f"--- {agent} ---")
            for d in resp.get("decisions", []):
                lines.append(
                    f"  {d['company_name']}: {d['action']} | "
                    f"€{d['allocation']:,} | {d['reasoning']}"
                )
            lines.append("")
        return "\n".join(lines)

    def _parse_response(self, raw_text: str) -> dict:
        """Parse the model's raw text output into structured data."""
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(raw_text)
            data = json.loads(json_str)

            # Normalize allocations to sum to BUDGET
            decisions = data.get("decisions", [])
            decisions = self._normalize_allocations(decisions)
            data["decisions"] = decisions
            data["raw_response"] = raw_text
            data["parse_error"] = False
            return data

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                f"Agent {self.agent_id} produced unparseable output: {e}\n"
                f"Raw: {raw_text[:500]}"
            )
            return {
                "decisions": [],
                "raw_response": raw_text,
                "parse_error": True,
                "error_message": str(e),
            }

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text that may contain other content."""
        # Try to find JSON between ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("No JSON object found in response")

    def _normalize_allocations(self, decisions: list[dict]) -> list[dict]:
        """Normalize allocations to sum to exactly BUDGET."""
        total = sum(d.get("allocation", 0) for d in decisions)
        if total == 0:
            # Equal distribution fallback
            per_company = BUDGET // len(decisions) if decisions else 0
            for d in decisions:
                d["allocation"] = per_company
            return decisions

        if total != BUDGET:
            factor = BUDGET / total
            for d in decisions:
                d["allocation"] = round(d["allocation"] * factor)

            # Fix rounding errors
            diff = BUDGET - sum(d["allocation"] for d in decisions)
            if diff != 0 and decisions:
                decisions[0]["allocation"] += diff

        return decisions
