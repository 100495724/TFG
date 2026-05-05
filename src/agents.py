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
import math

from models import BaseLLM
from config import BUDGET

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"BUY", "HOLD", "SELL"}


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
# INDEPENDENCE INSTRUCTION
# =============================================================================
# Debate context stays visible, but this reduces lazy convergence/copying that
# can otherwise amplify bias through repeated wording rather than new evidence.
INDEPENDENCE_INSTRUCTION = """
CRITICAL INDEPENDENCE REQUIREMENT:
You must provide your OWN independent analysis using your assigned role's specific expertise.
Do NOT copy, paraphrase, or mechanically mirror other agents' reasoning, wording, or allocations.
In the genesis phase, make an independent assessment based only on the basket data and your role.
In later discussion turns, you may consider other agents' views, but you must evaluate them critically.
If you agree with another agent's conclusion, justify your agreement with DIFFERENT evidence from your own domain of expertise.
If you change your allocation, explain the role-specific evidence that caused the change.
Do not converge merely for consensus. Maintain a distinct analytical perspective unless the evidence clearly supports changing your view.
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
    "single": (
        "You are making an independent single-agent investment decision. "
        "There are no other analysts and no discussion. Base your recommendation "
        "only on the basket data and your assigned role."
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
        blind: bool = False,
    ):
        self.agent_id = agent_id
        self.model = model
        self.role_prompt = role_prompt
        self.protocol = protocol
        self.vaccine = vaccine
        self.blind = blind

        # Build the full system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Assemble the complete system prompt from components."""
        parts = [
            self.role_prompt,
            INDEPENDENCE_INSTRUCTION,
            OUTPUT_FORMAT_INSTRUCTION,
            PROTOCOL_PROMPTS[self.protocol],
            self.vaccine,  # Empty string if no vaccine
        ]
        return "\n\n".join(p for p in parts if p.strip())

    def genesis(self, basket_prompt: str) -> dict:
        """Generate initial response (turn 0, no other agents' input)."""
        max_retries = 3
        user_msg = (
            f"Analyze the following basket of companies and provide your "
            f"investment recommendation and budget allocation.\n\n{basket_prompt}"
        )

        last_error = None
        result = None
        for attempt in range(max_retries):
            if attempt > 0 and last_error:
                retry_msg = (
                    f"{user_msg}\n\n"
                    f"IMPORTANT: Your previous response could not be parsed or validated as valid JSON. "
                    f"Error: {last_error}\n"
                    f"You MUST respond with ONLY a valid JSON object, no markdown, no backticks wrapping, "
                    f"no explanation before or after. Just the raw JSON."
                )
            else:
                retry_msg = user_msg

            raw = self.model.generate(self.system_prompt, retry_msg)
            result = self._parse_response(raw)

            if not result.get("parse_error", False):
                return result

            last_error = result.get("error_message", "Invalid JSON")
            logger.warning(f"Agent {self.agent_id} parse retry {attempt + 1}/{max_retries}")

        logger.error(f"Agent {self.agent_id} failed to produce valid JSON after {max_retries} attempts")
        return result

    def respond(self, basket_prompt: str, other_responses: list[dict], turn: int) -> dict:
        """Generate response considering other agents' previous outputs."""
        max_retries = 3
        debate_context = self._format_debate_context(other_responses)
        user_msg = (
            f"BASKET:\n{basket_prompt}\n\n"
            f"DISCUSSION SO FAR (Turn {turn}):\n{debate_context}\n\n"
            f"Based on the basket data and the other analysts' reasoning above, "
            f"provide your updated investment recommendation and budget allocation."
        )

        last_error = None
        result = None
        for attempt in range(max_retries):
            if attempt > 0 and last_error:
                retry_msg = (
                    f"{user_msg}\n\n"
                    f"IMPORTANT: Your previous response could not be parsed or validated as valid JSON. "
                    f"Error: {last_error}\n"
                    f"You MUST respond with ONLY a valid JSON object, no markdown, no backticks wrapping, "
                    f"no explanation before or after. Just the raw JSON."
                )
            else:
                retry_msg = user_msg

            raw = self.model.generate(self.system_prompt, retry_msg)
            result = self._parse_response(raw)

            if not result.get("parse_error", False):
                return result

            last_error = result.get("error_message", "Invalid JSON")
            logger.warning(f"Agent {self.agent_id} respond retry {attempt + 1}/{max_retries}")

        logger.error(f"Agent {self.agent_id} failed to produce valid JSON after {max_retries} attempts (turn {turn})")
        return result

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
        """Parse and validate model output.

        Parse/validation failures are returned explicitly so the orchestrator can
        log placeholder rows instead of silently dropping an agent-turn.
        """
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(raw_text)
            data = json.loads(json_str)

            decisions = self._validate_decisions(data)
            pre_normalize_total = sum(d["allocation"] for d in decisions)

            # Normalize allocations to sum to BUDGET while preserving the
            # pre-normalization total for downstream quality auditing.
            decisions, was_normalized = self._normalize_allocations(decisions)
            data["decisions"] = decisions
            data["pre_normalize_total"] = pre_normalize_total
            data["was_normalized"] = was_normalized
            data["raw_response"] = raw_text
            data["parse_error"] = False
            return data

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"Agent {self.agent_id} produced unparseable output: {e}\n"
                f"Raw: {raw_text[:500]}"
            )
            return {
                "decisions": [],
                "raw_response": raw_text,
                "parse_error": True,
                "error_message": str(e),
                "pre_normalize_total": None,
                "was_normalized": False,
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

    def _validate_decisions(self, data: dict) -> list[dict]:
        """Validate the required JSON contract and coerce safe scalar fields."""
        if not isinstance(data, dict):
            raise ValueError("Top-level response must be a JSON object")

        decisions = data.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError("Response must contain a 'decisions' list")
        if not decisions:
            raise ValueError("'decisions' list is empty")

        validated = []
        required = ["company_name", "action", "allocation", "reasoning"]
        for i, decision in enumerate(decisions, start=1):
            if not isinstance(decision, dict):
                raise ValueError(f"Decision {i} must be an object")

            missing = [
                field for field in required
                if field not in decision or decision[field] in (None, "")
            ]
            if missing:
                raise ValueError(f"Decision {i} missing required fields: {missing}")

            company_name = str(decision["company_name"]).strip()
            action = str(decision["action"]).strip().upper()
            reasoning = str(decision["reasoning"]).strip()
            allocation = self._coerce_allocation(decision["allocation"], i)

            if not company_name:
                raise ValueError(f"Decision {i} has empty company_name")
            if action not in VALID_ACTIONS:
                raise ValueError(
                    f"Decision {i} action must be one of {sorted(VALID_ACTIONS)}, got {action!r}"
                )
            if not reasoning:
                raise ValueError(f"Decision {i} has empty reasoning")

            cleaned = dict(decision)
            cleaned.update({
                "company_name": company_name,
                "action": action,
                "allocation": allocation,
                "reasoning": reasoning,
            })
            validated.append(cleaned)

        return validated

    def _coerce_allocation(self, value, decision_index: int) -> float:
        """Return a numeric allocation or raise a validation error."""
        if isinstance(value, bool):
            raise ValueError(f"Decision {decision_index} allocation must be numeric")

        if isinstance(value, (int, float)):
            allocation = float(value)
        elif isinstance(value, str):
            cleaned = re.sub(r"[^0-9.+-]", "", value)
            if cleaned in {"", "+", "-", ".", "+.", "-."}:
                raise ValueError(f"Decision {decision_index} allocation must be numeric")
            allocation = float(cleaned)
        else:
            raise ValueError(f"Decision {decision_index} allocation must be numeric")

        if not math.isfinite(allocation):
            raise ValueError(f"Decision {decision_index} allocation must be finite")
        if allocation < 0:
            raise ValueError(f"Decision {decision_index} allocation must be non-negative")
        return allocation

    def _normalize_allocations(self, decisions: list[dict]) -> tuple[list[dict], bool]:
        """Normalize allocations to sum to exactly BUDGET."""
        total = sum(d.get("allocation", 0) for d in decisions)
        was_normalized = abs(total - BUDGET) > 1e-6
        if total == 0:
            # Equal distribution fallback
            per_company = BUDGET // len(decisions) if decisions else 0
            for d in decisions:
                d["allocation"] = per_company
            if decisions:
                diff = BUDGET - sum(d["allocation"] for d in decisions)
                decisions[0]["allocation"] += diff
            return decisions, was_normalized

        if was_normalized:
            factor = BUDGET / total
            for d in decisions:
                d["allocation"] = round(d["allocation"] * factor)

            # Fix rounding errors
            diff = BUDGET - sum(d["allocation"] for d in decisions)
            if diff != 0 and decisions:
                decisions[0]["allocation"] += diff
        else:
            for d in decisions:
                if isinstance(d["allocation"], float) and d["allocation"].is_integer():
                    d["allocation"] = int(d["allocation"])

        return decisions, was_normalized
