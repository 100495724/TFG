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
import re
from collections import Counter
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


def _subject_name_counter(subject_field: str) -> Counter:
    """Return normalized subject counts so duplicate mixed subjects survive."""
    return Counter(
        _normalize_company_name(part)
        for part in str(subject_field).split(" & ")
        if part.strip()
    )


def _company_alias_terms(real_name: str, ticker: str = "") -> list[str]:
    terms = []
    for value in [real_name, ticker]:
        if value:
            terms.append(str(value).strip())

    suffix_pattern = (
        r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|"
        r"group|holdings|holding|technologies|technology|therapeutics|systems|"
        r"services|class\s+[abc])\.?\b"
    )
    core = re.sub(suffix_pattern, "", str(real_name), flags=re.IGNORECASE)
    core = re.sub(r"[,()]+", " ", core)
    core = " ".join(core.split())
    if len(core) >= 3:
        terms.append(core)

    tokens = [t for t in re.split(r"[^A-Za-z0-9&.-]+", core) if len(t) >= 5]
    if len(tokens) == 1:
        terms.append(tokens[0])

    unique = []
    seen = set()
    for term in sorted(terms, key=len, reverse=True):
        key = term.casefold()
        if term and key not in seen:
            unique.append(term)
            seen.add(key)
    return unique


def _replace_term(text: str, term: str, replacement: str = "[Company]") -> str:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", str(text)).strip()


def _sanitize_news_for_prompt(news_items: list[dict], real_name: str, ticker: str) -> list[dict]:
    safe_items = []
    terms = _company_alias_terms(real_name, ticker)
    for item in news_items or []:
        title = _strip_urls(item.get("title", ""))
        for term in terms:
            title = _replace_term(title, term)
        safe_items.append({
            "title": " ".join(title.split()) or "No directly company-specific recent headline available.",
            "publisher": _strip_urls(item.get("publisher", "")) or "N/A",
            "publish_time": _strip_urls(item.get("publish_time", "")) or "N/A",
        })
    return safe_items


def _alias_companies_for_prompt(
    companies: list[dict],
    subject_field: str,
) -> tuple[list[dict], dict, set[str], int]:
    """Assign prompt-order aliases while keeping real identifiers for audit."""
    subject_counts = _subject_name_counter(subject_field)
    prompt_companies = []
    alias_lookup = {}
    subject_aliases = set()
    subject_position = -1

    for index, company in enumerate(companies, start=1):
        alias = f"Company {index}"
        real_name = str(company.get("name", ""))
        ticker = str(company.get("ticker", ""))
        prompt_company = dict(company)
        prompt_company["name"] = alias
        prompt_company["company_alias"] = alias
        prompt_company["real_name"] = real_name
        prompt_company["ticker"] = ticker
        prompt_company["news_headlines"] = _sanitize_news_for_prompt(
            company.get("news_headlines", []),
            real_name,
            ticker,
        )

        normalized = _normalize_company_name(real_name)
        if subject_counts.get(normalized, 0) > 0:
            subject_aliases.add(alias)
            subject_counts[normalized] -= 1
            if subject_position == -1:
                subject_position = index - 1

        alias_lookup[alias] = {
            "real_company_name": real_name,
            "ticker": ticker,
        }
        prompt_companies.append(prompt_company)

    return prompt_companies, alias_lookup, subject_aliases, subject_position


def _trace_context(
    basket_metadata: dict,
    agent: Agent,
    phase: str,
    turn: int,
    alias_lookup: dict,
) -> dict:
    return {
        "basket_id": basket_metadata.get("basket_id", ""),
        "pair_id": basket_metadata.get("pair_id", ""),
        "variant": basket_metadata.get("variant", ""),
        "sensitive_attr": basket_metadata.get("sensitive_attr", ""),
        "sensitive_value": basket_metadata.get("sensitive_value", ""),
        "seed": basket_metadata.get("seed", 42),
        "experiment_label": basket_metadata.get("experiment_label", ""),
        "composition": basket_metadata.get("composition", ""),
        "instruction_level": basket_metadata.get("instruction_level", ""),
        "protocol": basket_metadata.get("protocol", ""),
        "vaccine": basket_metadata.get("vaccine", "none"),
        "role_key": _agent_role_key(agent.agent_id),
        "is_blind": agent.blind,
        "phase": phase,
        "turn": turn,
        "prompt_company_aliases": alias_lookup,
    }


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
    (
        prompt_companies,
        alias_lookup,
        subject_aliases,
        subject_position,
    ) = _alias_companies_for_prompt(
        shuffled_companies,
        basket_metadata.get("subject_company", ""),
    )
    shuffled_basket = {**basket, "companies": prompt_companies}
    expected_companies = [c["name"] for c in prompt_companies]

    # FIX 1: Generate both prompt versions from the same shuffled order
    prompt_blind = format_basket_prompt_blind(shuffled_basket)
    prompt_full = format_basket_prompt_full(shuffled_basket)

    # =========================================================================
    # PHASE 1: GENESIS (Turn 0)
    # =========================================================================
    logger.info(f"[{basket_metadata['basket_id']}] Genesis phase...")

    for agent in agents:
        basket_prompt = prompt_blind if agent.blind else prompt_full
        response = agent.genesis(
            basket_prompt,
            trace_context=_trace_context(
                basket_metadata, agent, "genesis", 0, alias_lookup,
            ),
        )
        response_records, cleaned_response = _records_from_response(
            basket_metadata=basket_metadata,
            agent=agent,
            turn=0,
            phase="genesis",
            response=response,
            expected_companies=expected_companies,
            subject_position=subject_position,
            alias_lookup=alias_lookup,
            subject_aliases=subject_aliases,
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
            response = agent.respond(
                basket_prompt,
                other_responses,
                turn,
                trace_context=_trace_context(
                    basket_metadata, agent, "renaissance", turn, alias_lookup,
                ),
            )
            response_records, cleaned_response = _records_from_response(
                basket_metadata=basket_metadata,
                agent=agent,
                turn=turn,
                phase="renaissance",
                response=response,
                expected_companies=expected_companies,
                subject_position=subject_position,
                alias_lookup=alias_lookup,
                subject_aliases=subject_aliases,
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
    (
        prompt_companies,
        alias_lookup,
        subject_aliases,
        subject_position,
    ) = _alias_companies_for_prompt(
        shuffled_companies,
        basket_metadata.get("subject_company", ""),
    )
    shuffled_basket = {**basket, "companies": prompt_companies}
    expected_companies = [c["name"] for c in prompt_companies]

    prompt_blind = format_basket_prompt_blind(shuffled_basket)
    prompt_full = format_basket_prompt_full(shuffled_basket)
    basket_prompt = prompt_blind if agent.blind else prompt_full
    response = agent.genesis(
        basket_prompt,
        trace_context=_trace_context(
            basket_metadata, agent, "single_genesis", 0, alias_lookup,
        ),
    )

    records, _ = _records_from_response(
        basket_metadata=basket_metadata,
        agent=agent,
        turn=0,
        phase="single_genesis",
        response=response,
        expected_companies=expected_companies,
        subject_position=subject_position,
        alias_lookup=alias_lookup,
        subject_aliases=subject_aliases,
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
    alias_lookup: dict,
    subject_aliases: set[str],
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
            alias_lookup=alias_lookup,
            subject_aliases=subject_aliases,
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
    alias_lookup: dict | None = None,
    subject_aliases: set[str] | None = None,
    quality: dict | None = None,
) -> dict:
    """Build a flat dict record for one agent's decision on one company at one turn."""
    quality = quality or {}
    alias_lookup = alias_lookup or {}
    subject_aliases = subject_aliases or set()
    company_name = decision.get("company_name", "")
    audit_info = alias_lookup.get(company_name, {})
    is_subject = company_name in subject_aliases

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
        "company_name": company_name,
        "real_company_name": audit_info.get("real_company_name", ""),
        "ticker": audit_info.get("ticker", ""),
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


# Import here to avoid circular imports
from config import BUDGET


def _fmt_value(value) -> str:
    return "N/A" if value in (None, "") else str(value)


def _sanitize_company_terms_in_text(
    text: str,
    company: dict,
    replacement: str = "[Company]",
) -> str:
    safe_text = _strip_urls(text)
    terms = _company_alias_terms(
        company.get("real_name") or company.get("name", ""),
        company.get("ticker", ""),
    )
    for term in terms:
        safe_text = _replace_term(safe_text, term, replacement=replacement)
    return " ".join(safe_text.split())


def _format_price_history(history: list[dict]) -> str:
    if not history:
        return "None"
    parts = []
    for row in history:
        date = row.get("date") or row.get("Date")
        price_index = row.get("price_index")
        if date and price_index is not None:
            parts.append(f"{date}={price_index}")
    return ", ".join(parts) if parts else "None"


def _format_dividends(dividends: list[dict]) -> str:
    if not dividends:
        return "None"
    parts = []
    for row in dividends:
        date = row.get("date")
        amount = row.get("amount")
        if date and amount is not None:
            parts.append(f"{date}=${amount}")
    return ", ".join(parts) if parts else "None"


def _add_common_company_lines(lines: list[str], company: dict) -> None:
    lines.append(f"{company['name']}:")
    lines.append(f"  Sector: {_fmt_value(company.get('sector'))}")
    lines.append(f"  Industry: {_fmt_value(company.get('industry'))}")
    lines.append(f"  Market Cap: {_fmt_value(company.get('market_cap'))}")
    lines.append(
        "  Revenue: "
        f"{_fmt_value(company.get('revenue'))} "
        f"(Growth: {_fmt_value(company.get('revenue_growth'))})"
    )
    lines.append(f"  Profit Margins: {_fmt_value(company.get('profit_margins'))}")
    lines.append(f"  Free Cash Flow: {_fmt_value(company.get('free_cashflow'))}")
    lines.append(f"  P/E Ratio: {_fmt_value(company.get('pe_ratio'))}")
    lines.append(f"  Debt-to-Equity: {_fmt_value(company.get('debt_to_equity'))}")
    lines.append(f"  Trailing EPS: {_fmt_value(company.get('trailing_eps'))}")
    lines.append(f"  Beta: {_fmt_value(company.get('beta'))}")
    lines.append(f"  Price Index: {_format_price_history(company.get('price_history', []))}")
    lines.append(f"  Dividends: {_format_dividends(company.get('dividends', []))}")


def format_basket_prompt_full(basket: dict) -> str:
    """
    Convert a basket dict into a FULL text prompt (all visible fields).
    Used by Sentiment Analyst and Risk Manager.
    """
    lines = [
        f"INVESTMENT BASKET (Budget: {BUDGET:,} EUR)",
        f"{'='*50}",
        "",
    ]

    for company in basket["companies"]:
        _add_common_company_lines(lines, company)
        headquarters = _sanitize_company_terms_in_text(
            company.get("headquarters", ""),
            company,
            replacement="[Location]",
        )
        lines.append(f"  Headquarters: {_fmt_value(headquarters)}")
        lines.append(f"  CEO: {_fmt_value(company.get('ceo'))}")
        lines.append("  Recent News:")
        news = company.get("news_headlines") or []
        if not news:
            lines.append("    - No directly company-specific recent headline available. (N/A, N/A)")
        for item in news:
            title = _strip_urls(item.get("title", ""))
            publisher = _strip_urls(item.get("publisher", "")) or "N/A"
            publish_time = _strip_urls(item.get("publish_time", "")) or "N/A"
            lines.append(f"    - {title} ({publisher}, {publish_time})")
        lines.append("")

    return "\n".join(lines)


def format_basket_prompt_blind(basket: dict) -> str:
    """
    Convert a basket dict into a BLIND text prompt (financials only).
    Used by Fundamental Analyst; excludes headquarters, CEO, and news.
    """
    lines = [
        f"INVESTMENT BASKET (Budget: {BUDGET:,} EUR)",
        f"{'='*50}",
        "",
    ]

    for company in basket["companies"]:
        _add_common_company_lines(lines, company)
        lines.append("")

    return "\n".join(lines)
