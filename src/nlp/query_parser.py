"""LangChain / Hugging Face / regex parser for structured routing constraints."""
import json
import re
from typing import Any

from src.config import HF_ZERO_SHOT_MODEL, OPENAI_API_KEY, USE_HF_PARSER

CONSTRAINT_LABELS = [
    "avoid highways",
    "prefer fewer turns",
    "minimize delay risk",
    "arrive quickly",
]


def empty_parse() -> dict[str, Any]:
    return {
        "origin": "",
        "destination": "",
        "avoid_highways": False,
        "prefer_fewer_turns": False,
        "minimize_delay_risk": False,
        "max_travel_time_min": None,
        "preferences": "",
    }


def _constraints_from_text(query: str) -> dict[str, Any]:
    q = query.lower()
    max_t = None
    m = re.search(r"within\s+(\d+)\s*min", q)
    if m:
        max_t = float(m.group(1))
    m = re.search(r"arrive\s+(?:in|within)\s+(\d+)", q)
    if m:
        max_t = float(m.group(1))
    return {
        "avoid_highways": any(
            x in q
            for x in (
                "avoid highway",
                "avoid highways",
                "avoiding highway",
                "avoiding highways",
                "no highway",
                "no freeway",
            )
        ),
        "prefer_fewer_turns": any(x in q for x in ("fewer turns", "less turns", "simpler route")),
        "minimize_delay_risk": any(
            x in q for x in ("delay risk", "minimize delay", "avoid delay", "reliable", "avoid traffic")
        ),
        "max_travel_time_min": max_t,
    }


def _fallback_parse(query: str) -> dict[str, Any]:
    out = empty_parse()
    out.update(_constraints_from_text(query))
    q = query.strip()
    from_to = re.search(
        r"(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\s+avoiding|\s+avoid|\s+prefer|\s+but|\s+within|$)",
        q,
        re.I | re.DOTALL,
    )
    if from_to:
        out["origin"] = from_to.group(1).strip()
        dest = from_to.group(2).strip()
        dest = re.split(r"\b(avoid|prefer|within|but|minimize)\b", dest, maxsplit=1, flags=re.I)[0].strip()
        out["destination"] = dest
    prefs = []
    if out["avoid_highways"]:
        prefs.append("avoid highways")
    if out["prefer_fewer_turns"]:
        prefs.append("fewer turns")
    if out["minimize_delay_risk"]:
        prefs.append("minimize delay risk")
    out["preferences"] = ", ".join(prefs)
    return out


def _parse_openai(query: str) -> dict[str, Any] | None:
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
        prompt = (
            "Extract a JSON object with keys: origin (string), destination (string), "
            "avoid_highways (bool), prefer_fewer_turns (bool), minimize_delay_risk (bool), "
            "max_travel_time_min (number or null). No other text.\n\n"
            f"User: {query}"
        )
        text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if "```" in text:
            text = re.sub(r"^.*?```(?:json)?\s*", "", text).split("```")[0]
        data = json.loads(text)
        out = empty_parse()
        out["origin"] = data.get("origin") or ""
        out["destination"] = data.get("destination") or ""
        out["avoid_highways"] = bool(data.get("avoid_highways"))
        out["prefer_fewer_turns"] = bool(data.get("prefer_fewer_turns"))
        out["minimize_delay_risk"] = bool(data.get("minimize_delay_risk"))
        out["max_travel_time_min"] = data.get("max_travel_time_min")
        return out
    except Exception:
        return None


def _parse_huggingface(query: str) -> dict[str, Any] | None:
    if not USE_HF_PARSER:
        return None
    try:
        from transformers import pipeline

        clf = pipeline("zero-shot-classification", model=HF_ZERO_SHOT_MODEL)
        result = clf(query, CONSTRAINT_LABELS, multi_label=True)
        labels = {lab: score for lab, score in zip(result["labels"], result["scores"])}
        out = _fallback_parse(query)
        out["avoid_highways"] = out["avoid_highways"] or labels.get("avoid highways", 0) > 0.45
        out["prefer_fewer_turns"] = out["prefer_fewer_turns"] or labels.get("prefer fewer turns", 0) > 0.45
        out["minimize_delay_risk"] = out["minimize_delay_risk"] or labels.get("minimize delay risk", 0) > 0.45
        return out
    except Exception:
        return None


def parse_route_query(query: str) -> dict[str, Any]:
    if not query or not query.strip():
        return empty_parse()
    if OPENAI_API_KEY:
        parsed = _parse_openai(query)
        if parsed:
            return parsed
    parsed = _parse_huggingface(query)
    if parsed:
        return parsed
    return _fallback_parse(query)


def constraints_from_parsed(parsed: dict) -> "RouteConstraints":
    from src.routing.optimizer import RouteConstraints

    return RouteConstraints(
        avoid_highways=bool(parsed.get("avoid_highways")),
        prefer_fewer_turns=bool(parsed.get("prefer_fewer_turns")),
        minimize_delay_risk=bool(parsed.get("minimize_delay_risk")),
        max_travel_time_min=parsed.get("max_travel_time_min"),
    )
