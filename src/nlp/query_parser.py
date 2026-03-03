"""LangChain NLP/LLM layer to parse natural-language route queries."""
import json
import os
import re
from typing import Any

from src.config import OPENAI_API_KEY

# Structured output schema for route intent
ROUTE_QUERY_SCHEMA = """
Extract from the user message:
1. origin: string (place or description, e.g. "downtown", "airport", "123 Main St")
2. destination: string (place or description)
3. preferences: string (optional: "avoid traffic", "shortest", "fastest", "scenic", or empty)
Return ONLY a JSON object with keys: origin, destination, preferences. No other text.
"""


def _fallback_parse(query: str) -> dict[str, Any]:
    """Rule-based fallback when LLM is not available."""
    query = query.strip().lower()
    # Patterns: "from X to Y", "X to Y", "route from X to Y"
    from_to = re.search(r"(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\s+avoiding|\s+avoid|\s+prefer|\s+want)?$", query, re.I | re.DOTALL)
    if from_to:
        origin = from_to.group(1).strip()
        dest = from_to.group(2).strip()
        preferences = ""
        if "avoid" in query or "traffic" in query:
            preferences = "avoid traffic"
        elif "shortest" in query:
            preferences = "shortest"
        elif "fast" in query:
            preferences = "fastest"
        return {"origin": origin, "destination": dest, "preferences": preferences}
    return {"origin": "", "destination": "", "preferences": ""}


def parse_route_query(query: str) -> dict[str, Any]:
    """
    Parse natural-language route query into structured origin, destination, preferences.
    Uses LangChain/LLM when OPENAI_API_KEY is set; otherwise rule-based fallback.
    """
    if not query or not query.strip():
        return {"origin": "", "destination": "", "preferences": ""}

    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
            prompt = (
                f"{ROUTE_QUERY_SCHEMA}\n\nUser message: {query}"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip()
            # Extract JSON (handle markdown code blocks)
            if "```" in text:
                text = re.sub(r"^.*?```(?:json)?\s*", "", text).split("```")[0]
            data = json.loads(text)
            return {
                "origin": data.get("origin", ""),
                "destination": data.get("destination", ""),
                "preferences": data.get("preferences", ""),
            }
        except Exception:
            pass
    return _fallback_parse(query)
