"""An idea blocked by client limits is never shown (deck slides 8 and 12).

The Fit & Eligibility Critic fails such an idea (rule E1) and it is regenerated. If it still reaches the RM
(for example after R10 escalation), the product is withheld from everything the RM sees: its name, ticker and
text are replaced, and any mention of the ticker elsewhere in the payload is masked."""

import json
import re
from typing import Any, Dict, List, Set

from src.graph.state import AdvisoryState

WITHHELD = "[withheld: blocked by client limits]"

def withheld_tickers(state: AdvisoryState) -> Set[str]:
    return blocked_by_limits(state.get("assessment"), state.get("candidate_ideas"))

def blocked_by_limits(assessment: Any, ideas: Any) -> Set[str]:
    """Tickers of the ideas in this critic assessment that failed E1 (blocked by client limits)."""
    if not assessment or not ideas:
        return set()
    blocked_ids = {ie.idea_id for ie in assessment.idea_evaluations
                   if any(r.rule_id == "E1" and r.outcome == "FAIL" for r in ie.rule_results)}
    return {i.ticker for i in ideas.ideas if i.idea_id in blocked_ids}

def mask_tickers(payload: Any, tickers: Set[str]) -> Any:
    """Replace every whole-word mention of a withheld ticker anywhere in a JSON-serialisable payload."""
    if not tickers or payload is None:
        return payload
    text = json.dumps(payload, default=str)
    for t in tickers:
        text = re.sub(rf"\b{re.escape(t)}\b", "[withheld]", text)
    return json.loads(text)

def redact_ideas(ideas: List[Dict[str, Any]], tickers: Set[str]) -> List[Dict[str, Any]]:
    """Idea dicts with the blocked ones stripped of their product. idea_id and slot stay so the RM can regenerate the slot."""
    out = []
    for idea in ideas:
        if idea.get("ticker") in tickers:
            idea = {**idea, "ticker": "-", "instrument_id": "-", "name": WITHHELD, "thesis": WITHHELD, "why_now": WITHHELD,
                    "client_fit_rationale": WITHHELD, "alternative_considered": "", "key_risks": [WITHHELD],
                    "invalidation_trigger": WITHHELD, "withheld": True}
        out.append(idea)
    return mask_tickers(out, tickers)
