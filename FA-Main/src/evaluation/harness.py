"""Baseline-vs-multi-agent evaluation harness (slides 30-32).

Research question: does a human-led specialist multi-agent workflow produce safer, more personalised and
more explainable investment memos than a single general-purpose LLM?
"""

import copy
import time
from typing import Any, Callable, Dict, List, Optional
from src.agents.client_profile import ClientProfileAgent
from src.api.service import AdvisorySessionService
from src.llm.base import LLMClientProtocol
from src.schemas.advisory import CandidateIdeaSet
from src.schemas.workspace import AdvisorActionRequest
from src.evaluation.baseline import single_llm_baseline
from src.evaluation import metrics as m

METRIC_ROWS = [
    ("suitability_violations", "Suitability violations (delivered)", "lower"),
    ("unsupported_claims", "Unsupported claims", "lower"),
    ("personalization", "Personalization (sensitivity to changed inputs)", "higher"),
    ("explainability", "Explainability (5 questions answered)", "higher"),
    ("consistency", "Output consistency (repeat runs)", "higher"),
    ("latency_s", "Latency (s)", "lower"),
    ("llm_calls", "Cost proxy: LLM calls", "lower"),
]

def perturb(raw: Dict[str, Any]) -> Dict[str, Any]:
    """A materially different client: shorter horizon and a more conservative stated appetite."""
    changed = copy.deepcopy(raw)
    changed["stated_risk_willingness"] = "Conservative"
    changed["time_horizon_months"] = min(int(changed.get("time_horizon_months") or 24), 24)
    return changed

def drive_workflow(service: AdvisorySessionService, raw: Dict[str, Any], sid: str) -> Optional[CandidateIdeaSet]:
    """Act as an advisor who locks every lockable idea and approves every other checkpoint.

    Returns the ideas that reached the memo, or None when the workflow blocked (nothing is delivered).
    """
    service.start_session(raw, session_id=sid)
    for _ in range(40):
        review = service.get_pending_review(sid)
        if review is None:
            break
        if "approve" in review["allowed_actions"]:
            service.execute_action(sid, AdvisorActionRequest(action="approve"))
        elif "lock_idea" in review["allowed_actions"]:
            locked = set(review.get("locked_idea_ids", []))
            nxt = next((i for i in review["candidate_ideas"] if i["idea_id"] not in locked), None)
            try:
                service.execute_action(sid, AdvisorActionRequest(
                    action="lock_idea", idea_id=nxt["idea_id"], notes="Evaluation harness: exception documented."))
            except ValueError:
                return None
        else:
            return None
    state = service.get_state(sid)
    return state["candidate_ideas"] if state.get("memo") else None

def _profile_and_brief(raw: Dict[str, Any], llm_client: LLMClientProtocol):
    from src.agents.market_intelligence import MarketIntelligenceAgent
    profile = ClientProfileAgent(llm_client).run(raw)
    return profile, MarketIntelligenceAgent(llm_client).run(profile)

def _calls(llm: Any) -> int:
    return len(getattr(llm, "call_history", []) or [])

def evaluate_system(name: str, produce: Callable[[Dict[str, Any], str], Optional[CandidateIdeaSet]],
                    llm_client: LLMClientProtocol, raw: Dict[str, Any], repeats: int = 2) -> Dict[str, Any]:
    profile, brief = _profile_and_brief(raw, llm_client)
    calls_before = _calls(llm_client)
    start = time.time()
    base = produce(raw, f"{name}-base")
    latency = round(time.time() - start, 3)
    calls = _calls(llm_client) - calls_before
    changed = produce(perturb(raw), f"{name}-changed")
    reruns = [base] + [produce(raw, f"{name}-rerun{i}") for i in range(repeats - 1)]
    delivered = base.ideas if base else []
    return {
        "system": name,
        "delivered": base is not None,
        "suitability_violations": m.suitability_violations(profile, base, brief) if base else 0,
        "unsupported_claims": m.unsupported_claims(delivered, brief),
        # None when the changed-input run was blocked (no recommendations to compare).
        "personalization": m.personalization_sensitivity(delivered, changed.ideas) if base and changed else None,
        "explainability": m.explainability(delivered),
        "consistency": m.output_consistency([r.ideas for r in reruns if r]),
        "latency_s": latency,
        "llm_calls": calls,
    }

def run_comparison(raw: Dict[str, Any], llm_factory: Callable[[], LLMClientProtocol]) -> Dict[str, Dict[str, Any]]:
    """Score the single-LLM baseline and the multi-agent workflow on one client."""
    base_llm = llm_factory()
    baseline = evaluate_system("single_llm", lambda r, sid: single_llm_baseline(r, base_llm), base_llm, raw)
    ma_llm = llm_factory()
    service = AdvisorySessionService(llm_client=ma_llm)
    multi = evaluate_system("multi_agent", lambda r, sid: drive_workflow(service, r, sid), ma_llm, raw)
    return {"single_llm": baseline, "multi_agent": multi}

def _fmt(value: Any) -> str:
    return "n/a (blocked)" if value is None else str(value)

def render_markdown(results: Dict[str, Dict[str, Dict[str, Any]]], offline: bool) -> str:
    lines = ["# Baseline vs proposed architecture", "",
             "Research question: does a human-led specialist multi-agent workflow produce safer, more personalized "
             "and more explainable investment memos than a single general-purpose LLM?", ""]
    if offline:
        lines += ["> Scores below come from the offline deterministic LLM, which returns scripted outputs. They verify the "
                  "harness, not model quality: set `OPENROUTER_API_KEY` and run again for a real comparison.", ""]
    for client, res in results.items():
        lines += [f"## {client}", "", "| Metric | Single LLM | Multi-agent | Better |", "|---|---|---|---|"]
        for key, label, better in METRIC_ROWS:
            lines.append(f"| {label} | {_fmt(res['single_llm'][key])} | {_fmt(res['multi_agent'][key])} | {better} is better |")
        if not res["multi_agent"]["delivered"]:
            lines.append("| Workflow outcome | n/a | blocked before memo (nothing unsafe delivered) | |")
        lines.append("")
    return "\n".join(lines)
