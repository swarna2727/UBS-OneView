"""Performance and cost (deck slide 27): is the workflow operationally viable?

End-to-end latency, LLM calls per run, estimated cost per run, where the latency comes from (Research / Ideas / Checks /
Record), and how much selective regeneration saves. Read from the run log, so it works for any model."""

from typing import Any, Dict

from src.evaluation.runner import RunResult

# Assumed list prices (US$ per million tokens), used only for the estimate. Replace with the contracted rates.
ASSUMED_PRICE_PER_MTOK = {"prompt": 3.0, "completion": 15.0}

BUCKETS = {
    "Research": ("client_profile", "client_limits", "market_intelligence"),
    "Ideas": ("investment_ideas",),
    "Checks": ("suitability_critic", "profile_change_check"),
    "Record": ("meeting_brief", "memo_formatter", "meeting_record"),
}
STAGE_BUCKET = {stage: bucket for bucket, stages in BUCKETS.items() for stage in stages}

def llm_calls(run: RunResult) -> int:
    history = getattr(run.llm, "call_history", None)
    if history is not None:
        return len(history)
    return sum(1 for e in run.log if e["event"] == "agent_output" and e.get("usage"))

def performance_report(run: RunResult) -> Dict[str, Any]:
    outputs = [e for e in run.log if e["event"] == "agent_output"]
    by_bucket = {b: 0.0 for b in BUCKETS}
    prompt_t = completion_t = 0
    for e in outputs:
        by_bucket[STAGE_BUCKET.get(e["stage"], "Research")] += e.get("duration_ms") or 0.0
        usage = e.get("usage") or {}
        prompt_t += usage.get("prompt_tokens", 0)
        completion_t += usage.get("completion_tokens", 0)
    total = sum(by_bucket.values())
    cost = prompt_t / 1e6 * ASSUMED_PRICE_PER_MTOK["prompt"] + completion_t / 1e6 * ASSUMED_PRICE_PER_MTOK["completion"]
    regens = [e for e in run.log if e["event"] == "auto_regeneration"]
    regenerated = sum(len(e["slots"]) for e in regens)
    return {
        "end_to_end_ms": round(total, 1),
        "llm_calls": llm_calls(run),
        "tokens": {"prompt": prompt_t, "completion": completion_t},
        "cost_usd": round(cost, 4) if (prompt_t or completion_t) else None,  # None: the model reported no token usage
        "latency_by_bucket_ms": {b: round(v, 1) for b, v in by_bucket.items()},
        "latency_share": {b: round(v / total, 3) if total else 0.0 for b, v in by_bucket.items()},
        "slowest_bucket": max(by_bucket, key=by_bucket.get) if total else None,
        "regeneration_rounds": len(regens),
        "idea_generations_saved": 3 * len(regens) - regenerated,  # slots kept instead of regenerated
    }
