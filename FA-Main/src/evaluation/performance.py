"""Performance and cost (deck slides 4 and 31): is the workflow operationally viable?

End-to-end latency, LLM calls per run, estimated cost per run, where the latency comes from (Research / Ideas / Checks /
Record), how much selective regeneration saves, whether the run needed manual recovery, and the unapproved action
rate. Read from the run log, so it works for any model."""

from typing import Any, Dict, List

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
    """Logical model calls in the run (a retry is not a new call: see `llm_attempts`)."""
    history = getattr(run.llm, "call_history", None)
    if history is not None:
        return len(history)
    return sum((e.get("usage") or {}).get("calls", 1) for e in run.log if e["event"] == "agent_output" and e.get("usage"))

def llm_attempts(run: RunResult) -> int:
    """Every HTTP attempt, including retries after a schema-validation failure: what the provider actually billed."""
    history = getattr(run.llm, "call_history", None)
    usage_attempts = sum((e.get("usage") or {}).get("attempts", 0) for e in run.log if e["event"] == "agent_output")
    return usage_attempts or (len(history) if history is not None else 0)

def performance_report(run: RunResult) -> Dict[str, Any]:
    outputs = [e for e in run.log if e["event"] == "agent_output"]
    by_bucket = {b: 0.0 for b in BUCKETS}
    prompt_t = completion_t = 0
    for e in outputs:
        by_bucket[STAGE_BUCKET.get(e["stage"], "Research")] += e.get("duration_ms") or 0.0
        usage = e.get("usage") or {}  # cumulative over the agent's calls and retries, so retries are not lost
        prompt_t += usage.get("prompt_tokens", 0)
        completion_t += usage.get("completion_tokens", 0)
    total = sum(by_bucket.values())
    cost = prompt_t / 1e6 * ASSUMED_PRICE_PER_MTOK["prompt"] + completion_t / 1e6 * ASSUMED_PRICE_PER_MTOK["completion"]
    regens = [e for e in run.log if e["event"] == "auto_regeneration"]
    regenerated = sum(len(e["slots"]) for e in regens)
    return {
        "end_to_end_ms": round(total, 1),
        "llm_calls": llm_calls(run),
        "llm_attempts": llm_attempts(run),
        "tokens": {"prompt": prompt_t, "completion": completion_t, "total": prompt_t + completion_t},
        "cost_usd": round(cost, 4) if (prompt_t or completion_t) else None,  # None: the model reported no token usage
        "latency_by_bucket_ms": {b: round(v, 1) for b, v in by_bucket.items()},
        "latency_share": {b: round(v / total, 3) if total else 0.0 for b, v in by_bucket.items()},
        "slowest_bucket": max(by_bucket, key=by_bucket.get) if total else None,
        "regeneration_rounds": len(regens),
        "idea_generations_saved": 3 * len(regens) - regenerated,  # slots kept instead of regenerated
        "manual_recovery_actions": recovery_actions(run),
        "consequential_actions": len(consequential_actions(run.log)),
        "unapproved_actions": sum(1 for a in consequential_actions(run.log) if not a["approved"]),
    }

# RM actions that repair a run rather than move it forward at an intended checkpoint (slide 31: "ex. intended RM checkpoints").
RECOVERY_ACTIONS = {"regenerate", "revise", "edit_allocation", "unlock_idea", "reject"}
# Outputs with consequences outside the workspace: each must follow an RM approval (slide 4: unapproved action rate).
CONSEQUENTIAL_EVENTS = {"pdf_generated", "profile_updates_approved"}

def recovery_actions(run: RunResult) -> List[str]:
    return [e["decision"].get("action") for e in run.log
            if e["event"] == "advisor_decision" and (e.get("decision") or {}).get("action") in RECOVERY_ACTIONS]

def consequential_actions(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each PDF or profile update, and whether the RM decision right before it approved it at the final checkpoint."""
    from src.evaluation.trajectory import FINAL_CHECKPOINTS
    out, last_approving = [], False
    for e in log:
        if e["event"] == "advisor_decision":
            last_approving = (e.get("decision") or {}).get("action") == "approve" and e.get("stage") in FINAL_CHECKPOINTS
        elif e["event"] in CONSEQUENTIAL_EVENTS:
            out.append({"event": e["event"], "approved": last_approving})
    return out

def operational_summary(runs: List[RunResult]) -> Dict[str, Any]:
    """Across runs: share completed without manual recovery, and the unapproved action rate (target 0%)."""
    clean = [r for r in runs if r.completed and not recovery_actions(r)]
    actions = [a for r in runs for a in consequential_actions(r.log)]
    unapproved = [a for a in actions if not a["approved"]]
    return {"runs": len(runs), "runs_without_manual_recovery": len(clean),
            "runs_without_manual_recovery_pct": round(100.0 * len(clean) / len(runs), 1) if runs else None,
            "consequential_actions": len(actions), "unapproved_actions": len(unapproved),
            "unapproved_action_rate_pct": round(100.0 * len(unapproved) / len(actions), 1) if actions else 0.0}
