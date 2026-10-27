"""CLI: python -m src.evaluation.run [--out reports/baseline_vs_multi_agent.md]"""

import argparse
import json
import os
import tempfile
os.environ.setdefault("RUN_LOG_DIR", tempfile.mkdtemp(prefix="ubs_eval_runlog_"))

from src.llm.fake import FakeAdvisoryLLMClient
from src.llm.openrouter import default_llm_client
from src.evaluation.harness import run_comparison, render_markdown

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "samples", "client_profiles.json")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/baseline_vs_multi_agent.md")
    parser.add_argument("--clients", nargs="*", default=["balanced_client", "demo_sarah"])
    args = parser.parse_args()
    probe = default_llm_client()
    offline = isinstance(probe, FakeAdvisoryLLMClient)
    factory = FakeAdvisoryLLMClient if offline else default_llm_client
    with open(SAMPLES, encoding="utf-8") as f:
        samples = json.load(f)
    results = {name: run_comparison(samples[name], factory) for name in args.clients}
    report = render_markdown(results, offline)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    main()
