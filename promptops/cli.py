import argparse
import json
import sys

from . import evals, redteam
from .assistant import Assistant
from .config import EVAL_CONFIG_PATH, GOLDEN_PATH, REDTEAM_PATH
from .llm import get_llm
from .prompts import list_prompts, load_prompt
from .retriever import default_index
from .tracing import Tracer


def build(prompt: str, provider: str, guardrails: bool, trace: bool = True) -> Assistant:
    cfg = evals.load_config()
    return Assistant(load_prompt(prompt), get_llm(provider), default_index(), guardrails=guardrails,
                     tracer=Tracer() if trace else None, top_k=cfg.get("retrieval", {}).get("top_k", 3))


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="mock", choices=["mock", "groq", "openai"])
    p.add_argument("--guardrails", action="store_true", help="enable input/output guardrails")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="promptops", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="ask one question"); _common(p)
    p.add_argument("question"); p.add_argument("--prompt", default="qa_v3")

    p = sub.add_parser("eval", help="run the golden set for one prompt"); _common(p)
    p.add_argument("--prompt", default="qa_v3")

    p = sub.add_parser("compare", help="compare prompt variants on the golden set"); _common(p)
    p.add_argument("--prompts", nargs="*", default=None)

    p = sub.add_parser("redteam", help="run the adversarial suite"); _common(p)
    p.add_argument("--prompt", default="qa_v3")

    p = sub.add_parser("gate", help="fail (exit 1) if quality thresholds or the baseline regress"); _common(p)
    p.add_argument("--prompt", default="qa_v3")

    p = sub.add_parser("baseline", help="save the current scores as the regression baseline"); _common(p)
    p.add_argument("--prompt", default="qa_v3")

    sub.add_parser("export-promptfoo", help="write Promptfoo config, prompts and tests")
    args = ap.parse_args(argv)

    if args.cmd == "export-promptfoo":
        from .promptfoo_export import export
        print(f"Wrote Promptfoo files to {export()}")
        return 0

    if args.cmd == "ask":
        a = build(args.prompt, args.provider, args.guardrails).ask(args.question)
        print(a.answer)
        print(f"\nsources: {', '.join(a.sources) or '-'}   latency: {a.latency_s:.3f}s   events: {a.events or '-'}")
        return 0

    if args.cmd == "compare":
        names = args.prompts or list_prompts()
        reports = [evals.evaluate(build(n, args.provider, args.guardrails)) for n in names]
        md = evals.comparison_markdown(reports)
        evals.write_report({"reports": reports}, f"comparison_{args.provider}.json")
        print(md)
        return 0

    assistant = build(args.prompt, args.provider, args.guardrails)

    if args.cmd == "eval":
        report = evals.evaluate(assistant)
        path = evals.write_report(report, f"eval_{args.provider}_{args.prompt}.json")
        print(json.dumps(report["summary"], indent=2)); print(f"full report: {path}")
        return 0

    if args.cmd == "redteam":
        report = redteam.evaluate(assistant)
        evals.write_report(report, f"redteam_{args.provider}_{args.prompt}.json")
        path = redteam.write_recommendations(report)
        print(json.dumps(report["summary"], indent=2)); print(f"recommendations: {path}")
        return 0

    if args.cmd == "baseline":
        report = evals.evaluate(assistant)
        print(f"Saved baseline to {evals.save_baseline(report)}")
        return 0

    if args.cmd == "gate":
        report = evals.evaluate(assistant)
        evals.write_report(report, f"eval_{args.provider}_{args.prompt}.json")
        bp = evals.baseline_path(args.provider, args.prompt)
        baseline = json.loads(bp.read_text()) if bp.exists() else None
        fails = evals.gate(report["summary"], evals.load_config(), baseline)
        print(json.dumps(report["summary"], indent=2))
        if fails:
            print("\nGATE FAILED")
            for f in fails:
                print(f"  - {f}")
            return 1
        print(f"\nGATE PASSED ({'with' if baseline else 'no'} baseline comparison)")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
