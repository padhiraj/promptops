"""Golden-set evaluation, prompt comparison and the regression gate."""
import json
from pathlib import Path
from typing import Optional

import yaml

from .assistant import Assistant
from .config import BASELINES_DIR, EVAL_CONFIG_PATH, GOLDEN_PATH, RESULTS_DIR
from .metrics import correctness, faithfulness, is_refusal, mean, percentile

HIGHER_IS_BETTER = ["correctness", "faithfulness", "refusal_accuracy"]
LOWER_IS_BETTER = ["hallucination_rate", "over_refusal_rate"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path: Path = EVAL_CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run_golden(assistant: Assistant, cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        a = assistant.ask(case["question"])
        refused = is_refusal(a.answer)
        faith = faithfulness(a.answer, a.context)
        if case["answerable"]:
            corr: Optional[float] = correctness(a.answer, case["key_facts"])
            halluc = faith < 0.7 and not refused
        else:
            corr = None
            halluc = not refused
        results.append({
            "id": case["id"], "question": case["question"], "answerable": case["answerable"],
            "answer": a.answer, "refused": refused, "correctness": corr, "faithfulness": faith,
            "hallucinated": halluc, "latency_s": round(a.latency_s, 4), "sources": a.sources, "events": a.events,
        })
    return results


def summarize(results: list[dict]) -> dict:
    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    lat = [r["latency_s"] for r in results]
    return {
        "n_cases": len(results), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "correctness": round(mean(r["correctness"] for r in ans), 4),
        "faithfulness": round(mean(r["faithfulness"] for r in results), 4),
        "hallucination_rate": round(mean(r["hallucinated"] for r in results), 4),
        "refusal_accuracy": round(mean(r["refused"] for r in unans), 4),
        "over_refusal_rate": round(mean(r["refused"] for r in ans), 4),
        "latency_p50_s": round(percentile(lat, 50), 4), "latency_p95_s": round(percentile(lat, 95), 4),
    }


def evaluate(assistant: Assistant, golden_path: Path = GOLDEN_PATH) -> dict:
    results = run_golden(assistant, load_jsonl(golden_path))
    return {"prompt": assistant.prompt.name, "provider": assistant.llm.name, "guardrails": assistant.guardrails,
            "summary": summarize(results), "cases": results}


def baseline_path(provider: str, prompt: str) -> Path:
    return BASELINES_DIR / f"{provider}_{prompt}.json"


def save_baseline(report: dict) -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    p = baseline_path(report["provider"], report["prompt"])
    p.write_text(json.dumps(report["summary"], indent=2), encoding="utf-8")
    return p


def gate(summary: dict, cfg: dict, baseline: Optional[dict] = None) -> list[str]:
    """Return a list of failure messages. An empty list means the gate passes."""
    t, fails = cfg["thresholds"], []
    checks = [
        ("correctness", summary["correctness"] >= t["min_correctness"], f">= {t['min_correctness']}"),
        ("faithfulness", summary["faithfulness"] >= t["min_faithfulness"], f">= {t['min_faithfulness']}"),
        ("hallucination_rate", summary["hallucination_rate"] <= t["max_hallucination_rate"], f"<= {t['max_hallucination_rate']}"),
        ("refusal_accuracy", summary["refusal_accuracy"] >= t["min_refusal_accuracy"], f">= {t['min_refusal_accuracy']}"),
        ("over_refusal_rate", summary["over_refusal_rate"] <= t["max_over_refusal_rate"], f"<= {t['max_over_refusal_rate']}"),
    ]
    for name, ok, rule in checks:
        if not ok:
            fails.append(f"threshold: {name} = {summary[name]} (needs {rule})")
    if baseline:
        drop = cfg.get("regression", {}).get("max_drop", 0.03)
        for m in HIGHER_IS_BETTER:
            if summary[m] < baseline[m] - drop:
                fails.append(f"regression: {m} fell from {baseline[m]} to {summary[m]} (allowed drop {drop})")
        for m in LOWER_IS_BETTER:
            if summary[m] > baseline[m] + drop:
                fails.append(f"regression: {m} rose from {baseline[m]} to {summary[m]} (allowed rise {drop})")
    return fails


def comparison_markdown(reports: list[dict]) -> str:
    cols = ["correctness", "faithfulness", "hallucination_rate", "refusal_accuracy", "over_refusal_rate", "latency_p95_s"]
    head = "| Prompt | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    rows = "".join(f"| {r['prompt']} | " + " | ".join(str(r["summary"][c]) for c in cols) + " |\n" for r in reports)
    note = f"\nProvider: `{reports[0]['provider']}`, guardrails: `{'on' if reports[0]['guardrails'] else 'off'}`, " \
           f"cases: {reports[0]['summary']['n_cases']}.\n"
    return head + rows + note


def write_report(report: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / name
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
