"""Export prompts and the golden set to Promptfoo format so prompt variants can be compared with Promptfoo too."""
from pathlib import Path

import yaml

from .config import GOLDEN_PATH, REFUSAL, ROOT
from .evals import load_jsonl
from .prompts import list_prompts, load_prompt
from .retriever import default_index

OUT = ROOT / "promptfoo"


def export(top_k: int = 3, model_id: str = "groq:llama-3.1-8b-instant") -> Path:
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    names = list_prompts()
    for n in names:
        p = load_prompt(n)
        body = p.system + "\n\n" + p.user.replace("{context}", "{{context}}").replace("{question}", "{{question}}")
        (OUT / "prompts" / f"{n}.txt").write_text(body + "\n", encoding="utf-8")

    index, tests = default_index(), []
    for case in load_jsonl(GOLDEN_PATH):
        ctx = "\n\n".join(f"[{c.id}] {c.text}" for c, _ in index.search(case["question"], k=top_k))
        t = {"description": case["id"], "vars": {"question": case["question"], "context": ctx}}
        if case["answerable"]:
            t["assert"] = [{"type": "icontains-any", "value": [a.strip() for a in f.split("|")]} for f in case["key_facts"]]
        else:
            t["assert"] = [{"type": "icontains", "value": REFUSAL}]
        tests.append(t)
    (OUT / "tests.yaml").write_text(yaml.safe_dump(tests, sort_keys=False, allow_unicode=True), encoding="utf-8")

    cfg = {
        "description": "PromptOps prompt variants on the SOP golden set",
        "prompts": [f"file://prompts/{n}.txt" for n in names],
        "providers": [{"id": model_id, "config": {"temperature": 0}}],
        "tests": "file://tests.yaml",
    }
    (OUT / "promptfooconfig.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return OUT
