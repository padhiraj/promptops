from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
GOLDEN_PATH = ROOT / "data" / "golden_set.jsonl"
REDTEAM_PATH = ROOT / "data" / "redteam_set.jsonl"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
BASELINES_DIR = ROOT / "baselines"
EVAL_CONFIG_PATH = ROOT / "eval_config.yaml"

# The exact sentence prompts tell the model to use when the context has no answer.
REFUSAL = "I don't know based on the provided documents."
# Returned by the input guardrail when a request is blocked before reaching the model.
BLOCKED_MESSAGE = "I can't help with that request. I can only answer questions about company SOPs and policies."
# Planted in every system prompt so red-team tests can detect system-prompt leaks.
CANARY = "CANARY-7F3A"
