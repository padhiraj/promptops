# Guardrail design

PromptOps uses layers, because no single layer is reliable on its own.

| Layer | Where | What it does |
|---|---|---|
| 1. Prompt rules | `prompts/qa_v2.yaml`, `qa_v3.yaml` | Grounding, citations, exact refusal sentence, untrusted-input rule, safety and privacy rules |
| 2. Retrieval | `promptops/retriever.py` | The model only sees the top-k relevant chunks, so unrelated questions arrive with weak or no context |
| 3. Input filter | `promptops/guardrails.py` | Blocks known injection phrases and requests for personal data before they reach the model |
| 4. Output filter | `promptops/guardrails.py` | Redacts answers that leak the internal canary token |
| 5. Human review | Process | Customer-facing output is reviewed by a person; wrong answers are reported through AI-FB-01 |

## How it is tested
- `data/redteam_set.jsonl` holds adversarial cases in six categories: prompt injection, off-topic, hallucination bait, unsafe requests, personal data and off-brand requests.
- `python -m promptops redteam --prompt qa_v3 [--guardrails]` reports failure rate per category and writes `results/guardrail_recommendations.md`.
- Run it with and without `--guardrails` to see what the input and output filters add.

## Known limits
- The filters are pattern based. They are easy to read and test, but a determined attacker can rephrase around them. Treat them as one layer, not a guarantee.
- A passing red-team run does not prove safety. Add every real failure to the set.
