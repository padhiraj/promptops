# PromptOps: prompt library, evals and guardrails for a workplace AI assistant

A small, end-to-end project for making an internal AI assistant **reliable and testable**. It builds an SOP question-answering assistant (retrieval + LLM) and, more importantly, the quality layer around it:

- a **versioned prompt library** (three variants of the same prompt)
- a **golden test set** (30 questions: 24 answerable, 6 unanswerable) and a **red-team set** (18 adversarial prompts in 6 categories)
- **evals** for correctness, faithfulness (groundedness), hallucination rate, refusal accuracy and over-refusal
- a **regression gate** that fails CI when a prompt or model change makes quality worse
- **input/output guardrails** and a red-team report that writes guardrail recommendations
- **tracing** of every call to JSONL (optional Langfuse)
- **enablement material** for non-technical colleagues (how-to guide, quick reference, training outline)

The documents are synthetic SOPs for a fictional EV company (`data/docs/`), so the repo is safe to share.

## Architecture

```
question ──► input guardrail ──► BM25 retriever ──► prompt (qa_v1/v2/v3) ──► LLM ──► output guardrail ──► answer + sources
                                                                                                          │
                              golden set / red-team set ──► metrics ──► report ──► regression gate (CI) ◄─┘
                                                                         └──► JSONL trace (+ optional Langfuse)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 23 tests, no API key needed
python -m promptops compare --provider mock           # compare the 3 prompt variants
python -m promptops redteam --prompt qa_v3 --provider mock --guardrails
python -m promptops gate --prompt qa_v3 --provider mock

# Real model (free Groq key from console.groq.com)
export GROQ_API_KEY=...
python -m promptops compare --provider groq
python -m promptops redteam --prompt qa_v3 --provider groq
python -m promptops ask "How long is the battery warranty?" --provider groq
python -m promptops baseline --prompt qa_v3 --provider groq   # save scores; later runs fail on regressions
```

Commands: `ask`, `eval`, `compare`, `redteam`, `gate`, `baseline`, `export-promptfoo`. Add `--guardrails` to switch the input/output filters on.

## Metrics

| Metric | Meaning |
|---|---|
| correctness | Share of expected key facts present in the answer (answerable questions) |
| faithfulness | Share of answer sentences whose content is supported by the retrieved context |
| hallucination_rate | Unsupported answers, plus any answer given to an unanswerable question |
| refusal_accuracy | Unanswerable questions correctly refused |
| over_refusal_rate | Answerable questions wrongly refused |

These are deterministic lexical heuristics: fast, free and reproducible, which is what a CI gate needs. They are not a substitute for an LLM judge or human review when making final quality calls. Thresholds live in `eval_config.yaml`.

## The `mock` provider

`--provider mock` is a **deterministic simulator**, not a language model. It extracts sentences from the retrieved context and follows or ignores the rules in the prompt. It lets tests and CI run without an API key and proves the harness can tell good prompts from bad ones. **Its scores say nothing about real model quality.** Sample output, for illustration of the report format only:

| Prompt | correctness | faithfulness | hallucination_rate | refusal_accuracy | over_refusal_rate |
|---|---|---|---|---|---|
| qa_v1 | 0.875 | 0.733 | 0.267 | 0.0 | 0.0 |
| qa_v2 | 0.875 | 1.0 | 0.0 | 1.0 | 0.083 |
| qa_v3 | 0.875 | 1.0 | 0.0 | 1.0 | 0.083 |

Red-team failures out of 18 (mock): qa_v1 14, qa_v2 4, qa_v3 1; with `--guardrails`: 9, 0, 0.

## Results with a real model

Run `python -m promptops compare --provider groq` and paste the table here, together with the red-team failure rates from `python -m promptops redteam`. Report your own numbers.

| Prompt | correctness | faithfulness | hallucination_rate | refusal_accuracy | over_refusal_rate |
|---|---|---|---|---|---|
| qa_v1 | _fill in_ | | | | |
| qa_v2 | _fill in_ | | | | |
| qa_v3 | _fill in_ | | | | |

## CI regression gate

`.github/workflows/evals.yml` runs the tests and `promptops gate` on every push and pull request. With a `GROQ_API_KEY` repository secret it also gates on a live model. The gate fails when a metric crosses a threshold or worsens versus the saved baseline (`baselines/`) by more than `regression.max_drop`.

## Optional integrations

- **Promptfoo**: `python -m promptops export-promptfoo` writes `promptfoo/` (prompts, tests, config). Then run `npx promptfoo@latest eval -c promptfoo/promptfooconfig.yaml` with a `GROQ_API_KEY`.
- **Langfuse**: `pip install langfuse`, set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`, and every call is also sent as a trace. Local JSONL traces in `results/traces.jsonl` are always written.

## Project layout

```
promptops/        assistant, retriever, prompts, llm backends, metrics, evals, redteam, guardrails, tracing, cli
prompts/          qa_v1.yaml, qa_v2.yaml, qa_v3.yaml
data/             docs/ (6 synthetic SOPs), golden_set.jsonl, redteam_set.jsonl
docs/             how_to_guide.md, quick_reference.md, training_outline.md, guardrails.md
tests/            23 tests (data integrity, metrics, guardrails, gate, red-team)
baselines/        saved scores used by the regression gate
promptfoo/        generated Promptfoo files
```

## Limitations

- Heuristic metrics can miss paraphrased correct answers (false negatives) and fluent but wrong answers that reuse context words (false positives). Spot-check with a judge model or a person.
- The guardrails are pattern based and can be rephrased around.
- The corpus is small and synthetic. Add real SOPs and real failure reports to make the sets representative.
