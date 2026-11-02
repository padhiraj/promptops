# 30-minute enablement session: "Using the SOP Assistant safely and well"

**Audience:** non-technical colleagues in operations, service, supply chain and quality.
**Before the session:** pick three real questions from the audience and test them so you know the answers.

| Time | Segment | Content |
|---|---|---|
| 0-5 min | Why it matters | What the assistant is, and why it only answers from approved documents |
| 5-12 min | Live demo | One good question, one vague question, one "I don't know" answer. Show the `[source]` tag |
| 12-18 min | Do's and don'ts | Walk through the table in `how_to_guide.md`. Stress: no personal data, human review for customer text |
| 18-25 min | Hands-on | Pairs ask three questions each, verify the source and note anything wrong |
| 25-30 min | Reporting and Q&A | Show form AI-FB-01. Explain that reports become test cases |

**Success check:** each attendee can (1) ask a specific question, (2) find the source, (3) explain when to escalate.
**Follow-up:** collect reports from AI-FB-01 for two weeks, add them to `data/golden_set.jsonl` and `data/redteam_set.jsonl`, and re-run `python -m promptops gate`.
