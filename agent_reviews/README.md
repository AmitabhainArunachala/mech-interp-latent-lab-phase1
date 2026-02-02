# Agent Reviews (drop-in folder)

This folder is designed so you can ask multiple agents the same question and collect their writeups **without copy/paste back into chat**.

## How to use

1. Give each agent the prompt in `agent_reviews/REQUEST_TOP_FINDINGS_LEDGER.md`.
2. Instruct them to **write their response as a markdown file** into:
   - `agent_reviews/responses/`
3. Require the filename + header schema below.

## Required filename format

`YYYYMMDD__MODELNAME__TOP_FINDINGS_LEDGER.md`

Examples:
- `20251215__claude-opus-4-5__TOP_FINDINGS_LEDGER.md`
- `20251215__gpt-5-2__TOP_FINDINGS_LEDGER.md`

## Required header schema (must be present at top of each response)

```markdown
Title: TOP FINDINGS LEDGER (DNA → CELL → ANIMAL)
Date: YYYY-MM-DD
Model: <model name + version>
Repo commit: <optional if available>
Prompt bank version: <PromptLoader.version if you ran code; otherwise "not checked">
```

## What success looks like

- Every claim includes **exact file paths**.
- Claims are labeled **VERIFIED / UNCERTAIN / CONTRADICTED**.
- Key metrics include **N** and stats (means / effect sizes / p-values or CIs) when available.
- The agent identifies **measurement inconsistencies** (especially R_V) and points to where they occur.


