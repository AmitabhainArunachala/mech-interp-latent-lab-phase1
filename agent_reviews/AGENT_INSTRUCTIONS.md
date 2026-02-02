# Instructions for Agents

## Quick Start

1. Read: `agent_reviews/REQUEST_TOP_FINDINGS_LEDGER.md`
2. Write response to: `agent_reviews/responses/YYYYMMDD__MODELNAME__TOP_FINDINGS_LEDGER.md`
3. Include the required header (see below)
4. Follow all constraints and deliverables from the request file

## Task

1. **Read the full request**: `agent_reviews/REQUEST_TOP_FINDINGS_LEDGER.md`
   - This contains the complete task description, hard constraints, and deliverables

2. **Write your response** as a single markdown file and save it to:
   ```
   agent_reviews/responses/YYYYMMDD__MODELNAME__TOP_FINDINGS_LEDGER.md
   ```

3. **Required filename format**:
   - Format: `YYYYMMDD__MODELNAME__TOP_FINDINGS_LEDGER.md`
   - Date format: YYYYMMDD (e.g., 20251215)
   - Model name: use a clear identifier (e.g., `claude-opus-4-5`, `gpt-5-2`)
   - Examples:
     - `20251215__claude-opus-4-5__TOP_FINDINGS_LEDGER.md`
     - `20251215__gpt-5-2__TOP_FINDINGS_LEDGER.md`

4. **Required header** (must be at the very top of your response file, before any other content):
   ```markdown
   Title: TOP FINDINGS LEDGER (DNA → CELL → ANIMAL)
   Date: YYYY-MM-DD
   Model: <model name + version>
   Repo commit: <optional if available>
   Prompt bank version: <PromptLoader.version if you ran code; otherwise "not checked">
   ```

## Success Criteria

- Every claim includes **exact file paths** to supporting evidence
- Claims are labeled **VERIFIED / UNCERTAIN / CONTRADICTED**
- Key metrics include **N** (sample size) and stats (means / effect sizes / p-values or CIs) when available
- Measurement inconsistencies (especially R_V) are identified with file locations
- Follow all hard constraints and deliverables listed in `REQUEST_TOP_FINDINGS_LEDGER.md`

## Output Location

All responses go into: `agent_reviews/responses/`

This allows direct comparison of outputs across different agents by simply scanning the `responses/` directory.

