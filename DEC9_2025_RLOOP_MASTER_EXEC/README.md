# DEC 9, 2025 - RLoop Master Execution Folder

This folder is for **running** the December 9th RLoop master directive:

- **00_DIRECTIVES/**: immutable specs only
  - `CURSOR_MASTER_DIRECTIVE_v2_COMPLETE.md` (execution plan)
  - `DEC9_CONFOUND_AUDIT_RESULTS.md` (what's missing / to be run)

- **01_CONFOUND_FALSIFICATION/**: code + results for Part A
  - `code/`: scripts to run the 60 confound prompts (repetition, pseudo-recursive, long control, banana test, etc.)
  - `results/`: CSVs + plots for confound tests
  - `logs/`: textual run logs

- **02_RLOOP_VALIDATION/**: code + results for Part B
  - `code/`: unified 4-phase R_V / V-patch / KV-patch / α-mix scripts
  - `results/`: master CSVs, summaries, and figures
  - `logs/`: run logs and sanity checks

Nothing is deleted from prior DEC folders. This is a clean execution workspace for DEC9 only.
