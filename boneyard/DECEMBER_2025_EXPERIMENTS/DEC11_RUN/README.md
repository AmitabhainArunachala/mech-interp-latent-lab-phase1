# DEC 11, 2025 - Run Log

## Structure
- `00_SETUP/` — env checks, quick sanity scripts, tokenizer/model notes
- `01_EXPERIMENTS/` — code for today’s runs (scripts/notebooks)
- `results/` — CSVs, logs, plots saved per experiment
- `WRITEUPS/` — narrative session logs, audits, summaries
- `NOTES/` — scratch planning, TODOs, observations

## Session Template
- **Date/Time:**  
- **Hardware:** (GPU/VRAM, drivers)  
- **Models:** (exact HF id + commit)  
- **Seeds / dtype:**  
- **Primary goals:**  
- **Key configs:** window size, layers, batch, α-mix, temp  

## Logging Checklist
- [ ] Record seed, model revision, tokenizer
- [ ] Save raw CSV to `results/`
- [ ] Note window size & target layers
- [ ] Note prompt subset used
- [ ] Document controls (random/shuffle/wrong-layer/cross-baseline)
- [ ] Push writeup to `WRITEUPS/`


