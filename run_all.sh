#!/bin/sh
# Run the whole pipeline in order. Step 3 (the LLM extraction) needs
# ANTHROPIC_API_KEY set and is skipped if it is not.
set -e
python scripts/10_generate_notes.py
python scripts/20_run_baseline.py
if [ -n "$ANTHROPIC_API_KEY" ]; then
  python scripts/30_run_llm.py
else
  echo "ANTHROPIC_API_KEY not set: skipping the LLM extraction step."
fi
python scripts/40_evaluate.py
python scripts/50_downstream.py
python scripts/45_figures.py
