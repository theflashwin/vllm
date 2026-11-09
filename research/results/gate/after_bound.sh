#!/bin/zsh
cd /Users/home/Documents/GitHub/vllm-swap/research
while pgrep -f bound.py >/dev/null; do sleep 60; done
PYTHONPATH=. /Users/home/Documents/GitHub/vllm/.venv/bin/python -m kvplace.gate_sweep --out results/gate/gate.jsonl > results/gate/gate.log 2>&1
