#!/bin/zsh
cd /Users/home/Documents/GitHub/vllm-swap/research
while pgrep -f "gate/bimodal.py" >/dev/null; do sleep 20; done
PYTHONPATH=. /Users/home/Documents/GitHub/vllm/.venv/bin/python results/gate/shared.py > results/gate/shared.log 2>&1
