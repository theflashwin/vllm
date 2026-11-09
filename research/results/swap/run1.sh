#!/bin/zsh
cd /Users/home/Documents/GitHub/vllm-swap/research
PY=/Users/home/Documents/GitHub/vllm/.venv/bin/python
R="0.004 0.0045 0.005 0.0055 0.006 0.0065 0.007 0.008 0.009 0.01"
T="traces/fx/tl300_sp_s1.jsonl traces/fx/tl300_sp_s3.jsonl traces/fx/tl300_sp_s5.jsonl"
PYTHONPATH=. $PY -m kvplace.swap_sweep ${=T} --rates ${=R} --cost-json costs_scan_a100.json --sec-blocks 0 --out results/swap/tl_nossd.jsonl
PYTHONPATH=. $PY -m kvplace.swap_sweep traces/fx/cat.jsonl --rates 0.005 0.0075 0.01 0.0125 0.015 0.0175 0.02 0.025 --cost-json costs_scan_a100.json --sec-blocks 0 --out results/swap/cat_nossd.jsonl
