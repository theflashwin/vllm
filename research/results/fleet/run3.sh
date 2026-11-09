#!/bin/zsh
ARMS=(base ttl1h@0.5 ttl1h ka1h@0.5 ka1h ka1h_notail ka1h_touch)
TL=(traces/fx/tl300_sp_s2.jsonl traces/fx/tl300_sp_s4.jsonl traces/fx/tl300_sp_s5.jsonl)
go() { PYTHONPATH=. $PY -m kvplace.retention_sweep "$@" --arms $ARMS --cost-json costs_scan_a100.json --workers 6; }
go $TL --rates 0.003 0.004 0.005 0.006 0.007 --out results/fleet/f3_tl_128_ssd.json
go $TL --rates 0.003 0.004 0.005 0.006 0.007 --sec-blocks 0 --out results/fleet/f3_tl_128_nossd.json
