#!/bin/zsh
ARMS=(base lru ttl30m ttl1h@0.25 ttl1h@0.5 ttl1h ka30m ka1h@0.25 ka1h@0.5 ka1h ka1h_notail ka1h_touch)
T=(traces/fx/tl300_sp_s1.jsonl traces/fx/tl300_sp_s3.jsonl traces/fx/cat.jsonl)
R=(0.005 0.01 0.015 0.02)
C=(--cost-json costs_scan_a100.json --workers 6)
PYTHONPATH=. $1 -m kvplace.retention_sweep $T --arms $ARMS --rates $R --out results/fleet/cpu128_ssd.json $C
PYTHONPATH=. $1 -m kvplace.retention_sweep $T --arms $ARMS --rates $R --sec-blocks 0 --out results/fleet/cpu128_nossd.json $C
PYTHONPATH=. $1 -m kvplace.retention_sweep $T --arms $ARMS --rates $R --cpu-blocks 35000 --sec-blocks 0 --out results/fleet/cpu32_nossd.json $C
