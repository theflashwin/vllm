#!/bin/zsh
ARMS=(base ttl30m ttl1h@0.25 ttl1h@0.5 ttl1h ka30m ka1h@0.25 ka1h@0.5 ka1h ka1h_notail ka1h_touch)
TL=(traces/fx/tl300_sp_s1.jsonl traces/fx/tl300_sp_s3.jsonl)
go() { PYTHONPATH=. $PY -m kvplace.retention_sweep "$@" --arms $ARMS --cost-json costs_scan_a100.json --workers 6; }
for cfg in "128_ssd 140000 10000000" "128_nossd 140000 0" "64_nossd 70000 0"; do
  set -- ${=cfg}
  go $TL --rates 0.003 0.004 0.005 0.006 0.007 0.008 --cpu-blocks $2 --sec-blocks $3 --out results/fleet/f2_tl_$1.json
  go traces/fx/cat.jsonl --rates 0.005 0.0075 0.01 0.0125 0.015 --cpu-blocks $2 --sec-blocks $3 --out results/fleet/f2_cat_$1.json
done
