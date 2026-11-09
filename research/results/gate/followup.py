import json,sys
sys.path.insert(0,'.')
from concurrent.futures import ProcessPoolExecutor as P
import kvplace.gate_sweep as g
g.TRACES.update({f"tl_s{i}":f"traces/fx/tl300_sp_s{i}.jsonl" for i in (2,4,5)})
TL=[round(0.0035+0.0005*i,5) for i in range(11)]
CAT=[round(0.008+0.001*i,5) for i in range(9)]
jobs=[dict(model='70B',cpu_gb=256,disk_gbps=d,trace=t,arm=a,rate=r) for d in (0.96,3.0) for t in g.TRACES for a in ('lru','ours') for r in (CAT if t=='cat' else TL)]
if __name__=='__main__':
    print(len(jobs),flush=True)
    with P(8) as p,open('results/gate/followup.jsonl','w') as f:
        for o in p.map(g.run_one,jobs):
            f.write(json.dumps(o)+'\n'); f.flush()
