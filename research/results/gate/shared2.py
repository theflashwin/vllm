import json,sys
sys.path.insert(0,'results/gate')
from concurrent.futures import ProcessPoolExecutor as P
import shared as s
R=[round(0.010+0.001*i,4) for i in range(13)]
jobs=[dict(trace=f'traces/fx/tl300_sp_s{i}.jsonl',pool=59,aff=0.25,gbps=0.96,arm=m,rate=r) for i in (1,2,3,4,5) for m in ('lru','ours') for r in R]
if __name__=='__main__':
    print(len(jobs),flush=True)
    with P(8) as p,open('results/gate/shared2.jsonl','w') as f:
        for o in p.map(s.run,jobs):
            f.write(json.dumps(o)+'\n'); f.flush()
