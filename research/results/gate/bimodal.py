import json,sys
sys.path.insert(0,'.')
from concurrent.futures import ProcessPoolExecutor as P
import kvplace.gate_sweep as g
from kvplace.sim import Simulator
PRED='toolres-nfh2'
g.Simulator=lambda s,pol,pred,*a,**k: Simulator(s,pol,PRED if pred!='none' else pred,*a,**k)
CFG=[("7B",7,0.96),("7B",32,0.96),("7B",7,3.0),("70B",256,0.96)]
jobs=[dict(model=m,cpu_gb=c,disk_gbps=d,trace=t,arm='ours',rate=r,pred=PRED) for m,c,d in CFG for t in g.TRACES for r in g.RATES[m,t.split('_')[0]]]
if __name__=='__main__':
    print(len(jobs),flush=True)
    with P(8) as p,open('results/gate/bimodal.jsonl','w') as f:
        for o in p.map(g.run_one,jobs):
            f.write(json.dumps(o)+'\n'); f.flush()
