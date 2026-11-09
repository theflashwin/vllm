import json,sys
sys.path.insert(0,'.')
from concurrent.futures import ProcessPoolExecutor as P
from dataclasses import replace
from kvplace.sim import CostModel,Simulator,_pct,get_policy
from kvplace.trace import read_trace
GB=10**9
T=[f'traces/fx/tl300_sp_s{i}.jsonl' for i in (1,3,5)]
R=[0.008,0.012,0.016,0.02,0.024,0.028,0.032,0.04]
jobs=[dict(trace=t,pool=p,aff=a,gbps=0.96,arm=m,rate=r) for t in T for p in (29,59) for a in (0.25,0.5,0.75) for m in ('lru','ours') for r in R]
jobs+=[dict(trace=t,pool=59,aff=0.25,gbps=3.0,arm=m,rate=r) for t in T for m in ('lru','ours') for r in R]
def run(j):
    ss=read_trace(j['trace']); span=max(s.start_s for s in ss); f=(len(ss)/j['rate'])/span
    for s in ss: s.start_s*=f
    c=replace(CostModel.from_json('costs_scan_a100.json'),block_tokens=256,sec_cpu_gbps=j['gbps'])
    blocks=lambda gb:int(gb*GB/c.block_bytes)
    ours=j['arm']=='ours'
    sim=Simulator(ss,get_policy('reuse_evict' if ours else 'lru'),'tool-nfh2' if ours else 'none',c,66500//16,blocks(j['pool']),10_000_000,contend_sec=True,write_model='shared',gpu_admission=True,server_hints=True,replicas=4,affinity=j['aff'])
    recs=sim.run().records; res=[r for r in recs if r.turn>0]; t=[r.ttft_s for r in recs]
    return dict(j,p50=_pct(t,50),p90=_pct(t,90),p99=_pct(t,99),disk_share=sum(r.sec_hit_blocks>0 for r in res)/len(res),disk_gb=sum(r.sec_hit_blocks for r in recs)*c.block_bytes/GB)
if __name__=='__main__':
    print(len(jobs),flush=True)
    with P(8) as p,open('results/gate/shared.jsonl','w') as f:
        for o in p.map(run,jobs):
            f.write(json.dumps(o)+'\n'); f.flush()
