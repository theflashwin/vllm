import json,sys
sys.path.insert(0,'.')
from concurrent.futures import ProcessPoolExecutor as P
from dataclasses import replace
from kvplace.gate_sweep import llama70,GPUS_70B
from kvplace.sim import CostModel,Simulator,_pct,get_policy
from kvplace.trace import read_trace
GB=10**9
INF=1e6
FX='traces/fx/'
TL=lambda i:f'{FX}tl300_sp_s{i}.jsonl'
CAT=FX+'cat.jsonl'
SPEEDS=[0.96,3.0,7.0,INF]
R7TL=[0.003,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.0075,0.009]
R7CAT=[0.006,0.009,0.011,0.0125,0.014,0.017,0.021,0.026,0.032]
R70TL=[0.002,0.003,0.0045,0.006,0.008,0.012,0.016,0.022]
R70CAT=[0.005,0.0075,0.011,0.015,0.02,0.03,0.04,0.055]
RSH=[round(0.008+0.004*i,3) for i in range(13)]
SETUPS=[
 ('7B_7GB','7B',7,None,[(TL(1),R7TL),(TL(3),R7TL),(CAT,R7CAT)]),
 ('70B_256GB','70B',256,None,[(TL(1),R70TL),(TL(3),R70TL),(CAT,R70CAT)]),
 ('shared_59GB_a25','7B',59,0.25,[(TL(i),RSH) for i in (1,3,5)]),
 ('shared_29GB_a25','7B',29,0.25,[(TL(i),RSH) for i in (1,3,5)]),
]
jobs=[dict(setup=n,model=m,cpu=c,aff=a,trace=t,gbps=g,arm=x,rate=r)
      for n,m,c,a,ts in SETUPS for t,rs in ts for g in SPEEDS for x in ('lru','ours') for r in rs]
def mean(x):
    x=list(x); return sum(x)/len(x) if x else 0.0
def parts(rs,c):
    def p(r):
        link=c.cpu_to_gpu_s(r.cpu_hit_blocks+r.sec_hit_blocks)
        return dict(queue=r.queue_s,disk=r.load_s-link,link=link,prefill=r.prefill_s,ttft=r.ttft_s)
    d=[p(r) for r in rs]
    return {k:mean(x[k] for x in d) for k in ('queue','disk','link','prefill','ttft')}
def run(j):
    ss=read_trace(j['trace']); span=max(s.start_s for s in ss); f=(len(ss)/j['rate'])/span
    for s in ss: s.start_s*=f
    c=CostModel.from_json('costs_scan_a100.json')
    shared=j['aff'] is not None
    gb=61
    if j['model']=='70B':
        c=llama70(c); gb=61*GPUS_70B
    c=replace(c,block_tokens=256,sec_cpu_gbps=j['gbps'])
    blocks=lambda x:int(x*GB/c.block_bytes)
    ours=j['arm']=='ours'
    gpu=66500//16 if shared else blocks(gb)
    kw=dict(replicas=4,affinity=j['aff']) if shared else {}
    sim=Simulator(ss,get_policy('reuse_evict' if ours else 'lru'),'tool-nfh2' if ours else 'none',c,gpu,blocks(j['cpu']),10_000_000,contend_sec=True,write_model='shared',gpu_admission=True,server_hints=True,**kw)
    recs=sim.run().records
    res=[r for r in recs if r.turn>0]
    t=sorted(r.ttft_s for r in recs)
    lo,hi=_pct(t,85),_pct(t,95)
    band=[r for r in recs if lo<=r.ttft_s<=hi]
    return dict(j,n=len(recs),p50=_pct(t,50),p90=_pct(t,90),p99=_pct(t,99),
      disk_share=sum(r.sec_hit_blocks>0 for r in res)/len(res),
      disk_gb=sum(r.sec_hit_blocks for r in recs)*c.block_bytes/GB,
      all=parts(recs,c),near_p90=parts(band,c))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='one':
        import time
        j=[x for x in jobs if x['setup']=='shared_59GB_a25' and x['gbps']==0.96 and x['arm']=='lru' and x['rate']==0.012 and x['trace']==TL(1)][0]
        a=time.time(); print(json.dumps(run(j),indent=0)[:900]); print(time.time()-a); sys.exit()
    print(len(jobs),flush=True)
    with P(7) as p,open('results/gate/disk_speed.jsonl','w') as f:
        for o in p.map(run,jobs):
            f.write(json.dumps(o)+'\n'); f.flush()
