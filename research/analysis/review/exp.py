"""Read-only review experiments: every change is a monkeypatch or parameter here."""
import json,sys,random,math
sys.path.insert(0,'.')
from concurrent.futures import ProcessPoolExecutor as P
from dataclasses import replace
from kvplace.gate_sweep import llama70,GPUS_70B
from kvplace.sim import CostModel,Simulator,_pct,get_policy,_block_keys
from kvplace.trace import read_trace
GB=10**9
TR=lambda i:f'traces/fx/tl300_sp_s{i}.jsonl'
G70=[round(0.0035+0.0005*i,5) for i in range(15)]+[0.012,0.014,0.017,0.02,0.025,0.03]
GSH=[round(0.010+0.001*i,4) for i in range(13)]+[0.024,0.028,0.032,0.04,0.05]
GEO=[round(0.0025*1.13**i,5) for i in range(30)]
class TouchAll(Simulator):
    """LRU arm that refreshes the whole prefix in CPU, as vLLM's prepare_store /
    on_request_finished does for GPU-hit keys (scheduler.py next_stored_chunk_idx=0)."""
    def _arrive(self,now,s,k,arrived=None):
        super()._arrive(now,s,k,arrived)
        keys=_block_keys(s,s.prompt_lengths()[k],self.cost.block_tokens)
        self.cpu.touch(keys,now)
def run(j):
    ss=read_trace(j['trace'])
    if j.get('draw') is not None:
        r=random.Random(1000+j['draw']); t=0.0
        for s in ss:
            t+=r.expovariate(1.0); s.start_s=t
    span=max(s.start_s for s in ss); f=(len(ss)/j['rate'])/span
    for s in ss: s.start_s*=f
    c=CostModel.from_json('costs_scan_a100.json')
    setup=j['setup']; arm=j['arm']; mod=j.get('mod','base')
    gpu_gb=61; kw={}
    if setup.startswith('70B'): c=llama70(c); gpu_gb=61*GPUS_70B
    cpu_gb={'70B_256':256,'7B_7':7,'sh59':59}[setup]
    gbps=0.96
    if 'disk' in mod: gbps=1e6
    c=replace(c,block_tokens=256,sec_cpu_gbps=gbps)
    if 'prefill10' in mod:
        c=replace(c,prefill_tok_per_s=c.prefill_tok_per_s*10,prefill_s_per_tok_per_ctx=c.prefill_s_per_tok_per_ctx/10)
    if 'scan0' in mod: c=replace(c,decode_s_per_ctx_tok=1e-12)
    blocks=lambda g:int(g*GB/c.block_bytes)
    gpu=66500//16 if setup=='sh59' else blocks(gpu_gb)
    if 'mem4' in mod: gpu*=4
    if setup=='sh59': kw=dict(replicas=4,affinity=0.25)
    cpu=blocks(cpu_gb); sec=10_000_000
    cls=Simulator; pol='lru'; pred='none'
    if arm=='ours': pol,pred='reuse_evict','tool-nfh2'
    elif arm=='oracle': pol,pred='reuse_evict','oracle'
    elif arm=='lru_touch': cls=TouchAll
    elif arm=='nodisk': sec=0
    elif arm=='infcpu': cpu=10**9
    sim=cls(ss,get_policy(pol),pred,c,gpu,cpu,sec,contend_sec=True,write_model='shared',gpu_admission=True,server_hints=True,**kw)
    recs=sim.run().records
    res=[r for r in recs if r.turn>0]; t=[r.ttft_s for r in recs]
    return dict(j,p50=_pct(t,50),p90=_pct(t,90),p99=_pct(t,99),disk=sum(r.sec_hit_blocks>0 for r in res)/len(res),
      miss=sum(r.miss_blocks for r in recs)*c.block_tokens)
def group(gj):
    out=[];bad=0
    for rate in gj['rates']:
        j={k:v for k,v in gj.items() if k!='rates'}; j['rate']=rate
        o=run(j); out.append(o)
        bad=bad+1 if o['p90']>60 else 0
        if bad>=2: break
    return out
if __name__=='__main__':
    groups=[]
    for i in range(1,6):
        for setup,rates in (('70B_256',G70),('sh59',GSH)):
            for arm in ('lru','ours','oracle','lru_touch','nodisk','infcpu'):
                groups.append(dict(exp='arms',setup=setup,trace=TR(i),arm=arm,rates=rates))
    for i in (1,3):
        for d in range(5):
            for setup,rates in (('70B_256',G70),('sh59',GSH)):
                for arm in ('lru','ours'):
                    groups.append(dict(exp='redraw',setup=setup,trace=TR(i),arm=arm,draw=d,rates=rates))
    for i in (1,3):
        for setup in ('70B_256','7B_7'):
            for mod in ('base','disk','mem4','prefill10','mem4+disk','prefill10+disk'):
                groups.append(dict(exp='ablate',setup=setup,trace=TR(i),arm='lru',mod=mod,rates=GEO))
    print(len(groups),flush=True)
    with P(6) as p,open(sys.argv[1],'w') as f:
        for o in p.map(group,groups):
            for r in o: f.write(json.dumps(r)+'\n')
            f.flush()
