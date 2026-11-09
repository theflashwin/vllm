import sys,csv,statistics as st
sys.path.insert(0,'.')
from dataclasses import replace
from kvplace.sim import CostModel,Simulator,get_policy,_pct
from kvplace.trace import read_trace
L='/Users/home/Documents/GitHub/vllm-context/research/results/live/live35_stable_a10g/'
def live(t):
    x=[float(r['ttft_s']) for r in csv.DictReader(open(L+t+'/requests.csv')) if r['kind']=='turn' and int(r['turn'])>0 and r['ttft_s']]
    return [round(_pct(x,p),3) for p in (50,90,99)],len(x)
print('live lru',live('tier_lru'),'live reuse',live('tier_reuse'))
for cj in ('costs_scan_a10g.json','costs_a10g.json'):
  try: c=CostModel.from_json(cj)
  except Exception as e: print(e); continue
  print(cj,{k:getattr(c,k) for k in ('prefill_tok_per_s','decode_s_per_tok','decode_s_per_ctx_tok','prefill_s_per_tok_per_ctx','sec_cpu_gbps','gpu_cpu_gbps')})
  for hints in (False,True):
    for pol,pred in (('lru','tool-nfh2'),('reuse_evict','tool-nfh2')):
      ss=read_trace('traces/live/live35.jsonl')
      sim=Simulator(ss,get_policy(pol),pred,c,3800,2000,10_000_000,time_scale=0.5,contend_sec=True,write_model='shared',gpu_admission=True,server_hints=hints)
      r=sim.run().records; x=[a.ttft_s for a in r if a.turn>0]
      print(' hints',('server' if hints else 'request'),pol,[round(_pct(x,p),3) for p in (50,90,99)],'n',len(x),'sec%',round(100*sum(a.sec_hit_blocks>0 for a in r if a.turn>0)/len(x),1))
