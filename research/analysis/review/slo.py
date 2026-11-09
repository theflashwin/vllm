import json,collections,statistics as st,sys
import os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cap import cap,load
B='results/gate/'
T5=2.776
for name,rows,keyf in (('70B-256-0.96',[r for r in load(B+'followup.jsonl') if r['disk_gbps']==0.96 and r['trace']!='cat'],None),('70B-256-3',[r for r in load(B+'followup.jsonl') if r['disk_gbps']==3.0 and r['trace']!='cat'],None),('shared59-a25',load(B+'shared2.jsonl'),None)):
    d=collections.defaultdict(list)
    for r in rows: d[r['trace'],r['arm']].append(r)
    print(name)
    ts=sorted({t for t,_ in d})
    for metric,slos in (('p90',(2,3,5,10,30,100)),('p99',(5,10,30,60,120))):
        for slo in slos:
            gs=[]
            for t in ts:
                L=[(r['rate'],r[metric]) for r in d[t,'lru']];O=[(r['rate'],r[metric]) for r in d[t,'ours']]
                l,fl=cap(L,slo);o,fo=cap(O,slo)
                gs.append((o/l-1)*100 if l else 0.0)
            m=st.fmean(gs); sd=st.stdev(gs) if len(gs)>1 else 0
            print(f'  {metric}<={slo}: gains {[round(x,1) for x in gs]} mean {m:.1f} sd {sd:.1f} 95%CI [{m-T5*sd/len(gs)**.5:.1f},{m+T5*sd/len(gs)**.5:.1f}]')
