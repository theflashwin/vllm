import json,collections,statistics as st,sys
def cap(pts,slo=5.0):
    pts=sorted(pts)
    if pts[0][1]>slo: return 0.0,'below'
    for (a,x),(b,y) in zip(pts,pts[1:]):
        if x<=slo<y: return a+(b-a)*(slo-x)/(y-x),''
    return pts[-1][0],'cap'
def load(f): return [json.loads(l) for l in open(f)]
B='results/gate/'
def table(rows,key,tkey='trace',label=''):
    d=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows: d[key(r)][(r[tkey],r['arm'])].append((r['rate'],r['p90']))
    print('##',label)
    for k,v in sorted(d.items(),key=str):
        ts=sorted({t for t,_ in v}); out=[];gs=[]
        for t in ts:
            if (t,'lru') not in v or (t,'ours') not in v: continue
            l,lf=cap(v[t,'lru']);o,of=cap(v[t,'ours'])
            g=(o/l-1)*100 if l else float('nan'); gs.append(g)
            out.append(f"{t.split('/')[-1][:14]}: {l:.4f}{lf} {o:.4f}{of} {g:+.1f}% (n={len(v[t,'lru'])})")
        print(k,'mean',round(st.fmean(gs),1) if gs else None); [print('   ',x) for x in out]
if __name__=='__main__':
    table(load(B+'gate.jsonl'),lambda r:(r['model'],r['cpu_gb'],r['disk_gbps']),label='gate')
    table(load(B+'followup.jsonl'),lambda r:(r['model'],r['cpu_gb'],r['disk_gbps']),label='followup')
    table(load(B+'shared.jsonl'),lambda r:(r['pool'],r['aff'],r['gbps']),label='shared')
    table(load(B+'shared2.jsonl'),lambda r:(r['pool'],r['aff'],r['gbps']),label='shared2')
    # bimodal vs gate ours
    g=load(B+'gate.jsonl');b=load(B+'bimodal.jsonl')
    print('## bimodal')
    idx=collections.defaultdict(list);ib=collections.defaultdict(list)
    for r in g:
        if r['arm']=='ours':idx[(r['model'],r['cpu_gb'],r['disk_gbps'],r['trace'])].append((r['rate'],r['p90']))
        else: pass
    for r in b: ib[(r['model'],r['cpu_gb'],r['disk_gbps'],r['trace'])].append((r['rate'],r['p90']))
    for k in sorted(ib): print(k,'tool',cap(idx[k]),'toolres',cap(ib[k]))
    # disk speed
    print('## disk speed')
    rows=load(B+'disk_speed.jsonl')
    d=collections.defaultdict(dict)
    for r in rows: d[(r['setup'],r['trace'],r['gbps'],r['arm'])].setdefault('pts',[]).append((r['rate'],r['p90']))
    setups=sorted({r['setup'] for r in rows}); 
    for s in setups:
        for g in (0.96,3.0,7.0,1e6):
            base=[];gain=[];inf=[]
            for t in sorted({k[1] for k in d if k[0]==s}):
                b0,_=cap(d[(s,t,0.96,'lru')]['pts']);l,fl=cap(d[(s,t,g,'lru')]['pts']);o,fo=cap(d[(s,t,g,'ours')]['pts'])
                base.append((l/b0-1)*100 if b0 else float('nan'));gain.append((o/l-1)*100 if l else float('nan'));inf.append(fl+fo)
            print(s,g,'lru rel 0.96 lru',[round(x,1) for x in base],'mean',round(st.fmean(base),1),' ours/lru',[round(x,1) for x in gain],'mean',round(st.fmean(gain),1),inf)
