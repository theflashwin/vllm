import json
rows=[json.loads(l) for l in open('results/gate/shared2.jsonl')]
def cap(p,slo):
    p=sorted(p)
    if not p or p[0][1]>slo:return 0,True
    for (a,x),(c,y) in zip(p,p[1:]):
        if x<=slo<y:return a+(slo-x)/(y-x)*(c-a),False
    return p[-1][0],True
T=sorted({r['trace'] for r in rows})
for m,slo in (('p90',5),('p90',10),('p90',30),('p99',30)):
    v=[];tops=0
    for t in T:
        f=lambda a:cap([(r['rate'],r[m]) for r in rows if r['trace']==t and r['arm']==a],slo)
        l,lt=f('lru');o,ot=f('ours');tops+=lt+ot;v.append(round((o/l-1)*100,1))
    print(m,slo,v,'mean',round(sum(v)/len(v),1),'capped',tops)
print(max(r['rate'] for r in rows))
f=[json.loads(l) for l in open('results/gate/followup.jsonl')]
print(sorted(f[0].keys()))
s=[r for r in f if 's1' in r['trace'] and r['arm']=='lru' and r.get('model')=='70B' and r.get('cpu_gb')==256 and r.get('disk_gbps')==0.96]
print([(r['rate'],round(r['p90'],1)) for r in sorted(s,key=lambda r:r['rate'])])
