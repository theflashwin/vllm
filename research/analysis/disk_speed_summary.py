import json
rows=[json.loads(l) for l in open('results/gate/disk_speed.jsonl')]
def cap(p,slo=5):
    p=sorted(p)
    if not p or p[0][1]>slo:return 0,True
    for (a,x),(c,y) in zip(p,p[1:]):
        if x<=slo<y:return a+(slo-x)/(y-x)*(c-a),False
    return p[-1][0],True
S=[]
for r in rows:
    if r['setup'] not in S:S.append(r['setup'])
SP=[0.96,3.0,7.0,1e6]
for st in S:
    T=sorted({r['trace'] for r in rows if r['setup']==st})
    print('##',st)
    for t in T:
        out=[]
        for g in SP:
            for a in ('lru','ours'):
                c,top=cap([(r['rate'],r['p90']) for r in rows if (r['setup'],r['trace'],r['gbps'],r['arm'])==(st,t,g,a)])
                out.append(f"{a[0]}{c:.4f}{'^' if top else ''}")
        print(t.split('/')[-1][:12],' | '.join(out[i]+' '+out[i+1] for i in range(0,8,2)))
    # mean ratios
    for a in ('lru','ours'):
        gains=[]
        for g in SP:
            v=[]
            for t in T:
                b=cap([(r['rate'],r['p90']) for r in rows if (r['setup'],r['trace'],r['gbps'],r['arm'])==(st,t,0.96,'lru')])[0]
                c=cap([(r['rate'],r['p90']) for r in rows if (r['setup'],r['trace'],r['gbps'],r['arm'])==(st,t,g,a)])[0]
                v.append((c/b-1)*100)
            gains.append(f"{g:g}: {sum(v)/len(v):+.0f}% ({min(v):+.0f},{max(v):+.0f})")
        print(' vs LRU@0.96',a,' ; '.join(gains))
    # ours vs lru per speed
    g2=[]
    for g in SP:
        v=[]
        for t in T:
            b=cap([(r['rate'],r['p90']) for r in rows if (r['setup'],r['trace'],r['gbps'],r['arm'])==(st,t,g,'lru')])[0]
            c=cap([(r['rate'],r['p90']) for r in rows if (r['setup'],r['trace'],r['gbps'],r['arm'])==(st,t,g,'ours')])[0]
            v.append((c/b-1)*100)
        g2.append(f"{g:g}: {sum(v)/len(v):+.0f}%")
    print(' ours/lru',' ; '.join(g2))
    # breakdown near LRU capacity at 0.96: highest grid rate with p90<=5
    for t in T:
      for g in (0.96,1e6):
        rr=[r for r in rows if (r['setup'],r['trace'],r['gbps'],r['arm'])==(st,t,g,'lru') and r['p90']<=5]
        if not rr: print(' none',t,g); continue
        r=max(rr,key=lambda r:r['rate']); n=r['near_p90']
        print(f"  {t.split('/')[-1][:12]} g={g:g} rate={r['rate']} p90={r['p90']:.2f} near: queue {n['queue']:.2f} disk {n['disk']:.2f} link {n['link']:.2f} prefill {n['prefill']:.2f} ttft {n['ttft']:.2f} diskshare {r['disk_share']:.3f}")
