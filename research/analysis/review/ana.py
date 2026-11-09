import json,collections,statistics as st,sys,math
import os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cap import cap,load
R=load(sys.argv[1])
T=2.776
print('rows',len(R))
d=collections.defaultdict(list)
for r in R: d[r['exp'],r['setup'],r['trace'],r.get('arm'),r.get('draw'),r.get('mod')].append((r['rate'],r['p90']))
c={k:cap(v) for k,v in d.items()}
def g(exp,setup,t,a,dr=None,m=None): return c.get((exp,setup,t,a,dr,m))
traces=sorted({r['trace'] for r in R})
for setup in ('70B_256','sh59'):
    print('==',setup,'capacity by arm, gain vs lru per trace; flag cap/below')
    for a in ('ours','oracle','lru_touch','nodisk','infcpu'):
        gs=[];fl=[]
        for t in traces:
            l=g('arms',setup,t,'lru'); o=g('arms',setup,t,a)
            if l is None or o is None: continue
            gs.append((o[0]/l[0]-1)*100 if l[0] else float('nan')); fl.append(o[1])
        m=st.fmean(gs); sd=st.stdev(gs) if len(gs)>1 else 0
        print(f'  {a:10s}',[round(x,1) for x in gs],f'mean {m:.1f} sd {sd:.1f} CI[{m-T*sd/len(gs)**.5:.1f},{m+T*sd/len(gs)**.5:.1f}]',fl)
    print('  lru capacity',[round(g('arms',setup,t,'lru')[0],4) for t in traces])
print('== redraw')
for setup in ('70B_256','sh59'):
    for t in sorted({k[2] for k in d if k[0]=='redraw'}):
        ls=[];os_=[];gs=[]
        for dr in range(5):
            l=g('redraw',setup,t,'lru',dr); o=g('redraw',setup,t,'ours',dr)
            ls.append(round(l[0],5));os_.append(round(o[0],5));gs.append((o[0]/l[0]-1)*100 if l[0] else float('nan'))
        print(setup,t.split('/')[-1][:14],'lru',ls,'ours',os_,'gain',[round(x,1) for x in gs],'mean',round(st.fmean(gs),1),'sd',round(st.stdev(gs),1),'lru cv%',round(100*st.stdev(ls)/st.fmean(ls),1))
print('== ablations (lru capacity)')
for setup in ('70B_256','7B_7'):
    for t in sorted({k[2] for k in d if k[0]=='ablate'}):
        b=g('ablate',setup,t,'lru',None,'base')
        out={m:(round(g('ablate',setup,t,'lru',None,m)[0],5),g('ablate',setup,t,'lru',None,m)[1]) for m in ('base','disk','mem4','prefill10','mem4+disk','prefill10+disk')}
        print(setup,t.split('/')[-1][:14],{k:(v[0],f"{(v[0]/b[0]-1)*100:+.0f}%",v[1]) for k,v in out.items()})
