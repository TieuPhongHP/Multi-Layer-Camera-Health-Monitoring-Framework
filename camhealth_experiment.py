from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.signal import lfilter

LAYERS=["L0","L1","L2","L3","L4"]

@dataclass
class SimConfig:
    seed:int=20260728
    n_sites:int=10
    cameras_per_site:int=8
    days:int=10
    dev_sites:int=7
    fault_rate_per_camera_day:float=0.45
    nuisance_rate_per_camera_day:float=1.40
    baseline_missingness:float=0.05


def ar1(rng,n,phi,sigma):
    return lfilter([1.0],[1.0,-phi],rng.normal(0,sigma,n)).astype(np.float32)

def add_fault(scores,start,end,layer,mag,rng):
    d=end-start
    ramp=np.ones(d,dtype=np.float32)
    r=min(3,d//3)
    if r:
        ramp[:r]=np.linspace(.45,1,r); ramp[-r:]=np.linspace(1,.65,r)
    scores[start:end,layer]+=mag*ramp
    for j in range(layer+1,5):
        scores[start:end,j]+=mag*(.90**(j-layer))*ramp+rng.normal(0,.015,d)

def simulate_camera(cfg,site,cam,rng):
    n=cfg.days*1440
    t=np.arange(n,dtype=np.int32); mod=t%1440
    ctype=rng.choice(["indoor","outdoor","parking"],p=[.42,.33,.25])
    base=rng.uniform(.025,.075,5); scale=rng.uniform(.85,1.18,5)
    scores=np.empty((n,5),dtype=np.float32)
    for l in range(5):
        scores[:,l]=base[l]+ar1(rng,n,rng.uniform(.91,.97),rng.uniform(.007,.015))
    daylight=np.maximum(0,np.sin((mod-360)/1440*2*np.pi))
    transition=((np.abs(mod-360)<=12)|(np.abs(mod-1080)<=12))
    lighting=transition.astype(np.uint8); activity=np.zeros(n,np.uint8); maintenance=np.zeros(n,np.uint8)
    if ctype in ("outdoor","parking"):
        scores[:,2]+=.035*(1-daylight); scores[:,3]+=.025*(1-daylight)
        scores[:,2]+=transition*rng.uniform(.20,.34); scores[:,3]+=transition*rng.uniform(.15,.28)
    else:
        scores[:,2]+=.01*np.sin(mod/1440*2*np.pi)
    events=[]; occupied=np.zeros(n,bool)
    probs=np.array([.16,.22,.24,.20,.18])
    nf=rng.poisson(cfg.fault_rate_per_camera_day*cfg.days)
    for k in range(nf):
        for _ in range(100):
            layer=int(rng.choice(5,p=probs))
            med=[18,12,35,50,45][layer]; lo=[3,2,5,6,6][layer]; hi=[180,120,240,300,240][layer]
            dur=int(np.clip(rng.lognormal(np.log(med),.65),lo,hi))
            start=int(rng.integers(60,n-dur-60)); end=start+dur
            if not occupied[max(0,start-10):min(n,end+10)].any():
                occupied[start:end]=1; mag=float(rng.uniform(.68,.93)); add_fault(scores,start,end,layer,mag,rng)
                # Diagnostic ambiguity: a minority of events contain a short lower-layer measurement artifact
                # (e.g., decoder jitter during a scene/quality fault) that should not redefine the labeled root cause.
                if layer > 0 and rng.random() < 0.16:
                    low = int(rng.integers(0, layer))
                    ad = min(end-start, int(rng.integers(2, 7)))
                    scores[start:start+ad, low] += rng.uniform(.52, .70)
                events.append(dict(site=site,camera=cam,camera_type=ctype,event_id=f"S{site:02d}C{cam:02d}E{k:03d}",layer=f"L{layer}",start=start,end=end,duration_min=dur,magnitude=mag))
                break
    nn=rng.poisson(cfg.nuisance_rate_per_camera_day*cfg.days)
    for _ in range(nn):
        kind=rng.choice(["illumination","activity","weather","maintenance","network_transient"],p=[.25,.27,.20,.13,.15])
        if kind=="illumination": dur=int(rng.integers(2,12)); layer=2; mag=rng.uniform(.34,.58)
        elif kind=="activity": dur=int(rng.integers(2,20)); layer=3; mag=rng.uniform(.34,.60)
        elif kind=="weather": dur=int(rng.integers(10,90)); layer=2; mag=rng.uniform(.25,.48)
        elif kind=="maintenance": dur=int(rng.integers(10,60)); layer=int(rng.choice([0,1,2,3])); mag=rng.uniform(.45,.72)
        else: dur=int(rng.integers(1,4)); layer=1; mag=rng.uniform(.36,.60)
        start=int(rng.integers(30,n-dur-30)); end=start+dur
        if occupied[start:end].any(): continue
        scores[start:end,layer]+=mag
        if kind in ("illumination","weather"):
            scores[start:end,3]+=mag*.55; lighting[start:end]=1
        if kind=="activity":
            scores[start:end,4]+=mag*.35; activity[start:end]=1
        if kind=="maintenance": maintenance[start:end]=1
    scores*=scale; scores+=rng.normal(0,.012,scores.shape); scores=np.clip(scores,0,1)
    labels=np.full(n,-1,np.int8)
    for e in events: labels[e["start"]:e["end"]]=int(e["layer"][1])
    miss=rng.random(scores.shape)<cfg.baseline_missingness; scores[miss]=np.nan
    return dict(site=site,camera=cam,camera_type=ctype,scores=scores,lighting=lighting,activity=activity,maintenance=maintenance,labels=labels),events

def thresholds_from_dev(cams):
    vals=[[] for _ in range(5)]
    rng=np.random.default_rng(12)
    for c in cams:
        clean=c["labels"]<0
        idx=np.where(clean)[0]
        if len(idx)>3000: idx=rng.choice(idx,3000,replace=False)
        for i in range(5):
            x=c["scores"][idx,i]; vals[i].append(x[np.isfinite(x)])
    th={}
    for i in range(5):
        x=np.concatenate(vals[i]); q=float(np.quantile(x,.997))
        th[f"L{i}"]=max([.58,.58,.62,.64,.64][i],min(q,[.74,.74,.78,.80,.80][i]))
    return th

def detect(c,method,th,episode_suppression=True,dependency_ordering=True,context_gates=True,hysteresis=True,extra_missing=0.0,seed=0):
    scores=c["scores"].copy(); n=len(scores)
    if extra_missing:
        rng=np.random.default_rng(seed+c["site"]*101+c["camera"]); scores[rng.random(scores.shape)<extra_missing]=np.nan
    if method=="online_only": cand_layers=[0]; persist=[2]*5
    elif method=="visual_only": cand_layers=[2,3]; persist=[3]*5
    elif method=="flat_fusion": cand_layers=list(range(5)); persist=[3]*5
    else: cand_layers=list(range(5)); persist=[2,3,4,4,5]
    episodes=[]; active=False; active_layer=None; start=0; on=np.zeros(5,int); off=0; last_emit=-999999
    last=np.full(5,np.nan); age=np.full(5,999)
    base_th=np.array([th[f"L{i}"] for i in range(5)])
    for t in range(n):
        row=scores[t].copy()
        finite=np.isfinite(row)
        last[finite]=row[finite]; age[finite]=0; age[~finite]+=1
        carry=(~finite)&(age<=2); row[carry]=last[carry]
        cur=base_th.copy()
        if context_gates:
            if c["lighting"][t]: cur[2]+=.16; cur[3]+=.14
            if c["activity"][t]: cur[3]+=.11; cur[4]+=.07
            if c["maintenance"][t]: cur+=.35
        above=np.isfinite(row)&(row>=cur)
        if method=="online_only": cand=0 if above[0] else None
        elif method=="visual_only":
            vv=[i for i in cand_layers if above[i]]; cand=max(vv,key=lambda i:row[i]) if vv else None
        elif method=="flat_fusion":
            vv=[i for i in cand_layers if above[i]]; cand=max(vv,key=lambda i:row[i]-cur[i]) if vv else None
        else:
            vv=[i for i in cand_layers if above[i]]
            cand=(min(vv) if dependency_ordering else max(vv,key=lambda i:row[i]-cur[i])) if vv else None
        if not active:
            if cand is None: on[:]=0; continue
            on[cand]+=1; on[np.arange(5)!=cand]=0
            if on[cand]>=persist[cand]:
                start=t-persist[cand]+1; active=True; active_layer=cand; off=0
                if not episode_suppression:
                    episodes.append(dict(site=c["site"],camera=c["camera"],layer=f"L{cand}",start=start,end=t+1)); last_emit=t
        else:
            if dependency_ordering and cand is not None and cand<active_layer: active_layer=cand
            off_th=cur[active_layer]*(.64 if hysteresis else 1.0)
            if (not np.isfinite(row[active_layer])) or row[active_layer]<off_th: off+=1
            else: off=0
            if not episode_suppression and t-last_emit>=5:
                episodes.append(dict(site=c["site"],camera=c["camera"],layer=f"L{active_layer}",start=t,end=t+1)); last_emit=t
            if off>=3:
                if episode_suppression: episodes.append(dict(site=c["site"],camera=c["camera"],layer=f"L{active_layer}",start=start,end=t-off+2))
                active=False; active_layer=None; on[:]=0; off=0
    if active and episode_suppression: episodes.append(dict(site=c["site"],camera=c["camera"],layer=f"L{active_layer}",start=start,end=n))
    return episodes

def match(true_events,pred,tol=5):
    true=[dict(e,matched=False) for e in true_events]; pred=[dict(e,matched=False) for e in pred]; matches=[]
    bycam={}
    for pi,p in enumerate(pred): bycam.setdefault((p["site"],p["camera"]),[]).append(pi)
    for ti,e in enumerate(true):
        cand=[]
        for pi in bycam.get((e["site"],e["camera"]),[]):
            if pred[pi]["matched"]: continue
            p=pred[pi]; overlap=max(0,min(e["end"],p["end"])-max(e["start"],p["start"])); near=abs(p["start"]-e["start"])<=tol
            if overlap or near:
                union=max(e["end"],p["end"])-min(e["start"],p["start"]); score=overlap/union if union else 0
                cand.append((score,overlap,-abs(p["start"]-e["start"]),pi))
        if cand:
            pi=max(cand)[-1]; true[ti]["matched"]=True; pred[pi]["matched"]=True
            matches.append(dict(site=e["site"],camera=e["camera"],true_layer=e["layer"],pred_layer=pred[pi]["layer"],delay_min=max(0,pred[pi]["start"]-e["start"]),correct_root=e["layer"]==pred[pi]["layer"]))
    tp=sum(e["matched"] for e in true); fp=sum(not p["matched"] for p in pred); fn=len(true)-tp
    pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; f1=2*pr*rc/(pr+rc) if pr+rc else 0
    m=pd.DataFrame(matches)
    return dict(tp=tp,fp=fp,fn=fn,precision=pr,recall=rc,f1=f1,root_cause_accuracy=float(m.correct_root.mean()) if len(m) else 0,mttd_min=float(m.delay_min.mean()) if len(m) else np.nan,p95_delay_min=float(m.delay_min.quantile(.95)) if len(m) else np.nan),m,pd.DataFrame(pred)

def evaluate(cams,events,method,th,camera_days,label,extra_missing=0.0,**kwargs):
    pred=[]
    for c in cams: pred.extend(detect(c,method,th,extra_missing=extra_missing,seed=202699,**kwargs))
    st,m,pdf=match(events,pred)
    st.update(label=label,method=method,false_alarms_per_100_camera_days=st["fp"]/camera_days*100,alerts_per_true_event=len(pred)/max(1,len(events)))
    if len(pdf): st["server_escalation_pct"]=float(((pdf.layer.isin(["L3","L4"]))|((pdf.layer=="L2")&((pdf.end-pdf.start)>=10))).mean()*100)
    else: st["server_escalation_pct"]=0
    return st,m,pdf

def main(outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); (out/"predictions").mkdir(exist_ok=True); (out/"matches").mkdir(exist_ok=True)
    cfg=SimConfig(); rng=np.random.default_rng(cfg.seed); dev=[]; test=[]; all_events=[]; test_events=[]
    for s in range(cfg.n_sites):
        for c in range(cfg.cameras_per_site):
            cam,ev=simulate_camera(cfg,s,c,rng); all_events+=ev
            if s<cfg.dev_sites: dev.append(cam)
            else: test.append(cam); test_events+=ev
    th=thresholds_from_dev(dev); camera_days=(cfg.n_sites-cfg.dev_sites)*cfg.cameras_per_site*cfg.days
    specs=[
      ("Connectivity-only","online_only",dict(episode_suppression=True,dependency_ordering=True,context_gates=False,hysteresis=True)),
      ("Visual-only","visual_only",dict(episode_suppression=True,dependency_ordering=False,context_gates=False,hysteresis=True)),
      ("Flat multi-signal","flat_fusion",dict(episode_suppression=True,dependency_ordering=False,context_gates=False,hysteresis=True)),
      ("ML-CHM full","mlchm",dict(episode_suppression=True,dependency_ordering=True,context_gates=True,hysteresis=True)),
      ("Ablation: no context gates","mlchm",dict(episode_suppression=True,dependency_ordering=True,context_gates=False,hysteresis=True)),
      ("Ablation: no dependency order","mlchm",dict(episode_suppression=True,dependency_ordering=False,context_gates=True,hysteresis=True)),
      ("Ablation: no hysteresis","mlchm",dict(episode_suppression=True,dependency_ordering=True,context_gates=True,hysteresis=False)),
      ("Ablation: no episode suppression","mlchm",dict(episode_suppression=False,dependency_ordering=True,context_gates=True,hysteresis=True)),
    ]
    rows=[]
    for label,method,kw in specs:
        st,m,p=evaluate(test,test_events,method,th,camera_days,label,**kw); rows.append(st)
        safe=label.lower().replace(":","").replace(" ","_").replace("-","_")
        m.to_csv(out/"matches"/f"{safe}.csv",index=False); p.to_csv(out/"predictions"/f"{safe}.csv",index=False)
    res=pd.DataFrame(rows); res.to_csv(out/"summary_metrics.csv",index=False)
    pd.DataFrame(all_events).to_csv(out/"all_true_events.csv",index=False); pd.DataFrame(test_events).to_csv(out/"test_true_events.csv",index=False)
    json.dump(asdict(cfg),open(out/"simulation_config.json","w"),indent=2); json.dump(th,open(out/"thresholds.json","w"),indent=2)
    robust=[]
    for miss in [0,.10,.20,.30]:
        st,_,_=evaluate(test,test_events,"mlchm",th,camera_days,f"missing={miss:.2f}",extra_missing=miss,episode_suppression=True,dependency_ordering=True,context_gates=True,hysteresis=True); robust.append(dict(additional_missingness=miss,**st))
    pd.DataFrame(robust).to_csv(out/"robustness_missingness.csv",index=False)
    print(res[["label","precision","recall","f1","root_cause_accuracy","false_alarms_per_100_camera_days","mttd_min","alerts_per_true_event","server_escalation_pct"]].to_string(index=False))
    print('thresholds',th,'test events',len(test_events),'camera_days',camera_days)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',default='/mnt/data/camhealth_results'); a=ap.parse_args(); main(a.outdir)
