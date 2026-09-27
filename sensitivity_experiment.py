"""Supplementary stress tests for the ML-CHM benchmark (camera-ready revision).

Runs the unchanged ML-CHM / baseline detectors from camhealth_experiment.py on
generator variants that weaken or violate the lower-to-higher dependency
assumption, and on concurrent independent faults. The main-paper results are
NOT changed by this script.

Usage:  python sensitivity_experiment.py --outdir sensitivity_results
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import camhealth_experiment as ce

def make_add_fault(prop_decay):
    def add_fault(scores,start,end,layer,mag,rng):
        d=end-start; ramp=np.ones(d,dtype=np.float32); r=min(3,d//3)
        if r: ramp[:r]=np.linspace(.45,1,r); ramp[-r:]=np.linspace(1,.65,r)
        scores[start:end,layer]+=mag*ramp
        if prop_decay>0:
            for j in range(layer+1,5):
                scores[start:end,j]+=mag*(prop_decay**(j-layer))*ramp+rng.normal(0,.015,d)
    return add_fault

def simulate_camera(cfg,site,cam,rng,artefact_p=.16,concurrent_p=0.0):
    """Copy of ce.simulate_camera with the lower-layer artefact probability and an
    optional concurrent independent second fault exposed as parameters."""
    n=cfg.days*1440; t=np.arange(n,dtype=np.int32); mod=t%1440
    ctype=rng.choice(["indoor","outdoor","parking"],p=[.42,.33,.25])
    base=rng.uniform(.025,.075,5); scale=rng.uniform(.85,1.18,5)
    scores=np.empty((n,5),dtype=np.float32)
    for l in range(5): scores[:,l]=base[l]+ce.ar1(rng,n,rng.uniform(.91,.97),rng.uniform(.007,.015))
    daylight=np.maximum(0,np.sin((mod-360)/1440*2*np.pi))
    transition=((np.abs(mod-360)<=12)|(np.abs(mod-1080)<=12))
    lighting=transition.astype(np.uint8); activity=np.zeros(n,np.uint8); maintenance=np.zeros(n,np.uint8)
    if ctype in ("outdoor","parking"):
        scores[:,2]+=.035*(1-daylight); scores[:,3]+=.025*(1-daylight)
        scores[:,2]+=transition*rng.uniform(.20,.34); scores[:,3]+=transition*rng.uniform(.15,.28)
    else: scores[:,2]+=.01*np.sin(mod/1440*2*np.pi)
    events=[]; occupied=np.zeros(n,bool); probs=np.array([.16,.22,.24,.20,.18])
    med=[18,12,35,50,45]; lo=[3,2,5,6,6]; hi=[180,120,240,300,240]
    nf=rng.poisson(cfg.fault_rate_per_camera_day*cfg.days)
    for k in range(nf):
        for _ in range(100):
            layer=int(rng.choice(5,p=probs))
            dur=int(np.clip(rng.lognormal(np.log(med[layer]),.65),lo[layer],hi[layer]))
            start=int(rng.integers(60,n-dur-60)); end=start+dur
            if not occupied[max(0,start-10):min(n,end+10)].any():
                occupied[start:end]=1; mag=float(rng.uniform(.68,.93)); ce.add_fault(scores,start,end,layer,mag,rng)
                if layer>0 and rng.random()<artefact_p:
                    low=int(rng.integers(0,layer)); ad=min(end-start,int(rng.integers(2,7)))
                    scores[start:start+ad,low]+=rng.uniform(.52,.70)
                events.append(dict(site=site,camera=cam,camera_type=ctype,event_id=f"S{site:02d}C{cam:02d}E{k:03d}",layer=f"L{layer}",start=start,end=end,duration_min=dur,magnitude=mag,secondary=False))
                if concurrent_p>0 and rng.random()<concurrent_p and dur>=10:
                    l2=int(rng.choice([x for x in range(5) if x!=layer]))
                    s2=start+int(rng.integers(dur//4,max(dur//4+1,dur//2)))
                    d2=int(np.clip(rng.lognormal(np.log(med[l2]),.65),lo[l2],hi[l2])); e2=min(n-1,s2+d2)
                    m2=float(rng.uniform(.68,.93)); ce.add_fault(scores,s2,e2,l2,m2,rng); occupied[s2:e2]=1
                    events.append(dict(site=site,camera=cam,camera_type=ctype,event_id=f"S{site:02d}C{cam:02d}E{k:03d}b",layer=f"L{l2}",start=s2,end=e2,duration_min=e2-s2,magnitude=m2,secondary=True))
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
        if kind in ("illumination","weather"): scores[start:end,3]+=mag*.55; lighting[start:end]=1
        if kind=="activity": scores[start:end,4]+=mag*.35; activity[start:end]=1
        if kind=="maintenance": maintenance[start:end]=1
    scores*=scale; scores+=rng.normal(0,.012,scores.shape); scores=np.clip(scores,0,1)
    labels=np.full(n,-1,np.int8)
    for e in events: labels[e["start"]:e["end"]]=int(e["layer"][1])
    miss=rng.random(scores.shape)<cfg.baseline_missingness; scores[miss]=np.nan
    return dict(site=site,camera=cam,camera_type=ctype,scores=scores,lighting=lighting,activity=activity,maintenance=maintenance,labels=labels),events

SPECS=[("Flat multi-signal","flat_fusion",dict(episode_suppression=True,dependency_ordering=False,context_gates=False,hysteresis=True)),
       ("ML-CHM full","mlchm",dict(episode_suppression=True,dependency_ordering=True,context_gates=True,hysteresis=True)),
       ("ML-CHM, no dependency order","mlchm",dict(episode_suppression=True,dependency_ordering=False,context_gates=True,hysteresis=True))]

VARIANTS={
 "S0 reference (paper setting)":dict(prop=.90,artefact_p=.16,concurrent_p=0.0),
 "S1 weak propagation (decay 0.45)":dict(prop=.45,artefact_p=.16,concurrent_p=0.0),
 "S2 no propagation (independent layers)":dict(prop=0.0,artefact_p=.16,concurrent_p=0.0),
 "S3 frequent lower-layer artefacts (p=0.50)":dict(prop=.90,artefact_p=.50,concurrent_p=0.0),
 "S4 concurrent independent faults (p=0.25)":dict(prop=.90,artefact_p=.16,concurrent_p=.25),
}

def run_variant(name,v):
    cfg=ce.SimConfig(); rng=np.random.default_rng(cfg.seed)
    ce.add_fault=make_add_fault(v["prop"])
    dev,test,test_ev=[],[],[]
    for s in range(cfg.n_sites):
        for c in range(cfg.cameras_per_site):
            cam,ev=simulate_camera(cfg,s,c,rng,v["artefact_p"],v["concurrent_p"])
            (dev if s<cfg.dev_sites else test).append(cam)
            if s>=cfg.dev_sites: test_ev+=ev
    th=ce.thresholds_from_dev(dev); cd=(cfg.n_sites-cfg.dev_sites)*cfg.cameras_per_site*cfg.days
    rows=[]
    n_sec=sum(e["secondary"] for e in test_ev)
    for label,method,kw in SPECS:
        st,m,p=ce.evaluate(test,test_ev,method,th,cd,label,**kw)
        esc=int(round(st["server_escalation_pct"]*len(p)/100)) if len(p) else 0
        row=dict(variant=name,method=label,n_test_events=len(test_ev),n_secondary=n_sec,
                 precision=st["precision"],recall=st["recall"],f1=st["f1"],root_cause_accuracy=st["root_cause_accuracy"],
                 fa_per_100_cd=st["false_alarms_per_100_camera_days"],mttd_min=st["mttd_min"],p95_delay_min=st["p95_delay_min"],
                 server_escalations=esc)
        if n_sec:
            sec=pd.DataFrame([e for e in test_ev if e["secondary"]])
            # a secondary fault counts as found only if it received its own matched episode
            key=set(zip(m.site,m.camera,m.true_layer)) if len(m) else set()
            row["secondary_recall_upper_bound"]=float(np.mean([(r.site,r.camera,r.layer) in key for r in sec.itertuples()]))
        rows.append(row)
    return rows,th

def main(outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); orig=ce.add_fault; allrows=[]; ths={}
    for name,v in VARIANTS.items():
        rows,th=run_variant(name,v); allrows+=rows; ths[name]=th; print(name,th)
    ce.add_fault=orig
    df=pd.DataFrame(allrows); df.to_csv(out/"sensitivity_metrics.csv",index=False)
    json.dump(dict(variants=VARIANTS,thresholds=ths),open(out/"sensitivity_config.json","w"),indent=2)
    pd.set_option("display.width",250); print(df.round(3).to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--outdir",default="sensitivity_results"); main(ap.parse_args().outdir)
