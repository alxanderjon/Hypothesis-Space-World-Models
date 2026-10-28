#!/usr/bin/env python3
import json, hashlib, os, sys
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def states(seed,n): return np.random.default_rng(seed).uniform(-1,1,size=(n,4))

def affine_fraction(x,d):
    z=np.concatenate([np.ones((len(x),1)),x],axis=1)
    b=np.linalg.lstsq(z,d,rcond=None)[0]; fit=z@b
    den=float(np.sum((d-d.mean(axis=0,keepdims=True))**2))
    resid=float(np.sum((d-fit)**2))
    return 1-resid/max(den,1e-30), resid, den

def cos(a,b):
    a=a.ravel(); b=b.ravel(); return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))

def main():
    root=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=False)
    specs=[('development','common_panel_labelblind_callable_recurrence_v1',2026104300),('confirmation','common_panel_labelblind_callable_recurrence_v1_confirmation',2026104400)]
    rows=[]
    for phase,name,world in specs:
        p=root/name/'predictions_before_hidden_labels.npz'; a=np.load(p); x=states(world+102,64)
        for seed in [2026104301,2026104302]:
            for i in range(16):
                no=a[f'no_module_{i}']; al=a[f'aligned_{seed}_{i}']-no; sh=a[f'shuffled_{seed}_{i}']-no
                af_a,ar_a,ad_a=affine_fraction(x,al); af_s,ar_s,ad_s=affine_fraction(x,sh)
                rows.append({'phase':phase,'seed':seed,'system':i,'group':'reuse' if i<8 else 'unrelated','aligned_affine_fraction':af_a,'shuffled_affine_fraction':af_s,'aligned_shuffled_cosine':cos(al,sh),'aligned_centered_shuffled_centered_cosine':cos(al-al.mean(0),sh-sh.mean(0)),'aligned_correction_rms':float(np.sqrt(np.mean(al*al))),'shuffled_correction_rms':float(np.sqrt(np.mean(sh*sh)))})
    summary=[]
    for phase in ['development','confirmation']:
      for seed in [2026104301,2026104302]:
       for group in ['reuse','unrelated']:
        q=[r for r in rows if r['phase']==phase and r['seed']==seed and r['group']==group]
        summary.append({'phase':phase,'seed':seed,'group':group,**{k:float(np.median([r[k] for r in q])) for k in ['aligned_affine_fraction','shuffled_affine_fraction','aligned_shuffled_cosine','aligned_centered_shuffled_centered_cosine','aligned_correction_rms','shuffled_correction_rms']}})
    result={'classification':'DESCRIPTIVE_SAVED_OUTPUT_ATTRIBUTION','counts':{'saved_prediction_arrays_read':2,'affine_lstsq':128,'derived_system_rows':64},'summary':summary,'rows':rows,'qualification':'No target labels were used. Affine correction content can explain a shortcut but does not by itself establish why an arm predicts well.'}
    tmp=out/'result.json.tmp'; tmp.write_text(json.dumps(result,indent=2,sort_keys=True)); os.replace(tmp,out/'result.json')
    (out/'completion.json').write_text(json.dumps({'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':result['counts']},indent=2,sort_keys=True))

if __name__=='__main__': main()
