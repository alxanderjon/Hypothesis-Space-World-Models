#!/usr/bin/env python3
import argparse,importlib.util,json,hashlib,os
from pathlib import Path
import numpy as np
def sha(p): h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def write(p,x): p=Path(p); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)); os.replace(t,p)
def add(a,b):
 keys=set(a)|set(b); return {k:int(a.get(k,0))+int(b.get(k,0)) for k in keys}
def main(a):
 root=Path(a.root); rp=json.load(open(a.recovery_plan)); op=json.load(open(a.original_plan)); selbar=json.load(open(root/'selection_barrier.json'))
 spec=importlib.util.spec_from_file_location('orig',a.driver); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); g=m.Guard(rp['recovery_ceilings'])
 write(root/'terminal_error.json',{'status':'ORIGINAL_ERROR_PRESERVED','error_type':'TypeError','message':'Object of type bool is not JSON serializable','failed_artifact':'admission_decision.json','original_spent_counts':rp['original_spent_counts'],'scientific_values_persisted':False})
 seed=2026105800; systems=m.make_systems(seed,7); x0=m.states(seed+1,32); xv=m.states(seed+2,32); pool=m.states(seed+3,32); y0=[m.trans(x0,s,g) for s in systems]; yv=[m.trans(xv,s,g) for s in systems]
 r0=m.quotient(x0,y0,g); sel,signs=m.signature(r0,g)
 if sel.tolist()!=selbar['selected_systems']: raise RuntimeError('SELECTION_RECONSTRUCTION_MISMATCH')
 arms={'original32':(x0,y0)}
 for name,ii0 in selbar['indices'].items():
  ii=np.array(ii0); ys=[m.trans(pool[ii],s,g) for s in systems]; arms[name]=(np.concatenate([x0,pool[ii]]),[np.concatenate([y0[i],ys[i]]) for i in range(16)])
 admissions=[]; funcs={}
 for name,(x,ys) in arms.items():
  rr=m.quotient(x,ys,g) if name!='original32' else r0; h=m.kernel(x,rr,sel,signs,g); funcs[name]=h; val,vals=m.admission(x,ys,xv,yv,h,sel,g); lin=[]
  for i in sel: lin.append(m.score(xv+m.DT*(xv@m.fit_linear(x,ys[i],g)),yv[i],g))
  ratio=float(val/np.mean(lin)); wins=int(sum(q<w for q,w in zip(vals,lin))); admissions.append({'arm':name,'ratio':ratio,'wins':wins,'passed':bool(ratio<=.8 and wins>=6)})
 write(root/'recovered_admission_decision.json',{'status':'RECOVERED_AND_FROZEN_BEFORE_TARGETS','admissions':admissions,'recovery_counts':g.n.copy(),'original_spent_counts':rp['original_spent_counts']})
 context=[r for r in admissions if r['arm']=='learned_context48'][0]
 if not context['passed']:
  cumulative=add(rp['original_spent_counts'],g.n); res={'classification':'CONTEXT_POLICY_NOT_SUPPORTED_AT_ADMISSION','admissions':admissions,'counts':{'original_spent':rp['original_spent_counts'],'recovery':g.n,'cumulative':cumulative},'qualification':op['qualification'],'recovery_qualification':'No policy training repeated; reserved-family response and admission work repeated because values were not persisted.'}; write(root/'result.json',res); write(root/'completion.json',{'status':'COMPLETE_AFTER_ADDITIVE_RECOVERY','result_sha256':sha(root/'result.json'),'counts':res['counts']}); return
 xd=m.states(seed+101,32); xq=m.states(seed+102,64); rt=np.random.default_rng(seed+1000); shared=systems[0][2]; target=[(m.stable_A(rt),rt.uniform(.6,1.4),shared if i<8 else m.M(rt),i<8,7) for i in range(16)]; dev=[(s,m.trans(xd,s,g),m.trans(xq,s,g)) for s in target]
 preds={}
 for name,h in funcs.items():
  hd=h(xd); hq=h(xq); g.charge('prediction_blocks',2)
  for i,(s,yd,yq) in enumerate(dev): preds[f'{name}_{i}']=m.pred_aug(xq,hq,m.fit_aug(xd,yd,hd,g))
 for i,(s,yd,yq) in enumerate(dev): preds[f'linear_{i}']=xq+m.DT*(xq@m.fit_linear(xd,yd,g))
 np.savez_compressed(root/'predictions_before_hidden_labels.npz',**preds); write(root/'prediction_barrier.json',{'status':'PERSISTED_BEFORE_HIDDEN_SCORING','sha256':sha(root/'predictions_before_hidden_labels.npz'),'recovery_counts':g.n.copy()})
 rows=[]
 for i,(s,yd,yq) in enumerate(dev):
  for arm in list(arms)+['linear']: rows.append({'system':i,'group':'reuse' if s[3] else 'unrelated','arm':arm,'MSE':m.score(preds[f'{arm}_{i}'],yq,g)})
 def v(group,arm):return [r['MSE'] for r in rows if r['group']==group and r['arm']==arm]
 cc=v('reuse','learned_context48'); rr=v('reuse','random48'); ss=v('reuse','learned_state_only48'); hh=v('reuse','frozen_disagreement48'); oo=v('reuse','original32'); au=v('unrelated','learned_context48'); ll=v('unrelated','linear'); rat=lambda x,y:sum(x)/sum(y)
 comp={'context_over_random':rat(cc,rr),'context_wins_random':sum(x<y for x,y in zip(cc,rr)),'context_over_state_only':rat(cc,ss),'context_wins_state_only':sum(x<y for x,y in zip(cc,ss)),'context_over_heuristic':rat(cc,hh),'context_wins_heuristic':sum(x<y for x,y in zip(cc,hh)),'context_over_original':rat(cc,oo),'context_wins_original':sum(x<y for x,y in zip(cc,oo)),'context_unrelated_over_linear':rat(au,ll)}
 passed=comp['context_over_random']<=.85 and comp['context_wins_random']>=6 and comp['context_over_state_only']<=.85 and comp['context_wins_state_only']>=6 and comp['context_over_heuristic']<=.85 and comp['context_wins_heuristic']>=6 and comp['context_over_original']<=.8 and comp['context_wins_original']>=6 and comp['context_unrelated_over_linear']<=1.1
 cumulative=add(rp['original_spent_counts'],g.n); res={'classification':'LEARNED_CONTEXT_ACQUISITION_SUPPORTED_DEVELOPMENT_ONLY' if passed else 'NOT_SUPPORTED','admissions':admissions,'comparison':comp,'scores':rows,'counts':{'original_spent':rp['original_spent_counts'],'recovery':g.n,'cumulative':cumulative},'qualification':op['qualification'],'recovery_qualification':'No policy training repeated; only unpersisted reserved-family work was repeated.'}; write(root/'result.json',res); write(root/'completion.json',{'status':'COMPLETE_AFTER_ADDITIVE_RECOVERY','result_sha256':sha(root/'result.json'),'counts':res['counts']})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--driver');ap.add_argument('--original-plan');ap.add_argument('--recovery-plan');ap.add_argument('--root');main(ap.parse_args())
