#!/usr/bin/env python3
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

DT = 0.05
SEEDS = [2026104301, 2026104302]
WORLD_SEED = 2026104300

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

class Guard:
    def __init__(self, ceilings):
        self.c={k:int(v) if isinstance(v,int) else v for k,v in ceilings.items()}
        self.n={k:0 for k in ceilings if k not in ('wall_seconds','peak_RSS_bytes','artifact_bytes')}
        self.t=time.time()
    def check_resource(self):
        if time.time()-self.t > self.c['wall_seconds']: raise RuntimeError('WALL_CAP')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform!='darwin': rss*=1024
        if rss > self.c['peak_RSS_bytes']: raise RuntimeError('RSS_CAP')
    def charge(self,k,v=1):
        self.check_resource()
        if self.n[k]+v > self.c[k]: raise RuntimeError(f'CAP:{k}:{self.n[k]}+{v}>{self.c[k]}')
        self.n[k]+=v

class Callable(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,4))
    def forward(self,x): return self.net(x)

def gen_net(rng):
    w1=rng.normal(size=(4,16))/2; b1=rng.normal(scale=.2,size=16)
    w2=rng.normal(size=(16,4))/4; b2=rng.normal(scale=.1,size=4)
    return (w1,b1,w2,b2)

def apply_net(x,p):
    w1,b1,w2,b2=p
    return np.tanh(x@w1+b1)@w2+b2

def stable_A(rng):
    a=rng.normal(scale=.22,size=(4,4))
    a-=np.eye(4)*(0.45+0.15*rng.random())
    return a

def states(seed,n):
    return np.random.default_rng(seed).uniform(-1,1,size=(n,4))

def transition(x,A,alpha,resid,guard):
    guard.charge('generated_transitions',len(x)); guard.charge('generated_rhs_evaluations',len(x))
    f=x@A.T+alpha*apply_net(x,resid)
    return x+DT*f

def fit_linear(x,y,guard,ridge=1e-6):
    guard.charge('coefficient_factorizations')
    f=(y-x)/DT
    b=np.linalg.solve(x.T@x+ridge*np.eye(4),x.T@f)
    return b

def residual(x,y,b): return (y-x)/DT-x@b

def norm_rows(r):
    v=r.reshape(len(r),-1); n=np.linalg.norm(v,axis=1,keepdims=True)
    return v/np.maximum(n,1e-12)

def select_and_targets(xs, rs, shuffled, seed, guard):
    # rs: systems x rows x 4.  Selection is label blind and fixed top-half centrality.
    z=[]; perms=[]
    for i,r in enumerate(rs):
        p=np.arange(len(r))
        if shuffled:
            p=np.random.default_rng(seed+1009*i).permutation(len(r))
        perms.append(p); q=r[p].reshape(-1); q=q/max(np.linalg.norm(q),1e-12); z.append(q)
    z=np.stack(z); sim=np.abs(z@z.T); np.fill_diagonal(sim,np.nan)
    central=np.nanmedian(sim,axis=1); selected=np.argsort(-central,kind='stable')[:8]
    guard.charge('svds')
    _,_,vh=np.linalg.svd(z[selected],full_matrices=False)
    template=vh[0]
    signs=np.sign(z@template); signs[signs==0]=1
    return selected, signs, central, perms

def train_module(xtrain, residuals, selected, signs, shuffled, seed, guard):
    torch.manual_seed(seed); np.random.seed(seed%2**32)
    m=Callable().double(); opt=torch.optim.Adam(m.parameters(),lr=.003)
    xx=[]; yy=[]
    for i in selected:
        r=residuals[i].copy(); x=xtrain.copy()
        if shuffled:
            p=np.random.default_rng(seed+2003*i).permutation(len(x)); r=r[p]
        scale=np.sqrt(np.mean(r*r))+1e-12
        xx.append(x); yy.append(signs[i]*r/scale)
    xx=torch.tensor(np.concatenate(xx),dtype=torch.float64)
    yy=torch.tensor(np.concatenate(yy),dtype=torch.float64)
    guard.charge('shared_fits')
    hist=[]
    for step in range(400):
        guard.charge('gradient_updates'); guard.charge('module_training_rows',len(xx))
        opt.zero_grad(); pred=m(xx); loss=((pred-yy)**2).mean(); loss.backward(); opt.step()
        if step in (0,99,199,299,399): hist.append([step+1,float(loss.detach())])
    return m,hist

def module_np(m,x,guard):
    guard.charge('prediction_blocks')
    with torch.no_grad(): return m(torch.tensor(x,dtype=torch.float64)).numpy()

def fit_augmented(x,y,h,guard,ridge=1e-6):
    guard.charge('coefficient_factorizations')
    f=((y-x)/DT).reshape(-1); n=len(x); z=np.zeros((n*4,17))
    for i in range(n):
        for j in range(4):
            z[4*i+j,4*j:4*j+4]=x[i]
            z[4*i+j,16]=h[i,j]
    beta=np.linalg.solve(z.T@z+ridge*np.eye(17),z.T@f)
    return beta

def pred_aug(x,h,beta):
    A=beta[:16].reshape(4,4); a=beta[16]
    return x+DT*(x@A.T+a*h)

def score(yhat,y,guard):
    guard.charge('score_blocks'); return float(np.mean((yhat-y)**2))

def bytes_under(root): return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def write_json(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)); os.replace(tmp,path)

def preflight():
    m=Callable().double(); assert sum(p.numel() for p in m.parameters())==1348
    g=Guard({'coefficient_factorizations':2,'prediction_blocks':1,'score_blocks':1,'svds':1,'shared_fits':1,'gradient_updates':1,'module_training_rows':8,'generated_transitions':1,'generated_rhs_evaluations':1,'wall_seconds':30,'peak_RSS_bytes':1073741824,'artifact_bytes':1000000})
    x=np.arange(32,dtype=float).reshape(8,4)/31; y=x+.01
    b=fit_linear(x,y,g); assert b.shape==(4,4)
    h=np.ones_like(x); beta=fit_augmented(x,y,h,g); assert beta.shape==(17,)
    # cap rejects before mutation
    before=g.n['score_blocks']
    try: g.charge('score_blocks',2); raise AssertionError('guard failed')
    except RuntimeError: pass
    assert g.n['score_blocks']==before
    print(json.dumps({'status':'PREFLIGHT_OK','module_parameters':1348,'linear_shape':list(b.shape),'augmented_shape':list(beta.shape)}))

def main(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    plan=json.load(open(args.plan)); guard=Guard(plan['ceilings'])
    ledger={'status':'SEALED_PRECALL','plan_sha256':sha(args.plan),'driver_sha256':sha(__file__),'ceilings':plan['ceilings'],'counts':guard.n.copy()}
    write_json(out/'ledger.json',ledger)
    rng=np.random.default_rng(WORLD_SEED); shared=gen_net(rng)
    banks={k:states(WORLD_SEED+s,n) for k,s,n in [('base',11,32),('cluster',12,16),('module',13,32),('admission',14,16)]}
    systems=[]
    for i in range(16):
        reuse=i<8; systems.append({'A':stable_A(rng),'alpha':rng.uniform(.6,1.4),'resid':shared if reuse else gen_net(rng),'reuse':reuse})
    data={k:[] for k in banks}
    for s in systems:
        for k,x in banks.items(): data[k].append(transition(x,s['A'],s['alpha'],s['resid'],guard))
    bs=[]; rcluster=[]; rmodule=[]; radmit=[]
    for i in range(16):
        b=fit_linear(banks['base'],data['base'][i],guard); bs.append(b)
        rcluster.append(residual(banks['cluster'],data['cluster'][i],b)); rmodule.append(residual(banks['module'],data['module'][i],b)); radmit.append(residual(banks['admission'],data['admission'][i],b))
    rcluster=np.stack(rcluster); rmodule=np.stack(rmodule); radmit=np.stack(radmit)
    arm_meta={}
    models={}
    for arm,shuf in [('aligned',False),('shuffled',True)]:
        sel,signs,central,perms=select_and_targets(banks['cluster'],rcluster,shuf,WORLD_SEED+(1 if shuf else 0),guard)
        arm_meta[arm]={'selected':[int(i) for i in sel],'centrality':[float(x) for x in central]}
        for seed in SEEDS:
            m,hist=train_module(banks['module'],rmodule,sel,signs,shuf,seed,guard); models[(arm,seed)]=m
            arm_meta[arm][str(seed)]={'history':hist}
    # Admission uses only acquisition data. Fit scalar jointly with linear A on cluster, score admission.
    admissions=[]
    for arm in ['aligned','shuffled']:
        sel=arm_meta[arm]['selected']
        for seed in SEEDS:
            m=models[(arm,seed)]; rows=[]
            for i in sel:
                hc=module_np(m,banks['cluster'],guard); hv=module_np(m,banks['admission'],guard)
                beta=fit_augmented(banks['cluster'],data['cluster'][i],hc,guard)
                ya=pred_aug(banks['admission'],hv,beta); mse=score(ya,data['admission'][i],guard)
                y0=banks['admission']+DT*(banks['admission']@bs[i]); mse0=score(y0,data['admission'][i],guard)
                rows.append({'system':int(i),'module_MSE':mse,'no_module_MSE':mse0})
            ratio=sum(r['module_MSE'] for r in rows)/sum(r['no_module_MSE'] for r in rows)
            wins=sum(r['module_MSE']<r['no_module_MSE'] for r in rows)
            admissions.append({'arm':arm,'seed':seed,'ratio':ratio,'wins':wins,'passed':ratio<=.8 and wins>=6,'rows':rows})
    decision={'status':'ADMISSION_FROZEN_BEFORE_DEVELOPMENT_GENERATION','hidden_reuse_labels_read':False,'arm_metadata':arm_meta,'admissions':admissions,'counts':guard.n.copy()}
    write_json(out/'admission_decision.json',decision)
    aligned_pass=all(x['passed'] for x in admissions if x['arm']=='aligned')
    if not aligned_pass:
        result={'classification':'NOT_SUPPORTED_AT_ACQUISITION_ADMISSION','admissions':admissions,'counts':guard.n,'qualification':plan['scope']['qualification']}
        write_json(out/'result.json',result); write_json(out/'completion.json',{'status':'COMPLETE','result_sha256':sha(out/'result.json'),'counts':guard.n})
        return
    # Development is generated only after the acquisition decision is persisted.
    dev_context=states(WORLD_SEED+101,32); dev_query=states(WORLD_SEED+102,64)
    dev=[]
    for i in range(16):
        reuse=i<8; s={'A':stable_A(rng),'alpha':rng.uniform(.6,1.4),'resid':shared if reuse else gen_net(rng),'reuse':reuse}
        yc=transition(dev_context,s['A'],s['alpha'],s['resid'],guard); yq=transition(dev_query,s['A'],s['alpha'],s['resid'],guard)
        dev.append((s,yc,yq))
    torch.manual_seed(WORLD_SEED+999); random_model=Callable().double()
    predictions={}; score_rows=[]
    for seed in SEEDS:
        for arm in ['aligned','shuffled']:
            m=models[(arm,seed)]
            hc=module_np(m,dev_context,guard); hq=module_np(m,dev_query,guard)
            for i,(s,yc,yq) in enumerate(dev):
                beta=fit_augmented(dev_context,yc,hc,guard); yh=pred_aug(dev_query,hq,beta)
                predictions[f'{arm}_{seed}_{i}']=yh
    hrc=module_np(random_model,dev_context,guard); hrq=module_np(random_model,dev_query,guard)
    for i,(s,yc,yq) in enumerate(dev):
        beta=fit_augmented(dev_context,yc,hrc,guard); predictions[f'random_{i}']=pred_aug(dev_query,hrq,beta)
        b=fit_linear(dev_context,yc,guard); predictions[f'no_module_{i}']=dev_query+DT*(dev_query@b)
    np.savez_compressed(out/'predictions_before_hidden_labels.npz',**predictions)
    write_json(out/'prediction_barrier.json',{'status':'PREDICTIONS_PERSISTED_BEFORE_HIDDEN_GROUP_SCORING','prediction_sha256':sha(out/'predictions_before_hidden_labels.npz'),'counts':guard.n.copy()})
    for seed in SEEDS:
        for i,(s,yc,yq) in enumerate(dev):
            for arm in ['aligned','shuffled']:
                mse=score(predictions[f'{arm}_{seed}_{i}'],yq,guard); score_rows.append({'seed':seed,'system':i,'group':'reuse' if s['reuse'] else 'unrelated','arm':arm,'MSE':mse})
            for arm in ['random','no_module']:
                mse=score(predictions[f'{arm}_{i}'],yq,guard); score_rows.append({'seed':seed,'system':i,'group':'reuse' if s['reuse'] else 'unrelated','arm':arm,'MSE':mse})
    comparisons=[]
    for seed in SEEDS:
        d=[r for r in score_rows if r['seed']==seed]
        def vals(group,arm): return [r['MSE'] for r in d if r['group']==group and r['arm']==arm]
        rr={a:vals('reuse',a) for a in ['aligned','shuffled','random','no_module']}; uu={a:vals('unrelated',a) for a in ['aligned','no_module']}
        def rat(a,b): return sum(a)/sum(b)
        c={'seed':seed,'reuse_aligned_over_no_module':rat(rr['aligned'],rr['no_module']),'reuse_wins_no_module':sum(a<b for a,b in zip(rr['aligned'],rr['no_module'])),'reuse_aligned_over_shuffled':rat(rr['aligned'],rr['shuffled']),'reuse_wins_shuffled':sum(a<b for a,b in zip(rr['aligned'],rr['shuffled'])),'reuse_aligned_over_random':rat(rr['aligned'],rr['random']),'reuse_wins_random':sum(a<b for a,b in zip(rr['aligned'],rr['random'])),'unrelated_aligned_over_no_module':rat(uu['aligned'],uu['no_module'])}
        c['passed']=c['reuse_aligned_over_no_module']<=.8 and c['reuse_wins_no_module']>=6 and c['reuse_aligned_over_shuffled']<=.85 and c['reuse_wins_shuffled']>=6 and c['reuse_aligned_over_random']<=.85 and c['reuse_wins_random']>=6 and c['unrelated_aligned_over_no_module']<=1.1
        comparisons.append(c)
    classification='SUPPORTED_DEVELOPMENT_ONLY' if all(c['passed'] for c in comparisons) else 'NOT_SUPPORTED'
    result={'classification':classification,'admissions':admissions,'comparisons':comparisons,'scores':score_rows,'counts':guard.n,'resources':{'wall_seconds':time.time()-guard.t,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)},'qualification':plan['scope']['qualification']}
    write_json(out/'result.json',result)
    if bytes_under(out)>plan['ceilings']['artifact_bytes']: raise RuntimeError('ARTIFACT_CAP')
    write_json(out/'completion.json',{'status':'COMPLETE','classification':classification,'result_sha256':sha(out/'result.json'),'counts':guard.n})

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',action='store_true'); ap.add_argument('--plan'); ap.add_argument('--out'); a=ap.parse_args()
    if a.preflight: preflight()
    else: main(a)
