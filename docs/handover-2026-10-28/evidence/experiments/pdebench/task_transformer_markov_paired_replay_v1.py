#!/usr/bin/env python3
import copy,json,os,resource,time,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent))
from task_transformer_markov_v1 import Model,load,batch,loss,evaluate,DATA,SRC,TGT,SEEDS

torch.set_num_threads(1);START=time.monotonic();OUT=DATA/'task_transformer_markov_paired_replay_v1';OLD=DATA/'task_transformer_markov_v1'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();src=load(SRC);tgt=load(TGT);train=np.concatenate([a[:20] for _,a,_,_ in src]);mean=float(train.mean());std=float(train.std());dstd=float(np.diff(train,axis=1).std())
 src_train=[(n,a[:20],x,t) for n,a,x,t in src];src_dev=[(n,a[20:25],x,t) for n,a,x,t in src];tgt_adapt=[(n,a[:10],x,t) for n,a,x,t in tgt];tgt_eval=[(n,a[10:25],x,t) for n,a,x,t in tgt]
 old=json.load(open(OLD/'result.json'));newscores=[];logs=[]
 for seed in SEEDS:
  frozen=Model();frozen.load_state_dict(torch.load(OLD/f'frozen_seed{seed}.pt',weights_only=True));replay=copy.deepcopy(frozen);noreplay=copy.deepcopy(frozen)
  orp=torch.optim.AdamW(replay.parameters(),lr=3e-4);onr=torch.optim.AdamW(noreplay.parameters(),lr=3e-4);rng=np.random.default_rng(seed+404);hist=[]
  for u in range(100):
   tb=batch(tgt_adapt,rng,2,mean,std,dstd);sb=batch(src_train,rng,2,mean,std,dstd)
   ltr=loss(replay,tb);lsr=loss(replay,sb);zr=ltr+.5*lsr;orp.zero_grad();zr.backward();orp.step()
   ltn=loss(noreplay,tb);lsn=loss(noreplay,sb);zn=ltn+0*lsn;onr.zero_grad();zn.backward();onr.step()
   if u in (0,99):hist.append({'update':u+1,'replay_target':float(ltr),'replay_source':float(lsr),'replay_objective':float(zr),'no_replay_target':float(ltn),'no_replay_source':float(lsn),'no_replay_objective':float(zn)})
  logs.append({'seed':seed,'history':hist})
  for arm,m in (('replay',replay),('no_replay',noreplay)):
   torch.save(m.state_dict(),OUT/f'{arm}_seed{seed}.pt')
   for r in evaluate(m,src_dev,mean,std,dstd):newscores.append({'seed':seed,'arm':arm,'role':'source','mode':'intact',**r})
   for mode in ('intact','history_shuffle','current_shuffle'):
    for r in evaluate(m,tgt_eval,mean,std,dstd,mode):newscores.append({'seed':seed,'arm':arm,'role':'target','mode':mode,**r})
 comparisons=[]
 for seed in SEEDS:
  def new(a,r,m='intact'):return np.array([x['mse'] for x in newscores if x['seed']==seed and x['arm']==a and x['role']==r and x['mode']==m])
  def prior(a,r,m='intact'):return np.array([x['mse'] for x in old['scores'] if x['seed']==seed and x['arm']==a and x['role']==r and x['mode']==m])
  q=new('replay','target');nr=new('no_replay','target');ret=new('replay','source');oldsrc=prior('frozen','source');f=prior('frozen','target');s=prior('scratch','target');hs=new('replay','target','history_shuffle');cs=new('replay','target','current_shuffle')
  d={'seed':seed,'replay_over_paired_no_replay':float(q.sum()/nr.sum()),'replay_wins_paired_no_replay':int(np.sum(q<nr)),'replay_over_frozen':float(q.sum()/f.sum()),'replay_over_saved_scratch':float(q.sum()/s.sum()),'source_after_over_frozen':float(ret.sum()/oldsrc.sum()),'source_blocks_within_10pct':int(np.sum(ret<=1.1*oldsrc)),'source_blocks':len(ret),'history_shuffle_over_intact':float(hs.sum()/q.sum()),'current_shuffle_over_intact':float(cs.sum()/q.sum())}
  d['passed']=bool(d['replay_over_paired_no_replay']<=.8 and d['source_after_over_frozen']<=1.1 and d['source_blocks_within_10pct']/len(ret)>=.8 and d['history_shuffle_over_intact']>=1.05 and d['current_shuffle_over_intact']>=1.1);comparisons.append(d)
 result={'status':'COMPLETE','classification':'SUPPORTED_DEVELOPMENT_ONLY' if all(x['passed'] for x in comparisons) else 'NOT_SUPPORTED','comparisons':comparisons,'logs':logs,'scores':newscores,'counts':{'fits':4,'optimizer_updates':400,'unique_sampled_windows':1600,'model_window_forwards':3200,'new_evaluation_blocks':len(newscores),'reused_old_score_blocks':len(old['scores'])},'resources':{'wall_seconds':time.monotonic()-START,'peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},'qualifications':['Exactly paired target and source batches isolate inclusion of source replay loss.','Old frozen and scratch scores are reused; their models are not re-evaluated.','Opened development data only; no new responses or confirmation.','Replay stores real source examples and adaptation updates all shared weights.']}
 p=OUT/'result.json';tmp=p.with_name('.result.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,p);print(json.dumps({k:result[k] for k in ('classification','comparisons','counts','resources')},indent=2))
if __name__=='__main__':main()
