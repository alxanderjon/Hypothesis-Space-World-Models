"""Scoped historical quota visit router; no placement or joint-constraint solver."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from experiments.sho.hierarchical_allocation_poc import conserve
from experiments.sho.train_geometry_allocation_curriculum import GeometryAllocator, feature_tensor, random_partition, record
OUT=Path('outputs/sho/stage1B/integer_quota_realization_screen'); OUT.mkdir(exist_ok=True)
BASE=Path('outputs/sho/stage1B'); rng=np.random.default_rng(29092026)
torch.set_num_threads(1); torch.manual_seed(29092026)
files=list(BASE.rglob('*.pt'))
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if OUT not in p.parents}
allocator=GeometryAllocator(); allocator.load_state_dict(torch.load(BASE/'geometry_allocation_curriculum/stage_4_best.pt',weights_only=False)['state_dict']); allocator.eval()
# Inputs are generic counter comparison states, not dimensions or axis-specific features.
# Integer subtraction/comparison, operand selection, cursor, membership remain scaffolding.
model=torch.nn.Linear(2,3,bias=False,dtype=torch.float64)
initial=model.weight.detach().tolist()
x=torch.tensor([[0.,1.],[1.,1.],[1.,0.]],dtype=torch.float64)
y=torch.tensor([0,1,2]) # SKIP, REQUEST_REMAINDER, REQUEST_FULL_BLOCK
# Historical count traces reconstructed into visit states; no invented saved accept trace.
rows=list(csv.DictReader(open(BASE/'hierarchical_allocation_poc/hierarchical_allocation_counts.csv')))
historical=[]
for i,r in enumerate(rows):
    if r['record_type']=='allocated_split':historical.append((i,np.array([int(r['left']),int(r['right'])]),int(r['level'])))
train_states=[]
for i,quota,level in historical:
    if i%5:
        for n in quota:
            for rem in range(int(n),0,-128):train_states.append([float(rem>0),float(rem<128)])
train_states.extend([[0.,1.]]*len(historical))
a=np.array(train_states); labels=np.where(a[:,0]==0,0,np.where(a[:,1]==1,1,2))
np.savez_compressed(OUT/'training_states.npz',comparison_states=a,teacher_actions=labels,
                    source='reconstructed historical counts with full-batch acceptance; not saved online transitions')
# Balance complete finite states; historical repeats contain no additional numerical function.
opt=torch.optim.Adam(model.parameters(),lr=.08)
for epoch in range(500):
    opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step()
    if epoch>=20 and float(loss)<.02:break
weights=model.weight.detach().numpy()
def actions(rem):
    rem=np.asarray(rem); z=np.stack((rem>0,rem<128),axis=-1).astype(np.float64)
    return np.argmax(z@weights.T,axis=-1)
def requests(rem):
    act=actions(rem)
    # Generic action dispatcher selects stored operands; no teacher min/condition in inference.
    operands=np.stack((np.zeros_like(rem),rem,np.full_like(rem,128)),axis=-1)
    return np.take_along_axis(operands,act[...,None],axis=-1)[...,0]
results={}; breakdown={}; schedules=[]
def test_quota(quota,split,dimension,level,identifier,partial=False):
    quota=np.asarray(quota,dtype=np.int64); budget=int(quota.sum())
    if partial:
        occupancy=np.zeros_like(quota); visits=0; errors=0
        while np.any(occupancy<quota):
            rem=quota-occupancy; got=requests(rem); expected=np.minimum(rem,128)
            errors+=int(np.count_nonzero(got!=expected));visits+=len(rem)
            accepted=np.array([rng.integers(1,int(n)+1) if n else 0 for n in got])
            occupancy+=accepted
            assert np.all(occupancy<=quota)
    else:
        chunks=[np.arange(int(n),0,-128,dtype=np.int64) for n in quota]
        rem=np.concatenate(chunks+[np.array([0],dtype=np.int64)])
        got=requests(rem); errors=int(np.count_nonzero(got!=np.minimum(rem,128))); visits=len(rem)
        starts=np.cumsum([0]+[len(c) for c in chunks]);occupancy=np.array([got[starts[j]:starts[j+1]].sum() for j in range(len(quota))])
    count_error=int(np.max(np.abs(occupancy-quota),initial=0)); conserved=int(occupancy.sum())==budget
    for key,container in [(split,results),(f'{split}/d{dimension}/level{level}',breakdown)]:
        s=container.setdefault(key,dict(sequences=0,visits=0,action_errors=0,count_failures=0,budget_failures=0,max_count_error=0))
        s['sequences']+=1;s['visits']+=visits;s['action_errors']+=errors;s['count_failures']+=int(count_error!=0);s['budget_failures']+=int(not conserved);s['max_count_error']=max(s['max_count_error'],count_error)
    schedules.append(dict(id=identifier,split=split,dimension=dimension,level=level,budget=budget,quotas=quota.tolist(),realized=occupancy.tolist()))
    return occupancy
for i,q,l in historical:test_quota(q,'historical_train' if i%5 else 'historical_heldout',2,l,str(i))
def allocate_records(records):
    prepared=[]
    for r in records:
        r=dict(r)
        for k in ['lower','upper','parent_lower','parent_upper']:r[k]=np.asarray(r[k])
        prepared.append(r)
    f,mask,_=feature_tensor(prepared)
    with torch.no_grad():fractions=allocator(f,mask).numpy()
    return [conserve(r['budget'],v[:len(r['target'])]) for r,v in zip(prepared,fractions)]
z=json.load(open(BASE/'geometry_allocation_curriculum/stage_4_partitions.json'))
for split,records in z.items():
    for start in range(0,len(records),128):
        rr=records[start:start+128]
        for r,q in zip(rr,allocate_records(rr)):test_quota(q,'geometry_'+split,r['dimension'],1,r['name'])
# Fresh consistent hierarchical assignments: arbitrary dimensions, generic region identities.
# Each child allocation is conditioned on its parent's integer budget, preserving ancestors.
coarse_checks=0;coarse_errors=0
for d in (1,2,3):
    for trial in range(20):
        root_budget=int(rng.choice([127,997,8600])); nodes=[(root_budget,'root')]; tree={}; leaves=[]
        for level in range(1,5):
            following=[]
            for n,path in nodes:
                r=record(random_partition(rng,d,2),n,path,d,'fresh_hierarchy')
                q=allocate_records([r])[0];tree[path]=n
                for j,v in enumerate(q):following.append((int(v),path+str(j)))
            nodes=following
            occ=test_quota([n for n,p in nodes],'fresh',d,level,f'd{d}_{trial}_{level}',partial=True)
            for path,n in tree.items():
                actual=sum(int(o) for o,(_,p) in zip(occ,nodes) if p.startswith(path))
                coarse_checks+=1;coarse_errors+=int(actual!=n)
# Original realized coordinate arrays: replay the exact historical controlled projection gate.
original=[]
with np.load(BASE/'hierarchical_allocation_poc/generated_interiors.npz') as pools:
    for name in pools.files:
        points=pools[name];axis={'x':0,'y':1}[name[0]] # historical artifact adapter only, never model logic
        counts=np.bincount(np.minimum((points[:,axis]*4).astype(int),3),minlength=4)
        original.append(dict(name=name,quarter_counts=counts.tolist(),max_count_error=int(np.max(abs(counts-2150))),half_counts=counts.reshape(2,2).sum(1).tolist()))
truth_errors=int(np.count_nonzero(model(x).argmax(1).detach().numpy()!=y.numpy()))
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
passed=truth_errors==0 and coarse_errors==0 and unchanged and all(s['action_errors']==s['count_failures']==s['budget_failures']==0 for s in results.values())
report=dict(status='FROZEN_CANDIDATE' if passed else 'FAILED',scope='disjoint regions and consistent laminar integer quotas only; arbitrary overlapping projection targets unsupported',parameter_count=6,architecture='float64 bias-free 2->3 linear categorical router',initial_weights=initial,weights=weights.tolist(),epochs=epoch+1,truth_table_errors=truth_errors,training_state_count=len(a),training_state_errors=int(np.count_nonzero(np.argmax(a@weights.T,axis=1)!=labels)),results=results,dimension_level_results=breakdown,coarse_retention_checks=coarse_checks,coarse_retention_errors=coarse_errors,original_controlled_projection_replay=original,preexisting_checkpoint_count=len(hashes),preexisting_checkpoints_unchanged=unchanged)
if passed:
    torch.save(dict(state_dict=model.state_dict(),status='FROZEN_CANDIDATE',scope=report['scope'],actions=['SKIP','REQUEST_REMAINDER','REQUEST_FULL_BLOCK'],features=['remaining_quota_positive','remaining_quota_less_than_128']),OUT/'integer_quota_visit_router.pt')
    report['checkpoint_sha256']=hashlib.sha256((OUT/'integer_quota_visit_router.pt').read_bytes()).hexdigest()
(OUT/'results.json').write_text(json.dumps(report,indent=2))
(OUT/'quota_replays.json').write_text(json.dumps(schedules))
(OUT/'preserved_checkpoint_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['dimension_level_results','original_controlled_projection_replay','initial_weights','weights']},indent=2))
