#!/usr/bin/env python3
from __future__ import annotations
import hashlib, io, json, os, resource, time
from pathlib import Path
import h5py, numpy as np, requests

OUT=Path('/Users/alexanderrichardson/Documents/Codex/2026-10-14/task-2/data/pdebench_range_subset_v1')
CAP_NET=170<<20;CAP_DISK=32<<20;START=time.monotonic()
FILES=[
 ('advection_beta0.2','source','https://darus.uni-stuttgart.de/api/access/datafile/255671','d47b52c1be4b6bcaa8748a8df2f5fbb8'),
 ('advection_beta1.0','source','https://darus.uni-stuttgart.de/api/access/datafile/255675','1fe41923a4123db55bf4e89bea32e142'),
 ('advection_beta4.0','source','https://darus.uni-stuttgart.de/api/access/datafile/255676','a87b03ff425496360e2732921805f47d'),
 ('burgers_nu0.004','source','https://darus.uni-stuttgart.de/api/access/datafile/268191','435e1fecb8a64a0b4563bb8d09f81c33'),
 ('burgers_nu0.04','source','https://darus.uni-stuttgart.de/api/access/datafile/281362','16f4c7afaf8c16238be157e54c9297c7'),
 ('burgers_nu0.4','source','https://darus.uni-stuttgart.de/api/access/datafile/268192','58327a6107e8f9defb5075677cc42533'),
 ('reacdiff_nu0.5_rho1','development','https://darus.uni-stuttgart.de/api/access/datafile/133177','69a429239778d529cd419ed5888ea835'),
 ('reacdiff_nu1_rho2','development','https://darus.uni-stuttgart.de/api/access/datafile/133183','112c01a76447162bd67c8c1073f58ca'),
 ('reacdiff_nu5_rho10','development','https://darus.uni-stuttgart.de/api/access/datafile/133190','264c80af0e2a6cc1f70e87275e0f6ac4'),
]
net=0;requests_count=0
class Remote(io.RawIOBase):
 def __init__(self,url):
  global net,requests_count
  self.s=requests.Session();r=self.s.get(url,headers={'Range':'bytes=0-0'},timeout=30);r.raise_for_status();requests_count+=1;net+=len(r.content);self.url=r.url;self.size=int(r.headers['Content-Range'].rsplit('/',1)[1]);self.etag=r.headers.get('ETag');self.p=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.p
 def seek(self,o,w=io.SEEK_SET):self.p=o if w==io.SEEK_SET else self.p+o if w==io.SEEK_CUR else self.size+o;return self.p
 def readinto(self,b):
  global net,requests_count
  if self.p>=self.size:return 0
  n=min(len(b),self.size-self.p)
  if net+n>CAP_NET:raise RuntimeError(f'NETWORK_CAP:{net+n}>{CAP_NET}')
  r=self.s.get(self.url,headers={'Range':f'bytes={self.p}-{self.p+n-1}'},timeout=60);r.raise_for_status();d=r.content;requests_count+=1;net+=len(d)
  if len(d)!=n:raise RuntimeError(('RANGE_LENGTH',len(d),n,r.headers.get('Content-Range')))
  b[:n]=d;self.p+=n;return n
def rng_indices(name):
 seed=int.from_bytes(hashlib.sha256(('pdebench-range-subset-v1:'+name).encode()).digest()[:8],'little');a=np.sort(np.random.default_rng(seed).choice(10000,30,replace=False));return seed,a[:20],a[20:25],a[25:]
def atomic_npz(path,**a):
 t=path.with_name('.'+path.name+'.tmp')
 with open(t,'wb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
 os.replace(t,path)
def atomic_json(path,x):
 t=path.with_name('.'+path.name+'.tmp')
 with open(t,'w') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,path)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);manifest=[]
 for name,role,url,md5 in FILES:
  before=net;rf=Remote(url)
  with h5py.File(rf,'r') as h:
   d=h['tensor'];shape=tuple(d.shape);dtype=np.dtype(d.dtype);offset=d.id.get_offset();x=np.asarray(h['x-coordinate']);t=np.asarray(h['t-coordinate']);attrs={k:str(v) for k,v in h.attrs.items()}
  if shape not in ((10000,201,1024),(10000,101,1024)) or dtype!=np.dtype('float32') or offset!=2048:raise RuntimeError((name,shape,dtype,offset))
  seed,tr,dv,res=rng_indices(name);selected=np.r_[tr,dv];row_bytes=shape[1]*shape[2]*4;rows=[]
  for idx in selected:
   rf.seek(offset+int(idx)*row_bytes);raw=rf.read(row_bytes)
   if len(raw)!=row_bytes:raise RuntimeError('SHORT_TRAJECTORY')
   z=np.frombuffer(raw,dtype='<f4').reshape(shape[1],shape[2]);rows.append(z[::2 if shape[1]==201 else 1,::16].copy())
  arr=np.stack(rows)
  if arr.shape!=(25,101,64):raise RuntimeError((name,arr.shape))
  atomic_npz(OUT/f'{name}.npz',field=arr,x=x[::16],t=t[:201:2] if shape[1]==201 else t[:101],train_indices=tr,development_indices=dv)
  manifest.append({'name':name,'role':role,'source_url':url,'resolved_url':rf.url,'remote_file_bytes':rf.size,'etag':rf.etag,'official_full_file_md5_unverified':md5,'root_attrs':attrs,'original_shape':shape,'retained_shape':arr.shape,'selection_seed':seed,'training_indices':tr.tolist(),'development_indices':dv.tolist(),'reserved_not_downloaded_indices':res.tolist(),'network_bytes':net-before})
  if time.monotonic()-START>600:raise RuntimeError('WALL_CAP')
 disk=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
 if disk>CAP_DISK:raise RuntimeError(f'DISK_CAP:{disk}>{CAP_DISK}')
 atomic_json(OUT/'manifest.json',{'status':'COMPLETE','protocol':'pdebench_range_subset_v1','files':manifest,'actual':{'network_bytes':net,'http_range_requests':requests_count,'persisted_bytes_before_manifest':disk,'wall_seconds':time.monotonic()-START,'peak_RSS_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},'limits':{'network_bytes':CAP_NET,'persisted_bytes':CAP_DISK,'wall_seconds':600,'peak_RSS_bytes':1<<30},'confirmation':'five reserved indices per file were selected but not downloaded; diffusion-sorption mechanism entirely unopened'})
 print(json.dumps({'network_bytes':net,'requests':requests_count,'disk_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'files':len(manifest),'wall_seconds':time.monotonic()-START},indent=2))
if __name__=='__main__':main()
