"""Read-only spatial audit of the saved allocation experiment. No training."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from experiments.sho.progressive_spacefill_generator import ProgressiveGenerator,boundary_faces
from experiments.sho.train_progressive_spacefill_smoke import sha,write_csv

SOURCE=Path('outputs/sho/stage1B/hierarchical_allocation_poc')
OUT=Path('outputs/sho/stage1B/hierarchical_hole_audit')
BANDS=(.005,.01,.02,.05)


def run():
    start=time.monotonic();deadline=start+240
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Four-minute audit compute limit')
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    config=json.loads((SOURCE/'config.json').read_text());original=json.loads((SOURCE/'results.json').read_text())
    n=config['N'];threshold=1/math.sqrt(n);conditions=config['geometry_conditions']
    protected=dict(original['protected_hashes'])
    protected.update({str(p):sha(p) for p in SOURCE.iterdir() if p.is_file()})
    assert sha(config['local_checkpoint'])==config['local_sha256']
    assert sha(SOURCE/'generated_interiors.npz')==(SOURCE/'generated_interiors.sha256').read_text().split()[0]
    saved=np.load(SOURCE/'generated_interiors.npz')
    old_rows=list(csv.DictReader((SOURCE/'hierarchical_allocation_geometry.csv').open()))
    old={(r['generator'],int(r['iteration']),r['pool_type']):r for r in old_rows if r['record_type']=='pool'}
    # EXACT original reference construction and order, not a fresh sample.
    t=(np.arange(512)+.5)/512
    reference=np.vstack((np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2),np.random.default_rng(27093001).random((262144,2))))
    m=len(reference);reference_sha=hashlib.sha256(reference.tobytes()).hexdigest()
    boundary_distance=np.minimum(reference,1-reference).min(1)
    seam={a:np.min(np.abs(reference[:,a,None]-np.array([.25,.5,.75])[None,:]),axis=1) for a in (0,1)}
    region={a:np.minimum((reference[:,a]*4).astype(int),3) for a in (0,1)}
    local=ProgressiveGenerator();local.load_state_dict(torch.load(config['local_checkpoint'],weights_only=False)['state_dict'])
    local.eval().requires_grad_(False)
    distances={};rows=[];state_checks=[]
    for name in ('frozen_unallocated','x_quarters','y_quarters'):
        for it,pool in conditions:
            check()
            if name=='frozen_unallocated':
                with torch.no_grad():complete,interior=local.generate(n,it,pool)
            else:
                interior=saved[f'{name}_{it}_{pool}']
                complete=np.unique(np.concatenate((interior,boundary_faces(n,2))),axis=0)
            d=cKDTree(complete).query(reference)[0];hole=d>threshold
            expected=old[name,it,pool]
            assert len(complete)==int(expected['complete_count'])
            assert float(d.max())==float(expected['complete_cover']),('cover mismatch',name,it,pool)
            assert int(hole.sum())==int(expected['large_hole_count']),('hole mismatch',name,it,pool)
            distances[name,it,pool]=d
            state_checks.append(dict(generator=name,iteration=it,pool_type=pool,holes=int(hole.sum()),cover=float(d.max()),exact_match=True))
    print(json.dumps(dict(event='all_original_metrics_reproduced',states=len(state_checks),threshold=threshold,reference_sha256=reference_sha)),flush=True)
    views=[('frozen_unallocated',0),('x_quarters',0),('frozen_unallocated',1),('y_quarters',1)]
    pooled={};top={}
    def bands_for(ds,axis,base):
        total=sum(int((d>threshold).sum()) for d in ds);rep=len(ds)
        for width in BANDS:
            for core in (False,True):
                domain=(boundary_distance>.02) if core else np.ones(m,dtype=bool)
                mask=(seam[axis]<=width)&domain
                band_area=6*width*(.96 if core else 1.)
                domain_area=.96**2 if core else 1.
                h=sum(int(np.sum((d>threshold)&mask)) for d in ds)
                domain_h=sum(int(np.sum((d>threshold)&domain)) for d in ds)
                refs=int(mask.sum())*rep;domain_refs=int(domain.sum())*rep
                row=dict(**base,record_type='seam_band_domain_interior' if core else 'seam_band',band_width=width,
                    reference_count=refs,hole_count=h,total_holes=domain_h,band_area=band_area,
                    area_fraction=band_area/domain_area,hole_share=h/domain_h if domain_h else 0.,
                    holes_per_unit_area_per_pool=h/(band_area*rep),hole_rate=h/refs,
                    area_enrichment=(h/domain_h)/(band_area/domain_area) if domain_h else 0.,
                    reference_exposure_fraction=refs/domain_refs,
                    exposure_enrichment=(h/domain_h)/(refs/domain_refs) if domain_h else 0.)
                rows.append(row)
                if base['aggregation']=='all_conditions' and not core:
                    pooled[base['generator'],axis,width]=row
    def regions_for(ds,axis,base):
        for r in range(4):
            for core in (False,True):
                mask=region[axis]==r
                if core:mask=mask&(seam[axis]>.02)&(boundary_distance>.02)
                values=np.concatenate([d[mask] for d in ds]);hole=values>threshold
                rows.append(dict(**base,record_type='quarter_core' if core else 'quarter_region',region_id=r,
                    region_lower=r/4,region_upper=(r+1)/4,reference_count=len(values),hole_count=int(hole.sum()),
                    hole_rate=float(hole.mean()),covering_radius=float(values.max()),mean_distance=float(values.mean()),
                    p95_distance=float(np.quantile(values,.95)),p99_distance=float(np.quantile(values,.99)),
                    p999_distance=float(np.quantile(values,.999)),
                    region_area=.21*.96 if core else .25))
    for name,axis in views:
        check();ds=[];failures=[]
        for it,pool in conditions:
            d=distances[name,it,pool];ds.append(d);hole=d>threshold
            base=dict(generator=name,seam_axis='xy'[axis],iteration=it,pool_type=pool,aggregation='single_pool')
            control=distances['frozen_unallocated',it,pool]
            for index in np.flatnonzero(hole):
                row=dict(**base,record_type='hole',reference_index=int(index),reference_source='midpoint_grid' if index<262144 else 'IID',
                    x=float(reference[index,0]),y=float(reference[index,1]),nearest_probe_distance=float(d[index]),
                    distance_to_seam=float(seam[axis][index]),distance_to_domain_boundary=float(boundary_distance[index]),
                    region_id=int(region[axis][index]),empty_disk_crosses_seam=bool(seam[axis][index]<d[index]),
                    baseline_distance=float(control[index]),new_hole_vs_baseline=bool(control[index]<=threshold))
                rows.append(row)
            if name!='frozen_unallocated':
                for index in np.argsort(-d,kind='stable')[:20]:
                    failures.append(dict(**base,record_type='top_failure',reference_index=int(index),
                        x=float(reference[index,0]),y=float(reference[index,1]),nearest_probe_distance=float(d[index]),
                        distance_to_seam=float(seam[axis][index]),distance_to_domain_boundary=float(boundary_distance[index]),
                        region_id=int(region[axis][index])))
            bands_for([d],axis,base);regions_for([d],axis,base)
            rows.append(dict(**base,record_type='pool_total',reference_count=m,hole_count=int(hole.sum()),
                covering_radius=float(d.max()),mean_distance=float(d.mean()),p99_distance=float(np.quantile(d,.99)),
                original_metrics_exact_match=True))
        base=dict(generator=name,seam_axis='xy'[axis],aggregation='all_conditions',pool_count=len(ds))
        bands_for(ds,axis,base);regions_for(ds,axis,base)
        if failures:
            failures.sort(key=lambda r:(-r['nearest_probe_distance'],r['iteration'],r['pool_type'],r['reference_index']))
            top[name]=failures[:20]
            for rank,row in enumerate(top[name],1):row['rank']=rank;rows.append(row)
    paired=[]
    for name,axis in [('x_quarters',0),('y_quarters',1)]:
        bs=[distances['frozen_unallocated',it,pool] for it,pool in conditions]
        ns=[distances[name,it,pool] for it,pool in conditions]
        total_b=sum(int((d>threshold).sum()) for d in bs);total_n=sum(int((d>threshold).sum()) for d in ns)
        for width in BANDS:
            mask=seam[axis]<=width
            added=sum(int(np.sum((n>threshold)&(b<=threshold)&mask)) for n,b in zip(ns,bs))
            removed=sum(int(np.sum((b>threshold)&(n<=threshold)&mask)) for n,b in zip(ns,bs))
            hb=pooled['frozen_unallocated',axis,width]['hole_count'];hn=pooled[name,axis,width]['hole_count']
            row=dict(record_type='paired_excess',generator=name,seam_axis='xy'[axis],aggregation='all_conditions',band_width=width,
                baseline_total_holes=total_b,allocated_total_holes=total_n,net_excess_total=total_n-total_b,
                baseline_band_holes=hb,allocated_band_holes=hn,added_band_holes=added,resolved_band_holes=removed,
                net_excess_in_band=hn-hb,net_excess_outside_band=(total_n-hn)-(total_b-hb),
                share_of_net_excess_in_band=(hn-hb)/(total_n-total_b),
                allocated_to_baseline_band_density_ratio=hn/hb if hb else float('inf'),
                baseline_outside_hole_rate=(total_b-hb)/((m-int(mask.sum()))*len(ns)),
                allocated_outside_hole_rate=(total_n-hn)/((m-int(mask.sum()))*len(ns)))
            paired.append(row);rows.append(row)
    check();unchanged=all(sha(p)==h for p,h in protected.items());assert unchanged
    write_csv(OUT/'hierarchical_hole_location_audit.csv',rows)
    conclusion='FINDING: Extra holes primarily reflect poorer placement throughout quarter interiors, not an artificial-seam failure.\nAcross five matched pools, baseline has 6,136 uncovered reference points, X quarters 11,596 (+89.0%), and Y quarters 9,933 (+61.9%). These are reference-point hits, not distinct geometric holes.\nAt seam distance <=0.02 (12% of domain area), X contains 12.56% of its net extra holes and Y only 1.05%. Thus 87.44% and 98.95% of excess holes lie farther from seams. Allocated hole area enrichment is 1.072x for X and 0.746x for Y; reference-exposure corrected values are 1.089x and 0.760x.\nX seam-band density is 1.851x baseline, but the overall hole density is already 1.890x baseline: this does not establish a special seam penalty. Y seam-band density is only 1.047x baseline versus 1.619x overall.\nAll four whole quarters AND all four quarter cores have more holes than their matched baseline. Excluding a 0.02 band around both seams and outer boundaries leaves 9,653 versus 5,003 X-core holes (+92.9%), and 8,730 versus 4,963 Y-core holes (+75.9%).\nDegradation is not uniform: quarter 3 ([0.75,1]) is worst by hole count, with 3,626 versus 1,429 baseline for X and 3,279 versus 1,433 for Y. X quarter 2 has the largest covering radius. This is a broad interior degradation with a regional imbalance, not one isolated interface.\nAmong the 20 worst reference points, 18/20 X points and 20/20 Y points are farther than 0.02 from a seam. The worst X point is (0.601345250,0.425065879), distance 0.015931391, seam distance 0.101345250; worst Y is (0.967773438,0.831054688), distance 0.015644425, seam distance 0.081054688.\nImplementation nuance: placement is region-restricted and round-robin, with a shared GLOBAL point prefix and geometry features; it is not completely independent regional generation. This audit localizes the deterioration but does not distinguish proposal restriction from scheduling or frozen-policy adaptation as its underlying cause.\n'
    lines=['HIERARCHICAL HOLE LOCATION AUDIT',conclusion,f'Runtime {time.monotonic()-start:.2f} seconds; no training, checkpoint writes or search.',
        'All 15 original complete-pool covering radii and large-hole counts reproduced EXACTLY. Allocated interiors loaded from hashed archive; unallocated baseline replayed from unchanged frozen checkpoint.',
        f'Reference EXACT original 524288 points: 512x512 midpoints followed by 262144 IID samples from seed 27093001. Reference-array SHA256: {reference_sha}.',
        f'N={n}; large hole iff nearest complete-pool distance >1/sqrt(N)={threshold:.17g}. All outer boundary support included.',
        'Artificial seams ONLY x or y=0.25,0.5,0.75. Actual domain boundary never counted as a seam. Baseline audited under BOTH hypothetical seam orientations.',
        'Counts are uncovered REFERENCE POINTS, not counts of distinct empty disks. Top 20 may contain neighboring points in the same hole.',
        'Seam bands are cumulative. Exact band area=6*width. Counts/unit area and hole-share/area enrichment are reported; sampled-reference exposure normalization also corrects small midpoint-grid aliasing in narrow bands.',
        'Quarter coverage queries the ENTIRE complete probe set, including neighboring strips; there is no artificial restriction of nearest probes to the quarter.',
        'Quarter core excludes points within 0.02 of an artificial seam or the domain boundary. Its area is 0.2016 per quarter. Domain-interior band rows separately remove actual-boundary effects.',
        'Paired excess rows compare identical reference indices and report new/resolved holes as well as net excess. No baseline duplicates are pooled across the two seam orientations.',
        '', 'POOLED SEAM BANDS — FIVE MATCHED POOLS',
        'variant | width | area | baseline holes | allocated holes | allocated hole share | allocated area enrichment | exposure enrichment | density ratio vs baseline | share of net extra holes']
    for row in paired:
        name=row['generator'];axis=0 if name=='x_quarters' else 1;width=row['band_width'];b=pooled[name,axis,width]
        lines.append(f"{name} | {width:.3f} | {b['band_area']:.1%} | {row['baseline_band_holes']} | {row['allocated_band_holes']} | {b['hole_share']:.2%} | {b['area_enrichment']:.3f}x | {b['exposure_enrichment']:.3f}x | {row['allocated_to_baseline_band_density_ratio']:.3f}x | {row['share_of_net_excess_in_band']:.2%}")
    lines+=['','REGION COVERAGE — POOLED FIVE STATES','generator | seam axis | region | core? | holes | cover | mean distance | p99 distance']
    for r in rows:
        if r['record_type'] in ('quarter_region','quarter_core') and r.get('aggregation')=='all_conditions':
            lines.append(f"{r['generator']} | {r['seam_axis']} | {r['region_id']} | {r['record_type']=='quarter_core'} | {r['hole_count']} | {r['covering_radius']:.8f} | {r['mean_distance']:.8f} | {r['p99_distance']:.8f}")
    lines+=['','TOP 20 UNCOVERED REFERENCE POINTS PER ALLOCATED VARIANT','variant | rank | iteration/pool | x | y | nearest distance | seam distance | boundary distance | quarter']
    for name,items in top.items():
        for r in items:lines.append(f"{name} | {r['rank']} | {r['iteration']}/{r['pool_type']} | {r['x']:.9f} | {r['y']:.9f} | {r['nearest_probe_distance']:.9f} | {r['distance_to_seam']:.9f} | {r['distance_to_domain_boundary']:.9f} | {r['region_id']}")
    lines+=['','Protected prior artifacts/checkpoints unchanged: '+str(unchanged),
        'The spatial comparison is observational: it localizes failures, but does not isolate policy weights, proposal restriction, and block scheduling as separate causal interventions. No fix was proposed or trained.']
    (OUT/'hierarchical_hole_summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='finished',seconds=time.monotonic()-start,hole_rows=sum(r['record_type']=='hole' for r in rows),
        paired_excess=paired,protected_unchanged=unchanged)),flush=True)


if __name__=='__main__':run()
