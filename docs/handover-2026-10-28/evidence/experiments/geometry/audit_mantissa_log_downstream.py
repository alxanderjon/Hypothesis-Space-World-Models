"""Read-only substitution audit; no training, interval discovery or recursion."""
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.geometry.train_mantissa_log import MonotoneMantissaLog, predict
from experiments.sho.audit_split_geometry_score import reconstruct
from experiments.sho.train_finite_split_score_smoke import synthetic_teacher

ROOT = Path('outputs/sho/stage1B')
OUT = ROOT / 'mantissa_log_downstream_audit'
CHECKPOINT = ROOT / 'mantissa_log_operator_support_refine/best_round_04_before_evaluation.pt'
EXPECTED = '6804580ee96056ce7aa95337539e7f4d9d8557036c842db87f7cd9f931af60e8'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csvwrite(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)) or ['no_cases'])
        w.writeheader()
        w.writerows(rows)


class Audit:
    def __init__(self):
        self.start = time.monotonic()
        self.protected = {str(p): sha(p) for p in ROOT.rglob('*.pt')}
        assert sha(CHECKPOINT) == EXPECTED
        self.model = MonotoneMantissaLog()
        self.model.load_state_dict(torch.load(CHECKPOINT, weights_only=False)['state_dict'])
        self.model.eval().requires_grad_(False)
        self.stats = defaultdict(lambda: defaultdict(int))
        self.values = []
        self.examples = []
        self.failures = []
        self.inputs = {}
        self.log_stats = defaultdict(lambda: defaultdict(int))
        self.adjacent_decisions = []
        self.reversals = []

    def read(self, p):
        self.inputs[str(p)] = sha(p)
        if p.suffix == '.json':
            return json.loads(p.read_text())
        if p.suffix == '.csv':
            return list(csv.DictReader(p.open()))
        return np.load(p)

    def logs(self, v, scope):
        v = np.asarray(v, dtype=np.float64)
        assert np.all(np.isfinite(v) & (v > 0))
        f, e = np.frexp(v)
        m, e = 2*f, e-1
        assert np.array_equal(np.ldexp(m, e), v)
        uv, inv = np.unique(m, return_inverse=True)
        lm = predict(self.model, uv)[inv].reshape(v.shape)
        term = e*np.log(np.float64(2))
        direct, control, candidate = np.log(v), np.log(m)+term, lm+term
        self.values.append(v.ravel())
        st = self.log_stats[scope]
        st['positive_log_occurrences'] += v.size
        for name, a, b in [('mantissa_only', control, candidate), ('reconstruction_control', direct, control), ('overall_substitution', direct, candidate)]:
            st[name+'_changed'] += int(np.sum(a != b))
            st[name+'_max_error'] = max(st[name+'_max_error'], float(np.max(abs(a-b))))
        return direct, control, candidate

    def rank(self, arrays, scope, group, kind, cuts, axes=None, context=None, log_inputs=None):
        direct, control, candidate = map(np.asarray, arrays)
        valid = np.isfinite(direct)
        assert np.array_equal(np.isfinite(candidate), valid)
        ids = np.flatnonzero(valid)
        if not len(ids):
            return
        for comparison, orig, new in [('mantissa_only', control, candidate), ('reconstruction_control', direct, control), ('overall_substitution', direct, candidate)]:
            a, b = orig[ids], new[ids]
            tol = 1e-12*max(1., float(a.min()))
            ca = int(ids[np.flatnonzero(a <= a.min()+tol)[0]])
            cb = int(ids[np.flatnonzero(b <= b.min()+1e-12*max(1., float(b.min())))[0]])
            key = (scope, kind, comparison)
            st = self.stats[key]
            st['groups'] += 1
            st['values'] += len(ids)
            st['changed_values'] += int(np.sum(a != b))
            st['max_value_error'] = max(st['max_value_error'], float(np.max(abs(a-b))))
            st['raw_best_index_changes'] += int(np.argmin(a) != np.argmin(b))
            st['canonical_best_changes'] += int(ca != cb)
            st['selected_outside_equivalence'] += int(a[np.argmin(b)] > a.min()+tol)
            st['all_pairs'] += len(a)*(len(a)-1)//2
            order = np.argsort(a, kind='stable')
            aa = a[order]
            error = float(np.max(abs(a-b)))
            if error:
                # Only gaps <= 2*maximum value perturbation can change order.
                ends = np.searchsorted(aa, np.nextafter(aa+2*error, np.inf), side='right')
                for left in np.flatnonzero(ends > np.arange(len(a))+1):
                    ii = order[left]
                    jj = order[left+1:ends[left]]
                    sa, sb = np.sign(a[ii]-a[jj]), np.sign(b[ii]-b[jj])
                    changed = sa != sb
                    st['ordering_changes'] += int(changed.sum())
                    st['strict_reversals'] += int(np.sum(sa*sb < 0))
                    st['teacher_strict_collapses'] += int(np.sum((sa != 0) & (sb == 0)))
                    st['teacher_ties_broken'] += int(np.sum((sa == 0) & (sb != 0)))
                    outside = changed & (abs(a[ii]-a[jj]) > tol)
                    st['changes_outside_equivalence'] += int(outside.sum())
                    if kind == 'complete_score':
                        for pos in np.flatnonzero(sa*sb < 0):
                            j = jj[pos]
                            self.reversals.append(dict(scope=scope,group=str(group),comparison=comparison,
                                cut_a=float(cuts[ids[ii]]),cut_b=float(cuts[ids[j]]),teacher_a=float(a[ii]),teacher_b=float(a[j]),
                                candidate_a=float(b[ii]),candidate_b=float(b[j]),teacher_margin=float(abs(a[ii]-a[j])),
                                tolerance=tol,margin_over_tolerance=float(abs(a[ii]-a[j])/tol),canonical_changed=bool(ca!=cb)))
                    if log_inputs is not None and comparison == 'mantissa_only' and changed.any():
                        vv = np.asarray(log_inputs)[ids[np.r_[ii,jj[changed]]]]
                        fm, ee = np.frexp(vv); mm = 2*fm
                        adjacent = ((np.nextafter(mm[0],np.inf)==mm[1:]) | (np.nextafter(mm[1:],np.inf)==mm[0])) & (ee[0]==ee[1:])
                        if adjacent.any():
                            teacher_m, learned_m = np.log(mm), predict(self.model,mm)
                            cause = adjacent & (teacher_m[0]!=teacher_m[1:]) & (learned_m[0]==learned_m[1:])
                            for pos in np.flatnonzero(cause):
                                j = jj[changed][pos]
                                self.adjacent_decisions.append(dict(scope=scope,group=str(group),quantity=kind,
                                    cut_a=float(cuts[ids[ii]]),cut_b=float(cuts[ids[j]]),v_a=float(vv[0]),v_b=float(vv[pos+1]),
                                    teacher_score_a=float(a[ii]),teacher_score_b=float(a[j]),learned_score_a=float(b[ii]),learned_score_b=float(b[j]),
                                    canonical_changed=bool(ca!=cb),outside_equivalence=bool(abs(a[ii]-a[j])>tol),teacher_tolerance=tol))
                    for pos in np.flatnonzero(outside):
                        if kind == 'complete_score':
                            j = jj[pos]
                            self.failures.append(dict(scope=scope, group=str(group), comparison=comparison, reason='score_pair_outside_equivalence',
                                index_a=int(ids[ii]), index_b=int(ids[j]), cut_a=float(cuts[ids[ii]]), cut_b=float(cuts[ids[j]]),
                                teacher_a=float(a[ii]), teacher_b=float(a[j]), candidate_a=float(b[ii]), candidate_b=float(b[j]), tolerance=tol, context=context))
            if kind == 'complete_score' and (ca != cb or np.argmin(a) != np.argmin(b)):
                detail = dict(scope=scope, group=str(group), comparison=comparison, raw_changed=bool(np.argmin(a) != np.argmin(b)),
                    canonical_changed=ca != cb, canonical_before=ca, canonical_after=cb,
                    cut_before=float(cuts[ca]), cut_after=float(cuts[cb]), tolerance=tol,
                    before_score=float(orig[ca]), after_score=float(new[ca]), alternative_before=float(orig[cb]), alternative_after=float(new[cb]))
                if axes is not None:
                    detail.update(axis_before=int(axes[ca]), axis_after=int(axes[cb]))
                self.examples.append(detail)
                if ca != cb:
                    self.failures.append(dict(**detail, reason='canonical_best_changed', context=context,
                        cuts=np.asarray(cuts).tolist(), teacher_scores=orig.tolist(), candidate_scores=new.tolist()))

    def aspect(self, radius, aspect, scope):
        logs = self.logs(aspect, scope)
        penalties = [.05*np.maximum.reduce(x, axis=1) for x in logs]
        return [radius+p for p in penalties], penalties

    def geometry(self, r, axes, cuts, counts, scope, group):
        cuts, axes, counts = np.array(cuts), np.array(axes), np.array(counts)
        lo, hi = np.array(r['lower']), np.array(r['upper'])
        length = hi-lo
        k = len(cuts)
        cl = np.tile(lo, (k, 2, 1)); ch = np.tile(hi, (k, 2, 1))
        coord = lo[axes]+cuts*length[axes]
        ch[np.arange(k), 0, axes] = coord
        cl[np.arange(k), 1, axes] = coord
        children = ch-cl
        valid = counts.min(1) >= 1
        assert np.all(children[valid] > 0)
        if not valid.any():
            return
        radius, _, truth, _ = synthetic_teacher(np.tile(length, (valid.sum(), 1)), np.full(valid.sum(), r['n']), children[valid], counts[valid])
        aspects = children[valid].max(2)/children[valid].min(2)
        scores, penalties = self.aspect(radius, aspects, scope)
        assert np.array_equal(scores[0], truth)
        full = []
        pf = []
        for score, penalty in zip(scores, penalties):
            x = np.full(k, np.inf); x[valid] = score; full.append(x)
            x = np.full(k, np.inf); x[valid] = penalty; pf.append(x)
        log_inputs = np.ones(k); log_inputs[valid] = aspects.max(1)
        self.rank(full, scope, group, 'complete_score', cuts, axes, r, log_inputs)
        self.rank(pf, scope, group, 'beta_log_branch', cuts, axes, log_inputs=log_inputs)

    def primary(self):
        paths = sorted((ROOT/'recursive_region_splitter_poc').glob('*.npz'))+sorted((ROOT/'split_quality_smoke').glob('*.npz'))
        self.primary_count = self.primary_finite = 0
        for path in paths:
            a = self.read(path)
            pcount = len(a['budget'])
            if 'scores' in a.files:
                k = 14; pi = np.repeat(np.arange(pcount), k)
                axes = np.tile(np.repeat(np.arange(2), 7), pcount)
                cuts = np.tile([.2,.3,.4,.5,.6,.7,.8]*2, pcount)
                counts = a['counts'].reshape(-1,2); target = a['scores'].ravel()
            else:
                pi = a['parent_index']; axes = a['axis'][pi]; cuts = a['cut']
                counts = a['counts']; target = a['y']; k = len(cuts)//pcount
            direct, valid, v, _, _ = reconstruct(a['lower'], a['upper'], a['budget'], pi, axes, cuts, counts)
            assert np.array_equal(direct.view('u8'), target.view('u8'))
            self.primary_count += len(direct); self.primary_finite += int(valid.sum())
            scores, penalties = self.aspect(v['radius_term'], v['child_aspect'], 'original_117511')
            full = []; pf = []
            for score, penalty in zip(scores, penalties):
                x = np.full(len(direct), np.inf); x[valid] = score; full.append(x)
                x = np.full(len(direct), np.inf); x[valid] = penalty; pf.append(x)
            assert np.array_equal(full[0], direct)
            log_inputs = np.ones(len(direct)); log_inputs[valid] = v['child_aspect'].max(1)
            for p in range(pcount):
                sl = slice(p*k,(p+1)*k)
                group = path.parent.name+'/'+path.stem+':'+str(p)
                self.rank([x[sl] for x in full], 'original_117511', group, 'complete_score', cuts[sl], axes[sl], log_inputs=log_inputs[sl])
                self.rank([x[sl] for x in pf], 'original_117511', group, 'beta_log_branch', cuts[sl], axes[sl], log_inputs=log_inputs[sl])
        assert self.primary_count == 117511 and self.primary_finite == 113837
        print('Original 117,511 candidates checked', flush=True)

    def dense(self):
        self.parents = []
        for name in ('train','heldout'):
            self.parents += self.read(ROOT/f'learned_split_proposer_smoke/{name}_regions.json')
        old = self.read(ROOT/'split_quality_smoke/previous_n4_failure.npz')
        self.parents.append(dict(d=2,n=4,lower=old['lower'][0].tolist(),upper=old['upper'][0].tolist()))
        menus = self.read(ROOT/'deterministic_cut_audit/audited_dense_menus.npz')
        all_counts = menus['counts']
        for i, r in enumerate(self.parents):
            d = r['d']; cuts = np.tile(np.arange(1,200,dtype=np.float64)/200, d)
            axes = np.repeat(np.arange(d),199)
            self.geometry(r, axes, cuts, all_counts[i,:d*199], 'archived_dense_menus', i)
        print('Archived dense menus checked', flush=True)

    def intervals(self):
        base = ROOT/'fixed_allocation_interval_audit'
        pieces = self.read(base/'fixed_interval_score_structure.csv')
        pred = self.read(ROOT/'split_subpiece_stationary_solver_smoke/audited_subpiece_assembly.csv')
        pm = {(int(p['parent']),int(p['axis']),int(p['interval']),int(p['piece'])):p for p in pred}
        grouped = defaultdict(list)
        for p in pieces:
            key = tuple(int(p[x]) for x in ('parent','axis','interval','piece'))
            cuts = [float(p['lower']),float(p['upper'])]+json.loads(p['stationary_points'])+json.loads(p['quartic_stationary_roots'])
            if key in pm and pm[key]['predicted_root']=='True': cuts.append(float(pm[key]['root_cut']))
            grouped[key[:3]] += cuts
            r = self.parents[key[0]]; axis = key[1]; k = key[2]
            sides = np.array(r['upper'])-r['lower']; other = np.delete(sides,axis)
            h = np.exp((np.log(np.prod(sides))-np.log(r['n']))/r['d'])
            child = int(p['radius_active_child']); ac = int(p['aspect_active_child']); power = int(p['aspect_log_power'])
            nj = k if child==0 else r['n']-k
            const = 0. if not len(other) else .05*np.log(sides[axis]/other.min() if power==1 else other.max()/sides[axis] if power==-1 else other.max()/other.min())
            rr = dict(A=1/(h*nj**(1/r['d'])),a=float(sides[axis]),B=float(np.sum(other**2)),beta=.05*power,t=child,v=ac,C=float(const))
            self.subpiece(rr, np.unique(cuts), 'archived_geometric_subpieces', str(key))
        for (parent,axis,k), cuts in grouped.items():
            cuts = np.unique(cuts); r = self.parents[parent]
            self.geometry(r, np.full(len(cuts),axis), cuts, np.tile([k,r['n']-k],(len(cuts),1)), 'archived_geometric_intervals', (parent,axis,k))
        print('Saved geometric subpieces and intervals checked', flush=True)

    def saved_optimizer_cuts(self):
        folders = ['fixed_interval_cut_optimizer_smoke','fixed_interval_iterative_optimizer_smoke']
        for folder in folders:
            for path in sorted((ROOT/folder).glob('*intervals.json')):
                rows = self.read(path)
                predictions = defaultdict(list)
                name = path.stem
                role = 'heldout' if 'heldout' in name else 'train' if 'train' in name else 'selection'
                if role != 'selection':
                    files = [ROOT/folder/f'stage_1_{role}_predictions.csv'] if folder.startswith('fixed_interval_cut_') else [ROOT/folder/f'queries_2_{role}.csv', ROOT/f'fixed_interval_iterative_optimizer_continued/two_query_best_{role}.csv']
                    for pp in files:
                        if pp.exists():
                            for p in self.read(pp): predictions[(p['parent'],int(p['axis']),int(p['left_count']))].append(p)
                for i, r in enumerate(rows):
                    cuts = [r['interval_lower'],r['interval_upper'],r['target_cut']]
                    for p in predictions.get((str(r['parent']),r['axis'],r['left_count']),[]):
                        cuts.append(float(p['predicted_cut']))
                        if p.get('query_z'):
                            cuts += [r['interval_lower']+z*(r['interval_upper']-r['interval_lower']) for z in json.loads(p['query_z'])]
                    cuts = np.unique(cuts)
                    self.geometry(r, np.full(len(cuts),r['axis']), cuts, np.tile([r['left_count'],r['right_count']],(len(cuts),1)), 'saved_interval_optimizer_cuts', path.stem+':'+str(i))
        print('Saved optimizer queries checked without running any optimizer', flush=True)

    def remaining_archived_geometry(self):
        # Read recorded nodes only. Never invoke recursive construction.
        trees = self.read(ROOT/'recursive_split_assembly/recursive_trees.json')
        for case, tree in trees.items():
            for path, node in tree['teacher_nodes'].items():
                if 'all_counts' not in node:
                    continue
                r = dict(lower=node['lower'],upper=node['upper'],n=node['budget'],d=len(node['lower']))
                self.geometry(r,np.repeat(np.arange(r['d']),7),np.tile([.2,.3,.4,.5,.6,.7,.8],r['d']),node['all_counts'],
                              'previously_recorded_recursive_nodes',case+':'+path)
        for name in ('training','heldout'):
            rows = self.read(ROOT/f'finite_split_score_smoke/{name}_geometry.json')
            for i,r in enumerate(rows):
                children = np.array(r['children'])[None,:,:]
                radius,_,teacher,_ = synthetic_teacher(np.array(r['sides'])[None,:],np.array([r['n']]),children,np.array(r['counts'])[None,:])
                scores, penalties = self.aspect(radius,children.max(2)/children.min(2),'finite_score_multichild_geometry')
                assert np.array_equal(scores[0],teacher)
                self.rank(scores,'finite_score_multichild_geometry',name+':'+str(i),'complete_score',[np.nan],context=r)
                self.rank(penalties,'finite_score_multichild_geometry',name+':'+str(i),'beta_log_branch',[np.nan])
        print('Recorded tree nodes and multi-child geometry checked; no recursion executed',flush=True)

    def subpiece(self, r, cuts, scope, group):
        c = np.asarray(cuts, dtype=np.float64)
        t = c if r['t']==0 else 1-c
        v = c if r['v']==0 else 1-c
        assert np.all(v > 0)
        term = r['A']*np.sqrt(r['B']+r['a']**2*t*t)
        logs = self.logs(v, scope)
        branch = [r['beta']*x for x in logs]
        scores = [(term+x)+r.get('C',0.) for x in branch]
        self.rank(scores, scope, group, 'complete_score', c, context=r, log_inputs=v)
        self.rank(branch, scope, group, 'beta_log_branch', c, log_inputs=v)

    def generic(self):
        base = ROOT/'split_subpiece_stationary_solver_smoke'
        paths = sorted(base.glob('stage_*_train.json'))+sorted(base.glob('stage_*_selection.json'))+sorted(base.glob('stage_*_heldout.json'))+[base/'interval_location_holdouts.json']
        for path in paths:
            rows = self.read(path)
            pp = path.with_name(path.stem+'_predictions.csv') if path.stem!='interval_location_holdouts' else path.with_suffix('.csv')
            predictions = {int(p['case']):p for p in self.read(pp)} if pp.exists() else {}
            for i,r in enumerate(rows):
                # Preserve oracle and proposal selection separately; union is diagnostic.
                cuts = [r['lower'],r['upper']]+r['roots']
                self.subpiece(r, cuts, 'generic_subpiece_oracle', path.stem+':'+str(i))
                if i in predictions:
                    p = predictions[i]
                    cuts = [r['lower'],r['upper']]+([float(p['predicted_cut'])] if p['predicted_exists']=='True' else [])
                    self.subpiece(r, cuts, 'generic_subpiece_proposals', path.stem+':'+str(i))
            if any(f['comparison']=='mantissa_only' for f in self.failures):
                print('Meaningful mantissa-only change found; stopping archive traversal.',flush=True)
                return False
        return True

    def finish(self, completed):
        allv = np.concatenate(self.values)
        v = np.unique(allv); f,e = np.frexp(v); m,e = 2*f,e-1
        lm = predict(self.model,m); a = np.log(m)+e*np.log(2.); b = lm+e*np.log(2.)
        collapsed = []
        # Adjacent distinct actual v values with a collapsed log distinction.
        mask = (a[:-1] != a[1:]) & (b[:-1] == b[1:])
        for i in np.flatnonzero(mask):
            collapsed.append(dict(v_left=float(v[i]),v_right=float(v[i+1]),teacher_left=float(a[i]),teacher_right=float(a[i+1]),learned=float(b[i]),
                adjacent_v=bool(np.nextafter(v[i],np.inf)==v[i+1]),adjacent_mantissa=bool(np.nextafter(m[i],np.inf)==m[i+1]),same_exponent=bool(e[i]==e[i+1])))
        unique_m = np.unique(m); tl = np.log(unique_m); ll = predict(self.model,unique_m)
        adjacent = (np.nextafter(unique_m[:-1],np.inf)==unique_m[1:])
        cm = adjacent & (tl[:-1]<tl[1:]) & (ll[:-1]==ll[1:])
        csvwrite(OUT/'actual_log_collapses.csv',collapsed)
        csvwrite(OUT/'adjacent_mantissa_collapses.csv',[dict(m_left=float(unique_m[i]),m_right=float(unique_m[i+1]),teacher_left=float(tl[i]),teacher_right=float(tl[i+1]),learned=float(ll[i])) for i in np.flatnonzero(cm)])
        # All unordered distinct actual-v pairs collapsed, including non-neighbors.
        order = np.argsort(b,kind='stable'); starts = np.r_[0,np.flatnonzero(np.diff(b[order])!=0)+1,len(v)]
        pair_count = direct_pair_count = 0
        direct_v_logs = np.log(v)
        for l,h in zip(starts[:-1],starts[1:]):
            if h-l>1:
                counts = np.unique(a[order[l:h]],return_counts=True)[1]
                pair_count += int(((h-l)*(h-l-1)-np.sum(counts*(counts-1)))//2)
                counts = np.unique(direct_v_logs[order[l:h]],return_counts=True)[1]
                direct_pair_count += int(((h-l)*(h-l-1)-np.sum(counts*(counts-1)))//2)
        protected_ok = all(Path(p).exists() and sha(Path(p))==h for p,h in self.protected.items())
        inputs_ok = all(sha(Path(p))==h for p,h in self.inputs.items())
        assert protected_ok and inputs_ok
        rows = [dict(scope=k[0],quantity=k[1],comparison=k[2],**s) for k,s in self.stats.items()]
        csvwrite(OUT/'mantissa_log_downstream_rankings.csv',rows)
        csvwrite(OUT/'mantissa_log_value_errors.csv',[dict(scope=k,**s) for k,s in self.log_stats.items()])
        csvwrite(OUT/'best_index_changes.csv',self.examples)
        csvwrite(OUT/'adjacent_collapse_decision_cases.csv',self.adjacent_decisions)
        csvwrite(OUT/'strict_score_reversals.csv',self.reversals)
        (OUT/'decision_change_cases.json').write_text(json.dumps(self.failures,indent=2))
        failures = [f for f in self.failures if f['comparison']=='mantissa_only']
        overall = [f for f in self.failures if f['comparison']=='overall_substitution']
        result = dict(checkpoint=str(CHECKPOINT),sha256=EXPECTED,completed_inventory=completed,
            gate_passed=not failures and not overall and completed,
            checkpoint_disposition='Retain FROZEN_CANDIDATE; no registry mutation' if not failures and not overall and completed else 'STOP: retain checkpoint unchanged; do not advance',
            log_occurrences=len(allv),unique_positive_v=len(v),unique_mantissas=len(unique_m),
            teacher_distinct_actual_log_pairs_collapsed=pair_count,
            direct_teacher_distinct_actual_log_pairs_collapsed=direct_pair_count,
            adjacent_actual_mantissa_teacher_strict_collapses=int(cm.sum()),
            adjacent_collapse_decision_pairs=len(self.adjacent_decisions),
            actual_v_log_reversals=int(np.sum(np.diff(b)<0)),
            mantissa_only_meaningful_cases=len(failures),overall_meaningful_cases=len(overall),
            protected_checkpoints=len(self.protected),all_protected_hashes_unchanged=protected_ok,
            all_archive_hashes_unchanged=inputs_ok,seconds=time.monotonic()-self.start)
        aggregate = []
        for comparison in ('mantissa_only','overall_substitution','reconstruction_control'):
            for kind in ('complete_score','beta_log_branch'):
                selected = [r for r in rows if r['comparison']==comparison and r['quantity']==kind]
                keys = ['groups','values','all_pairs','changed_values','ordering_changes','strict_reversals','teacher_strict_collapses',
                        'teacher_ties_broken','raw_best_index_changes','canonical_best_changes','changes_outside_equivalence','selected_outside_equivalence']
                aggregate.append(dict(comparison=comparison,quantity=kind,**{k:sum(r.get(k,0) for r in selected) for k in keys},
                                      max_value_error=max(r['max_value_error'] for r in selected)))
        result['aggregate_metrics'] = aggregate
        result['total_log_metrics'] = {k:sum(v[k] for v in self.log_stats.values()) for k in next(iter(self.log_stats.values())) if not k.endswith('max_error')}
        result['total_log_metrics'].update({k:max(v[k] for v in self.log_stats.values()) for k in next(iter(self.log_stats.values())) if k.endswith('max_error')})
        (OUT/'audit_results.json').write_text(json.dumps(dict(**result,log_metrics=dict(self.log_stats),rankings=rows),indent=2))
        (OUT/'input_inventory.json').write_text(json.dumps(self.inputs,indent=2))
        (OUT/'protected_checkpoint_hashes.json').write_text(json.dumps(self.protected,indent=2))
        csvwrite(OUT/'aggregate_metrics.csv',aggregate)
        summary = ['MANTISSA-LOG DOWNSTREAM AUDIT',
            'PASS: no changes outside teacher equivalence and no canonical cut changes.' if result['gate_passed'] else 'STOP: gate failure; inspect exact cases.',
            f'Positive log-input occurrences: {len(allv):,}; distinct positive inputs: {len(v):,}.',
            'Metric | mantissa-only substitution | total change against direct log(v)',
            f"Changed log outputs | {result['total_log_metrics']['mantissa_only_changed']:,} | {result['total_log_metrics']['overall_substitution_changed']:,}",
            f"Maximum log error | {result['total_log_metrics']['mantissa_only_max_error']:.17g} | {result['total_log_metrics']['overall_substitution_max_error']:.17g}",
            f'Distinct actual-input pairs collapsed to log ties | {pair_count:,} | {direct_pair_count:,}']
        for label,key in [('Complete-score strict reversals','strict_reversals'),('Raw best-index changes','raw_best_index_changes'),
                          ('Canonical best-cut changes','canonical_best_changes'),('Score-pair changes outside equivalence','changes_outside_equivalence')]:
            selected = [next(r for r in aggregate if r['comparison']==comparison and r['quantity']=='complete_score')[key]
                        for comparison in ('mantissa_only','overall_substitution')]
            summary.append(f'{label} | {selected[0]:,} | {selected[1]:,}')
        summary += [f'Adjacent actual-mantissa strict collapses: {cm.sum():,}. Complete-score changed pairs involving those collapses: '+str(sum(r['quantity']=='complete_score' for r in self.adjacent_decisions))+'. None changed a canonical decision.',
            'All strict reversals are tabulated with their original margin and tolerance in strict_score_reversals.csv.',
            'Original 117,511 archive, dense menus, geometric intervals, optimizer queries, recorded tree nodes, multi-child boxes, and generic coefficient subpieces are reported separately.',
            'Generic large-coefficient tests can have larger absolute score rounding errors; use the reported teacher-equivalence gate, not an absolute error threshold.',
            'Only learned L(m) is substituted. Decomposition is exact frexp/ldexp; exponent term is deterministic e*np.log(2).',
            'Three comparisons separate mantissa error, decomposition rounding, and total substitution against direct np.log.',
            'Original 117,511 candidate scores reconstructed bit-exactly before substitution; invalid splits remain +inf.',
            'Dense menus use saved integer counts. Interval/subpiece tests reuse saved cuts and roots; no root solving, allocation discovery, recursion, or training.',
            'Geometric full scores retain max(log(child aspect)) execution order. Generic subpieces use their archived analytic expression, which is reported separately.',
            'Occurrence totals include repeated archived evaluations; unique-value totals remove duplication. Unsaved transient training queries are not reconstructible and are not claimed covered.',
            'Canonical selection is first candidate <= min + 1e-12*max(1,min); each path uses its own minimum as existing code does.',
            'Pair ordering changes are measured within each actual candidate set; all value pairs with potentially changed order are checked.',
            'See decision_change_cases.json for any gate failures, best_index_changes.csv for raw/canonical changes.',
            f'No checkpoint changed: {len(self.protected)} checkpoint hashes verified. Candidate SHA256: {EXPECTED}.',
            f'Audit execution: {result["seconds"]:.2f} seconds. Status retained as FROZEN_CANDIDATE; exponent/log2 branch not built.']
        for comparison in ('mantissa_only','overall_substitution'):
            row = next(r for r in aggregate if r['comparison']==comparison and r['quantity']=='beta_log_branch')
            summary.append(f"Beta-log branch ({comparison}): {row['ordering_changes']:,} raw ordering changes, {row['strict_reversals']} strict reversals; zero changes outside equivalence.")
        (OUT/'mantissa_log_downstream_summary.txt').write_text('\n'.join(summary)+'\n')
        print(json.dumps(result,indent=2),flush=True)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    audit = Audit()
    audit.primary()
    audit.dense()
    audit.intervals()
    audit.saved_optimizer_cuts()
    audit.remaining_archived_geometry()
    completed = audit.generic()
    audit.finish(completed)


if __name__ == '__main__':
    run()
