"""Small learned Brent control heads and exact compositions of frozen arithmetic.

The instrumented interpreter preserves the audited assignment order. Dynamic
geometry/function evaluation remains a caller-supplied callback.
"""
import inspect
import numpy as np
import torch
from torch import nn
from experiments.geometry import audit_exact_branch_brent as authority
from experiments.geometry.brent_interpolation_acceptance import BrentInterpolationAcceptance
from experiments.geometry.validate_locked_brent_iq import LockedIQInterpolation, INPUTS, GRAPH
from experiments.geometry.train_brent_interpolation_branch_selector import BrentBranchSelector
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold


class CategoricalHead(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, dtype=torch.float64)

    def predict(self, x):
        with torch.no_grad():
            logits = self.linear(torch.as_tensor(np.asarray(x), dtype=torch.float64))
            return (logits[..., 0] > 0).long().numpy() if logits.shape[-1] == 1 else logits.argmax(-1).numpy()


class LearnedBrent:
    def __init__(self):
        self.ops = BrentInterpolationAcceptance()
        self.iq = LockedIQInterpolation()
        meta = self.iq.linear_manifest['branch_selector']
        self.branch = BrentBranchSelector(MagnitudeThreshold())
        self.branch.load_state_dict(torch.load(meta['checkpoint'], weights_only=False)['state_dict'])
        self.branch.eval().requires_grad_(False)
        self.step_head = None
        self.stop_head = None
        self.return_head = None
        self.counts = {}

    def sign(self, x):
        with torch.no_grad():
            return self.ops.threshold.decision(torch.as_tensor(np.asarray(x), dtype=torch.float64)).numpy()

    def and_head(self, a, b):
        a, b = np.broadcast_arrays(a, b)
        with torch.no_grad():
            return self.ops.eligibility.head(torch.from_numpy(np.stack((a, b), -1).astype(float))).argmax(-1).numpy().astype(bool)

    def use_raw(self, scur, delta):
        return self.sign(self.ops.subtract(np.abs(scur), delta)) > 0

    def step(self, xc, sc, sb, delta):
        xc, sc, sb, delta = np.broadcast_arrays(xc, sc, sb, delta)
        features = np.stack((self.use_raw(sc, delta), self.sign(sb) > 0), -1).astype(float)
        action = self.step_head.predict(features)
        # Categorical decoder selects stored float64 operands; no rule override.
        stored = np.stack((sc, delta, -delta), -1)
        applied = np.take_along_axis(stored, action[..., None], -1)[..., 0]
        return self.ops.add(xc, applied), applied, action

    def refresh(self, fp, fc):
        s1, s2 = self.sign(fp), self.sign(fc)
        both_nonzero = self.and_head(s1 != 0, s2 != 0)
        return self.and_head(both_nonzero, np.signbit(fp) != np.signbit(fc))

    def promote(self, fb, fc):
        return self.sign(self.ops.subtract(np.abs(fb), np.abs(fc))) < 0

    def zero(self, x):
        # Reuse the exact equality branch selector; class 0 means equal.
        v = np.asarray(x, dtype=float)
        return (self.branch.predict(np.column_stack((v.ravel(), np.zeros(v.size)))) == 0).reshape(v.shape)

    def stopping(self, fc, sb, delta):
        zero = self.zero(fc)
        width = self.sign(self.ops.subtract(np.abs(sb), delta)) < 0
        z, w = np.broadcast_arrays(zero, width)
        done = self.stop_head.predict(np.stack((z, w), -1).astype(float)).astype(bool)
        return zero, width, done

    def initial_zero(self, fp, fc):
        z1, z2 = bool(self.zero(fp)), bool(self.zero(fc))
        return z1, z2, bool(self.stop_head.predict([[z1, z2]])[0])

    def returned(self, xp, xc, initial_left_zero):
        idx = int(self.return_head.predict([[float(initial_left_zero)]])[0])
        return (xc, xp)[idx]

    def build(self, level):
        """A=1, B=2, C=3, D/E=4; later operations stay teacher until their gate."""
        source = inspect.getsource(authority.traced_brent)
        def scalar(v): return float(np.asarray(v).reshape(-1)[0])
        def count(name): self.counts[name] = self.counts.get(name, 0) + 1
        def eligibility(sp, d, fc, fp):
            count('eligibility');return bool(self.ops.eligibility.predict([[sp,d,fc,fp]])[1][0])
        def branch(xp, xb):
            count('branch');return int(self.branch.predict([[xp,xb]])[0]) == 0
        def linear(fc, fp, xc, xp):
            count('linear');return float(self.iq.linear.predict([[fc,fp,xc,xp]])[0])
        def iq(fc, fp, fb, xc, xp, xb):
            count('IQ')
            # Inference executes frozen primitives only; no teacher intermediate
            # comparisons or substitution occur in this path.
            values={k:np.array([v]) for k,v in zip(INPUTS,(fc,fp,fb,xc,xp,xb))}
            for name, op, lhs, rhs in GRAPH:
                values[name]=self.iq.operation(op,values[lhs],values[rhs] if rhs else None)
            return tuple(float(values[k][0]) for k in ('dpre','dblk','stry'))
        def accept(t, sp, sb, d):
            count('acceptance');return bool(self.ops.evaluate(np.array([t]),np.array([sp]),np.array([sb]),np.array([d]))[0][0])
        def context(xc, xb):
            count('context');v=self.ops.context(np.array([xc]),np.array([xb]));return float(v['delta'][0]),float(v['sbis'][0])
        def step(xc, sc, sb, d):
            count('step');return scalar(self.step(xc,sc,sb,d)[0])
        substitutions=[
            ('delta=(XTOL+RTOL*abs(xc))/2\n        sbis=(xb-xc)/2', 'delta,sbis=context_hook(xc,xb)'),
            ('eligible=previous_step_large and residual_decreasing','eligible=eligibility_hook(sp,delta,fc,fp)'),
            ('if xp==xb:', 'if branch_hook(xp,xb):'),
            ('trial=-fc*(xc-xp)/(fc-fp)', 'trial=linear_hook(fc,fp,xc,xp)'),
            ('dp=(fp-fc)/(xp-xc);db=(fb-fc)/(xb-xc)\n                trial=-fc*(fb*db-fp*dp)/(db*dp*(fb-fp))', 'dp,db,trial=iq_hook(fc,fp,fb,xc,xp,xb)'),
            ('accepted=lhs<limit', 'accepted=acceptance_hook(trial,sp,sbis,delta)'),
            ('fullstep=abs(sc)>delta\n        if fullstep:xc+=sc\n        else:xc+=(delta if sbis>0 else -delta)',
             'fullstep=bool(engine.use_raw(sc,delta))\n        xc=step_hook(xc,sc,sbis,delta)'),
        ]
        if level>=2:
            substitutions.extend([
                ('reset=fp!=0 and fc!=0 and opposite(fp,fc)', 'reset=bool(engine.refresh(fp,fc))'),
                ('sc=xc-xp;sp=sc', 'sc=scalar(engine.ops.subtract(xc,xp));sp=sc'),
                ('promote=abs(fb)<abs(fc)', 'promote=bool(engine.promote(fb,fc))')])
        if level>=3:
            substitutions.extend([
                ("if fp==0:return xp,rows,calls,'INITIAL_LEFT_ZERO'\n    if fc==0:return xc,rows,calls,'INITIAL_RIGHT_ZERO'",
                 "iz_left,iz_right,iz_stop=engine.initial_zero(fp,fc)\n    if iz_stop:\n        if iz_left:return xp,rows,calls,'INITIAL_LEFT_ZERO'\n        return xc,rows,calls,'INITIAL_RIGHT_ZERO'"),
                ('stopzero=fc==0;stopwidth=abs(sbis)<delta', 'stopzero,stopwidth,stopdecision=engine.stopping(fc,sbis,delta)\n        stopzero=bool(stopzero);stopwidth=bool(stopwidth);stopdecision=bool(stopdecision)'),
                ('stop=stopzero or stopwidth,', 'stop=stopdecision,'),
                ('if stopzero or stopwidth:', 'if stopdecision:')])
        if level>=4:
            substitutions.extend([
                ("if iz_left:return xp,rows,calls,'INITIAL_LEFT_ZERO'\n        return xc,rows,calls,'INITIAL_RIGHT_ZERO'",
                 "chosen=engine.returned(xp,xc,iz_left)\n        return chosen,rows,calls,'INITIAL_LEFT_ZERO' if iz_left else 'INITIAL_RIGHT_ZERO'"),
                ("rows.append(row);return xc,rows,calls,row['stop_reason']", "rows.append(row);return engine.returned(xp,xc,False),rows,calls,row['stop_reason']")])
        for old,new in substitutions:
            assert source.count(old)==1, old
            source=source.replace(old,new)
        env=dict(authority.__dict__,engine=self,scalar=scalar,context_hook=context,
                 eligibility_hook=eligibility,branch_hook=branch,linear_hook=linear,iq_hook=iq,
                 acceptance_hook=accept,step_hook=step)
        exec(compile(source,'learned_brent_level_'+str(level),'exec'),env)
        return env['traced_brent'],source
