"""Frozen acceptance composition only. No solver mutation or new parameters."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.geometry import train_division_operator as source
from experiments.geometry.validate_locked_brent_linear import LockedLinearInterpolation
from experiments.geometry.train_brent_interpolation_eligibility import BrentEligibility
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold

ROOT=Path('outputs/sho/stage1B')


class BrentInterpolationAcceptance:
    """Existing strict comparisons -> original frozen two-input AND head."""
    def __init__(self):
        self.foundation=LockedLinearInterpolation()
        self.arithmetic=self.foundation.arithmetic
        self.division=self.foundation.division
        self.threshold=self.division.threshold
        self.head_manifest=json.loads((ROOT/'brent_interpolation_eligibility_training/brent_interpolation_eligibility_locked.json').read_text())
        m=self.head_manifest
        assert m['status']=='LOCKED' and source.sha(Path(m['checkpoint']))==m['sha256']
        self.eligibility=BrentEligibility(MagnitudeThreshold())
        self.eligibility.load_state_dict(torch.load(m['checkpoint'],weights_only=False)['state_dict'])
        self.eligibility.eval().requires_grad_(False)

    def multiply(self,a,b):return self.arithmetic._call(self.arithmetic.multiply,a,b)
    def subtract(self,a,b):return self.arithmetic._call(self.arithmetic.subtract,a,b)
    def add(self,a,b):return self.arithmetic._call(self.arithmetic.add,a,b)
    def divide(self,a,b):
        a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
        return self.division.predict(np.column_stack((a.ravel(),b.ravel())))[0].reshape(a.shape)

    def context(self,xcur,xblk,xtol=5e-324,rtol=1e-15):
        abs_xcur=np.abs(xcur)
        relative_tolerance=self.multiply(rtol,abs_xcur)
        tolerance_sum=self.add(xtol,relative_tolerance)
        delta=self.divide(tolerance_sum,2.)
        bracket_difference=self.subtract(xblk,xcur)
        sbis=self.divide(bracket_difference,2.)
        return dict(abs_xcur=abs_xcur,relative_tolerance=relative_tolerance,tolerance_sum=tolerance_sum,
                    delta=delta,bracket_difference=bracket_difference,sbis=sbis)

    def compare_limits(self,lhs,lim1,lim2):
        margin1=self.subtract(lhs,lim1);margin2=self.subtract(lhs,lim2)
        with torch.no_grad():
            c1=self.threshold.decision(torch.from_numpy(np.asarray(margin1)))<0
            c2=self.threshold.decision(torch.from_numpy(np.asarray(margin2)))<0
            features=torch.stack((c1,c2),-1).double()
            accepted=self.eligibility.head(features).argmax(-1).numpy().astype(bool)
        return accepted,dict(lhs_lim1_margin=margin1,lhs_lim2_margin=margin2,lhs_lt_lim1=c1.numpy(),lhs_lt_lim2=c2.numpy())

    def evaluate(self,stry,spre,sbis,delta):
        abs_stry=np.abs(stry);lhs=self.multiply(2.,abs_stry)
        lim1=np.abs(spre);abs_sbis=np.abs(sbis)
        three_times_abs_sbis=self.multiply(3.,abs_sbis)
        lim2=self.subtract(three_times_abs_sbis,delta)
        accepted,comparisons=self.compare_limits(lhs,lim1,lim2)
        return accepted,dict(abs_stry=abs_stry,lhs=lhs,lim1=lim1,abs_sbis=abs_sbis,
            three_times_abs_sbis=three_times_abs_sbis,lim2=lim2,**comparisons)
