"""Deterministic normalized residual interface; frozen arithmetic only.

The stored pair denotes sign(x) * 2**scale_exponent * (hi + lo), where the
sum and scaling in this statement are exact real operations, NOT instructions
to materialize a single unscaled float. No rounding comparison/router lives here.
"""
from dataclasses import dataclass
import numpy as np
import torch

from experiments.geometry import train_division_operator as source
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_multiplication_operator import QuadraticArithmetic
from experiments.geometry.train_subtraction_operator import Subtraction


@dataclass(frozen=True)
class ResidualExpansion:
    hi: np.ndarray
    lo: np.ndarray
    scale_exponent: np.ndarray
    numerator_negative: np.ndarray


class CompensatedDivisionResidual:
    """Product-tail composition and exact high/low residual on the audited domain.

    Contract: finite x,y,q, y!=0, quotient sign matches x/y, normalized rounded
    product is zero or within a factor two of mx. The last condition ensures the
    ordinary high subtraction is exact. Out-of-contract calls raise; no fallback.
    Normalization, unary negation, constant, and wiring remain deterministic.
    """
    def __init__(self):
        self.multiply,mm=source.load(QuadraticArithmetic,'multiplication_operator_smoke')
        self.subtract,sm=source.load(Subtraction,'subtraction_operator_training','subtraction_operator_locked.json')
        self.add,am=source.load(Addition,'addition_operator_smoke')
        self.manifests={'multiply':mm,'subtract':sm,'add':am}

    @staticmethod
    def _call(operator,a,b):
        a,b=np.broadcast_arrays(np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64))
        shape=a.shape
        with torch.no_grad():out=operator(torch.from_numpy(np.column_stack((a.ravel(),b.ravel())))).numpy()
        return out.reshape(shape)

    def product_pair(self,a,b):
        m=lambda x,y:self._call(self.multiply,x,y)
        s=lambda x,y:self._call(self.subtract,x,y)
        add=lambda x,y:self._call(self.add,x,y)
        ph=m(a,b)
        ca=m(134217729.,a);ua=s(ca,a);ah=s(ca,ua);al=s(a,ah)
        cb=m(134217729.,b);ub=s(cb,b);bh=s(cb,ub);bl=s(b,bh)
        hh=m(ah,bh);s0=s(hh,ph)
        hl=m(ah,bl);s1=add(s0,hl)
        lh=m(al,bh);s2=add(s1,lh)
        ll=m(al,bl);pl=add(s2,ll)
        return ph,pl

    def construct(self,x,y,q):
        x,y,q=np.broadcast_arrays(np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64),np.asarray(q,dtype=np.float64))
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(q).all() and (y!=0).all()):
            raise ValueError('Finite operands and nonzero denominator required.')
        if np.any(np.signbit(q)!=(np.signbit(x)^np.signbit(y))):
            raise ValueError('Approximate quotient must carry the input-derived quotient sign, including zero.')
        mx,ex=np.frexp(abs(x));my,ey=np.frexp(abs(y))
        with np.errstate(over='ignore',under='ignore'):a=np.ldexp(abs(q),-(ex-ey))
        # Safe product-tail domain contains the audited [0,2] range and permits
        # nearby fresh values while keeping every split product far from extremes.
        if np.any((a!=0)&((a<.25)|(a>4.))) or not np.isfinite(a).all():
            raise ValueError('Normalized quotient is outside the supported product-tail domain.')
        ph,pl=self.product_pair(a,my)
        if np.any((ph!=0)&((ph<mx*.5)|(ph>mx*2.))):
            raise ValueError('High subtraction is outside the exact factor-two domain.')
        hi=self._call(self.subtract,mx,ph)
        # Exact IEEE unary sign flip: preserve -0.0 when pl is +0.0. This is
        # the diagnostic's operation; multiply(-1,pl) is NOT substituted.
        lo=np.negative(pl)
        return ResidualExpansion(hi,lo,ex,np.signbit(x))

    def rounded_normalized(self,expansion):
        """Optional diagnostic-compatible scalar; never replaces the stored pair."""
        return self._call(self.add,expansion.hi,expansion.lo)
