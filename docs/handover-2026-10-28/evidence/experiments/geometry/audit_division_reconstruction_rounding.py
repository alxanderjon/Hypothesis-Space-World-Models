"""Diagnose platform ldexp separately from mantissa quotient rounding."""
import csv
import json
import math
import platform
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np


def run():
    out=Path('outputs/sho/stage1B/division_rounding_diagnosis')
    ref=Path('outputs/sho/stage1B/division_operator_refinement')
    z=np.load(ref/'refinement_inputs.npz');xy=z['inputs'];e=z['exponents']
    teacher=z['teacher'];model=z['before'];f=z['features']
    weights=json.loads((ref/'results.json').read_text())['refined_weights']
    mx=np.frexp(abs(xy[:,0]))[0];my=np.frexp(abs(xy[:,1]))[0]
    native_m=mx/my;model_m=weights[0]*f[:,0]+weights[1]*f[:,1]
    counts={g:Counter() for g in ['Brent','original_synthetic','neighbors','all']};rows=[]
    for i in np.flatnonzero(np.isfinite(teacher)&np.isfinite(model)):
        group='Brent' if i<45582 else 'original_synthetic' if i<119390 else 'neighbors'
        t=abs(float(teacher[i]));q=abs(float(model[i]));m=float(model_m[i]);ex=int(e[i]);scale=F(2)**ex
        correct_m_scale=float(F(m)*scale);correct_native_m_scale=float(F(float(native_m[i]))*scale)
        native_m_ldexp=float(np.ldexp(native_m[i],ex))
        flags=dict(cases=1,current_errors=q!=t,model_ldexp_not_single_round=q!=correct_m_scale,
            native_mantissa_double_round_errors=correct_native_m_scale!=t,
            native_mantissa_ldexp_errors=native_m_ldexp!=t,model_single_scale_round_errors=correct_m_scale!=t)
        for label in [group,'all']:counts[label].update({k:int(v) for k,v in flags.items()})
        if q!=correct_m_scale:
            split=math.ldexp(math.ldexp(m,-1022),ex+1022) if ex<-1022 else math.ldexp(m,ex)
            rows.append(dict(index=int(i),group=group,x=float(xy[i,0]),y=float(xy[i,1]),model_mantissa=m,
                mantissa_hex=m.hex(),exponent=ex,model=q,model_hex=q.hex(),teacher=t,teacher_hex=t.hex(),
                single_rounded_scaled_mantissa=correct_m_scale,single_rounded_hex=correct_m_scale.hex(),
                math_ldexp=math.ldexp(m,ex),math_ldexp_hex=math.ldexp(m,ex).hex(),
                split_scaling=split,split_scaling_matches=split==q))
    with (out/'platform_ldexp_rounding_cases.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    report=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
        counts={k:dict(v) for k,v in counts.items()},
        caveat='Observed arithmetic behavior, not a claim about the internal platform libm implementation; exact-rational power-of-two scaling supplies independent oracle.',
        cases=len(rows),split_scaling_matches=sum(row['split_scaling_matches'] for row in rows))
    (out/'platform_ldexp_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
