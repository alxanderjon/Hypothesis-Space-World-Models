"""Continuous surrogate extension only; all cell machinery is reused unchanged."""
import math,types
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from experiments.sho import dimension_general_edirect as base


def proposal(points, losses, case_id, iteration, traces):
    d=points.shape[1];surface=base.teacher.GaussianLocalResponse(points,losses)
    # Fixed normalized full-domain starts, independent of hidden parameters.
    n=2*d
    starts=qmc.Sobol(d,scramble=True,seed=base.teacher.SEARCH_SEED).random_base2(int(math.ceil(math.log2(n))))[:n]
    starts=np.vstack((points[int(np.argmin(losses))],starts))
    bounds=[(0.,1.)]*d;results=[]
    def surrogate(u):return float(surface.predict(np.asarray(u)[None])[0])
    for i,start in enumerate(starts):
        r=minimize(surrogate,start,method='Powell',bounds=bounds,
                   options=dict(xtol=1e-10,ftol=1e-12,maxiter=100,maxfev=2000))
        if not np.isfinite(r.fun) or not np.isfinite(r.x).all() or np.any((r.x<0)|(r.x>1)):
            raise RuntimeError('Invalid continuous surrogate minimizer')
        results.append(dict(start_id=i,start=start.tolist(),point=r.x.tolist(),prediction=float(r.fun),
                            nfev=int(r.nfev),nit=int(r.nit),success=bool(r.success),message=str(r.message)))
    chosen=min(range(len(results)),key=lambda i:(results[i]['prediction'],i))
    u=np.asarray(results[chosen]['point']);duplicate=bool(np.any(np.all(points==u,axis=1)))
    traces.append(dict(case_id=case_id,iteration=iteration,site_count=len(points),starts=results,
                       chosen_start=chosen,point=u.tolist(),prediction=results[chosen]['prediction'],
                       exact_duplicate_rejected=duplicate))
    return None if duplicate else u


def search(objective,lower,upper,case_id,cap=5000,gate=1e-4):
    traces=[]
    def composition(d):
        api=base.composition(d)
        api['_greedy_proposal']=lambda p,f,c,it:proposal(p,f,c,it,traces)
        return api
    namespace=dict(base.__dict__);namespace['composition']=composition
    runner=types.FunctionType(base.search.__code__,namespace,'search',base.search.__defaults__)
    result=runner(objective,lower,upper,case_id,cap,gate)
    result['continuous_traces']=traces
    return result
