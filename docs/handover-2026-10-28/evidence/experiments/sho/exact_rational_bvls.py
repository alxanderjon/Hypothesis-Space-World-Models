"""Isolated exact rational bounded LS; no network or production integration."""
from fractions import Fraction as F


def ldlt(matrix):
    n=len(matrix); L=[[F(int(i==j)) for j in range(n)] for i in range(n)];D=[]
    for j in range(n):
        pivot=matrix[j][j]-sum((L[j][k]*L[j][k]*D[k] for k in range(j)),F(0))
        if pivot<=0:raise RuntimeError('EXACT_RANK_OR_CONVEXITY_STOP: nonpositive LDL pivot')
        D.append(pivot)
        for i in range(j+1,n):
            L[i][j]=(matrix[i][j]-sum((L[i][k]*L[j][k]*D[k] for k in range(j)),F(0)))/pivot
    return L,D


def solve_ldlt(factors,rhs):
    L,D=factors;n=len(D);y=[]
    for i in range(n):y.append(rhs[i]-sum((L[i][j]*y[j] for j in range(i)),F(0)))
    z=[y[i]/D[i] for i in range(n)];x=[F(0)]*n
    for i in range(n-1,-1,-1):x[i]=z[i]-sum((L[j][i]*x[j] for j in range(i+1,n)),F(0))
    return x


def exact_bvls(A,r,lower,upper,max_iterations=200):
    """All arguments are exact Fractions. Bounds are saved rounded displacements.

    Returns only after exact box/KKT verification. The iteration cap is a
    resource stop, never a convergence tolerance. No fallback/rank truncation.
    """
    n=len(lower);m=len(r)
    if n==0 or len(upper)!=n or len(A)!=m or any(len(row)!=n for row in A):raise ValueError('shape mismatch')
    if any(not isinstance(v,F) for row in A for v in row) or any(not isinstance(v,F) for v in r+lower+upper):raise TypeError('exact Fraction inputs required')
    if max_iterations!=200:raise ValueError('declared resource cap is exactly 200')
    if any(lower[j]>upper[j] or not lower[j]<=0<=upper[j] for j in range(n)):raise ValueError('zero must be feasible')
    G=[[sum((A[k][i]*A[k][j] for k in range(m)),F(0)) for j in range(n)] for i in range(n)]
    c=[sum((A[k][i]*r[k] for k in range(m)),F(0)) for i in range(n)]
    full_factors=ldlt(G) # Exact all-mode full-rank check, no magnitude cutoff.
    x=[F(0)]*n;status=[0]*n
    for j in range(n):
        if lower[j]==upper[j]:status[j]=2
        elif lower[j]==0 and c[j]>=0:status[j]=-1
        elif upper[j]==0 and c[j]<=0:status[j]=1
    seen=set();history=[];cache={tuple(range(n)):full_factors}
    for iteration in range(1,max_iterations+1):
        key=(tuple(status),tuple(x))
        if key in seen:raise RuntimeError('EXACT_ACTIVE_SET_CYCLE')
        seen.add(key);free=[j for j in range(n) if status[j]==0];active=[j for j in range(n) if status[j]!=0]
        trial=[F(0)]*n
        for j in active:trial[j]=upper[j] if status[j]==1 else lower[j]
        if free:
            keyfree=tuple(free)
            if keyfree not in cache:cache[keyfree]=ldlt([[G[i][j] for j in free] for i in free])
            rhs=[-c[i]-sum((G[i][j]*trial[j] for j in active),F(0)) for i in free]
            solution=solve_ldlt(cache[keyfree],rhs)
            for j,v in zip(free,solution):trial[j]=v
            if any(c[i]+sum((G[i][j]*trial[j] for j in range(n)),F(0))!=0 for i in free):raise RuntimeError('EXACT_FACE_RESIDUAL_FAILURE')
        delta=[trial[j]-x[j] for j in range(n)];blockers=[]
        for j in free:
            if trial[j]<lower[j]:blockers.append(((lower[j]-x[j])/delta[j],j,-1))
            elif trial[j]>upper[j]:blockers.append(((upper[j]-x[j])/delta[j],j,1))
        row={'iteration':iteration,'status_before':status.copy(),'free':free,'action':None}
        if blockers:
            alpha=min(v[0] for v in blockers)
            if not 0<=alpha<=1:raise RuntimeError('EXACT_BLOCKING_FRACTION_FAILURE')
            x=[x[j]+alpha*delta[j] for j in range(n)]
            hits=[(j,s) for a,j,s in blockers if a==alpha]
            for j,s in hits:status[j]=s;x[j]=lower[j] if s==-1 else upper[j]
            row.update(action='block',alpha=alpha,blockers=hits)
        else:
            x=trial;g=[c[i]+sum((G[i][j]*x[j] for j in range(n)),F(0)) for i in range(n)]
            violations=[(-g[j],j) for j in active if status[j]==-1 and g[j]<0]+[(g[j],j) for j in active if status[j]==1 and g[j]>0]
            if violations:
                magnitude,j=min(violations,key=lambda q:(-q[0],q[1]));status[j]=0
                row.update(action='release',coordinate=j,violation=magnitude)
            else:row.update(action='return')
        if any(not lower[j]<=x[j]<=upper[j] for j in range(n)):raise RuntimeError('EXACT_FEASIBILITY_FAILURE')
        history.append(row)
        if row['action']=='return':
            # Independent residual route: do not reuse G or c for certificate.
            residual=[r[k]+sum((A[k][j]*x[j] for j in range(n)),F(0)) for k in range(m)]
            g=[sum((A[k][j]*residual[k] for k in range(m)),F(0)) for j in range(n)]
            valid=[]
            for j in range(n):
                valid.append(True if lower[j]==upper[j] else g[j]>=0 if x[j]==lower[j] else g[j]<=0 if x[j]==upper[j] else g[j]==0)
            if not all(valid):raise RuntimeError('EXACT_KKT_CERTIFICATE_FAILURE')
            return {'direction':x,'active':status,'gradient':g,'half_sum':sum((v*v for v in residual),F(0))/2,
                    'iterations':iteration,'history':history,'full_LDL_pivots':full_factors[1],
                    'exact_feasible':True,'exact_KKT':True,'full_rank_exact':True,'face_factorizations':len(cache)}
    raise RuntimeError('RESOURCE_LIMIT_200_EXACT_ACTIVE_SET_ITERATIONS')
