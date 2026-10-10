"""Finite-d table of paper/main.tex (Section "Finite d"): is the (b,j) block maximum attained on the sampled points
b in {k-1,k}, j in {1,m/2,m-1}?  Float64 vectorised full scan over b and j at the reported optimizers for
d = 1e4 and 1e6; D is the worst test dimension d-2J from the sampled run.  "diff" = full - sampled ln G_blk.
usage: python3 finite_d_fullscan.py"""
import numpy as np
from scipy.special import gammaln
def lb(a,b): return gammaln(a)+gammaln(b)-gammaln(a+b)
def dirlog(D,b,j,l,delta):
    Dp,n=D-1,D-b; a=(n-j)/2; a_s=(Dp-j)/2; b0=j/2; b1=(Dp-n+j-1)/2
    lA=lb(a_s,b0)-lb(a,b0)-(b-1)/2*np.log(1-delta)
    lB=lb(a,b1)-lb(a,b0)-(b-2)/2*np.log(delta)
    lEZ=gammaln(a-l)+gammaln(a_s)-gammaln(a)-gammaln(a_s-l)
    return np.logaddexp(lA,lB+lEZ)
def rho(D,b,l): n=D-b; return gammaln(n/2-l)+gammaln(D/2)-gammaln(n/2)-gammaln(D/2-l)
lam=10
for d,(kap,pi,mu) in [(10**4,(0.07588776354108091,0.47864251083758885,0.12941357593845532)),
                      (10**6,(0.03816778909618177,0.32728379025844495,0.024082246852806966))]:
    k,l,m=int(kap*d),int(pi*d/2),int(np.ceil(mu*d)); delta=1.001*(kap+mu)/(1-pi/(1-kap-mu))
    T=0.3411*d*lam if d==10**4 else 0.7786*d*lam; J=int(np.ceil(T/k)); D=d-2*J
    def blk(bs,js):
        best=-np.inf
        for b in bs:
            jj=np.array(list(js),dtype=float); v=np.max(dirlog(D,b,jj,l,delta)) if b>=2 else 0.0
            best=max(best,v+rho(D,b,l))
        return best
    s=blk([k-1,k],[1,max(1,m//2),m-1]); bs=range(2,k+1) if k<5000 else list(range(2,200))+list(range(k-2000,k+1))
    f=blk(bs,range(1,m))
    print(d,"k,m,l,D",k,m,l,D,"sampled lnG_blk",s,"full",f,"diff",f-s)
