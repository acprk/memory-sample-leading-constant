"""Independent re-implementation: recomputation of the asymptotic constant c = kappa*pi*ln2/g and of the finite-d effective constant."""
import numpy as np, mpmath as mp
mp.mp.dps = 50
LN2 = mp.log(2)
def R1(kap, pi):   # int_0^{pi/2} ln((1/2-x)/((1-kap)/2-x)) dx   via mp.quad (independent of closed form)
    return mp.quad(lambda x: -mp.log1p(-(kap/2)/(mp.mpf(1)/2 - x)), [0, pi/2])
def casym(kap, pi, mu, zeta=mp.mpf('1e-3'), lam=None):
    kap, pi, mu = mp.mpf(kap), mp.mpf(pi), mp.mpf(mu)
    delta = (1+zeta)*(kap+mu)/(1 - pi/(1-kap-mu))
    g = R1(kap, pi) + kap/2*(-mp.log(1-delta))
    cinf = kap*pi*LN2/g
    c10 = None if lam is None else kap*(pi*LN2*lam/g - 1)/lam
    return cinf, c10, g, delta
lam10 = mp.log(10)/LN2
print("2ln2 =", mp.nstr(2*LN2, 12))
for p in [("1/16","1/16","1/16"),("1e-3","2e-2","1e-4"),("1e-5","1e-3","1e-6"),("1e-7","1e-4","1e-8"),("1e-9","1e-5","1e-10")]:
    k,pi_,m = [mp.mpf(eval(s)) for s in p]
    ci, c10, g, de = casym(k, pi_, m, lam=lam10)
    print(p, "c_inf=", mp.nstr(ci, 10), " c(lambda=log2 10)=", mp.nstr(c10, 10), " ratio to 2^-16:", mp.nstr(c10*2**16, 6),
          " loss 1-c/2ln2 =", mp.nstr(1-ci/(2*LN2), 4), "  (kappa+mu)/pi=", mp.nstr((k+m)/pi_,4), " pi/2=", mp.nstr(pi_/2,4))
# memory-threshold implied: need ln(w)/m << d*g, i.e. M*ln2/(mu d) << d g  =>  M/d^2 << mu*g/ln2
for p in [("1e-3","2e-2","1e-4"),("1e-5","1e-3","1e-6"),("1e-7","1e-4","1e-8")]:
    k,pi_,m = [mp.mpf(eval(s)) for s in p]
    _,_,g,_ = casym(k,pi_,m)
    print(p, " memory scale at which state term equals 1% of block cost: M/d^2 =", mp.nstr(0.01*m*g/LN2, 4),
          "; min d for k>=2,m>=2:", mp.nstr(max(2/k, 2/m), 3))

# ---------- finite-d effective constant (exact beta products, worst D=d-2J) ----------
def lbeta(a,b): return mp.loggamma(a)+mp.loggamma(b)-mp.loggamma(a+b)
def block_lnG(d, D, k, m, l, delta, M, T):
    best = -mp.inf
    for b in [k-1, k]:
        Dp, n = D-1, D-b
        dirmax = mp.mpf(0)
        for j in [1, max(1, m//2), m-1]:
            a, a_s, b0, b1 = mp.mpf(n-j)/2, mp.mpf(Dp-j)/2, mp.mpf(j)/2, mp.mpf(Dp-n+j-1)/2
            A = mp.exp(lbeta(a_s,b0)-lbeta(a,b0) - mp.mpf(b-1)/2*mp.log(1-delta))
            B = mp.exp(lbeta(a,b1)-lbeta(a,b0) - mp.mpf(b-2)/2*mp.log(delta))
            EZ = mp.exp(mp.loggamma(a-l)+mp.loggamma(a_s)-mp.loggamma(a)-mp.loggamma(a_s-l))
            dirmax = max(dirmax, mp.log(A + B*EZ))
        rho = mp.loggamma(mp.mpf(n)/2-l)+mp.loggamma(mp.mpf(D)/2)-mp.loggamma(mp.mpf(n)/2)-mp.loggamma(mp.mpf(D)/2-l)
        best = max(best, dirmax + rho)
    w = (T+2)*mp.mpf(2)**M
    return best + mp.log(w)/m
def Tmin(d, lam, M, kap, pi, mu, zeta=1e-3):
    k, l, m = int(kap*d), int(pi*d/2), int(np.ceil(mu*d)); p = 2*l
    if k < 2 or m < 2 or l < 1: return None
    delta = mp.mpf((1+zeta)*(kap+mu)/(1-pi/(1-kap-mu)))
    T = 2*d*lam  # iterate: guess T, compute J, lnG
    for _ in range(6):
        J = int(np.ceil(T/k)); D = d - 2*J
        if D - k - m - p < 10: return None
        lnG = block_lnG(d, D, k, m, l, delta, M, T)
        need = (p*lam*LN2 + mp.log(mp.mpf(2)/3))/lnG   # success>=2/3 needs J >= need
        Jmin = int(mp.ceil(need)); T = max(0, k*(Jmin-1)+1)
    return T
import itertools
if __name__ != "__main__": raise SystemExit
for d in [10**4, 10**6, 10**8]:
    for lam in [10, 30]:
        for Mlab, M in [("Kacz d(lam+2log d)", int(d*(lam+2*np.log2(d)))), ("d^1.5", int(d**1.5))]:
            best = (0, None)
            for kap, pi, mu in itertools.product([3e-2,1e-2,3e-3,1e-3,3e-4,1e-4], [0.2,0.1,0.05,0.02,0.01], [3e-2,1e-2,3e-3,1e-3,1e-4]):
                T = Tmin(d, lam, M, kap, pi, mu)
                if T is not None and T/(d*lam) > best[0]: best = (T/(d*lam), (kap,pi,mu))
            print(f"d={d:.0e} lam={lam} M={Mlab}: best finite-d c_eff = {best[0]:.4f} at {best[1]}  (2ln2=1.3863; old 2^-16=1.5e-5)", flush=True)
