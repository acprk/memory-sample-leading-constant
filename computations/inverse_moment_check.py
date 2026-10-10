"""Check of the inverse-moment lemma (Lemma "Inverse moment" in paper/main.tex).
For Z ~ Beta(alpha, h1) with h1 = (b-1)/2: E Z^{-l} = prod_{t=1}^l (alpha+h1-t)/(alpha-t).
Part 1 lists cases where the tempting bound (alpha/(alpha-l))^{h1} fails, and confirms the stated
bound ((alpha-1)/(alpha-1-l))^{h1} in each; part 2 checks the stated bound exhaustively on
2alpha in [3,200), b in [2,40), 1 <= l < alpha-1 (40-digit mpmath).
usage: python3 inverse_moment_check.py"""
import mpmath as mp
mp.mp.dps=40
def EZ(a,h,l): return mp.fprod([(a+h-t)/(a-t) for t in range(1,l+1)])
bad=[]
for a2 in [5,9,21,101,1001,20001]:
  a=mp.mpf(a2)/2
  for b in [2,3,4,5,10]:
    h=mp.mpf(b-1)/2
    for l in sorted(set([1,2,int(a)//3,int(a)//2,int(a)-1])):
      if l<1 or l>=a: continue
      lhs=EZ(a,h,l); rhs=(a/(a-l))**h
      fix=((a-1)/(a-1-l))**h if l<a-1 else mp.inf
      if lhs>rhs: bad.append((float(a),b,l,float(lhs),float(rhs),float(fix), lhs<=fix))
for x in bad[:25]: print(x)
print(len(bad),"violations; all fixed bound ok:",all(x[-1] for x in bad))
# exhaustive check of fixed bound
ok=True
for a2 in range(3,200):
  a=mp.mpf(a2)/2
  for b in range(2,40):
    h=mp.mpf(b-1)/2
    for l in range(1,int(mp.ceil(a-1))):
      if l>=a-1: continue
      if EZ(a,h,l)>((a-1)/(a-1-l))**h*(1+mp.mpf(10)**-30): ok=False;print("fixfail",a,b,l)
print("fixed bound exhaustive ok:",ok)
