"""
analyses.py: analyses of the transient multiscale model of L-arabinose oxidation
in a recycled packed-bed/stirred-tank system.

Run:  python analyses.py   (about 40 min; writes results_*.json and figure PNGs)
"""


# ==============================================================================
# Identifiability, validation, alternative models, oxygen balance, convergence, Morris screening, design map
# ==============================================================================
import json, time, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
import tandem_model as tm

BETA, DELTA = 4.877, 2.287
E = tm.EXPERIMENTS; G097, G162 = ["E1","E2"], ["E3","E4","E5","E6"]
OUT = {}

def pred(key, k, beta=BETA, delta=DELTA, kd=0.0, Nz=10, Nx=6, rtol=1e-4, atol=1e-7, **kw):
    d = E[key]; t = np.array(d["t_min"])*60.0
    p = tm.make_params(k, d["flow"], d["mass"], beta, delta, Nz=Nz, Nx=Nx)
    for a,b in kw.items(): setattr(p,a,b)
    if kd == 0.0:
        res = tm.simulate(p, t.max()*1.05+300, n_eval=200, rtol=rtol, atol=atol)
        return np.interp(t, res["t"], res["XA"]), res
    # first-order deactivation k'(t)=k' exp(-kd t): integrate in segments
    tt = np.linspace(0, t.max()*1.05+300, 60); XA=[0.0]
    # time-varying k' via rate law wrapper
    base = tm.rate_second_order
    tm._kd, tm._t = kd, 0.0
    rhs, n = tm.build_rhs(p)
    def rhs_t(tc, y):
        p.k_prime = k*(d["mass"]/tm.MASS_REF)*np.exp(-kd*tc); return rhs(tc, y)
    from scipy.integrate import solve_ivp
    y0 = np.ones(n); Nzi=p.Nz-1; npore=Nzi*p.Nx
    y0[2*Nzi:3*Nzi]=0; y0[3*Nzi+2*npore:3*Nzi+3*npore]=0; y0[-1]=0
    sol = solve_ivp(rhs_t,(0,tt[-1]),y0,method="LSODA",t_eval=tt,rtol=rtol,atol=atol)
    return np.interp(t, sol.t, 1-sol.y[-3]), None

def sse(keys, k, **kw):
    return sum(np.sum((np.array(E[q]["XA"])-pred(q,k,**kw)[0])**2) for q in keys)

def fit_k(keys, **kw):
    r = minimize_scalar(lambda lk: sse(keys, np.exp(lk), **kw), bounds=(np.log(1e-6), np.log(1e-1)), method="bounded", options={"xatol":1e-5})
    return float(np.exp(r.x)), float(r.fun)

t0=time.time()
# ── 1. Refit on production grid, per-run statistics, both bias definitions 
k097,_ = fit_k(G097); k162,_ = fit_k(G162); K={**{q:k097 for q in G097},**{q:k162 for q in G162}}
runs={}; allo=[]; allp=[]
for q in E:
    xp,res = pred(q,K[q]); xo=np.array(E[q]["XA"]); r=(xp-xo)*100
    runs[q]=dict(N=len(xo),RMSE=float(np.sqrt(np.mean(r**2))),bias=float(r.mean()))
    allo+=list(xo); allp+=list(xp)
allo,allp=np.array(allo),np.array(allp); r=(allp-allo)*100
OUT["fit"]=dict(k097=k097,k162=k162,k162_eff=k162*1.6225/0.897,ratio=k097/k162,ratio_eff=k097/(k162*1.6225/0.897),
    R2=float(1-np.sum((allo-allp)**2)/np.sum((allo-allo.mean())**2)),RMSE=float(np.sqrt(np.mean(r**2))),
    bias_pooled=float(r.mean()),bias_mean_of_runs=float(np.mean([v["bias"] for v in runs.values()])),runs=runs)
print("fit",OUT["fit"],time.time()-t0,flush=True)

# ── 2. Joint 4-parameter Fisher matrix (log-parameters), correlations 
th0=np.log([BETA,DELTA,k097,k162]); 
def allpred(th):
    b,dl,a,c=np.exp(th); return np.concatenate([pred(q,(a if q in G097 else c),beta=b,delta=dl,rtol=1e-6,atol=1e-9)[0] for q in E])
f0=allpred(th0); J=np.zeros((len(f0),4)); h=0.02
for j in range(4):
    e=np.zeros(4); e[j]=h; J[:,j]=(allpred(th0+e)-allpred(th0-e))/(2*h)
s2=np.sum((allo-f0)**2)/(len(allo)-4); JtJ=J.T@J
cond=float(np.linalg.cond(JtJ))
try:
    C=s2*np.linalg.inv(JtJ); sd=np.sqrt(np.diag(C)); corr=C/np.outer(sd,sd)
    OUT["fim"]=dict(names=["beta_A","delta_A","k097","k162"],rel_se=list(map(float,sd)),corr=corr.round(3).tolist(),cond=cond,
        ci95=[[float(np.exp(th0[i]-1.96*sd[i])),float(np.exp(th0[i]+1.96*sd[i]))] for i in range(4)],
        sens_norm=[float(np.linalg.norm(J[:,i])) for i in range(4)])
except Exception as ex: OUT["fim"]=dict(error=str(ex),cond=cond)
print("fim",OUT["fim"],time.time()-t0,flush=True)

# ── 3. Profile likelihood in beta_A (k' re-fitted for both groups at each beta)
prof=[]
for b in [0.0005,0.005,0.05,0.5,2.0,4.877,10,50]:
    a,s_a=fit_k(G097,beta=b); c,s_c=fit_k(G162,beta=b)
    prof.append(dict(beta=b,k097=a,k162=c,RMSE=float(np.sqrt((s_a+s_c)/len(allo))*100)))
OUT["profile_beta"]=prof; print("profile",prof,time.time()-t0,flush=True)

# ── 4. Leave-one-condition-out validation 
loo={}
for grp in (G097,G162):
    for q in grp:
        tr=[x for x in grp if x!=q]; kk,_=fit_k(tr); xp,_=pred(q,kk); xo=np.array(E[q]["XA"])
        loo[q]=dict(k_train=kk,RMSE_heldout=float(np.sqrt(np.mean((xp-xo)**2))*100),bias_heldout=float(np.mean(xp-xo)*100))
OUT["loo"]=loo; print("loo",loo,time.time()-t0,flush=True)

# ── 5. Alternative explanations for the loading anomaly: deactivation and free LHHW 
alt={}
for name,grp,k0 in (("097",G097,k097),("162",G162,k162)):
    f=lambda th: sum(np.sum((np.array(E[q]["XA"])-pred(q,np.exp(th[0]),kd=np.exp(th[1]))[0])**2) for q in grp)
    r=minimize(f,[np.log(k0*1.3),np.log(1e-6)],method="Nelder-Mead",options={"xatol":1e-3,"fatol":1e-7,"maxiter":120})
    n=sum(len(E[q]["XA"]) for q in grp); base=sse(grp,k0)
    alt["deact_"+name]=dict(k=float(np.exp(r.x[0])),kd_per_h=float(np.exp(r.x[1])*3600),half_life_h=float(np.log(2)/np.exp(r.x[1])/3600),
        RMSE=float(np.sqrt(r.fun/n)*100),RMSE_base=float(np.sqrt(base/n)*100))
    # free LHHW fit (K_A from literature, k' fitted freely, not matched on initial rate)
    def sse_l(lk):
        tot=0
        for q in grp:
            d=E[q]; t=np.array(d["t_min"])*60
            p=tm.make_params(np.exp(lk),d["flow"],d["mass"],BETA,DELTA,K_A=11.44,K_B=0.0)
            res=tm.simulate(p,t.max()*1.05+300,rate_law="lhhw",n_eval=200,rtol=1e-4,atol=1e-7)
            tot+=np.sum((np.array(d["XA"])-np.interp(t,res["t"],res["XA"]))**2)
        return tot
    rl=minimize_scalar(sse_l,bounds=(np.log(1e-5),np.log(10)),method="bounded")
    alt["lhhw_"+name]=dict(k=float(np.exp(rl.x)),RMSE=float(np.sqrt(rl.fun/n)*100))
OUT["alternatives"]=alt; print("alt",alt,time.time()-t0,flush=True)

# ── 6. Residence times, Pe, Da, per-pass conversion and oxygen balance per run 
c0=133.0; cO2_air=0.103; cO2_pure=0.46   # mol m-3 at 70 C, 1 bar total (Henry; pure O2 corrected for water vapour)
dims={}
for q,d in E.items():
    p=tm.make_params(K[q],d["flow"],d["mass"],BETA,DELTA); xp,res=pred(q,K[q])
    i=np.argmin(abs(res["t"]-3600)); Xpass=float(1-res["y_bed"][0,-1,i]/res["y_tank"][0,i])
    Q=d["flow"]/60e6; V=tm.V_LIQ
    dims[q]=dict(Q=d["flow"],gamma=p.gamma,tau_LT=p.tau_LT,Pe=1/(p.alpha_A*p.gamma),Da=p.k_prime*p.gamma,X_pass_pct=Xpass*100,
        O2_demand_per_pass=0.5*c0*Xpass,ratio_demand_to_cstar_pure=0.5*c0*Xpass/cO2_pure,
        obs_rate_per_h=None, O2_ceiling_per_h_pure=2*cO2_pure*Q/(c0*V)*3600)
    t=np.array(d["t_min"])/60; X=np.array(d["XA"]); j=np.searchsorted(X,0.3); dims[q]["obs_rate_per_h"]=float(X[j]/t[j])
OUT["dims"]=dims; print("dims",dims,flush=True)

# ── 7. Dynamic check with O2 solved in bed and pores (liquid-only bed, sparged tank) 
o2={}
for q in ("E2","E6"):
    for lab,cs in (("pure_O2",cO2_pure),):
        xo,_=pred(q,K[q]); xs,res=pred(q,K[q],o2_active=True,o2_ratio=0.5*c0/cs)
        o2[q]=dict(cstar=cs,min_YB_bed=float(res["y_bed"][1].min()),min_YB_pore=float(res["Y_pore"][1].min()),
            X_end_sat=float(xo[-1]),X_at_obs_t50_sat=None,RMSE_O2model=float(np.sqrt(np.mean((xs-np.array(E[q]["XA"]))**2))*100),
            RMSE_sat=runs[q]["RMSE"])
OUT["o2_dynamic"]=o2; print("o2",o2,time.time()-t0,flush=True)

# ── 8. Mass conservation (A + C) in every compartment 
xp,res=pred("E6",k162,rtol=1e-6,atol=1e-9)
OUT["mass_balance_max_dev"]=float(max(np.abs(res["y_tank"][0]+res["y_tank"][2]-1).max(),
    np.abs(res["y_bed"][0]+res["y_bed"][2]-1).max(),np.abs(res["Y_pore"][0]+res["Y_pore"][2]-1).max()))
# NaOH volume added at full conversion (1 M NaOH, 1:1 neutralisation), 150 mL
OUT["naoh_volume_mL"]=0.133*tm.V_LIQ*1e6/1.0

# ── 9. Grid and tolerance convergence for E6 
conv=[]
for g in [(6,4),(10,6),(16,10),(24,14)]:
    t1=time.time(); xp,_=pred("E6",k162,Nz=g[0],Nx=g[1],rtol=1e-7,atol=1e-10)
    conv.append(dict(grid=g,X_mean=float(xp.mean()),RMSE=float(np.sqrt(np.mean((xp-np.array(E["E6"]["XA"]))**2))*100),cpu=time.time()-t1))
tol=[]
for rt in (1e-3,1e-4,1e-5,1e-6,1e-7):
    xp,_=pred("E6",k162,rtol=rt,atol=rt*1e-3); tol.append(dict(rtol=rt,X_mean=float(xp.mean())))
kfine,_=fit_k(G162,Nz=24,Nx=14)
OUT["convergence_E6"]=dict(grid=conv,tol=tol,k162_fine=kfine,k162_prod=k162); print("conv",OUT["convergence_E6"],flush=True)

# ── 10. Physically based Thiele modulus for 1.4-1.8 mm extrudates 
Dm=1.67e-9+(343.15-333.2)/20.0*(2.37e-9-1.67e-9)   # measured, Mogi et al. (2007), interpolated to 70 C
phys=[]
for dp in (1.4e-3,1.8e-3):
    for tau in (3.0,5.0):
        De=Dm*0.45/tau; L=dp/4   # characteristic length V/A of a long cylinder
        phi=L*np.sqrt(0.45*k097/De); eta=np.tanh(phi)/phi
        phys.append(dict(d_mm=dp*1e3,tortuosity=tau,De=De,beta_phys=De/(dp/2)**2,Phi=float(phi),eta=float(eta)))
OUT["thiele_physical"]=dict(Dm_70C=Dm,cases=phys,De_implied_by_fitted_beta=[BETA*(dp/2)**2 for dp in (1.4e-3,1.8e-3)])

# ── 11. Morris: 3 seeds x r=20 (nominal k' = fitted value for the 0.897 g runs)
tm.PARAM_NOM["k_prime"]=k097; tm.PARAM_BND.update({k:(v*0.5,v*1.5) for k,v in tm.PARAM_NOM.items()})
mor={}
for sd in (1,2,3):
    m=tm.morris_screening([1.0,5.0,10.0],r=20,p_levels=10,seed=sd)
    mor[sd]={str(t):dict(mu_star=v["mu_star"].tolist(),mu=v["mu"].tolist(),sigma=v["sigma"].tolist(),XA=v["XA_base"]) for t,v in m.items()}
OUT["morris"]=mor; OUT["morris_names"]=tm.PARAM_LABELS
fig,ax=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
for a,t in zip(ax,("1.0","5.0","10.0")):
    M=np.array([mor[s][t]["mu_star"] for s in mor]); XA=np.mean([mor[s][t]["XA"] for s in mor])
    mm=M.mean(0)/XA; lo=(M.min(0))/XA; hi=(M.max(0))/XA; o=np.arange(len(mm))
    a.barh(o,mm,color="#1d4ed8",alpha=.85); a.errorbar(mm,o,xerr=[mm-lo,hi-mm],fmt="none",ecolor="k",capsize=3)
    a.set_title(f"t = {float(t):.0f} h (X$_A$ ≈ {XA*100:.0f}%)"); a.set_xlabel(r"$\mu^*/X_A$")
ax[0].set_yticks(range(len(tm.PARAM_LABELS))); ax[0].set_yticklabels(tm.PARAM_LABELS)
plt.tight_layout(); plt.savefig("Fig7_morris.png",dpi=600)

# ── 12. Corrected design map with both calibrated k' groups 
masses=[0.5,1.0,2.0,4.0,8.0]; flows=[30,50,100,150,200,300,500]; dm={}
for lab,kk in (("0.897 g group",k097),("1.6225 g group",k162)):
    Z=np.zeros((len(masses),len(flows)))
    for i,m in enumerate(masses):
        for j,f in enumerate(flows):
            p=tm.make_params(kk,f,m,BETA,DELTA); res=tm.simulate(p,8*3600+100,n_eval=40,rtol=1e-5,atol=1e-8)
            Z[i,j]=np.interp(8*3600,res["t"],res["XA"])*100
    dm[lab]=Z.round(1).tolist()
OUT["design_map"]=dict(masses=masses,flows=flows,XA8h=dm)
fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for a,(lab,Z) in zip(ax,dm.items()):
    for i,m in enumerate(masses): a.plot(flows,Z[i],"o-",label=f"{m} g")
    a.axhline(90,ls="--",c="grey"); a.set_xscale("log"); a.set_xlabel("Q [mL min$^{-1}$]"); a.set_title(f"k′ from {lab}")
ax[0].set_ylabel("X$_A$ at 8 h [%]"); ax[1].legend(fontsize=8); plt.tight_layout(); plt.savefig("Fig8_design_map.png",dpi=600)
# profile plot
fig,a=plt.subplots(figsize=(5,3.8)); a.semilogx([p["beta"] for p in prof],[p["RMSE"] for p in prof],"o-k")
a.set_xlabel(r"$\beta_A$ [s$^{-1}$] (fixed, k′ re-fitted)"); a.set_ylabel("Global RMSE [%]"); plt.tight_layout(); plt.savefig("FigS_profile_beta.png",dpi=600)
json.dump(OUT,open("results_main.json","w"),indent=1,default=float); print("DONE",time.time()-t0)


# ==============================================================================
# Single vs per-run rate constants, joint scaling of beta_A and delta_A, apparent loop constant
# ==============================================================================
import json, numpy as np
from scipy.optimize import minimize_scalar
import tandem_model as tm
E=tm.EXPERIMENTS; B0,D0=4.877,2.287; G097,G162=["E1","E2"],["E3","E4","E5","E6"]; ALL=G097+G162
N097,N162=(sum(len(E[q]["XA"]) for q in g) for g in (G097,G162))
def pred(q,k,beta=B0,delta=D0):
    d=E[q]; t=np.array(d["t_min"])*60; p=tm.make_params(k,d["flow"],d["mass"],beta,delta)
    r=tm.simulate(p,t.max()*1.05+300,n_eval=200,rtol=1e-5,atol=1e-8); return np.interp(t,r["t"],r["XA"]),r
def fit(keys,**kw):
    f=lambda lk: sum(np.sum((np.array(E[q]["XA"])-pred(q,np.exp(lk),**kw)[0])**2) for q in keys)
    r=minimize_scalar(f,bounds=(np.log(1e-8),np.log(5.0)),method="bounded",options={"xatol":1e-5})
    n=sum(len(E[q]["XA"]) for q in keys); return float(np.exp(r.x)),float(np.sqrt(r.fun/n)*100)
O={}
# (a) one k' shared by all six runs
O["shared_k"]=dict(zip(("k","RMSE"),fit(ALL))); print("a",O["shared_k"],flush=True)
# (b) k' fitted to each run alone, to separate loading from flow (design confounds them)
O["per_run"]={q:dict(zip(("k","RMSE"),fit([q])),Q=E[q]["flow"],m=E[q]["mass"]) for q in ALL}; print("b",O["per_run"],flush=True)
# (c) beta and delta scaled together (both contain D_e), k' re-fitted for both loadings
sc=[]
for f in (1.0,1e-1,1e-2,1e-3,1e-4):
    a,ra=fit(G097,beta=B0*f,delta=D0*f); c,rc=fit(G162,beta=B0*f,delta=D0*f)
    rm=np.sqrt((ra**2*N097+rc**2*N162)/(N097+N162)); phi=np.sqrt(0.45*a/(B0*f))
    sc.append(dict(f=f,beta=B0*f,delta=D0*f,k097=a,k162=c,RMSE=float(rm),Phi=float(phi),eta=float(np.tanh(phi)/phi)))
    print("c",sc[-1],flush=True)
O["joint_scaling"]=sc
# (d) lower end of the physical beta range with delta fixed
a,ra=fit(G097,beta=2.0e-4); c,rc=fit(G162,beta=2.0e-4); phi=np.sqrt(0.45*a/2e-4)
O["beta_2e-4"]=dict(k097=a,k162=c,RMSE=float(np.sqrt((ra**2*N097+rc**2*N162)/(N097+N162))),Phi=float(phi),eta=float(np.tanh(phi)/phi)); print("d",O["beta_2e-4"],flush=True)
# (e) identifiable quantity: apparent first-order loop constant k_obs of the tank curve (E2), across beta
def kobs(k,beta=B0,delta=D0):
    d=E["E2"]; p=tm.make_params(k,d["flow"],d["mass"],beta,delta); r=tm.simulate(p,8*3600,n_eval=200,rtol=1e-6,atol=1e-9)
    m=(r["XA"]>0.02)&(r["XA"]<0.9); return float(-np.polyfit(r["t"][m],np.log(1-r["XA"][m]),1)[0]*3600)
prof=json.load(open("results_main.json"))["profile_beta"]
O["kobs_profile"]=[dict(beta=p["beta"],kobs_per_h=kobs(p["k097"],beta=p["beta"])) for p in prof if p["beta"]<=10]
kf=json.load(open("results_main.json"))["fit"]["k097"]   # fitted k' of the 0.897 g runs and its Fisher interval
pr={q:tm.predict_XA(tm.make_params(kf,E[q]["flow"],E[q]["mass"]),np.array(E[q]["t_min"])*60) for q in G097}
fc=tm.fisher_ci(kf,G097,pr); O["kobs_ci"]=dict(est=kobs(kf),lo=kobs(fc["lower"]),hi=kobs(fc["upper"]),k=kf,k_lo=fc["lower"],k_hi=fc["upper"])
print("e",O["kobs_profile"],O["kobs_ci"],flush=True)
json.dump(O,open("results_identifiability.json","w"),indent=1); print("DONE")


# ==============================================================================
# Transport criteria from observed rates: Weisz-Prater, Carberry, Ergun, column-to-particle ratio
# ==============================================================================
import json, numpy as np, itertools
import tandem_model as tm
mu, rho = 4.04e-4, 977.8              # Pa s, kg m-3
# arabinose in water, infinite dilution (Mogi et al., 2007, J. Chem. Eng. Data 52, 40-43; Taylor dispersion):
# 1.67e-9 m2/s at 333.2 K and 2.37e-9 m2/s at 353.2 K -> linear interpolation to 343.15 K
Dm = 1.67e-9 + (343.15-333.2)/(353.2-333.2)*(2.37e-9-1.67e-9)
c0, VL, Dt, Lb, epsB, epsP = 133.0, tm.V_LIQ, 0.010, 0.130, 0.40, 0.45
A = np.pi*Dt**2/4
out = {}
for q, d in tm.EXPERIMENTS.items():
    t = np.array(d["t_min"])/60; X = np.array(d["XA"]); j = np.searchsorted(X, 0.3)
    rate_X = X[j]/(t[j]*3600)                       # s-1, mean rate to X = 0.3
    n_dot = rate_X*c0*VL                            # mol s-1
    us = d["flow"]/60e6/A                           # superficial velocity
    res = {"Q": d["flow"], "m": d["mass"], "u_s_mm_s": us*1e3}
    WP, Ca = [], []
    for dp, tau, rp in itertools.product((1.4e-3, 1.8e-3), (3.0, 5.0), (0.9e3, 1.2e3)):
        Vcat = d["mass"]*1e-3/rp; r_obs = n_dot/Vcat           # mol m-3(cat) s-1
        De = Dm*epsP/tau; Lc = dp/4                            # V/A of a long cylinder
        c_s = c0*(1-X[j]/2)                                    # mean concentration over the interval
        WP.append(r_obs*Lc**2/(De*c_s))
        Re = rho*us*dp/mu; Sc = mu/(rho*Dm); Sh = (1.09/epsB*Re**(1/3) if Re < 55 else 0.25/epsB*Re**0.69)*Sc**(1/3)   # Wilson-Geankoplis (two Re ranges)
        kL = Sh*Dm/dp; a = 1/Lc
        Ca.append(r_obs/(kL*a*c_s))
    dp = 1.6e-3
    dP = (150*mu*(1-epsB)**2*us/(epsB**3*dp**2) + 1.75*rho*(1-epsB)*us**2/(epsB**3*dp))*Lb
    res.update(WeiszPrater=[min(WP), max(WP)], Carberry=[min(Ca), max(Ca)], Re_p=rho*us*dp/mu,
               dP_Pa=dP, pump_W=dP*d["flow"]/60e6)
    out[q] = res
out["Dt_over_dp"] = [Dt/1.8e-3, Dt/1.4e-3]
json.dump(out, open("results_transport.json", "w"), indent=1)
for q in tm.EXPERIMENTS:
    r = out[q]; print(q, r["Q"], r["m"], "WP %.1f-%.1f" % tuple(r["WeiszPrater"]), "Ca %.3f-%.3f" % tuple(r["Carberry"]),
          "Re %.0f dP %.0f Pa P %.4f W" % (r["Re_p"], r["dP_Pa"], r["pump_W"]))
print("Dt/dp", out["Dt_over_dp"])


# ==============================================================================
# Identifiability of k' and the bed residence time
# ==============================================================================

import json, numpy as np
from scipy.optimize import minimize_scalar
import tandem_model as tm
E=tm.EXPERIMENTS; out={}
for g0 in (20.0,3.5):
    res={}
    for grp in (["E1","E2"],["E3","E4","E5","E6"]):
        def sse(lk):
            s=0
            for q in grp:
                d=E[q]; t=np.array(d["t_min"])*60; p=tm.make_params(np.exp(lk),d["flow"],d["mass"])
                p.gamma=g0*70.0/d["flow"]; r=tm.simulate(p,t.max()*1.05+300,n_eval=200,rtol=1e-4,atol=1e-7)
                s+=np.sum((np.array(d["XA"])-np.interp(t,r["t"],r["XA"]))**2)
            return s
        r=minimize_scalar(sse,bounds=(np.log(1e-6),np.log(1)),method="bounded",options={"xatol":1e-5})
        n=sum(len(E[q]["XA"]) for q in grp); res["+".join(grp)]=dict(k=float(np.exp(r.x)),kgamma=float(np.exp(r.x)*g0),RMSE=float(np.sqrt(r.fun/n)*100))
    out[str(g0)]=res
print(json.dumps(out,indent=1)); json.dump(out,open("results_k_gamma.json","w"),indent=1)


# ==============================================================================
# Shared rate constant broken down by run, and sensitivity to the bed Peclet number
# ==============================================================================

import json, numpy as np
import tandem_model as tm
E = tm.EXPERIMENTS; out = {"shared": {}, "peclet": []}
k_sh = json.load(open("results_identifiability.json"))["shared_k"]["k"]
for q, d in E.items():                                     # one k' for all six runs
    t = np.array(d["t_min"]) * 60; r = (tm.predict_XA(tm.make_params(k_sh, d["flow"], d["mass"]), t) - np.array(d["XA"])) * 100
    out["shared"][q] = dict(RMSE=float(np.sqrt(np.mean(r ** 2))), bias=float(r.mean()), max_abs=float(np.abs(r).max()))
F = json.load(open("results_main.json"))["fit"]
for s in (1.0, 0.2, 0.05):                                 # alpha / s  ->  Pe x 1, 5, 20 (fitted k' kept)
    res = []
    for q, d in E.items():
        p = tm.make_params(F["k097"] if d["mass"] < 1 else F["k162"], d["flow"], d["mass"])
        for a in ("alpha_A", "alpha_B", "alpha_C"): setattr(p, a, getattr(p, a) * s)
        t = np.array(d["t_min"]) * 60; res += list((tm.predict_XA(p, t) - np.array(d["XA"])) * 100)
    out["peclet"].append(dict(Pe=float(1 / (p.alpha_A * p.gamma)), Pe_p=float(1 / (p.alpha_A * p.gamma)) * 1.6e-3 / 0.130,
                              RMSE=float(np.sqrt(np.mean(np.array(res) ** 2)))))
# nested-model test: one shared k' (p = 1) against one k' per loading (p = 2), same data points
N = sum(len(E[q]["XA"]) for q in E)
sse1 = N * (json.load(open("results_identifiability.json"))["shared_k"]["RMSE"] / 100) ** 2
sse2 = N * (F["RMSE"] / 100) ** 2
from scipy.stats import f as fdist
Fst = (sse1 - sse2) / (sse2 / (N - 2))
out["nested_test"] = dict(N=N, SSE_shared=sse1, SSE_two=sse2, SSE_ratio=sse1 / sse2, F=Fst, p_value=float(fdist.sf(Fst, 1, N - 2)),
                          dAIC=float(N * np.log(sse1 / N) + 2 * 1 - (N * np.log(sse2 / N) + 2 * 2)))
print(json.dumps(out, indent=1)); json.dump(out, open("results_shared_peclet.json", "w"), indent=1)


# ==============================================================================
# Product inhibition: r = k' Y_A / (1 + K_C Y_C), k' and K_C fitted per loading
# ==============================================================================

import json, numpy as np
from scipy.optimize import minimize
import tandem_model as tm
E = tm.EXPERIMENTS; F = json.load(open("results_main.json"))["fit"]; out = {}
for name, grp, k0 in (("097", ["E1", "E2"], F["k097"]), ("162", ["E3", "E4", "E5", "E6"], F["k162"])):
    def sse(th):
        tot = 0.0
        for q in grp:
            d = E[q]; t = np.array(d["t_min"]) * 60
            p = tm.make_params(np.exp(th[0]), d["flow"], d["mass"]); p.K_C = np.exp(th[1])
            r = tm.simulate(p, t.max() * 1.05 + 300, rate_law="product_inhibition", n_eval=200, rtol=1e-4, atol=1e-7)
            tot += np.sum((np.array(d["XA"]) - np.interp(t, r["t"], r["XA"])) ** 2)
        return tot
    best = None
    for kc0 in (1e-2, 1.0, 10.0):                         # three starts; K_C -> 0 recovers first order
        r = minimize(sse, [np.log(k0 * (1 + kc0 / 2)), np.log(kc0)], method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 150})
        if best is None or r.fun < best.fun: best = r
    n = sum(len(E[q]["XA"]) for q in grp)
    out["pinhib_" + name] = dict(k=float(np.exp(best.x[0])), K_C=float(np.exp(best.x[1])), RMSE=float(np.sqrt(best.fun / n) * 100))
print(json.dumps(out, indent=1)); json.dump(out, open("results_product_inhibition.json", "w"), indent=1)

# ==============================================================================
# Figures 7 and 8
# ==============================================================================
# Redraws Figures 7 and 8 from results_main.json with cleaner labels (no recomputation)
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
R=json.load(open("results_main.json"))
lab=[r"$\beta_A$ (pore diffusion)",r"$\delta_A$ (L–S coupling)",r"$\gamma$ (bed residence time)",r"$\varepsilon_P$ (porosity)",
     r"$\tau_{LT}$ (tank space time)",r"$\varepsilon_{LT}$ (tank holdup)",r"$k'$ (rate parameter)",r"$\alpha_A$ (axial dispersion)",r"$\lambda_A$ (1/Bi)"]
fig,ax=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
for a,t in zip(ax,("1.0","5.0","10.0")):
    M=np.array([R["morris"][s][t]["mu_star"] for s in R["morris"]]); XA=np.mean([R["morris"][s][t]["XA"] for s in R["morris"]])
    mm=M.mean(0)/XA; o=np.arange(len(mm))
    a.barh(o,mm,color="#1d4ed8",alpha=.85); a.errorbar(mm,o,xerr=[mm-M.min(0)/XA,M.max(0)/XA-mm],fmt="none",ecolor="k",capsize=3)
    a.set_title(f"t = {float(t):.0f} h ($X_A$ ≈ {XA*100:.0f} %)"); a.set_xlabel(r"$\mu^*/X_A$")
ax[0].set_yticks(range(len(lab))); ax[0].set_yticklabels(lab); plt.tight_layout(); plt.savefig("Fig7_morris.png",dpi=600)
dm=R["design_map"]; fig,ax=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for a,(k,Z) in zip(ax,dm["XA8h"].items()):
    for i,m in enumerate(dm["masses"]): a.plot(dm["flows"],Z[i],"o-",label=f"{m:g} g")
    a.axhline(90,ls="--",c="grey"); a.set_xscale("log"); a.set_xticks(dm["flows"])
    a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel(r"$Q$ [mL min$^{-1}$]"); a.set_title(("(a) " if "0.897" in k else "(b) ")+f"$k'$ from {k}")
ax[0].set_ylabel(r"$X_A$ at 8 h [%]"); ax[1].legend(fontsize=8,loc="center right"); plt.tight_layout(); plt.savefig("Fig8_design_map.png",dpi=600)
