"""
EuRAD 2026 — OFDM Radar RPO  (v2 + MC RMSE validation)
========================================================
Fig 1: CRB(Ω)+CRB(d) vs distance  ★ ML-RMSE markers
Fig 2: CRB vs K  (median, shading)
Fig 3: Design guidelines — BW + CPI
Fig 4: CRB vs spin rate
Fig 5: Link budget (σ=10 + σ=1 m²)

  python rpo_eurad_figures.py           # FAST  ~3 min
  python rpo_eurad_figures.py --full    # FULL  ~25 min
"""
import numpy as np, matplotlib, matplotlib.pyplot as plt
import os, sys, time

FAST = '--full' not in sys.argv
FIG_DIR = 'eurad/fig'; os.makedirs(FIG_DIR, exist_ok=True)

# ── style ────────────────────────────────────────────────────
IW,IH,FS,LFS,TFS = 3.4,2.5,9,7,8
matplotlib.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
    'mathtext.fontset':'cm','font.size':FS,'axes.labelsize':FS,
    'xtick.labelsize':TFS,'ytick.labelsize':TFS,'legend.fontsize':LFS,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.02,
    'lines.linewidth':1.2,'lines.markersize':4,'axes.linewidth':0.6,
    'grid.linewidth':0.4,'grid.alpha':0.3})
CB='#0072B2';CR='#D55E00';CG='#009E73';CO='#E69F00';CP='#CC79A7';CC='#56B4E9'

# ── constants ────────────────────────────────────────────────
c=3e8; kB=1.38e-23; fc=26e9; lam=c/fc; B=20e6; eta=0.6
Pt=30; Ds=0.5; Gs=eta*(np.pi*Ds/lam)**2; L=10**(6/10); Tss=500
vr0=-0.5; Om0=np.deg2rad(2.0)

# main OFDM grid
N_main=256 if FAST else 1024; Df_main=B/N_main; Ts_main=(1/Df_main)*1.25
M_main=max(1,int(0.05/Ts_main))
if FAST and M_main>200: M_main=200
sw2_main=kB*Tss*Df_main

# mutable state used by compute_crb
N=N_main; M=M_main; Df=Df_main; Ts=Ts_main; sw2=sw2_main

# ── rotation ─────────────────────────────────────────────────
wh=np.array([0.,0.,1.]); nh=np.array([1.,0.,0.])
Km=np.array([[0,-1,0],[1,0,0],[0,0,0]],dtype=float)  # skew(wh)
K2=Km@Km
def Rrod(t): return np.eye(3)+np.sin(t)*Km+(1-np.cos(t))*K2
def Rdot(t): return np.cos(t)*Km+np.sin(t)*K2

def gen_target(Ksc,seed=42):
    rng=np.random.RandomState(seed)
    return rng.randn(Ksc,3)*2.0,(rng.randn(Ksc)+1j*rng.randn(Ksc))/np.sqrt(2)

# ── fast signal vector (K-vectorized) ────────────────────────
def sig_fast(Ps,d,vr,Om,pk,sk, N_,M_,Df_,Ts_):
    """h ∈ ℂ^{N_·M_}, vectorized over scatterers."""
    beta=np.sqrt(Ps*Gs**2*lam**2/((4*np.pi)**3*d**4*L))
    tau0=2*d/c; p0=2*np.pi*fc*tau0; fd=-2*vr*fc/c
    ab=beta*sk*np.exp(-1j*p0)          # (K,)
    na=np.arange(N_); fn=fc+na*Df_     # (N,)
    phr=-2*np.pi*na*Df_*tau0           # (N,)
    h=np.zeros(N_*M_,dtype=complex)
    for m in range(M_):
        tm=m*Ts_; Rm=Rrod(Om*tm); phd=2*np.pi*fd*tm
        dtk=(2/c)*(pk@Rm.T@nh)                      # (K,)
        ph_k=-2*np.pi*fn[None,:]*dtk[:,None]         # (K,N)
        h[m*N_:(m+1)*N_]=np.sum(
            ab[:,None]*np.exp(1j*(-p0+phr[None,:]+phd+ph_k)), axis=0)
    return h

# ── CRB (module-level N,M,Df,Ts,sw2) ────────────────────────
def compute_crb(Ps,d,vr,Om,pk,sk):
    Ksc=len(sk); beta=np.sqrt(Ps*Gs**2*lam**2/((4*np.pi)**3*d**4*L))
    t0r=2*d/c; p0=2*np.pi*fc*t0r; fd=-2*vr*fc/c; ab=beta*sk*np.exp(-1j*p0)
    na=np.arange(N); fn=fc+na*Df; phr=-2*np.pi*na*Df*t0r; NM=N*M
    dh={k:np.zeros(NM,dtype=complex) for k in ('dd','dv','dO','dp')}
    for m in range(M):
        tm=m*Ts; thm=Om*tm; Rm=Rrod(thm); Rd=Rdot(thm)
        phd=2*np.pi*fd*tm; sl=slice(m*N,(m+1)*N)
        for k in range(Ksc):
            dtk=(2/c)*(nh@Rm@pk[k]); ddO=(2/c)*tm*(nh@Rd@pk[k])
            phm=-2*np.pi*fn*dtk; gk=ab[k]*np.exp(1j*(-p0+phr+phd+phm))
            dh['dd'][sl]+=(-1j*4*np.pi*na*Df/c)*gk
            dh['dv'][sl]+=(-1j*4*np.pi*fc*tm/c)*gk
            dh['dO'][sl]+=(-1j*2*np.pi*fn*ddO)*gk; dh['dp'][sl]+=(-1j)*gk
    ks=('dd','dv','dO','dp'); J=np.zeros((4,4))
    for i in range(4):
        for j in range(i,4):
            J[i,j]=(2/sw2)*np.real(np.conj(dh[ks[i]])@dh[ks[j]]); J[j,i]=J[i,j]
    J3=J[:3,:3]-J[:3,3:4]@J[3:4,:3]/J[3,3]
    try: return np.sqrt(np.diag(np.linalg.inv(J3)))
    except: return np.array([np.inf,np.inf,np.inf])

# ── ML estimator: 2-stage grid search ────────────────────────
def ml_estimate(z,Ps,d0,vr0_,Om0_,pk,sk, N_,M_,Df_,Ts_):
    """Phase-profiled ML: cost = |h^H z|²/||h||² ."""
    Ng=7     # 7³=343 per stage, ×2 stages = 686 evals
    dd=max(abs(d0)*5e-3, 1.0)
    dvr=max(abs(vr0_)*0.20, 0.01)
    dOm=max(abs(Om0_)*0.30, 1e-4)
    bd,bvr,bOm=d0,vr0_,Om0_
    for _ in range(2):
        dg=np.linspace(bd-dd,bd+dd,Ng)
        vg=np.linspace(bvr-dvr,bvr+dvr,Ng)
        Og=np.linspace(bOm-dOm,bOm+dOm,Ng)
        best=-np.inf
        for di in dg:
            for vi in vg:
                for Oi in Og:
                    h=sig_fast(Ps,di,vi,Oi,pk,sk,N_,M_,Df_,Ts_)
                    hh=np.real(np.conj(h)@h)
                    if hh<1e-30: continue
                    cost=np.abs(np.conj(h)@z)**2/hh
                    if cost>best: best=cost; bd,bvr,bOm=di,vi,Oi
        dd/=Ng; dvr/=Ng; dOm/=Ng
    return bd,bvr,bOm

def run_mc(Ps,d,vr,Om,pk,sk,N_,M_,Df_,Ts_,sw2_,Nt,seed=0):
    h0=sig_fast(Ps,d,vr,Om,pk,sk,N_,M_,Df_,Ts_)
    rng=np.random.RandomState(seed)
    sq_d=[]; sq_Om=[]
    for _ in range(Nt):
        noise=np.sqrt(sw2_/2)*(rng.randn(N_*M_)+1j*rng.randn(N_*M_))
        dh,vrh,Omh=ml_estimate(h0+noise,Ps,d,vr,Om,pk,sk,N_,M_,Df_,Ts_)
        sq_d.append((dh-d)**2); sq_Om.append((Omh-Om)**2)
    return np.sqrt(np.mean(sq_d)), np.sqrt(np.mean(sq_Om))

# ── targets ──────────────────────────────────────────────────
pk1,sk1=gen_target(1); pk2,sk2=gen_target(2)
pk5,sk5=gen_target(5); pk10,sk10=gen_target(10)
def savefig(n):
    for e in ('pdf','png'): plt.savefig(f'{FIG_DIR}/{n}.{e}')
    plt.close(); print(f"  {n}.pdf")

# ==============================================================
print("="*60)
print(f"EuRAD 2026 v2+MC  ({'FAST' if FAST else 'FULL'})")
print("="*60); T0=time.time()

# ── Fig 1 data: CRB vs distance ─────────────────────────────
print("[1] CRB vs distance …", end=' ', flush=True); t0=time.time()
Npt=8 if FAST else 20; darr=np.linspace(0.5e3,25e3,Npt)
kcfgs=[(pk1,sk1,1),(pk2,sk2,2),(pk5,sk5,5),(pk10,sk10,10)]
crb1={}
for pk,sk,K in kcfgs:
    crb1[K]=np.array([np.rad2deg(compute_crb(Pt,d,vr0,Om0,pk,sk)[2]) for d in darr])
crb1_gau_d=[]; crb1_d_K5=[]
for d in darr:
    aeff2=Pt*Gs**2*lam**2*np.sum(np.abs(sk5)**2)/((4*np.pi)**3*d**4*L)
    crb1_gau_d.append(np.sqrt(6*sw2/(aeff2*(2*np.pi*Df)**2*N*M*(N**2-1)))*c/2*1e3)
    crb1_d_K5.append(compute_crb(Pt,d,vr0,Om0,pk5,sk5)[0]*1e3)
crb1_gau_d=np.array(crb1_gau_d); crb1_d_K5=np.array(crb1_d_K5)
print(f"({time.time()-t0:.0f}s)")

# ── MC RMSE ──────────────────────────────────────────────────
print("[MC] ML estimation …")
t0=time.time()
N_mc=32 if FAST else 64; Df_mc=B/N_mc; Ts_mc=(1/Df_mc)*1.25
M_mc=max(1,int(0.05/Ts_mc));
if M_mc>80: M_mc=80
sw2_mc=kB*Tss*Df_mc; Ntrials=30 if FAST else 200

Nsv,Msv,Dfsv,Tssv,sw2sv=N,M,Df,Ts,sw2
N,M,Df,Ts,sw2=N_mc,M_mc,Df_mc,Ts_mc,sw2_mc

mc_dists=[1e3,2e3,5e3] if FAST else [1e3,2e3,5e3,8e3,12e3]
mc_crb_Om=[]; mc_rmse_Om=[]
for d_mc in mc_dists:
    cr=compute_crb(Pt,d_mc,vr0,Om0,pk5,sk5)
    mc_crb_Om.append(np.rad2deg(cr[2]))
    _,rOm=run_mc(Pt,d_mc,vr0,Om0,pk5,sk5,N_mc,M_mc,Df_mc,Ts_mc,sw2_mc,Ntrials)
    mc_rmse_Om.append(np.rad2deg(rOm))
    r=mc_rmse_Om[-1]/mc_crb_Om[-1] if mc_crb_Om[-1]>0 else 999
    print(f"  d={d_mc/1e3:>5.0f}km  CRB(Ω)={mc_crb_Om[-1]:.2e}  "
          f"RMSE={mc_rmse_Om[-1]:.2e}  ratio={r:.2f}x  ({time.time()-t0:.0f}s)")

N,M,Df,Ts,sw2=Nsv,Msv,Dfsv,Tssv,sw2sv
mc_km=[d/1e3 for d in mc_dists]

# ── Fig 2 data ───────────────────────────────────────────────
print("[2] CRB vs K …", end=' ', flush=True)
Karr=[1,2,3,5,8,10,15,20]; Nreal=5 if FAST else 30
crb2_avg=[]; crb2_std=[]; crb2d_avg=[]; crb2d_std=[]
for Kv in Karr:
    vals=[np.rad2deg(compute_crb(Pt,5e3,vr0,Om0,*gen_target(Kv,100+s))[2]) for s in range(Nreal)]
    dvals=[compute_crb(Pt,5e3,vr0,Om0,*gen_target(Kv,100+s))[0]*1e3 for s in range(Nreal)]
    crb2_avg.append(np.median(vals)); crb2_std.append(np.std(vals))
    crb2d_avg.append(np.median(dvals)); crb2d_std.append(np.std(dvals))
crb2_avg=np.array(crb2_avg); crb2_std=np.array(crb2_std)
crb2d_avg=np.array(crb2d_avg); crb2d_std=np.array(crb2d_std)
print("done")

# ── Fig 3 data ───────────────────────────────────────────────
print("[3] BW & CPI …", end=' ', flush=True); t0=time.time()
Nsv2,Msv2,Dfsv2,Tssv2,sw2sv2=N,M,Df,Ts,sw2
N_sw=128 if FAST else 256; B_arr=[5e6,10e6,20e6,40e6,80e6]
crb3_B_Om=[]; crb3_B_d=[]
for Bv in B_arr:
    Df_v=Bv/N_sw; Ts_v=(1/Df_v)*1.25; M_v=max(1,int(0.05/Ts_v))  # NO cap: M grows with B
    N,M,Df,Ts=N_sw,M_v,Df_v,Ts_v; sw2=kB*Tss*Df_v
    cr=compute_crb(Pt,5e3,vr0,Om0,pk5,sk5)
    crb3_B_Om.append(np.rad2deg(cr[2])); crb3_B_d.append(cr[0]*1e3)
N,M,Df,Ts,sw2=Nsv2,Msv2,Dfsv2,Tssv2,sw2sv2
crb3_B_Om=np.array(crb3_B_Om); crb3_B_d=np.array(crb3_B_d)
Tcpi_arr=[5e-3,10e-3,20e-3,50e-3,100e-3,200e-3]; crb3_T_Om=[]; crb3_T_d=[]
N_cpi=64; Df_cpi=B/1024; Ts_cpi=(1/Df_cpi)*1.25
for Tv in Tcpi_arr:
    M_v=max(1,int(Tv/Ts_cpi)); N,M,Df,Ts=N_cpi,M_v,Df_cpi,Ts_cpi; sw2=kB*Tss*Df_cpi
    cr=compute_crb(Pt,5e3,vr0,Om0,pk5,sk5); crb3_T_Om.append(np.rad2deg(cr[2])); crb3_T_d.append(cr[0]*1e3)
N,M,Df,Ts,sw2=Nsv2,Msv2,Dfsv2,Tssv2,sw2sv2
crb3_T_Om=np.array(crb3_T_Om); crb3_T_d=np.array(crb3_T_d)
print(f"({time.time()-t0:.0f}s)")

# ── Fig 4 data ───────────────────────────────────────────────
print("[4] Spin rate …", end=' ', flush=True); t0=time.time()
N_s7=128 if FAST else 256; Df_s7=B/N_s7; Ts_s7=(1/Df_s7)*1.25
M_s7=max(1,min(200 if FAST else 800,int(0.05/Ts_s7)))
N,M,Df,Ts=N_s7,M_s7,Df_s7,Ts_s7; sw2=kB*Tss*Df_s7
Om_deg=np.array([0.1,0.2,0.5,1,2,5,10,20]); Om_rad=np.deg2rad(Om_deg)
crb4_Om=[]; crb4_rel=[]
for Omv in Om_rad:
    cr=compute_crb(Pt,5e3,vr0,Omv,pk5,sk5)
    crb4_Om.append(np.rad2deg(cr[2])); crb4_rel.append(cr[2]/Omv*100)
N,M,Df,Ts,sw2=Nsv2,Msv2,Dfsv2,Tssv2,sw2sv2
crb4_Om=np.array(crb4_Om); crb4_rel=np.array(crb4_rel)
print(f"({time.time()-t0:.0f}s)")

# ── Fig 5 data ───────────────────────────────────────────────
print("[5] Link budget …", end=' ', flush=True)
d_lb=np.linspace(0.5e3,30e3,100); snr_lb={}
for Pv,Dv,lab in [(30,0.5,'30 W, $D$=0.5 m'),(100,0.5,'100 W, $D$=0.5 m'),(30,0.8,'30 W, $D$=0.8 m')]:
    Gv=eta*(np.pi*Dv/lam)**2
    snr_lb[lab]=10*np.log10(Pv*Gv**2*lam**2*10/((4*np.pi)**3*d_lb**4*L*kB*Tss*B))+10*np.log10(Msv2)
Gv0=eta*(np.pi*0.5/lam)**2
snr_lb_1m2=10*np.log10(30*Gv0**2*lam**2*1/((4*np.pi)**3*d_lb**4*L*kB*Tss*B))+10*np.log10(Msv2)
print("done")

# ==============================================================
#  PLOTS
# ==============================================================
print("\n[Plotting]")

# ── Fig 1: CRB + MC ─────────────────────────────────────────
fig,(ax,axd)=plt.subplots(1,2,figsize=(IW*2+0.3,IH))
for (pk,sk,K),(col,ls,mk) in zip(kcfgs,[(CB,'-','o'),(CR,'--','s'),(CG,'-.','^'),(CO,':','D')]):
    ax.semilogy(darr/1e3,crb1[K],ls=ls,color=col,marker=mk,markevery=2,ms=3,label=f'CRB, $K$={K}')
# MC overlay (Ω only, K=5)
ax.semilogy(mc_km,mc_crb_Om,'v',color='black',ms=5,mfc='none',mew=1.0,
            label=f'CRB ($N$={N_mc},$K$=5)',zorder=10)
ax.semilogy(mc_km,mc_rmse_Om,'*',color=CP,ms=8,mew=0.8,
            label=f'ML RMSE ({Ntrials} trials)',zorder=10)
ax.set_xlabel('Distance $d$ [km]'); ax.set_ylabel(r'$\sqrt{\mathrm{CRB}(\Omega)}$ [deg/s]')
ax.set_xlim([0.5,25]); ax.legend(loc='upper left',fontsize=5,ncol=1); ax.grid(True)
ax.set_title('(a) Spin Rate: CRB vs ML RMSE',fontsize=FS)
idx10=np.argmin(np.abs(darr-10e3))
if idx10<len(crb1[1]) and idx10<len(crb1[2]):
    r12=crb1[1][idx10]/crb1[2][idx10]
    ax.annotate('',xy=(darr[idx10]/1e3,crb1[2][idx10]),xytext=(darr[idx10]/1e3,crb1[1][idx10]),
                arrowprops=dict(arrowstyle='<->',color=CR,lw=1))
    ax.text(darr[idx10]/1e3+1.5,np.sqrt(crb1[1][idx10]*crb1[2][idx10]),f'{r12:.0f}'+r'$\times$',fontsize=7,color=CR,fontweight='bold')

axd.semilogy(darr/1e3,crb1_d_K5,'-^',color=CG,ms=3,markevery=2,lw=1.5,label='This work ($K$=5)')
axd.semilogy(darr/1e3,crb1_gau_d,'--',color=CB,lw=1.5,label='Point target [7]')
axd.set_xlabel('Distance $d$ [km]'); axd.set_ylabel(r'$\sqrt{\mathrm{CRB}(d)}$ [mm]')
axd.set_xlim([0.5,25]); axd.legend(loc='upper left',fontsize=6.5); axd.grid(True)
axd.set_title('(b) Range CRB vs [7]',fontsize=FS)
plt.tight_layout(); savefig('fig1_crb_vs_distance')

# ── Fig 2 ────────────────────────────────────────────────────
fig,ax1=plt.subplots(figsize=(IW,IH))
ax1.semilogy(Karr,crb2_avg,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$) [deg/s]')
ax1.fill_between(Karr,np.maximum(crb2_avg-crb2_std,1e-8),crb2_avg+crb2_std,alpha=0.12,color=CG)
ax1.set_xlabel('Number of scatterers $K$'); ax1.set_xticks(Karr)
ax1.set_ylabel(r'$\sqrt{\mathrm{CRB}(\Omega)}$ [deg/s]',color=CG); ax1.tick_params(axis='y',labelcolor=CG); ax1.grid(True)
ax2=ax1.twinx()
ax2.semilogy(Karr,crb2d_avg,'-o',color=CB,ms=4,lw=1.5,label=r'CRB($d$) [mm]')
ax2.fill_between(Karr,np.maximum(crb2d_avg-crb2d_std,1e-2),crb2d_avg+crb2d_std,alpha=0.08,color=CB)
ax2.set_ylabel(r'$\sqrt{\mathrm{CRB}(d)}$ [mm]',color=CB); ax2.tick_params(axis='y',labelcolor=CB)
r12=crb2_avg[0]/crb2_avg[1]
ax1.annotate(f'{r12:.0f}'+r'$\times$',xy=(1.5,np.sqrt(crb2_avg[0]*crb2_avg[1])),fontsize=7,color=CR,fontweight='bold',ha='center')
l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,b1+b2,loc='upper right',fontsize=6.5)
savefig('fig2_crb_vs_K')

# ── Fig 3 ────────────────────────────────────────────────────
fig,(a1,a2)=plt.subplots(1,2,figsize=(IW*2+0.3,IH))
a1.semilogy([b/1e6 for b in B_arr],crb3_B_Om,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$)')
a1b=a1.twinx(); a1b.semilogy([b/1e6 for b in B_arr],crb3_B_d,'-o',color=CB,ms=4,lw=1.5,label=r'CRB($d$)')
a1.set_xlabel('Bandwidth [MHz]'); a1.set_ylabel(r'CRB($\Omega$) [deg/s]',color=CG); a1b.set_ylabel(r'CRB($d$) [mm]',color=CB)
a1.tick_params(axis='y',labelcolor=CG); a1b.tick_params(axis='y',labelcolor=CB)
a1.set_title(r'(a) $T_{\mathrm{cpi}}$=50 ms',fontsize=FS)
l1,b1=a1.get_legend_handles_labels(); l2,b2=a1b.get_legend_handles_labels()
a1.legend(l1+l2,b1+b2,loc='upper right',fontsize=6); a1.grid(True)
a2.semilogy([t*1e3 for t in Tcpi_arr],crb3_T_Om,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$)')
a2b=a2.twinx(); a2b.semilogy([t*1e3 for t in Tcpi_arr],crb3_T_d,'-o',color=CB,ms=4,lw=1.5,label=r'CRB($d$)')
a2.set_xlabel('CPI Duration [ms]'); a2.set_ylabel(r'CRB($\Omega$) [deg/s]',color=CG); a2b.set_ylabel(r'CRB($d$) [mm]',color=CB)
a2.tick_params(axis='y',labelcolor=CG); a2b.tick_params(axis='y',labelcolor=CB)
a2.set_title(r'(b) $B$=20 MHz',fontsize=FS)
l1,b1=a2.get_legend_handles_labels(); l2,b2=a2b.get_legend_handles_labels()
a2.legend(l1+l2,b1+b2,loc='upper right',fontsize=6); a2.grid(True)
plt.tight_layout(); savefig('fig3_design_guidelines')

# ── Fig 4 ────────────────────────────────────────────────────
fig,ax1=plt.subplots(figsize=(IW,IH))
ax1.loglog(Om_deg,crb4_Om,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$) [deg/s]')
ax1.set_xlabel(r'Spin Rate $\Omega$ [deg/s]'); ax1.set_ylabel(r'$\sqrt{\mathrm{CRB}(\Omega)}$ [deg/s]',color=CG)
ax1.tick_params(axis='y',labelcolor=CG); ax1.grid(True,which='both',alpha=0.3)
ax2r=ax1.twinx(); ax2r.semilogx(Om_deg,crb4_rel,'-s',color=CR,ms=4,lw=1.5,label='Relative [%]')
ax2r.set_ylabel(r'Relative CRB/$\Omega$ [%]',color=CR); ax2r.tick_params(axis='y',labelcolor=CR)
l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2r.get_legend_handles_labels()
ax1.legend(l1+l2,b1+b2,loc='upper left',fontsize=6.5)
savefig('fig4_crb_vs_spinrate')

# ── Fig 5 ────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(IW,IH))
for lab,(col,ls) in zip(snr_lb.keys(),[(CB,'-'),(CR,'--'),(CG,'-.')]):
    ax.plot(d_lb/1e3,snr_lb[lab],ls=ls,color=col,lw=1.5,label=lab+', $\\sigma$=10')
ax.plot(d_lb/1e3,snr_lb_1m2,ls=':',color=CB,lw=1.0,label='30 W, $D$=0.5 m, $\\sigma$=1')
ax.axhline(10,color='gray',ls=':',lw=0.6); ax.text(27,12,'10 dB',fontsize=6,color='gray')
ax.axhline(0,color='gray',ls='--',lw=0.6); ax.text(27,2,'0 dB',fontsize=6,color='gray')
ax.axvspan(0.5,25,alpha=0.04,color=CB); ax.text(12,-15,'RPO range',fontsize=7,color=CB,ha='center',style='italic')
ax.set_xlabel('Distance [km]'); ax.set_ylabel('SNR [dB] (50 ms CPI)')
ax.set_xlim([0.5,30]); ax.set_ylim([-20,80]); ax.legend(loc='upper right',fontsize=5.5); ax.grid(True)
savefig('fig5_link_budget')

# ── composite ────────────────────────────────────────────────
fig,axes=plt.subplots(2,3,figsize=(14,8))
for i,n in enumerate(['crb_vs_distance','crb_vs_K','design_guidelines','crb_vs_spinrate','link_budget']):
    ax=axes.flat[i]; img=plt.imread(f'{FIG_DIR}/fig{i+1}_{n}.png')
    ax.imshow(img); ax.set_title(f'Fig. {i+1}',fontsize=10); ax.axis('off')
axes.flat[5].axis('off'); plt.tight_layout()
fig.savefig(f'{FIG_DIR}/all_figures.png',dpi=150); plt.close()

# ── summary ──────────────────────────────────────────────────
print(f"\n{'='*60}\nMC VALIDATION SUMMARY  (N={N_mc}, M={M_mc}, {Ntrials} trials)\n{'='*60}")
print(f"  {'d [km]':>8s}  {'CRB(Ω)':>12s}  {'RMSE(Ω)':>12s}  {'ratio':>8s}")
for i,dk in enumerate(mc_km):
    rO=mc_rmse_Om[i]/mc_crb_Om[i] if mc_crb_Om[i]>0 else 999
    print(f"  {dk:>7.0f}   {mc_crb_Om[i]:>11.2e}  {mc_rmse_Om[i]:>11.2e}  {rO:>7.2f}×")
print(f"\nTotal: {time.time()-T0:.1f}s  →  {FIG_DIR}/")