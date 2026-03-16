"""
EuRAD 2026 — OFDM Radar for RPO Tumbling Target
=================================================
Generates 5 figures for 4-page EuRAD paper (radar-centric framing).

Fig 1: CRB(Ω) vs distance (K=1,2,5,10) + CRB(d) vs Gaudio baseline
Fig 2: CRB vs K (averaged, with shading)
Fig 3: Design guidelines — CRB vs bandwidth + CPI (two-panel)
Fig 4: CRB vs spin rate (absolute + relative)
Fig 5: Link budget validation (SNR vs distance)

Run:   python rpo_eurad_figures.py [--full]
Output: eurad/fig/*.pdf + eurad/fig/all_figures.png
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys, time

FAST = '--full' not in sys.argv
FIG_DIR = 'eurad/fig'
os.makedirs(FIG_DIR, exist_ok=True)

# === IEEE/EuRAD Style ===
IW, IH, FS, LFS, TFS = 3.4, 2.5, 9, 7, 8
matplotlib.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
    'mathtext.fontset':'cm','font.size':FS,'axes.labelsize':FS,
    'xtick.labelsize':TFS,'ytick.labelsize':TFS,'legend.fontsize':LFS,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.02,
    'lines.linewidth':1.2,'lines.markersize':4,'axes.linewidth':0.6,
    'grid.linewidth':0.4,'grid.alpha':0.3,
})
CB='#0072B2'; CR='#D55E00'; CG='#009E73'; CO='#E69F00'; CP='#CC79A7'; CC='#56B4E9'

# === System Parameters ===
c=3e8; kB=1.38e-23; fc=26e9; lam=c/fc; B=20e6; eta=0.6
N=256 if FAST else 1024
Df=B/N; Ts=(1/Df)*1.25; M=max(1,int(0.05/Ts))
if FAST and M>200: M=200
Tcpi=M*Ts; Pt=30; Ds=0.5; Gs=eta*(np.pi*Ds/lam)**2
L=10**(6/10); Tss=500; sw2=kB*Tss*Df; vr0=-0.5; Om0=np.deg2rad(2.0)

wh=np.array([0.,0.,1.]); nh=np.array([1.,0.,0.])
def skew(v): return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
Km=skew(wh); K2=Km@Km
def Rrod(t): return np.eye(3)+np.sin(t)*Km+(1-np.cos(t))*K2
def Rdot(t): return np.cos(t)*Km+np.sin(t)*K2

def gen_target(Ksc,seed=42):
    rng=np.random.RandomState(seed)
    return rng.randn(Ksc,3)*2.0,(rng.randn(Ksc)+1j*rng.randn(Ksc))/np.sqrt(2)

def compute_crb(Ps,d,vr,Om,pk,sk):
    Ksc=len(sk); beta=np.sqrt(Ps*Gs**2*lam**2/((4*np.pi)**3*d**4*L))
    t0=2*d/c; p0=2*np.pi*fc*t0; fd=-2*vr*fc/c; ab=beta*sk*np.exp(-1j*p0)
    na=np.arange(N); fn=fc+na*Df; phr=-2*np.pi*na*Df*t0; NM=N*M
    dh={k:np.zeros(NM,dtype=complex) for k in ['dd','dv','dO','dp']}
    for m in range(M):
        tm=m*Ts; thm=Om*tm; Rm=Rrod(thm); Rd=Rdot(thm)
        phd=2*np.pi*fd*tm; sl=slice(m*N,(m+1)*N)
        for k in range(Ksc):
            dtk=(2/c)*(nh@Rm@pk[k]); ddO=(2/c)*tm*(nh@Rd@pk[k])
            phm=-2*np.pi*fn*dtk; gk=ab[k]*np.exp(1j*(-p0+phr+phd+phm))
            dh['dd'][sl]+=(-1j*4*np.pi*na*Df/c)*gk
            dh['dv'][sl]+=(-1j*4*np.pi*fc*tm/c)*gk
            dh['dO'][sl]+=(-1j*2*np.pi*fn*ddO)*gk; dh['dp'][sl]+=(-1j)*gk
    ks=['dd','dv','dO','dp']; J=np.zeros((4,4))
    for i in range(4):
        for j in range(i,4):
            J[i,j]=(2/sw2)*np.real(np.conj(dh[ks[i]])@dh[ks[j]]); J[j,i]=J[i,j]
    J3=J[:3,:3]-J[:3,3:4]@J[3:4,:3]/J[3,3]
    try: return np.sqrt(np.diag(np.linalg.inv(J3)))
    except: return np.array([np.inf,np.inf,np.inf])

pk1,sk1=gen_target(1); pk2,sk2=gen_target(2)
pk5,sk5=gen_target(5); pk10,sk10=gen_target(10)

def savefig(name):
    for ext in ['pdf','png']: plt.savefig(f'{FIG_DIR}/{name}.{ext}')
    plt.close(); print(f"  {name}.pdf")

# ============================================================
print("="*60)
print(f"EuRAD 2026 Figures ({'FAST' if FAST else 'FULL'} mode)")
print("="*60)
T0=time.time()

# --- DATA ---
print("[Data] Fig 1: CRB vs distance...", end=' ', flush=True)
t0=time.time()
Npt=8 if FAST else 20
darr=np.linspace(0.5e3,25e3,Npt)
kcfgs=[(pk1,sk1,1),(pk2,sk2,2),(pk5,sk5,5),(pk10,sk10,10)]
crb1={}
for pk,sk,K in kcfgs:
    crb1[K]=np.array([np.rad2deg(compute_crb(Pt,d,vr0,Om0,pk,sk)[2]) for d in darr])
# Gaudio baseline + our CRB(d) for K=5
crb1_gau_d=[]; crb1_d_K5=[]
for d in darr:
    aeff2=Pt*Gs**2*lam**2*np.sum(np.abs(sk5)**2)/((4*np.pi)**3*d**4*L)
    crb1_gau_d.append(np.sqrt(6*sw2/(aeff2*(2*np.pi*Df)**2*N*M*(N**2-1)))*c/2*1e3)
    crb1_d_K5.append(compute_crb(Pt,d,vr0,Om0,pk5,sk5)[0]*1e3)
crb1_gau_d=np.array(crb1_gau_d); crb1_d_K5=np.array(crb1_d_K5)
print(f"({time.time()-t0:.0f}s)")

print("[Data] Fig 2: CRB vs K...", end=' ', flush=True)
Karr=[1,2,3,5,8,10,15,20]; Nreal=5 if FAST else 30
crb2_avg=[]; crb2_std=[]; crb2d_avg=[]; crb2d_std=[]
for Kv in Karr:
    vals=[np.rad2deg(compute_crb(Pt,5e3,vr0,Om0,*gen_target(Kv,100+s))[2]) for s in range(Nreal)]
    dvals=[compute_crb(Pt,5e3,vr0,Om0,*gen_target(Kv,100+s))[0]*1e3 for s in range(Nreal)]
    # Use median to handle ill-conditioned realizations at small K
    crb2_avg.append(np.median(vals)); crb2_std.append(np.std(vals))
    crb2d_avg.append(np.median(dvals)); crb2d_std.append(np.std(dvals))
crb2_avg=np.array(crb2_avg); crb2_std=np.array(crb2_std)
crb2d_avg=np.array(crb2d_avg); crb2d_std=np.array(crb2d_std)
print("done")

print("[Data] Fig 3: BW & CPI...", end=' ', flush=True)
t0=time.time()
N_bak,M_bak,Df_bak,Ts_bak,sw2_bak=N,M,Df,Ts,sw2
N_sw=128 if FAST else 256
B_arr=[5e6,10e6,20e6,40e6,80e6]; crb3_B_Om=[]; crb3_B_d=[]
for Bv in B_arr:
    Df_v=Bv/N_sw; Ts_v=(1/Df_v)*1.25; M_v=max(1,min(200 if FAST else 800,int(0.05/Ts_v)))
    N,M,Df,Ts=N_sw,M_v,Df_v,Ts_v; sw2=kB*Tss*Df_v
    cr=compute_crb(Pt,5e3,vr0,Om0,pk5,sk5)
    crb3_B_Om.append(np.rad2deg(cr[2])); crb3_B_d.append(cr[0]*1e3)
N,M,Df,Ts,sw2=N_bak,M_bak,Df_bak,Ts_bak,sw2_bak
crb3_B_Om=np.array(crb3_B_Om); crb3_B_d=np.array(crb3_B_d)
Tcpi_arr=[5e-3,10e-3,20e-3,50e-3,100e-3,200e-3]; crb3_T_Om=[]; crb3_T_d=[]
N_cpi=64; Df_cpi=B/1024; Ts_cpi=(1/Df_cpi)*1.25
for Tv in Tcpi_arr:
    M_v=max(1,int(Tv/Ts_cpi)); N,M,Df,Ts=N_cpi,M_v,Df_cpi,Ts_cpi; sw2=kB*Tss*Df_cpi
    cr=compute_crb(Pt,5e3,vr0,Om0,pk5,sk5)
    crb3_T_Om.append(np.rad2deg(cr[2])); crb3_T_d.append(cr[0]*1e3)
N,M,Df,Ts,sw2=N_bak,M_bak,Df_bak,Ts_bak,sw2_bak
crb3_T_Om=np.array(crb3_T_Om); crb3_T_d=np.array(crb3_T_d)
print(f"({time.time()-t0:.0f}s)")

print("[Data] Fig 4: Spin rate...", end=' ', flush=True)
t0=time.time()
N_s7=128 if FAST else 256; Df_s7=B/N_s7; Ts_s7=(1/Df_s7)*1.25
M_s7=max(1,min(200 if FAST else 800,int(0.05/Ts_s7)))
N,M,Df,Ts=N_s7,M_s7,Df_s7,Ts_s7; sw2=kB*Tss*Df_s7
Om_deg=np.array([0.1,0.2,0.5,1,2,5,10,20]); Om_rad=np.deg2rad(Om_deg)
crb4_Om=[]; crb4_rel=[]
for Omv in Om_rad:
    cr=compute_crb(Pt,5e3,vr0,Omv,pk5,sk5)
    crb4_Om.append(np.rad2deg(cr[2])); crb4_rel.append(cr[2]/Omv*100)
N,M,Df,Ts,sw2=N_bak,M_bak,Df_bak,Ts_bak,sw2_bak
crb4_Om=np.array(crb4_Om); crb4_rel=np.array(crb4_rel)
print(f"({time.time()-t0:.0f}s)")

print("[Data] Fig 5: Link budget...", end=' ', flush=True)
d_lb=np.linspace(0.5e3,30e3,100)
snr_lb={}
for Pv,Dv,lab in [(30,0.5,'30 W, $D$=0.5 m'),(100,0.5,'100 W, $D$=0.5 m'),(30,0.8,'30 W, $D$=0.8 m')]:
    Gv=eta*(np.pi*Dv/lam)**2
    snr_single=Pv*Gv**2*lam**2*10/((4*np.pi)**3*d_lb**4*L*kB*Tss*B)
    snr_lb[lab]=10*np.log10(snr_single)+10*np.log10(M_bak)
# Also compute σ=1 m² for baseline
Gv_base=eta*(np.pi*0.5/lam)**2
snr_1m2=30*Gv_base**2*lam**2*1/((4*np.pi)**3*d_lb**4*L*kB*Tss*B)
snr_lb_1m2=10*np.log10(snr_1m2)+10*np.log10(M_bak)
print("done")

# --- PLOTS ---
print("\n[Plotting]")

# Fig 1: Two-panel CRB vs distance
fig,(ax,axd)=plt.subplots(1,2,figsize=(IW*2+0.3,IH))
for (pk,sk,K),(col,ls,mk) in zip(kcfgs,[(CB,'-','o'),(CR,'--','s'),(CG,'-.','^'),(CO,':','D')]):
    ax.semilogy(darr/1e3,crb1[K],ls=ls,color=col,marker=mk,markevery=2,ms=3,label=f'$K={K}$')
ax.set_xlabel('Distance $d$ [km]'); ax.set_ylabel(r'$\sqrt{\mathrm{CRB}(\Omega)}$ [deg/s]')
ax.set_xlim([0.5,25]); ax.legend(loc='upper left',fontsize=6.5); ax.grid(True)
ax.set_title('(a) Spin Rate CRB',fontsize=FS)
idx10=np.argmin(np.abs(darr-10e3))
if idx10<len(crb1[1]) and idx10<len(crb1[2]):
    r12=crb1[1][idx10]/crb1[2][idx10]
    ax.annotate('',xy=(darr[idx10]/1e3,crb1[2][idx10]),xytext=(darr[idx10]/1e3,crb1[1][idx10]),
                arrowprops=dict(arrowstyle='<->',color=CR,lw=1))
    ax.text(darr[idx10]/1e3+1.5,np.sqrt(crb1[1][idx10]*crb1[2][idx10]),
            f'{r12:.0f}'+r'$\times$',fontsize=7,color=CR,fontweight='bold')
axd.semilogy(darr/1e3,crb1_d_K5,'-^',color=CG,ms=3,markevery=2,lw=1.5,label='This work ($K$=5)')
axd.semilogy(darr/1e3,crb1_gau_d,'--',color=CB,lw=1.5,label='Point target [7]')
axd.set_xlabel('Distance $d$ [km]'); axd.set_ylabel(r'$\sqrt{\mathrm{CRB}(d)}$ [mm]')
axd.set_xlim([0.5,25]); axd.legend(loc='upper left',fontsize=6.5); axd.grid(True)
axd.set_title('(b) Range CRB vs [7]',fontsize=FS)
plt.tight_layout(); savefig('fig1_crb_vs_distance')

# Fig 2: CRB vs K (median + shading)
fig,ax1=plt.subplots(figsize=(IW,IH))
ax1.semilogy(Karr,crb2_avg,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$) [deg/s]')
ax1.fill_between(Karr,np.maximum(crb2_avg-crb2_std,1e-8),crb2_avg+crb2_std,alpha=0.12,color=CG)
ax1.set_xlabel('Number of scatterers $K$'); ax1.set_xticks(Karr)
ax1.set_ylabel(r'$\sqrt{\mathrm{CRB}(\Omega)}$ [deg/s]',color=CG)
ax1.tick_params(axis='y',labelcolor=CG); ax1.grid(True)
ax2=ax1.twinx()
ax2.semilogy(Karr,crb2d_avg,'-o',color=CB,ms=4,lw=1.5,label=r'CRB($d$) [mm]')
ax2.fill_between(Karr,np.maximum(crb2d_avg-crb2d_std,1e-2),crb2d_avg+crb2d_std,alpha=0.08,color=CB)
ax2.set_ylabel(r'$\sqrt{\mathrm{CRB}(d)}$ [mm]',color=CB); ax2.tick_params(axis='y',labelcolor=CB)
r12=crb2_avg[0]/crb2_avg[1]
ax1.annotate(f'{r12:.0f}'+r'$\times$',
             xy=(1.5,np.sqrt(crb2_avg[0]*crb2_avg[1])),fontsize=7,color=CR,fontweight='bold',ha='center')
l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,b1+b2,loc='upper right',fontsize=6.5)
savefig('fig2_crb_vs_K')

# Fig 3: Design guidelines (BW + CPI)
fig,(a1,a2)=plt.subplots(1,2,figsize=(IW*2+0.3,IH))
a1.semilogy([b/1e6 for b in B_arr],crb3_B_Om,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$)')
a1b=a1.twinx()
a1b.semilogy([b/1e6 for b in B_arr],crb3_B_d,'-o',color=CB,ms=4,lw=1.5,label=r'CRB($d$)')
a1.set_xlabel('Bandwidth [MHz]'); a1.set_ylabel(r'CRB($\Omega$) [deg/s]',color=CG)
a1b.set_ylabel(r'CRB($d$) [mm]',color=CB)
a1.tick_params(axis='y',labelcolor=CG); a1b.tick_params(axis='y',labelcolor=CB)
a1.set_title('(a) $T_{\\mathrm{cpi}}$=50 ms',fontsize=FS)
l1,b1=a1.get_legend_handles_labels(); l2,b2=a1b.get_legend_handles_labels()
a1.legend(l1+l2,b1+b2,loc='upper right',fontsize=6); a1.grid(True)
a2.semilogy([t*1e3 for t in Tcpi_arr],crb3_T_Om,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$)')
a2b=a2.twinx()
a2b.semilogy([t*1e3 for t in Tcpi_arr],crb3_T_d,'-o',color=CB,ms=4,lw=1.5,label=r'CRB($d$)')
a2.set_xlabel('CPI Duration [ms]'); a2.set_ylabel(r'CRB($\Omega$) [deg/s]',color=CG)
a2b.set_ylabel(r'CRB($d$) [mm]',color=CB)
a2.tick_params(axis='y',labelcolor=CG); a2b.tick_params(axis='y',labelcolor=CB)
a2.set_title('(b) $B$=20 MHz',fontsize=FS)
l1,b1=a2.get_legend_handles_labels(); l2,b2=a2b.get_legend_handles_labels()
a2.legend(l1+l2,b1+b2,loc='upper right',fontsize=6); a2.grid(True)
plt.tight_layout(); savefig('fig3_design_guidelines')

# Fig 4: CRB vs spin rate
fig,ax1=plt.subplots(figsize=(IW,IH))
ax1.loglog(Om_deg,crb4_Om,'-^',color=CG,ms=5,lw=1.5,label=r'CRB($\Omega$) [deg/s]')
ax1.set_xlabel(r'Spin Rate $\Omega$ [deg/s]')
ax1.set_ylabel(r'$\sqrt{\mathrm{CRB}(\Omega)}$ [deg/s]',color=CG)
ax1.tick_params(axis='y',labelcolor=CG); ax1.grid(True,which='both',alpha=0.3)
ax2r=ax1.twinx()
ax2r.semilogx(Om_deg,crb4_rel,'-s',color=CR,ms=4,lw=1.5,label='Relative [%]')
ax2r.set_ylabel(r'Relative CRB/$\Omega$ [%]',color=CR); ax2r.tick_params(axis='y',labelcolor=CR)
l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2r.get_legend_handles_labels()
ax1.legend(l1+l2,b1+b2,loc='upper left',fontsize=6.5)
savefig('fig4_crb_vs_spinrate')

# Fig 5: Link budget with σ=1 m² overlay
fig,ax=plt.subplots(figsize=(IW,IH))
for lab,(col,ls) in zip(snr_lb.keys(),[(CB,'-'),(CR,'--'),(CG,'-.')]):
    ax.plot(d_lb/1e3,snr_lb[lab],ls=ls,color=col,lw=1.5,label=lab+', $\\sigma$=10')
ax.plot(d_lb/1e3,snr_lb_1m2,ls=':',color=CB,lw=1.0,label='30 W, $D$=0.5 m, $\\sigma$=1')
ax.axhline(10,color='gray',ls=':',lw=0.6); ax.text(27,12,'10 dB',fontsize=6,color='gray')
ax.axhline(0,color='gray',ls='--',lw=0.6); ax.text(27,2,'0 dB',fontsize=6,color='gray')
ax.axvspan(0.5,25,alpha=0.04,color=CB)
ax.text(12,-15,'RPO range',fontsize=7,color=CB,ha='center',style='italic')
ax.set_xlabel('Distance [km]'); ax.set_ylabel('SNR [dB] (50 ms CPI)')
ax.set_xlim([0.5,30]); ax.set_ylim([-20,80])
ax.legend(loc='upper right',fontsize=5.5,ncol=1); ax.grid(True)
savefig('fig5_link_budget')

# Composite
fig,axes=plt.subplots(2,3,figsize=(14,8))
for i,n in enumerate(['crb_vs_distance','crb_vs_K','design_guidelines','crb_vs_spinrate','link_budget']):
    ax=axes.flat[i]; img=plt.imread(f'{FIG_DIR}/fig{i+1}_{n}.png')
    ax.imshow(img); ax.set_title(f'Fig. {i+1}',fontsize=10); ax.axis('off')
axes.flat[5].axis('off'); plt.tight_layout()
fig.savefig(f'{FIG_DIR}/all_figures.png',dpi=150); plt.close()

print(f"\nTotal: {time.time()-T0:.1f}s → {FIG_DIR}/")