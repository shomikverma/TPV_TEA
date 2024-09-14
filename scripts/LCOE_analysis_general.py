import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm
import TPV_model as tpv
from scipy.optimize import fsolve
from matplotlib.sankey import Sankey

plt.style.use(['science','grid'])

# TPV_data_2200_ideal = tpv.P_balance_2(Trad=2200,ref=1,EQE=1,series_R=0,VF=0.25,emis=1)
# TPV_data_2200_real = tpv.P_balance_2(Trad=2200,ref=0.95,EQE=0.97,series_R=0.1e-6,VF=0.25,emis=1)
# TPV_data_2200_ideal = tpv.P_balance_2(Trad=2123, ref=1, EQE=1, series_R=0, VF=0.31, emis=1)
# TPV_data_2200_ideal.Name = 'Ideal'
# TPV_data_2200_real = tpv.P_balance_2(Trad=2123, ref=0.95, EQE=0.7, series_R=6.5e-3, VF=0.31, emis=1)
# TPV_data_2200_real.Name = 'Real'
# TPV_data_2200_real_imp_ref = tpv.P_balance_2(Trad=2123, ref=0.99, EQE=0.7, series_R=6.5e-3, VF=0.31, emis=1)
# TPV_data_2200_real_imp_ref.Name = 'Real Imp Ref'
# TPV_data_2200_real_imp_eqe = tpv.P_balance_2(Trad=2123, ref=0.95, EQE=0.9, series_R=6.5e-3, VF=0.31, emis=1)
# TPV_data_2200_real_imp_eqe.Name = 'Real Imp EQE'
# TPV_data_2200_real_imp_resis = tpv.P_balance_2(Trad=2123, ref=0.95, EQE=0.7, series_R=6.5e-4, VF=0.31, emis=1)
# TPV_data_2200_real_imp_resis.Name = 'Real Imp Resis'

TPV_data_2200_ideal = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=1, BW=1)
TPV_data_2200_ideal.Name = 'Ideal'
TPV_data_2200_real = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real.Name = 'Real'
TPV_data_2200_real_imp_ref = tpv.P_balance_3(Trad=2123, SBR=0.99, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_ref.Name = 'Real Imp Ref'
TPV_data_2200_real_imp_eqe = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=1, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_eqe.Name = 'Real Imp EQE'
TPV_data_2200_real_imp_resis = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-4, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_resis.Name = 'Real Imp Resis'
TPV_data_2200_real_imp_ABR = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.03, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_ABR.Name = 'Real Imp ABR'
TPV_data_2200_real_imp_T = tpv.P_balance_3(Trad=2623, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_T.Name = 'Real Imp T'
TPV_data_2200_real_imp_VF = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=1, emis=0.85, BW=1)
TPV_data_2200_real_imp_VF.Name = 'Real Imp VF'
TPV_data_2200_real_imp_emis = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=1, BW=1)
TPV_data_2200_real_imp_emis.Name = 'Real Imp emis'
TPV_data_2200_ideal_10 = TPV_data_2200_ideal.iloc[::10, :]
TPV_data_2200_real_10 = TPV_data_2200_real.iloc[::10, :]

# Pdens_TPV = 5 # W/cm^2
# eta_TPV = 0.5 # efficiency of TPV

# Pdens_TPV = np.linspace(0, 10, 10) # W/cm^2
# eta_TPV = np.linspace(0, 1, 100) # efficiency of TPV

Pdens_TPV = np.meshgrid(np.linspace(0.5, 10, 10), np.linspace(0.2, 1, 10))[0] # W/cm^2
eta_TPV = np.meshgrid(np.linspace(0.5, 10, 10), np.linspace(0.2, 1, 10))[1] # efficiency of TPV
# Pdens_TPV = np.linspace(0.5, 10, 10)[0] # W/cm^2
# eta_TPV = np.linspace(0.2, 1, 10)[9] # efficiency of TPV

# case!; combustor with H2 fuel at 100MW scale
CPA = 5 # $/cm^2 TPV cost per area
n = 20 # lifetime in years
i = 0.04 # interest rate
CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
# print(CRF)
t_out = 8760 # hours in a year


def sweep_CPEs(ir,n,t_out):
    CRF = (ir*(1+ir)**n)/((1+ir)**n-1) # capital recovery factor
    C1 = CRF/t_out
    print(C1)
    # CPE_sys = np.logspace(-6,-3,4)
    # CPE_energy = np.logspace(-6,-3,4)
    CPA_TPV = np.array([0.5,1,5,20])

    # CPE_tots = []
    # for i,CPE_th_sys in enumerate(CPE_sys):
    #     for j,CPE_th_input in enumerate(CPE_energy):
    #         CPE_tot = CPE_th_sys*CRF + CPE_th_input
    #         CPE_tots.append(CPE_tot)
    
    # CPE_tots = np.logspace(np.log10(min(CPE_tots)), np.log10(max(CPE_tots)), 4)
    CPE_tots = np.logspace(-4,-6,3)
    fig = plt.figure(figsize=(10,7.5))

    for i, CPE_tot in enumerate(CPE_tots):
        for j, CPA in enumerate(CPA_TPV):
            # print(CPE_th_sys, CPE_th_input, CPA)
            # t_out = 8760
        
            LCOE_TPV = (CPA*C1/Pdens_TPV) # $/Wh for TPV contribution
            # LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
            # LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
            LCOE_sys = (CPE_tot/eta_TPV) # $/Wh for device contribution
            # LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
            LCOE = LCOE_TPV + LCOE_sys # $/Wh total LCOE
            LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
            LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
            # print(Pdens_TPV, eta_TPV, LCOE_MWh)
            # exit()
            # reshape LCOE_MWh to 2D
            # 2D plot of LCOE vs Pdens_TPV and eta_TPV
            # plt.figure(figsize=(6,4))
            # on sublot i,j plot LCOE_MWh
            ax = fig.add_subplot(3,4,4*i+j+1)
            # color contour lines
            # plt.plot(TPV_data_2200_ideal_10['Pgen'], TPV_data_2200_ideal_10['eff'], 'k-o', label='TPV')
            # plt.plot(TPV_data_2200_real_10['Pgen'], TPV_data_2200_real_10['eff'], 'k--s', label='TPV')
            plt.contour(Pdens_TPV, eta_TPV, LCOE_kWh, levels=20, colors='k', linewidths=0.5)
            # plt.contour(Pdens_TPV, eta_TPV, LCOE_kWh, levels=[10], colors='r', linewidths=0.5)
            # lev_exp = np.arange(np.floor(np.log10(1),np.ceil(np.log10(600))))
            # levs = np.power(10, lev_exp)
            # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())
            # round min value to nearest power of 10 
            vmin = 0.1
            vmax = 100
            cf = plt.contourf(Pdens_TPV, eta_TPV, LCOE_kWh, locator=ticker.LogLocator(), cmap='plasma', norm=colors.LogNorm(), levels=np.logspace(np.log10(vmin),np.log10(vmax), 200), vmin=vmin, vmax=vmax)
            for c in cf.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
            # plt.pcolor(Pdens_TPV, eta_TPV, LCOE_MWh, cmap='jet', norm=colors.LogNorm(vmin=1, vmax=600), shading='gouraud')
            plt.xticks([2,4,6,8,10])
            plt.xlim(0.5, 10)
            plt.ylim(0.2, 1)
            # remove x labels for all but bottom row
            if i < 2:
                plt.xticks([])
            # remove y labels for all but left column
            if j > 0:
                plt.yticks([])
            plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.925, 0.043, 0.025, 0.93])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    # cbar = plt.colorbar()
    cbar.set_ticks([0.1, 1, 10, 100])
    # cbar.set_ticks([0.1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,60])
    # cbar.set_ticks(np.logspace(np.log10(vmin),np.log10(vmax), 5))
    cbar.minorticks_off()
    # plt.tight_layout()
    plt.savefig('plots/LCOE_Pdens_eta_all_%0.2f_%d_%d.png' % (ir,n,t_out), dpi=300)
    plt.savefig('plots/LCOE_Pdens_eta_all_%0.2f_%d_%d.pdf' % (ir,n,t_out))
    CPE_tot = 9e-5
    CPA = 1
    Pdens_TPV_fixed = 2
    eta_TPV_sweep = np.linspace(0.2, 0.6, 100)
    LCOE_TPV = (CPA*CRF/Pdens_TPV_fixed/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_tot/eta_TPV_sweep) # $/Wh for device contribution
    LCOE = LCOE_TPV + LCOE_sys # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    plt.figure(figsize=(3,2.5))
    plt.plot(eta_TPV_sweep, LCOE_kWh,'b-')
    plt.xlabel('Efficiency',color='b')
    plt.ylabel('LCOE (\\textcent/kWh)')
    plt.twiny()
    Pdens_TPV_sweep = np.linspace(1+1/3,4,100)
    eta_TPV_fixed = 0.3
    LCOE_TPV = (CPA*CRF/Pdens_TPV_sweep/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_tot/eta_TPV_fixed) # $/Wh for device contribution
    LCOE = LCOE_TPV + LCOE_sys # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    plt.plot(Pdens_TPV_sweep, LCOE_kWh,'r-')
    plt.xlabel('Power density (W/cm$^2$)', color='r')
    plt.grid()
    plt.ylim(0,45)
    # plt.title(f'CPE_tot = {CPE_tot:.2e}, CPA = {CPA:.2e}, Pdens_TPV = {Pdens_TPV_fixed}')
    plt.tight_layout()
    plt.savefig('plots/LCOE_eta_%d_%d_%d.pdf' % (ir,n,t_out))

    CPE_tot = 3e-6
    CPA = 70 
    plt.figure(figsize=(3,2.5))
    eta_TPV_sweep = np.linspace(0.2, 0.6, 100)
    Pdens_TPV_fixed = 2
    LCOE_TPV = (CPA*CRF/Pdens_TPV_fixed/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_tot/eta_TPV_sweep) # $/Wh for device contribution
    LCOE = LCOE_TPV + LCOE_sys # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    plt.plot(eta_TPV_sweep, LCOE_kWh,'b-')
    plt.xlabel('Efficiency', color='b')
    plt.ylabel('LCOE (\\textcent/kWh)')
    plt.twiny()
    Pdens_TPV_sweep = np.linspace(1+1/3,4,100)
    eta_TPV_fixed = 0.3
    LCOE_TPV = (CPA*CRF/Pdens_TPV_sweep/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_tot/eta_TPV_fixed) # $/Wh for device contribution
    LCOE = LCOE_TPV + LCOE_sys # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    plt.plot(Pdens_TPV_sweep, LCOE_kWh,'r-')
    plt.xlabel('Power density (W/cm$^2$)',color='r')
    plt.grid()
    plt.ylim(0,45)
    # plt.title(f'CPE_tot = {CPE_tot:.2e}, CPA = {CPA:.2e}, eta_TPV = {eta_TPV_fixed}')
    plt.tight_layout()
    plt.savefig('plots/LCOE_Pdens_%d_%d_%d.pdf' % (ir,n,t_out))

    CPE_tot = 4.5e-5
    CPA = 35
    plt.figure(figsize=(3,2.5))
    eta_TPV_sweep = np.linspace(0.2, 0.6, 100)
    Pdens_TPV_fixed = 2
    LCOE_TPV = (CPA*CRF/Pdens_TPV_fixed/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_tot/eta_TPV_sweep) # $/Wh for device contribution
    LCOE = LCOE_TPV + LCOE_sys # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE  
    plt.plot(eta_TPV_sweep, LCOE_kWh,'b-')
    plt.xlabel('Efficiency', color='b')
    plt.ylabel('LCOE (\\textcent/kWh)')
    plt.twiny()
    Pdens_TPV_sweep = np.linspace(1+1/3,4,100)
    eta_TPV_fixed = 0.3
    LCOE_TPV = (CPA*CRF/Pdens_TPV_sweep/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_tot/eta_TPV_fixed) # $/Wh for device contribution
    LCOE = LCOE_TPV + LCOE_sys # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    plt.plot(Pdens_TPV_sweep, LCOE_kWh,'r-')
    plt.xlabel('Power density (W/cm$^2$)',color='r')
    plt.grid()
    plt.ylim(0,45)
    # plt.title(f'CPE_tot = {CPE_tot:.2e}, CPA = {CPA:.2e}, eta_TPV = {eta_TPV_fixed}')
    plt.tight_layout()
    plt.savefig('plots/LCOE_dual_%d_%d_%d.pdf' % (ir,n,t_out))


    # plt.show()


def sweep_CPE_regime_map():
    CRFs = np.linspace(0.15, 0.04, 4)
    # t_outs = np.logspace(np.log10(1), np.log10(8760), 4)
    # t_outs = np.linspace(1, 8760, 4)
    t_outs = np.array([1,450,2000,8760])
    C1_min = min(CRFs)/max(t_outs)
    C1_max = max(CRFs)/min(t_outs)
    C1s = np.linspace(C1_min, C1_max, 100)
    LCOH = np.linspace(0.1, 10, 100) # ¢/kWh-th
    CPA = np.linspace(0.5,70,100) # $/cm2
    C1 = 0.074/8760 # 1/hr
    LCOEs = []
    Pdens_TPV = 2 # W/cm^2
    eta_TPV = 0.3 # 
    for i, LCOH_i in enumerate(LCOH):
        for j, CPA_i in enumerate(CPA):
            LCOE_base = CPA_i*C1/Pdens_TPV*1000*100 + LCOH_i/eta_TPV # ¢/kWh
            LCOE_Pdens = CPA_i*C1/(Pdens_TPV*2)*1000*100 + LCOH_i/eta_TPV # ¢/kWh
            LCOE_eta = CPA_i*C1/Pdens_TPV*1000*100 + LCOH_i/(eta_TPV*2) # ¢/kWh
            LCOE_Pdens_ratio = LCOE_Pdens/LCOE_base
            LCOE_eta_ratio = LCOE_eta/LCOE_base
            LCOEs.append(LCOE_eta_ratio/LCOE_Pdens_ratio)
    LCOEs = np.array(LCOEs).reshape(100,100)
    fig = plt.figure(figsize=(4.5,4))
    cf = plt.contourf(CPA, LCOH, LCOEs, levels=np.logspace(np.log10(0.5), np.log10(2),2000), cmap='coolwarm', norm=colors.LogNorm())
    cb = plt.colorbar()
    cb.set_label('LCOE$_{2\eta}$/LCOE$_{2P}$', rotation=270, labelpad=15)
    cb.set_ticks(np.linspace(0.5,2,7))
    plt.contour(CPA, LCOH, LCOEs, levels=[0.9,1.11], colors='k', linewidths=0.5, linestyles='dashed')
    for c in cf.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.xlabel('CPA$_{TPV}$ (\\$/cm$^2$)')
    plt.ylabel('LCOH (\\textcent/kWh-th)')
    plt.annotate('Efficiency-limited', xy=(10, 8))
    plt.annotate('Power density-limited', xy=(37.5, 1.25))
    plt.annotate('Dual-limited', xy=(42, 6))
    # cb.ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('plots/LCOE_regime_map_C1_%0.2e.pdf' % C1)

    fig = plt.figure(figsize=(10,8.9))

    for i,CRF in enumerate(CRFs):
        for j,t_out in enumerate(t_outs):
            print(CRF, t_out)
            C1 = CRF/t_out
            LCOEs = []
            Pdens_TPV = 2 # W/cm^2
            eta_TPV = 0.3 # 
            for LCOH_i in LCOH:
                for CPA_i in CPA:
                    LCOE_base = CPA_i*C1/Pdens_TPV*1000*100 + LCOH_i/eta_TPV # ¢/kWh
                    LCOE_Pdens = CPA_i*C1/(Pdens_TPV*2)*1000*100 + LCOH_i/eta_TPV # ¢/kWh
                    LCOE_eta = CPA_i*C1/Pdens_TPV*1000*100 + LCOH_i/(eta_TPV*2) # ¢/kWh
                    LCOE_Pdens_ratio = LCOE_Pdens/LCOE_base
                    LCOE_eta_ratio = LCOE_eta/LCOE_base
                    LCOEs.append(LCOE_eta_ratio/LCOE_Pdens_ratio)
            LCOEs = np.array(LCOEs).reshape(100,100)
            # fig = plt.figure(figsize=(4.5,4))
            plt.subplot(4,4,4*i+j+1)
            cf = plt.contourf(CPA, LCOH, LCOEs, levels=np.logspace(np.log10(0.5), np.log10(2),200), cmap='coolwarm', norm=colors.LogNorm())
            plt.contour(CPA, LCOH, LCOEs, levels=[0.9,1.11], colors='k', linewidths=0.5, linestyles='dashed')
            # plt.title(f'CRF = {CRF:.2f}, t_out = {t_out}')
            for c in cf.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
            if i < 3:
                plt.xticks([])
            if j > 0:
                plt.yticks([])
            # plt.xlabel('CPA$_{TPV}$ (\\$/cm$^2$)')
            # plt.ylabel('LCOH (\\textcent/kWh-th)')
            # plt.annotate('Efficiency-limited', xy=(10, 8))
            # plt.annotate('Power density-limited', xy=(37.5, 1.25))
            # plt.annotate('Dual-limited', xy=(42, 6))
            # cb.ax.set_yticklabels([])
            plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.925, 0.043, 0.025, 0.93])
    cb = fig.colorbar(cf, cax=cbar_ax)
    # cb = plt.colorbar()
    cb.set_label('LCOE$_{2\eta}$/LCOE$_{2P}$', rotation=270, labelpad=15)
    cb.set_ticks(np.linspace(0.5,2,7))
    # plt.tight_layout()
    plt.savefig('plots/LCOE_regime_map_C1s.pdf' % C1)

def sweep_CPE_LCOH():
    # CPE_th_input = np.array([0,0.1,1,10]) # c/kWh-th
    CPE_th_input = np.linspace(0,10,100) # c/kWh-th
    # CPE_th_sys = np.array([0,1,10,100]) # c/kWh-th
    CPE_th_sys = np.linspace(0,300,100) # c/kWh-th
    CRF = np.linspace(0.04, 0.15, 4)
    fig = plt.figure(figsize=(10,3))
    for index,i in enumerate(CRF):
        LCOEs = []
        for j in CPE_th_sys:
            for k in CPE_th_input:
                LCOE = i*j + k
                LCOEs.append(LCOE)
        LCOEs = np.array(LCOEs).reshape(100,100)
        plt.subplot(1,4,index+1)
        cf = plt.contourf(CPE_th_input, CPE_th_sys, LCOEs, levels=np.linspace(0,60,100), cmap='plasma')
        plt.contour(CPE_th_input, CPE_th_sys, LCOEs, levels=10, colors='k', linewidths=0.5, linestyles='solid')
        for c in cf.collections:
            c.set_edgecolor("face")
            c.set_linewidth(0.000000000001)
        if index > 0:
            plt.yticks([])
        plt.xticks([0,2,4,6,8,10], ['0','2','4','6','8','10'])
        plt.xlabel('CPE$_{input}$ (\\textcent/kWh-th)')
        if index == 0:
            plt.ylabel('CPE$_{sys}$ (\\textcent/kWh-th)')
        plt.title(f'CRF = {i:.2f}')
        plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.925, 0.043, 0.025, 0.93])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label('LCOH (\\textcent/kWh)', rotation=270, labelpad=15)
    plt.savefig('plots/LCOH_CPEs_CRF.pdf')

    

def sweep_CPE_LCOE_2(Pdens_TPV, eta_TPV):
    CRF = np.linspace(0.04, 0.15, 100)
    t_out = np.linspace(1, 8760, 100)
    C1_min = min(CRF)/max(t_out)
    C1_max = max(CRF)/min(t_out)
    C1 = np.linspace(C1_min, C1_max, 100)
    LCOH = np.linspace(0.1, 10, 100) # ¢/kWh-th
    CPA = np.linspace(0.5,70,100) # $/cm2
    C1 = 0.074/8760 # 1/hr
    LCOEs = []
    # Pdens_TPV = 5 # W/cm^2
    # eta_TPV = 0.5 # 
    for i, LCOH_i in enumerate(LCOH):
        for j, CPA_i in enumerate(CPA):
            LCOE_base = CPA_i*C1/Pdens_TPV*1000*100 + LCOH_i/eta_TPV # ¢/kWh
            LCOE_Pdens = CPA_i*C1/(Pdens_TPV*2)*1000*100 + LCOH_i/eta_TPV # ¢/kWh
            LCOE_eta = CPA_i*C1/Pdens_TPV*1000*100 + LCOH_i/(eta_TPV*2) # ¢/kWh
            LCOE_Pdens_ratio = LCOE_Pdens/LCOE_base
            LCOE_eta_ratio = LCOE_eta/LCOE_base
            LCOEs.append(LCOE_base)
    LCOEs = np.array(LCOEs).reshape(100,100)
    # fig = plt.figure(figsize=(4.6,4))
    fig = plt.figure(figsize=(3.6,3))
    cf = plt.contourf(CPA, LCOH, LCOEs, levels=np.linspace(0.25,62.5,2000), cmap='plasma')
    cb = plt.colorbar(ticks = np.append(0.5,np.arange(0,65,5)))
    # cb = plt.colorbar()
    cb.set_label('LCOE (\\textcent/kWh)', rotation=270, labelpad=15)
    # cb.set_ticks(np.linspace(0.5,60,12))
    # plt.xlim(0,70)
    # plt.ylim(0,10)
    manual_locations = []
    for LCOE in [2.5, 5, 7.5, 10]:
        # find CPA when LCOH = 0.1
        CPA_01 = CPA[np.abs(LCOEs[0,:] - LCOE).argmin()]/2
        # find LCOH when CPA = 0.5
        LCOH_05 = LCOH[np.abs(LCOEs[:,0] - LCOE).argmin()]/2
        manual_locations.append((CPA_01, LCOH_05))
    # print(manual_locations)
    cs = plt.contour(CPA, LCOH, LCOEs, levels=[2.5, 5, 7.5, 10], colors='w', linewidths=0.5, linestyles='solid')
    plt.clabel(cs, inline=True, fontsize=8, fmt='%0.1f', colors='w', manual=manual_locations)
    for c in cf.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.xlabel('CPA$_{TPV}$ (\\$/cm$^2$)')
    plt.ylabel('LCOH (\\textcent/kWh-th)')
    plt.title('$\eta_{TPV}$ = %d' % (eta_TPV*100) + '\%' + ', $P_{dens,TPV}$ = %d' % Pdens_TPV + ' W/cm$^2$', fontsize=10)
    # plt.annotate('Efficiency-limited', xy=(10, 8))
    # plt.annotate('Power density-limited', xy=(37.5, 1.25))
    # plt.annotate('Dual-limited', xy=(42, 6))
    # cb.ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig('plots/LCOE_base_regime_map_C1_%0.2e_Pdens_%0.2f_eta_%0.2f.pdf' % (C1, Pdens_TPV, eta_TPV))


def sweep_CPEs_2(ir, n, t_out, CPA_TPV):
    CRF = (ir*(1+ir)**n)/((1+ir)**n-1) # capital recovery factor
    CPE_sys = np.logspace(-3,-5,3)
    CPE_energy = np.logspace(-6,-4,3)
    # CPA_TPV = np.array([0.5,1,5])

    # CPE_tots = []
    # for i,CPE_th_sys in enumerate(CPE_sys):
    #     for j,CPE_th_input in enumerate(CPE_energy):
    #         CPE_tot = CPE_th_sys*CRF + CPE_th_input
    #         CPE_tots.append(CPE_tot)
    
    # CPE_tots = np.logspace(np.log10(min(CPE_tots)), np.log10(max(CPE_tots)), 4)
    # CPE_tots = np.logspace(-6,-4,3)
    fig = plt.figure(figsize=(15,10))
    vmin = np.inf
    vmax = -np.inf
    for i, CPE_th_sys in enumerate(CPE_sys):
        for j, CPE_th_input in enumerate(CPE_energy):
            # print(CPE_th_sys, CPE_th_input, CPA)
            # t_out = 8760
        
            LCOE_TPV = (CPA_TPV*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
            LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
            LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
            # LCOE_sys = (CPE_tot/eta_TPV) # $/Wh for device contribution
            # LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
            LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
            LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
            vmin = min(vmin, np.min(LCOE_MWh))
            vmax = max(vmax, np.max(LCOE_MWh))

    for i, CPE_th_sys in enumerate(CPE_sys):
        for j, CPE_th_input in enumerate(CPE_energy):
            # print(CPE_th_sys, CPE_th_input, CPA)
            # t_out = 8760
        
            LCOE_TPV = (CPA_TPV*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
            LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
            LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
            # LCOE_sys = (CPE_tot/eta_TPV) # $/Wh for device contribution
            # LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
            LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
            LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
            # print(Pdens_TPV, eta_TPV, LCOE_MWh)
            # exit()
            # reshape LCOE_MWh to 2D
            # 2D plot of LCOE vs Pdens_TPV and eta_TPV
            # plt.figure(figsize=(6,4))
            # on sublot i,j plot LCOE_MWh
            ax = fig.add_subplot(4,4,4*i+j+1)
            # color contour lines
            # plt.plot(TPV_data_2200_ideal_10['Pgen'], TPV_data_2200_ideal_10['eff'], 'k-o', label='TPV')
            # plt.plot(TPV_data_2200_real_10['Pgen'], TPV_data_2200_real_10['eff'], 'k--s', label='TPV')
            vmin = 1
            vmax = 1000
            plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
            ctf = plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=np.logspace(np.log10(vmin),np.log10(vmax), 200), locator=ticker.LogLocator(), cmap='plasma', norm=colors.LogNorm(), vmin=vmin, vmax=vmax)
            # ctf.set_edgecolor("face")
            # plt.colorbar()
            # plt.xlabel('Power density (W/cm$^2$)')
            # plt.ylabel('Efficiency')
            cbar = plt.colorbar()
            # cbar.set_ticks([1, 10, 100, 1000])
            cbar.set_ticks(10**np.arange(np.floor(np.log10(vmin)), np.ceil(np.log10(vmax))+1))
            cbar.minorticks_off()
            plt.xlim(0.5, 10)
            plt.ylim(0.2, 1)
            # plt.title(f'CPE_tot = {CPE_tot:.2e}, CPA = {CPA:.2e}')
            plt.tight_layout()

    plt.savefig('plots/LCOE_Pdens_eta_all_CPA5_%0.2f_%d_%d_%0.2f.png' % (ir,n,t_out,CPA_TPV), dpi=300)
    plt.savefig('plots/LCOE_Pdens_eta_all_CPA5_%0.2f_%d_%d_%0.2f.pdf' % (ir,n,t_out,CPA_TPV))
    # plt.show()


def cost_map():
    fig = plt.figure(figsize=(3,6))
    sTPV_CPE = np.array([0.001902587519025875, 3e-7])*1e5
    waste_heat_CPE = np.array([3.159018264840183e-05, 3e-7])*1e5
    th_stor_CPE = np.array([0.00012049720953830543, 3.3333333333333335e-05])*1e5
    port_power_CPE = np.array([0.0005513372472276582, 5.857709049241056e-05])*1e5
    h2_comb_CPE = np.array([0.0001297564687975647, 0.00016666666666666666])*1e5
    systems = [sTPV_CPE, waste_heat_CPE, th_stor_CPE, port_power_CPE, h2_comb_CPE]
    colors = ['y', 'purple', 'b', 'g', 'r']
    labels=['Solar TPV','Waste heat','Thermal storage','Portable power','Power plant']
    for index, system in enumerate(systems):
        plt.loglog(system[1], system[0], 's', color = colors[index], markersize=10, markeredgecolor='k', markeredgewidth=0.5, clip_on=False, zorder=100, label=labels[index])
    plt.xlim(3e-2, 3e1)
    plt.ylim(3e-1, 3e2)
    # for tick in plt.gca().yaxis.get_major_ticks():
    #     tick.label.set_position((-0.02,0))
    plt.xticks([1e-1, 1e0, 1e1], labels=['0.1', '1', '10'])
    plt.yticks([1e0, 1e1, 1e2], labels=['1', '10', '100'])
    plt.xlabel('CPE$_{th,input}$ (\\textcent/kWh-th)')
    plt.ylabel('CPE$_{th,system}$ (\\textcent/kWh-th)')
    # make square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=[0.36,-0.75])
    plt.tight_layout()
    plt.savefig('plots/cost_map.pdf')
    plt.savefig('plots/cost_map.png', dpi=300)
    # plt.show()
    # plt.xlabel('CPE_th')


def specific_system(CPE_energy_i, CPE_sys_i, CF):
    # specific system
    for df in [TPV_data_2200_ideal, TPV_data_2200_real, TPV_data_2200_real_imp_ref, TPV_data_2200_real_imp_eqe, TPV_data_2200_real_imp_resis, TPV_data_2200_real_imp_ABR, TPV_data_2200_real_imp_T, TPV_data_2200_real_imp_VF, TPV_data_2200_real_imp_emis]:
        print(df.Name)
        t_out = CF*8760 # hours in a year
        # eta_eth = 0.9 # energy to heat conversion efficiency
        Pdens_TPV = np.array(df['Pgen'])
        eta_TPV = np.array(df['eff'])
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
        LCOE_sys = (CPE_sys_i*CRF/eta_TPV) # $/Wh for device contribution
        LCOE_fuel = (CPE_energy_i/eta_TPV) # $/Wh for fuel contribution
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
        LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
        # print max LCOE and corresponding Pdens_TPV and eta_TPV
        min_LCOE = np.min(LCOE_MWh)
        min_LCOE_index = np.where(LCOE_MWh == min_LCOE)
        # print(min_LCOE_index)
        min_LCOE_Pdens_TPV = Pdens_TPV[min_LCOE_index]
        min_LCOE_eta_TPV = eta_TPV[min_LCOE_index]
        print('min LCOE: %0.2f, Pdens_TPV: %0.2f, eta_TPV: %0.2f' % (min_LCOE, min_LCOE_Pdens_TPV[0], min_LCOE_eta_TPV[0]))
        # print(min_LCOE, min_LCOE_Pdens_TPV[0], min_LCOE_eta_TPV[0])

def specific_system_LCOH_CPA(LCOH, CPA, CF):
    t_out = CF*8760 # hours in a year
    ir = 0.04
    n = 20
    CRF = (ir*(1+ir)**n)/((1+ir)**n-1) # capital recovery factor
    Pdens_TPV = np.meshgrid(np.linspace(0.5, 10, 100), np.linspace(0.2, 1, 100))[0] # W/cm^2
    eta_TPV = np.meshgrid(np.linspace(0.5, 10, 100), np.linspace(0.2, 1, 100))[1] # efficiency of TPV
    LCOE_heat = LCOH/eta_TPV # $/Wh for heat contribution
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE = LCOE_TPV + LCOE_heat # $/Wh total LCOE
    LCOE_kWh = LCOE*1e6/1e3*1e2 # cents/kWh total LCOE
    plt.figure(figsize=(5,4))
    plt.contour(Pdens_TPV, eta_TPV, LCOE_kWh, levels=10, colors='k', linewidths=0.5)
    cf = plt.contourf(Pdens_TPV, eta_TPV, LCOE_kWh, levels=np.linspace(2,20,1000), cmap='coolwarm')
    for c in cf.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(2,21,2))
    cbar.set_label('LCOE (\\textcent/kWh)')
    plt.xlabel('TPV power density (W/cm$^2$)')
    plt.ylabel('TPV efficiency')
    plt.title('LCOH = %0.0f\\textcent/kWh, CPA = $\\$$%0.0f/cm$^2$' % (LCOH*1e3*1e2, CPA))
    plt.tight_layout()
    plt.savefig('plots/LCOE_Pdens_eta_LCOH_%0.0f_CPA_%0.0f.pdf' % (LCOH*1e3*1e2, CPA))


def individual_plot(CPE_th_sys, CPE_th_input, CF, filename):
    plt.figure()
    t_out = 8760*CF
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
    LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
    # LCOE_sys = (CPE_tot/eta_TPV) # $/Wh for device contribution
    # LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    # print(Pdens_TPV, eta_TPV, LCOE_MWh)
    # exit()
    # reshape LCOE_MWh to 2D
    # 2D plot of LCOE vs Pdens_TPV and eta_TPV
    # plt.figure(figsize=(6,4))
    # on sublot i,j plot LCOE_MWh
    # color contour lines
    plt.plot(TPV_data_2200_ideal_10['Pgen'], TPV_data_2200_ideal_10['eff'], 'k-o', label='ideal TPV')
    plt.plot(TPV_data_2200_real_10['Pgen'], TPV_data_2200_real_10['eff'], 'k--s', label='real TPV')
    plt.contour(Pdens_TPV, eta_TPV, LCOE_kWh, levels=20, colors='k', linewidths=0.5)
    cf = plt.contourf(Pdens_TPV, eta_TPV, LCOE_kWh, levels=np.logspace(np.log10(0.1),np.log10(100), 200), locator=ticker.LogLocator(), cmap='plasma', norm=colors.LogNorm(), vmin=0.1, vmax=100)
    for c in cf.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    # plt.legend(loc=[0,-0.5], ncol=2)
    # cb = plt.colorbar()
    # cb.set_label('LCOE (\$/MWh)')
    cbar = plt.colorbar()
    cbar.set_ticks([0.1, 1, 10, 100])
    cbar.minorticks_off()
    cbar.set_label('LCOE (\\textcent/kWh)')
    plt.xlabel('Power density (W/cm$^2$)')
    plt.ylabel('Efficiency')
    plt.title(filename.replace('_',' '))
    plt.xlim(0.5, 10)
    plt.ylim(0.2, 1)
    # plt.title(f'CPE_tot = {CPE_tot:.2e}, CPA = {CPA:.2e}')
    plt.tight_layout()
    plt.savefig('plots/LCOE_Pdens_eta_'+filename+'.pdf')
    plt.savefig('plots/LCOE_Pdens_eta_'+filename+'.png', dpi=300)
    pass

def individual_plot_no_TPV(CPE_th_sys, CPE_th_input, CF, filename):
    plt.figure()
    t_out = 8760*CF
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
    LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
    # LCOE_sys = (CPE_tot/eta_TPV) # $/Wh for device contribution
    # LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    # print(Pdens_TPV, eta_TPV, LCOE_MWh)
    # exit()
    # reshape LCOE_MWh to 2D
    # 2D plot of LCOE vs Pdens_TPV and eta_TPV
    # plt.figure(figsize=(6,4))
    # on sublot i,j plot LCOE_MWh
    # color contour lines
    # plt.plot(TPV_data_2200_ideal_10['Pgen'], TPV_data_2200_ideal_10['eff'], 'k-o', label='ideal TPV')
    # plt.plot(TPV_data_2200_real_10['Pgen'], TPV_data_2200_real_10['eff'], 'k--s', label='real TPV')
    plt.contour(Pdens_TPV, eta_TPV, LCOE_kWh, levels=20, colors='k', linewidths=0.5)
    cf = plt.contourf(Pdens_TPV, eta_TPV, LCOE_kWh, levels=np.logspace(np.log10(0.1),np.log10(100), 200), locator=ticker.LogLocator(), cmap='plasma', norm=colors.LogNorm(), vmin=0.1, vmax=100)
    for c in cf.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    # plt.legend(loc=[0,-0.5], ncol=2)
    # cb = plt.colorbar()
    # cb.set_label('LCOE (\$/MWh)')
    cbar = plt.colorbar()
    cbar.set_ticks([0.1, 1, 10, 100])
    cbar.minorticks_off()
    cbar.set_label('LCOE (\\textcent/kWh)')
    plt.xlabel('Power density (W/cm$^2$)')
    plt.ylabel('Efficiency')
    plt.title(filename.replace('_',' '))
    plt.xlim(0.5, 10)
    plt.ylim(0.2, 1)
    # plt.title(f'CPE_tot = {CPE_tot:.2e}, CPA = {CPA:.2e}')
    plt.tight_layout()
    plt.savefig('plots/LCOE_Pdens_eta_'+filename+'_noTPV.pdf')
    plt.savefig('plots/LCOE_Pdens_eta_'+filename+'_noTPV.png', dpi=300)
    pass


# def TEGS():
#     # TEGS
#     CPE_energy_i = 3e-5 # $/Wh cost per energy of fuel
#     CPE_sys_i = 2.1689497716894976e-05 # $/Wh cost per energy of system
#     t_out = 0.8*8760 # hours in a year
#     eta_eth = 0.9 # energy to heat conversion efficiency
#     Pdens_TPV = np.array(TPV_data_2200_ideal['Pgen']/4)
#     eta_TPV = np.array(TPV_data_2200_ideal['eff'])
#     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
#     CPE_storage = 20e-3 # $/Wh cost per energy of device
#     storage_energy = 1e9 # Wh energy of device
#     P_in = 250e6 # W
#     CPP_dis_not_TPV = 0.42 # $/W cost per power of discharging (not incl TPV)
#     dis_power = 0.2*P_in # W power of discharging
#     CPP_ch = 0.03 # $/W cost per power of charging
#     LCOE_sys = ((CPE_storage*storage_energy+CPP_ch*P_in+CPP_dis_not_TPV*dis_power*eta_TPV)*CRF/P_in/eta_eth/eta_TPV/t_out) # $/Wh for device 
#     LCOE_fuel = (CPE_energy_i/eta_eth/eta_TPV) # $/Wh for fuel contribution
#     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
#     LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
#     # print max LCOE and corresponding Pdens_TPV and eta_TPV
#     min_LCOE = np.min(LCOE_MWh)
#     min_LCOE_index = np.where(LCOE_MWh == min_LCOE)
#     print(min_LCOE_index)
#     min_LCOE_Pdens_TPV = Pdens_TPV[min_LCOE_index]
#     min_LCOE_eta_TPV = eta_TPV[min_LCOE_index]
#     print(min_LCOE, min_LCOE_Pdens_TPV, min_LCOE_eta_TPV)
#     P_dens_TPV = np.array(TPV_data_2200_real['Pgen']/4)
#     eta_TPV = np.array(TPV_data_2200_real['eff'])
#     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
#     LCOE_sys = (CPE_sys_i*CRF/eta_eth/eta_TPV) # $/Wh for device contribution
#     LCOE_fuel = (CPE_energy_i/eta_eth/eta_TPV) # $/Wh for fuel contribution
#     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
#     LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
#     # print max LCOE and corresponding Pdens_TPV and eta_TPV
#     min_LCOE = np.min(LCOE_MWh)
#     min_LCOE_index = np.where(LCOE_MWh == min_LCOE)
#     print(min_LCOE_index)
#     min_LCOE_Pdens_TPV = Pdens_TPV[min_LCOE_index]
#     min_LCOE_eta_TPV = eta_TPV[min_LCOE_index]
#     print(min_LCOE, min_LCOE_Pdens_TPV, min_LCOE_eta_TPV)

def sweep_cell_improvements(CPE_th_sys, CPE_th_input):
    SBRs = np.linspace(1, 0.9, 10)
    SBR_LCOEs = []
    SBR_Pdens_TPV = []
    SBR_eta_TPV = []
    SBR_BG_TPV = []
    SBR_LCOEs_ideal = []
    
    # print('SBR calcs')
    for SBR in SBRs:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=2123, SBR=SBR, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=SBR, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=0.9, BW=1)
        TPV_data_2200_ideal_imp = TPV_data_2200_ideal_imp.iloc[::10, :]
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
        LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
        LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
        LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
        min_LCOE = np.min(LCOE_kWh)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        SBR_LCOEs.append(min_LCOE)
        SBR_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        SBR_eta_TPV.append(min_LCOE_eta_TPV[0])
        SBR_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out) # $/Wh for TPV contribution
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal) # $/Wh for device contribution
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal) # $/Wh for fuel contribution
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal # $/Wh total LCOE
        LCOE_MWh_ideal = LCOE_ideal*1e6 # $/MWh total LCOE
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2 # cents/kWh total LCOE
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        SBR_LCOEs_ideal.append(min_LCOE_ideal)
    
    # plt.show()
    # print('NRR calcs')
    NRRs = np.linspace(1, 20, 10)
    NRRs = np.append(NRRs, np.arange(21,25,1))
    NRRs = np.insert(NRRs, 0, 0)
    NRR_LCOEs = []
    NRR_Pdens_TPV = []
    NRR_eta_TPV = []
    NRR_BG_TPV = []
    NRR_LCOEs_ideal = []
    for NRR in NRRs:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=NRR, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=NRR, series_R=0, VF=0.31, emis=0.9, BW=1)
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
        LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
        LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
        LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
        min_LCOE = np.min(LCOE_kWh)
        NRR_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        NRR_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        NRR_eta_TPV.append(min_LCOE_eta_TPV[0])
        NRR_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out) # $/Wh for TPV contribution
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal) # $/Wh for device contribution
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal) # $/Wh for fuel contribution
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal # $/Wh total LCOE
        LCOE_MWh_ideal = LCOE_ideal*1e6 # $/MWh total LCOE
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2 # cents/kWh total LCOE
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        NRR_LCOEs_ideal.append(min_LCOE_ideal)
    
    # plt.show()
    # print('Rseries calcs')
    Rseries_all = np.logspace(-6, -2, 10)
    Rseries_all = np.append(Rseries_all, 0.013)
    Rseries_LCOEs = []
    Rseries_Pdens_TPV = []
    Rseries_eta_TPV = []
    Rseries_BG_TPV = []
    R_series_LCOEs_ideal = []
    for Rseries in Rseries_all:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=Rseries, VF=0.31, emis=0.85, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=Rseries, VF=0.31, emis=0.9, BW=1)
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
        LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
        LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
        LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
        min_LCOE = np.min(LCOE_kWh)
        Rseries_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        Rseries_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        Rseries_eta_TPV.append(min_LCOE_eta_TPV[0])
        Rseries_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out) # $/Wh for TPV contribution
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal) # $/Wh for device contribution
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal) # $/Wh for fuel contribution
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal # $/Wh total LCOE
        LCOE_MWh_ideal = LCOE_ideal*1e6 # $/MWh total LCOE
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2 # cents/kWh total LCOE
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        R_series_LCOEs_ideal.append(min_LCOE_ideal)
    
    # print('ABR calcs')
    ABRs = np.linspace(0, 0.5, 10)
    ABRs = np.append(ABRs, np.array([0.55,0.6]))
    ABR_LCOEs = []
    ABR_Pdens_TPV = []
    ABR_eta_TPV = []
    ABR_BG_TPV = []
    ABR_LCOEs_ideal = []
    for ABR in ABRs:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=ABR, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=1, ABR=ABR, nonrad_ratio=0, series_R=0, VF=0.31, emis=0.9, BW=1)
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
        LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
        LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
        LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
        min_LCOE = np.min(LCOE_kWh)
        ABR_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        ABR_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        ABR_eta_TPV.append(min_LCOE_eta_TPV[0])
        ABR_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out) # $/Wh for TPV contribution
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal) # $/Wh for device contribution
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal) # $/Wh for fuel contribution
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal # $/Wh total LCOE
        LCOE_MWh_ideal = LCOE_ideal*1e6 # $/MWh total LCOE
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2 # cents/kWh total LCOE
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        ABR_LCOEs_ideal.append(min_LCOE_ideal)

    TPV_data_2200_ideal = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=0.9, BW=1)
    TPV_data_2200_ideal = TPV_data_2200_ideal.iloc[::10, :]
    Pdens_TPV = np.array(TPV_data_2200_ideal['Pgen'])
    eta_TPV = np.array(TPV_data_2200_ideal['eff'])
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
    LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
    LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    min_LCOE_ideal = np.min(LCOE_kWh)

    TPV_data_2200_real = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
    # TPV_data_2200_real = TPV_data_2200_real.iloc[::10, :]
    Pdens_TPV = np.array(TPV_data_2200_real['Pgen'])
    eta_TPV = np.array(TPV_data_2200_real['eff'])
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
    LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
    LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2 # cents/kWh total LCOE
    min_LCOE_real = np.min(LCOE_kWh)
    min_LCOE_index = np.where(LCOE_kWh == min_LCOE_real)
    min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
    min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
    min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]


    fig = plt.figure(figsize=(10,2.5))
    fig.add_subplot(141)
    plt.plot(SBRs, SBR_LCOEs, 'b-')
    plt.plot(SBRs, SBR_LCOEs_ideal, 'g-')
    plt.plot(1, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(0.95, min_LCOE_real, 'bo-', markersize=2.5)
    plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.gca().invert_xaxis()
    plt.xlabel('SBR')
    plt.ylabel('LCOE (\\textcent/kWh)')
    fig.add_subplot(142)
    plt.plot(NRRs, NRR_LCOEs, 'b-')
    plt.plot(NRRs, NRR_LCOEs_ideal, 'g-')
    plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(12, min_LCOE_real, 'bo-', markersize=2.5)
    plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.xlim(-1, 25)
    plt.xticks([0, 12, 24])
    plt.xlabel('NRR')
    plt.ylabel('LCOE (\\textcent/kWh)')
    fig.add_subplot(143)
    plt.plot(Rseries_all, Rseries_LCOEs, 'b-')
    plt.plot(Rseries_all, R_series_LCOEs_ideal, 'g-')
    plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(6.5e-3, min_LCOE_real, 'bo-', markersize=2.5)
    plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.xlim(-5e-4, 13.5e-3)
    plt.xticks([0, 6.5e-3, 13e-3],['0','0.0065','0.013'])
    plt.xlabel('Rseries')
    plt.ylabel('LCOE (\\textcent/kWh)')
    fig.add_subplot(144)
    plt.plot(ABRs, ABR_LCOEs, 'b-')
    plt.plot(ABRs, ABR_LCOEs_ideal, 'g-')
    plt.plot(0.3, min_LCOE_real, 'bo-', markersize=2.5, label='real cell')
    plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5, label='ideal cell')
    plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.xlim(-0.025, 0.6+0.025)
    plt.xticks([0, 0.3, 0.6])
    plt.xlabel('ABR')
    plt.ylabel('LCOE (\\textcent/kWh)')
    plt.legend(loc='lower right')
    plt.tight_layout()
 
    plt.savefig('plots/LCOE_cell_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))

    fig = plt.figure(figsize=(10,2.5))
    fig.add_subplot(141)
    plt.plot(SBRs, SBR_BG_TPV, 'b-')
    # plt.plot(SBRs, SBR_LCOEs_ideal, 'g-')
    # plt.plot(1, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(0.95, min_LCOE_BG_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1.3)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.gca().invert_xaxis()
    plt.xlabel('SBR')
    plt.ylabel('Optimal bandgap (eV)')
    fig.add_subplot(142)
    plt.plot(NRRs, NRR_BG_TPV, 'b-')
    # plt.plot(NRRs, NRR_LCOEs_ideal, 'g-')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(12, min_LCOE_BG_TPV, 'bo-', markersize=2.5)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 1.3)
    plt.xlim(-1, 25)
    plt.xticks([0, 12, 24])
    plt.xlabel('NRR')
    plt.ylabel('Optimal bandgap (eV)')
    fig.add_subplot(143)
    plt.plot(Rseries_all, Rseries_BG_TPV, 'b-')
    # plt.plot(Rseries_all, R_series_LCOEs_ideal, 'g-')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(6.5e-3, min_LCOE_BG_TPV, 'bo-', markersize=2.5)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 1.3)
    plt.xlim(-5e-4, 13.5e-3)
    plt.xticks([0, 6.5e-3, 13e-3],['0','0.0065','0.013'])
    plt.xlabel('Rseries')
    plt.ylabel('Optimal bandgap (eV)')
    fig.add_subplot(144)
    plt.plot(ABRs, ABR_BG_TPV, 'b-')
    # plt.plot(ABRs, ABR_LCOEs_ideal, 'g-')
    plt.plot(0.3, min_LCOE_BG_TPV, 'bo-', markersize=2.5, label='real cell')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5, label='ideal cell')
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 1.3)
    plt.xlim(-0.025, 0.6+0.025)
    plt.xticks([0, 0.3, 0.6])
    plt.xlabel('ABR')
    plt.ylabel('Optimal bandgap (eV)')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig('plots/BG_cell_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))

    fig = plt.figure(figsize=(10,2.5))
    fig.add_subplot(141)
    plt.plot(SBRs, SBR_eta_TPV, 'b-')
    # plt.plot(SBRs, SBR_LCOEs_ideal, 'g-')
    # plt.plot(1, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(0.95, min_LCOE_eta_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.gca().invert_xaxis()
    plt.xlabel('SBR')
    plt.ylabel('Efficiency')
    fig.add_subplot(142)
    plt.plot(NRRs, NRR_eta_TPV, 'b-')
    # plt.plot(NRRs, NRR_LCOEs_ideal, 'g-')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(12, min_LCOE_eta_TPV, 'bo-', markersize=2.5)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 1)
    plt.xlim(-1, 25)
    plt.xticks([0, 12, 24])
    plt.xlabel('NRR')
    plt.ylabel('Efficiency')
    fig.add_subplot(143)
    plt.plot(Rseries_all, Rseries_eta_TPV, 'b-')
    # plt.plot(Rseries_all, R_series_LCOEs_ideal, 'g-')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(6.5e-3, min_LCOE_eta_TPV, 'bo-', markersize=2.5)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 1)
    plt.xlim(-5e-4, 13.5e-3)
    plt.xticks([0, 6.5e-3, 13e-3],['0','0.0065','0.013'])
    plt.xlabel('Rseries')
    plt.ylabel('Efficiency')
    fig.add_subplot(144)
    plt.plot(ABRs, ABR_eta_TPV, 'b-')
    # plt.plot(ABRs, ABR_LCOEs_ideal, 'g-')
    plt.plot(0.3, min_LCOE_eta_TPV, 'bo-', markersize=2.5, label='real cell')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5, label='ideal cell')
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 1)
    plt.xlim(-0.025, 0.6+0.025)
    plt.xticks([0, 0.3, 0.6])
    plt.xlabel('ABR')
    plt.ylabel('Efficiency')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig('plots/eta_cell_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))

    fig = plt.figure(figsize=(10,2.5))
    fig.add_subplot(141)
    plt.plot(SBRs, SBR_Pdens_TPV, 'b-')
    # plt.plot(SBRs, SBR_LCOEs_ideal, 'g-')
    # plt.plot(1, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(0.95, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 6)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.gca().invert_xaxis()
    plt.xlabel('SBR')
    plt.ylabel('Power density (W/cm$^2$)')
    fig.add_subplot(142)
    plt.plot(NRRs, NRR_Pdens_TPV, 'b-')
    # plt.plot(NRRs, NRR_LCOEs_ideal, 'g-')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(12, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 6)
    plt.xlim(-1, 25)
    plt.xticks([0, 12, 24])
    plt.xlabel('NRR')
    plt.ylabel('Power density (W/cm$^2$)')
    fig.add_subplot(143)
    plt.plot(Rseries_all, Rseries_Pdens_TPV, 'b-')
    # plt.plot(Rseries_all, R_series_LCOEs_ideal, 'g-')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5)
    plt.plot(6.5e-3, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5)
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 6)
    plt.xlim(-5e-4, 13.5e-3)
    plt.xticks([0, 6.5e-3, 13e-3],['0','0.0065','0.013'])
    plt.xlabel('Rseries')
    plt.ylabel('Power density (W/cm$^2$)')
    fig.add_subplot(144)
    plt.plot(ABRs, ABR_Pdens_TPV, 'b-')
    # plt.plot(ABRs, ABR_LCOEs_ideal, 'g-')
    plt.plot(0.3, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5, label='real cell')
    # plt.plot(0, min_LCOE_ideal, 'go-', markersize=2.5, label='ideal cell')
    # plt.ylim(0, max(SBR_LCOEs)*1.1)
    plt.ylim(0, 6)
    plt.xlim(-0.025, 0.6+0.025)
    plt.xticks([0, 0.3, 0.6])
    plt.xlabel('ABR')
    plt.ylabel('Power density (W/cm$^2$)')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig('plots/Pdens_cell_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))
    # plt.show()


def sweep_config_improvements(CPE_th_sys, CPE_th_input):
    
    # print('Trad calcs')
    Trads = np.linspace(1723, 2623, 10)
    Trad_LCOEs = []
    Trad_Pdens_TPV = []
    Trad_eta_TPV = []
    Trad_BG_TPV = []
    Trad_LCOEs_ideal = []
    for Trad in Trads:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=Trad, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=0.9, BW=1)
        TPV_data_2200_ideal_imp = TPV_data_2200_ideal_imp.iloc[::10, :]
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
        LCOE_fuel = (CPE_th_input/eta_TPV)
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
        LCOE_MWh = LCOE*1e6
        LCOE_kWh = LCOE_MWh/1e3*1e2
        min_LCOE = np.min(LCOE_kWh)
        Trad_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        Trad_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        Trad_eta_TPV.append(min_LCOE_eta_TPV[0])
        Trad_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out)
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal)
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal)
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal
        LCOE_MWh_ideal = LCOE_ideal*1e6
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        Trad_LCOEs_ideal.append(min_LCOE_ideal)
    
    # print('VF calcs')
    VFs = np.linspace(0.1, 1, 10)
    VF_LCOEs = []
    VF_Pdens_TPV = []
    VF_eta_TPV = []
    VF_BG_TPV = []
    VF_LCOEs_ideal = []
    for VF in VFs:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=VF, emis=0.85, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=VF, emis=0.9, BW=1)
        TPV_data_2200_ideal_imp = TPV_data_2200_ideal_imp.iloc[::10, :]
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
        LCOE_fuel = (CPE_th_input/eta_TPV)
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
        LCOE_MWh = LCOE*1e6
        LCOE_kWh = LCOE_MWh/1e3*1e2
        min_LCOE = np.min(LCOE_kWh)
        VF_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        VF_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        VF_eta_TPV.append(min_LCOE_eta_TPV[0])
        VF_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out)
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal)
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal)
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal
        LCOE_MWh_ideal = LCOE_ideal*1e6
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        VF_LCOEs_ideal.append(min_LCOE_ideal)
    
    # print('emis calcs')
    emis = np.linspace(0.4, 1, 10)
    emis_LCOEs = []
    emis_Pdens_TPV = []
    emis_eta_TPV = []
    emis_BG_TPV = []
    emis_LCOEs_ideal = []
    for emi in emis:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=emi, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=emi, BW=1)
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
        LCOE_fuel = (CPE_th_input/eta_TPV)
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
        LCOE_MWh = LCOE*1e6
        LCOE_kWh = LCOE_MWh/1e3*1e2
        min_LCOE = np.min(LCOE_kWh)
        emis_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        emis_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        emis_eta_TPV.append(min_LCOE_eta_TPV[0])
        emis_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out)
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal)
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal)
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal
        LCOE_MWh_ideal = LCOE_ideal*1e6
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        emis_LCOEs_ideal.append(min_LCOE_ideal)
    
    # print('SBE calcs')
    # SBEs = np.append([0], np.linspace(0.01, 1, 10))
    SBEs = np.append(np.linspace(1,0.01,10), [0])
    SBE_LCOEs = []
    SBE_Pdens_TPV = []
    SBE_eta_TPV = []
    SBE_BG_TPV = []
    SBE_LCOEs_ideal = []
    for SBE in SBEs:
        TPV_data_2200_real_imp = tpv.P_balance_4(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, ABE=0.85, SBE=SBE, BW=1)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_4(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, ABE=0.9, SBE=SBE, BW=1)
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
        LCOE_fuel = (CPE_th_input/eta_TPV)
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
        LCOE_MWh = LCOE*1e6
        LCOE_kWh = LCOE_MWh/1e3*1e2
        min_LCOE = np.min(LCOE_kWh)
        SBE_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        SBE_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        SBE_eta_TPV.append(min_LCOE_eta_TPV[0])
        SBE_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out)
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal)
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal)
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal
        LCOE_MWh_ideal = LCOE_ideal*1e6
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        SBE_LCOEs_ideal.append(min_LCOE_ideal)

    # print('BW calcs')
    BWs = np.linspace(0.1, 1, 10)
    BW_LCOEs = []
    BW_Pdens_TPV = []
    BW_eta_TPV = []
    BW_BG_TPV = []
    BW_LCOEs_ideal = []
    for BW in BWs:
        TPV_data_2200_real_imp = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=BW)
        # TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=0.9, BW=BW)
        TPV_data_2200_ideal_imp = TPV_data_2200_ideal_imp.iloc[::10, :]
        Pdens_TPV = TPV_data_2200_real_imp['Pgen']
        eta_TPV = TPV_data_2200_real_imp['eff']
        BG_TPV = TPV_data_2200_real_imp['Eg']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
        LCOE_fuel = (CPE_th_input/eta_TPV)
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
        LCOE_MWh = LCOE*1e6
        LCOE_kWh = LCOE_MWh/1e3*1e2
        min_LCOE = np.min(LCOE_kWh)
        BW_LCOEs.append(min_LCOE)
        min_LCOE_index = np.where(LCOE_kWh == min_LCOE)
        min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
        min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
        min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
        BW_Pdens_TPV.append(min_LCOE_Pdens_TPV[0])
        BW_eta_TPV.append(min_LCOE_eta_TPV[0])
        BW_BG_TPV.append(min_LCOE_BG_TPV[0])
        Pdens_TPV_ideal = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV_ideal = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV_ideal = (CPA*CRF/Pdens_TPV_ideal/t_out)
        LCOE_sys_ideal = (CPE_th_sys*CRF/eta_TPV_ideal)
        LCOE_fuel_ideal = (CPE_th_input/eta_TPV_ideal)
        LCOE_ideal = LCOE_TPV_ideal + LCOE_sys_ideal + LCOE_fuel_ideal
        LCOE_MWh_ideal = LCOE_ideal*1e6
        LCOE_kWh_ideal = LCOE_MWh_ideal/1e3*1e2
        min_LCOE_ideal = np.min(LCOE_kWh_ideal)
        BW_LCOEs_ideal.append(min_LCOE_ideal)


    TPV_data_2200_ideal = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=0.9, BW=1)
    # TPV_data_2200_ideal = TPV_data_2200_ideal.iloc[::10, :]
    Pdens_TPV = np.array(TPV_data_2200_ideal['Pgen'])
    eta_TPV = np.array(TPV_data_2200_ideal['eff'])
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
    LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
    LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2
    min_LCOE_ideal = np.min(LCOE_kWh)
    

    TPV_data_2200_real = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
    # TPV_data_2200_real = TPV_data_2200_real.iloc[::10, :]
    Pdens_TPV = np.array(TPV_data_2200_real['Pgen'])
    eta_TPV = np.array(TPV_data_2200_real['eff'])
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
    LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
    LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    LCOE_kWh = LCOE_MWh/1e3*1e2
    min_LCOE_real = np.min(LCOE_kWh)
    min_LCOE_index = np.where(LCOE_kWh == min_LCOE_real)
    min_LCOE_Pdens_TPV = np.array(Pdens_TPV)[min_LCOE_index]
    min_LCOE_eta_TPV = np.array(eta_TPV)[min_LCOE_index]
    min_LCOE_BG_TPV = np.array(BG_TPV)[min_LCOE_index]
    
    fig = plt.figure(figsize=(12.5,2.5))
    fig.add_subplot(151)
    plt.plot(Trads-273, Trad_LCOEs, 'b-')
    plt.plot(Trads-273, Trad_LCOEs_ideal, 'g-')
    plt.plot(2123-273, min_LCOE_real, 'bo-', markersize=2.5)
    plt.plot(2123-273, min_LCOE_ideal, 'go', markersize=2.5)
    plt.ylim(0, max(Trad_LCOEs)*1.1)
    # plt.xlim(1450, 2250)
    # plt.xticks([1500, 1850, 2200])
    plt.xlabel('Trad ($^\circ$C)')
    plt.ylabel('LCOE (\\textcent/kWh)')
    fig.add_subplot(152)
    plt.plot(VFs, VF_LCOEs, 'b-')
    plt.plot(VFs, VF_LCOEs_ideal, 'g-')
    plt.plot(0.31, min_LCOE_real, 'bo-', markersize=2.5)
    plt.plot(0.31, min_LCOE_ideal, 'go', markersize=2.5)
    plt.ylim(0, max(Trad_LCOEs)*1.1)
    # plt.xlim(-0.43,1.05)
    # plt.xticks([0, 0.31, 1])
    plt.xlim(-0.05, 1.05)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], ['0.0', '0.25', '0.5', '0.75', '1.0'])
    plt.xlabel('VF')
    plt.ylabel('LCOE (\\textcent/kWh)')
    fig.add_subplot(153)
    plt.plot(emis, emis_LCOEs, 'b-')
    plt.plot(emis, emis_LCOEs_ideal, 'g-')
    plt.plot(0.85, min_LCOE_real, 'bo-', markersize=2.5)
    plt.plot(0.9, min_LCOE_ideal, 'go', markersize=2.5)
    plt.ylim(0, max(Trad_LCOEs)*1.1)
    # plt.xlim(0.65,1.05)
    # plt.xticks([0.7, 0.85, 1])
    plt.xlabel('ABE')
    plt.ylabel('LCOE (\\textcent/kWh)')
    fig.add_subplot(154)
    plt.gca().invert_xaxis()
    plt.plot(SBEs, SBE_LCOEs, 'b-')
    plt.plot(SBEs, SBE_LCOEs_ideal, 'g-')
    plt.plot(0.85, min_LCOE_real, 'bo-', markersize=2.5)
    plt.plot(0.9, min_LCOE_ideal, 'go', markersize=2.5)
    plt.ylim(0, max(Trad_LCOEs)*1.1)
    plt.xticks([0, 0.25, 0.5, 0.75, 1], ['0.0', '0.25', '0.5', '0.75', '1.0'])
    plt.xlabel('SBE')
    plt.ylabel('LCOE (\\textcent/kWh)')
    fig.add_subplot(155)
    plt.plot(BWs, BW_LCOEs, 'b-')
    plt.plot(1, min_LCOE_real, 'bo-', markersize=2.5, label='real cell')
    plt.plot(BWs, BW_LCOEs_ideal, 'g-')
    plt.plot(1, min_LCOE_ideal, 'go-', markersize=2.5, label='ideal cell')
    plt.ylim(0, max(Trad_LCOEs)*1.1)
    plt.xlim(-0.05, 1.05)
    plt.gca().invert_xaxis()
    plt.xlabel('BW')
    plt.xticks([0, 0.25, 0.5, 0.75, 1], ['0.0', '0.25', '0.5', '0.75', '1.0'])
    plt.ylabel('LCOE (\\textcent/kWh)')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig('plots/LCOE_config_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))


    fig = plt.figure(figsize=(12.5,2.5))
    fig.add_subplot(151)
    plt.plot(Trads-273, Trad_BG_TPV, 'b-')
    plt.plot(2123-273, min_LCOE_BG_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1.3)
    plt.xlabel('Trad ($^\circ$C)')
    plt.ylabel('Optimal bandgap (eV)')
    fig.add_subplot(152)
    plt.plot(VFs, VF_BG_TPV, 'b-')
    plt.plot(0.31, min_LCOE_BG_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1.3)
    plt.xlabel('VF')
    plt.ylabel('Optimal bandgap (eV)')
    fig.add_subplot(153)
    plt.plot(emis, emis_BG_TPV, 'b-')
    plt.plot(0.85, min_LCOE_BG_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1.3)
    plt.xlabel('ABE')
    plt.ylabel('Optimal bandgap (eV)')
    fig.add_subplot(154)
    plt.gca().invert_xaxis()
    plt.plot(SBEs, SBE_BG_TPV, 'b-')
    plt.plot(0.85, min_LCOE_BG_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1.3)
    plt.xlabel('SBE')
    plt.ylabel('Optimal bandgap (eV)')
    fig.add_subplot(155)
    plt.plot(BWs, BW_BG_TPV, 'b-')
    plt.plot(1, min_LCOE_BG_TPV, 'bo-', markersize=2.5, label='real cell')
    plt.ylim(0, 1.3)
    plt.xlim(-0.05, 1.05)
    plt.gca().invert_xaxis()
    plt.xlabel('BW')
    plt.ylabel('Optimal bandgap (eV)')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig('plots/BG_config_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))

    fig = plt.figure(figsize=(12.5,2.5))
    fig.add_subplot(151)
    plt.plot(Trads-273, Trad_eta_TPV, 'b-')
    plt.plot(2123-273, min_LCOE_eta_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1)
    plt.xlabel('Trad ($^\circ$C)')
    plt.ylabel('Efficiency')
    fig.add_subplot(152)
    plt.plot(VFs, VF_eta_TPV, 'b-')
    plt.plot(0.31, min_LCOE_eta_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1)
    plt.xlabel('VF')
    plt.ylabel('Efficiency')
    fig.add_subplot(153)
    plt.plot(emis, emis_eta_TPV, 'b-')
    plt.plot(0.85, min_LCOE_eta_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1)
    plt.xlabel('ABE')
    plt.ylabel('Efficiency')
    fig.add_subplot(154)
    plt.gca().invert_xaxis()
    plt.plot(SBEs, SBE_eta_TPV, 'b-')
    plt.plot(0.85, min_LCOE_eta_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 1)
    plt.xlabel('SBE')
    plt.ylabel('Efficiency')
    fig.add_subplot(155)
    plt.plot(BWs, BW_eta_TPV, 'b-')
    plt.plot(1, min_LCOE_eta_TPV, 'bo-', markersize=2.5, label='real cell')
    plt.ylim(0, 1)
    plt.xlim(-0.05, 1.05)
    plt.gca().invert_xaxis()
    plt.xlabel('BW')
    plt.ylabel('Efficiency')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig('plots/eta_config_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))

    fig = plt.figure(figsize=(12.5,2.5))
    fig.add_subplot(151)
    plt.plot(Trads-273, Trad_Pdens_TPV, 'b-')
    plt.plot(2123-273, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 10)
    plt.xlabel('Trad ($^\circ$C)')
    plt.ylabel('Power density (W/cm$^2$)')
    fig.add_subplot(152)
    plt.plot(VFs, VF_Pdens_TPV, 'b-')
    plt.plot(0.31, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 10)
    plt.xlabel('VF')
    plt.ylabel('Power density (W/cm$^2$)')
    fig.add_subplot(153)
    plt.plot(emis, emis_Pdens_TPV, 'b-')
    plt.plot(0.85, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 10)
    plt.xlabel('ABE')
    plt.ylabel('Power density (W/cm$^2$)')
    fig.add_subplot(154)
    plt.gca().invert_xaxis()
    plt.plot(SBEs, SBE_Pdens_TPV, 'b-')
    plt.plot(0.85, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5)
    plt.ylim(0, 10)
    plt.xlabel('SBE')
    plt.ylabel('Power density (W/cm$^2$)')
    fig.add_subplot(155)
    plt.plot(BWs, BW_Pdens_TPV, 'b-')
    plt.plot(1, min_LCOE_Pdens_TPV, 'bo-', markersize=2.5, label='real cell')
    plt.ylim(0, 10)
    plt.xlim(-0.05, 1.05)
    plt.gca().invert_xaxis()
    plt.xlabel('BW')
    plt.ylabel('Power density (W/cm$^2$)')
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig('plots/Pdens_config_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))

def sweep_BW_ideal(CPE_th_sys, CPE_th_input):
    BWs = np.linspace(0.1, 1, 10)
    BW_LCOEs = []
    for BW in BWs:
        TPV_data_2200_ideal_imp = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=1, BW=BW)
        # TPV_data_2200_ideal_imp = TPV_data_2200_ideal_imp.iloc[::10, :]
        Pdens_TPV = TPV_data_2200_ideal_imp['Pgen']
        eta_TPV = TPV_data_2200_ideal_imp['eff']
        LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
        LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
        LCOE_fuel = (CPE_th_input/eta_TPV)
        LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
        LCOE_MWh = LCOE*1e6
        min_LCOE = np.min(LCOE_MWh)
        BW_LCOEs.append(min_LCOE)

    plt.figure()
    plt.plot(BWs, BW_LCOEs, 'b-')
    # plt.plot(1, min_LCOE_real, 'b^', markersize=5, label='base case')
    # plt.plot(1, min_LCOE_ideal, 'bo', markersize=5, label='ideal case')
    plt.ylim(0, max(BW_LCOEs)*1.1)
    plt.xlabel('BW')
    plt.ylabel('LCOE (\$/MWh)')
    plt.tight_layout()
    plt.savefig('plots/LCOE_BWideal_improvements_%0.2e_%0.2e.pdf' % (CPE_th_sys, CPE_th_input))
    pass

def solve_CPA(CPA, Pdens_TPV, eta_TPV, CF, CPE_th_sys, CPE_th_input, LCOE_MWh_base):
    LCOE = (CPA*CRF/Pdens_TPV/(t_out*CF)) + (CPE_th_sys*CRF/eta_TPV) + (CPE_th_input/eta_TPV)
    LCOE_MWh = LCOE*1e6
    return LCOE_MWh - LCOE_MWh_base


def CPA_analysis(CPE_th_sys, CPE_th_input, CF, case):
    Pdens_TPV_base = 2
    eta_TPV_base = 0.3
    CPA_base = 5
    ir = 0.04
    n = 20
    t_out = 8760*CF
    CRF = (ir*(1+ir)**n)/((1+ir)**n-1)

    LCOE_TPV_base = (CPA_base*CRF/Pdens_TPV_base/t_out) # $/Wh for TPV contribution
    LCOE_sys_base = (CPE_th_sys*CRF/eta_TPV_base) # $/Wh for device contribution
    LCOE_fuel_base = (CPE_th_input/eta_TPV_base) # $/Wh for fuel contribution
    LCOE_base = LCOE_TPV_base + LCOE_sys_base + LCOE_fuel_base # $/Wh total LCOE
    LCOE_MWh_base = LCOE_base*1e6 # $/MWh total LCOE
    
    perc_imprv = np.linspace(0, 1, 10)
    CPAs = []
    for imprv in perc_imprv:
        eta_TPV = eta_TPV_base*(1+imprv)
        CPA_imp = fsolve(solve_CPA, CPA_base, args=(Pdens_TPV_base, eta_TPV, CF, CPE_th_sys, CPE_th_input, LCOE_MWh_base))
        LCOE_imp = (CPA_imp*CRF/Pdens_TPV_base/t_out) + (CPE_th_sys*CRF/eta_TPV) + (CPE_th_input/eta_TPV)
        LCOE_MWh_imp = LCOE_imp*1e6
        print(eta_TPV, CPA_imp, LCOE_MWh_imp, LCOE_MWh_base)
        CPAs.append(CPA_imp)
        # break
    CPAs = np.array(CPAs)
    CPAs_perc = (CPAs-CPA_base)/CPA_base

    # plt.figure()
    # plt.plot(perc_imprv*100, CPAs_perc*100, 'b-')
    # plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # top=False)      # ticks along the bottom edge are off
    # # top=False,         # ticks along the top edge are off
    # # labelbottom=False) # labels along the bottom edge are off
    # plt.xlabel('\% increase in efficiency')
    # plt.ylabel('\% increase in CPA')
    # plt.twinx()
    # plt.plot(perc_imprv*100, CPAs, 'b-')
    # plt.ylabel('CPA')
    # plt.grid()
    # plt.twiny()
    # plt.plot(eta_TPV_base*(1+perc_imprv), CPAs, 'b-')
    # plt.xlabel('efficiency')
    # plt.grid()
    # plt.tight_layout()

    CPAs_P = []
    for imprv in perc_imprv:
        Pdens_TPV = Pdens_TPV_base*(1+imprv)
        CPA_imp = fsolve(solve_CPA, CPA_base, args=(Pdens_TPV, eta_TPV_base, CF, CPE_th_sys, CPE_th_input, LCOE_MWh_base))
        LCOE_imp = (CPA_imp*CRF/Pdens_TPV/t_out) + (CPE_th_sys*CRF/eta_TPV_base) + (CPE_th_input/eta_TPV_base)
        LCOE_MWh_imp = LCOE_imp*1e6
        # print(Pdens_TPV, CPA_imp, LCOE_MWh_imp, LCOE_MWh_base)
        CPAs_P.append(CPA_imp)
        # break
    CPAs_P = np.array(CPAs_P)
    CPAs_perc_P = (CPAs_P-CPA_base)/CPA_base

    # plt.figure()
    # plt.plot(perc_imprv*100, CPAs_perc_P*100, 'b-')
    # plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # top=False)      # ticks along the bottom edge are off
    # # top=False,         # ticks along the top edge are off
    # # labelbottom=False) # labels along the bottom edge are off
    # plt.xlabel('\% increase in power density')
    # plt.ylabel('\% increase in CPA')
    # plt.twinx()
    # plt.plot(perc_imprv*100, CPAs_P, 'b-')
    # plt.ylabel('CPA')
    # plt.grid()
    # plt.twiny()
    # plt.plot(Pdens_TPV_base*(1+perc_imprv), CPAs_P, 'b-')
    # plt.xlabel('power density')
    # plt.grid()
    # plt.tight_layout()
    
    fig = plt.figure(figsize=(3.5,4))
    # host = fig.add_subplot(111)
    plt.plot(perc_imprv*100, CPAs_perc*100, 'b-', label='efficiency')
    plt.plot(perc_imprv*100, CPAs_perc_P*100, 'r-', label='power density')
    plt.xlabel('\% increase in metric')
    plt.ylabel('allowable \% increase in CPA')
    # plt.legend()
    host = plt.twinx()
    if max(CPAs) > max(CPAs_P):
        plt.plot(perc_imprv*100, CPAs, 'b--')
    else:
        plt.plot(perc_imprv*100, CPAs_P, 'r--')
    plt.grid()
    plt.ylabel('allowable CPA (\$/cm$^2$)')
    host.twiny()
    plt.grid()
    plt.plot(eta_TPV_base*(1+perc_imprv), CPAs, 'b-', label='efficiency')
    plt.xticks(0.3*np.array([1, 1.25, 1.5, 1.75, 2]))
    plt.xlabel('Efficiency', color='b')
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    # plt.tight_layout()
    host.twiny()
    plt.plot(Pdens_TPV_base*(1+perc_imprv), CPAs_P, 'r-', label='power density')
    # move the spine higher
    plt.gca().spines['top'].set_position(('outward', 30))
    plt.xticks(2*np.array([1, 1.25, 1.5, 1.75, 2]))
    plt.xlabel('Power density', color='r')
    # # remove minor ticks
    # plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    # plt.tight_layout()
    # plt.title(case, loc='left')
    # plt.text(case, xy=(0.15, 0.95), xycoords='axes fraction', ha='center', va='center', backgroundcolor='w', edgecolor='k')
    plt.text(0.05, 0.9, case,  transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    plt.tight_layout()
    plt.savefig('plots/CPA_analysis_%s.pdf' % case.replace(' ', '_'))
    plt.savefig('plots/CPA_analysis_%s.png' % case.replace(' ', '_'), dpi=300)
    pass

def rank_improvements(CPE_th_sys, CPE_th_input):
    Trads = [2123]
    SBRs = np.linspace(1, 0.9, 10)
    NRRs = np.linspace(1, 20, 10)
    NRRs = np.append(NRRs, np.arange(21,25,1))
    NRRs = np.insert(NRRs, 0, 0)
    Rseries_all = np.logspace(-6, -2, 10)
    Rseries_all = np.append(Rseries_all, 0.013)
    ABRs = np.linspace(0, 0.5, 10)
    ABRs = np.append(ABRs, np.array([0.55,0.6]))
    VFs = np.linspace(0.1, 1, 10)
    emis_all = np.linspace(0.4, 1, 10)
    BWs = np.linspace(0.1, 1, 10)

    params_all = [Trads, SBRs, ABRs, NRRs, Rseries_all, VFs, emis_all, BWs]
    labels = ['Trads', 'SBR', 'ABR', 'NRR', 'Rseries', 'VF', 'emis', 'BW']

    LCOE_Trad = []
    LCOE_SBR = []
    LCOE_ABR = []
    LCOE_NRR = []
    LCOE_Rseries = []
    LCOE_VF = []
    LCOE_emis = []
    LCOE_BW = []
    LCOEs = [LCOE_Trad, LCOE_SBR, LCOE_ABR, LCOE_NRR, LCOE_Rseries, LCOE_VF, LCOE_emis, LCOE_BW]

    Trad = 2123
    SBR = 0.95
    ABR = 0.3
    nonrad_ratio = 12
    series_R = 6.5e-3
    VF = 0.31
    emis = 0.85
    BW = 1
    params_base = [Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW]

    TPV_data_2200_real = tpv.P_balance_3(Trad=Trad, SBR=SBR, ABR=ABR, nonrad_ratio=nonrad_ratio, series_R=series_R, VF=VF, emis=emis, BW=BW)
    TPV_data_2200_real = TPV_data_2200_real.iloc[::10, :]
    Pdens_TPV = np.array(TPV_data_2200_real['Pgen'])
    eta_TPV = np.array(TPV_data_2200_real['eff'])
    LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
    LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
    LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
    LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
    LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
    min_LCOE_real = np.min(LCOE_MWh)
    LCOE_base = min_LCOE_real
    print(LCOE_base)
    
    while True:
        LCOEs = [[] for i in range(len(params_all))]
        for index, param in enumerate(params_all):
            param_vals = params_base.copy()
            for i in range(len(param_vals)):
                if i == index:
                    param_vals[index] = param
                else:
                    param_vals[i] = params_base[i]*np.ones(len(param))
            # print(param_vals)
            for i in range(len(param)):
                TPV_data_2200_real_imp = tpv.P_balance_3(Trad=param_vals[0][i], SBR=param_vals[1][i], ABR=param_vals[2][i], nonrad_ratio=param_vals[3][i], series_R=param_vals[4][i], VF=param_vals[5][i], emis=param_vals[6][i], BW=param_vals[7][i])
                TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
                Pdens_TPV = TPV_data_2200_real_imp['Pgen']
                eta_TPV = TPV_data_2200_real_imp['eff']
                LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
                LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
                LCOE_fuel = (CPE_th_input/eta_TPV)
                LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
                LCOE_MWh = LCOE*1e6
                min_LCOE = np.min(LCOE_MWh)
                LCOEs[index].append(min_LCOE)
        min_LCOE = min([min(LCOE) for LCOE in LCOEs])
        max_len = -1
        for LCOE in LCOEs:
            if len(LCOE) > max_len:
                max_len = len(LCOE)
        if max_len == 1:
            break
        # print([min(LCOE) for LCOE in LCOEs])
        # print(labels)
        min_index = -1
        for index, LCOE in enumerate(LCOEs):
            if np.min(LCOE) == min_LCOE:
                min_index = index
        prop = labels[min_index]
        # print(prop)
        # print(LCOEs[min_index])
        val = params_all[min_index][np.where(LCOEs[min_index] == min_LCOE)[0][0]]
        print(prop, val, '%0.2f' % min_LCOE)
        params_all[min_index] = [val]
        params_base[min_index] = val
        
        # print(params_all)
        # pause = input('Continue?')
        # break

def VF_3D():
    # A1 = 2*np.pi*R1*H
    # A2 = 2*np.pi*R2*H
    h = np.linspace(1,100,100) #H/R1
    R = 1/0.31 # A2/A1
    f1 = h**2 + R**2 - 1
    f2 = h**2 - R**2 + 1
    f3 = np.sqrt((f1+2)**2 - 4*R**2)
    f4 = f3*np.arccos(f2/(R*f1)) + f2*np.arcsin(1/R) - np.pi*f1/2
    F12 = 1 - 1/np.pi*(np.arccos(f2/f1)-f4/(2*h))
    F13 = (1 - F12)/2
    plt.figure()
    plt.plot(h, F13, 'b-')
    plt.xlabel('L/R$_1$')
    plt.ylabel('F$_{emit\\rightarrow env}$')
    plt.tight_layout()
    plt.savefig('plots/VF_3D.pdf')
    pass


def plot_eff_Pdens_lit():
    df = pd.read_csv('data/efficiency_Pdens_data.csv')
    plt.figure(figsize=(3,3))
    plt.plot(df['Pdens'], df['eff'], 'bo')
    plt.xlabel('Power density (W/cm$^2$)')
    plt.ylabel('TPV Efficiency (\%)')
    plt.xlim(0,6)
    plt.ylim(0,50)
    plt.tight_layout()
    plt.savefig('plots/eff_Pdens_lit.pdf')

    df = pd.read_csv('data/efficiency_power_cTPV_data.csv')
    df2 = pd.read_csv('data/efficiency_power_sTPV_data.csv')
    plt.figure(figsize=(3,3.5))
    plt.semilogx(df['Pout'], df['eff'], 'ro', label='combustion')
    plt.semilogx(df2['Pout'], df2['eff'], 'go', label='solar')
    plt.xlabel('Power output (W)')
    plt.ylabel('System Efficiency (\%)')
    plt.xlim(1e-1, 1e3)
    plt.ylim(0,50)
    plt.legend(loc=[0,-0.3], ncol=2)
    plt.tight_layout()
    plt.savefig('plots/eff_Pout_lit.pdf')


    # for i in range(len(SBRs)):
    #     TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=SBRs[i], ABR=ABR, nonrad_ratio=nonrad_ratio, series_R=series_R, VF=VF, emis=emis, BW=BW)
    #     TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
    #     Pdens_TPV = TPV_data_2200_real_imp['Pgen']
    #     eta_TPV = TPV_data_2200_real_imp['eff']
    #     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
    #     LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
    #     LCOE_fuel = (CPE_th_input/eta_TPV)
    #     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
    #     LCOE_MWh = LCOE*1e6
    #     min_LCOE = np.min(LCOE_MWh)
    #     LCOE_SBR.append(min_LCOE)

    # for i in range(len(NRRs)):
    #     TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=SBR, ABR=ABR, nonrad_ratio=NRRs[i], series_R=series_R, VF=VF, emis=emis, BW=BW)
    #     TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
    #     Pdens_TPV = TPV_data_2200_real_imp['Pgen']
    #     eta_TPV = TPV_data_2200_real_imp['eff']
    #     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
    #     LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
    #     LCOE_fuel = (CPE_th_input/eta_TPV)
    #     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
    #     LCOE_MWh = LCOE*1e6
    #     min_LCOE = np.min(LCOE_MWh)
    #     LCOE_NRR.append(min_LCOE)
    
    # for i in range(len(Rseries_all)):
    #     TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=SBR, ABR=ABR, nonrad_ratio=nonrad_ratio, series_R=Rseries_all[i], VF=VF, emis=emis, BW=BW)
    #     TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
    #     Pdens_TPV = TPV_data_2200_real_imp['Pgen']
    #     eta_TPV = TPV_data_2200_real_imp['eff']
    #     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
    #     LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
    #     LCOE_fuel = (CPE_th_input/eta_TPV)
    #     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
    #     LCOE_MWh = LCOE*1e6
    #     min_LCOE = np.min(LCOE_MWh)
    #     LCOE_Rseries.append(min_LCOE)
    
    # for i in range(len(ABRs)):
    #     TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=SBR, ABR=ABRs[i], nonrad_ratio=nonrad_ratio, series_R=series_R, VF=VF, emis=emis, BW=BW)
    #     TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
    #     Pdens_TPV = TPV_data_2200_real_imp['Pgen']
    #     eta_TPV = TPV_data_2200_real_imp['eff']
    #     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
    #     LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
    #     LCOE_fuel = (CPE_th_input/eta_TPV)
    #     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
    #     LCOE_MWh = LCOE*1e6
    #     min_LCOE = np.min(LCOE_MWh)
    #     LCOE_ABR.append(min_LCOE)
    
    # for i in range(len(VFs)):
    #     TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=SBR, ABR=ABR, nonrad_ratio=nonrad_ratio, series_R=series_R, VF=VFs[i], emis=emis, BW=BW)
    #     TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
    #     Pdens_TPV = TPV_data_2200_real_imp['Pgen']
    #     eta_TPV = TPV_data_2200_real_imp['eff']
    #     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
    #     LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
    #     LCOE_fuel = (CPE_th_input/eta_TPV)
    #     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
    #     LCOE_MWh = LCOE*1e6
    #     min_LCOE = np.min(LCOE_MWh)
    #     LCOE_VF.append(min_LCOE)
    
    # for i in range(len(emis_all)):
    #     TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=SBR, ABR=ABR, nonrad_ratio=nonrad_ratio, series_R=series_R, VF=VF, emis=emis_all[i], BW=BW)
    #     TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
    #     Pdens_TPV = TPV_data_2200_real_imp['Pgen']
    #     eta_TPV = TPV_data_2200_real_imp['eff']
    #     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
    #     LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
    #     LCOE_fuel = (CPE_th_input/eta_TPV)
    #     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
    #     LCOE_MWh = LCOE*1e6
    #     min_LCOE = np.min(LCOE_MWh)
    #     LCOE_emis.append(min_LCOE)
    
    # for i in range(len(BWs)):
    #     TPV_data_2200_real_imp = tpv.P_balance_3(Trad=Trad, SBR=SBR, ABR=ABR, nonrad_ratio=nonrad_ratio, series_R=series_R, VF=VF, emis=emis, BW=BWs[i])
    #     TPV_data_2200_real_imp = TPV_data_2200_real_imp.iloc[::10, :]
    #     Pdens_TPV = TPV_data_2200_real_imp['Pgen']
    #     eta_TPV = TPV_data_2200_real_imp['eff']
    #     LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out)
    #     LCOE_sys = (CPE_th_sys*CRF/eta_TPV)
    #     LCOE_fuel = (CPE_th_input/eta_TPV)
    #     LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel
    #     LCOE_MWh = LCOE*1e6
    #     min_LCOE = np.min(LCOE_MWh)
    #     LCOE_BW.append(min_LCOE)
    
    # LCOE_SBR = np.array(LCOE_SBR)
    # LCOE_ABR = np.array(LCOE_ABR)
    # LCOE_NRR = np.array(LCOE_NRR)
    # LCOE_Rseries = np.array(LCOE_Rseries)
    # LCOE_VF = np.array(LCOE_VF)
    # LCOE_emis = np.array(LCOE_emis)
    # LCOE_BW = np.array(LCOE_BW)
    # LCOEs = [LCOE_SBR, LCOE_ABR, LCOE_NRR, LCOE_Rseries, LCOE_VF, LCOE_emis, LCOE_BW]
    # labels = ['SBR', 'ABR', 'NRR', 'Rseries', 'VF', 'emis', 'BW']
    # min_LCOE = min([np.min(LCOE) for LCOE in LCOEs])
    # for index, LCOE in enumerate(LCOEs):
    #     if np.min(LCOE) == min_LCOE:
    #         prop = labels[index]
    #         val = params[index][np.where(LCOE == min_LCOE)[0][0]]
    #         print(prop,val, min_LCOE)
    

sweep_CPEs(ir=0.04,n=20,t_out=8760)
sweep_CPEs_2(ir=0.04,n=20,t_out=8760,CPA_TPV=5)
sweep_CPEs_2(ir=0.1,n=10,t_out=8760,CPA_TPV=5)
sweep_CPEs_2(ir=0.04,n=20,t_out=10,CPA_TPV=5)
sweep_CPEs_2(ir=0.04,n=20,t_out=8760,CPA_TPV=0.5)
cost_map()
sweep_CPE_regime_map()
sweep_CPE_LCOH()
sweep_CPE_LCOE_2(2, 0.6)
sweep_CPE_LCOE_2(4, 0.3)
sweep_CPE_LCOE_2(2, 0.3)
sweep_CPE_LCOE_2(4, 0.6)

CPE_th_sys = 0.0001297564687975647
CPE_th_input = 0.00016666666666666666
CF = 1
individual_plot_no_TPV(CPE_th_sys, CPE_th_input, CF, 'Power_plant')
individual_plot(CPE_th_sys, CPE_th_input, CF, 'Power_plant')

CPE_th_sys = 3.159018264840183e-05
CPE_th_input = 0
CF = 1
individual_plot_no_TPV(CPE_th_sys, CPE_th_input, CF, 'Waste_heat')
individual_plot(CPE_th_sys, CPE_th_input, CF, 'Waste_heat')

CPE_th_sys = 0.0005513372472276582
CPE_th_input = 5.857709049241056e-05
CF = 0.25
individual_plot_no_TPV(CPE_th_sys, CPE_th_input, CF, 'Portable_power')
individual_plot(CPE_th_sys, CPE_th_input, CF, 'Portable_power')

CPE_th_sys = 0.001902587519025875
CPE_th_input = 0
CF = 0.2
individual_plot_no_TPV(CPE_th_sys, CPE_th_input, CF, 'Solar_TPV')
individual_plot(CPE_th_sys, CPE_th_input, CF, 'Solar_TPV')

CPE_th_sys = 0.00012049720953830543
CPE_th_input = 3.3333333333333335e-05
CF = 0.8
individual_plot_no_TPV(CPE_th_sys, CPE_th_input, CF, 'Thermal_storage')
individual_plot(CPE_th_sys, CPE_th_input, CF, 'Thermal_storage')

VF_3D()
plot_eff_Pdens_lit()

# efficiency-limited case
CPE_th_sys = 0.00012049720953830543
CPE_th_input = 0.00016666666666666666
individual_plot(CPE_th_sys, CPE_th_input, 1, 'eff')
CPA_analysis(CPE_th_sys, CPE_th_input, 1, 'Efficiency-limited')
sweep_cell_improvements(CPE_th_sys, CPE_th_input)
sweep_config_improvements(CPE_th_sys, CPE_th_input)
sweep_BW_ideal(CPE_th_sys, CPE_th_input)
rank_improvements(CPE_th_sys, CPE_th_input)

# power-limited case
CPE_th_sys = 0.000012049720953830543
CPE_th_input = 0
individual_plot(CPE_th_sys, CPE_th_input, 1, 'power')
CPA_analysis(CPE_th_sys, CPE_th_input, 1, 'Power-limited')
sweep_cell_improvements(CPE_th_sys, CPE_th_input)
sweep_config_improvements(CPE_th_sys, CPE_th_input)
sweep_BW_ideal(CPE_th_sys, CPE_th_input)
rank_improvements(CPE_th_sys, CPE_th_input)

# # dual-limited case
CPE_th_sys = 0.00012049720953830543
CPE_th_input = 3.3333333333333335e-05
individual_plot(CPE_th_sys, CPE_th_input, 0.8, 'dual')
CPA_analysis(CPE_th_sys, CPE_th_input, 0.8, 'Dual-limited')
sweep_cell_improvements(CPE_th_sys, CPE_th_input)
sweep_config_improvements(CPE_th_sys, CPE_th_input)
sweep_BW_ideal(CPE_th_sys, CPE_th_input)
rank_improvements(CPE_th_sys, CPE_th_input)

LCOH = 3 # cents/kWh
LCOH = LCOH/100/1e3 # $/Wh
CPA = 2 # $/cm^2

specific_system_LCOH_CPA(LCOH, CPA, 1)


print('h2 comb 1')
specific_system(3.3333333333333335e-05, 3.7342719431760526e-05, 1)
print('h2 comb 2')
specific_system(0.00016666666666666666, 0.00012049720953830543, 0.8)
print('TEGS')
specific_system(3.3333333333333335e-05, 0.00012049720953830543, 0.8)
print('solar TPV')
specific_system(0, 0.001902587519025875, 0.2)
print('portable power generation')
specific_system(5.857709049241056e-05, 7.917482061317677e-05, 0.25)
print('waste heat recovery')
specific_system(0, 0.00012049720953830543, 1)

plt.show()