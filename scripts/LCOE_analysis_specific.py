import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import TPV_model as tpv
from matplotlib import ticker, cm
import matplotlib.colors as colors

plt.style.use(['science','grid'])

# TPV_data_2200_ideal = tpv.P_balance_2(Trad=2123, ref=1, EQE=1, series_R=0, VF=0.31, emis=1)
# TPV_data_2200_real = tpv.P_balance_2(Trad=2123, ref=0.95, EQE=0.7, series_R=6.5e-3, VF=0.31, emis=1)
# TPV_data_2200_real_imp_ref = tpv.P_balance_2(Trad=2123, ref=0.99, EQE=0.7, series_R=6.5e-3, VF=0.31, emis=1)
# TPV_data_2200_real_imp_eqe = tpv.P_balance_2(Trad=2123, ref=0.95, EQE=0.9, series_R=6.5e-3, VF=0.31, emis=1)
# TPV_data_2200_real_imp_resis = tpv.P_balance_2(Trad=2123, ref=0.95, EQE=0.7, series_R=6.5e-4, VF=0.31, emis=1)
TPV_data_2200_ideal = tpv.P_balance_3(Trad=2123, SBR=1, ABR=0, nonrad_ratio=0, series_R=0, VF=0.31, emis=1, BW=1)
TPV_data_2200_real = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_ref = tpv.P_balance_3(Trad=2123, SBR=0.99, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_eqe = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=1, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_resis = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-4, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_ABR = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.03, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_T = tpv.P_balance_3(Trad=2623, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.85, BW=1)
TPV_data_2200_real_imp_VF = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=1, emis=0.85, BW=1)
TPV_data_2200_real_imp_emis = tpv.P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=1, BW=1)
TPV_data_2200_ideal = TPV_data_2200_ideal.iloc[::10, :]
TPV_data_2200_real = TPV_data_2200_real.iloc[::10, :]
TPV_data_2200_real_imp_ref = TPV_data_2200_real_imp_ref.iloc[::10, :]
TPV_data_2200_real_imp_eqe = TPV_data_2200_real_imp_eqe.iloc[::10, :]
TPV_data_2200_real_imp_resis = TPV_data_2200_real_imp_resis.iloc[::10, :]
TPV_data_2200_real_imp_ABR = TPV_data_2200_real_imp_ABR.iloc[::10, :]
TPV_data_2200_real_imp_T = TPV_data_2200_real_imp_T.iloc[::10, :]
TPV_data_2200_real_imp_VF = TPV_data_2200_real_imp_VF.iloc[::10, :]
TPV_data_2200_real_imp_emis = TPV_data_2200_real_imp_emis.iloc[::10, :]

# Pdens_TPV = 5 # W/cm^2
# eta_TPV = 0.5 # efficiency of TPV

# Pdens_TPV = np.linspace(0, 10, 10) # W/cm^2
# eta_TPV = np.linspace(0, 1, 100) # efficiency of TPV

i_vals = np.arange(0.01, 0.1, 0.01)
n_vals = np.arange(10,30,1)
CRF = np.zeros((len(i_vals), len(n_vals)))
for i in range(len(i_vals)):
    for j in range(len(n_vals)):
        # print(i,j)
        CRF[i, j] = (i_vals[i]*(1+i_vals[i])**n_vals[j])/((1+i_vals[i])**n_vals[j]-1)
# print(CRF[i_vals==0.04, n_vals==20])
plt.figure()
plt.contour(n_vals, i_vals, CRF, levels=20, colors='k', linewidths=0.5)
plt.contourf(n_vals, i_vals, CRF, levels=200, cmap='plasma')
cb = plt.colorbar(label='CRF', ticks=[0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
# cb.set_ticks([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15])
plt.xlabel('Lifetime (years)')
plt.ylabel('Interest rate')
plt.tight_layout()
plt.savefig('CRF.png', dpi=300)

CRF_vals = np.linspace(0.036, 0.15,10)
CF_vals = np.logspace(-4,0,50)
C_1 = np.zeros((len(CRF_vals), len(CF_vals)))
for i in range(len(CRF_vals)):
    for j in range(len(CF_vals)):
        C_1[i, j] = CRF_vals[i]/(CF_vals[j]*8760)
plt.figure()
plt.contour(CF_vals, CRF_vals, C_1, levels=np.logspace(np.log10(3e-6),np.log10(1e-3), 20), norm=colors.LogNorm(), vmin=3e-6, vmax=3e-2, locator=ticker.LogLocator(),colors='k', linewidths=0.5)
plt.contourf(CF_vals, CRF_vals, C_1, levels=np.logspace(np.log10(3e-6),np.log10(3e-2), 200), locator=ticker.LogLocator(), cmap='plasma', norm=colors.LogNorm(), vmin=3e-6, vmax=3e-2)

cbar = plt.colorbar(label='C$_1$',format='%.0e')
# cbar.set_ticks([1, 10, 100, 600])
cbar.minorticks_off()
plt.xlabel('Capacity factor')
plt.ylabel('CRF')
plt.tight_layout()
plt.savefig('C_1.png', dpi=300)
# plt.show()
# exit()

Pdens_TPV = np.meshgrid(np.linspace(0.5, 10, 10), np.linspace(0.2, 1, 10))[0] # W/cm^2
eta_TPV = np.meshgrid(np.linspace(0.5, 10, 10), np.linspace(0.2, 1, 10))[1] # efficiency of TPV

# Pdens_TPV = np.linspace(0.5, 10, 10)[0] # W/cm^2
# eta_TPV = np.linspace(0.2, 1, 10)[9] # efficiency of TPV

# case1; combustor with H2 fuel at 100MW scale
CPA = 5 # $/cm^2 TPV cost per area
n = 20 # lifetime in years
i = 0.04 # interest rate
CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
print(CRF)
t_out = 8760 # hours in a year
CPV_device = 481500 # $/m^3 cost per volume of device
V_device = 135 # m^3 volume of device
P_in = 250e6 # W
CPV_insulation = 10000 # $/m^3 cost per volume of insulation
V_insulation = 860 # m^3 volume of insulation
CPM_fuel = 1 #$/kg cost per mass of fuel
eta_eth = 0.9 # energy to heat conversion efficiency
CPE_fuel = CPM_fuel/120e6/eta_eth*3600 # $/Wh cost per energy of fuel
CPE_system = (CPV_device*V_device + CPV_insulation*V_insulation)/(P_in*eta_eth*t_out) # $/Wh cost per energy of system
LCOH = CPE_system*CRF + CPE_fuel
LCOH_kWh = LCOH*1e3*1e2 # cents/kWh
print('h2 combustor liberal')
print(CPE_system, CPE_fuel, t_out/8760, LCOH_kWh)


LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
LCOE_system = (CPE_system*CRF/eta_TPV) # $/Wh for device contribution
LCOE_fuel = (CPE_fuel/eta_TPV) # $/Wh for fuel contribution
LCOE = LCOE_TPV + LCOE_system + LCOE_fuel # $/Wh total LCOE
LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
# print(Pdens_TPV, eta_TPV, LCOE_MWh)
# exit()
# reshape LCOE_MWh to 2D
# 2D plot of LCOE vs Pdens_TPV and eta_TPV
plt.figure(figsize=(6,4))
# color contour lines
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='TPV')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='TPV')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_comb.png', dpi=300)
# plt.show()

# case1.5; combustor with H2 fuel at 100MW scale (higher costs for fuel and infra)
CPA = 5 # $/cm^2 TPV cost per area
n = 20 # lifetime in years
i = 0.04 # interest rate
CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
t_out = 8760 # hours in a year
CPV_device = 481500 # $/m^3 cost per volume of device
V_device = 500 # m^3 volume of device
P_in = 250e6 # W
CPV_insulation = 10000 # $/m^3 cost per volume of insulation
V_insulation = 1500 # m^3 volume of insulation
CPM_fuel = 5 #$/kg cost per mass of fuel
eta_eth = 0.9 # energy to heat conversion efficiency
CPE_fuel = CPM_fuel/120e6/eta_eth*3600 # $/Wh cost per energy of fuel
CPE_system = (CPV_device*V_device + CPV_insulation*V_insulation)/(P_in*eta_eth*t_out) # $/Wh cost per energy of system
LCOH = CPE_system*CRF + CPE_fuel
LCOH_kWh = LCOH*1e3*1e2 # cents/kWh
# CPE_system = 0.00012049720953830543
print('h2 combustor conservative numbers')
print(CPE_system, CPE_fuel, t_out/8760, LCOH_kWh)

LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
LCOE_system = (CPE_system*CRF/eta_TPV) # $/Wh for device contribution
LCOE_fuel = (CPE_fuel/eta_TPV) # $/Wh for fuel contribution
LCOE = LCOE_TPV + LCOE_system + LCOE_fuel # $/Wh total LCOE
LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
# print(Pdens_TPV, eta_TPV, LCOE_MWh)
# exit()
# reshape LCOE_MWh to 2D
# 2D plot of LCOE vs Pdens_TPV and eta_TPV
plt.figure(figsize=(6,4))
# color contour lines
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='ideal')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='ref')
plt.plot(TPV_data_2200_real_imp_ref['Pgen'], TPV_data_2200_real_imp_ref['eff'], 'k--v', label='imp ref')
plt.plot(TPV_data_2200_real_imp_eqe['Pgen'], TPV_data_2200_real_imp_eqe['eff'], 'k--^', label='imp eqe')
plt.plot(TPV_data_2200_real_imp_resis['Pgen'], TPV_data_2200_real_imp_resis['eff'], 'k--P', label='imp resis')
plt.plot(TPV_data_2200_real_imp_ABR['Pgen'], TPV_data_2200_real_imp_ABR['eff'], 'k--D', label='imp ABR')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_comb_2.png', dpi=300)

plt.figure(figsize=(6,4))
# color contour lines
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='ideal')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='ref')
plt.plot(TPV_data_2200_real_imp_T['Pgen'], TPV_data_2200_real_imp_T['eff'], 'k--v', label='imp T')
plt.plot(TPV_data_2200_real_imp_VF['Pgen'], TPV_data_2200_real_imp_VF['eff'], 'k--^', label='imp VF')
plt.plot(TPV_data_2200_real_imp_emis['Pgen'], TPV_data_2200_real_imp_emis['eff'], 'k--P', label='imp emis')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_comb_2_config.png', dpi=300)
# plt.show()


# case 2: energy storage
CPA = 5 # $/cm^2 TPV cost per area
n = 20 # lifetime in years
i = 0.04 # interest rate
CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
t_out = 8760*0.8 # hours in a year
# CPV_device = 5000 # $/m^3 cost per volume of device
# V_device = 2500 # m^3 volume of device
CPE_storage = 20e-3 # $/Wh cost per energy of device
storage_energy = 1e9 # Wh energy of device
P_in = 250e6 # W
# CPV_insulation = 10000 # $/m^3 cost per volume of insulation
# V_insulation = 7500 # m^3 volume of insulation
CPP_dis_not_TPV = 0.42 # $/W cost per power of discharging (not incl TPV)
dis_power = 0.2*P_in # W power of discharging
CPP_ch = 0.03 # $/W cost per power of charging
# CPM_fuel = 0 #$/kg cost per mass of fuel
# CPE_fuel = CPM_fuel/120e6*3600 # $/Wh cost per energy of fuel
eta_eth = 0.9 # energy to heat conversion efficiency
CPE_fuel = 0.03e-3/eta_eth # $/Wh cost per energy of electricity
LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
LCOE_system = ((CPE_storage*storage_energy+CPP_ch*P_in+CPP_dis_not_TPV*dis_power*eta_TPV)*CRF/(P_in*4/20)/eta_eth/eta_TPV/t_out) # $/Wh for device contribution
LCOE_fuel = (CPE_fuel/eta_TPV) # $/Wh for fuel contribution
LCOE = LCOE_TPV + LCOE_system + LCOE_fuel # $/Wh total LCOE
LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
CPE_system = (CPE_storage*storage_energy+CPP_dis_not_TPV*dis_power*0.5+CPP_ch*P_in)/(P_in*(4/20)*eta_eth*t_out) # $/Wh cost per energy of system
LCOH = CPE_system*CRF + CPE_fuel
LCOH_kWh = LCOH*1e3*1e2 # cents/kWh
print('energy storage')
print(CPE_system, CPE_fuel, t_out/8760, LCOH_kWh)
# print(Pdens_TPV, eta_TPV, LCOE_MWh)
# exit()
# reshape LCOE_MWh to 2D
# 2D plot of LCOE vs Pdens_TPV and eta_TPV
plt.figure(figsize=(6,4))
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='TPV')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='TPV')
plt.plot(TPV_data_2200_real_imp_ref['Pgen'], TPV_data_2200_real_imp_ref['eff'], 'k--v', label='imp ref')
plt.plot(TPV_data_2200_real_imp_eqe['Pgen'], TPV_data_2200_real_imp_eqe['eff'], 'k--^', label='imp eqe')
plt.plot(TPV_data_2200_real_imp_resis['Pgen'], TPV_data_2200_real_imp_resis['eff'], 'k--P', label='imp resis')
plt.plot(TPV_data_2200_real_imp_ABR['Pgen'], TPV_data_2200_real_imp_ABR['eff'], 'k--D', label='imp ABR')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_storage.png', dpi=300)

plt.figure(figsize=(6,4))
# color contour lines
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='ideal')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='ref')
plt.plot(TPV_data_2200_real_imp_T['Pgen'], TPV_data_2200_real_imp_T['eff'], 'k--v', label='imp T')
plt.plot(TPV_data_2200_real_imp_VF['Pgen'], TPV_data_2200_real_imp_VF['eff'], 'k--^', label='imp VF')
plt.plot(TPV_data_2200_real_imp_emis['Pgen'], TPV_data_2200_real_imp_emis['eff'], 'k--P', label='imp emis')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_storage_config.png', dpi=300)
# plt.show()

# case3: solar TPV
CPA = 5 # $/cm^2 TPV cost per area
n = 20 # lifetime in years
i = 0.04 # interest rate
CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
CF = 0.2
t_out = 8760*CF # hours in a year
lens = 100 # $
shield = 10 # $
absorber = 15 # $
V_insulation = 0.125 # m^3 volume of insulation
CPV_insulation = 1000 # $/m^3 cost per volume of insulation
insulation = CPV_insulation*V_insulation # $
P_in = 100 # W
eta_lth = 0.75 # efficiency of light to heat
CF = 0.2
t_out = 8760*CF # hours in a year
CPE_th_input = 0
CPE_th_sys = (lens+shield+absorber+insulation)/(P_in*eta_lth*t_out) # $/Wh cost per energy of system
LCOH = CPE_th_sys*CRF + CPE_th_input
LCOH_kWh = LCOH*1e3*1e2 # cents/kWh
print('solar TPV')
print(CPE_th_sys, CPE_th_input, CF, LCOH_kWh)

LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
LCOE_sys = (CPE_th_sys*CRF/eta_TPV) # $/Wh for device contribution
LCOE_fuel = (CPE_th_input/eta_TPV) # $/Wh for fuel contribution
LCOE = LCOE_TPV + LCOE_sys + LCOE_fuel # $/Wh total LCOE
LCOE_MWh = LCOE*1e6 # $/MWh total LCOE

plt.figure(figsize=(6,4))
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='TPV')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='TPV')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_sTPV.png', dpi=300)
# plt.show()

# # case 2: solar TPV
# CPA = 5 # $/cm^2 TPV cost per area
# n = 20 # lifetime in years
# i = 0.04 # interest rate
# CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
# t_out = 8760/4 # hours in a year
# CPV_device = 5000 # $/m^3 cost per volume of device
# V_device = 2500 # m^3 volume of device
# P_in = 250e6 # W
# CPV_insulation = 10000 # $/m^3 cost per volume of insulation
# V_insulation = 7500 # m^3 volume of insulation
# CPM_fuel = 0 #$/kg cost per mass of fuel
# CPE_fuel = CPM_fuel/120e6*3600 # $/Wh cost per energy of fuel
# eta_eth = 0.9 # energy to heat conversion efficiency
# LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
# LCOE_device = (CPV_device*V_device*CRF/P_in/eta_eth/eta_TPV/t_out) # $/Wh for device contribution
# LCOE_insulation = (CPV_insulation*V_insulation*CRF/P_in/eta_eth/eta_TPV/t_out) # $/Wh for insulation contribution
# LCOE_fuel = (CPE_fuel/eta_eth/eta_TPV) # $/Wh for fuel contribution
# LCOE = LCOE_TPV + LCOE_device + LCOE_insulation + LCOE_fuel # $/Wh total LCOE
# LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
# # print(Pdens_TPV, eta_TPV, LCOE_MWh)
# # exit()
# # reshape LCOE_MWh to 2D
# # 2D plot of LCOE vs Pdens_TPV and eta_TPV
# plt.figure(figsize=(6,4))
# plt.plot(TPV_data_2200_ideal['Pgen']/4, TPV_data_2200_ideal['eff'], 'k-o', label='TPV')
# plt.plot(TPV_data_2200_real['Pgen']/4, TPV_data_2200_real['eff'], 'k--s', label='TPV')
# plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
# plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
# plt.colorbar(label='LCOE (MWh$^{-1}$)')
# plt.xlabel('Power density (W/cm$^2$)')
# plt.ylabel('Efficiency')
# plt.xlim(0.5, 10)
# plt.ylim(0.2, 1)
# plt.tight_layout()
# plt.savefig('LCOE_Pdens_eta_solarTPV.png', dpi=300)

# case4; combustor with propane fuel at 100W scale
CPA = 5 # $/cm^2 TPV cost per area
n = 20 # lifetime in years
i = 0.04 # interest rate
CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
CF = 0.25
t_out = 8760*CF # hours in a year
CPV_device = 337680 # $/m^3 cost per volume of device
V_device = 2.5e-4 # m^3 volume of device
P_in = 100 # W
CPV_insulation = 100 # $/m^3 cost per volume of insulation
V_insulation = 10e-4 # m^3 volume of insulation
CPM_fuel = 1 #$/gal cost per mass of fuel
HV_fuel = 19834*4.24*0.29 # Wh/gal
eta_eth = 0.7 # energy to heat conversion efficiency
CPE_fuel = CPM_fuel/HV_fuel/eta_eth # $/Wh cost per energy of fuel
CPE_system = (CPV_device*V_device + CPV_insulation*V_insulation)/(P_in*eta_eth*t_out) # $/Wh cost per energy of system
LCOH = CPE_system*CRF + CPE_fuel
LCOH_kWh = LCOH*1e3*1e2 # cents/kWh
print('portable power')
print(CPE_system, CPE_fuel, CF, LCOH_kWh)


LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
LCOE_system = (CPE_system*CRF/eta_TPV) # $/Wh for device contribution
LCOE_fuel = (CPE_fuel/eta_TPV) # $/Wh for fuel contribution
LCOE = LCOE_TPV + LCOE_system + LCOE_fuel # $/Wh total LCOE
LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
# print(Pdens_TPV, eta_TPV, LCOE_MWh)
# exit()
# reshape LCOE_MWh to 2D
# 2D plot of LCOE vs Pdens_TPV and eta_TPV
plt.figure(figsize=(6,4))
# color contour lines
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='TPV')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='TPV')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_portable.png', dpi=300)

# case5; waste heat recovery
CPA = 5 # $/cm^2 TPV cost per area
n = 20 # lifetime in years
i = 0.04 # interest rate
CRF = (i*(1+i)**n)/((1+i)**n-1) # capital recovery factor
CF = 1
t_out = 8760*CF # hours in a year
CPE_fuel=0
P_in = 1e6 # W
CPV_tin = 102000 # $/m^3 cost per volume of tin
V_tin = 0 # m^3 volume of tin
CPV_insulation = 1812 # $/m^3 cost per volume of insulation
V_insulation = 50 # m^3 volume of insulation
CPV_piping = 15845.7 # $/m^3 cost per volume of piping from skyline
V_piping = 10 # m^3 volume of piping
eta_eth = 0.9 # energy to heat conversion efficiency
CPE_system = (CPV_tin*V_tin + CPV_piping*V_piping + CPV_insulation*V_insulation)/(P_in*eta_eth*t_out) # $/Wh cost per energy of system
# CPE_system = 0.00012049720953830543
LCOH = CPE_system*CRF + CPE_fuel
LCOH_kWh = LCOH*1e3*1e2 # cents/kWh
print('waste heat recovery')
print(CPE_system, CPE_fuel, CF, LCOH_kWh)

LCOE_TPV = (CPA*CRF/Pdens_TPV/t_out) # $/Wh for TPV contribution
LCOE_system = (CPE_system*CRF/eta_TPV) # $/Wh for device contribution
LCOE_fuel = (CPE_fuel/eta_TPV) # $/Wh for fuel contribution
LCOE = LCOE_TPV + LCOE_system + LCOE_fuel # $/Wh total LCOE
LCOE_MWh = LCOE*1e6 # $/MWh total LCOE
# print(Pdens_TPV, eta_TPV, LCOE_MWh)
# exit()
# reshape LCOE_MWh to 2D
# 2D plot of LCOE vs Pdens_TPV and eta_TPV
plt.figure(figsize=(6,4))
# color contour lines
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='TPV')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='TPV')
plt.plot(TPV_data_2200_real_imp_ref['Pgen'], TPV_data_2200_real_imp_ref['eff'], 'k--v', label='imp ref')
plt.plot(TPV_data_2200_real_imp_eqe['Pgen'], TPV_data_2200_real_imp_eqe['eff'], 'k--^', label='imp eqe')
plt.plot(TPV_data_2200_real_imp_resis['Pgen'], TPV_data_2200_real_imp_resis['eff'], 'k--P', label='imp resis')
plt.plot(TPV_data_2200_real_imp_ABR['Pgen'], TPV_data_2200_real_imp_ABR['eff'], 'k--D', label='imp ABR')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_wasteheat.png', dpi=300)

plt.figure(figsize=(6,4))
# color contour lines
plt.plot(TPV_data_2200_ideal['Pgen'], TPV_data_2200_ideal['eff'], 'k-o', label='ideal')
plt.plot(TPV_data_2200_real['Pgen'], TPV_data_2200_real['eff'], 'k--s', label='ref')
plt.plot(TPV_data_2200_real_imp_T['Pgen'], TPV_data_2200_real_imp_T['eff'], 'k--v', label='imp T')
plt.plot(TPV_data_2200_real_imp_VF['Pgen'], TPV_data_2200_real_imp_VF['eff'], 'k--^', label='imp VF')
plt.plot(TPV_data_2200_real_imp_emis['Pgen'], TPV_data_2200_real_imp_emis['eff'], 'k--P', label='imp emis')
plt.contour(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, colors='k', linewidths=0.5)
plt.contourf(Pdens_TPV, eta_TPV, LCOE_MWh, levels=20, cmap='jet')
plt.colorbar(label='LCOE ($\$$/MWh$^{-1}$)')
plt.xlabel('Power density (W/cm$^2$)')
plt.ylabel('Efficiency')
plt.xlim(0.5, 10)
plt.ylim(0.2, 1)
plt.tight_layout()
plt.savefig('LCOE_Pdens_eta_wasteheat_config.png', dpi=300)

plt.show()