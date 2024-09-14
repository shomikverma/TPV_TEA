import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from tqdm import tqdm
import pandas as pd
from scipy.optimize import fsolve, curve_fit
from tqdm import tqdm
from p_tqdm import p_map
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import os

e = 1.60217662e-19  # coulombs
h = 6.62607004e-34  # m2 kg / s
c = 299792458  # m / s
kb = 1.38064852e-23  # m2 kg s-2 K-1
JtoeV = 6.242e+18
Tc = 300.0

E = np.linspace(1e-5, 10, 1000)
EJ = E * 1/JtoeV
Trad = 2400
P = 2*np.pi / (c**2 * h**3) * EJ**3 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV
nu = E * 241799050402293 / 1e12


def Prad_total_SB(E, Eg, Trad, Tc, V):
    E = E[E < Eg]
    EJ = E * 1/JtoeV
    P = 2*np.pi / (c**2 * h**3) * EJ**3 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV
    Prad = np.trapz(P, E)
    return Prad


def Prad_total_AB(E, Eg, Trad, Tc, V, BW):
    E = E[E >= Eg]
    full_BW = (1240/Eg - 1240/max(E))
    new_BW = full_BW * BW
    max_E = 1240/(1240/Eg - new_BW)
    E = E[E <= max_E]
    EJ = E * 1/JtoeV
    P = 2*np.pi / (c**2 * h**3) * EJ**3 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV
    Prad = np.trapz(P, E)
    return Prad

def Ptherm(E, Eg, Trad, Tc, V, BW):
    E = E[E >= Eg]
    full_BW = (1240/Eg - 1240/max(E))
    new_BW = full_BW * BW
    max_E = 1240/(1240/Eg - new_BW)
    E = E[E <= max_E]
    EJ = E * 1/JtoeV
    EgJ = Eg * 1/JtoeV
    P = 2*np.pi / (c**2 * h**3) * EJ**3 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV - \
        EgJ * 2*np.pi / (c**2 * h**3) * EJ**2 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV
    Pgen = np.trapz(P, E)
    return Pgen

def PBG_Voc(E, Eg, Trad, Tc, V, BW):
    E = E[E >= Eg]
    full_BW = (1240/Eg - 1240/max(E))
    new_BW = full_BW * BW
    max_E = 1240/(1240/Eg - new_BW)
    E = E[E <= max_E]
    EJ = E * 1/JtoeV
    EgJ = Eg * 1/JtoeV
    P = EgJ * 2*np.pi / (c**2 * h**3) * EJ**2 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV - \
        e*V * 2*np.pi / (c**2 * h**3) * EJ**2 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV
    Pgen = np.trapz(P, E)
    return Pgen

def Ptherm_plot(E, Eg, Trad, Tc, V, BW):
    E = E[E >= Eg]
    full_BW = (1240/Eg - 1240/max(E))
    new_BW = full_BW * BW
    max_E = 1240/(1240/Eg - new_BW)
    E = E[E <= max_E]
    EJ = E * 1/JtoeV
    P = 2*np.pi / (c**2 * h**3) * EJ**3 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV - \
        e*V * 2*np.pi / (c**2 * h**3) * EJ**2 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV
    Pgen = np.trapz(P, E)
    # return Pgen
    plt.figure()
    plt.plot(E,2*np.pi / (c**2 * h**3) * EJ**3 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV)
    plt.plot(E,e*V * 2*np.pi / (c**2 * h**3) * EJ**2 * (1/(np.exp(EJ/(kb*Trad))-1)) / JtoeV)
    plt.show()

def Precomb_PV(E, Eg, Trad, Tc, V, BW):
    E = E[E >= Eg]
    full_BW = (1240/Eg - 1240/max(E))
    new_BW = full_BW * BW
    max_E = 1240/(1240/Eg - new_BW)
    E = E[E <= max_E]
    EJ = E * 1/JtoeV
    P = e*V*2*np.pi / (c**2 * h**3) * EJ**2 * (1/(np.exp((EJ - e*V)/(kb*Tc)) - 1)) / JtoeV
    Prad = np.trapz(P, E)
    return Prad


def P_balance_2_ideal(Trad=2200):
    Egs = np.linspace(0.5, 2, 100)
    # Egs = [1.4]
    # Pouts = []
    effs = []
    all_Ps = []
    ref = 1
    EQE = 1
    series_R = 0 # ohm cm2
    # for Trad in np.arange(3000, 9000, 1000):
    for Eg in tqdm(Egs):
        E = np.linspace(0.01, 10, 10000)
        Ps = []
        effs_V = []
        Vs = np.linspace(Eg/2, Eg*0.99, 1000)
        for V in Vs:
            P_rad_SB = Prad_total_SB(E, Eg, Trad, Tc, V)
            P_rad_AB = Prad_total_AB(E, Eg, Trad, Tc, V)
            P_ref_loss = Pref_loss(E, Eg, Trad, Tc, V, P_rad_SB, ref)
            P_recomb = Precomb_PV(E, Eg, Trad, Tc, V)
            P_therm = Ptherm(E, Eg, Trad, Tc, V)
            P_reabs = P_rad_SB - P_ref_loss
            P_inc = P_rad_AB - P_therm - P_recomb
            P_EQE = P_inc * (1-EQE)
            P_gen = P_inc - P_EQE
            I_gen = P_gen / V
            P_series = I_gen**2 * series_R
            P_gen = P_gen - P_series
            eff = P_gen / (P_rad_AB + P_ref_loss - P_recomb)
            Ps.append([P_rad_SB, P_rad_AB, P_ref_loss, P_recomb, P_therm, P_reabs, P_EQE, P_series, P_gen])
            effs_V.append(eff)
        df = pd.DataFrame(Ps, columns=['Prad_SB', 'Prad_AB', 'Pref_loss',
                          'Precomb', 'Ptherm', 'Preabs', 'PEQE', 'Pseries','Pgen'])
        df['effs'] = effs_V
        idx = df.index[df['Pgen'] == max(df['Pgen'])]
        data = df.loc[idx]
        all_Ps.append(data.values[0][:-1])
        effs.append(data.values[0][-1])
        del df

    all_Ps = np.array(all_Ps)/100**2  # W cm-2
    df = pd.DataFrame(all_Ps, columns=['Prad_SB', 'Prad_AB', 'Pref_loss',
                                       'Precomb', 'Ptherm', 'Preabs', 'PEQE', 'Pseries','Pgen'])
    df['Eg'] = Egs
    df.to_csv('data/P_balance_all_ideal_%0.0f.csv' % Trad)
    # fig = plt.figure(num=2, figsize=[8, 4], dpi=300, clear=True)
    # fig.add_subplot(121)
    # plt.plot(df['Eg'], df['Prad_total'],  label='Prad_total')
    # plt.plot(df['Eg'], df['Pref'],  label='Pref')
    # plt.plot(df['Eg'], df['Precomb'], label='Precomb')
    # plt.plot(df['Eg'], df['Prad'],  label='Prad_cell')
    # plt.plot(df['Eg'], df['Pgen'], label='Pgen')
    # plt.stackplot(df['Eg'], df['Pgen'], df['Pseries'], df['PEQE'], df['Precomb'], df['Ptherm'], df['Pref_loss'],
    #               df['Preabs'],  labels=['Pgen', 'Pseries', 'P_EQE', 'Prad_recomb', 'Ptherm', 'Pref_loss', 'Preabs'])
    # plt.legend()
    # plt.xlabel('Eg (eV)')
    # plt.ylabel('Power (W/cm^2)')
    # plt.title('T = {}K'.format(Trad))
    # plt.xlim(0.5, 2)
    # plt.ylim(0, max(df['Prad_SB'] + df['Prad_AB']))
    # plt.tight_layout()
    # fig.add_subplot(122)
    df2 = df[['Pgen', 'Pseries', 'PEQE', 'Ptherm', 'Pref_loss']]
    df_perc = df2.divide(df2.sum(axis=1), axis=0)
    # plt.stackplot(df['Eg'], df_perc['Pgen'], df_perc['Pseries'], df_perc['PEQE'],  np.zeros(len(df['Eg'])),
    #               df_perc['Ptherm'], df_perc['Pref_loss'],
    #               labels=['Pgen', 'Pseries', 'PEQE', '', 'Ptherm', 'Prefloss'])
    # plt.xlabel('Eg (eV)')
    # plt.ylabel('Frac Pabs')
    # plt.title('Efficiency')
    # plt.xlim(0.5, 2)
    # plt.ylim(0, 1)
    # plt.legend(loc='lower right')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('plots/P_balance_ideal_%0.0f.png' % Trad)
    df['P_net'] = df['Pgen'] + df['Pseries'] + df['PEQE'] + df['Ptherm'] + df['Pref_loss']
    df_data = df[['Eg', 'Pgen']]
    df_data['Trad'] = Trad
    df_data['eff'] = df_perc['Pgen']
    df_data.to_csv('data/P_balance_ideal_%0.0f.csv' % Trad)
    return df_data


def P_balance_2(Trad=2200, ref=1, EQE=1, series_R=0, VF=1, emis=1):
    if os.path.exists('data/P_balance_all_Trad_%0.0f_ref_%0.2f_EQE_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, ref, EQE, series_R, VF, emis)):
        df = pd.read_csv('data/P_balance_all_Trad_%0.0f_ref_%0.2f_EQE_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, ref, EQE, series_R, VF, emis))
        df_data = pd.read_csv('data/P_balance_Trad_%0.0f_ref_%0.2f_EQE_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, ref, EQE, series_R, VF, emis))
        return df_data
    ABR = 0
    series_R = series_R * 1e-4  # ohm m2
    Egs = np.linspace(0.5, 2, 100)
    # Egs = [1.4]
    # Pouts = []
    effs = []
    all_Ps = []
    # Trad = 2200
    # ref = 0.95
    # EQE = 0.97
    # series_R = 0.1e-6 # ohm cm2
    # for Trad in np.arange(3000, 9000, 1000):
    for Eg in tqdm(Egs):
        E = np.linspace(0.01, 10, 10000)
        Ps = []
        effs_V = []
        Vs = np.linspace(Eg/2, Eg*0.99, 1000)
        for V in Vs:
            P_rad_SB = Prad_total_SB(E, Eg, Trad, Tc, V)*VF*emis
            P_rad_AB = Prad_total_AB(E, Eg, Trad, Tc, V)*VF*emis
            P_rad_AB_ref = P_rad_AB * ABR
            P_rad_AB_abs = P_rad_AB * (1-ABR)
            P_recomb = Precomb_PV(E, Eg, Trad, Tc, V)*VF*emis
            P_therm = Ptherm(E, Eg, Trad, Tc, V)*VF*emis
            P_ref_loss = (1-ref)*P_rad_SB
            P_reabs = P_rad_SB - P_ref_loss
            P_inc = P_rad_AB - P_therm - P_recomb
            P_EQE = P_inc * (1-EQE)
            P_gen = P_inc - P_EQE
            I_gen = P_gen / V
            P_series = I_gen**2 * series_R
            P_gen = P_gen - P_series
            eff = P_gen / (P_rad_AB + P_ref_loss - P_recomb)
            Ps.append([P_rad_SB, P_rad_AB, P_ref_loss, P_recomb, P_therm, P_reabs, P_EQE, P_series, P_gen])
            effs_V.append(eff)
        df = pd.DataFrame(Ps, columns=['Prad_SB', 'Prad_AB', 'Pref_loss',
                          'Precomb', 'Ptherm', 'Preabs', 'PEQE', 'Pseries','Pgen'])
        df['effs'] = effs_V
        idx = df.index[df['Pgen'] == max(df['Pgen'])]
        data = df.loc[idx]
        all_Ps.append(data.values[0][:-1])
        effs.append(data.values[0][-1])
        del df

    all_Ps = np.array(all_Ps)/100**2  # W cm-2
    series_R = series_R * 1e4  # ohm cm2
    df = pd.DataFrame(all_Ps, columns=['Prad_SB', 'Prad_AB', 'Pref_loss',
                                       'Precomb', 'Ptherm', 'Preabs', 'PEQE', 'Pseries','Pgen'])
    df['Eg'] = Egs
    df.to_csv('data/P_balance_all_Trad_%0.0f_ref_%0.2f_EQE_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, ref, EQE, series_R, VF, emis))
    df2 = df[['Pgen', 'Pseries', 'PEQE', 'Ptherm', 'Pref_loss']]
    df_perc = df2.divide(df2.sum(axis=1), axis=0)
    df['P_net'] = df['Pgen'] + df['Pseries'] + df['PEQE'] + df['Ptherm'] + df['Pref_loss']
    df_data = df[['Eg', 'Pgen']]
    df_data['Trad'] = Trad
    df_data['eff'] = df_perc['Pgen']
    df_data.to_csv('data/P_balance_Trad_%0.0f_ref_%0.2f_EQE_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, ref, EQE, series_R, VF, emis))
    return df_data

def P_balance_3(Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW):
    series_R = series_R * 1e-4
    if BW == 1:
        if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis)):
            df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
            df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
            return df_data
        else:
            if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW)):
                df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
                df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
                return df_data
    else:
        if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW)):
            df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
            df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
            return df_data
    Egs = np.linspace(0.5, 2, 100)
    # Egs = [1.4]
    # Pouts = []
    # series_R given in ohm cm2, convert to ohm m2
    effs = []
    all_Ps = []
    # Trad = 2200
    # ref = 0.95
    # EQE = 0.97
    # series_R = 0.1e-6 # ohm cm2
    # for Trad in np.arange(3000, 9000, 1000):
     
    def get_data(Eg):
        E = np.linspace(0.01, 10, 10000)
        Ps = []
        effs_V = []
        I_gens = []
        Vs = np.linspace(Eg/2, Eg*0.99, 1000)
        # Voc = calc_Voc(E, Eg, Trad, Tc)
        # print(Eg-Voc)
        for V in Vs:
            P_rad_SB = Prad_total_SB(E, Eg, Trad, Tc, V)*VF*emis
            P_rad_AB = Prad_total_AB(E, Eg, Trad, Tc, V, BW)*VF*emis
            P_rad_AB_ref = P_rad_AB * ABR
            P_rad_AB_abs = P_rad_AB * (1-ABR)
            P_recomb_rad = Precomb_PV(E, Eg, Trad, Tc, V, BW)
            P_recomb_nonrad = P_recomb_rad*nonrad_ratio
            P_recomb = P_recomb_rad + P_recomb_nonrad
            P_therm = Ptherm(E, Eg, Trad, Tc, V, BW)*VF*emis*(1-ABR)
            P_BG_Voc = PBG_Voc(E, Eg, Trad, Tc, V, BW)*VF*emis*(1-ABR)
            P_rad_SB_ref = P_rad_SB * SBR
            P_rad_SB_abs = P_rad_SB * (1-SBR)
            P_reabs = P_rad_SB_ref + P_rad_AB_ref
            P_inc = P_rad_AB_abs - P_therm - P_BG_Voc - P_recomb_rad
            # P_IQE = P_recomb_nonrad
            P_gen = P_inc - P_recomb_nonrad
            I_gen = P_gen / V
            P_series = I_gen**2 * series_R
            # P_Voc_offset = max((0.0-(Eg-Voc)) * I_gen,0)
            P_gen = P_gen - P_series
            eff = P_gen / (P_rad_AB_abs + P_rad_SB_abs - P_recomb_rad)
            Ps.append([P_rad_SB_ref, P_rad_SB_abs, P_rad_AB_ref, P_rad_AB_abs, P_recomb_rad, P_therm, P_BG_Voc, P_reabs, P_recomb_nonrad, P_series, P_gen])
            effs_V.append(eff)
            I_gens.append(I_gen)
        df = pd.DataFrame(Ps, columns=['P_rad_SB_ref', 'P_rad_SB_abs', 'P_rad_AB_ref', 'P_rad_AB_abs', 
                          'P_rad_recomb', 'Ptherm', 'P_BG_Voc', 'Preabs', 'P_nonrad_recomb', 'Pseries','Pgen'])
        # plt.figure()
        # plt.plot(Vs, df['Pgen'])
        # plt.ylim(0, max(df['Pgen'])*1.1)
        # plt.show()
        df['effs'] = effs_V
        idx = df.index[df['Pgen'] == max(df['Pgen'])]
        data = df.loc[idx]
        return data

    datas = p_map(get_data, Egs)
    all_Ps = [data.values[0][:-1] for data in datas]
    effs = [data.values[0][-1] for data in datas]

    # for Eg in tqdm(Egs):
    #     Eg = Eg
    #     E = np.linspace(0.01, 10, 10000)
    #     Ps = []
    #     effs_V = []
    #     I_gens = []
    #     Vs = np.linspace(Eg/2, Eg*0.99, 1000)
    #     # Voc = calc_Voc(E, Eg, Trad, Tc)
    #     # print(Eg-Voc)
    #     for V in Vs:
    #         P_rad_SB = Prad_total_SB(E, Eg, Trad, Tc, V)*VF*emis
    #         P_rad_AB = Prad_total_AB(E, Eg, Trad, Tc, V, BW)*VF*emis
    #         P_rad_AB_ref = P_rad_AB * ABR
    #         P_rad_AB_abs = P_rad_AB * (1-ABR)
    #         P_recomb_rad = Precomb_PV(E, Eg, Trad, Tc, V, BW)
    #         P_recomb_nonrad = P_recomb_rad*nonrad_ratio
    #         P_recomb = P_recomb_rad + P_recomb_nonrad
    #         P_therm = Ptherm(E, Eg, Trad, Tc, V, BW)*VF*emis*(1-ABR)
    #         P_BG_Voc = PBG_Voc(E, Eg, Trad, Tc, V, BW)*VF*emis*(1-ABR)
    #         P_rad_SB_ref = P_rad_SB * SBR
    #         P_rad_SB_abs = P_rad_SB * (1-SBR)
    #         P_reabs = P_rad_SB_ref + P_rad_AB_ref
    #         P_inc = P_rad_AB_abs - P_therm - P_BG_Voc - P_recomb_rad
    #         # P_IQE = P_recomb_nonrad
    #         P_gen = P_inc - P_recomb_nonrad
    #         I_gen = P_gen / V
    #         P_series = I_gen**2 * series_R
    #         # P_Voc_offset = max((0.0-(Eg-Voc)) * I_gen,0)
    #         P_gen = P_gen - P_series
    #         eff = P_gen / (P_rad_AB_abs + P_rad_SB_abs - P_recomb_rad)
    #         Ps.append([P_rad_SB_ref, P_rad_SB_abs, P_rad_AB_ref, P_rad_AB_abs, P_recomb_rad, P_therm, P_BG_Voc, P_reabs, P_recomb_nonrad, P_series, P_gen])
    #         effs_V.append(eff)
    #         I_gens.append(I_gen)
    #     df = pd.DataFrame(Ps, columns=['P_rad_SB_ref', 'P_rad_SB_abs', 'P_rad_AB_ref', 'P_rad_AB_abs', 
    #                       'P_rad_recomb', 'Ptherm', 'P_BG_Voc', 'Preabs', 'P_nonrad_recomb', 'Pseries','Pgen'])
    #     # plt.figure()
    #     # plt.plot(Vs, df['Pgen'])
    #     # plt.ylim(0, max(df['Pgen'])*1.1)
    #     # plt.show()
    #     df['effs'] = effs_V
    #     idx = df.index[df['Pgen'] == max(df['Pgen'])]
    #     data = df.loc[idx]
    #     maxV = Vs[idx]
    #     # Ptherm_plot(E, Eg, Trad, Tc, maxV, BW)
    #     all_Ps.append(data.values[0][:-1])
    #     effs.append(data.values[0][-1])
    #     del df

    all_Ps = np.array(all_Ps)/100**2  # W cm-2
    df = pd.DataFrame(all_Ps, columns=['P_rad_SB_ref', 'P_rad_SB_abs', 'P_rad_AB_ref', 'P_rad_AB_abs', 
                                       'P_rad_recomb', 'Ptherm', 'P_BG_Voc', 'Preabs', 'P_nonrad_recomb', 'Pseries','Pgen'])
    df['Eg'] = Egs
    df.to_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
    
    df2 = df[['Pgen', 'Pseries', 'P_nonrad_recomb', 'Ptherm', 'P_BG_Voc', 'P_rad_SB_abs']]
    df_perc = df2.divide(df2.sum(axis=1), axis=0)
    
    df['P_net'] = df['Pgen'] + df['Pseries'] + df['P_nonrad_recomb'] + df['Ptherm'] + df['P_BG_Voc'] + df['P_rad_SB_abs']
    df_data = df[['Eg', 'Pgen']]
    df_data['Trad'] = Trad
    df_data['eff'] = df_perc['Pgen']
    df_data.to_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
    # plt.show()
    return df_data

def P_balance_4(Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW):
    series_R = series_R * 1e-4
    if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW)):
        df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW))
        df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW))
        return df_data
    Egs = np.linspace(0.5, 2, 100)
    effs = []
    all_Ps = []
     
    def get_data(Eg):
        E = np.linspace(0.01, 10, 10000)
        Ps = []
        effs_V = []
        I_gens = []
        Vs = np.linspace(Eg/2, Eg*0.99, 1000)
        # Voc = calc_Voc(E, Eg, Trad, Tc)
        # print(Eg-Voc)
        for V in Vs:
            P_rad_SB = Prad_total_SB(E, Eg, Trad, Tc, V)*VF*SBE
            P_rad_AB = Prad_total_AB(E, Eg, Trad, Tc, V, BW)*VF*ABE
            P_rad_AB_ref = P_rad_AB * ABR
            P_rad_AB_abs = P_rad_AB * (1-ABR)
            P_recomb_rad = Precomb_PV(E, Eg, Trad, Tc, V, BW)
            P_recomb_nonrad = P_recomb_rad*nonrad_ratio
            P_recomb = P_recomb_rad + P_recomb_nonrad
            P_therm = Ptherm(E, Eg, Trad, Tc, V, BW)*VF*ABE*(1-ABR)
            P_BG_Voc = PBG_Voc(E, Eg, Trad, Tc, V, BW)*VF*ABE*(1-ABR)
            P_rad_SB_ref = P_rad_SB * SBR
            P_rad_SB_abs = P_rad_SB * (1-SBR)
            P_reabs = P_rad_SB_ref + P_rad_AB_ref
            P_inc = P_rad_AB_abs - P_therm - P_BG_Voc - P_recomb_rad
            # P_IQE = P_recomb_nonrad
            P_gen = P_inc - P_recomb_nonrad
            I_gen = P_gen / V
            P_series = I_gen**2 * series_R
            # P_Voc_offset = max((0.0-(Eg-Voc)) * I_gen,0)
            P_gen = P_gen - P_series
            eff = P_gen / (P_rad_AB_abs + P_rad_SB_abs - P_recomb_rad)
            Ps.append([P_rad_SB_ref, P_rad_SB_abs, P_rad_AB_ref, P_rad_AB_abs, P_recomb_rad, P_therm, P_BG_Voc, P_reabs, P_recomb_nonrad, P_series, P_gen])
            effs_V.append(eff)
            I_gens.append(I_gen)
        df = pd.DataFrame(Ps, columns=['P_rad_SB_ref', 'P_rad_SB_abs', 'P_rad_AB_ref', 'P_rad_AB_abs', 
                          'P_rad_recomb', 'Ptherm', 'P_BG_Voc', 'Preabs', 'P_nonrad_recomb', 'Pseries','Pgen'])
        # plt.figure()
        # plt.plot(Vs, df['Pgen'])
        # plt.ylim(0, max(df['Pgen'])*1.1)
        # plt.show()
        df['effs'] = effs_V
        idx = df.index[df['Pgen'] == max(df['Pgen'])]
        data = df.loc[idx]
        return data

    datas = p_map(get_data, Egs)
    all_Ps = [data.values[0][:-1] for data in datas]
    effs = [data.values[0][-1] for data in datas]


    all_Ps = np.array(all_Ps)/100**2  # W cm-2
    df = pd.DataFrame(all_Ps, columns=['P_rad_SB_ref', 'P_rad_SB_abs', 'P_rad_AB_ref', 'P_rad_AB_abs', 
                                       'P_rad_recomb', 'Ptherm', 'P_BG_Voc', 'Preabs', 'P_nonrad_recomb', 'Pseries','Pgen'])
    df['Eg'] = Egs
    df.to_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW))
    
    df2 = df[['Pgen', 'Pseries', 'P_nonrad_recomb', 'Ptherm', 'P_BG_Voc', 'P_rad_SB_abs']]
    df_perc = df2.divide(df2.sum(axis=1), axis=0)
    
    df['P_net'] = df['Pgen'] + df['Pseries'] + df['P_nonrad_recomb'] + df['Ptherm'] + df['P_BG_Voc'] + df['P_rad_SB_abs']
    df_data = df[['Eg', 'Pgen']]
    df_data['Trad'] = Trad
    df_data['eff'] = df_perc['Pgen']
    df_data.to_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW))
    # plt.show()
    return df_data

def plot_P_balance_3(Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW):
    P_balance_3(Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW)
    series_R = series_R * 1e-4
    if BW == 1:
        if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis)):
            df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
            df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
    if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW)):
        df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
        df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW))
    df2 = df[['Pgen', 'Pseries', 'P_nonrad_recomb', 'Ptherm', 'P_BG_Voc', 'P_rad_SB_abs']]
    df_perc = df2.divide(df2.sum(axis=1), axis=0)
    fig = plt.figure(num=2, figsize=[8, 4], dpi=300, clear=True)
    fig.add_subplot(121)
    # plt.plot(df['Eg'], df['Prad_total'],  label='Prad_total')
    # plt.plot(df['Eg'], df['Pref'],  label='Pref')
    # plt.plot(df['Eg'], df['Precomb'], label='Precomb')
    # plt.plot(df['Eg'], df['Prad'],  label='Prad_cell')
    # plt.plot(df['Eg'], df['Pgen'], label='Pgen')
    plt.stackplot(df['Eg'], df['Pgen'], df['Pseries'], df['P_nonrad_recomb'], df['P_rad_recomb'], df['Ptherm'], df['P_BG_Voc'], df['P_rad_SB_abs'],
                  df['Preabs'],  labels=['Pgen', 'Pseries', 'P_nonrad_recomb','Prad_recomb', 'Ptherm', 'P_BG_Voc', 'P_SB_abs', 'Preabs'])
    plt.legend()
    plt.xlabel('Eg (eV)')
    plt.ylabel('Power (W/cm$^2$)')
    plt.title('T = {}K'.format(Trad))
    plt.xlim(0.5, 2)
    plt.ylim(0, max(df['P_rad_SB_ref'] + df['P_rad_SB_abs'] + df['P_rad_AB_ref'] + df['P_rad_AB_abs']))
    plt.tight_layout()
    fig.add_subplot(122)
    plt.stackplot(df['Eg'], df_perc['Pgen'], df_perc['Pseries'], df_perc['P_nonrad_recomb'],  np.zeros(len(df['Eg'])),
                  df_perc['Ptherm'], df_perc['P_BG_Voc'], df_perc['P_rad_SB_abs'],
                  labels=['Pgen', 'Pseries', 'P_nonrad_recomb', '', 'Ptherm', 'P_BG_Voc', 'P_SB_abs'])
    plt.xlabel('Eg (eV)')
    plt.ylabel('Frac Pabs')
    plt.title('Efficiency')
    plt.xlim(0.5, 2)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/P_balance_nonideal_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f_BW_%0.2f.png' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis, BW), dpi=300)
    plt.show()

def plot_P_balance_4(Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW):
    P_balance_4(Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW)
    series_R = series_R * 1e-4
    if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW)):
        df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW))
        df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW))
    df2 = df[['Pgen', 'Pseries', 'P_nonrad_recomb', 'Ptherm', 'P_BG_Voc', 'P_rad_SB_abs']]
    df_perc = df2.divide(df2.sum(axis=1), axis=0)
    fig = plt.figure(num=2, figsize=[8, 4], dpi=300, clear=True)
    fig.add_subplot(121)
    # plt.plot(df['Eg'], df['Prad_total'],  label='Prad_total')
    # plt.plot(df['Eg'], df['Pref'],  label='Pref')
    # plt.plot(df['Eg'], df['Precomb'], label='Precomb')
    # plt.plot(df['Eg'], df['Prad'],  label='Prad_cell')
    # plt.plot(df['Eg'], df['Pgen'], label='Pgen')
    plt.stackplot(df['Eg'], df['Pgen'], df['Pseries'], df['P_nonrad_recomb'], df['P_rad_recomb'], df['Ptherm'], df['P_BG_Voc'], df['P_rad_SB_abs'],
                  df['Preabs'],  labels=['Pgen', 'Pseries', 'P_nonrad_recomb','Prad_recomb', 'Ptherm', 'P_BG_Voc', 'P_SB_abs', 'Preabs'])
    plt.legend()
    plt.xlabel('Eg (eV)')
    plt.ylabel('Power (W/cm$^2$)')
    plt.title('T = {}K'.format(Trad))
    plt.xlim(0.5, 2)
    plt.ylim(0, max(df['P_rad_SB_ref'] + df['P_rad_SB_abs'] + df['P_rad_AB_ref'] + df['P_rad_AB_abs']))
    plt.tight_layout()
    fig.add_subplot(122)
    plt.stackplot(df['Eg'], df_perc['Pgen'], df_perc['Pseries'], df_perc['P_nonrad_recomb'],  np.zeros(len(df['Eg'])),
                  df_perc['Ptherm'], df_perc['P_BG_Voc'], df_perc['P_rad_SB_abs'],
                  labels=['Pgen', 'Pseries', 'P_nonrad_recomb', '', 'Ptherm', 'P_BG_Voc', 'P_SB_abs'])
    plt.xlabel('Eg (eV)')
    plt.ylabel('Frac Pabs')
    plt.title('Efficiency')
    plt.xlim(0.5, 2)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/P_balance_nonideal_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_ABE_%0.2f_SBE_%0.2f_BW_%0.2f.png' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, ABE, SBE, BW), dpi=300)
    plt.show()


def P_balance_3_old(Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis):
    series_R = series_R * 1e-4
    if os.path.exists('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis)):
        df = pd.read_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
        df_data = pd.read_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
        return df_data
    Egs = np.linspace(0.5, 2, 100)
    # Egs = [1.4]
    # Pouts = []
    # series_R given in ohm cm2, convert to ohm m2
    effs = []
    all_Ps = []
    # Trad = 2200
    # ref = 0.95
    # EQE = 0.97
    # series_R = 0.1e-6 # ohm cm2
    # for Trad in np.arange(3000, 9000, 1000):
    for Eg in tqdm(Egs):
        Eg = Eg
        E = np.linspace(0.01, 10, 10000)
        Ps = []
        effs_V = []
        Vs = np.linspace(Eg/2, Eg*0.99, 1000)
        # Voc = calc_Voc(E, Eg, Trad, Tc)
        # print(Eg-Voc)
        for V in Vs:
            P_rad_SB = Prad_total_SB(E, Eg, Trad, Tc, V)*VF*emis
            P_rad_AB = Prad_total_AB(E, Eg, Trad, Tc, V)*VF*emis
            P_rad_AB_ref = P_rad_AB * ABR
            P_rad_AB_abs = P_rad_AB * (1-ABR)
            P_recomb_rad = Precomb_PV(E, Eg, Trad, Tc, V)
            P_recomb_nonrad = P_recomb_rad*nonrad_ratio
            P_recomb = P_recomb_rad + P_recomb_nonrad
            P_therm = Ptherm(E, Eg, Trad, Tc, V)*VF*emis*(1-ABR)
            P_rad_SB_ref = P_rad_SB * SBR
            P_rad_SB_abs = P_rad_SB * (1-SBR)
            P_reabs = P_rad_SB_ref + P_rad_AB_ref
            P_inc = P_rad_AB_abs - P_therm - P_recomb_rad
            # P_IQE = P_recomb_nonrad
            P_gen = P_inc - P_recomb_nonrad
            I_gen = P_gen / V
            P_series = I_gen**2 * series_R
            # P_Voc_offset = max((0.0-(Eg-Voc)) * I_gen,0)
            P_gen = P_gen - P_series
            eff = P_gen / (P_rad_AB_abs + P_rad_SB_abs - P_recomb_rad)
            Ps.append([P_rad_SB_ref, P_rad_SB_abs, P_rad_AB_ref, P_rad_AB_abs, P_recomb_rad, P_therm, P_reabs, P_recomb_nonrad, P_series, P_gen])
            effs_V.append(eff)
        df = pd.DataFrame(Ps, columns=['P_rad_SB_ref', 'P_rad_SB_abs', 'P_rad_AB_ref', 'P_rad_AB_abs', 
                          'P_rad_recomb', 'Ptherm', 'Preabs', 'P_nonrad_recomb', 'Pseries','Pgen'])
        df['effs'] = effs_V
        idx = df.index[df['Pgen'] == max(df['Pgen'])]
        data = df.loc[idx]
        all_Ps.append(data.values[0][:-1])
        effs.append(data.values[0][-1])
        del df

    all_Ps = np.array(all_Ps)/100**2  # W cm-2
    df = pd.DataFrame(all_Ps, columns=['P_rad_SB_ref', 'P_rad_SB_abs', 'P_rad_AB_ref', 'P_rad_AB_abs', 
                                       'P_rad_recomb', 'Ptherm', 'Preabs', 'P_nonrad_recomb', 'Pseries','Pgen'])
    df['Eg'] = Egs
    df.to_csv('data/P_balance_all_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
    # fig = plt.figure(num=2, figsize=[8, 4], dpi=300, clear=True)
    # fig.add_subplot(121)
    # # plt.plot(df['Eg'], df['Prad_total'],  label='Prad_total')
    # # plt.plot(df['Eg'], df['Pref'],  label='Pref')
    # # plt.plot(df['Eg'], df['Precomb'], label='Precomb')
    # # plt.plot(df['Eg'], df['Prad'],  label='Prad_cell')
    # # plt.plot(df['Eg'], df['Pgen'], label='Pgen')
    # plt.stackplot(df['Eg'], df['Pgen'], df['Pseries'], df['P_nonrad_recomb'], df['P_rad_recomb'], df['Ptherm'], df['P_rad_SB_abs'],
    #               df['Preabs'],  labels=['Pgen', 'Pseries', 'P_nonrad_recomb','Prad_recomb', 'Ptherm', 'P_SB_abs', 'Preabs'])
    # plt.legend()
    # plt.xlabel('Eg (eV)')
    # plt.ylabel('Power (W/cm$^2$)')
    # plt.title('T = {}K'.format(Trad))
    # plt.xlim(0.5, 2)
    # plt.ylim(0, max(df['P_rad_SB_ref'] + df['P_rad_SB_abs'] + df['P_rad_AB_ref'] + df['P_rad_AB_abs']))
    # plt.tight_layout()
    # fig.add_subplot(122)
    df2 = df[['Pgen', 'Pseries', 'P_nonrad_recomb', 'Ptherm', 'P_rad_SB_abs']]
    df_perc = df2.divide(df2.sum(axis=1), axis=0)
    # plt.stackplot(df['Eg'], df_perc['Pgen'], df_perc['Pseries'], df_perc['P_nonrad_recomb'],  np.zeros(len(df['Eg'])),
    #               df_perc['Ptherm'], df_perc['P_rad_SB_abs'],
    #               labels=['Pgen', 'Pseries', 'P_nonrad_recomb', '', 'Ptherm', 'P_SB_abs'])
    # plt.xlabel('Eg (eV)')
    # plt.ylabel('Frac Pabs')
    # plt.title('Efficiency')
    # plt.xlim(0.5, 2)
    # plt.ylim(0, 1)
    # plt.legend(loc='upper right')
    # plt.tight_layout()
    # plt.savefig('plots/P_balance_nonideal_%0.0f.pdf' % Trad)
    df['P_net'] = df['Pgen'] + df['Pseries'] + df['P_nonrad_recomb'] + df['Ptherm'] + df['P_rad_SB_abs']
    df_data = df[['Eg', 'Pgen']]
    df_data['Trad'] = Trad
    df_data['eff'] = df_perc['Pgen']
    df_data.to_csv('data/P_balance_Trad_%0.0f_SBR_%0.2f_ABR_%0.2f_nonrad_%0.2f_seriesR_%0.2e_VF_%0.2f_emis_%0.2f.csv' % (Trad, SBR, ABR, nonrad_ratio, series_R, VF, emis))
    plt.show()
    return df_data


if __name__ == '__main__':

    plt.figure()
    plt.plot(E, P, label=Trad)
    plt.xlim(0, 3)
    plt.xlabel('energy (eV)')
    plt.ylabel('power (W/m2/eV)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('2400_P_eV.png')
    # plt.show()
    # Prad = np.trapz(P, E) / 100**2  # W cm-2
    # Trad_const()
    # BG_const()
    # single_T_BG_const(2400, 1.4)
    # single_T_BG_const(2400, 1.2)
    # multi_2_T_BG_const(2400, 1.4, 1.2)
    # multi_2_T_BG(2400, 1.4)


    # Prad_total_e, Trad = P_balance_14()
    # A_emit = 1e-5
    # polyfit_prods_h, polyfit_H2_h, polyfit_H2_T, polyfit_O2_h, polyfit_O2_T, \
    #     h_reacts, h_H2, h_O2, Y_H2, Y_O2 = H2_combustion()
    # flowrate_H2, T_avg, Tout_prods, Tout_prods_O2HX, Tout_exhaust, Tout_O2HX, Tout_H2HX = calc_H2_flowrate(
    #     Prad_total_e, Trad, A_emit, polyfit_prods_h, polyfit_H2_h, polyfit_O2_T, h_reacts, Y_H2, Y_O2)
    # print('flowrate_H2:\t %0.2e kg/s' % flowrate_H2)
    # print('T_avg:\t %0.2f K' % T_avg)
    # print('Tout_prods:\t %0.2f K' % Tout_prods)
    # print('Tout_prods_O2HX:\t %0.2f K' % Tout_prods_O2HX)
    # print('Tout_exhaust:\t %0.2f K' % Tout_exhaust)
    # print('Tout_O2HX:\t %0.2f K' % Tout_O2HX)
    # print('Tout_H2HX:\t %0.2f K' % Tout_H2HX)
    # P_balance_14_test()
    # P_balance_2_ideal(Trad=2200)
    # P_balance_2(Trad=2123, ref=1, EQE=1, series_R=0, VF=1, emis=1)
    # P_balance_2(Trad=2123, ref=1, EQE=1, series_R=0, VF=0.31, emis=1)
    # P_balance_2(Trad=2123, ref=0.95, EQE=0.7, series_R=6.5e-3, VF=0.31, emis=1)
    # P_balance_2(Trad=2123, ref=0.95, EQE=0.7, series_R=6.5e-3, VF=0.31, emis=0.85)
    # P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, emis=0.95, BW=0.1)
    plot_P_balance_4(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=12, series_R=6.5e-3, VF=0.31, ABE=0.85, SBE=0.85, BW=1)
    # plot_P_balance_3(Trad=2123, SBR=0.95, ABR=0.3, nonrad_ratio=0, series_R=6.5e-3, VF=0.31, emis=0.95, BW=1)
    # sweep_P_balance()
    # P_balance_T(1.4)
    # P_therm_14_test()
    # get_P_therm_frac_func_multi()
    # P_balance_T(0.9)
    # analyze_effective_emissivity()
    # analyze_Pnet_vs_Trad()
    # H2_combustion_sweep()
