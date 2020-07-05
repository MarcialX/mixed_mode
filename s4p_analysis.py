# -*- coding: utf-8 -*-
"""
===================================================================
S4P mixed mode. To get the de-embedded S4P matrix
===================================================================
"""

import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

plt.ion()


def get_data_text(filename, ports):

    # Open file
    data_file = open(filename)

    trig = False
    data = []
    freq = []

    ports = ports-1
    S4P = np.ones((4,4), dtype=complex)

    cnt = 0
    len_array = 9
    lines = data_file.readlines()
    for line in lines:
        if trig:
            col = 0
            value = ""
            s_data = [[]]*len_array
            for char in line:
                if char == ' ':
                    s_data[col] = float(value)
                    col = col + 1
                    char = ""
                    value = ""
                else:
                    value += char

            s_data[-1] = float(value)

            if cnt%4 == 0:
                len_array = 8
                f = s_data[0]
                S4P[ports[0]][ports[0]] = s_data[1] + 1j*s_data[2]
                S4P[ports[0]][ports[1]] = s_data[3] + 1j*s_data[4]
                S4P[ports[0]][ports[2]] = s_data[5] + 1j*s_data[6]
                S4P[ports[0]][ports[3]] = s_data[7] + 1j*s_data[8]
            elif cnt%4 == 1:
                S4P[ports[1]][ports[0]] = s_data[0] + 1j*s_data[1]
                S4P[ports[1]][ports[1]] = s_data[2] + 1j*s_data[3]
                S4P[ports[1]][ports[2]] = s_data[4] + 1j*s_data[5]
                S4P[ports[1]][ports[3]] = s_data[6] + 1j*s_data[7]
            elif cnt%4 == 2:
                S4P[ports[2]][ports[0]] = s_data[0] + 1j*s_data[1]
                S4P[ports[2]][ports[1]] = s_data[2] + 1j*s_data[3]
                S4P[ports[2]][ports[2]] = s_data[4] + 1j*s_data[5]
                S4P[ports[2]][ports[3]] = s_data[6] + 1j*s_data[7]
            elif cnt%4 == 3:
                len_array = 9
                S4P[ports[3]][ports[0]] = s_data[0] + 1j*s_data[1]
                S4P[ports[3]][ports[1]] = s_data[2] + 1j*s_data[3]
                S4P[ports[3]][ports[2]] = s_data[4] + 1j*s_data[5]
                S4P[ports[3]][ports[3]] = s_data[6] + 1j*s_data[7]

                freq.append(f)
                data.append([S4P[0][0],S4P[0][1],S4P[0][2],S4P[0][3],
                            S4P[1][0],S4P[1][1],S4P[1][2],S4P[1][3],
                            S4P[2][0],S4P[2][1],S4P[2][2],S4P[2][3],
                            S4P[3][0],S4P[3][1],S4P[3][2],S4P[3][3]])
            cnt += 1

        if line.startswith('#'):
            trig = True

    data_file.close()

    return np.array(freq), np.array(data)

def s4ptosmm(s4p):
    # Differential-Differential
    SDD11 = 0.5*(s4p[0]-s4p[2]-s4p[8]+s4p[10])
    SDD12 = 0.5*(s4p[1]-s4p[3]-s4p[9]+s4p[11])
    SDD21 = 0.5*(s4p[4]-s4p[6]-s4p[12]+s4p[14])
    SDD22 = 0.5*(s4p[5]-s4p[7]-s4p[13]+s4p[15])
    SDD = np.array([SDD11, SDD12, SDD21, SDD22])

    # Differential-Common
    SDC11 = 0.5*(s4p[0]+s4p[2]-s4p[8]-s4p[10])
    SDC12 = 0.5*(s4p[1]+s4p[3]-s4p[9]-s4p[11])
    SDC21 = 0.5*(s4p[4]+s4p[6]-s4p[12]-s4p[14])
    SDC22 = 0.5*(s4p[5]+s4p[7]-s4p[13]-s4p[15])
    SDC = np.array([SDC11, SDC12, SDC21, SDC22])

    # Common-Differential
    SCD11 = 0.5*(s4p[0]-s4p[2]+s4p[8]-s4p[10])
    SCD12 = 0.5*(s4p[1]-s4p[3]+s4p[9]-s4p[11])
    SCD21 = 0.5*(s4p[4]-s4p[6]+s4p[12]-s4p[14])
    SCD22 = 0.5*(s4p[5]-s4p[7]+s4p[13]-s4p[15])
    SCD = np.array([SCD11, SCD12, SCD21, SCD22])

    # Differential-Differential
    SCC11 = 0.5*(s4p[0]+s4p[2]+s4p[8]+s4p[10])
    SCC12 = 0.5*(s4p[1]+s4p[3]+s4p[9]+s4p[11])
    SCC21 = 0.5*(s4p[4]+s4p[6]+s4p[12]+s4p[14])
    SCC22 = 0.5*(s4p[5]+s4p[7]+s4p[13]+s4p[15])
    SCC = np.array([SCC11, SCC12, SCC21, SCC22])

    S = np.array([SDD, SDC, SCD, SCC])

    return S

def smmtos4p(smm):
    S4P11 = 0.5*(smm[0]+smm[2]+smm[8]+smm[10])
    S4P12 = 0.5*(-smm[0]+smm[2]-smm[8]+smm[10])
    S4P13 = 0.5*(smm[1]+smm[3]+smm[9]+smm[11])
    S4P14 = 0.5*(-smm[1]+smm[3]-smm[9]+smm[11])

    S4P21 = 0.5*(-smm[0]-smm[2]+smm[8]+smm[10])
    S4P22 = 0.5*(smm[0]-smm[2]-smm[8]+smm[10])
    S4P23 = 0.5*(-smm[1]-smm[3]+smm[9]+smm[11])
    S4P24 = 0.5*(smm[1]-smm[3]-smm[9]+smm[11])

    S4P31 = 0.5*(smm[4]+smm[6]+smm[12]+smm[14])
    S4P32 = 0.5*(-smm[4]+smm[6]-smm[12]+smm[14])
    S4P33 = 0.5*(smm[5]+smm[7]+smm[13]+smm[15])
    S4P34 = 0.5*(-smm[5]+smm[7]-smm[13]+smm[15])

    S4P41 = 0.5*(-smm[4]-smm[6]+smm[12]+smm[14])
    S4P42 = 0.5*(smm[4]-smm[6]-smm[12]+smm[14])
    S4P43 = 0.5*(-smm[5]-smm[7]+smm[13]+smm[15])
    S4P44 = 0.5*(smm[5]-smm[7]-smm[13]+smm[15])

    S = np.array([S4P11, S4P12, S4P13, S4P14,
                  S4P21, S4P22, S4P23, S4P24,
                  S4P31, S4P32, S4P33, S4P34,
                  S4P41, S4P42, S4P43, S4P44])

    return S

def stowcm(S):

    det = S[0][0]*S[1][1] - S[0][1]*S[1][0]

    wcm = np.ones((2,2),  dtype=complex)
    wcm[0][0] = -det
    wcm[0][1] = S[0][0]
    wcm[1][0] = -S[1][1]
    wcm[1][1] = 1

    wcm = (1/S[1][0])*wcm

    return wcm

def eqn_order_2(a,b,c):
    x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

    return x1,x2

def root_criteria(x1, x2):
    if abs(x1) < abs(x2):
        return x2, x1
    else:
        return x1, x2

def get_eigenvals(T, ac, b):

    lambda_1 = (ac*T[0][0] - b*ac*T[1][0] - b*T[1][1] + T[0][1])/(ac - b)
    lambda_2 = (ac - b)/(ac*T[1][1] + b*ac*T[1][0] - b*T[0][0] - T[0][1])

    return lambda_1, lambda_2

def get_abc(S1, S2):

    ABC = []
    L = []

    for f in range(len(S1)):

        # Convert from S to WCM
        S1p = np.ones((2,2), dtype=complex)
        S1p[0][0] = S1[f][0]
        S1p[0][1] = S1[f][1]
        S1p[1][0] = S1[f][2]
        S1p[1][1] = S1[f][3]

        M1 = stowcm(S1p)

        # Convert from S to WCM
        S2p = np.ones((2,2), dtype=complex)
        S2p[0][0] = S2[f][0]
        S2p[0][1] = S2[f][1]
        S2p[1][0] = S2[f][2]
        S2p[1][1] = S2[f][3]

        M2 = stowcm(S2p)

        T = M1.dot(np.linalg.inv(M2))

        # Get a/c y b
        x1, x2 = eqn_order_2(T[1][0], (T[1][1] - T[0][0]), -T[0][1])
        #x1, x2 = np.roots([T[1][0], (T[1][1] - T[0][0]), -T[0][1]])
        ac, b = root_criteria(x1, x2)

        # Get lambda_1 y lambda_2
        lambda_1, lambda_2 = get_eigenvals(T, ac, b)

        ABC.append([ac, b])
        L.append([lambda_1, lambda_2])

    return ABC, L


def get_propag(L, l1, l2, val=0):

    alpha = []
    for f in range(len(L)):
        ln = np.log(L[f][val])

        alpha.append(ln*(1/(l2-l1)))

    return np.array(alpha)

def beta_disc_corrector(freq, beta, region=[10,40], smooth=False, threshold=10):

    # Locate discontinuities
    betap = np.diff(beta)/np.diff(freq)

    # Ajuste de los datos en el intervalo est_area
    fit = np.polyfit(freq[region[0]:region[1]], beta[region[0]:region[1]], 1)
    p = np.poly1d(fit)
    m = fit[0]

    # Discontinuities, position
    if smooth:
        window_size = 4
        absbeta = abs(betap[:len(betap)/2])
        min_beta = []
        for b in range(window_size/2, len(absbeta)-window_size/2):
            beta_min = np.min(absbeta[b-window_size/2:b+window_size/2])
            min_beta.append(beta_min)

        popt, pcov = curve_fit(lambda t, a, b, mx, my: a*(t-mx)**b + my, freq[window_size/2:len(absbeta)-window_size/2], min_beta, p0=(min_beta[0],-0.5, freq[0], min_beta[-1]))

        baseline = popt[0]*(freq[:-1]-popt[2])**popt[1] + popt[3]
        betap = abs(betap) - baseline
        m = m - baseline[0]

    discon = np.where(abs(betap) > abs(m)*threshold)[0]
    discon = discon + 1
    discon = np.concatenate(([0], discon, [len(beta)]))

    # Fine adjust
    for i in range(len(discon)-2):
        freq_sec = freq[discon[i]:discon[i+1]]
        beta_sec = beta[discon[i]:discon[i+1]]

        fit_sec = np.polyfit(freq_sec, beta_sec, 1)
        p_sec = np.poly1d(fit_sec)
        offset_sec = p_sec(freq[discon[i+1]]) - beta[discon[i+1]]

        beta[discon[i+1]:discon[i+2]] = offset_sec + beta[discon[i+1]:discon[i+2]]

    return beta

def get_ediel_us(eff, h, w):

    if w/h < 1:
        k = (1+12*(h/w))**(-0.5) + 0.04*(1-(w/h))**2
    else:
        k = (1+12*(h/w))**(-0.5)

    er = (2*eff + k - 1)/(1+k)

    return er

def get_eff(freq, propag, ind_corr = False):

    alpha = propag.real
    beta = propag.imag

    w = 2*np.pi*freq

    if ind_corr:
        e_eff = ( (beta-alpha)*c/w )**2
    else:
        e_eff = ( (beta)*c/w )**2

    return e_eff

def weight_avg_prop(freq, X_dat, ispan=10):

    X_av = np.zeros(len(X_dat[0])-2*ispan)

    # Read all the data
    for x in range(ispan, len(freq)-ispan):

        # Get m_n|f_x
        pn = np.zeros(len(X_dat))
        for n in range(len(X_dat)):
            mn = 0.
            for i in range(-ispan, ispan-1):
                mn += abs( (X_dat[n][x+i+1]-X_dat[n][x+i])/(freq[x+i+1]-freq[x+i]) )
            # pn
            mn = (1/(2.*ispan))*mn
            pn[n] = 1/mn

        # Get the avg average
        x_av_p = 0.
        ptot = sum(pn)
        for j in range(len(pn)):
            x_av_p += (pn[j]/ptot)*X_dat[j][x]

        X_av[x-ispan] = x_av_p

    freq = freq[ispan:len(X_dat[0])-ispan]

    return freq, X_av


def get_Zc_real(f_beta, C):

    f = f_beta[0]
    beta = f_beta[1]

    # Get Zc real. Zc = propagacion/(j*w*C)
    w = 2*np.pi*f
    Zc_real = beta/(w*C)

    return Zc_real

def get_Zc_imag(f, propag, tand, C):

    alpha = propag.real
    beta = propag.imag

    # Get Zc imag. Zc = propagacion/(j*w*C)
    w = 2*np.pi*f
    Zc_imag = - (alpha-beta*tand)/(w*C)

    return Zc_imag

# Conversion S-matrix to ABCD-matrix
def stoabcd(S, Z0):

    abcd = np.ones((2,2),  dtype=complex)
    abcd[0][0] = ( (1+S[0][0])*(1-S[1][1]) + S[0][1]*S[1][0] )/(2*S[1][0])
    abcd[0][1] = Z0*( (1+S[0][0])*(1+S[1][1]) - S[0][1]*S[1][0] )/(2*S[1][0])
    abcd[1][0] = (1/Z0)*( (1-S[0][0])*(1-S[1][1]) - S[0][1]*S[1][0] )/(2*S[1][0])
    abcd[1][1] = ( (1-S[0][0])*(1+S[1][1]) + S[0][1]*S[1][0] )/(2*S[1][0])

    return abcd

# Conversion ABCD-matrix to S-matrix
def abcdtos(ABCD, Z0):

    s = np.ones((2,2),  dtype=complex)
    ABCD_den = ABCD[0][0]+ABCD[0][1]/Z0+ABCD[1][0]*Z0+ABCD[1][1]
    s[0][0] = (ABCD[0][0]+ABCD[0][1]/Z0-ABCD[1][0]*Z0-ABCD[1][1])/ABCD_den
    s[0][1] = (2*(ABCD[0][0]*ABCD[1][1]-ABCD[0][1]*ABCD[1][0]))/ABCD_den
    s[1][0] = 2./ABCD_den
    s[1][1] = (-ABCD[0][0]+ABCD[0][1]/Z0-ABCD[1][0]*Z0+ABCD[1][1])/ABCD_den

    return s

# Get Zc from ABCD matrix
def get_Zc_from_ABCD(S, Z0):

    Zc = np.zeros(len(S))
    for x in range(len(S)):
        Sp = np.ones((2,2), dtype=complex)
        Sp[0][0] = S[x][0]
        Sp[0][1] = S[x][1]
        Sp[1][0] = S[x][2]
        Sp[1][1] = S[x][3]

        ABCD = stoabcd(Sp, Z0)
        Zc[x] = np.sqrt(ABCD[0][1]/ABCD[1][0])

    return Zc

# Get RGLC parameters, Zc and propagation constant are complex
def get_RGLC(f, propag, Zc):

    L = (propag*Zc).imag/(2*np.pi*f)
    C = (propag/Zc).imag/(2*np.pi*f)

    R = (propag*Zc).real
    G = (propag/Zc).real

    return R,G,L,C

# Get ABCD of a coupled transmission line from gamma c, Zc and length
def abcd_from_zc_gc(gamma_c, Zc, l):

    ABCD = []
    for x in range(len(gamma_c)):
        ABCD_ax = np.ones((2,2), dtype=complex)
        ABCD_ax[0][0] = np.cosh(gamma_c[x]*l)
        ABCD_ax[0][1] = Zc[x]*np.sinh(gamma_c[x]*l)
        ABCD_ax[1][0] = (1/Zc[x])*np.sinh(gamma_c[x]*l)
        ABCD_ax[1][1] = np.cosh(gamma_c[x]*l)

        ABCD.append(ABCD_ax)

    return ABCD

# Build Mixed mode matrix de-embedded
def smm_de(SDD, SCC):

    SMM = []
    for x in range(len(SDD)):
        SMM_ax = [[]]*16
        SMM_ax[0] = SDD[x][0][0]
        SMM_ax[1] = SDD[x][0][1]
        SMM_ax[2] = 0
        SMM_ax[3] = 0

        SMM_ax[4] = SDD[x][1][0]
        SMM_ax[5] = SDD[x][1][1]
        SMM_ax[6] = 0
        SMM_ax[7] = 0

        SMM_ax[8] = 0
        SMM_ax[9] = 0
        SMM_ax[10] = SCC[x][0][0]
        SMM_ax[11] = SCC[x][0][1]

        SMM_ax[12] = 0
        SMM_ax[13] = 0
        SMM_ax[14] = SCC[x][1][0]
        SMM_ax[15] = SCC[x][1][1]

        SMM.append(SMM_ax)

    return np.array(SMM)


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='18')

c = 299.792458e6

# Cargamos los archivos
ports = np.array([1,3,2,4])
f1, S1 = get_data_text('TOP_90_30400um.s4p', ports)
f2, S2 = get_data_text('TOP_90_62320um.s4p', ports)

# Convert to Mixed Mode
SMM1 = []
SMM2 = []

SDD_L1, SCC_L1 = [], []
SDD_L2, SCC_L2 = [], []

for i in range(len(S1)):
    SMM1_aux = s4ptosmm(S1[i])
    SMM2_aux = s4ptosmm(S2[i])
    SMM1.append(SMM1_aux)
    SMM2.append(SMM2_aux)

    SDD_L1.append(SMM1_aux[0])
    SCC_L1.append(SMM1_aux[3])

    SDD_L2.append(SMM2_aux[0])
    SCC_L2.append(SMM2_aux[3])

# Graphics
heat = plt.cm.get_cmap('rainbow', 6)

# Test all the combinations
cnt = 0
m, n = 0, 0

lon = np.array([30400E-6, 62320E-6])

print "Lineas: ", lon[0], lon[1]

print "Matrices DD"
ABC_DD, L_DD = get_abc(SDD_L1, SDD_L2)

propag_DD = get_propag(L_DD, lon[0], lon[1])
alpha_DD = propag_DD.real
beta_DD = beta_disc_corrector(f1, propag_DD.imag)
propag_DD_corr = alpha_DD+1j*beta_DD

print "Matrices CC"
ABC_CC, L_CC = get_abc(SCC_L1, SCC_L2)

propag_CC = get_propag(L_CC, lon[0], lon[1])
alpha_CC = propag_CC.real
beta_CC = beta_disc_corrector(f1, propag_CC.imag)
propag_CC_corr= alpha_CC+1j*beta_CC

# plt.figure()
# plt.title(r'Attenuation per unit length')
# plt.plot(f1, alpha_DD, 'r', label=r'$S_{DD}$')
# plt.plot(f1, alpha_CC, 'b', label=r'$S_{CC}$')
# plt.xlabel(r'Frequency [Hz]')
# plt.ylabel(r'$\alpha$[Np/m]')
# plt.legend()
#
# plt.figure()
# plt.title('Phase per unit length')
# plt.plot(f1, beta_DD, 'r', label=r'$S_{DD}$')
# plt.plot(f1, beta_CC, 'b', label=r'$S_{CC}$')
# plt.xlabel(r'Frequency [Hz]')
# plt.ylabel(r'$\beta$ [rad/m]')
# plt.legend()

# To get Zc of Differential-Differential mode
Z0_DD = 100.
Zc_DD = get_Zc_from_ABCD(SDD_L1, Z0_DD)

Zc_DD = Zc_DD[1:100]
X_beta_DD = beta_DD[1:100]
freq_zc = f1[1:100]

# Get C fitting Zc curve
popt_DD, pcov_DD = curve_fit(get_Zc_real, (freq_zc, X_beta_DD), Zc_DD, p0=10e-10)
Cap_DD = popt_DD[0]
print "Capacitance C = ", Cap_DD/1e-12, "pF"

# Get Zc
Zc_DD_real = get_Zc_real((f1, beta_DD), Cap_DD)

tand = 0.002
Zc_DD_imag = get_Zc_imag(f1, propag_DD, tand, Cap_DD)
Zc_DD_cmplx = Zc_DD_real + 1j*Zc_DD_imag


# To get Zc of Common-Common mode
Z0_CC = 25.
Zc_CC = get_Zc_from_ABCD(SCC_L1, Z0_CC)

Zc_CC = Zc_CC[1:100]
X_beta_CC = beta_CC[1:100]

# Get C fitting Zc curve
popt_CC, pcov_CC = curve_fit(get_Zc_real, (freq_zc, X_beta_CC), Zc_CC, p0=10e-10)
Cap_CC = popt_CC[0]
print "Capacitance C = ", Cap_CC/1e-12, "pF"

# Get Zc
Zc_CC_real = get_Zc_real((f1, beta_CC), Cap_CC)

Zc_CC_imag = get_Zc_imag(f1, propag_CC, tand, Cap_CC)
Zc_CC_cmplx = Zc_CC_real + 1j*Zc_CC_imag


# Get ABCD_DD from gamma_c, Zc and length
ABCD_DD_L1 = abcd_from_zc_gc(propag_DD_corr, Zc_DD_cmplx, lon[0])
ABCD_DD_L2 = abcd_from_zc_gc(propag_DD_corr, Zc_DD_cmplx, lon[1])

SDD1_de_embed = []
SDD2_de_embed = []
SDD_L1_de_embed = []
SDD_L2_de_embed = []
for i, f in enumerate(f1):
    SDD1_de_embed = abcdtos(ABCD_DD_L1[i], Z0_DD)
    SDD_L1_de_embed.append(SDD1_de_embed)
    SDD2_de_embed = abcdtos(ABCD_DD_L2[i], Z0_DD)
    SDD_L2_de_embed.append(SDD2_de_embed)

# Get ABCD_CC from gamma_c, Zc and length
ABCD_CC_L1 = abcd_from_zc_gc(propag_CC_corr, Zc_CC_cmplx, lon[0])
ABCD_CC_L2 = abcd_from_zc_gc(propag_CC_corr, Zc_CC_cmplx, lon[1])

SCC1_de_embed = []
SCC2_de_embed = []
SCC_L1_de_embed = []
SCC_L2_de_embed = []
for i, f in enumerate(f1):
    SCC1_de_embed = abcdtos(ABCD_CC_L1[i], Z0_CC)
    SCC_L1_de_embed.append(SCC1_de_embed)
    SCC2_de_embed = abcdtos(ABCD_CC_L2[i], Z0_CC)
    SCC_L2_de_embed.append(SCC2_de_embed)

# SMM linea 1
SMM_L1 = smm_de(SDD_L1_de_embed, SCC_L1_de_embed)

# SMM linea 2
SMM_L2 = smm_de(SDD_L2_de_embed, SCC_L2_de_embed)

S4P_L1_DE = []
S4P_L2_DE = []
for i, f in enumerate(f1):
    S4P_L1_DE.append(smmtos4p(SMM_L1[i]))
    S4P_L2_DE.append(smmtos4p(SMM_L2[i]))

S4P_L1_DE = np.array(S4P_L1_DE)
S4P_L2_DE = np.array(S4P_L2_DE)

# Línea 1. Original vs De-embedded
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# S11
axs[0,0].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,0])),'r')
axs[0,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,0])),'b')
axs[0,0].text(30, -70, r'\textbf{S$_{11}$}')
axs[0,0].set_ylabel('[dB]')
# S12
axs[0,1].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,1])),'r')
axs[0,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,2])),'b')
axs[0,1].text(30, -70, r'\textbf{S$_{12}$}')
# S13
axs[0,2].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,2])),'r')
axs[0,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,1])),'b')
axs[0,2].text(30, -70, r'\textbf{S$_{13}$}')
# S14
axs[0,3].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,3])),'r', label=r'Original')
axs[0,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,3])),'b', label=r'De-embedded')
axs[0,3].text(30, -70, r'\textbf{S$_{14}$}')
axs[0,3].legend()
# S21
axs[1,0].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,4])),'r')
axs[1,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,8])),'b')
axs[1,0].text(30, -70, r'\textbf{S$_{21}$}')
axs[1,0].set_ylabel('[dB]')
# S22
axs[1,1].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,5])),'r')
axs[1,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,10])),'b')
axs[1,1].text(30, -70, r'\textbf{S$_{22}$}')
# S23
axs[1,2].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,6])),'r')
axs[1,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,9])),'b')
axs[1,2].text(30, -70, r'\textbf{S$_{23}$}')
# S24
axs[1,3].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,7])),'r')
axs[1,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,11])),'b')
axs[1,3].text(30, -70, r'\textbf{S$_{24}$}')
# S31
axs[2,0].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,8])),'r')
axs[2,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,4])),'b')
axs[2,0].set_ylabel('[dB]')
axs[2,0].text(30, -70, r'\textbf{S$_{31}$}')
# S32
axs[2,1].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,9])),'r')
axs[2,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,6])),'b')
axs[2,1].text(30, -70, r'\textbf{S$_{32}$}')
# S33
axs[2,2].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,10])),'r')
axs[2,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,5])),'b')
axs[2,2].text(30, -70, r'\textbf{S$_{33}$}')
# S34
axs[2,3].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,11])),'r')
axs[2,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,7])),'b')
axs[2,3].text(30, -70, r'\textbf{S$_{34}$}')
# S41
axs[3,0].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,12])),'r')
axs[3,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,12])),'b')
axs[3,0].set_ylabel('[dB]')
axs[3,0].set_xlabel('Frequency[GHz]')
axs[3,0].text(30, -70, r'\textbf{S$_{41}$}')
# S42
axs[3,1].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,13])),'r')
axs[3,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,14])),'b')
axs[3,1].set_xlabel('Frequency[GHz]')
axs[3,1].text(30, -70, r'\textbf{S$_{42}$}')
# S43
axs[3,2].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,14])),'r')
axs[3,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,13])),'b')
axs[3,2].set_xlabel('Frequency[GHz]')
axs[3,2].text(30, -70, r'\textbf{S$_{43}$}')
# S44
axs[3,3].plot(f1*1e-9, 20*np.log10(np.abs(S1[:,15])),'r')
axs[3,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L1_DE[:,15])),'b')
axs[3,3].set_xlabel('Frequency[GHz]')
axs[3,3].text(30, -70, r'\textbf{S$_{44}$}')

fig.suptitle(r'Line 1. Comparison Original-De embedded')


# Línea 2. Original vs De-embedded
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# S11
axs[0,0].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,0])),'r')
axs[0,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,0])),'b')
axs[0,0].text(30, -70, r'\textbf{S$_{11}$}')
axs[0,0].set_ylabel('[dB]')
# S12
axs[0,1].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,1])),'r')
axs[0,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,2])),'b')
axs[0,1].text(30, -70, r'\textbf{S$_{12}$}')
# S13
axs[0,2].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,2])),'r')
axs[0,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,1])),'b')
axs[0,2].text(30, -70, r'\textbf{S$_{13}$}')
# S14
axs[0,3].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,3])),'r', label=r'Original')
axs[0,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,3])),'b', label=r'De-embedded')
axs[0,3].text(30, -70, r'\textbf{S$_{14}$}')
axs[0,3].legend()
# S21
axs[1,0].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,4])),'r')
axs[1,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,8])),'b')
axs[1,0].text(30, -70, r'\textbf{S$_{21}$}')
axs[1,0].set_ylabel('[dB]')
# S22
axs[1,1].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,5])),'r')
axs[1,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,10])),'b')
axs[1,1].text(30, -70, r'\textbf{S$_{22}$}')
# S23
axs[1,2].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,6])),'r')
axs[1,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,9])),'b')
axs[1,2].text(30, -70, r'\textbf{S$_{23}$}')
# S24
axs[1,3].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,7])),'r')
axs[1,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,11])),'b')
axs[1,3].text(30, -70, r'\textbf{S$_{24}$}')
# S31
axs[2,0].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,8])),'r')
axs[2,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,4])),'b')
axs[2,0].set_ylabel('[dB]')
axs[2,0].text(30, -70, r'\textbf{S$_{31}$}')
# S32
axs[2,1].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,9])),'r')
axs[2,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,6])),'b')
axs[2,1].text(30, -70, r'\textbf{S$_{32}$}')
# S33
axs[2,2].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,10])),'r')
axs[2,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,5])),'b')
axs[2,2].text(30, -70, r'\textbf{S$_{33}$}')
# S34
axs[2,3].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,11])),'r')
axs[2,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,7])),'b')
axs[2,3].text(30, -70, r'\textbf{S$_{34}$}')
# S41
axs[3,0].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,12])),'r')
axs[3,0].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,12])),'b')
axs[3,0].set_ylabel('[dB]')
axs[3,0].set_xlabel('Frequency[GHz]')
axs[3,0].text(30, -70, r'\textbf{S$_{41}$}')
# S42
axs[3,1].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,13])),'r')
axs[3,1].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,14])),'b')
axs[3,1].set_xlabel('Frequency[GHz]')
axs[3,1].text(30, -70, r'\textbf{S$_{42}$}')
# S43
axs[3,2].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,14])),'r')
axs[3,2].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,13])),'b')
axs[3,2].set_xlabel('Frequency[GHz]')
axs[3,2].text(30, -70, r'\textbf{S$_{43}$}')
# S44
axs[3,3].plot(f1*1e-9, 20*np.log10(np.abs(S2[:,15])),'r')
axs[3,3].plot(f1*1e-9, 20*np.log10(np.abs(S4P_L2_DE[:,15])),'b')
axs[3,3].set_xlabel('Frequency[GHz]')
axs[3,3].text(30, -70, r'\textbf{S$_{44}$}')

fig.suptitle(r'Line 2. Comparison Original-De embedded')
