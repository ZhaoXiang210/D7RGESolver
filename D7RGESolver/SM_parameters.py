import numpy as np
from math import  sqrt, pi 
import ckmutil
import ckmutil.ckm




# Default values for SM parameters: MSbar parameters at M_Z (except GF)
GF = 1.1663787e-5

Vus = 0.2243
Vub = 3.62e-3
Vcb = 4.221e-2
gamma = 1.27
m_e = 0.000511
m_mu = 0.1057
m_tau = 1.777
m_u = 0.00127  # mu(2 GeV)=0.0022
m_c = 0.635  # mc(mc)=1.28
m_d = 0.00270  # md(2 GeV)=0.0047
m_s = 0.0551  # ms(2 GeV)=0.095
m_b = 2.85  # mb(mb)=4.18
m_t = 169.0
m_h = 130.6
MW = 80.20
v=sqrt(1 / sqrt(2) / GF)

alpha_e = 1 / 127.9
alpha_s = 0.1185
e=sqrt(4 * pi * alpha_e)
g = 2 * MW / v
gp = e * g / sqrt(g**2 - e**2)
gs = sqrt(4 * pi * alpha_s)
Lambda = m_h**2 / (2 * v**2)
muh = -Lambda * v**2








CKM = ckmutil.ckm.ckm_tree(Vus, Vub, Vcb, gamma)
Me = np.diag([m_e, m_mu, m_tau])

Mu_down = CKM.conj().T @ np.diag([m_u, m_c, m_t])
Md_down = np.diag([m_d, m_s, m_b])
Yd_down = Md_down / (v / sqrt(2))
Yu_down = Mu_down / (v / sqrt(2))
Yl = Me / (v / sqrt(2))

C_in_SM_down = {}

# Yu_downbasis
for i in range(3):
    for j in range(3):
        C_in_SM_down[f"Yu{i+1}{j+1}"] = Yu_down[i][j]

# Yd_downbasis
for i in range(3):
    for j in range(3):
        C_in_SM_down[f"Yd{i+1}{j+1}"] = Yd_down[i][j]

# Yl
for i in range(3):
    for j in range(3):
        C_in_SM_down[f"Yl{i+1}{j+1}"] = Yl[i][j]



C_in_SM_down["gp"] = gp
C_in_SM_down["g"] = g
C_in_SM_down["gs"] = gs
C_in_SM_down["Lambda"] = Lambda
C_in_SM_down["muh"] = muh



### up quark mass basis
Mu_up = np.diag([m_u, m_c, m_t])
Md_up = CKM @ np.diag([m_d, m_s, m_b])




Yd_up = Md_up / (v / sqrt(2))
Yu_up = Mu_up / (v / sqrt(2))


C_in_SM_up = {}

# Yu_downbasis
for i in range(3):
    for j in range(3):
        C_in_SM_up[f"Yu{i+1}{j+1}"] = Yu_up[i][j]

# Yd_downbasis
for i in range(3):
    for j in range(3):
        C_in_SM_up[f"Yd{i+1}{j+1}"] = Yd_up[i][j]

# Yl
for i in range(3):
    for j in range(3):
        C_in_SM_up[f"Yl{i+1}{j+1}"] = Yl[i][j]

C_in_SM_up["gp"] = gp
C_in_SM_up["g"] = g
C_in_SM_up["gs"] = gs
C_in_SM_up["Lambda"] = Lambda
C_in_SM_up["muh"] = muh