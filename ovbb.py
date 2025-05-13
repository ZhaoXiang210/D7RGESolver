import numpy as np
from RGE_dim7 import solve_rge
from sympy import I, re, Abs, conjugate
import sys
sys.path.append("./external_tools/nudobe/src")
sys.path.append("./external_tools/nudobe")
import nudobe
from EFT import SMEFT, LEFT
from constants import *
from tabulate import tabulate
import scipy.constants
pc = scipy.constants.physical_constants
m_e     = pc["electron mass energy equivalent in MeV"][0] * 1e-3 #electron mass in GeV

############################################################################################
#                                                                                          #
#    Define the functions to generate the limit on the single SMEFT operator from 0vbb     #
#    We define two kinds of function to generate the limit                                 #
#    1. Solve SMEFT RGEs via D7RGESover                                                    #
#    2. Solve SMEFT RGEs via nuDoBe                                                        #    
#                                                                                          #
############################################################################################

############################################################################################
#                  1. Solve SMEFT RGEs via D7RGESover                                      #
############################################################################################


# Calculate the half-life using LEFT operators at EW scale via a fixed reference point   
def compute_half_life(C_out_0vbb_ref, rescale_factor, isotope="136Xe", method = "IBM2", unknown_LECs = False, PSF_scheme ="A"):
   
    C_out_0vbb_scaled = {key: val * rescale_factor for key, val in C_out_0vbb_ref.items()}
    model = LEFT(C_out_0vbb_scaled, method = method, unknown_LECs = unknown_LECs, PSF_scheme =PSF_scheme)
    half_life=model.t_half(isotope)
    return half_life

# Find the uppper limit of the WC at NP scale for a given targeted half-life             
def find_0vbb_limit_WC(C_in_Ops, C_in_val_ref, isotope="136Xe", method = "IBM2", unknown_LECs = False, PSF_scheme ="A",
                       scale_in=10e3, scale_out=80, target_half_life=2.3e26,
                       exp_scan_range=(-12, 10), exp_scan_points=200, lin_scan_points=500,
                       tol=1e-2, basis="down"):
 
    C_in_ref = {C_in_Ops: C_in_val_ref}
    
    C_out_ref = solve_rge(scale_in, scale_out, C_in_ref, basis=basis)  # RGEs are solved by D7RGESolver by function solve_rge

    if basis == "down":
        C_out_0vbb_ref = extract_0vbb_LEFT(C_out_ref,basis="down")  # Use the down basis extraction function
    elif basis == "up":
        C_out_0vbb_ref = extract_0vbb_LEFT(C_out_ref,basis="up")  # Use the up basis extraction function
    else:
        raise ValueError("Invalid basis: choose either 'down' or 'up'.")

    x_values = np.logspace(exp_scan_range[0], exp_scan_range[1], num=exp_scan_points)  # Scan over the WC at NP scale to find the uppper limit of the WC
    half_lives = np.array([compute_half_life(C_out_0vbb_ref, x, isotope=isotope, method = method, unknown_LECs = unknown_LECs, PSF_scheme =PSF_scheme) for x in x_values])

    idx_min = np.argmin(np.abs(half_lives - target_half_life))
    if idx_min == 0:
        x_min, x_max = x_values[0] / 10, x_values[1]
    elif idx_min == len(x_values) - 1:
        x_min, x_max = x_values[-2], x_values[-1] * 10
    else:
        x_min, x_max = x_values[idx_min - 1], x_values[idx_min + 1]

    x_linear_values = np.linspace(x_min, x_max, num=lin_scan_points)
    half_lives_linear = np.array([compute_half_life(C_out_0vbb_ref, x, isotope=isotope, method = method, unknown_LECs = unknown_LECs, PSF_scheme =PSF_scheme) for x in x_linear_values])

    idx_min_linear = np.argmin(np.abs(half_lives_linear - target_half_life))
    optimal_x = x_linear_values[idx_min_linear]
    optimal_half_life = half_lives_linear[idx_min_linear]

    optimal_C_in = optimal_x * C_in_val_ref
    print(f"Limit on WC {C_in_Ops} at {scale_in:.3e} GeV is {optimal_C_in:.3e} GeV^4-d, Corresponding half-life = {optimal_half_life:.3e} yr")

    if np.abs(optimal_half_life - target_half_life) < tol * target_half_life:
        return optimal_C_in, optimal_half_life
    else:
        print(f"Couldn't reach the desired precision. Best result: half-life = {optimal_half_life:.3e}")
        return optimal_C_in, optimal_half_life
    

 
# scan over a group of NP scale to find a group of upper limit of WCs                   
def scan_scale_in(find_0vbb_limit_WC, scale_in_range, num_points, scan_type="log", 
                  *args, basis="down", method="IBM2", unknown_LECs=False, PSF_scheme="A", **kwargs):
    """
    This function scans the input scale (NP scale)
    """
    if scan_type == "log":
        scale_in_values = np.logspace(np.log10(scale_in_range[0]), np.log10(scale_in_range[1]), num_points)
    elif scan_type == "linear":
        scale_in_values = np.linspace(scale_in_range[0], scale_in_range[1], num_points)
    else:
        raise ValueError("scan_type must be 'linear' or 'log'")
    
    results = []
    for scale_in in scale_in_values:
        A = find_0vbb_limit_WC(*args, scale_in=scale_in, basis=basis, 
                               method=method, unknown_LECs=unknown_LECs, PSF_scheme=PSF_scheme, **kwargs)  
        results.append((scale_in, A[0], A[1]))
    
    return results

# scan over a group of NP scale and a group of operators                                
def find_limit_0vbb_operators_and_scale(find_0vbb_limit_WC, scan_ranges, operators, 
                                        C_in_val_ref=1e-10, basis="down", 
                                        method="IBM2", unknown_LECs=False, PSF_scheme="A", **kwargs):
    
    """
    This function scans the input scale (NP scale) and the operators at NP scale
    """
    results = {}
    for operator in operators:
        operator_results = []
        
        for scale_in_range, scan_type, num_points in scan_ranges:
            operator_results.extend(scan_scale_in(find_0vbb_limit_WC, scale_in_range, num_points, scan_type, 
                                                   operator, C_in_val_ref, basis=basis, 
                                                   method=method, unknown_LECs=unknown_LECs, PSF_scheme=PSF_scheme, **kwargs))  
        
        results[operator] = operator_results
    
    table_data = []
    for operator, data in results.items():
        first_row = [operator] + [f"{data[0][0]:.3e}", f"{data[0][1]:.3e}"]
        table_data.append(first_row)

        for entry in data[1:]:
            table_data.append([""] + [f"{entry[0]:.3e}", f"{entry[1]:.3e}"])

    headers = ["", "Scale_in (GeV)", r"Limit on WCs ($\text{GeV}^{4-d}$)"]
    print("\n=== Summary Table ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))  

    return results
 
# Scan over a group of operators with fixed NP scale                                    
def find_0vbb_limit( 
    operators,
    scale, 
    basis="down",
    target_half_life=2.3e26,
    isotope="136Xe",
    method="IBM2",
    unknown_LECs=False,
    PSF_scheme="A"
):
    return find_limit_0vbb_operators_and_scale(
        find_0vbb_limit_WC,
        [((scale, scale), "log", 1)],  
        operators=operators,
        scale_out=80,
        target_half_life=target_half_life,
        isotope=isotope,
        basis=basis,
        method=method,
        unknown_LECs=unknown_LECs,
        PSF_scheme=PSF_scheme
    )

# Function to calculate the half-life of 0vbb decay with a given NP scale and the corresponding WC                                      
def halflife_scale(operator,NPscale, WC,method = "IBM2", unknown_LECs = False, PSF_scheme ="A",isostope="136Xe"):
  C_in = {operator: WC}      
  scale_in = NPscale                  
  scale_out = 80                  
  C_out = solve_rge(scale_in, scale_out, C_in, basis="down",method="integrate")   # choose the down-quark flavor basis
  LEFT_WCs=extract_0vbb_LEFT(C_out,basis="down") 
  model = LEFT(LEFT_WCs, method = method, unknown_LECs = unknown_LECs, PSF_scheme =PSF_scheme)
  halflife=model.t_half(isostope)   
  return halflife

# scan the NP scale with fixed WCs to obtain the correspondence between NP scale and half-life of 0vbb   
def scan_halflife(wc_name, wc_value, 
                 min_scale=1e3, max_scale=1e10, num_points=5, method = "IBM2", unknown_LECs = False, PSF_scheme ="A",isostope="136Xe"):

    np_scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_points)   
    results = np.zeros((num_points, 2))  
    for i, scale in enumerate(np_scales):
        half_life = halflife_scale(wc_name, NPscale=scale, WC=wc_value, method = method, unknown_LECs = unknown_LECs, PSF_scheme =PSF_scheme,isostope=isostope)
        results[i, 0] = scale
        results[i, 1] = half_life   
    return results

############################################################################################
#          2. Solve SMEFT RGEs and calculate the half-life using nuDoBe                    #
############################################################################################

 
# Calculate the half-life using SMEFT operators at EW scale via a fixed reference point 
def compute_half_life_nudobe(C_out_0vbb_ref, x, isotope="136Xe", method="IBM2", unknown_LECs=False, PSF_scheme="A"):
   
    C_out_0vbb_scaled = {key: val * x for key, val in C_out_0vbb_ref.items()}
    model_instance = SMEFT(C_out_0vbb_scaled, method=method, unknown_LECs=unknown_LECs, PSF_scheme=PSF_scheme)  
    half_life = model_instance.t_half(isotope)
    return half_life

  
# Find the uppper limit of the WC at NP scale for a given targeted half-life
def find_0vbb_limit_nudobe(C_in_Ops, C_in_val_ref, isotope, 
                   scale_in=10e3, scale_out=80, target_half_life=2.3e26, 
                   exp_scan_range=(-10, 10), exp_scan_points=200, lin_scan_points=500, tol=1e-3,basis="down", method="IBM2", unknown_LECs=False, PSF_scheme="A"):
    
    C_in_ref = {C_in_Ops: C_in_val_ref}
    model = SMEFT(C_in_ref)
    """
    RGEs are solved by nudobe by function model.run.
    """
    C_out_0vbb_ref=model.run(initial_scale = scale_in, inplace = True)
    x_values = np.logspace(exp_scan_range[0], exp_scan_range[1], num=exp_scan_points)
    half_lives = np.array([compute_half_life_nudobe(C_out_0vbb_ref, x, isotope, method=method, unknown_LECs=unknown_LECs, PSF_scheme=PSF_scheme) for x in x_values])

   
    idx_min = np.argmin(np.abs(half_lives - target_half_life))
    if idx_min == 0:
        x_min, x_max = x_values[0] / 10, x_values[1]
    elif idx_min == len(x_values) - 1:
        x_min, x_max = x_values[-2], x_values[-1] * 10
    else:
        x_min, x_max = x_values[idx_min - 1], x_values[idx_min + 1]

  
    x_linear_values = np.linspace(x_min, x_max, num=lin_scan_points)
    half_lives_linear = np.array([compute_half_life_nudobe(C_out_0vbb_ref, x, isotope, method=method, unknown_LECs=unknown_LECs, PSF_scheme=PSF_scheme) for x in x_linear_values])

    
    idx_min_linear = np.argmin(np.abs(half_lives_linear - target_half_life))
    optimal_x = x_linear_values[idx_min_linear]
    optimal_half_life = half_lives_linear[idx_min_linear]

   
    optimal_C_in = optimal_x * C_in_val_ref
    print(f"Limit on WC {C_in_Ops} at scale {scale_in:.3e} = {optimal_C_in:.3e}, Corresponding half-life = {optimal_half_life:.3e}")

    
    if np.abs(optimal_half_life - target_half_life) < tol * target_half_life:
        return optimal_C_in, optimal_half_life
    else:
        print(f"Couldn't reach the desired precision. Best result: half-life = {optimal_half_life:.3e}")
        return optimal_C_in, optimal_half_life
    
 
#  Scan over a group of operators with fixed NP scale                                    
def find_limit_nudobe( 
    operators,
    scale, 
    target_half_life=2.3e26,
    isotope="136Xe",
    method="IBM2",
    unknown_LECs=False,
    PSF_scheme="A"
):
    
    return find_limit_0vbb_operators_and_scale(
        find_0vbb_limit_nudobe,
        [((scale, scale), "log", 1)],  
        operators=operators,
        scale_out=80,
        target_half_life=target_half_life,
        isotope=isotope,
        method=method,
        unknown_LECs=unknown_LECs,
        PSF_scheme=PSF_scheme
    )


## low energy input, consistent with the nudobe
import ckmutil
import ckmutil.ckm
Vus = 0.2243
Vub = 3.62e-3
Vcb = 4.221e-2
gamma = 1.27
CKM = ckmutil.ckm.ckm_tree(Vus, Vub, Vcb, gamma)
V_ud=CKM[0][0]
GeV=1
MeV=1e-3*GeV
m_u = 2.16*MeV        
m_d = 4.67*MeV 
vev=246*GeV       

###########################################################################################                                                                                                                              
#                                                                                         #                                        
#                 Match the 0vbb SMEFT operators in D7RGESolver to LEFT operators         #                                                                                                                                                                                                                                                                                         
#                                                                                         #                                                                                                                               
###########################################################################################
def extract_0vbb_LEFT_up(C_out):
   
    LEFT_WCs= {}   
    LH5_11=C_out.get("LH5_11", 0)
    LH_11=C_out.get("LH_11", 0)
    LeDH_11=C_out.get("LeDH_11", 0)
    LHW_11=C_out.get("LHW_11", 0)
    dLueH_1111=C_out.get("dLueH_1111", 0)
    dLQLH1_1111=C_out.get("dLQLH1_1111", 0)
    dLQLH2_1111=C_out.get("dLQLH2_1111", 0)
    DLDH1_11=C_out.get("DLDH1_11", 0)
    DLDH2_11=C_out.get("DLDH2_11", 0)
    QuLLH_1111=C_out.get("QuLLH_1111", 0)
    QuLLH_2111=C_out.get("QuLLH_2111", 0)
    QuLLH_3111=C_out.get("QuLLH_3111", 0)
    duLDL_1111=C_out.get("duLDL_1111", 0)
      
    LEFT_WCs["m_bb"] = -vev**2 * LH5_11 - vev**4/2 * LH_11

    LEFT_WCs["VL(6)"] = (vev**3 * V_ud*(- 1/np.sqrt(2) * LeDH_11.conjugate()
                                            + 4*m_e/vev * LHW_11.conjugate()/0.652))
    LEFT_WCs["VR(6)"] = (vev**3/(2*np.sqrt(2)) * dLueH_1111.conjugate())

    LEFT_WCs["SR(6)"] = (vev**3 *( 1/(2*np.sqrt(2)) * dLQLH1_1111.conjugate()
                                      -V_ud/2*m_d/vev   * DLDH2_11.conjugate()))

        
    LEFT_WCs["SL(6)"] = (vev**3 * (  1/(np.sqrt(2))   * (CKM[0][0]*QuLLH_1111.conjugate()+CKM[1][0]*QuLLH_2111.conjugate()+CKM[2][0]*QuLLH_3111.conjugate())
                                       + V_ud/2 * m_u/vev * DLDH2_11.conjugate()))

    LEFT_WCs["T(6)"]  = (vev**3/(8*np.sqrt(2)) * (2 * dLQLH2_1111.conjugate()
                                                      +   dLQLH1_1111.conjugate()))

    LEFT_WCs["VL(7)"] = -(vev**3*V_ud/2 * (+ 2 * DLDH1_11.conjugate()
                                              +  DLDH2_11.conjugate()
                                              + 8 * LHW_11.conjugate()/0.652))

    LEFT_WCs["VR(7)"] = ( vev**3 * -1 * duLDL_1111.conjugate())

    LEFT_WCs["1L(9)"] = -(vev**3 * V_ud**2*(+ 2*DLDH1_11.conjugate()
                                               + 8*LHW_11.conjugate()/0.652))

    LEFT_WCs["4L(9)"] = (-vev**3 * 2*V_ud * duLDL_1111.conjugate())
    return LEFT_WCs


def extract_0vbb_LEFT_down(C_out):
   
    LEFT_WCs= {}    
    LH5_11=C_out.get("LH5_11", 0)
    LH_11=C_out.get("LH_11", 0)
    LeDH_11=C_out.get("LeDH_11", 0)
    LHW_11=C_out.get("LHW_11", 0)
    dLueH_1111=C_out.get("dLueH_1111", 0)
    dLQLH1_1111=C_out.get("dLQLH1_1111", 0)
    dLQLH1_1121=C_out.get("dLQLH1_1121", 0)
    dLQLH1_1131=C_out.get("dLQLH1_1131", 0)
    dLQLH2_1111=C_out.get("dLQLH2_1111", 0)
    dLQLH2_1121=C_out.get("dLQLH2_1121", 0)
    dLQLH2_1131=C_out.get("dLQLH2_1131", 0)
    DLDH1_11=C_out.get("DLDH1_11", 0)
    DLDH2_11=C_out.get("DLDH2_11", 0)
    QuLLH_1111=C_out.get("QuLLH_1111", 0)
    duLDL_1111=C_out.get("duLDL_1111", 0)
        
    LEFT_WCs["m_bb"] = -vev**2 * LH5_11 - vev**4/2 * LH_11

    LEFT_WCs["VL(6)"] = (vev**3 * V_ud*(- 1/np.sqrt(2) * LeDH_11.conjugate()
                                            + 4*m_e/vev * LHW_11.conjugate()/0.652))
    LEFT_WCs["VR(6)"] = (vev**3/(2*np.sqrt(2)) * dLueH_1111.conjugate())

    LEFT_WCs["SR(6)"] = (vev**3 *( 1/(2*np.sqrt(2)) * (CKM[0][0]*dLQLH1_1111.conjugate()+CKM[0][1]*dLQLH1_1121.conjugate()+CKM[0][2]*dLQLH1_1131.conjugate())
                                      -V_ud/2*m_d/vev   * DLDH2_11.conjugate()))

        
    LEFT_WCs["SL(6)"] = (vev**3 * (  1/(np.sqrt(2))   * QuLLH_1111.conjugate()
                                       + V_ud/2 * m_u/vev * DLDH2_11.conjugate()))

    LEFT_WCs["T(6)"]  = (vev**3/(8*np.sqrt(2)) * (CKM[0][0]*(2 * dLQLH2_1111.conjugate()
                                                      +   dLQLH1_1111.conjugate())     
                                                 +CKM[0][1]*(2 * dLQLH2_1121.conjugate()
                                                      +   dLQLH1_1121.conjugate())
                                                 +CKM[0][2]*(2 * dLQLH2_1131.conjugate()
                                                      +   dLQLH1_1131.conjugate())))

    LEFT_WCs["VL(7)"] = -(vev**3*V_ud/2 * (+ 2 * DLDH1_11.conjugate()
                                              +  DLDH2_11.conjugate()
                                              + 8 * LHW_11.conjugate()/0.652))

    LEFT_WCs["VR(7)"] = ( vev**3 * -1 * duLDL_1111.conjugate())

    LEFT_WCs["1L(9)"] = -(vev**3 * V_ud**2*(+ 2*DLDH1_11.conjugate()
                                               + 8*LHW_11.conjugate()/0.652))

    LEFT_WCs["4L(9)"] = (-vev**3 * 2*V_ud * duLDL_1111.conjugate())
    return LEFT_WCs

def extract_0vbb_LEFT(C_out, basis="down"):
    if basis == "up":
        return extract_0vbb_LEFT_up(C_out)
    elif basis == "down":
        return extract_0vbb_LEFT_down(C_out)
    else:
        raise ValueError("Invalid basis specified. Choose 'up' or 'down'.")
    
###################################################################################################
#                                                                                                 #
#           Convert the 0vbb SMEFT operators in D7RGESolver to other SMEFT operators basis        #                  
#                                                                                                 #
###################################################################################################
def extract_0vbb_operators_downbasis(C_out):
   
    C_out_0vbb = {}  

    LH5_11=C_out.get("LH5_11", 0)
    LH_11=C_out.get("LH_11", 0)
    LeDH_11=C_out.get("LeDH_11", 0)
    LHW_11=C_out.get("LHW_11", 0)
    dLueH_1111=C_out.get("dLueH_1111", 0)
    dLQLH1_1111=C_out.get("dLQLH1_1111", 0)
    dLQLH1_1121=C_out.get("dLQLH1_1121", 0)
    dLQLH1_1131=C_out.get("dLQLH1_1131", 0)
    dLQLH2_1111=C_out.get("dLQLH2_1111", 0)
    dLQLH2_1121=C_out.get("dLQLH2_1121", 0)
    dLQLH2_1131=C_out.get("dLQLH2_1131", 0)
    DLDH1_11=C_out.get("DLDH1_11", 0)
    DLDH2_11=C_out.get("DLDH2_11", 0)
    QuLLH_1111=C_out.get("QuLLH_1111", 0)
    duLDL_1111=C_out.get("duLDL_1111", 0)

 
    C_out_0vbb["LH(5)"] = LH5_11
    C_out_0vbb["LH(7)"] = LH_11
    C_out_0vbb["LHDe(7)"] = 1j * LeDH_11
    C_out_0vbb["LHD1(7)"] =  DLDH1_11 
    C_out_0vbb["LHD2(7)"] =  DLDH2_11  
    C_out_0vbb["LHW(7)"] =  LHW_11/0.652
    C_out_0vbb["LLduD1(7)"] = 1j*duLDL_1111  
    C_out_0vbb["LLQdH1(7)"] =  dLQLH1_1111*CKM.conjugate()[0,0]+dLQLH1_1121*CKM.conjugate()[0,1]+dLQLH1_1131*CKM.conjugate()[0,2]
    C_out_0vbb["LLQdH2(7)"] =  dLQLH2_1111*CKM.conjugate()[0,0]+dLQLH2_1121*CKM.conjugate()[0,1]+dLQLH2_1131*CKM.conjugate()[0,2]
    C_out_0vbb["LLQuH(7)"] =  QuLLH_1111
    C_out_0vbb["LeudH(7)"] =  1/2*dLueH_1111
    return C_out_0vbb



def extract_0vbb_operators_upbasis(C_out):
   
    C_out_0vbb = {}  

    LH5_11=C_out.get("LH5_11", 0)
    LH_11=C_out.get("LH_11", 0)
    LeDH_11=C_out.get("LeDH_11", 0)
    LHW_11=C_out.get("LHW_11", 0)
    dLueH_1111=C_out.get("dLueH_1111", 0)
    dLQLH1_1111=C_out.get("dLQLH1_1111", 0)
    dLQLH2_1111=C_out.get("dLQLH2_1111", 0)
    DLDH1_11=C_out.get("DLDH1_11", 0)
    DLDH2_11=C_out.get("DLDH2_11", 0)
    QuLLH_1111=C_out.get("QuLLH_1111", 0)
    QuLLH_2111=C_out.get("QuLLH_2111", 0)
    QuLLH_3111=C_out.get("QuLLH_3111", 0)
    duLDL_1111=C_out.get("duLDL_1111", 0)

    C_out_0vbb["LH(5)"] = LH5_11
    C_out_0vbb["LH(7)"] = LH_11
    C_out_0vbb["LHDe(7)"] = 1j* LeDH_11
    C_out_0vbb["LHD1(7)"] =  DLDH1_11
    C_out_0vbb["LHD2(7)"] =  DLDH2_11
    C_out_0vbb["LHW(7)"] =  LHW_11/0.652
    C_out_0vbb["LLduD1(7)"] =  1j* duLDL_1111
    C_out_0vbb["LLQdH1(7)"] =  dLQLH1_1111
    C_out_0vbb["LLQdH2(7)"] =  dLQLH2_1111
    C_out_0vbb["LLQuH(7)"] =  QuLLH_1111*CKM.conjugate()[0,0]+QuLLH_2111*CKM.conjugate()[1,0]+QuLLH_3111*CKM.conjugate()[2,0]
    C_out_0vbb["LeudH(7)"] =  1/2*dLueH_1111
    return C_out_0vbb


def replace_I_with_1j(expr):
    """
    Recursively replaces all instances of 'I' in a SymPy expression with '1j'.
    
    """
    if expr.is_Add:
        return sum(replace_I_with_1j(arg) for arg in expr.args)
    elif expr.is_Mul:
        return np.prod([replace_I_with_1j(arg) for arg in expr.args])
    elif expr.is_Pow:
        base, exp = expr.as_base_exp()
        return replace_I_with_1j(base) ** replace_I_with_1j(exp)
    elif expr.is_Function:
        return expr.func(*[replace_I_with_1j(arg) for arg in expr.args])
    elif expr == I:
        return 1j
    else:
        return complex(expr.evalf()) if expr.is_number else expr


#################################################################################################################################
#                                                                                                                               #
#           analytical halftime expression of 0vbb calculated by master formula method:1708.09390 and 1806.02780                #
#   The following results is obtained by using nuDobe 2304.05415 via command:"print(model.generate_formula("136Xe"))"           #                  
#                                                                                                                               #
#################################################################################################################################
def half_life_136Xe(C):
    # Get the values of the Wilson coefficients from the input dictionary
    C_LH5 = C.get('LH(5)', 0)
    C_LH = C.get('LH(7)', 0) + C_LH5 * 2 / (246**2)
    C_LHD1 = C.get('LHD1(7)', 0)
    C_LHD2 = C.get('LHD2(7)', 0)
    C_LHDe = C.get('LHDe(7)', 0)
    C_LHW = C.get('LHW(7)', 0)
    C_LLduD1 = C.get('LLduD1(7)', 0)
    C_LLQdH1 = C.get('LLQdH1(7)', 0)
    C_LLQdH2 = C.get('LLQdH2(7)', 0)
    C_LLQuH = C.get('LLQuH(7)', 0)
    C_LeudH = C.get('LeudH(7)', 0)

    # Half-life inverse expression
    
    term_1 = 1.59e13 * Abs(C_LH)**2
    term_2 = 3.84e-4 * Abs(C_LHD1)**2
    term_3 = 1.17e-4 * Abs(C_LHD2)**2
    term_4 = 1.89e5 * Abs(C_LHDe)**2
    term_5 = 6.54e-3 * Abs(C_LHW)**2
    term_6 = 1.23e3 * Abs(C_LLduD1)**2
    term_7 = 7.18e5 * Abs(C_LLQdH1)**2
    term_8 = 4.18e4 * Abs(C_LLQdH2)**2
    term_9 = 3.61e6 * Abs(C_LLQuH)**2
    term_10 = 3.81e1 * Abs(C_LeudH)**2
    
    term_11 = 2*((1.59e13)**0.5)*((3.84e-4)**0.5) * re(C_LH * conjugate(C_LHD1))
    term_12 = -2*((1.59e13)**0.5)*((1.17e-4)**0.5) * re(C_LH * conjugate(C_LHD2))
    term_13 = 2*((1.59e13)**0.5)*((6.54e-3)**0.5) * re(C_LH * conjugate(C_LHW))
    term_14 = 1.95e-3 * re(C_LH * conjugate(C_LLduD1))
    term_15 = 2*((1.59e13)**0.5)*((7.18e5)**0.5) * re(C_LH * conjugate(C_LLQdH1))
    term_16 = -2*((1.59e13)**0.5)*((4.18e4)**0.5) * re(C_LH * conjugate(C_LLQdH2))
    term_17 = -2*((1.59e13)**0.5)*((3.61e6)**0.5) * re(C_LH * conjugate(C_LLQuH))
    term_18 = -1.07e7 * re(C_LH * conjugate(C_LeudH))

    term_19 = -4.23e-4 * re(C_LHD1 * conjugate(C_LeudH))
    term_20 = 2.91e-11 * re(C_LHD1 * conjugate(C_LeudH))
    term_21 = 3.16e-3 * re(C_LHD1 * conjugate(C_LeudH))
    term_22 = 3.32e1 * re(C_LHD1 * conjugate(C_LeudH))
    term_23 = -8.01e0 * re(C_LHD1 * conjugate(C_LeudH))
    term_24 = -7.44e1 * re(C_LHD1 * conjugate(C_LeudH))
    term_25 = -5.23e-2 * re(C_LHD1 * conjugate(C_LeudH))

    term_26 = 2.91e-11 * re(C_LHD2 * conjugate(C_LeudH))
    term_27 = -1.74e-3 * re(C_LHD2 * conjugate(C_LeudH))
    term_28 = -1.83e1 * re(C_LHD2 * conjugate(C_LeudH))
    term_29 = 4.41e0 * re(C_LHD2 * conjugate(C_LeudH))
    term_30 = 4.1e1 * re(C_LHD2 * conjugate(C_LeudH))
    term_31 = 2.88e-2 * re(C_LHD2 * conjugate(C_LeudH))

    term_32 = 2.91e-11 * re(C_LHDe * conjugate(C_LeudH))
    term_33 = 1.41e4 * re(C_LHDe * conjugate(C_LeudH))
    term_34 = -1.16e-10 * re(C_LHDe * conjugate(C_LeudH))
    term_35 = 2.91e-11 * re(C_LHDe * conjugate(C_LeudH))
    term_36 = 4.66e-10 * re(C_LHDe * conjugate(C_LeudH))
    term_37 = 2.91e-11 * re(C_LHDe * conjugate(C_LeudH))

    term_38 = 1.37e2 * re(C_LHW * conjugate(C_LeudH))
    term_39 = -3.3e1 * re(C_LHW * conjugate(C_LeudH))
    term_40 = -3.07e2 * re(C_LHW * conjugate(C_LeudH))
    term_41 = -2.1e-1 * re(C_LHW * conjugate(C_LeudH))

    term_42 = -2.27e-13 * re(C_LLduD1 * conjugate(C_LeudH))
    term_43 = -3.46e5 * re(C_LLQdH1 * conjugate(C_LeudH))
    term_44 = -3.22e6 * re(C_LLQdH1 * conjugate(C_LeudH))
    term_45 = -2.26e3 * re(C_LLQdH1 * conjugate(C_LeudH))
    term_46 = 7.76e5 * re(C_LLQdH2 * conjugate(C_LeudH))
    term_47 = 5.46e2 * re(C_LLQdH2 * conjugate(C_LeudH))
    term_48 = 5.07e3 * re(C_LLQuH * conjugate(C_LeudH))
    term_49 = -((7.18e5 * 4.18e4)**0.5) * 2 * re(C_LLQdH1 * conjugate(C_LLQdH2))

    # Sum all terms
    T_inv = (
        term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9 + term_10 +
        term_11 + term_12 + term_13 + term_14 + term_15 + term_16 + term_17 + term_18 + 
        term_19 + term_20 + term_21 + term_22 + term_23 + term_24 + term_25 +
        term_26 + term_27 + term_28 + term_29 + term_30 + term_31 +
        term_32 + term_33 + term_34 + term_35 + term_36 + term_37 +
        term_38 + term_39 + term_40 + term_41 +
        term_42 + term_43 + term_44 + term_45 + term_46 + term_47 + term_48 + term_49
    )

    return 1 / T_inv
































































