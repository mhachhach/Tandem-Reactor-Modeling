"""
tandem_model.py
===============
Fully self-contained Python implementation of the Salmi multi-scale
model for the selective oxidation of L-arabinose to L-arabinonic acid over
Au/γ-Al₂O₃ in a tandem packed-bed / completely-recycled CSTR.

Usage
-----
  pip install numpy scipy plotly kaleido==0.2.1
  python tandem_model.py

Dependencies
------------
  numpy >= 1.24   [Harris et al., Nature 585 2020]
  scipy >= 1.10   [Virtanen et al., Nat. Methods 17 2020]
  plotly >= 5.14  [for figures]
  kaleido==0.2.1  [for PNG export]

Output: ./results/  (11 publication-quality PNG figures + printed summary)
"""
# Requirements: see requirements.txt

import os, time, warnings
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# §1  EXPERIMENTAL DATA
#     182 data points from 6 conditions spanning two catalyst loadings
#     Sources: Hachhach et al. 2025a,b; Hachhach PhD thesis 2024; later points of E1, E3, E4 and E6 and run E5 unpublished
#     Source: Simakova et al. CEJ 2023; Hachhach PhD thesis ÅAU 2024
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "E1": {"flow": 40,  "mass": 0.897,   "label": "40 mL/min, 0.897 g",
        "t_min": [0.0,15.0,60.0,120.0,155.0,184.0,243.0,300.0,365.0,1174.0,
                  1200.0,1260.0,1321.0,1380.0,1440.0,1500.0,1573.0,1620.0,
                  1713.0,2625.0,2702.0,2765.0,2850.0,3006.0,3061.0,3120.0,
                  3195.0,3310.0,4024.0,4095.0,4153.0,4200.0,4452.0],
        "XA":    [0.0,0.153985,0.235836,0.308814,0.352911,0.393534,0.482053,
                  0.554738,0.628833,0.986979,0.990555,0.990672,0.990819,0.99143,
                  0.991181,0.991249,0.991056,0.990931,0.991136,0.991534,0.991383,
                  0.991672,0.991063,0.991383,0.991389,0.991044,0.990738,0.991191,
                  0.991037,0.990937,0.991043,0.990925,0.991]},
    "E2": {"flow": 70,  "mass": 0.897,   "label": "70 mL/min, 0.897 g",
        "t_min": [0.0,15.0,60.0,120.0,150.0,180.0,241.0,301.0,360.0,420.0,
                  480.0,1276.0,1320.0,1380.0,1440.0,1500.0,1520.0],
        "XA":    [0.0,0.198832,0.272232,0.37237,0.411494,0.411201,0.519496,
                  0.596631,0.689256,0.764788,0.847743,0.988861,0.991485,
                  0.99174,0.991583,0.991685,0.99164]},
    "E3": {"flow": 150, "mass": 1.6225,  "label": "150 mL/min, 1.6225 g",
        "t_min": [0.0,15.0,60.0,120.0,150.0,180.0,260.0,342.0,400.0,422.0,
                  480.0,540.0,1333.0,1381.0,1440.0,1500.0,1540.0,1570.0,1620.0,
                  1680.0,1740.0,1800.0,1861.0,1920.0,1981.0,2040.0,2792.0,2860.0,
                  2887.0,2947.0,3000.0,3090.0,3150.0,3257.0,4315.0,4383.0],
        "XA":    [0.0,0.004613,0.080886,0.105025,0.118889,0.133526,0.163765,
                  0.181195,0.205022,0.229798,0.247874,0.275967,0.585699,0.605448,
                  0.623706,0.643608,0.656631,0.666113,0.683268,0.701092,0.716181,
                  0.735599,0.753084,0.756957,0.782979,0.797261,0.943693,0.953578,
                  0.958345,0.965277,0.970446,0.977763,0.982501,0.988761,0.999759,1.0]},
    "E4": {"flow": 200, "mass": 1.6225,  "label": "200 mL/min, 1.6225 g",
        "t_min": [0.0,15.0,60.0,120.0,152.0,180.0,260.0,342.0,422.0,488.0,
                  542.0,644.0,1395.0,1440.0,1500.0,1540.0,1570.0,1622.0,1680.0,
                  1740.0,1801.0,1860.0,1920.0,1980.0,2038.0,2876.0,2941.0,3000.0,
                  3061.0,3080.0],
        "XA":    [0.0,0.009336,0.072239,0.096911,0.115403,0.128857,0.168455,
                  0.20977,0.247033,0.279002,0.308321,0.333396,0.637626,0.652819,
                  0.676784,0.690457,0.704136,0.718701,0.739837,0.759268,0.779657,
                  0.796642,0.812578,0.831643,0.84727,0.990494,0.994056,0.99614,
                  0.997541,0.997926]},
    "E5": {"flow": 225, "mass": 1.6225,  "label": "225 mL/min, 1.6225 g",
        "t_min": [0.0,15.0,63.0,122.0,150.0,181.0,260.0,342.0,402.0,424.0,
                  486.0,540.0,1360.0,1420.0,1480.0,1540.0,1576.0,1621.0,1686.0,
                  1740.0,1801.0,1875.0,1908.0,2153.0,2877.0,2942.0,3002.0,3063.0,
                  3120.0,3181.0,3240.0,3304.0,4241.0,4303.0,4364.0,4420.0,4483.0,
                  4603.0,4673.0,5688.0],
        "XA":    [0.0,0.020402,0.098157,0.120233,0.132263,0.141443,0.171517,
                  0.190865,0.212026,0.220166,0.27243,0.261803,0.508339,0.534137,
                  0.54764,0.561199,0.575114,0.585298,0.599562,0.612282,0.621413,
                  0.639025,0.648429,0.699257,0.822708,0.834155,0.843646,0.850524,
                  0.8588,0.866836,0.877002,0.883127,0.956757,0.963828,0.968798,
                  0.969557,0.969955,0.977016,0.980281,0.998156]},
    "E6": {"flow": 250, "mass": 1.6225,  "label": "250 mL/min, 1.6225 g",
        "t_min": [0.0, 15.0, 60.0, 120.0, 152.0, 180.0, 260.0, 420.0, 488.0, 542.0, 640.0, 721.0, 1456.0, 1500.0, 1540.0, 1570.0, 1620.0, 1682.0, 1760.0, 1802.0, 1862.0, 1922.0, 1980.0, 2042.0, 2102.0, 2885.0],
        "XA":    [0.000000, 0.012379, 0.100413, 0.131404, 0.148481, 0.156757, 0.197089, 0.276749, 0.309483, 0.341351, 0.384480, 0.425413, 0.704644, 0.732930, 0.746069, 0.755979, 0.769204, 0.789441, 0.810058, 0.818065, 0.831449, 0.849563, 0.864344, 0.879668, 0.891279, 0.996219]},
}

# Assign each experiment to its catalyst batch
EXP_BATCH = {k: "097" if EXPERIMENTS[k]["mass"] < 1.0 else "162"
             for k in EXPERIMENTS}


# ══════════════════════════════════════════════════════════════════════════════
# §2  MODEL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelParams:
    """
    Complete dimensionless parameter set for the Hachhach–Salmi model.

    Notation follows the model report [Hachhach & Salmi, ÅAU 2024] and
    Table 2 of the manuscript.

    Packed-bed transport (per component i; A=arabinose, B=O₂, C=arabinonate)
    -------------------------------------------------------------------------
    alpha_i  [s⁻¹]  Axial dispersion   αᵢ = D_Li / L²
    beta_i   [s⁻¹]  Pore diffusion     βᵢ = D_ei / R²          ← key param
    delta_i  [s⁻¹]  Pore–bulk coupling δᵢ = D_ei c₀ a_LS/(ε_L c₀ R)
    lam_i    [—]    Reciprocal Biot    λᵢ = D_ei / (R k_Li)

    System
    ------
    eps_P  [—]  Catalyst porosity
    eps_LT [—]  Tank liquid holdup
    gamma  [s]  Bed residence-time parameter γ = τ_L ε_L
    tau_LT [s]  Tank space time τ_LT = V_LT / Q

    Kinetics
    --------
    k_prime [s⁻¹]  Merged rate constant  k' = k ρ_P / ε_P  ← estimated
    K_A     [—]    LHHW arabinose adsorption constant
    K_B     [—]    LHHW O₂ adsorption constant  (≈0 on gold)

    Grid
    ----
    Nz   Number of axial nodes (Nzi = Nz-1 interior nodes)
    Nx   Number of radial pore nodes
    """
    # Component A (arabinose)
    alpha_A : float = 0.010
    beta_A  : float = 4.877      # fitted
    delta_A : float = 2.287      # fitted
    lam_A   : float = 0.001

    # Component B (dissolved O₂ — frozen at Y_B ≡ 1)
    alpha_B : float = 0.030      # 3× D_A (O₂ diffuses faster than arabinose)
    beta_B  : float = 14.631     # 3 × beta_A
    delta_B : float = 6.861      # 3 × delta_A
    lam_B   : float = 0.001

    # Component C (arabinonic acid — similar size to A)
    alpha_C : float = 0.010
    beta_C  : float = 4.877
    delta_C : float = 2.287
    lam_C   : float = 0.001

    # System
    eps_P   : float = 0.45
    eps_LT  : float = 0.85
    gamma   : float = 3.50       # [s] geometric bed residence time at Q_ref = 70 mL/min
    tau_LT  : float = 200.0      # [s] calibrated at Q_ref = 70 mL/min

    # Kinetics
    k_prime : float = 2.728e-3
    K_A     : float = 0.0        # 0 → second-order pseudo-first-order kinetics
    K_B     : float = 0.0        # negligible O₂ adsorption on gold [Villa 2023]
    K_C     : float = 0.0        # product (arabinonate) inhibition constant, K_C,dim·c₀

    # Grid
    o2_active : bool  = False
    o2_ratio  : float = 0.0
    Nz      : int   = 10
    Nx      : int   = 6


# Reference conditions
Q_REF    = 70.0     # mL/min
V_LIQ    = 150e-6   # total liquid in the loop [m3] (150 mL, Hachhach et al. 2025a)
EPS_LT   = 0.85     # liquid holdup of the sparged tank
V_LIQ_BED = 0.40*np.pi*0.005**2*0.130   # liquid in the bed [m3]
TAU_LT_REF = (V_LIQ - V_LIQ_BED)/(EPS_LT*Q_REF/60e6)   # tank space time at Q_REF [s]
MASS_REF = 0.897    # g
GAMMA_REF = 0.40*np.pi*0.005**2*0.130/(70.0/60e6)


def make_params(k_prime: float,
                flow_ml_min: float = 70.0,
                mass_g: float      = 0.897,
                beta_A: float      = 4.877,
                delta_A: float     = 2.287,
                Nz: int            = 10,
                Nx: int            = 6,
                K_A: float         = 0.0,
                K_B: float         = 0.0) -> ModelParams:
    """
    Create a scaled ModelParams for a given operating condition.

    Scaling rules (constant bed and tank volumes, V_bed and V_tank fixed):
        γ    ∝ 1/Q     because τ_L = V_bed ε_L / Q
        τ_LT ∝ 1/Q     because τ_LT = V_LT / Q
        k'   ∝ m_cat   proportional to total active sites

    Parameters
    ----------
    k_prime     : rate constant at (Q_ref, mass_ref) [s⁻¹]
    flow_ml_min : desired flow rate [mL/min]
    mass_g      : catalyst loading [g]
    beta_A      : pore diffusion parameter [s⁻¹]
    delta_A     : pore–bulk coupling [s⁻¹]
    Nz, Nx      : grid resolution
    K_A, K_B    : LHHW adsorption constants (0 → second-order)
    """
    fsc = Q_REF / flow_ml_min       # flow scaling factor

    # so alpha = D_L/L^2 scales with Q and the bed Peclet number Pe = 1/(alpha*gamma) is the same for all runs
    msc = mass_g / MASS_REF         # mass scaling factor
    return ModelParams(
        alpha_A = 0.010/fsc,   beta_A = beta_A,    delta_A = delta_A,   lam_A = 0.001,
        alpha_B = 0.030/fsc,   beta_B = beta_A*3,  delta_B = delta_A*3, lam_B = 0.001,
        alpha_C = 0.010/fsc,   beta_C = beta_A,    delta_C = delta_A,   lam_C = 0.001,
        eps_P   = 0.45,    eps_LT = EPS_LT,
        gamma   = GAMMA_REF * fsc,
        tau_LT  = TAU_LT_REF * fsc,
        k_prime = k_prime * msc,
        K_A=K_A, K_B=K_B, Nz=Nz, Nx=Nx,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3  INTRINSIC KINETICS
# ══════════════════════════════════════════════════════════════════════════════

def rate_second_order(Y_A: np.ndarray, Y_B: np.ndarray, p: ModelParams) -> np.ndarray:
    """
    Bimolecular rate, pseudo-first-order under Y_B ≡ 1.

        r_A = -k' · Y_A · Y_B  →  -k' · Y_A      (eq. 13)

    The merged rate constant k' = k ρ_P / ε_P already incorporates catalyst
    density and porosity, giving units [s⁻¹] for the dimensionless pore ODE.
    """
    return -p.k_prime * Y_A * Y_B


def rate_lhhw(Y_A: np.ndarray, Y_B: np.ndarray, p: ModelParams) -> np.ndarray:
    """
    Langmuir–Hinshelwood–Hougen–Watson (LHHW) rate.

        r_A = -k' · Y_A · Y_B / (1 + K_A·Y_A + K_B·Y_B)²      (eq. 14)

    With K_A = 11.44 (dimensionless, from K_A,lit × c₀ = 0.086 × 133),
    K_B = 0 (O₂ adsorption negligible on Au [Villa et al. CatSciTech 2023]),
    and k' normalised so both models share the same initial rate:
        k'_LHHW = k'_2nd × (1 + K_A)² = k'_2nd × 154.8
    """
    denom = (1.0 + p.K_A * Y_A + p.K_B * Y_B) ** 2
    return -p.k_prime * Y_A * Y_B / np.maximum(denom, 1e-30)


def rate_product_inhibition(Y_A: np.ndarray, Y_B: np.ndarray, p: ModelParams) -> np.ndarray:
    """
    First-order rate inhibited by adsorbed product (arabinonate):

        r_A = -k' · Y_A · Y_B / (1 + K_C·Y_C),   Y_C = 1 - Y_A

    Y_C = 1 - Y_A holds in every compartment because arabinose and arabinonic
    acid share the same transport parameters, the pores start filled with feed
    and the conversion is 1:1 (closure checked to 4e-11, Table S14).
    """
    return -p.k_prime * Y_A * Y_B / (1.0 + p.K_C * np.clip(1.0 - Y_A, 0.0, 1.0))


def get_rate_fn(name: str) -> Callable:
    if name == "second_order":
        return rate_second_order
    elif name == "lhhw":
        return rate_lhhw
    elif name == "product_inhibition":
        return rate_product_inhibition
    raise ValueError(f"Unknown rate law '{name}'")


# ══════════════════════════════════════════════════════════════════════════════
# §4  BUILD_RHS — METHOD OF LINES DISCRETISATION
#
#   Converts the coupled PDE system (packed-bed + intraparticle + tank)
#   into an ODE system via the Method of Lines (MOL).
#   [Schiesser & Griffiths 2009; Vande Wouwer et al. 2023]
#
#   State vector layout  (Nzi = Nz-1 interior bed nodes)
#   ─────────────────────────────────────────────────────
#     [0        : Nzi]              y_bedA  bed bulk arabinose
#     [Nzi      : 2·Nzi]            y_bedB  bed bulk O₂ (frozen ≡ 1)
#     [2·Nzi    : 3·Nzi]            y_bedC  bed bulk arabinonic acid
#     [3·Nzi    : 3·Nzi + Nzi·Nx]  Y_poreA pore arabinose  (Nzi × Nx)
#     [... + Nzi·Nx : ... + 2·Nzi·Nx]  Y_poreB pore O₂ (frozen ≡ 1)
#     [... + 2·Nzi·Nx : ... + 3·Nzi·Nx] Y_poreC pore arabinonate
#     [-3]   y_tankA
#     [-2]   y_tankB  (frozen ≡ 1)
#     [-1]   y_tankC
#
#   ODE count: 3·Nzi·(1+Nx) + 3
#   At Nz=10, Nx=6 (Nzi=9): 3·9·7 + 3 = 192; frozen B → active = 128
# ══════════════════════════════════════════════════════════════════════════════

def build_rhs(p: ModelParams, rate_law: str = "second_order") -> Tuple[Callable, int]:
    """
    Build and return (rhs_function, n_state).

    The RHS implements (step by step):
      Step 1  Unpack state vector into y_bed[3, Nzi], Y_pore[3, Nzi, Nx], y_tank[3]
      Step 2  Compute pore PDEs   (eq. 5: β·d²Y/dx² + r_norm)
      Step 3  Ghost-cell Danckwerts inlet BC   (eq. 3)
      Step 4  Compute bed PDEs    (eq. 1: α·d²y/dz² − (1/γ)·dy/dz + δ/λ·(Y_surf − y_bed))
      Step 5  Compute tank ODEs   (eq. 9: (y_bed_out − y_tank)/(ε_LT·τ_LT))
      Step 6  Zero B-component dynamics (Y_B ≡ 1 enforced globally, hypothesis H4)
      Step 7  Assemble output dydt vector
    """
    Nz, Nx  = p.Nz, p.Nx
    Nzi     = Nz - 1                  # interior bed nodes
    dz      = 1.0 / (Nz - 1)         # axial grid spacing
    dx      = 1.0 / (Nx - 1)         # radial pore grid spacing
    n_pore  = Nzi * Nx
    n_state = 3*Nzi + 3*n_pore + 3

    # Component-indexed parameter arrays  (A=0, B=1, C=2)
    alpha = np.array([p.alpha_A, p.alpha_B, p.alpha_C])
    beta  = np.array([p.beta_A,  p.beta_B,  p.beta_C])
    delta = np.array([p.delta_A, p.delta_B, p.delta_C])
    lam   = np.array([p.lam_A,   p.lam_B,   p.lam_C])

    # δᵢ/λᵢ — liquid-solid coupling strength (large when film resistance small)
    delta_over_lam = delta / np.maximum(lam, 1e-12)

    # Stoichiometry: A + B → C  (ν_A = -1, ν_B = -1, ν_C = +1)
    nu = np.array([-1.0, -1.0, 1.0])

    rate_fn = get_rate_fn(rate_law)

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        # ── Step 1: Unpack state ─────────────────────────────────────────
        y_bed  = np.empty((3, Nzi))
        Y_pore = np.empty((3, Nzi, Nx))
        idx = 0
        for i in range(3):
            y_bed[i] = state[idx:idx+Nzi];  idx += Nzi
        for i in range(3):
            Y_pore[i] = state[idx:idx+n_pore].reshape(Nzi, Nx);  idx += n_pore
        y_tank = state[-3:]

        # ── Step 2: Pore PDEs (eq. 5) ────────────────────────────────────
        #
        #   ε_P · ∂Yᵢ/∂t = βᵢ · ∂²Yᵢ/∂x²  +  rᵢ·ρ_P/(ε_P·c₀)
        #
        #   Dividing by ε_P and using k' = k·ρ_P/ε_P (so r·ρ_P/(ε_P·c₀) = r_norm):
        #
        #   ∂Yᵢ/∂t = (βᵢ/ε_P) · ∂²Yᵢ/∂x²  +  r_norm_i        ← eq. in code
        #
        #   IMPORTANT: r_norm already has k' (NOT k), so NO extra /ε_P on reaction term.
        #   The merged k' = k·ρ_P/ε_P absorbs the ε_P in the denominator.
        #
        #   Radial ghost cells (eq. 7 in paper):
        #     x=0 (symmetry):    Y_ghost_left  = Y[:, 1]
        #     x=1 (Biot BC):     λᵢ·∂Y/∂x = y_bed - Y → Y_ghost_right via central diff.
        #
        r_A_pore = rate_fn(Y_pore[0], Y_pore[1], p)   # shape (Nzi, Nx)

        dYdt = np.zeros((3, Nzi, Nx))
        for i in range(3):
            lam_i = max(lam[i], 1e-12)

            # Ghost cells for Biot and symmetry BCs
            Y_gl = Y_pore[i, :, 1]                                          # symmetry at x=0
            Y_gr = Y_pore[i, :, -2] + (2.0*dx/lam_i)*(y_bed[i] - Y_pore[i, :, -1])  # Biot at x=1

            # d²Y/dx²: central differences (eq. 16c)
            d2Y = np.empty((Nzi, Nx))
            d2Y[:, 0]    = (Y_pore[i,:,1]   - 2*Y_pore[i,:,0]   + Y_gl) / dx**2
            d2Y[:, 1:-1] = (Y_pore[i,:,2:]  - 2*Y_pore[i,:,1:-1] + Y_pore[i,:,:-2]) / dx**2
            d2Y[:, -1]   = (Y_gr             - 2*Y_pore[i,:,-1]   + Y_pore[i,:,-2]) / dx**2

            # Reaction source term
            # r_A_pore is NEGATIVE (arabinose consumed by reaction).
            # A (i=0): r_norm =  r_A_pore  (negative → A decreases)
            # B (i=1): r_norm =  r_A_pore  (frozen anyway)
            # C (i=2): r_norm = -r_A_pore  (positive → C increases, stoichiometry 1:1)
            if i == 0:
                r_norm = r_A_pore
            elif i == 1:
                r_norm = p.o2_ratio * r_A_pore
            else:   # i == 2, arabinonic acid produced
                r_norm = -r_A_pore

            # ∂Yᵢ/∂t = (βᵢ/ε_P)·d²Yᵢ/dx² + r_norm
            dYdt[i] = (beta[i] / max(p.eps_P, 1e-12)) * d2Y + r_norm

        # ── Step 3 + 4: Bed PDEs (eq. 1, Danckwerts BC eq. 2/3) ─────────
        #
        #   ∂yᵢ/∂t = αᵢ·∂²yᵢ/∂z² − (1/γ)·∂yᵢ/∂z + δᵢ/λᵢ·(Y_surf − y_bed)
        #
        #   Danckwerts inlet ghost cell (eq. 3):
        #     y_ghost = (y_tank - y_bed[0]·(0.5 - αᵢγ/Δz)) / (0.5 + αᵢγ/Δz)
        #
        #   Discretisation (eqs. 16a, 16b):
        #     Convection:  upwind  (y_bed[i] - y_left) / Δz
        #     Dispersion:  central (y_right - 2y + y_left) / Δz²
        #
        dy_bed_dt = np.zeros((3, Nzi))
        for i in range(3):
            ai_g  = alpha[i] * max(p.gamma, 1e-12)
            denom = 0.5 + ai_g / dz
            y_g   = (y_tank[i] - y_bed[i, 0] * (0.5 - ai_g/dz)) / denom  # Danckwerts ghost

            # Extended array: [ghost_inlet | interior | ghost_outlet (Neumann z=1)]
            y_ext       = np.empty(Nzi + 2)
            y_ext[0]    = y_g
            y_ext[1:-1] = y_bed[i]
            y_ext[-1]   = y_bed[i, -1]     # dydz=0 at z=1

            # d²y/dz² (central, eq. 16b) and dy/dz (upwind, eq. 16a)
            d2y  = (y_ext[2:] - 2*y_ext[1:-1] + y_ext[:-2]) / dz**2
            dy_up = (y_ext[1:-1] - y_ext[:-2]) / dz

            # Coupling: δᵢ/λᵢ · (Y_surf - y_bed)  — negative when y_bed > Y_surf
            coupling = -delta_over_lam[i] * (y_bed[i] - Y_pore[i, :, -1])

            dy_bed_dt[i] = alpha[i]*d2y - (1.0/max(p.gamma, 1e-12))*dy_up + coupling

        # ── Step 5: Tank ODE (eq. 9) ─────────────────────────────────────
        #   dyᵢ/dt = (y_bed_outlet - y_tank) / (ε_LT · τ_LT)
        dy_tank_dt = np.empty(3)
        for i in range(3):
            dy_tank_dt[i] = (y_bed[i, -1] - y_tank[i]) / (p.eps_LT * max(p.tau_LT, 1e-12))

        # ── Step 6: B-saturation enforcement (hypothesis H4) ─────────────
        #   Y_B ≡ 1 everywhere; zero all dynamics for component B (i=1).
        #   Justification: c*_O2 / c₀ = 0.103/133 = 7.7e-4 → per-pass depletion < 0.01%
        if not p.o2_active:
            dYdt[1]       = 0.0
            dy_bed_dt[1]  = 0.0
        dy_tank_dt[1] = 0.0

        # ── Step 7: Assemble output ───────────────────────────────────────
        out = np.empty(n_state)
        idx = 0
        for i in range(3):
            out[idx:idx+Nzi] = dy_bed_dt[i];  idx += Nzi
        for i in range(3):
            out[idx:idx+n_pore] = dYdt[i].ravel();  idx += n_pore
        out[-3:] = dy_tank_dt
        return out

    return rhs, n_state


# ══════════════════════════════════════════════════════════════════════════════
# §5  SIMULATE — LSODA INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate(p: ModelParams,
             t_end_s: float,
             rate_law: str  = "second_order",
             n_eval: int    = 120,
             rtol: float    = 1e-5,
             atol: float    = 1e-8) -> dict:
    """
    Integrate the ODE system using LSODA.

    LSODA (Livermore Solver for ODEs with Automatic stiffness detection)
    automatically switches between Adams-Moulton (non-stiff) and BDF
    (stiff) methods.  The system is stiff during the initial pore transient
    (fast β/ε_P dynamics) and non-stiff at late times.
    [Hindmarsh et al., TOMS 2005; Virtanen et al., Nat. Methods 2020]

    Initial conditions (t = 0)
    --------------------------
    The system is fully loaded at t=0:
        y_bedA = 1, Y_poreA = 1, y_tankA = 1  (arabinose everywhere at c₀)
        y_bedB = 1, Y_poreB = 1, y_tankB = 1  (O₂ at saturation)
        y_bedC = 0, Y_poreC = 0, y_tankC = 0  (no product yet)

    Returns
    -------
    dict:
        t      [s]   shape (n_eval,)
        XA     [—]   arabinose conversion = 1 - y_tankA, shape (n_eval,)
        y_tank [—]   tank concentrations,  shape (3, n_eval)  [A, B, C]
        y_bed  [—]   bed axial profiles,   shape (3, Nzi, n_eval)
        Y_pore [—]   pore radial profiles, shape (3, Nzi, Nx, n_eval)
    """
    Nzi   = p.Nz - 1
    Nx    = p.Nx
    n_pore = Nzi * Nx

    rhs_fn, n_state = build_rhs(p, rate_law)

    # Initial conditions
    y0 = np.zeros(n_state)
    y0[0:Nzi]   = 1.0                                         # y_bedA
    y0[Nzi:2*Nzi] = 1.0                                       # y_bedB
    y0[3*Nzi           : 3*Nzi +   n_pore] = 1.0              # Y_poreA
    y0[3*Nzi +   n_pore: 3*Nzi + 2*n_pore] = 1.0              # Y_poreB
    y0[-3] = 1.0    # y_tankA
    y0[-2] = 1.0    # y_tankB

    t_eval = np.linspace(0.0, t_end_s, n_eval)
    sol = solve_ivp(rhs_fn, (0.0, t_end_s), y0,
                    method="LSODA", t_eval=t_eval,
                    rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"LSODA failed: {sol.message}")

    y_bedA = sol.y[0:Nzi];     y_bedB = sol.y[Nzi:2*Nzi];   y_bedC = sol.y[2*Nzi:3*Nzi]
    Y_pA   = sol.y[3*Nzi          : 3*Nzi+n_pore].reshape(Nzi, Nx, -1)
    Y_pB   = sol.y[3*Nzi+n_pore   : 3*Nzi+2*n_pore].reshape(Nzi, Nx, -1)
    Y_pC   = sol.y[3*Nzi+2*n_pore : 3*Nzi+3*n_pore].reshape(Nzi, Nx, -1)

    return {
        "t":      sol.t,
        "XA":     np.clip(1.0 - sol.y[-3], 0.0, 1.0),
        "y_tank": sol.y[-3:],                                   # (3, n_eval)
        "y_bed":  np.stack([y_bedA, y_bedB, y_bedC]),           # (3, Nzi, n_eval)
        "Y_pore": np.stack([Y_pA,   Y_pB,   Y_pC]),             # (3, Nzi, Nx, n_eval)
    }


def predict_XA(p: ModelParams, t_s: np.ndarray, rate_law="second_order") -> np.ndarray:
    """Integrate and interpolate XA at the requested times t_s [s]."""
    t_end = float(np.max(t_s)) * 1.05 + 300.0
    res   = simulate(p, t_end, rate_law=rate_law, n_eval=80, rtol=1e-4, atol=1e-7)
    return np.interp(t_s, res["t"], res["XA"])


# ══════════════════════════════════════════════════════════════════════════════
# §6  PARAMETER ESTIMATION — NELDER-MEAD WITH MULTI-START
#     [Nelder & Mead, Comput. J. 1965; scipy.optimize.minimize]
# ══════════════════════════════════════════════════════════════════════════════

def estimate_k(exp_keys: list,
               beta_A: float  = 4.877,
               delta_A: float = 2.287,
               n_starts: int  = 12,
               verbose: bool  = True) -> tuple:
    """
    Estimate k' for a set of experiments by minimising the MSE:

        F(k') = (1/N) Σⱼ [XA_j^exp − XA_j^mod(k')]²          (Eq. 5 of the paper)

    Two-stage strategy:
      1) Grid search: 12 log-spaced points over k' ∈ [10⁻⁸, 10⁻²] s⁻¹
         to identify the global basin.
      2) Nelder–Mead refinement in log-space (θ = log k') for positivity
         and better numerical conditioning.

    Returns
    -------
    k_opt : optimal k' [s⁻¹]
    F_opt : optimal MSE
    """
    def obj(log_k_arr: np.ndarray) -> float:
        k = np.exp(float(log_k_arr[0]))
        sse = 0.0;  n = 0
        for key in exp_keys:
            d   = EXPERIMENTS[key]
            t_s = np.array(d["t_min"]) * 60.0
            p   = make_params(k, d["flow"], d["mass"], beta_A, delta_A, Nz=10, Nx=6)
            try:
                pred = predict_XA(p, t_s)
                sse += np.sum((np.array(d["XA"]) - pred)**2)
                n   += len(d["XA"])
            except Exception:
                return 1e9
        return sse / max(n, 1)

    # Grid search — 12 log-spaced starting points
    best_lk, best_F = -4.0, np.inf
    for lk in np.linspace(-8, -2, n_starts):
        F = obj([lk])
        if F < best_F:
            best_F  = F;  best_lk = lk
    if verbose:
        print(f"    Grid best: k'={np.exp(best_lk):.3e}  RMSE={np.sqrt(best_F)*100:.1f}%")

    # Nelder–Mead refinement
    res   = minimize(obj, [best_lk], method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-8})
    k_opt = np.exp(float(res.x[0]))
    F_opt = float(res.fun)
    if verbose:
        print(f"    Converged: k'={k_opt:.4e}  RMSE={np.sqrt(F_opt)*100:.2f}%")
    return k_opt, F_opt


# ══════════════════════════════════════════════════════════════════════════════
# §7  CONFIDENCE INTERVALS — FISHER INFORMATION MATRIX
#     [Bard 1974; Stigter & Molenaar, Math. Biosci. Eng. 2021]
# ══════════════════════════════════════════════════════════════════════════════

def fisher_ci(k_opt: float, exp_keys: list, xa_preds: dict,
              beta_A: float = 4.877, delta_A: float = 2.287,
              alpha_lvl: float = 0.05) -> dict:
    """
    Asymptotic 95% confidence interval for k' via the Fisher information matrix.

        Var(k') = s² · (JᵀJ)⁻¹           (eq. 19)
        s²      = SSR / (N − 1)           residual variance
        J_j     = ∂XA_j/∂k'              Jacobian (central finite differences, Δk'=1%)
        CI      = k' ± t_{α/2}(N−1) √Var(k')
    """
    dk = k_opt * 0.01
    all_obs, all_pred, all_J = [], [], []

    for key in exp_keys:
        d    = EXPERIMENTS[key]
        t_s  = np.array(d["t_min"]) * 60.0
        all_obs.extend(d["XA"])
        all_pred.extend(xa_preds[key])

        p_hi = make_params(k_opt+dk, d["flow"], d["mass"], beta_A, delta_A, Nz=10, Nx=6)
        p_lo = make_params(k_opt-dk, d["flow"], d["mass"], beta_A, delta_A, Nz=10, Nx=6)
        J    = (predict_XA(p_hi, t_s) - predict_XA(p_lo, t_s)) / (2.0 * dk)
        all_J.extend(J)

    xa_obs = np.array(all_obs);   xa_pred = np.array(all_pred);   J_arr = np.array(all_J)
    n   = len(xa_obs)
    SSR = np.sum((xa_obs - xa_pred)**2)
    s2  = SSR / max(n - 1, 1)
    JtJ = np.dot(J_arr, J_arr)
    se  = np.sqrt(s2 / max(JtJ, 1e-30))
    tc  = t_dist.ppf(1.0 - alpha_lvl/2, df=n-1)
    return {"lower": k_opt - tc*se, "upper": k_opt + tc*se, "se": se, "df": n-1}


# ══════════════════════════════════════════════════════════════════════════════
# §8  GRID CONVERGENCE — RICHARDSON EXTRAPOLATION
#     [Schiesser & Griffiths 2009; Roache 1998]
# ══════════════════════════════════════════════════════════════════════════════

def grid_convergence(k_prime: float = 2.728e-3, t_check_h: float = 6.0) -> dict:
    """
    Compute XA at t_check_h for increasing grid resolutions.
    Richardson extrapolation (refinement ratio r ≈ 1.25) provides the
    asymptotic reference value.
    """
    grids = [(4,3), (6,4), (8,5), (10,6)]
    XA_v, cpu_v = [], []
    t_s = t_check_h * 3600.0
    for (Nz_g, Nx_g) in grids:
        p  = make_params(k_prime, Nz=Nz_g, Nx=Nx_g)
        t0 = time.perf_counter()
        res = simulate(p, t_s+100, n_eval=20, rtol=1e-5, atol=1e-8)
        cpu_v.append(time.perf_counter() - t0)
        XA_v.append(float(np.interp(t_s, res["t"], res["XA"])))
    XA_rich = XA_v[-1] + (XA_v[-1] - XA_v[-2]) / (1.25**2 - 1)
    return {"grids": grids, "XA_values": XA_v, "XA_richardson": XA_rich, "cpu_times": cpu_v}


# ══════════════════════════════════════════════════════════════════════════════
# §9  MORRIS ELEMENTARY EFFECTS — GLOBAL SENSITIVITY ANALYSIS
#     [Morris 1991; Campolongo et al. 2007; Saltelli et al. 2021]
# ══════════════════════════════════════════════════════════════════════════════

PARAM_NAMES  = ["beta_A","delta_A","gamma","eps_P","tau_LT","eps_LT","k_prime","alpha_A","lam_A"]
PARAM_LABELS = ["β_A (pore diff.)", "δ_A (L-S coupling)", "γ (bed res. time)",
                "ε_P (porosity)", "τ_LT (tank space time)", "ε_LT (tank holdup)",
                "k′ (rate const.)", "α_A (axial disp.)", "λ_A (Biot⁻¹)"]
PARAM_NOM    = {"beta_A":4.877,"delta_A":2.287,"gamma":3.50,"eps_P":0.45,
                "tau_LT":round(TAU_LT_REF,1),"eps_LT":0.85,"k_prime":9.78e-3,"alpha_A":0.010,"lam_A":0.001}
PARAM_BND    = {k: (v*0.5, v*1.5) for k, v in PARAM_NOM.items()}


def _morris_eval(theta: np.ndarray, t_s: float) -> float:
    """Evaluate XA at t_s for unit-cube parameter vector theta."""
    kw = {pn: PARAM_BND[pn][0] + theta[i]*(PARAM_BND[pn][1]-PARAM_BND[pn][0])
          for i, pn in enumerate(PARAM_NAMES)}
    p = ModelParams(Nz=10, Nx=6, **{k: kw[k] for k in kw})
    try:
        res = simulate(p, t_s+100, n_eval=12, rtol=1e-4, atol=1e-7)
        return float(np.interp(t_s, res["t"], res["XA"]))
    except Exception:
        return np.nan


def morris_screening(t_eval_hours: list,
                     r: int = 20, p_levels: int = 10, seed: int = 42) -> dict:
    """
    Morris elementary effects (EE) global sensitivity screening.

        EE_i(x) = [F(x₁,…,xᵢ+Δ,…,xₖ) − F(x)] / Δ        (eq. 20)

    Statistics:
        μ* = mean|EE_i|   → first-order importance
        σ  = std(EE_i)    → interaction / non-linearity
        EI = sign(mean EE) × μ* / XA_base   → normalised for heatmap

    Parameters: r=4 trajectories, p=10 grid levels (Δ=1/9), k=9 parameters.
    Total evaluations: r·(k+1) = 40 per time point.
    [Morris, Technometrics 1991; Campolongo et al., Env. Mod. Softw. 2007]
    """
    rng = np.random.default_rng(seed)
    k   = len(PARAM_NAMES)
    delta = 1.0 / (p_levels - 1)
    out = {}
    for t_h in t_eval_hours:
        t_s  = t_h * 3600.0
        EEs  = np.full((r, k), np.nan)
        for traj in range(r):
            x    = rng.integers(0, p_levels-1, size=k) / (p_levels-1)
            perm = rng.permutation(k)
            F0   = _morris_eval(x, t_s)
            if np.isnan(F0):
                continue
            for pidx in perm:
                xn    = x.copy()
                sign  = 1.0 if x[pidx]+delta <= 1.0 else -1.0
                xn[pidx] += sign * delta
                F1    = _morris_eval(xn, t_s)
                if not np.isnan(F1):
                    EEs[traj, pidx] = (F1 - F0) / (sign * delta)
                x  = xn
                F0 = F1 if not np.isnan(F1) else F0
        mu_star  = np.nanmean(np.abs(EEs), axis=0)
        mu_signed = np.nanmean(EEs, axis=0)
        sigma    = np.nanstd(EEs, axis=0)
        XA_base  = _morris_eval(np.full(k, 0.5), t_s)
        EI       = np.sign(np.mean(EEs, axis=0)) * mu_star / max(abs(XA_base), 1e-6)
        out[t_h] = {"mu_star": mu_star, "mu": mu_signed, "sigma": sigma, "EI": EI, "XA_base": XA_base}
        print(f"  Morris t={t_h}h: XA_base={XA_base*100:.1f}%  top: {PARAM_LABELS[np.argmax(mu_star)]}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §10  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    if pio.kaleido.scope is not None:        # ← safe guard
        pio.kaleido.scope.mathjax = None
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
except AttributeError:                       # ← also catch unexpected kaleido errors
    PLOTLY_OK = True                         # kaleido is installed, just skip mathjax


COLORS = ["#1d4ed8","#dc2626","#16a34a","#9333ea","#ea580c","#0891b2"]
AX   = dict(showline=True, linecolor="#999", mirror=True, gridcolor="#f0f0f0", ticks="outside")
BASE = dict(font=dict(family="Arial,sans-serif", size=13), plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="#ddd", borderwidth=1),
            margin=dict(l=70, r=50, t=100, b=70))


def sv(fig, name: str, w: int = 960, h: int = 520):
    if not PLOTLY_OK:
        return
    path = f"results/{name}"
    m = fig.layout.margin; fig.update_layout(title=None, margin=dict(l=m.l, r=m.r, b=m.b, t=60))   # captions go in the manuscript, not in the image
    try:
        fig.write_image(path, width=w, height=h, scale=4)
        print(f"  Saved {name}")
    except Exception as e:
        print(f"  Could not save {name}: {e}")


def make_all_figures(k_097, k_162, beta_A, delta_A,
                     preds, all_obs, all_pred_arr, R2, RMSE,
                     gc, morris, ci_097, ci_162,
                     res_2nd, res_lhw, KA_lit,
                     res_dyn, masses, flows_su, scaleup, flows_t, t50_arr, t90_arr):
    """Generate all 11 publication-quality figures and save to ./results/."""

    keys = list(EXPERIMENTS.keys())

    # ── Fig 01: Three-component tank dynamics ─────────────────────────────
    t_h = res_dyn["t"] / 3600
    Nzi_d = res_dyn["y_bed"].shape[1]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["(a) Dimensionless Tank Concentrations", "(b) Arabinose Conversion X_A"])
    for ci_i, (name, clr, ls) in enumerate([
        ("Arabinose (A)", "#1d4ed8", "solid"),
        ("Dissolved O₂ (B)  [≡1]", "#f97316", "dot"),
        ("Arabinonic Acid (C)", "#16a34a", "dash")]):
        yv = res_dyn["y_tank"][0] if ci_i==0 else np.ones_like(t_h) if ci_i==1 else res_dyn["y_tank"][2]
        fig.add_trace(go.Scatter(x=t_h, y=yv, name=name, line=dict(color=clr, width=2.5, dash=ls)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_h, y=res_dyn["XA"]*100,
        line=dict(color="#1d4ed8", width=3), showlegend=False), row=1, col=2)
    for xv in [50, 90]:
        fig.add_hline(y=xv, line_dash="dot", line_color="#bbb", line_width=1.2, row=1, col=2)
    fig.update_xaxes(**AX, title_text="Time [h]")
    fig.update_yaxes(**AX, title_text="Dimensionless Concentration [-]", row=1, col=1)
    fig.update_yaxes(**AX, title_text="X_A [%]", row=1, col=2)
    fig.update_layout(**BASE, width=1100, height=480,
        title="Fig. 1 — Three-component model dynamics (70 mL/min, 0.897 g, second-order)")
    sv(fig, "Fig01_dynamics.png", w=1100, h=480)

    # ── Fig 02: Axial bed profiles ────────────────────────────────────────
    z_arr = np.linspace(0, 1, Nzi_d)
    times_h = [0.5, 1, 3, 6, 12, 24]
    pal = ["#bfdbfe","#93c5fd","#60a5fa","#3b82f6","#1d4ed8","#1e3a8a"]
    fig2 = go.Figure()
    for th, clr in zip(times_h, pal):
        tidx = np.argmin(np.abs(res_dyn["t"] - th*3600))
        fig2.add_trace(go.Scatter(x=z_arr, y=res_dyn["y_bed"][0, :, tidx],
            name=f"t = {th} h", line=dict(color=clr, width=2.5)))
    fig2.update_layout(**BASE, width=820, height=500,
        title="Fig. 2 — Arabinose axial concentration profiles y_A(z,t) in packed bed",
        xaxis=dict(**AX, title="Dimensionless axial coordinate z = l/L"),
        yaxis=dict(**AX, title="y_A [-]", range=[0, 1.05]))
    sv(fig2, "Fig02_bed_profiles.png", w=820, h=500)

    # ── Fig 03: Pore profiles ─────────────────────────────────────────────
    Nx_d  = res_dyn["Y_pore"].shape[2]
    x_arr = np.linspace(0, 1, Nx_d)
    mid   = Nzi_d // 2
    fig3  = make_subplots(rows=1, cols=2,
        subplot_titles=["(a) Arabinose A  (fitted β_A, Φ = 0.035)",
                        "(b) Dissolved O₂ B  (Y_B ≡ 1 imposed)"])
    for th, clr in [(1,"#93c5fd"),(3,"#3b82f6"),(6,"#1d4ed8"),(12,"#1e3a8a")]:
        tidx = np.argmin(np.abs(res_dyn["t"] - th*3600))
        fig3.add_trace(go.Scatter(x=x_arr, y=res_dyn["Y_pore"][0, mid, :, tidx],
            name=f"t={th}h", line=dict(color=clr, width=2.5)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=x_arr, y=np.ones(Nx_d),
            line=dict(color=clr, width=2.5, dash="dash"), showlegend=False), row=1, col=2)
    fig3.update_xaxes(**AX, title_text="x (slab: 0 = centre, 1 = surface)")
    fig3.update_yaxes(**AX, title_text="Y [-]", row=1, col=1)
    fig3.update_yaxes(**AX, title_text="Y_B [-]", range=[0.99, 1.01], row=1, col=2)
    fig3.update_layout(**BASE, width=1100, height=480,
        title="Fig. 3 — Intraparticle concentration profiles at bed mid-point (z = 0.5)")
    sv(fig3, "Fig03_pore_profiles.png", w=1100, h=480)

    # ── Fig 04: Second-order vs LHHW ─────────────────────────────────────
    t90_2nd = float(np.interp(0.9, res_2nd["XA"], res_2nd["t"])) / 60
    t90_lhw = float(np.interp(0.9, res_lhw["XA"], res_lhw["t"])) / 60
    fig4 = go.Figure([
        go.Scatter(x=res_2nd["t"]/60, y=res_2nd["XA"]*100,
            name=f"Second-order  (t₉₀={t90_2nd:.0f} min)",
            line=dict(color="#1d4ed8", width=3)),
        go.Scatter(x=res_lhw["t"]/60, y=res_lhw["XA"]*100,
            name=f"LHHW  K_A={KA_lit}  (t₉₀={t90_lhw:.0f} min)",
            line=dict(color="#dc2626", width=3, dash="dash")),
    ])
    for xv in [50, 90]:
        fig4.add_hline(y=xv, line_dash="dot", line_color="#bbb", line_width=1.2)
    fig4.add_annotation(x=1100, y=38,
        text=(f"Denominator (1+K_A·Y_A)²<br>"
              f"Y_A=1.0: {(1+KA_lit)**2:.0f}<br>"
              f"Y_A=0.5: {(1+KA_lit*0.5)**2:.1f}<br>"
              f"Y_A=0.1: {(1+KA_lit*0.1)**2:.2f}"),
        showarrow=False, align="left", font=dict(size=11),
        bgcolor="rgba(255,240,240,0.9)", bordercolor="#dc2626", borderwidth=1)
    fig4.update_layout(**BASE, width=860, height=500,
        title="Fig. 4 — Second-order vs LHHW kinetics (equal initial rates; K_A=11.44, K_B=0)",
        xaxis=dict(**AX, title="Time [min]", range=[0, 1800]),
        yaxis=dict(**AX, title="X_A [%]", range=[0, 105]))
    sv(fig4, "Fig04_kinetics.png", w=860, h=500)

    # ── Fig 05: Parameter estimation diagnostics ──────────────────────────
    fig5 = make_subplots(rows=1, cols=2,
        subplot_titles=["(a) Fitted k′ with 95 % CI", "(b) Residuals (predicted − observed)"],
        column_widths=[0.32, 0.68])
    for i_r, (lbl, val, lo, hi, clr) in enumerate([
        ("0.897 g", k_097*1e3, ci_097["lower"]*1e3, ci_097["upper"]*1e3, "#1d4ed8"),
        ("1.6225 g (×10)", k_162*1e4, ci_162["lower"]*1e4, ci_162["upper"]*1e4, "#dc2626"),
    ]):
        fig5.add_trace(go.Scatter(y=[lbl], x=[val], mode="markers",
            marker=dict(color=clr, size=14, symbol="diamond"),
            error_x=dict(type="data", symmetric=False,
                         array=[hi-val], arrayminus=[val-lo],
                         color=clr, thickness=2, width=8), name=lbl, showlegend=False), row=1, col=1)
    fig5.update_xaxes(**AX, title_text="k′ [10⁻³ s⁻¹]", row=1, col=1)
    for idx, key in enumerate(keys):
        d   = EXPERIMENTS[key]
        xa_p = preds[key]["xa_pred"]
        resid = (xa_p - np.array(d["XA"])) * 100
        fig5.add_trace(go.Scatter(x=xa_p*100, y=resid, mode="markers",
            marker=dict(color=COLORS[idx], size=7, opacity=0.85),
            name=d["label"]), row=1, col=2)
    fig5.add_hline(y=0, line_color="#555", line_width=1.5, row=1, col=2)
    fig5.add_hrect(y0=-RMSE, y1=RMSE, fillcolor="rgba(180,180,180,0.15)",
        line_width=0, row=1, col=2)
    fig5.add_annotation(x=90, y=-RMSE-1.5, text=f"±RMSE={RMSE:.1f}%",
        showarrow=False, font=dict(size=10, color="#888"), row=1, col=2)
    fig5.update_xaxes(**AX, title_text="Predicted X_A [%]", row=1, col=2)
    fig5.update_yaxes(**AX, title_text="Residual [%]", row=1, col=2)
    fig5.update_layout(font=dict(family="Arial,sans-serif",size=13),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1100, height=480,
        legend=dict(x=1.02, y=1.0, xanchor="left", bgcolor="rgba(255,255,255,0.92)", bordercolor="#ddd", borderwidth=1),
        margin=dict(l=70,r=220,t=100,b=70),
        title="Fig. 5 — Parameter estimation: (a) 95% CIs from FIM; (b) residual plot")
    sv(fig5, "Fig05_estimation.png", w=1250, h=480)

    # ── Fig 06: Grid convergence ──────────────────────────────────────────
    nz_lbl = [f"Nz={g[0]}, Nx={g[1]}" for g in gc["grids"]]
    err_pct = [(v - gc["XA_richardson"])/gc["XA_richardson"]*100 for v in gc["XA_values"]]
    fig6 = make_subplots(rows=1, cols=2,
        subplot_titles=["(a) X_A(6h) vs. grid resolution", "(b) Error vs. Richardson reference [%]"])
    fig6.add_trace(go.Scatter(x=nz_lbl, y=[v*100 for v in gc["XA_values"]],
        mode="lines+markers", marker=dict(size=11, color="#1d4ed8"),
        line=dict(color="#1d4ed8", width=2.5), name="X_A"), row=1, col=1)
    fig6.add_hline(y=gc["XA_richardson"]*100, line_dash="dot", line_color="#dc2626",
        line_width=1.5, annotation_text=f"Richardson {gc['XA_richardson']*100:.2f}%",
        annotation_font_size=11, row=1, col=1)
    fig6.add_trace(go.Bar(x=nz_lbl, y=err_pct, marker_color="#1d4ed8",
        opacity=0.75, name="Error"), row=1, col=2)
    fig6.add_hline(y=0, line_color="#555", line_width=1, row=1, col=2)
    fig6.update_xaxes(**AX, title_text="Grid", row=1, col=1)
    fig6.update_yaxes(**AX, title_text="X_A [%]", row=1, col=1)
    fig6.update_xaxes(**AX, title_text="Grid", row=1, col=2)
    fig6.update_yaxes(**AX, title_text="Error [%]", row=1, col=2)
    fig6.update_layout(**BASE, width=1100, height=460,
        title="Fig. 6 — Grid convergence: X_A(t=6h, 70 mL/min, 0.897g) vs. Nz/Nx")
    sv(fig6, "Fig06_grid_convergence.png", w=1100, h=460)

    # ── Fig 07: Six-panel calibration ──────────────────────────────────────
    fig7 = make_subplots(rows=2, cols=3,
        subplot_titles=[EXPERIMENTS[k]["label"] for k in keys],
        horizontal_spacing=0.08, vertical_spacing=0.12)
    for idx, key in enumerate(keys):
        d   = EXPERIMENTS[key]
        res = preds[key]["res"]
        xa_p = preds[key]["xa_pred"]
        rmse_k = np.sqrt(np.mean((np.array(d["XA"])-xa_p)**2))*100
        bias_k = np.mean((xa_p-np.array(d["XA"]))*100)
        r, c = idx//3+1, idx%3+1
        fig7.add_trace(go.Scatter(x=np.array(d["t_min"])/60, y=np.array(d["XA"])*100,
            mode="markers", marker=dict(color=COLORS[idx], size=7, opacity=0.9,
            line=dict(color="white",width=0.5)), showlegend=False), row=r, col=c)
        fig7.add_trace(go.Scatter(x=res["t"]/3600, y=res["XA"]*100,
            line=dict(color=COLORS[idx], width=2.5), showlegend=False), row=r, col=c)
        fig7.add_annotation(row=r, col=c, x=0.97, y=0.06,
            xref="x domain", yref="y domain",
            text=f"RMSE={rmse_k:.1f}%  bias={bias_k:+.1f}%",
            showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.88)", align="right")
    fig7.update_xaxes(**AX, title_text="Time [h]")
    fig7.update_yaxes(**AX, title_text="X_A [%]")
    fig7.update_layout(**BASE, width=1100, height=680,
        title=f"Model calibration: symbols=experiment, lines=model. R²={R2:.4f}  RMSE={RMSE:.1f}%  N={sum(len(EXPERIMENTS[k]["XA"]) for k in EXPERIMENTS)}")
    sv(fig7, "Fig07_validation.png", w=1100, h=680)

    # ── Fig 08: Parity + residual histogram ───────────────────────────────
    resids_all = (all_pred_arr - all_obs)*100
    fig8 = make_subplots(rows=1, cols=2,
        subplot_titles=["(a) Parity Plot", "(b) Residual Distribution"])
    x_id = list(range(-2, 105))
    fig8.add_trace(go.Scatter(x=x_id, y=[v+RMSE for v in x_id], mode="lines",
        line=dict(color="#ddd",width=1), showlegend=False), row=1, col=1)
    fig8.add_trace(go.Scatter(x=x_id, y=[v-RMSE for v in x_id], mode="lines",
        line=dict(color="#ddd",width=1), fill="tonexty",
        fillcolor="rgba(180,180,180,0.15)", showlegend=False), row=1, col=1)
    fig8.add_trace(go.Scatter(x=x_id, y=x_id, mode="lines",
        line=dict(color="#333",width=1.5,dash="dash"), name="y = x"), row=1, col=1)
    for idx, key in enumerate(keys):
        d = EXPERIMENTS[key]
        fig8.add_trace(go.Scatter(x=np.array(d["XA"])*100,
            y=preds[key]["xa_pred"]*100, mode="markers",
            marker=dict(color=COLORS[idx],size=7,opacity=0.85),
            name=d["label"]), row=1, col=1)
    fig8.add_annotation(x=5, y=92, align="left", xanchor="left",
        text=f"<b>R² = {R2:.4f}</b><br>RMSE = {RMSE:.1f}%<br>N = {sum(len(EXPERIMENTS[k]["XA"]) for k in EXPERIMENTS)}",
        showarrow=False, font=dict(size=12),
        bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc", borderwidth=1, row=1, col=1)
    fig8.update_xaxes(**AX, title_text="Observed X_A [%]", range=[-2,103], row=1, col=1)
    fig8.update_yaxes(**AX, title_text="Predicted X_A [%]", range=[-2,103], row=1, col=1)
    fig8.add_trace(go.Histogram(x=resids_all, nbinsx=25,
        marker_color="#1d4ed8", opacity=0.75, showlegend=False), row=1, col=2)
    fig8.add_vline(x=0, line_color="#dc2626", line_width=1.5, row=1, col=2)
    fig8.add_vline(x=float(np.mean(resids_all)), line_color="#f97316",
        line_width=1.5, line_dash="dash",
        annotation_text=f"mean={np.mean(resids_all):+.1f}%", annotation_font_size=11, row=1, col=2)
    fig8.update_xaxes(**AX, title_text="Residual [%]", row=1, col=2)
    fig8.update_yaxes(**AX, title_text="Count", row=1, col=2)
    fig8.update_layout(font=dict(family="Arial,sans-serif",size=13),
        plot_bgcolor="white", paper_bgcolor="white", width=1100, height=500,
        margin=dict(l=70,r=50,t=100,b=90),
        legend=dict(x=0.0, y=-0.22, orientation="h", bgcolor="rgba(255,255,255,0.92)"),
        title=f"Fig. 8 — Parity plot and residual distribution (mean={np.mean(resids_all):+.2f}%)")
    sv(fig8, "Fig08_parity.png", w=1100, h=500)

    # ── Fig 09: Morris sensitivity heatmap ────────────────────────────────
    t_hrs   = list(morris.keys())
    t_lbl   = [f"{t}h\n(X_A≈{morris[t]['XA_base']*100:.0f}%)" for t in t_hrs]

    z_mat   = [[morris[t]["mu_star"][i]/max(morris[t]["XA_base"],1e-6) for t in t_hrs] for i in range(len(PARAM_NAMES))]
    vmax    = max(abs(v) for row in z_mat for v in row) or 1.0
    fig9    = go.Figure(go.Heatmap(
        z=z_mat, x=t_lbl, y=PARAM_LABELS,
        colorscale=[[0,"#ffffff"],[1,"#1d4ed8"]],
        zmin=0, zmax=vmax,
        text=[[f"{v:.2f}" for v in row] for row in z_mat],
        texttemplate="%{text}", textfont=dict(size=12),
        colorbar=dict(title="μ*/X_A")))
    fig9.update_layout(**BASE, width=860, height=520,
        title="Fig. 9 — Morris EE global sensitivity heatmap: EI across conversion regimes",
        xaxis=dict(title="Operating regime"),
        yaxis=dict(title="Parameter"))
    sv(fig9, "Fig09_sensitivity.png", w=860, h=520)

    # ── Fig 10: Scale-up map ──────────────────────────────────────────────
    pal10 = ["#bfdbfe","#60a5fa","#1d4ed8","#1e40af","#1e3a8a"]
    fig10 = go.Figure()
    for mi, mass in enumerate(masses):
        fig10.add_trace(go.Scatter(x=flows_su, y=scaleup[mi],
            mode="lines", name=f"m = {mass} g",
            line=dict(width=2.5, color=pal10[mi])))
    fig10.add_hline(y=90, line_dash="dot", line_color="#555", line_width=1.5)
    fig10.add_annotation(x=505, y=90, text="90% target", showarrow=False,
        font=dict(size=11), xanchor="left")
    fig10.update_layout(**BASE, width=860, height=500,
        title="Fig. 10 — Scale-up prediction map: X_A(t=8h) vs. Q and catalyst loading (k′ ∝ m_cat)",
        xaxis=dict(**AX, title="Recirculation Flow Rate Q [mL/min]"),
        yaxis=dict(**AX, title="X_A at 8 h [%]", range=[0, 105]))
    sv(fig10, "Fig10_scaleup.png", w=860, h=500)

    # ── Fig 11: t50/t90 vs flow ────────────────────────────────────────────
    fig11 = go.Figure([
        go.Scatter(x=flows_t, y=t50_arr, name="t₅₀%",
            mode="lines+markers", line=dict(color="#1d4ed8", width=2.5), marker=dict(size=9)),
        go.Scatter(x=flows_t, y=t90_arr, name="t₉₀%",
            mode="lines+markers", line=dict(color="#dc2626", width=2.5, dash="dash"),
            marker=dict(size=9)),
    ])
    for fl, lbl in [(40,"E1 (40)"),(70,"E2 (70)")]:
        fig11.add_vline(x=fl, line_dash="dot", line_color="#aaa", line_width=1.5,
            annotation_text=lbl, annotation_font_size=10)
    fig11.update_layout(**BASE, width=860, height=500,
        title="Fig. 11 — Time to 50% and 90% arabinose conversion vs. Q (0.897 g, k′ fixed)",
        xaxis=dict(**AX, title="Flow Rate Q [mL/min]"),
        yaxis=dict(**AX, title="Time to Conversion [min]"))
    sv(fig11, "Fig11_flow_effect.png", w=860, h=500)


# ══════════════════════════════════════════════════════════════════════════════
# §11  MAIN — orchestrates all computations
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Arabinose Oxidation Model — Hachhach & Salmi  (CEJ 2025)")
    print("  Full model · All computations · 11 figures")
    print("=" * 70)

    # Fixed structural parameters
    BETA_A  = 4.877    # pore diffusion parameter [s⁻¹]
    DELTA_A = 2.287    # pore-bulk coupling        [s⁻¹]

    # ─── A: Parameter Estimation ──────────────────────────────────────────
    print("\n[A] Parameter Estimation (Nelder-Mead, multi-start)")
    print("  0.897 g batch (E1, E2):")
    k_097, _ = estimate_k(["E1","E2"],   BETA_A, DELTA_A)
    print("  1.6225 g batch (E3-E6):")
    k_162, _ = estimate_k(["E3","E4","E5","E6"], BETA_A, DELTA_A)
    print(f"\n  k′(0.897g)  = {k_097:.4e} s⁻¹")
    print(f"  k′(1.6225g) = {k_162:.4e} s⁻¹")
    print(f"  k′ ratio    = {k_097/k_162:.2f}   (mass ratio = {1.6225/0.897:.2f})")

    K_PRIME = {"097": k_097, "162": k_162}

    # ─── B: Validation predictions (production grid) ─────────────────────
    print("\n[B] Validation predictions (Nz=10, Nx=6)")
    preds = {}
    all_obs, all_pred_flat = [], []
    for key in EXPERIMENTS:
        d    = EXPERIMENTS[key]
        t_s  = np.array(d["t_min"]) * 60.0
        batch = EXP_BATCH[key]
        p    = make_params(K_PRIME[batch], d["flow"], d["mass"], BETA_A, DELTA_A, Nz=10, Nx=6)
        t_end = max(t_s)*1.05 + 300
        res  = simulate(p, t_end, n_eval=120)
        xa_p = np.interp(t_s, res["t"], res["XA"])
        preds[key] = {"res": res, "xa_pred": xa_p}
        rmse_k = np.sqrt(np.mean((np.array(d["XA"])-xa_p)**2))*100
        bias_k = np.mean((xa_p-np.array(d["XA"]))*100)
        all_obs.extend(d["XA"]); all_pred_flat.extend(xa_p)
        print(f"  {key} ({d['label']}): RMSE={rmse_k:.1f}%  bias={bias_k:+.1f}%")

    all_obs       = np.array(all_obs)
    all_pred_arr  = np.array(all_pred_flat)
    R2   = 1.0 - np.sum((all_obs-all_pred_arr)**2) / np.sum((all_obs-all_obs.mean())**2)
    RMSE = np.sqrt(np.mean((all_obs-all_pred_arr)**2)) * 100
    print(f"\n  Overall: R²={R2:.4f}  RMSE={RMSE:.2f}%  N={len(all_obs)}")

    # ─── C: Confidence Intervals ──────────────────────────────────────────
    print("\n[C] Fisher Information Matrix — 95% CIs")
    ci_097 = fisher_ci(k_097, ["E1","E2"], {k:preds[k]["xa_pred"] for k in ["E1","E2"]},
                       BETA_A, DELTA_A)
    ci_162 = fisher_ci(k_162, ["E3","E4","E5","E6"],
                       {k:preds[k]["xa_pred"] for k in ["E3","E4","E5","E6"]},
                       BETA_A, DELTA_A)
    print(f"  k′(0.897g)  = {k_097:.4e}  [{ci_097['lower']:.4e}, {ci_097['upper']:.4e}]")
    print(f"  k′(1.6225g) = {k_162:.4e}  [{ci_162['lower']:.4e}, {ci_162['upper']:.4e}]")

    # ─── D: Dimensionless groups ──────────────────────────────────────────
    print("\n[D] Dimensionless Groups")
    phi_097 = np.sqrt(0.45 * k_097 / BETA_A)
    eta_097 = np.tanh(phi_097)/phi_097 if phi_097>1e-9 else 1.0
    phi_162 = np.sqrt(0.45 * k_162 / BETA_A)

    Da_E2   = k_097 * GAMMA_REF
    for key, d in EXPERIMENTS.items():
        pr = make_params(K_PRIME[EXP_BATCH[key]], d["flow"], d["mass"], BETA_A, DELTA_A)
        print(f"  {key}: gamma={pr.gamma:5.1f} s  tau_LT={pr.tau_LT:6.1f} s  Pe={1/(pr.alpha_A*pr.gamma):5.1f}  Da={pr.k_prime*pr.gamma:.4f}")
    print(f"  Thiele Φ(0.897g) = {phi_097:.4f}   η = {eta_097:.4f}")
    print(f"  Thiele Φ(1.625g) = {phi_162:.4f}   η ≈ 1.000")
    print(f"  Damköhler Da(E2) = {Da_E2:.3f}")
    print("  Note: Phi is for the fitted (lumped) beta_A only; see analyses.py for the physically consistent case")

    # ─── E: Grid convergence ──────────────────────────────────────────────
    print("\n[E] Grid Convergence (XA at t=6h, 70 mL/min, 0.897 g)")
    gc = grid_convergence(k_097)
    for i,(Nz_g,Nx_g) in enumerate(gc["grids"]):
        err = (gc["XA_values"][i]-gc["XA_richardson"])/gc["XA_richardson"]*100
        print(f"  Nz={Nz_g}, Nx={Nx_g}: XA={gc['XA_values'][i]*100:.2f}%  "
              f"err={err:+.2f}%  CPU={gc['cpu_times'][i]:.2f}s")
    print(f"  Richardson reference: {gc['XA_richardson']*100:.2f}%")

    # ─── F: Morris sensitivity ────────────────────────────────────────────
    print("\n[F] Morris Elementary Effects (r=20 trajectories, p=10 levels)")
    morris = morris_screening([1.0, 5.0, 10.0], r=20, p_levels=10)

    # ─── G: LHHW vs second-order ──────────────────────────────────────────
    print("\n[G] LHHW vs Second-Order Kinetics")
    KA_lit  = 11.44
    k_lhhw  = k_097 * (1.0 + KA_lit)**2
    p_2nd = make_params(k_097, 70.0, 0.897, BETA_A, DELTA_A, Nz=8, Nx=5)
    p_lhw = make_params(k_lhhw, 70.0, 0.897, BETA_A, DELTA_A, Nz=8, Nx=5, K_A=KA_lit, K_B=0.0)
    res_2nd = simulate(p_2nd, 110000, rate_law="second_order", n_eval=200)
    res_lhw = simulate(p_lhw, 110000, rate_law="lhhw",         n_eval=200)
    t90_2nd = float(np.interp(0.9, res_2nd["XA"], res_2nd["t"]))/60
    t90_lhw = float(np.interp(0.9, res_lhw["XA"], res_lhw["t"]))/60
    print(f"  k′_LHHW = k′ × {(1+KA_lit)**2:.1f} = {k_lhhw:.4e} s⁻¹")
    print(f"  t₉₀ (2nd-order) = {t90_2nd:.0f} min")
    print(f"  t₉₀ (LHHW)      = {t90_lhw:.0f} min  ({t90_2nd/t90_lhw:.1f}× faster)")

    # ─── H: Full-resolution dynamics ──────────────────────────────────────
    print("\n[H] Full-resolution dynamics (Nz=10, Nx=6)")
    p_dyn   = make_params(k_097, 70.0, 0.897, BETA_A, DELTA_A, Nz=10, Nx=6)
    res_dyn = simulate(p_dyn, 108000, n_eval=200)
    print(f"  XA(1h) = {np.interp(3600,res_dyn['t'],res_dyn['XA'])*100:.1f}%")
    print(f"  XA(6h) = {np.interp(21600,res_dyn['t'],res_dyn['XA'])*100:.1f}%")

    # ─── I: Scale-up predictions ──────────────────────────────────────────
    print("\n[I] Scale-Up Analysis")
    masses   = [0.5, 1.0, 2.0, 4.0, 8.0]
    flows_su = np.linspace(20, 500, 20)
    t_8h     = 8.0 * 3600.0
    scaleup  = np.zeros((len(masses), len(flows_su)))
    for mi, mass in enumerate(masses):
        for fi, flow in enumerate(flows_su):
            p   = make_params(k_097, flow, mass, BETA_A, DELTA_A, Nz=10, Nx=6)
            res = simulate(p, t_8h+100, n_eval=35)
            scaleup[mi, fi] = float(np.interp(t_8h, res["t"], res["XA"])) * 100
        hit = np.where(scaleup[mi] >= 90)[0]
        q90 = f"{flows_su[hit[0]]:.0f}" if len(hit)>0 else ">500"
        print(f"  {mass}g:  90% at 8h requires Q ≥ {q90} mL/min")

    flows_t = np.array([30,40,50,70,100,150,200,300,400,500])
    t50_arr, t90_arr = [], []
    for flow in flows_t:
        p   = make_params(k_097, flow, 0.897, BETA_A, DELTA_A, Nz=6, Nx=4)
        res = simulate(p, 600000, n_eval=200)
        t50_arr.append(float(np.interp(0.50, res["XA"], res["t"]))/60)
        t90_arr.append(float(np.interp(0.90, res["XA"], res["t"]))/60)

    # ─── J: Generate all figures ──────────────────────────────────────────
    if PLOTLY_OK:
        print("\n[J] Generating 11 figures ...")
        make_all_figures(
            k_097, k_162, BETA_A, DELTA_A,
            preds, all_obs, all_pred_arr, R2, RMSE,
            gc, morris, ci_097, ci_162,
            res_2nd, res_lhw, KA_lit,
            res_dyn, masses, flows_su, scaleup, flows_t, t50_arr, t90_arr
        )
    else:
        print("\n  Install plotly+kaleido for figures.")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  k′(0.897g)  = {k_097:.4e} s⁻¹  95%CI [{ci_097['lower']:.4e}, {ci_097['upper']:.4e}]")
    print(f"  k′(1.6225g) = {k_162:.4e} s⁻¹  95%CI [{ci_162['lower']:.4e}, {ci_162['upper']:.4e}]")
    print(f"  β_A = {BETA_A} s⁻¹   δ_A = {DELTA_A} s⁻¹")
    print(f"  Thiele Φ    = {phi_097:.4f}  →  η = {eta_097:.4f}  (fitted beta_A only)")
    print(f"  R²  = {R2:.4f}   RMSE = {RMSE:.2f}%   N = {len(all_obs)}")
    print(f"  t₉₀ 2nd-order = {t90_2nd:.0f} min")
    print(f"  t₉₀ LHHW     = {t90_lhw:.0f} min  ({t90_2nd/t90_lhw:.1f}× faster)")
    print(f"\n  Figures → ./results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
