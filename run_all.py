"""
run_all.py: regenerates every number, table and figure of the revised manuscript.

    python run_all.py            # full run (about 45 min on a laptop)
    python run_all.py --reuse    # rebuild tables/figures from existing JSON results (seconds)

Outputs (no narrative text; everything is written to files):
    results/tables/*.csv          one CSV per manuscript/SI table
    results/all_tables.xlsx       the same tables, one sheet each
    results/figures/*.png         all figures (600 dpi)
    results/logs/*.log            solver/console logs of each step
"""
import os, sys, json, runpy, contextlib, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results"); TAB = os.path.join(OUT, "tables"); FIG = os.path.join(OUT, "figures"); LOG = os.path.join(OUT, "logs")
for d in (TAB, FIG, LOG): os.makedirs(d, exist_ok=True)
REUSE = "--reuse" in sys.argv
import tandem_model as tm

def run(script, product):
    """Run one analysis script with its console output sent to a log file."""
    if REUSE and os.path.exists(product): return
    with open(os.path.join(LOG, script.replace(".py", ".log")), "w") as f, contextlib.redirect_stdout(f):
        runpy.run_path(script, run_name="__main__")

# ── 1. Analyses ───────────────────────────────────────────────────────────────
run("analyses.py", "results_main.json")
if not REUSE or not os.path.exists(os.path.join(HERE, "results", "Fig01_dynamics.png")):
    run("tandem_model.py", "__none__")           # model figures (plotly, 600 dpi)

R  = json.load(open("results_main.json")); R2 = json.load(open("results_identifiability.json"))
C  = json.load(open("results_transport.json"));     G  = json.load(open("results_k_gamma.json"))
SP = json.load(open("results_shared_peclet.json"))
PI = json.load(open("results_product_inhibition.json"))
E  = tm.EXPERIMENTS; G097, G162 = ["E1", "E2"], ["E3", "E4", "E5", "E6"]
T  = {}

# ── 2. Tables ─────────────────────────────────────────────────────────────────
f = R["fit"]
T["Table1_fit_statistics"] = pd.DataFrame(
    [dict(run=q, mass_g=E[q]["mass"], Q_mL_min=E[q]["flow"], N=v["N"], RMSE_pct=v["RMSE"], bias_pct=v["bias"],
          heldout_RMSE_pct=R["loo"][q]["RMSE_heldout"], heldout_bias_pct=R["loo"][q]["bias_heldout"]) for q, v in f["runs"].items()]
    + [dict(run="All", N=sum(v["N"] for v in f["runs"].values()), RMSE_pct=f["RMSE"], bias_pct=f["bias_pooled"],
            heldout_RMSE_pct=float(np.sqrt(sum(f["runs"][q]["N"] * R["loo"][q]["RMSE_heldout"] ** 2 for q in E) / sum(v["N"] for v in f["runs"].values()))))])

# k' confidence intervals (FIM, conditional on fixed beta_A, delta_A)
ci = {}
for name, grp in (("k_prime_0.897g", G097), ("k_prime_1.6225g", G162)):
    with contextlib.redirect_stdout(open(os.path.join(LOG, "estimation.log"), "a")):
        k, _ = tm.estimate_k(grp, 4.877, 2.287)          # Section 4.2: log-grid search + Nelder-Mead
    preds = {q: tm.predict_XA(tm.make_params(k, E[q]["flow"], E[q]["mass"]), np.array(E[q]["t_min"]) * 60) for q in grp}
    with contextlib.redirect_stdout(open(os.path.join(LOG, "fisher_ci.log"), "a")):
        out = tm.fisher_ci(k, grp, preds)
    ci[name] = (k, out["lower"], out["upper"])
T["Table3_parameters"] = pd.DataFrame(
    [dict(parameter=n, estimate=v[0], CI95_low=v[1], CI95_high=v[2], unit="s-1") for n, v in ci.items()]
    + [dict(parameter="beta_A", estimate=4.877, CI95_low="not identifiable", CI95_high="", unit="s-1"),
       dict(parameter="delta_A", estimate=2.287, CI95_low="not identifiable", CI95_high="", unit="s-1"),
       dict(parameter="k_obs_E2 (identifiable)", estimate=R2["kobs_ci"]["est"], CI95_low=R2["kobs_ci"]["lo"], CI95_high=R2["kobs_ci"]["hi"], unit="h-1")])

T["Table4_dimensionless_groups"] = pd.DataFrame(
    [dict(run=q, Q_mL_min=v["Q"], gamma_s=v["gamma"], tau_LT_s=v["tau_LT"], Pe=v["Pe"], Da=v["Da"],
          X_pass_pct_1h=v["X_pass_pct"], O2_demand_over_cstar=v["ratio_demand_to_cstar_pure"],
          observed_rate_per_h=v["obs_rate_per_h"], O2_ceiling_per_h=v["O2_ceiling_per_h_pure"]) for q, v in R["dims"].items()])

fim = R["fim"]
T["TableS1_FIM_correlation"] = pd.DataFrame(fim["corr"], index=fim["names"], columns=fim["names"]).reset_index().rename(columns={"index": "parameter"})
T["TableS1b_FIM_summary"] = pd.DataFrame([dict(condition_number=fim["cond"], **{f"relSE_{n}": s for n, s in zip(fim["names"], fim["rel_se"])})])

rows = []
for t in ("1.0", "5.0", "10.0"):
    XA = np.mean([R["morris"][s][t]["XA"] for s in R["morris"]])
    for i, n in enumerate(R["morris_names"]):
        g = lambda k: [R["morris"][s][t][k][i] / XA for s in R["morris"]]
        rows.append(dict(t_h=float(t), XA_base=XA, parameter=n, mu_star_norm=np.mean(g("mu_star")), mu_star_min=min(g("mu_star")),
                         mu_star_max=max(g("mu_star")), mu_norm=np.mean(g("mu")), sigma_norm=np.mean(g("sigma"))))
T["TableS2_morris"] = pd.DataFrame(rows)
T["TableS3_per_run_k"] = pd.DataFrame([dict(run=q, mass_g=v["m"], Q_mL_min=v["Q"], k_prime=v["k"], RMSE_pct=v["RMSE"],
                                           RMSE_shared_pct=SP["shared"][q]["RMSE"], bias_shared_pct=SP["shared"][q]["bias"]) for q, v in R2["per_run"].items()]
                                      + [dict(run="shared (all six)", k_prime=R2["shared_k"]["k"], RMSE_pct=R2["shared_k"]["RMSE"])])
T["TableS4_joint_beta_delta_scaling"] = pd.DataFrame(R2["joint_scaling"])
T["TableS5_kobs_profile"] = pd.DataFrame(R2["kobs_profile"])
T["TableS6_transport_criteria"] = pd.DataFrame(
    [dict(run=q, mass_g=C[q]["m"], Q_mL_min=C[q]["Q"], Re_p=C[q]["Re_p"], WeiszPrater_min=C[q]["WeiszPrater"][0], WeiszPrater_max=C[q]["WeiszPrater"][1],
          Carberry_min=C[q]["Carberry"][0], Carberry_max=C[q]["Carberry"][1], dP_Pa=C[q]["dP_Pa"], pump_W=C[q]["pump_W"], Dt_over_dp_min=C["Dt_over_dp"][0],
          Dt_over_dp_max=C["Dt_over_dp"][1]) for q in E])
T["TableS7_profile_beta"] = pd.DataFrame(R["profile_beta"])
T["TableS8_alternative_models"] = pd.DataFrame([dict(model=k, **v) for k, v in R["alternatives"].items()] + [dict(model=k, **v) for k, v in PI.items()])
T["TableS9_oxygen_dynamic"] = pd.DataFrame([dict(run=k, **v) for k, v in R["o2_dynamic"].items()])
T["TableS10_convergence_E6_grid"] = pd.DataFrame([dict(Nz=g["grid"][0], Nx=g["grid"][1], X_mean=g["X_mean"], RMSE_pct=g["RMSE"], cpu_s=g["cpu"]) for g in R["convergence_E6"]["grid"]])
T["TableS11_convergence_E6_tol"] = pd.DataFrame(R["convergence_E6"]["tol"])
dm = R["design_map"]
T["TableS12_design_map_XA8h"] = pd.concat([pd.DataFrame(v, index=pd.Index(dm["masses"], name="mass_g"), columns=[f"Q={q}" for q in dm["flows"]]).assign(k_from=k).reset_index()
                                          for k, v in dm["XA8h"].items()])
T["TableS13_k_gamma_identifiability"] = pd.DataFrame([dict(gamma_ref_s=g, group=grp, **v) for g, d in G.items() for grp, v in d.items()])
T["TableS15_peclet_sensitivity"] = pd.DataFrame(SP["peclet"])
T["TableS14_mass_balance"] = pd.DataFrame([dict(max_deviation_A_plus_C=R["mass_balance_max_dev"], NaOH_volume_mL=R["naoh_volume_mL"])])

with pd.ExcelWriter(os.path.join(OUT, "all_tables.xlsx")) as xw:
    for name, df in T.items():
        df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
        df.to_excel(xw, sheet_name=name[:31], index=False)

# ── 3. Collect figures ────────────────────────────────────────────────────────
for src in [os.path.join("results", x) for x in os.listdir("results") if x.endswith(".png")] + \
           ["Fig7_morris.png", "Fig8_design_map.png", "FigS_profile_beta.png"]:
    if os.path.exists(src) and not src.startswith(os.path.join("results", "figures")):
        shutil.copy(src, FIG)
for old in ("Fig09_sensitivity.png", "Fig10_scaleup.png"):   # superseded by Fig7/Fig8 revised
    p = os.path.join(FIG, old)
    if os.path.exists(p): os.remove(p)
