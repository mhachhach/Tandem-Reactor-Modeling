# Tandem-Reactor-Modeling

Transient multiscale model of L-arabinose oxidation over Au/γ-Al₂O₃ extrudates in a recycled packed-bed / stirred-tank reactor system.

M. Hachhach, D. Yu. Murzin, T. Salmi. *Transient multiscale reactor modelling of L-arabinose oxidation in a recycled packed-bed/tank reactor system.* Chemical Engineering Science.

## Files

| File | Content |
|---|---|
| `tandem_model.py` | Reactor model: axially dispersed packed bed, intraparticle diffusion–reaction (slab), recycled stirred tank; method-of-lines discretisation; parameter estimation; confidence intervals; grid convergence; Morris screening; model figures. |
| `analyses.py` | Identifiability (Fisher matrix, profile likelihood), leave-one-condition-out validation, alternative kinetic models (deactivation, free LHHW, product inhibition), one shared rate constant versus one per loading (with a nested-model F-test and AIC), sensitivity to the bed Péclet number, oxygen balance, numerical convergence, transport criteria (Weisz–Prater, Carberry, Ergun), Morris screening with several seeds, design maps. |
| `run_all.py` | Runs the model and the analyses and writes every table (CSV and Excel) and figure. |

The conversion data (six runs, 182 points) are in the `EXPERIMENTS` dictionary at the top of `tandem_model.py`. E1 and E2 (fresh 2 wt% Au, 0.897 g) were reported in Hachhach et al., Chem. Process Eng. New Front. 46 (2025) e85; E3, E4 and E6 (spent 3 wt% Au, 1.6225 g) in Hachhach et al., Chem. Eng. Sci. 311 (2025) 121580. All values are conversions computed from the HPLC arabinose peak area, X = 1 − A/A0. E5 and the long-time points of E1, E3, E4 and E6 are unpublished records of the same experimental campaign. The loop holds 150 mL of liquid (`V_LIQ` in `tandem_model.py`).

## Usage

```bash
pip install -r requirements.txt
python run_all.py
```

Outputs are written to `results/`: `results/tables/*.csv`, `results/all_tables.xlsx` and `results/figures/*.png`. A full run takes about one hour on a laptop; one simulation of an experiment on the production grid (Nz = 10, Nx = 6, 128 active ODEs) takes about 0.3 s.

## Solver

The ODE system is integrated with LSODA through `scipy.integrate.solve_ivp(method="LSODA")`; no separate solver file is needed.

## Licence

MIT
