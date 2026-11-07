# Tandem-Reactor-Modeling

Transient multiscale model of L-arabinose oxidation over Au/γ-Al₂O₃ extrudates in a recycled packed-bed / stirred-tank reactor system.

M. Hachhach, D. Yu. Murzin, T. Salmi. *Transient multiscale reactor modelling of L-arabinose oxidation in a recycled packed-bed/tank reactor system.* Chemical Engineering Science.

## Files

| File | Content |
|---|---|
| `tandem_model.py` | Reactor model: axially dispersed packed bed, intraparticle diffusion–reaction (slab), recycled stirred tank; method-of-lines discretisation; parameter estimation; confidence intervals; grid convergence; Morris screening; model figures. |
| `analyses.py` | Identifiability (Fisher matrix, profile likelihood), leave-one-condition-out validation, alternative kinetic models, oxygen balance, numerical convergence, transport criteria (Weisz–Prater, Carberry, Ergun), Morris screening with several seeds, design maps. |
| `run_all.py` | Runs the model and the analyses and writes every table (CSV and Excel) and figure. |

The experimental conversion data (six runs, 200 points) are in the `EXPERIMENTS` dictionary at the top of `tandem_model.py` (published in Hachhach et al., Chem. Eng. Sci. 311 (2025) 121580).

## Usage

```bash
pip install -r requirements.txt
python run_all.py
```

Outputs are written to `results/`: `results/tables/*.csv`, `results/all_tables.xlsx` and `results/figures/*.png`. A full run takes about 45 minutes on a laptop; one simulation of an experiment on the production grid (Nz = 10, Nx = 6, 128 active ODEs) takes about 0.3 s.

## Solver

The ODE system is integrated with LSODA through `scipy.integrate.solve_ivp(method="LSODA")`; no separate solver file is needed.

## Licence

MIT
