# PMPG-NN-reconstruction
Physics-informed (PMPG-based) neural network model for denoising and reconstructing axisymmetric flow over a sphere in spherical coordinates.

This repository contains a **research prototype** (Jupyter notebook + utilities) for a simple, axisymmetric test case:
**inviscid potential flow over a sphere** in θ–r coordinates, with **synthetic noise added** to the velocity field and a
**neural streamfunction** trained to denoise/recover the field while enforcing physics/constraints.

> Main notebook: `Flow_over_sphere.ipynb`  
> Helper utilities: `functions.py`

---

## What’s inside

- **Grid generation** in spherical coordinates (axisymmetric slice).
- **Analytic reference flow** over a sphere (streamfunction-based potential flow).
- **Synthetic noise injection** into θ/r velocity components.
- **Neural streamfunction model** ψ(r,θ) trained with a weighted loss:
  - Data fit to noisy velocities
  - No-penetration at the sphere surface (r = 1)
  - Far-field uniform flow condition (r = R_∞)
  - Optional physics penalties (e.g., curl of convective acceleration)
  - A PMPG/Appellian-style integral term (see notes below)

---

## Quick start

### 1) Create an environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
