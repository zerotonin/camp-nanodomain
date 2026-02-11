# Model Architecture: cAMP Compartmentalization in Drosophila Kenyon Cell Gamma Lobes

## 1. Biological Context

### 1.1 System Overview

A single Kenyon cell (KC) axon traverses the mushroom body gamma lobe, forming synaptic boutons in five compartments (γ1–γ5). Each compartment receives distinct dopaminergic (DAN) innervation:

| Compartment | DAN Input | Valence |
|-------------|-----------|---------|
| γ1 | Aversive DANs | Punishment (e.g., electric shock) |
| γ2 | Aversive DANs | Punishment |
| γ3 | Mixed DANs | Aversive + Appetitive |
| γ4 | Appetitive DANs | Reward (e.g., sugar) |
| γ5 | Appetitive DANs | Reward |

During aversive conditioning, a neutral odorant (activating the KC) is paired with electric shocks (activating DANs in γ1–γ2). Inside each bouton, adenylyl cyclases (ACs) serve as coincidence detectors: they produce cAMP constitutively, but Ca²⁺ influx from DAN activation greatly enhances cAMP production.

### 1.2 The Dunce Hypothesis

Dunce is the Drosophila PDE4 homolog. Anatomically, dunce localizes to synaptic boutons but is absent from inter-bouton axonal segments. We hypothesize that dunce acts as a **cAMP barrier** — a degradation "fence" that confines cAMP signals to the bouton where they are produced.

**Key experimental finding:** In *dunce⁻* mutants, after repeated paired stimulation (odorant + shock), cAMP signal appears in γ4–γ5 — compartments that were NOT directly activated by aversive DANs. This suggests that without dunce-mediated degradation, cAMP produced in γ1–γ2 diffuses along the axon into distal compartments.

### 1.3 Connecting to the Lohse Framework

The Lohse lab established two critical concepts we must incorporate:

1. **Absorptive action η** (Lohse et al. 2017): The ability of a PDE to create a concentration sink is captured by η = k_cat / (4π·R_PDE·D·K_m). With literature values for free diffusion, η ≈ 10⁻⁴ — far too small. Experimental fitting gives η ≈ 6, requiring restricted diffusion.

2. **Buffered diffusion** (Bock et al. 2020): At physiological concentrations, most cAMP is bound to intracellular binding sites (PKA subunits, Epac, etc.), giving an effective cAMP buffering capacity of ~20 μM. Only a small fraction of total cAMP is free and diffusible. This "catch-and-release" dynamics dramatically reduces effective cAMP mobility, reconciling the discrepancy in η values.

---

## 2. Model Geometry

### 2.1 Spatial Discretization: 1D Compartmental Chain

We model the KC axon as a **1D chain of compartments** alternating between boutons and axonal segments:

```
[Bouton γ1] —— axon —— [Bouton γ2] —— axon —— [Bouton γ3] —— axon —— [Bouton γ4] —— axon —— [Bouton γ5]
```

Each bouton is a well-mixed compartment (justified by the small bouton size ~1–2 μm diameter). Axonal segments between boutons are modeled as thin tubes (~0.2–0.5 μm diameter, ~5–10 μm long).

For the numerical implementation, the axonal segments are further subdivided into small spatial bins (dx ≈ 0.1–0.5 μm) to resolve the diffusion gradient.

### 2.2 Geometric Parameters

| Parameter | Symbol | Value | Source/Justification |
|-----------|--------|-------|---------------------|
| Bouton diameter | d_b | 1.5 μm | Typical Drosophila NMJ/MB bouton |
| Bouton volume | V_b | ~1.77 μm³ | Sphere: 4/3·π·(0.75)³ |
| Axon diameter | d_a | 0.3 μm | Drosophila KC axon |
| Axon cross-section | A_a | 0.071 μm² | π·(0.15)² |
| Inter-bouton distance | L_ab | 5 μm | Approximate, can be varied |
| Number of compartments | N | 5 | γ1 through γ5 |

---

## 3. State Variables

For each spatial element *i* (bouton or axon segment), we track:

| Variable | Symbol | Units | Description |
|----------|--------|-------|-------------|
| Free cAMP | c_free,i | μM | Unbound, diffusible cAMP |
| Bound cAMP | c_bound,i | μM | cAMP bound to buffering sites |
| Total cAMP | c_total,i | μM | c_free + c_bound |
| Calcium | [Ca²⁺]_i | μM | Intracellular Ca²⁺ (drives AC) |
| Buffer occupancy | B_i | μM | Occupied binding sites |

---

## 4. Governing Equations

### 4.1 cAMP Reaction-Diffusion (Core Equation)

In each spatial element, the free cAMP concentration evolves as:

```
∂c_free/∂t = D_free · ∇²c_free  +  J_AC  −  J_PDE  −  J_bind  +  J_unbind
```

Where:
- **D_free · ∇²c_free**: Fickian diffusion of free cAMP (only free cAMP diffuses)
- **J_AC**: cAMP production by adenylyl cyclase
- **J_PDE**: cAMP degradation by dunce (PDE4)
- **J_bind / J_unbind**: Binding/unbinding to intracellular buffering sites

### 4.2 Buffered Diffusion (from Bock et al. 2020)

Following Bock et al., we implement cAMP buffering as a rapid equilibrium binding process:

```
c_free + B_free  ⇌  c_bound
                k_on
                k_off
```

With:
- **k_on**: Association rate (~10⁷ M⁻¹s⁻¹, diffusion-limited)
- **k_off**: Dissociation rate (k_off = k_on · K_D_buffer)
- **K_D_buffer**: Dissociation constant of buffering sites (~1–5 μM, typical for PKA-RIα)
- **B_total**: Total concentration of buffering sites (~20 μM, conservative estimate from Bock et al.)

The buffering equations:

```
∂c_bound/∂t = k_on · c_free · (B_total − c_bound) − k_off · c_bound
∂c_free/∂t  = ... − k_on · c_free · (B_total − c_bound) + k_off · c_bound
```

**Rapid equilibrium approximation:** If binding/unbinding is fast compared to diffusion and production, we can use the quasi-steady-state:

```
c_bound = B_total · c_free / (K_D_buffer + c_free)
```

This gives an **effective diffusion coefficient**:

```
D_eff = D_free / (1 + B_total / (K_D_buffer + c_free)²  ·  K_D_buffer)
```

At low [cAMP] (c_free << K_D_buffer):
```
D_eff ≈ D_free / (1 + B_total / K_D_buffer)
```

For B_total = 20 μM, K_D_buffer = 2 μM → D_eff ≈ D_free / 11, reducing effective diffusion by about an order of magnitude. This is the core mechanism from Bock et al.

### 4.3 cAMP Production: Adenylyl Cyclase

The AC acts as a coincidence detector (odor-evoked depolarization + DAN-evoked Ca²⁺):

```
J_AC(t) = V_basal + V_max_AC · f_Ca([Ca²⁺]) · f_odor(t)
```

Where:
- **V_basal**: Basal cAMP production rate (~0.1–1 μM/s)
- **V_max_AC**: Maximal Ca²⁺-stimulated production rate (~5–20 μM/s)
- **f_Ca([Ca²⁺])**: Calcium activation function (Hill equation):
  ```
  f_Ca = [Ca²⁺]^n_H / (K_Ca^n_H + [Ca²⁺]^n_H)
  ```
  with K_Ca ≈ 0.5 μM, n_H ≈ 2 (cooperative binding of Ca²⁺/calmodulin to AC)
- **f_odor(t)**: Binary or graded function representing KC activity (= 1 during odor presentation)

### 4.4 cAMP Degradation: Dunce (PDE4)

Following Lohse et al. 2017, dunce follows Michaelis-Menten kinetics:

```
J_PDE = V_max_PDE · c_free / (K_m + c_free)
```

Where:
- **V_max_PDE** = [PDE]_total · k_cat
- **k_cat** ≈ 5 s⁻¹ (literature PDE4 value) or ~160 s⁻¹ (Bock et al. in-cell estimate for PDE4A1 monomer)
- **K_m** ≈ 2–4 μM (PDE4 family)

**Critical:** Dunce is present ONLY in boutons, NOT in axonal segments. This is the key spatial feature:

```
V_max_PDE(x) = { V_max_PDE_bouton   if x ∈ bouton
               { 0                   if x ∈ axon segment
```

In *dunce⁻* mutants: V_max_PDE = 0 everywhere.

### 4.5 Calcium Dynamics

Calcium enters boutons upon DAN activation and decays exponentially:

```
∂[Ca²⁺]/∂t = J_Ca_influx(t) − ([Ca²⁺] − [Ca²⁺]_rest) / τ_Ca
```

Where:
- **[Ca²⁺]_rest** ≈ 0.05–0.1 μM (resting intracellular Ca²⁺)
- **τ_Ca** ≈ 0.5–2 s (Ca²⁺ clearance time constant)
- **J_Ca_influx(t)**: Pulse of Ca²⁺ during each shock stimulus, applied only to the relevant compartments:
  - Aversive conditioning: Ca²⁺ pulses in γ1, γ2 (and partially γ3)
  - Appetitive conditioning: Ca²⁺ pulses in γ4, γ5 (and partially γ3)

For a train of N shocks at interval Δt_shock:
```
J_Ca_influx(t) = Σ_k  A_Ca · δ(t − t_k)    for k = 1, ..., N
```
where A_Ca sets the amplitude of each Ca²⁺ transient.

### 4.6 Diffusion Between Compartments (Discrete Coupling)

The flux of free cAMP between adjacent compartments through the connecting axon segment is:

```
J_diff = D_free · A_a · (c_free,i − c_free,j) / L_ij
```

Where:
- **A_a**: Cross-sectional area of the axon
- **L_ij**: Distance between compartment centers
- **D_free**: Free cAMP diffusion coefficient

The narrow axon acts as a **diffusion bottleneck** due to its small cross-section relative to the bouton volume. The effective coupling rate between two boutons is:

```
k_coupling = D_free · A_a / (V_b · L_ab)
```

This geometric restriction contributes to compartmentalization even independent of dunce.

---

## 5. Parameter Table

### 5.1 Diffusion Parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Free cAMP diffusion coefficient | D_free | 130 μm²/s | Bock et al. 2020; Nikolaev et al. 2004 |
| Effective diffusion (buffered) | D_eff | ~10–15 μm²/s | Calculated from buffering |
| cAMP buffer concentration | B_total | 20 μM | Bock et al. 2020 (conservative) |
| Buffer dissociation constant | K_D_buffer | 2 μM | Walker-Gray et al. 2017; PKA-RIα |

### 5.2 Enzymatic Parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| PDE4 (dunce) k_cat | k_cat | 5 s⁻¹ (lit.) / 160 s⁻¹ (in-cell) | Bender & Beavo 2006 / Bock et al. 2020 |
| PDE4 K_m | K_m | 2.4 μM | Bender & Beavo 2006 |
| AC basal rate | V_basal | 0.5 μM/s | Estimated |
| AC maximal rate | V_max_AC | 10 μM/s | Estimated |
| AC Ca²⁺ half-activation | K_Ca | 0.5 μM | Typical Ca²⁺/CaM activation |
| AC Hill coefficient | n_H | 2 | Cooperative CaM binding |

### 5.3 Calcium Parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Resting [Ca²⁺] | [Ca²⁺]_rest | 0.05 μM | Standard |
| Ca²⁺ transient amplitude | A_Ca | 1–5 μM | Estimated for DAN activation |
| Ca²⁺ decay time constant | τ_Ca | 1 s | Typical neuronal Ca²⁺ clearance |

### 5.4 Stimulation Protocol

| Parameter | Value |
|-----------|-------|
| Odor duration | 5 s |
| Shock duration | 1 s (during last second of odor) |
| Number of pairings | 5–12 (typical training protocol) |
| Inter-trial interval | 30–60 s |
| DAN compartments (aversive) | γ1, γ2 |

---

## 6. Model Variants / Conditions to Simulate

### 6.1 Wild-Type (WT)

- Dunce active in all boutons (V_max_PDE > 0 in boutons)
- Dunce absent from axonal segments
- Predict: cAMP confined to γ1–γ2 during aversive training

### 6.2 *dunce⁻* Mutant

- V_max_PDE = 0 everywhere
- Predict: cAMP diffuses from γ1–γ2 into γ3, γ4, γ5 over time
- Accumulation in γ4–γ5 after repeated pairings

### 6.3 PDE Inhibitor (Pharmacological)

- Equivalent to dunce⁻ but potentially partial inhibition
- Can model dose-response by scaling V_max_PDE

### 6.4 Sensitivity Analysis

Critical unknowns to sweep:
- D_free (or D_eff): What effective diffusion is needed to see spreading in dunce⁻?
- k_cat: Literature (5 s⁻¹) vs in-cell (160 s⁻¹) — which matches the data?
- B_total: How much buffering is in Drosophila neurons?
- Axon geometry: How sensitive is compartmentalization to axon diameter?

---

## 7. Implementation Plan

### 7.1 Numerical Method

**Spatial discretization:** Finite difference method on a 1D grid.
- Boutons: single well-mixed compartments (few grid points)
- Axon segments: subdivided into ~10–50 bins each
- Total grid: ~60–260 spatial bins

**Temporal integration:** Explicit Euler with adaptive time step, or scipy `solve_ivp` with a stiff solver (e.g., BDF or Radau) since buffering kinetics can be fast.

**Time step constraint (if explicit):** dt < dx² / (2·D_free) ≈ (0.1)² / (2·130) ≈ 38 μs → use implicit solver.

### 7.2 Code Structure

```
model/
├── geometry.py        # Spatial grid, bouton/axon assignment, cross-sections
├── parameters.py      # All physical parameters, organized by category
├── dynamics.py        # ODE right-hand side: diffusion + reactions
├── stimulation.py     # Calcium/odor stimulus protocols
├── simulation.py      # Time integration, initial conditions
├── visualization.py   # Plotting: kymographs, time courses per compartment
└── run_experiments.py  # WT vs dunce⁻ comparisons, parameter sweeps
```

### 7.3 Key Outputs / Figures to Generate

1. **Kymograph**: cAMP concentration (color) vs. position along axon (x) vs. time (y). Show WT and dunce⁻ side by side.
2. **Time courses per compartment**: [cAMP]_free in γ1–γ5 over the training protocol. WT vs dunce⁻.
3. **Spatial profiles at key time points**: Snapshots of [cAMP] along the axon at end of stimulus, 10 s after, 60 s after.
4. **Parameter sensitivity**: How D_eff, k_cat, B_total affect the degree of cAMP spreading.
5. **Dose-response for PDE activity**: Fraction of WT dunce activity needed to confine cAMP.

### 7.4 Validation Criteria

The model should reproduce:
- **WT**: cAMP signal confined to γ1–γ2 during and after aversive training
- **dunce⁻**: cAMP accumulates in γ4–γ5 after stimulus train (not after single pulse)
- **dunce⁻ kinetics**: Spreading should require multiple pairings (cumulative effect)
- **PDE inhibition**: Should phenocopy dunce⁻ (partial inhibition → partial spreading)

---

## 8. Mathematical Summary

### The Full PDE System

For spatial position x along the KC axon, with geometry-dependent cross-section A(x) and volume elements V(x):

```
∂c_free/∂t = (1/A(x)) · ∂/∂x [D_free · A(x) · ∂c_free/∂x]
             + V_basal + V_max_AC · f_Ca([Ca²⁺](x,t)) · f_odor(t)
             − V_max_PDE(x) · c_free / (K_m + c_free)
             − k_on · c_free · (B_total − c_bound) + k_off · c_bound

∂c_bound/∂t = k_on · c_free · (B_total − c_bound) − k_off · c_bound

∂[Ca²⁺]/∂t = J_Ca(x, t) − ([Ca²⁺] − [Ca²⁺]_rest) / τ_Ca
```

The variable cross-section A(x) captures the bouton-axon-bouton geometry:
- A(x) = A_bouton = π·(d_b/2)² in boutons
- A(x) = A_axon = π·(d_a/2)² in axonal segments

This creates natural diffusion bottlenecks at the bouton-axon junctions.

### Dimensionless Absorptive Action (from Lohse et al. 2017)

For a single dunce molecule in a bouton, the absorptive action is:

```
η = k_cat / (4π · R_PDE · D_eff · K_m)
```

Using the **effective** (buffered) diffusion coefficient D_eff rather than D_free is the key insight from Bock et al. that resolves the Lohse discrepancy. With D_eff ≈ 10 μm²/s instead of D_free ≈ 130 μm²/s, η increases by >10-fold, bringing it closer to the experimentally observed η ≈ 6.

For the bouton-level model, the total PDE activity in a bouton is the sum of all dunce molecules. With N_PDE molecules per bouton:

```
V_max_PDE_bouton = N_PDE · k_cat
```

---

## 9. Simplifications and Assumptions

1. **1D geometry**: We collapse the 3D bouton to a point/well-mixed compartment. Justified by small bouton size (~1.5 μm) relative to diffusion length scales.

2. **No cAMP production in axonal segments**: ACs are localized to boutons (synaptic sites).

3. **No dunce in axonal segments**: Based on the anatomical data showing dunce in boutons only.

4. **Uniform buffering**: B_total is the same in boutons and axon. Could be relaxed if data suggests otherwise.

5. **Ignore cAMP export**: MRP-mediated cAMP export is neglected (likely slow relative to PDE degradation).

6. **Ca²⁺ does not diffuse between compartments**: Ca²⁺ is heavily buffered and local. Each bouton has independent Ca²⁺ dynamics driven by its own DAN input.

7. **No cGMP cross-talk**: We model only the cAMP pathway.

8. **Steady-state for single stimuli, accumulation across trials**: The key phenomenon (spreading in dunce⁻) emerges over multiple training trials due to incomplete cAMP clearance between trials.

---

## 10. Extensions for Future Versions

- **PKA activation**: Add downstream PKA dynamics to predict actual synaptic plasticity
- **Presynaptic vs. postsynaptic**: Distinguish cAMP in KC boutons vs. MBON dendrites
- **3D geometry**: Full 3D bouton model with dunce clustered at bouton periphery (membrane-associated)
- **Stochastic model**: At nanodomain scale, molecule numbers are small → stochastic effects may matter
- **Rutabaga (AC) mutants**: Model the coincidence detection failure
- **Multiple training protocols**: Different ISI, number of pairings, massed vs. spaced training
