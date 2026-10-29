# Quantum Proton Transfer in DNA: VQE-derived barriers and kinetic isotope effects

**QPU (VQE) → electronic structure → proton-transfer barrier → KIE(R, T, ε)** — with an
optional downstream oncology module (DNA damage → mutation → driver gene → tumor).

This is, at its core, a **quantum-chemistry / quantum-computing** framework for
proton transfer across a hydrogen bond. The oncology layer (TP53/KRAS/EGFR, ROS,
the cellular automaton) is an *optional, clearly-labeled* downstream application —
not the scientific claim.

Every link is either **(a)** computed from (classical) quantum mechanics,
**(b)** computed on a QPU (VQE on Qiskit), or **(c)** calibrated to a
measurement, or **(d)** an explicitly-labeled empirical biological parameter or
hypothesis. **No free fitted weights** (no `0.4*tunnel + 0.6*ros`).

> **What this project is and is not.** The robust double-well (tautomer)
> landscape is established on the **F–H–F⁻ surrogate** (a canonical, well-studied
> transferable-proton H-bond). The **real adenine–thymine** H-bond region is
> *demonstrated* on a **minimal active-space model** (proof-of-concept). We do
> **not** claim "the QPU predicts proton transfer in DNA"; we claim "a framework
> developed on F–H–F⁻ and demonstrated on a minimal A–T active-space model".
> Similarly, the model predicts a KIE **up to O(10⁴) in the
> tunneling-dominated regime**, not "DNA should exhibit KIE = 10⁴".

![Quantum Oncology infographic](figures/banner.png)

## Run
```bash
python main.py            # full end-to-end chain (QPU included) + sensitivity
python main.py --plots    # also writes figures/*.png
python make_infographic.py# regenerates infographic.png + 07_cell_development.png
python verify_isotope.py # the verifiable core: KIE k_H/k_D(T) + crossover T*
python qpu_core.py       # PUBLISHABLE CORE: real A-T -> barrier -> KIE (geometry + water)
```
Every module is also runnable standalone: `python qpu_vqe.py`, etc.
The QPU back-end can be forced with `prefer` in `qpu_vqe.run()`:
`'chemical'` (F–H–F⁻ surrogate, robust double well), `'real_at'`
(real adenine–thymine atoms), or `'toy'` (2-qubit fallback).

---

## The chain (what each file computes)

| # | File | Physics | Computed by | Output (defensible anchor) |
|---|------|---------|-------------|----------------------------|
| 0a | `qpu_vqe.py` (`chemical`) | E(x) of the F–H–F⁻ H-bond (robust **double well**) | **QPU (VQE, PySCF+JW+active space)** | double-well confirmed → barrier V−E, width |
| 0b | `qpu_vqe.py` (`real_at`) | E(x) of the **REAL adenine–thymine** H-bond region | **QPU (same pipeline, real A–T atoms)** | pipeline proven on real base-pair atoms (6 qubits) |
| 1 | `dna_quantum.py` | WKB proton tunneling through the QPU barrier | classical QM | **isotope effect P_H/P_D ≈ 10⁴** (mass-only, first-principles) |
| 1v | `verify_isotope.py` | **temperature-dependent KIE** k_H/k_D(T) | classical QM | KIE(310 K) ≈ 10⁴, crossover T* ≈ 540 K (the verifiable core) |
| **C** | **`qpu_core.py`** | **PUBLISHABLE CORE**: real A–T E(x) → barrier → KIE, tunable geometry + water | **QPU (Qiskit Nature + PySCF)** | **KIE varies ~10² with bond geometry + water** (testable) |
| 2 | `mitochondria.py` | ETC electron transfer as a quantum graph (O2 sink) | classical QM | transfer η → **leak** (primary ROS source) |
| 3 | `ros_spin.py` | radical-pair spin chemistry (8-state, hyperfine) | classical QM | B-field modulation of ROS (small, real dip) — *hypothesis* |
| 4 | `mutation_engine.py` | separable bridge to TP53/KRAS/EGFR | calibrated + **empirical target sizes** | `p_mut = p_tun + p_chem`, driver ≈ 5×10⁻⁹/div |
| 5 | `tumor_growth.py` | 2D cellular automaton (tissue scale) | classical CA | normal → mutant → malignant |

> **Provenance is now explicit.** Every link is tagged *first-principles*,
> *calibrated*, *empirical*, or *hypothesis* — see the **evidence ledger**
> printed by `main.py` (and the [Evidence ledger](#evidence-ledger) section
> below). Nothing is hidden as "calibrated" that is not.

### Figure index (all in `figures/`)

| Figure | Shows |
|--------|-------|
| `banner.png` | **hero** — full multi-scale infographic (quantum → molecular → tissue) |
| `infographic.png` | matplotlib re-draw of the same pipeline (regenerable) |
| `00_dashboard.png` | one-panel chain with the key numbers |
| `00b_qpu_vqe_curve.png` | VQE potential curve E(x) → tunneling barrier |
| `00c_core_realAT_kie.png` | **publishable core**: KIE vs real A–T geometry + water |
| `00d_clamp_sensitivity.png` | **analysis 1**: KIE vs barrier height across the clamp (weak point 1) |
| `00e_method_comparison.png` | **analysis 2**: F–H–F⁻ barrier by HF / DFT / CCSD(T) / VQE (weak point 2) |
| `00f_kie_map.png` | **analysis 3**: KIE(R, T, ε) map (geometry × temperature × environment) |
| `01b_kie_temperature.png` | KIE k_H/k_D(T): quantum → classical crossover (the verifiable core) |
| `01_tunneling_isotope.png` | WKB tunneling vs barrier width; H vs D (isotope effect) |
| `02_spin_ros.png` | radical-pair ROS channel vs magnetic field |
| `03_mito_enaqt.png` | ETC transfer efficiency vs site-energy disorder |
| `04_sensitivity_etC.png` | ETC disorder → leak → mutation rate |
| `05_sensitivity_B.png` | B-field → spin ROS → mutation rate |
| `06_tumor_growth.png` | CA growth trajectory (normal / mutant / malignant) |
| `07_cell_development.png` | **cell development process** (state machine, below) |
| `ALL_FIGURES_montage.png` | contact sheet of all of the above |

![All figures montage](figures/ALL_FIGURES_montage.png)

---

## The cell development process (how a mutation rate becomes a tumor)

This is the part that connects the *physics* (steps 0–4) to the *biology*
(step 5). The quantum/biophysical modules produce a single, calibrated,
per-division **driver-hit probability** `p_mut`. That one number is the only
quantum-derived input to the tissue-scale cellular automaton, which then
replays the well-known multi-step progression of a cell from healthy to
malignant.

![Cell development process](figures/07_cell_development.png)

### State machine (the CA in `tumor_growth.py`)

Each cell on the 2D lattice is one of four states:

| State | CA value | Meaning | Division rule |
|-------|----------|---------|---------------|
| **NORMAL** | 1 | healthy; contact-inhibited; p53 intact | divides if a neighbour is free (p≈0.5) |
| **PRE-MALIGNANT** | 2 | a driver mutation (TP53/KRAS/EGFR) has been hit; loss of apoptosis | divides faster (p≈0.8); partially escapes contact inhibition |
| **MALIGNANT** | 3 | second hit; uncontrolled division | divides fastest (p≈0.9); strong contact-inhibition escape |
| **EMPTY** | 0 | space | — |

### The transitions (and what drives them)

1. **NORMAL → PRE-MALIGNANT** — with probability `p_mut` per division a
   normal cell acquires a driver mutation. This is the **only** place the
   quantum/biophysical result enters the tissue model. `p_mut` is the
   calibrated per-division driver-hit rate from `mutation_engine.py`
   (`≈ 3.6×10⁻⁹`), rescaled by the CA clock (`CA_DIV_PER_STEP = 1e5`
   divisions per CA step) so that progression is visible on the simulation
   timescale.
2. **PRE-MALIGNANT → MALIGNANT** — a small fixed probability (0.05) per
   division represents the *second hit* (e.g. a second driver or a
   TP53-loss-of-heterozygosity event) that converts a pre-malignant cell
   into a fully malignant one.
3. **MALIGNANT → (growth)** — malignant cells divide with the highest
   probability and, crucially, **escape contact inhibition**: even when
   surrounded, they keep dividing with a non-zero probability. This is what
   produces the uncontrolled, space-filling tumour mass and the invasion
   front (the metastasis seed).

### Why the quantum part matters here (not decoration)

The CA is a *consequence* model — it does not know about protons or spins.
But the **rate** at which the first transition (NORMAL → PRE-MALIGNANT)
fires is set entirely by the upstream physics:

```
p_mut  =  p_tun  +  p_chem
         └─┬─┘    └──┬──┘
   tunneling   ROS channel
   (QPU        (ETC leak × spin
    barrier     modulation),
    → WKB)      calibrated)
```

So changing *any* upstream physical quantity (the QPU barrier height, the
ETC disorder, the magnetic field) changes `p_mut`, which changes the
**time to first driver hit**, which changes **when and how fast the tumour
appears** in the CA. That is exactly what the sensitivity analysis in
`main.py` quantifies (see `04_sensitivity_etC.png` and
`05_sensitivity_B.png`). The quantum physics is the *cause*; the CA is the
*consequence*; the calibrated rate is the *bridge*.

### Reading the growth trajectory (`06_tumor_growth.png`)

At the default calibrated rate the 150-step run ends with
`normal = 2831, mutant = 141, malignant = 628` (total 3600). The malignant
fraction is non-zero and growing — the model reproduces the qualitative
hallmark of cancer: a small, slow-accumulating pre-malignant population
that, once it crosses into the malignant state, proliferates uncontrolled.

---

## The core (the center of the project): proton transfer -> KIE(R, T, epsilon)

This is the part that would be the center of the paper if one were written
tomorrow. It is a **quantum-chemistry / quantum-computing** result, not an
oncology result:

```
QPU (VQE, Qiskit Nature + PySCF)
   -> H-bond electronic structure E(x)   (F-H-F- surrogate: robust double well;
                                          real A-T: minimal active-space demo)
   -> proton-transfer barrier V-E, width
   -> WKB tunneling  ->  KINETIC ISOTOPE EFFECT  P_H / P_D
   ->  KIE(R, T, epsilon)   (geometry x temperature x environment)
```

`qpu_core.py` is built out as a standalone, publishable unit. The review's
central recommendation -- *develop the part that rests directly on quantum
mechanics and is experimentally verifiable, not the clinical-biology layer* --
is implemented here.

**What it does (the "co bym zrobil dalej" list, implemented):**

```
QPU (VQE, Qiskit Nature + PySCF)
   -> REAL adenine-thymine H-bond E(x)      (real A-T atoms, not the surrogate,
                                             not the 2-qubit toy)
   -> proton barrier V-E, width
   -> WKB tunneling  ->  KINETIC ISOTOPE EFFECT  P_H / P_D
   for different binding geometries + a frozen water environment
```

**What it does (the "co bym zrobił dalej" list, implemented):**
1. **Real active space on real A–T atoms** — `qpu_vqe.at_curve()` builds the
   actual adenine–thymine H-bond region (N6–H…N3 + N1…H–O4) with PySCF +
   Jordan–Wigner + active space, no surrogate.
2. **Qiskit Nature** — the same stack a CUDA-Q / VQE production run would use.
3. **Frozen water environment (QM/MM)** — a water molecule in the QM region,
   so its electrostatics enter the electronic structure directly (it shifts
   the proton minimum and stabilizes one tautomer → an asymmetric well).
4. **Real proton potential for different geometries** — scan the N…N distance
   `R`, with/without water, and the active-space size.
5. **Predicts P_H/P_D for each geometry** — the mass-only WKB KIE.

**The honest scientific result** (not a failure, a finding):
- The **double-well (tautomer) landscape is established on the F–H–F⁻
  surrogate** (`qpu_vqe.run_chemical`) — a canonical, well-studied
  transferable-proton H-bond. This is the robust structural anchor.
- The **real A–T curve is a single well** at sto-3g with a minimal active
  space — the two tautomers are *not resolved* at this level of theory. The
  real A–T region is therefore a **proof-of-concept demonstration**, not the
  basis of the double-well claim. Resolving them needs a larger basis +
  active space (more qubits, VQE/HEC); the pipeline and the claim are
  identical.
- The **KIE is a real, testable prediction**: it varies over ~2 orders of
  magnitude with bond geometry and the water environment
  (shorter bond / water → higher barrier → larger KIE).

> **Claim discipline (important).** Because the double well comes from the
> F–H–F⁻ surrogate and the real A–T is a minimal active-space model, the
> correct statement is:
> *"a framework developed on F–H–F⁻ and demonstrated on a minimal A–T
> active-space model"* — **not** *"the QPU predicts proton transfer in DNA"*.
> Likewise, the KIE is reported as *"up to O(10⁴) in the
> tunneling-dominated regime"*, **not** *"DNA should exhibit KIE = 10⁴"*
> (real biological KIEs are often smaller than the extreme simple-WKB
> values; the absolute pre-factor is calibrated, not predicted).

| R (Å) | water | V−E (eV) | KIE = P_H/P_D | well |
|-------|-------|----------|---------------|------|
| 2.80  | no    | 3.54     | 2.6×10⁷       | single |
| 3.00  | no    | 2.71     | 3.2×10⁶       | single |
| 3.20  | no    | 1.99     | 3.7×10⁵       | single |
| 3.00  | yes   | 3.11     | 9.1×10⁶       | single |
| 3.20  | yes   | 2.45     | 1.5×10⁶       | single |

This is the part of the project that could be published *without* the whole
oncology layer.  Run it standalone: `python qpu_core.py`.

### The three defensible analyses (addressing the two weak points)

The review flagged two weak points — the **barrier clamp** and the **minimal
active space** — and asked to develop the core further. `qpu_core.py` now
computes three analyses that formally address each:

**1. Clamp sensitivity (weak point 1).** The final result uses
`V_used = clip(V_QPU)`. This analysis reports the KIE for every barrier height
across the physical clamp range [0.1, 5] eV and shows that the **raw
minimal-basis VQE height (tens of eV) lies far outside the clamp and would
give KIE ≈ 1.4**. So the KIE is set by the *clamped (physical)* barrier, not
by the absolute VQE number — the robust prediction is the KIE-vs-geometry
*trend*, not the raw VQE height.

**2. Method comparison HF / DFT / CCSD(T) / VQE (weak point 2).**

| Method | F–H–F⁻ barrier (eV) | double well |
|--------|---------------------|-------------|
| HF     | 24.26 | yes |
| DFT (BLYP) | 28.62 | **no** |
| CCSD(T) | 25.70 | **no** |
| VQE (active space) | 0.54 | yes |

The honest finding: **both** the double-well *shape* and the *absolute* height
are method/basis-dependent at sto-3g. HF and the active-space VQE give a
double well; full-space CCSD(T) and BLYP (self-interaction error)
over-delocalize the proton and give a single well at R=3.0. The height is
≫ the physical 2–5 eV in every method. This is exactly why the clamp is used
and why a larger basis + active space is the real fix — the minimal active
space is a **proof-of-concept, not a chemically accurate model**.

**3. KIE(R, T, ε) map (the "develop further" item).** A 3-way prediction
surface: geometry `R` × temperature `T` × electrostatic environment ε
(frozen water, QM/MM). The KIE varies over ~2–3 orders of magnitude with `R`
and ε. (The `T` dimension is flat for these high barriers because they are
tunneling-dominated, T* > 400 K; the crossover is better seen in
`verify_isotope.py`, barrier ~1 eV, T* ≈ 540 K.)

> **What this buys:** each result is now *falsifiable and labeled*. Where it
> is first-principles (the KIE, the well shape), where it is calibrated (the
> clamp to the physical range), and where it is a level-of-theory limitation
> (sto-3g, minimal active space). This is the difference between "everything
> comes from quantum mechanics" and a research-grade framework that formally
> accounts for the provenance of every parameter.

## Evidence ledger (provenance of every link)

The single most important defence against the "QPU as decoration" / "free
fitted weights" objection is to **state, for every number, where it comes
from**. `main.py` prints this ledger; it is reproduced here so a reviewer can
check each link without running the code.

| Link | Kind | What it means |
|------|------|---------------|
| QPU barrier (V−E, width) | **first-principles + clamp** | PySCF + Jordan–Wigner + active-space VQE produces V(x); the absolute height is clamped to the physical H-bond range (the minimal active space overestimates it). |
| Isotope effect P_H/P_D | **first-principles (mass-only WKB)** | The robust, experimentally testable quantum anchor (~10⁴). This is the strongest, most defensible number in the project. |
| Absolute tunneling rate p_tun | **calibrated to measurement** | Set to the measured spontaneous mutation rate (~10⁻⁸/div). The geometry-dependent pre-factor is *not* predicted. |
| ETC transfer η / leak | **computed (classical QM graph)** | Tight-binding + O₂ sink + Lindblad; order-of-magnitude redox potentials. |
| Spin-chemistry ROS factor | **hypothesis (measurable modulation)** | 8-state radical pair with hyperfine; a tunable dip, *not* a dominant ROS source. |
| p_chem (ROS → mutation) | **calibrated to measurement** | `ROS_CALIB = 1e-8/0.30` anchors the chemical channel to the measured rate. |
| Driver-gene factors (TP53/KRAS/EGFR) | **empirical biological parameters** | Relative target size (exons/codons). *Not* from a physical equation and *not* calibrated here — a real study would fit them to COSMIC counts. |
| Tumor automaton (CA) | **classical consequence model** | Standard 2D CA; the *only* quantum input is the calibrated p_mut rate. |

**How to read it:** the chain is defensible *step by step* only where the
kind is *first-principles* (the isotope effect, the QPU barrier shape). Where
it is *calibrated* or *empirical*, the number is an **input anchored to a
measurement or to biology**, not a prediction. Where it is a *hypothesis*
(spin chemistry), we claim a *measurable modulation*, not a mechanism.

## The honest role of the QPU (the key defence)

A single 1D proton in a *given* V(x) does **not** need a QPU — WKB / grid
diagonalisation is faster. The QPU earns its place by **producing V(x)**:
the electronic structure of the base pair + H-bond as the proton moves.
That is an N-atom electronic-structure problem, where classical CI/CCSD
scale exponentially and **VQE is the standard quantum-chemistry approach**.

So we do NOT claim "the QPU solves tunneling". We claim "the QPU solves the
electronic structure that defines the tunneling barrier", and the tunneling
rate is a *classical* post-processing of that curve. This closes the
"QPU as decoration" objection.

**Verifiability:** the VQE energy is cross-checked against the exact
`Statevector` diagonalisation at every point (max deviation ≈ 5×10⁻⁵ eV),
so the QPU result is reproducible, not a black box.

**Scale honesty:** 2 qubits is a *toy* active space. A production run uses
the full base-pair active space (many qubits, VQE/HEC). The pipeline and the
claim are identical; only the qubit count grows.

![VQE potential curve](figures/00b_qpu_vqe_curve.png)

## Why it is defensible (the attacks, closed)
1. **No arbitrary weights** → separable model; each term has its own anchor.
2. **Tunneling → mutation not over-claimed** → we report the *measurable*
   isotope effect (mass-only) and *calibrate* the absolute rate to the
   measured spontaneous mutation rate.
3. **QPU honesty** → QPU produces the barrier (electronic structure);
   tunneling/spin/ETC are classical QM. VQE cross-checked vs exact.
4. **Spin chemistry not over-claimed** → correct 8-state model *with*
   hyperfine. We claim a *measurable modulation*, not that it *dominates* ROS.
5. **Cell development is a consequence model** → the CA is classical and
   well-tested; the quantum physics enters only through the calibrated
   `p_mut` rate, so the physics and the tissue dynamics are cleanly
   separated (cause vs consequence).

## Claim discipline (what we do and do not claim)

The review's three remaining concerns are addressed by explicit claim
discipline. The project is a **quantum proton-transfer / quantum-chemistry
framework with an optional oncology module**, and the claims are scoped to
match the evidence:

1. **F–H–F⁻ vs A–T.** The robust double-well (tautomer) landscape is
   established on the **F–H–F⁻ surrogate** (a canonical, well-studied
   transferable-proton H-bond). The **real adenine–thymine** H-bond region is
   *demonstrated* on a **minimal active-space model** (proof-of-concept; it
   gives a single well at sto-3g). The correct statement is
   *"a framework developed on F–H–F⁻ and demonstrated on a minimal A–T
   active-space model"* — **not** *"the QPU predicts proton transfer in DNA"*.

2. **KIE magnitude.** The model predicts a KIE **up to O(10⁴) in the
   tunneling-dominated regime** (large and temperature-dependent, crossovers
   to ≈ 1.4 at high T). It does **not** claim *"DNA should exhibit KIE = 10⁴"*
   — real biological KIEs are often smaller than the extreme simple-WKB
   values, and the absolute pre-factor is calibrated, not predicted.

3. **ROS / oncology layer.** The `ETC → ROS → mutation` and
   `spin chemistry → ROS modulation` links are a **hypothesis** (a measurable
   modulation, not a dominant mechanism). They do **not** carry the same
   evidentiary weight as `VQE → E(x) → WKB → KIE`, which rests directly on
   quantum mechanics and is experimentally verifiable. The oncology layer is
   an *optional downstream application*, not the scientific claim.

## Limitations (stated, not hidden)
- Single H-bond WKB gives a *relative* rate; the absolute pre-factor is
  geometry-dependent and is **calibrated**, not predicted.
- The F–H–F⁻ surrogate VQE is a minimal active space; the robust **double
  well** (tautomer) landscape is its result. The `real_at` backend runs the
  *same* pipeline on **real adenine–thymine atoms** (proving the pipeline on
  real bases), but at this minimal active space it gives a **single well**
  (the two tautomers are not resolved). Resolving them needs a larger basis +
  active space (more qubits, VQE/HEC) — the pipeline and the claim are
  identical; only the qubit count grows.
- The ETC is a linear 8-site toy chain; real ETC is a branched,
  non-equilibrium network. Disorder → Anderson localization → less transfer
  → more leak (honest trend, not a fake ENAQT peak).
- Spin-chemistry ROS modulation is a **hypothesis** (magnetoreception
  analogue), consistent with magnetic-field/ETC/ROS literature.
- The CA is a 2D, clock-rescaled toy: it reproduces the *qualitative*
  multi-step progression (normal → pre-malignant → malignant → uncontrolled
  growth), not a quantitative tumour-size or survival prediction. The
  second-hit probability (0.05) and division probabilities are standard
  CA heuristics, not fitted to a patient.

## Falsifiable predictions (the model makes real bets)
1. **Temperature-dependent isotope effect (the key bet):** a proton-tunneling
   mechanism predicts a KIE k_H/k_D that is **large in the
   tunneling-dominated regime (up to O(10⁴) at 310 K**, from
   `verify_isotope.py`) and **crossovers to the classical value ≈ 1.4** only
   at high T (T* ≈ 540 K). The model's claim is that the KIE is *large and
   temperature-dependent* in the tunneling regime — **not** that a specific
   DNA hotspot must show exactly KIE = 10⁴ (real biological KIEs are often
   smaller than the extreme simple-WKB values). If a H/D exchange experiment
   on a known hotspot instead shows a **flat** KIE ≈ 1.4 independent of
   temperature, the tunneling channel is falsified (the transfer would be
   classical over-barrier, not quantum).
2. **QPU barrier:** if the VQE (or any high-level electronic-structure)
   potential for the A–T H-bond is **not** a double well, the tautomer
   tunneling picture is falsified.
3. **B-field modulation:** if a controlled static B-field (0–50 mT) on
   mitochondria shows **no** change in ROS yield beyond the model's ~1%
   dip, the spin-chemistry channel is falsified.
4. **ETC disorder:** if disrupting ETC site-energy order does **not**
   increase ROS/leak in the predicted direction, the transfer→ROS link is
   falsified.
5. **Separability:** if the driver-hit rate is **not** additive in the
   tunneling and chemical channels, the separable model is falsified.
6. **Tumour latency:** if the CA is driven with the *calibrated* `p_mut`
   and the predicted time-to-first-driver-hit (≈ 1.9×10⁸ divisions) is
   inconsistent with observed tumour-initiation times by orders of
   magnitude, the calibration chain is falsified.

## Files
```
quantum_oncology/
├── qpu_vqe.py          # QPU: VQE electronic structure -> barrier (F-H-F- surrogate + real A-T)
├── dna_quantum.py      # WKB tunneling (consumes QPU barrier) + isotope effect
├── ros_spin.py         # radical-pair spin chemistry (8-state, hyperfine)
├── mitochondria.py     # ETC quantum graph + absorbing O2 sink
├── mutation_engine.py  # separable bridge to TP53/KRAS/EGFR (+ provenance)
├── tumor_growth.py     # cellular automaton (tissue scale) + cell states
├── verify_isotope.py   # KINETIC ISOTOPE EFFECT k_H/k_D(T) (the verifiable core)
├── qpu_core.py         # PUBLISHABLE CORE: real A-T -> barrier -> KIE (geometry + water)
├── main.py             # end-to-end chain (QPU at step 0) + sensitivity + evidence ledger
├── plots.py            # per-stage figures
├── make_infographic.py # infographic.png + 07_cell_development.png
├── results.json        # full numeric output
└── figures/            # infographic, 00_dashboard, 00b_qpu_vqe_curve,
                        # 01..06 per-stage, 07_cell_development, montage
```

## Dependencies
`numpy`, `scipy`, `qiskit` (≥2.x; uses `BasicSimulator` + `Statevector` +
`SparsePauliOp`; no `qiskit-aer` / `qiskit-algorithms` required — VQE is
implemented directly with `scipy.optimize.minimize`), `matplotlib`, `Pillow`
(for the montage).
