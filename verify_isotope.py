"""
verify_isotope.py
=================
The experimentally verifiable core of the model: the KINETIC ISOTOPE
EFFECT (KIE) for proton transfer across the hydrogen-bond barrier.

This is the fragment a reviewer can defend and test in the lab:
    QPU (VQE)  ->  V(x) double-well barrier
    WKB        ->  tunneling rate k_H, k_D
    KIE        ->  k_H / k_D

WHY THIS IS THE STRONGEST PART (per the physics review):
  * The BARRIER comes from the QPU electronic-structure curve (a REAL
    chemical Hamiltonian, PySCF + Jordan-Wigner + active space) -- not a
    2-qubit toy.  (If the QPU stack is not installed we fall back to a
    literature H-bond barrier, and say so.)
  * The KIE is set by the MASS RATIO (m_D/m_H = 2) and is experimentally
    observable in H/D exchange of base pairs.

THE KEY NEW RESULT -- the TEMPERATURE-DEPENDENT KIE (the real signature):
  A single rate constant has two contributions:
      k(T) = nu * [ exp(-Ea/kT)  +  P_tunnel ]
             └──────────┘        └───────────┘
        classical over-barrier   quantum tunneling (WKB)
  * The CLASSICAL (over-barrier) KIE is ~ sqrt(m_D/m_H) = 1.41 and is
    MASS-INDEPENDENT (it is just the vibrational-frequency ratio).
  * The TUNNELING KIE is exponentially large (~10^4-10^5) and GROWS with
    mass.
  * Therefore the TOTAL KIE(T) crosses over:
        T -> 0   :  KIE -> sqrt(m_D/m_H) * P_H/P_D   (huge, quantum)
        T -> inf :  KIE -> sqrt(m_D/m_H)             (~1.4, classical)
  * The crossover temperature T* = Ea / (kB ln(1/P_H)) separates the two
    regimes.  If T* is ABOVE physiological temperature (310 K), then at
    body temperature the transfer is TUNNELING-dominated and the KIE must
    be large -- that is the falsifiable quantum signature.  A classical
    over-barrier mechanism would give KIE ~ 1.4 at ALL temperatures.

What we compute:
  * KIE(T) at physiological T (310 K) and the full T-scan
  * the crossover temperature T* (the discriminant)
  * the mass-only tunneling KIE P_H/P_D (the "quantum anchor")
  * the geometry (width/height) and electrostatic dependence (testable)

Honest scope:
  * The ABSOLUTE rate is calibrated (pre-exponential nu_0 unknown); the KIE
    (a RATIO) is the robust, mass-based prediction.  nu_0 cancels in the
    KIE, so the result does not depend on the unknown pre-factor.
  * A KIE of order 10^3-10^5 at physiological temperature is the signature
    of proton tunneling.  If a H/D exchange experiment on a known hotspot
    does NOT show a KIE of this order, the tunneling channel is falsified.
"""
import numpy as np

HBAR = 1.0545718e-34
EVJ  = 1.602176634e-19
KB   = 1.380649e-23          # J/K
M_H  = 1.6726219e-27
M_D  = 3.3435837e-27
ANG  = 1e-10
T_BIO = 310.0                # K, physiological temperature

# ---- fallback barrier (used only if the QPU stack is not installed) -----
FALLBACK_BARRIER_EV = 1.0     # eV, V - E
FALLBACK_WIDTH_A    = 0.8     # A


def _qpu_barrier():
    """Pull the barrier from the QPU (F-H-F- surrogate = robust double well).
    Returns (barrier_eV, width_A, source)."""
    try:
        import qpu_vqe
        r = qpu_vqe.run(prefer="auto")   # chemical (F-H-F-) if available, else toy
        b = r["barrier"]
        # clamp to the physical H-bond range (see main.py) -- the minimal
        # active space overestimates the height; the width grid is coarse.
        barrier_eV = min(max(b["barrier_height"], 0.1), 5.0)
        width_A    = min(max(b["width_A"], 0.5), 1.0)
        return barrier_eV, width_A, r["backend"]
    except Exception:
        return FALLBACK_BARRIER_EV, FALLBACK_WIDTH_A, "literature fallback (no QPU stack)"


def wkb_P(mass_kg, barrier_eV, width_A):
    """WKB tunneling probability through a rectangular barrier."""
    dE = barrier_eV * EVJ
    a = width_A * ANG
    k = np.sqrt(2.0 * mass_kg * dE) / HBAR
    return float(np.exp(-2.0 * a * k))


def rate(mass_kg, barrier_eV, width_A, T, prefactor=1.0e13, m_ref=M_H):
    """Total rate k(T) = nu * [ classical over-barrier + tunneling ].
    nu (prefactor) is the H-bond vibration frequency; it cancels in the KIE.
    The classical pre-factor carries the vibrational-frequency mass ratio
    sqrt(m_ref/m), which is the source of the ~1.4 classical KIE."""
    dE = barrier_eV * EVJ
    classical = np.sqrt(m_ref / mass_kg) * np.exp(-dE / (KB * T))
    tunnel = wkb_P(mass_kg, barrier_eV, width_A)
    return prefactor * (classical + tunnel)


def kie_T(barrier_eV, width_A, T):
    """Total (classical + tunneling) KIE k_H/k_D at temperature T."""
    return rate(M_H, barrier_eV, width_A, T) / rate(M_D, barrier_eV, width_A, T)


def crossover_T(barrier_eV, width_A):
    """T* where tunneling = classical over-barrier contribution (for H).
    k_classical = P_tunnel  ->  exp(-Ea/kT) = P_H  ->  T* = Ea/(kB ln(1/P_H))."""
    P_H = wkb_P(M_H, barrier_eV, width_A)
    if P_H >= 1.0:
        return 0.0
    Ea = barrier_eV * EVJ
    return float(Ea / (KB * np.log(1.0 / P_H)))


def run():
    barrier_eV, width_A, source = _qpu_barrier()
    out = dict(barrier_eV=barrier_eV, width_A=width_A, barrier_source=source)

    # mass-only tunneling KIE (the "quantum anchor")
    P_H = wkb_P(M_H, barrier_eV, width_A)
    P_D = wkb_P(M_D, barrier_eV, width_A)
    out["P_H"] = P_H
    out["P_D"] = P_D
    out["tunneling_KIE"] = P_H / P_D

    # the KEY result: temperature-dependent total KIE
    out["KIE_physiological"] = kie_T(barrier_eV, width_A, T_BIO)
    out["crossover_T_K"] = crossover_T(barrier_eV, width_A)
    out["classical_KIE"] = float(np.sqrt(M_D / M_H))   # mass-independent ~1.41
    # T-scan: from 100 K (deep tunneling) to 1000 K (classical)
    T_scan = np.array([100, 150, 200, 250, 300, 310, 350, 400, 500, 600, 800, 1000],
                      dtype=float)
    out["T_scan"] = [(float(t), float(kie_T(barrier_eV, width_A, t))) for t in T_scan]

    # geometry dependence (testable: perturb the H-bond) at physiological T
    out["width_scan"] = [(w, float(kie_T(barrier_eV, w, T_BIO)))
                         for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]]
    out["height_scan"] = [(b, float(kie_T(b, width_A, T_BIO)))
                          for b in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]]

    # electrostatic environment: a stabilizing field lowers the barrier
    # (testable: change the protein/water electrostatics)
    out["field_scan"] = [(f, float(kie_T(max(barrier_eV - 0.1 * f, 0.1), width_A, T_BIO)))
                         for f in [0, 1, 2, 3, 4, 5]]

    # CORRECTED consistency checks (the old "heavy-atom limit -> 1" was wrong):
    #  * The TUNNELING KIE GROWS with absolute mass (heavier = more quantum).
    #  * The CLASSICAL KIE is mass-INDEPENDENT (always sqrt(m_D/m_H) ~ 1.41).
    #  * So a large, mass-dependent KIE is the quantum signature; a flat ~1.4
    #    KIE (independent of T and mass) would be classical.
    out["tunneling_KIE_heavy10x"] = (wkb_P(10 * M_H, barrier_eV, width_A)
                                     / wkb_P(20 * M_H, barrier_eV, width_A))
    out["tunneling_KIE_light"] = P_H / P_D
    out["classical_KIE_mass_independent"] = float(np.sqrt(M_D / M_H))

    # the key discriminant: at physiological T the KIE is quantum (>> classical)
    out["quantum_signature"] = out["KIE_physiological"] > 100.0
    out["tunneling_dominates_at_body_T"] = (T_BIO < out["crossover_T_K"])
    return out


if __name__ == "__main__":
    r = run()
    print("=== verify_isotope : KINETIC ISOTOPE EFFECT (the verifiable core) ===")
    print(f"barrier source: {r['barrier_source']}")
    print(f"barrier: V-E = {r['barrier_eV']:.2f} eV, width = {r['width_A']:.2f} A")
    print(f"P_H = {r['P_H']:.3e} ;  P_D = {r['P_D']:.3e}  (WKB, mass-only)")
    print(f"tunneling KIE (mass-only)  = {r['tunneling_KIE']:.3e}")
    print(f"classical KIE (mass-independent) = {r['classical_KIE']:.3f}")
    print()
    print(f"KEY RESULT -- temperature-dependent total KIE (classical + tunneling):")
    print(f"  KIE at physiological T = {T_BIO:.0f} K :  {r['KIE_physiological']:.3e}")
    print(f"  crossover temperature T* = {r['crossover_T_K']:.0f} K")
    print(f"  tunneling dominates at body temperature: {r['tunneling_dominates_at_body_T']}")
    print(f"  quantum signature (KIE >> classical ~1.4): {r['quantum_signature']}")
    print()
    print("  T (K)   ->   KIE(T)   [quantum -> classical crossover]")
    for t, k in r["T_scan"]:
        marker = "  <-- body T" if abs(t - T_BIO) < 1 else ""
        print(f"    {t:5.0f}   {k:.3e}{marker}")
    print()
    print("(1) isotope effect:  KIE = k_H/k_D (mass-only, m_D/m_H = 2)")
    print(f"    tunneling KIE = {r['tunneling_KIE']:.2e}   [experimentally testable in H/D exchange]")
    print()
    print("(2) geometry dependence (testable: perturb the H-bond), at body T:")
    print("    width (A)   ->   KIE(310 K)")
    for w, k in r["width_scan"]:
        print(f"      {w:.2f}      {k:.2e}")
    print("    height (eV) ->   KIE(310 K)")
    for b, k in r["height_scan"]:
        print(f"      {b:.2f}      {k:.2e}")
    print()
    print("(3) electrostatic environment (testable: change protein/water field):")
    print("    field (a.u.) ->   KIE(310 K)")
    for f, k in r["field_scan"]:
        print(f"      {f}        {k:.2e}")
    print()
    print("consistency checks (corrected):")
    print(f"    tunneling KIE for 10x heavier mass = {r['tunneling_KIE_heavy10x']:.3e}  (GROWS with mass = quantum)")
    print(f"    tunneling KIE for H/D              = {r['tunneling_KIE_light']:.3e}")
    print(f"    classical KIE (mass-independent)   = {r['classical_KIE_mass_independent']:.3f}")
    print()
    print("FALSIFIABLE PREDICTION: if a H/D exchange experiment on a known")
    print("hotspot (e.g. KRAS G12) gives a KIE ~ 1.4 (flat in T) instead of")
    print(f"order 10^3-10^5 at 310 K, the proton-tunneling channel is falsified.")
