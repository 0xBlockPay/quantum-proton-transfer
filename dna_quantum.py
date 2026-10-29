"""
dna_quantum.py
==============
Proton tunneling in a DNA base-pair hydrogen bond (A*...T* tautomerisation).

Honest scope (this is what makes it defensible):
  * We do NOT claim to predict the absolute tunneling rate from first
    principles -- the geometry-dependent pre-exponential factor (how often
    the two bases are in the tautomeric configuration) is not known.
  * What we DO compute from first principles (mass-dependent, WKB):
        - the *relative* tunneling probabilities for H vs D
        - the ISOTOPE EFFECT  P_H / P_D  ~ 10^5   <-- the quantum anchor
        - the exponential sensitivity to barrier width / height
  * The *absolute* per-division tunneling probability is CALIBRATED to the
    measured spontaneous point-mutation rate of the relevant hotspot.

All of this is CLASSICAL quantum mechanics (Schrödinger / WKB). No QPU is
required for a single H-bond; a QPU is only justified for the full
electronic structure of the base pair + protein environment (see main.py).
"""
import numpy as np

# ---- constants ----------------------------------------------------------
HBAR = 1.0545718e-34          # J s
EVJ  = 1.602176634e-19        # J per eV
M_H  = 1.6726219e-27          # kg  (proton)
M_D  = 3.3435837e-27          # kg  (deuteron)
ANG  = 1e-10                  # m per Angstrom

# ---- barrier parameters (A-T hydrogen-bond tautomerisation ridge) -------
V0   = 1.00 * EVJ             # barrier height (J)
E    = 0.50 * EVJ             # proton energy (J)  -> V-E = 0.5 eV
A_W  = 1.00 * ANG             # effective barrier width (m)

# ---- calibration to measured biology ------------------------------------
# Spontaneous point-mutation rate per nucleotide per division is ~1e-8..1e-9.
# We assign the tunneling/tautomerisation channel a calibrated share of a
# hotspot base-pair. This is a measured anchor, NOT a fitted 0.4/0.6 weight.
P_TUN_CALIB_H = 1.0e-8        # per-division tunneling probability (H), calibrated

# ---- HONEST PROVENANCE (per the physics review) -------------------------
# The tunneling module produces two DIFFERENT kinds of numbers. We label
# them so the evidence ledger does not overclaim:
#   * ISOTOPE EFFECT P_H/P_D  -> FIRST PRINCIPLES (mass-only, WKB). This is
#     the robust, defensible, experimentally testable quantum anchor.
#   * ABSOLUTE per-division rate (P_TUN_CALIB_H) -> CALIBRATED to the
#     measured spontaneous mutation rate. It is NOT predicted from first
#     principles: the geometry-dependent pre-exponential factor (how often
#     the bases are in the tautomeric configuration) is unknown.
#   * BARRIER (height/width) -> from the QPU (electronic structure), but the
#     absolute height is clamped to the physical H-bond range (the minimal
#     active space overestimates it).
PROVENANCE = {
    "isotope_effect": "first-principles (mass-only WKB; experimentally testable)",
    "absolute_rate":  "calibrated to measured spontaneous mutation rate (~1e-8/div)",
    "barrier":        "QPU electronic structure (height clamped to physical H-bond range)",
}


def wkb_P(mass_kg, V0=V0, E=E, a=A_W):
    """WKB tunneling probability through a rectangular barrier."""
    if E >= V0:
        return 1.0
    k = np.sqrt(2.0 * mass_kg * (V0 - E)) / HBAR
    return float(np.exp(-2.0 * a * k))


def run():
    P_H = wkb_P(M_H)
    P_D = wkb_P(M_D)
    iso = P_H / P_D

    # calibrate absolute rate: H is set by measurement, D follows the isotope effect
    p_H_div = P_TUN_CALIB_H
    p_D_div = P_TUN_CALIB_H / iso

    # sensitivity: how P_H changes with barrier width (exponential!)
    width_scan = []
    for a_A in [0.5, 0.75, 1.0, 1.25, 1.5]:
        width_scan.append((a_A, wkb_P(M_H, a=a_A * ANG)))

    return dict(
        P_H=P_H, P_D=P_D, isotope_effect=iso,
        p_H_per_div=p_H_div, p_D_per_div=p_D_div,
        width_scan=width_scan,
        mass_ratio=M_D / M_H,
        provenance=PROVENANCE,
    )


if __name__ == "__main__":
    r = run()
    print("=== dna_quantum : proton tunneling in A-T H-bond (WKB) ===")
    print(f"barrier: V-E = {(V0-E)/EVJ:.2f} eV, width = {A_W/ANG:.2f} A")
    print(f"P_H (rel)  = {r['P_H']:.3e}")
    print(f"P_D (rel)  = {r['P_D']:.3e}")
    print(f"ISOTOPE EFFECT P_H/P_D = {r['isotope_effect']:.3e}   (m_D/m_H = {r['mass_ratio']:.2f})")
    print(f"calibrated per-division:  H = {r['p_H_per_div']:.1e}   D = {r['p_D_per_div']:.1e}")
    print("barrier-width sensitivity (P_H):")
    for a_A, p in r["width_scan"]:
        print(f"   a = {a_A:.2f} A  ->  P_H ~ {p:.2e}")


def tunnel_from_barrier(barrier_eV, width_A, calib_H=P_TUN_CALIB_H):
    """WKB tunneling from a barrier produced by the QPU (qpu_vqe.py).

    barrier_eV : effective barrier height (V - E) in eV, from the VQE curve
    width_A    : barrier width in Angstrom, from the VQE curve
    Returns P_H, P_D, isotope effect, and the calibrated per-division rate.
    """
    dE = barrier_eV * EVJ
    a = width_A * ANG
    P_H = float(np.exp(-2.0 * a * np.sqrt(2.0 * M_H * dE) / HBAR))
    P_D = float(np.exp(-2.0 * a * np.sqrt(2.0 * M_D * dE) / HBAR))
    iso = P_H / P_D
    return dict(P_H=P_H, P_D=P_D, isotope_effect=iso,
                p_H_per_div=calib_H, p_D_per_div=calib_H / iso,
                barrier_eV=barrier_eV, width_A=width_A,
                provenance=PROVENANCE)
