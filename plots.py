"""
plots.py — visualization of each link in the quantum chain -> tumor.
Generuje figures/*.png (matplotlib, bez okna / Agg backend).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("   ->", p)


def fig_tunneling(dna):
    fig, ax = plt.subplots(figsize=(6, 4))
    a, p = zip(*dna["width_scan"])
    ax.semilogy(a, p, "o-", color="#c0392b", label="proton (H)")
    a_d = [x for x, _ in dna["width_scan"]]
    # deuteron: przeskaluj przez efekt izotopowy
    p_d = [x / dna["isotope_effect"] for x in p]
    ax.semilogy(a_d, p_d, "s--", color="#2980b9", label="deuteron (D)")
    ax.set_xlabel("barrier width a (Å)")
    ax.set_ylabel("tunneling probability P (WKB)")
    ax.set_title(f"Proton tunneling vs barrier width\nisotope effect P_H/P_D = {dna['isotope_effect']:.1e}")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    _save(fig, "01_tunneling_isotope.png")


def fig_spin(spin):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.asarray(spin["B"], dtype=float) * 1000.0, np.asarray(spin["ROS_rel"], dtype=float), color="#8e44ad", lw=2)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("magnetic field B (mT)")
    ax.set_ylabel("ROS channel (rel. to B=0)")
    ax.set_title(f"Radical-pair spin chemistry\nJ={spin['J_eV']:.0e} eV, A={spin['A_eV']:.0e} eV")
    ax.grid(True, alpha=0.3)
    _save(fig, "02_spin_ros.png")


def fig_mito(mito):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mito["noise"], mito["eta"], "o-", color="#16a085", lw=2)
    ax.set_xlabel("site-energy disorder (eV)")
    ax.set_ylabel("ETC transfer efficiency η")
    ax.set_title(f"ETC electron transfer (quantum graph)\nclean η = {mito['eta_clean']:.3f}")
    ax.grid(True, alpha=0.3)
    _save(fig, "03_mito_enaqt.png")


def fig_mutation(out):
    fig, ax = plt.subplots(figsize=(6, 4))
    wn, eta, drv = zip(*out["rowsA"])
    ax.semilogy(wn, drv, "o-", color="#d35400", lw=2)
    ax.set_xlabel("ETC disorder (eV)")
    ax.set_ylabel("driver-hit rate per division")
    ax.set_title("Sensitivity: ETC disorder -> leak -> mutation rate")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "04_sensitivity_etC.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    b, sf, drv2 = zip(*out["rowsB"])
    ax.semilogy([x * 1000 for x in b], drv2, "o-", color="#2c3e50", lw=2)
    ax.set_xlabel("magnetic field B (mT)")
    ax.set_ylabel("driver-hit rate per division")
    ax.set_title("Sensitivity: B-field -> spin ROS -> mutation rate")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "05_sensitivity_B.png")


def fig_tumor(out):
    hist = out["tumor_history"]
    steps = [h[0] for h in hist]
    N = [h[1] for h in hist]
    M = [h[2] for h in hist]
    Mal = [h[3] for h in hist]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, N, label="normal", color="#27ae60")
    ax.plot(steps, M, label="mutant", color="#f39c12")
    ax.plot(steps, Mal, label="malignant", color="#c0392b")
    ax.set_xlabel("CA step")
    ax.set_ylabel("cell count")
    ax.set_title(f"Tumor growth CA (final: {out['tumor_counts']})")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "06_tumor_growth.png")


def fig_dashboard(out):
    """One panel: the full chain with numbers."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")
    q = out["qpu"]; b = q["barrier"]
    stages = [
        ("0. QPU/VQE", f"{q['n_qubits']} qubits\nV-E={b['barrier_height']:.2f} eV"),
        ("1. TUNNEL", f"p_tun={out['p_tun']:.0e}\nP_H/P_D={out['tun']['isotope_effect']:.1e}"),
        ("2. ETC LEAK", f"η={out['eta']:.2f}\nleak={out['leak']:.2f}"),
        ("3. SPIN ROS", f"factor={out['spin_factor']:.3f}\nJ,A ~ meV"),
        ("4. MUTATION", f"p_bpmut={out['mut']['p_bpmut']:.0e}\ndriver={out['per_div_driver']:.0e}"),
        ("5. TUMOR", f"N={out['tumor_counts']['normal']}\nMal={out['tumor_counts']['malignant']}"),
    ]
    n = len(stages)
    for i, (title, body) in enumerate(stages):
        x = i / n
        ax.add_patch(plt.Rectangle((x + 0.02, 0.15), 0.16, 0.5,
                                   transform=ax.transAxes,
                                   facecolor="#ecf0f1", edgecolor="#2c3e50", lw=1.5))
        ax.text(x + 0.10, 0.60, title, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.10, 0.35, body, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, family="monospace")
        if i < n - 1:
            ax.annotate("", xy=(x + 0.20, 0.40), xytext=(x + 0.185, 0.40),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))
    ax.set_title("Quantum Oncology: multi-scale chain (quantum → tumor)",
                 fontsize=12, fontweight="bold", pad=10)
    _save(fig, "00_dashboard.png")



def fig_qpu(out):
    """VQE electronic-structure potential curve E(x) -> tunneling barrier."""
    q = out["qpu"]
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, E, Ee = q["xs"], q["E"], q["E_exact"]
    ax.plot(xs, E, "o-", color="#2980b9", lw=2, label="VQE (Qiskit)")
    ax.plot(xs, Ee, ":", color="#7f8c8d", label="exact (Statevector)")
    b = q["barrier"]
    ax.axhline(b["V0"], color="#c0392b", ls="--", lw=1, label=f"barrier ridge {b['V0']:.2f} eV")
    ax.axhline(b["Emin"], color="#27ae60", ls="--", lw=1, label=f"well min {b['Emin']:.2f} eV")
    ax.set_xlabel("proton position x (A)")
    ax.set_ylabel("electronic ground-state energy E(x) (eV)")
    ax.set_title(f"VQE potential curve -> barrier (V-E={b['barrier_height']:.2f} eV, w={b['width_A']:.1f} A)\n"
                 f"{q['n_qubits']} qubits, VQE-vs-exact dev = {q['vqe_exact_max_dev']:.0e} eV")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, "00b_qpu_vqe_curve.png")

def fig_kie(out):
    """Temperature-dependent KINETIC ISOTOPE EFFECT (the verifiable core).
    Shows the quantum -> classical crossover: large KIE at low T -> ~1.4 at
    high T.  Reads from verify_isotope.run() (stored under out['kie'])."""
    r = out.get("kie")
    if not r:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    t, k = zip(*r["T_scan"])
    ax.semilogy(t, k, "o-", color="#c0392b", lw=2, label="KIE(T) = k_H/k_D")
    ax.axhline(r["classical_KIE"], color="#2980b9", ls="--", lw=1.5,
               label=f"classical limit = {r['classical_KIE']:.2f}")
    ax.axvline(310.0, color="#27ae60", ls=":", lw=1.5, label="body T = 310 K")
    ax.axvline(r["crossover_T_K"], color="#f39c12", ls=":", lw=1.5,
               label=f"crossover T* = {r['crossover_T_K']:.0f} K")
    ax.set_xlabel("temperature T (K)")
    ax.set_ylabel("KIE  k_H/k_D")
    ax.set_title(f"KINETIC ISOTOPE EFFECT (verifiable core)\n"
                 f"KIE(310 K) = {r['KIE_physiological']:.2e}  (tunneling-dominated: "
                 f"{r['tunneling_dominates_at_body_T']})")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    _save(fig, "01b_kie_temperature.png")


def fig_core(out):
    """Publishable core: KIE P_H/P_D vs real A-T binding geometry + water.
    Reads from out['core_scan'] (the geometry/environment scan in main.py)."""
    rows = out.get("core_scan")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    # no-water (solid) vs water (dashed), each as a function of R
    nw = sorted([r for r in rows if not r["water"]], key=lambda r: r["R"])
    w  = sorted([r for r in rows if r["water"]], key=lambda r: r["R"])
    if nw:
        ax.semilogy([r["R"] for r in nw], [r["KIE"] for r in nw], "o-",
                    color="#c0392b", lw=2, label="no water (gas-phase-like)")
    if w:
        ax.semilogy([r["R"] for r in w], [r["KIE"] for r in w], "s--",
                    color="#2980b9", lw=2, label="+ frozen water (QM/MM)")
    ax.set_xlabel("N...N bond distance R (A)")
    ax.set_ylabel("KIE  P_H/P_D  (mass-only WKB)")
    ax.set_title("Publishable core: real A-T barrier -> KIE\n"
                 "KIE varies ~10^2 with geometry + water (testable)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    _save(fig, "00c_core_realAT_kie.png")


def fig_clamp(out):
    """Analysis 1: CLAMP SENSITIVITY -- KIE vs barrier height across the clamp."""
    cs = out.get("clamp_sensitivity")
    if not cs:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    h = [r["barrier_eV"] for r in cs["rows"]]
    km = [r["KIE_mass"] for r in cs["rows"]]
    k310 = [r["KIE_310K"] for r in cs["rows"]]
    ax.semilogy(h, km, "o-", color="#c0392b", lw=2, label="KIE (mass-only)")
    ax.semilogy(h, k310, "s--", color="#2980b9", lw=2, label="KIE (310 K, classical+tunnel)")
    ax.axvspan(cs["clamp_range"][0], cs["clamp_range"][1], color="#27ae60", alpha=0.12,
               label=f"physical clamp [{cs['clamp_range'][0]},{cs['clamp_range'][1]}] eV")
    ax.set_xlabel("barrier height V-E (eV)")
    ax.set_ylabel("KIE  P_H/P_D")
    ax.set_title("Clamp sensitivity: KIE is set by the CLAMPED (physical) barrier\n"
                 "(raw minimal-basis VQE height lies far right -> KIE ~ 1.4)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    _save(fig, "00d_clamp_sensitivity.png")


def fig_methods(out):
    """Analysis 2: METHOD COMPARISON -- F-H-F- barrier by level of theory."""
    m = out.get("methods")
    if not m:
        return
    res = m["results"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names, vals, wells = [], [], []
    for k in ("HF", "DFT", "CCSD(T)", "VQE"):
        d = res.get(k)
        if d is None or "error" in d:
            continue
        names.append(k); vals.append(d["barrier_height_eV"])
        wells.append("double" if d["double_well"] else "single")
    colors = ["#c0392b" if w == "double" else "#f39c12" for w in wells]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("F-H-F- barrier height (eV)")
    ax.set_title(f"Method comparison (basis = {m['basis']}): shape AND height are\n"
                 "method/basis-dependent (red = double well, amber = single well)")
    for b, v, w in zip(bars, vals, wells):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f} eV\n{w}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(5.0, color="#27ae60", ls=":", lw=1.5, label="physical H-bond range (2-5 eV)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "00e_method_comparison.png")


def fig_basis_scan(out):
    """Analysis 4: BASIS + ACTIVE-SPACE SCAN -- does the clamp earn its keep?"""
    bs = out.get("basis_set_scan")
    asc = out.get("active_space_scan")
    if not bs and not asc:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: barrier vs basis
    if bs:
        names = [r["basis"] for r in bs["rows"] if not r.get("error")]
        vals = [r["barrier_eV"] for r in bs["rows"] if not r.get("error")]
        bars = ax1.bar(names, vals, color="#2980b9")
        ax1.set_ylabel("F-H-F- barrier height (eV)")
        ax1.set_title(f"V(R, basis)  [active space {bs['active']}, fixed]")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7)
        ax1.axhline(5.0, color="#27ae60", ls=":", lw=1.5, label="clamp top (5 eV)")
        ax1.axhline(0.1, color="#27ae60", ls=":", lw=1.5, label="clamp floor (0.1 eV)")
        ax1.legend(fontsize=7); ax1.grid(True, axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=30, labelsize=7)
    # right: barrier vs active space
    if asc:
        names = [str(r["active"]) for r in asc["rows"] if not r.get("error")]
        vals = [r["barrier_eV"] for r in asc["rows"] if not r.get("error")]
        bars = ax2.bar(names, vals, color="#8e44ad")
        ax2.set_ylabel("F-H-F- barrier height (eV)")
        ax2.set_title(f"V(R, active space)  [basis {asc['basis']}, fixed]")
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7)
        ax2.axhline(5.0, color="#27ae60", ls=":", lw=1.5, label="clamp top (5 eV)")
        ax2.axhline(0.1, color="#27ae60", ls=":", lw=1.5, label="clamp floor (0.1 eV)")
        ax2.legend(fontsize=7); ax2.grid(True, axis="y", alpha=0.3)
        ax2.tick_params(axis="x", labelsize=7)
    fig.suptitle("The raw active-space barrier is already inside the clamp and is "
                 "stable in basis AND active space -> the clamp is a safety net, "
                 "not an active filter", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "00g_basis_scan.png")


def fig_kie_map(out):
    """Analysis 3: KIE(R, T, epsilon) MAP -- geometry x temperature x environment."""
    km = out.get("kie_map")
    if not km:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    # plot KIE(T) vs T for each (R, epsilon), no-water solid, water dashed
    import itertools
    for (R, eps), style, col in [((2.8, False), "o-", "#c0392b"),
                                  ((3.0, False), "s-", "#e67e22"),
                                  ((3.2, False), "^-", "#f1c40f"),
                                  ((2.8, True), "o--", "#2980b9"),
                                  ((3.0, True), "s--", "#16a085"),
                                  ((3.2, True), "^--", "#8e44ad")]:
        pts = [g for g in km["grid"] if g["R"] == R and g["epsilon"] == eps]
        if not pts:
            continue
        pts = sorted(pts, key=lambda g: g["T"])
        lab = f"R={R}, {'water' if eps else 'no water'}"
        ax.semilogy([g["T"] for g in pts], [g["KIE_T"] for g in pts], style,
                    color=col, lw=2, label=lab)
    ax.set_xlabel("temperature T (K)")
    ax.set_ylabel("KIE  P_H/P_D")
    ax.set_title("KIE(R, T, epsilon) map: geometry x temperature x environment\n"
                 "(solid = no water, dashed = +water; T flat here = tunneling-dominated)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, which="both", alpha=0.3)
    _save(fig, "00f_kie_map.png")


def make_all(out):
    print("[plots] generating figures...")
    fig_dashboard(out)
    fig_qpu(out)
    fig_core(out)
    fig_clamp(out)
    fig_methods(out)
    fig_basis_scan(out)
    fig_kie_map(out)
    fig_tunneling(out["dna"])
    fig_kie(out)
    fig_spin(out["spin"])
    fig_mito(out["mito"])
    fig_mutation(out)
    fig_tumor(out)
