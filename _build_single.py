"""Builds quantum_oncology_single.py from the individual modules.

STALE / DEPRECATED — DO NOT RUN.
================================
quantum_oncology_single.py is now HAND-MAINTAINED and is the source of
truth.  It was extended well beyond what this generator produces (added
verify_isotope, qpu_core, clamp_sensitivity, method_comparison, kie_map,
and a rewritten run_pipeline), and the generator's name-replacement
assumptions no longer match qpu_vqe.py (run(prefer=...) / _hamiltonian_qpu).

Running this script would therefore either crash on its own assertions or,
worse, overwrite the good single file with an incomplete rebuild.  It is
disabled on purpose.  Edit quantum_oncology_single.py directly.
"""
import sys

if __name__ == "__main__":
    sys.stderr.write(
        "STALE: _build_single.py is DEPRECATED and disabled.\n"
        "  quantum_oncology_single.py is now the hand-maintained source of truth.\n"
        "  Do not run this script; edit quantum_oncology_single.py directly.\n"
    )
    raise SystemExit(1)

# Everything below is kept only as historical reference and is never executed.
import re, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(f): return open(os.path.join(HERE, f)).read()

def strip_main_block(src):
    """Remove the `if __name__ == "__main__":` block but keep any later
    top-level defs (e.g. tunnel_from_barrier in dna_quantum)."""
    lines = src.splitlines()
    out, skip = [], False
    for ln in lines:
        if ln.startswith('if __name__ == "__main__":'):
            skip = True
            continue
        if skip:
            # a top-level def/class ends the main block
            if re.match(r'^(def |class )', ln):
                skip = False
                out.append(ln)
            # else drop the line (part of the main block)
        else:
            out.append(ln)
    return "\n".join(out).rstrip() + "\n"

# ---- qpu_vqe : rename hamiltonian -> hamiltonian_qpu, run -> run_qpu_vqe ----
q = rd("qpu_vqe.py")
q = strip_main_block(q)
q = q.replace("def hamiltonian(", "def hamiltonian_qpu(")
q = q.replace("op = hamiltonian(x)", "op = hamiltonian_qpu(x)")
q = q.replace("def run():", "def run_qpu_vqe():")
assert "def hamiltonian_qpu(" in q and "def run_qpu_vqe(" in q
assert "def run():" not in q

# ---- dna_quantum : keep tunnel_from_barrier, rename run -> run_dna_quantum ----
d = rd("dna_quantum.py")
d = strip_main_block(d)
d = d.replace("def run():", "def run_dna_quantum():")
assert "def run_dna_quantum(" in d and "def tunnel_from_barrier(" in d
assert "def run():" not in d

# ---- ros_spin : rename run -> run_ros_spin (hamiltonian stays, unique now) ----
rs = rd("ros_spin.py")
rs = strip_main_block(rs)
rs = rs.replace("def run():", "def run_ros_spin():")
assert "def run_ros_spin(" in rs and "def hamiltonian(" in rs

# ---- mitochondria : rename run -> run_mitochondria ----
m = rd("mitochondria.py")
m = strip_main_block(m)
m = m.replace("def run():", "def run_mitochondria():")
assert "def run_mitochondria(" in m

# ---- mutation_engine : rename run -> run_mutation_engine ----
me = rd("mutation_engine.py")
me = strip_main_block(me)
me = me.replace("def run(", "def run_mutation_engine(")
assert "def run_mutation_engine(" in me

# ---- tumor_growth : rename run -> run_tumor_growth ----
t = rd("tumor_growth.py")
t = strip_main_block(t)
t = t.replace("def run(", "def run_tumor_growth(")
assert "def run_tumor_growth(" in t

# ---- plots & make_infographic : keep as-is (their names are unique) ----
pl = strip_main_block(rd("plots.py"))
mi = strip_main_block(rd("make_infographic.py"))
assert "def make_all(" in pl and "def infographic(" in mi and "def cell_development(" in mi

# ---- assemble ----
header = '''"""
quantum_oncology_single.py
==========================
Single-file build of the Quantum Oncology multi-scale model:
    QPU (VQE) -> quantum phenomena -> DNA damage -> mutations
    -> cell transformation -> tumor growth

All modules are inlined below, in dependency order:
    1. qpu_vqe          (QPU: VQE electronic structure -> tunneling barrier)
    2. dna_quantum      (WKB proton tunneling + isotope effect)
    3. ros_spin         (radical-pair spin chemistry, 8-state + hyperfine)
    4. mitochondria     (ETC electron transfer as a quantum graph)
    5. mutation_engine  (separable bridge to TP53/KRAS/EGFR)
    6. tumor_growth     (cellular automaton, tissue scale)
    7. plots            (per-stage figures)
    8. make_infographic (infographic.png + 07_cell_development.png)
    9. pipeline         (end-to-end chain + sensitivity analysis)

Every number is either (a) computed from (classical) quantum mechanics,
(b) computed on a QPU (VQE on Qiskit), or (c) calibrated to a measurement.
No free fitted weights.

Run:
    python quantum_oncology_single.py                 # full chain (QPU) + sensitivity
    python quantum_oncology_single.py --plots         # also writes figures/*.png
    python quantum_oncology_single.py --infographics  # also regenerates infographic.png
    python quantum_oncology_single.py --module qpu_vqe   # run one module standalone
"""
import sys
import os
import json
import numpy as np

# ===================================================================== #
#  1. QPU (VQE) module                                                 #
# ===================================================================== #
'''

sections = [
    ("1. QPU (VQE) module", q),
    ("2. dna_quantum module", d),
    ("3. ros_spin module", rs),
    ("4. mitochondria module", m),
    ("5. mutation_engine module", me),
    ("6. tumor_growth module", t),
    ("7. plots module", pl),
    ("8. make_infographic module", mi),
]

body = header
for i, (name, code) in enumerate(sections):
    if i == 0:
        body += code + "\n"
    else:
        body += (
            "# ===================================================================== #\n"
            f"#  {name}                                                            #\n"
            "# ===================================================================== #\n"
            + code + "\n"
        )

# ---- pipeline (hand-written orchestrator calling the renamed functions) ----
pipeline = '''
# ===================================================================== #
#  9. pipeline (end-to-end chain + sensitivity analysis)               #
# ===================================================================== #
def run_pipeline(argv=None):
    """Full chain: QPU -> tunneling -> ETC -> spin -> mutation -> tumor."""
    if argv is None:
        argv = sys.argv[1:]

    # 0. QPU (VQE): electronic structure -> tunneling barrier
    qpu = run_qpu_vqe()
    bar = qpu["barrier"]
    barrier_eV = bar["barrier_height"]     # V - E from the VQE curve
    width_A    = bar["width_A"]

    # 1. Tunneling (CLASSICAL WKB) -- barrier from the QPU
    tun = tunnel_from_barrier(barrier_eV, width_A)
    p_tun = tun["p_H_per_div"]            # calibrated to measurement: 1e-8 /division
    iso   = tun["isotope_effect"]         # P_H/P_D ~ 1e4..1e5  <- QUANTUM ANCHOR

    # 2. ETC -> electron leak = source of ROS
    mito = run_mitochondria()
    eta   = mito["eta_clean"]
    leak  = 1.0 - eta

    # 3. Spin modulation of ROS (tunable by magnetic field)
    spin = run_ros_spin()
    spin_factor = float(spin["ROS_rel"].max())

    # 4. Chemical ROS scale -> mutations (calibrated to measurement)
    ROS_CALIB = 1e-8 / 0.30
    p_chem = ROS_CALIB * leak * spin_factor

    # 5. Bridge to driver genes (SEPARABLE: p_tun + p_chem)
    mut = run_mutation_engine(p_tun=p_tun, p_chem=p_chem)
    per_div_driver = mut["per_div_driver"]

    # 6. Cellular automaton (tissue scale)
    CA_DIV_PER_STEP = 1e5
    p_mut_per_step  = per_div_driver * CA_DIV_PER_STEP
    tumor = run_tumor_growth(steps=150, size=60, p_mut=p_mut_per_step, seed=42)

    # ---------------- RESULT ----------------
    print("=== QUANTUM ONCOLOGY: full chain (QPU end-to-end) ===")
    print(f"[0] QPU/VQE:     {qpu['n_qubits']} qubits, {qpu['n_layers']} layers, "
          f"VQE-vs-exact dev = {qpu['vqe_exact_max_dev']:.1e} eV")
    print(f"    barrier from VQE: V-E = {barrier_eV:.3f} eV, width = {width_A:.2f} A")
    print(f"[1] tunneling:  p_tun = {p_tun:.1e}  |  isotope effect P_H/P_D = {iso:.2e}")
    print(f"[2] ETC:          eta = {eta:.3f}  ->  leak = {leak:.3f}")
    print(f"[3] spin ROS:     B factor = {spin_factor:.3f}  (J={spin['J_eV']:.0e}, A={spin['A_eV']:.0e} eV)")
    print(f"[4] p_chem = {p_chem:.2e}   (calibrated, not a free weight)")
    print(f"[5] mutations:      p_bpmut = {mut['p_bpmut']:.1e}  |  driver/division = {per_div_driver:.2e}")
    print(f"    divisions to 50% driver hit = {mut['n_div_to_50pct']:.2e}")
    print(f"[6] tumor (CA):     {tumor['counts']}  (total = {tumor['total_cells']})")

    # ---------------- SENSITIVITY ANALYSIS ----------------
    print("\\n=== SENSITIVITY: how physics drives the mutation rate ===")
    print(" (A) ETC order (Anderson localization) -> leak -> p_chem:")
    rowsA = []
    for wn in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        H, t = build_chain(8, 0.2, 0.3, eps_noise=wn, seed=1)
        e = transfer_efficiency(H, t)
        pc = ROS_CALIB * (1 - e) * spin_factor
        mm = run_mutation_engine(p_tun=p_tun, p_chem=pc)
        rowsA.append((wn, e, mm["per_div_driver"]))
        print(f"    disorder={wn:.2f} eV  eta={e:.3f}  ->  driver/division = {mm['per_div_driver']:.2e}")

    print(" (B) magnetic field B -> spin modulation of ROS:")
    rowsB = []
    for b in [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        sf = float(spin["ROS_rel"][np.argmin(np.abs(spin["B"] - b))])
        pc = ROS_CALIB * leak * sf
        mm = run_mutation_engine(p_tun=p_tun, p_chem=pc)
        rowsB.append((b, sf, mm["per_div_driver"]))
        print(f"    B={b*1000:5.1f} mT  spin_factor={sf:.4f}  ->  driver/division = {mm['per_div_driver']:.2e}")

    print(" (C) QPU barrier -> tunneling -> isotope effect:")
    print(f"    VQE barrier (V-E={barrier_eV:.3f} eV, w={width_A:.2f} A) -> "
          f"P_H={tun['P_H']:.2e}, P_D={tun['P_D']:.2e}, P_H/P_D={iso:.2e}")

    # ---------------- SAVE FOR PLOTS ----------------
    out = dict(
        qpu=qpu, tun=tun, dna=run_dna_quantum(), spin=spin, mito=mito, mut=mut,
        tumor_counts=tumor["counts"], tumor_total=tumor["total_cells"],
        tumor_history=tumor["history"],
        rowsA=rowsA, rowsB=rowsB,
        p_tun=p_tun, p_chem=p_chem, eta=eta, leak=leak,
        spin_factor=spin_factor, per_div_driver=per_div_driver,
    )
    def _py(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, dict): return {k: _py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_py(v) for v in o]
        return o
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f:
        json.dump(_py(out), f, indent=2)
    print("\\n[saved] results.json")

    # ---------------- PLOTS ----------------
    if "--plots" in argv:
        make_all(out)
        print("[saved] figures/*.png")
    if "--infographics" in argv:
        infographic()
        cell_development()
    return out


# ===================================================================== #
#  standalone module runners (for --module <name>)                     #
# ===================================================================== #
def _module_standalone(name):
    if name == "qpu_vqe":
        r = run_qpu_vqe()
        print(f"barrier: V-E={r['barrier']['barrier_height']:.3f} eV, "
              f"w={r['barrier']['width_A']:.2f} A, VQE-vs-exact dev={r['vqe_exact_max_dev']:.2e} eV")
    elif name == "dna_quantum":
        r = run_dna_quantum()
        print(f"isotope effect P_H/P_D = {r['isotope_effect']:.3e}, "
              f"calibrated p_H/div = {r['p_H_per_div']:.1e}")
    elif name == "ros_spin":
        r = run_ros_spin()
        print(f"baseline ROS = {r['baseline']:.4f}, max ROS_rel = {r['ROS_rel'].max():.4f}")
    elif name == "mitochondria":
        r = run_mitochondria()
        print(f"clean eta = {r['eta_clean']:.4f}, peak eta = {r['eta_peak']:.4f}")
    elif name == "mutation_engine":
        r = run_mutation_engine()
        print(f"p_bpmut = {r['p_bpmut']:.1e}, driver/div = {r['per_div_driver']:.2e}")
    elif name == "tumor_growth":
        r = run_tumor_growth()
        print(f"final counts = {r['counts']}")
    else:
        raise SystemExit(f"unknown module: {name}")


if __name__ == "__main__":
    if "--module" in sys.argv:
        _module_standalone(sys.argv[sys.argv.index("--module") + 1])
    else:
        run_pipeline()
'''

body += pipeline

open(os.path.join(HERE, "quantum_oncology_single.py"), "w").write(body)
print("WROTE quantum_oncology_single.py  (%d bytes)" % len(body))
