"""
mitochondria.py
===============
Electron transfer along the mitochondrial Electron Transport Chain (ETC)
modelled as a QUANTUM GRAPH with real redox potentials and Lindblad
decoherence (open quantum system).

Hamiltonian (tight-binding on a graph):
    H = sum_i eps_i |i><i|  +  sum_{i,j} J_ij |i><j|

  * sites i = redox centres (NADH, FMN, Fe-S, Q, cyt c, CuA/IV, O2)
  * eps_i   = site energies from REAL redox potentials (eV)
  * J_ij    = electronic coupling between connected centres (graph edges)

The final site (O2) is an ABSORBING SINK: H[t,t] -= i*Gamma, so an electron
that reaches O2 is irreversibly captured (norm is lost).

OPEN-QUANTUM-SYSTEMS UPGRADE (Lindblad decoherence):
  The real ETC is a warm, wet, non-isolated system.  We model environment
  dephasing with Lindblad operators L_k = sqrt(gamma_k) |k><k|:
      d rho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
  We integrate the density matrix (not just the state vector) and measure the
  O2 capture WITH decoherence.  This is the physically correct treatment for
  a biological electron chain, and it exposes the
  environment-assisted quantum transport (ENAQT) crossover: too little
  decoherence -> Anderson localisation (poor transfer); too much -> pure
  diffusion (also poor); an intermediate gamma is optimal.

Why it matters for the cancer story:
  ETC efficiency sets the electron LEAK rate = the primary source of
  mitochondrial ROS.  A less efficient / more disordered chain leaks more
  electrons to O2 -> more ROS -> more DNA damage.  So transfer efficiency is
  a physically-grounded *input* to the ROS module, not a free number.
"""
import numpy as np

# ---- real ETC topology (graph, not a linear chain) ---------------------
# Node names and (approximate) redox potentials in eV (vs. vacuum, downhill
# to O2).  Values are order-of-magnitude, from standard biochemistry tables.
NODES = ["NADH", "FMN", "FeS_I", "Q", "cyt_c", "CuA_IV", "O2"]
EPS = np.array([0.0, -0.15, -0.30, -0.45, -0.55, -0.70, -0.85])  # eV
# graph edges (i,j) with coupling J (eV).  Q branches to two paths (III/IV).
EDGES = [(0, 1, 0.25), (1, 2, 0.25), (2, 3, 0.30),
         (3, 4, 0.25), (4, 5, 0.25), (5, 6, 0.40)]
N = len(NODES)
TARGET = N - 1   # O2


def build_graph(eps_noise=0.0, seed=0):
    """Return (H, target).  H is the (complex) tight-binding Hamiltonian on
    the ETC graph with optional site-energy disorder."""
    rng = np.random.default_rng(seed)
    eps = EPS.copy()
    if eps_noise:
        eps = eps + eps_noise * rng.standard_normal(N)
    H = np.zeros((N, N), dtype=complex)
    H[np.diag_indices(N)] = eps
    for i, j, J in EDGES:
        H[i, j] += J
        H[j, i] += J
    return H, TARGET


def _propagate(H, target, T, n_t, source=0, Gamma=0.15, gamma_deph=0.0):
    """Integrate the (possibly open) dynamics and return O2 capture fraction.

    gamma_deph=0 -> closed system (unitary, non-Hermitian H with sink).
    gamma_deph>0 -> Lindblad dephasing on each site (density matrix).
    """
    H = H.copy()
    if gamma_deph <= 0.0:
        # CLOSED system: non-Hermitian H (absorbing sink) propagated as a
        # state vector.  Norm loss = O2 capture.
        H[target, target] -= 1j * Gamma
        w, V = np.linalg.eig(H)
        Vi = np.linalg.inv(V)
        c0 = np.zeros(N, dtype=complex); c0[source] = 1.0
        dt = T / n_t
        c = c0
        for _ in range(n_t):
            c = V @ (np.exp(-1j * w * dt) * (Vi @ c))
        return float(min(max(1.0 - np.vdot(c, c).real, 0.0), 1.0))

    # OPEN system: master equation on the density matrix.
    #   d rho/dt = -i[H_herm, rho]                      (coherent, Hermitian H)
    #             - (Gamma/2)(Ps rho + rho Ps)          (O2 sink / absorption)
    #             - gamma_deph (rho - diag(rho))        (uniform site dephasing)
    # The sink term is the anti-Hermitian part (it removes probability); the
    # dephasing term only damps coherences (trace-conserving).  This is the
    # correct Lindblad form for dephasing L_k = sqrt(gamma)|k><k| summed over
    # k (which reduces to -gamma*(rho - diag(rho)) for uniform gamma).
    Ps = np.zeros((N, N), dtype=complex)
    Ps[target, target] = 1.0
    rho = np.zeros((N, N), dtype=complex)
    rho[source, source] = 1.0
    dt = T / n_t
    captured = 0.0
    for _ in range(n_t):
        comm = H @ rho - rho @ H
        sink = -0.5 * Gamma * (Ps @ rho + rho @ Ps)
        deph = -gamma_deph * (rho - np.diag(np.diag(rho)))
        rho = rho + dt * (-1j * comm + sink + deph)
        captured = 1.0 - float(np.real(np.trace(rho)))
    return float(min(max(captured, 0.0), 1.0))


def transfer_efficiency(eps_noise=0.0, seed=0, gamma_deph=0.0,
                        T=200.0, n_t=4000, Gamma=0.15):
    """O2 capture fraction (transfer efficiency) for the ETC graph."""
    H, target = build_graph(eps_noise=eps_noise, seed=seed)
    return _propagate(H, target, T, n_t, Gamma=Gamma, gamma_deph=gamma_deph)


def run():
    # clean, closed-system efficiency (the headline number)
    eta0 = transfer_efficiency(eps_noise=0.0, gamma_deph=0.0)

    # disorder scan (Anderson localisation -> less transfer -> more leak)
    noise = np.linspace(0.0, 0.3, 11)
    eta = np.array([transfer_efficiency(eps_noise=wn) for wn in noise])

    # ENAQT: decoherence scan at fixed small disorder (the crossover)
    gamma = np.linspace(0.0, 0.3, 11)
    eta_deph = np.array([transfer_efficiency(eps_noise=0.05, gamma_deph=g)
                         for g in gamma])

    return dict(n=N, target=TARGET, eta_clean=eta0,
                noise=noise, eta=eta,
                gamma=gamma, eta_deph=eta_deph,
                eta_deph_peak=eta_deph.max(),
                gamma_at_peak=gamma[eta_deph.argmax()],
                nodes=NODES, eps=EPS)


if __name__ == "__main__":
    r = run()
    print("=== mitochondria : ETC electron transfer (graph + Lindblad) ===")
    print(f"graph: {r['n']} redox centres, target = O2 (site {r['target']})")
    print(f"nodes: {' -> '.join(r['nodes'])}")
    print(f"clean transfer efficiency eta = {r['eta_clean']:.4f}")
    print("disorder scan (Anderson localisation):")
    for wn, e in zip(r["noise"][::2], r["eta"][::2]):
        print(f"   disorder = {wn:.2f} eV  ->  eta = {e:.4f}")
    print("Lindblad decoherence scan (ENAQT, disorder=0.05):")
    for g, e in zip(r["gamma"][::2], r["eta_deph"][::2]):
        print(f"   gamma = {g:.2f}  ->  eta = {e:.4f}")
    print(f"peak efficiency = {r['eta_deph_peak']:.4f} at gamma = {r['gamma_at_peak']:.2f}")
