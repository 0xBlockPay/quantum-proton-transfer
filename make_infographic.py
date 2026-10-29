"""
make_infographic.py
===================
Generates two infographic-style figures (dark theme, matplotlib) that mirror
the "Quantum Oncology: from molecular quantum events to cancer progression"
narrative, plus a dedicated cell-development process diagram.

Outputs:
  figures/infographic.png        - full multi-scale pipeline (6 stages)
  figures/07_cell_development.png - cell development: normal -> pre-malignant
                                    -> malignant -> tumor (state machine)

Run:  python make_infographic.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette (dark theme) ----
BG      = "#0b1020"
PANEL   = "#121a33"
PANEL2  = "#182449"
EDGE    = "#3b4a7a"
ACCENT  = "#4fd1c5"   # teal
ACCENT2 = "#f6ad55"   # amber
ACCENT3 = "#fc8181"   # red
TEXT    = "#e2e8f0"
MUT     = "#94a3b8"

def _set_dark_theme():
    plt.rcParams["text.color"] = TEXT
    plt.rcParams["axes.facecolor"] = BG
    plt.rcParams["figure.facecolor"] = BG


def _reset_theme():
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def _panel(ax, x, y, w, h, title, body, color=ACCENT, title_fs=11, body_fs=8.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                         linewidth=1.6, edgecolor=color, facecolor=PANEL)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.045, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold", color=color)
    ax.text(x + w/2, y + h/2 - 0.02, body, ha="center", va="center",
            fontsize=body_fs, color=TEXT, linespacing=1.5)


def _arrow(ax, x0, y0, x1, y1, color=ACCENT, lw=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=18, lw=lw, color=color))


def _band(ax, y, label, color):
    ax.add_patch(FancyBboxPatch((0.012, y), 0.10, 0.16,
                 boxstyle="round,pad=0.006,rounding_size=0.01",
                 linewidth=1.4, edgecolor=color, facecolor=PANEL2))
    ax.text(0.062, y + 0.08, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=color, rotation=90)


# ===================================================================== #
#  FIGURE 1: full multi-scale pipeline (mirrors the prompt infographic) #
# ===================================================================== #
def infographic():
    _set_dark_theme()
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # title
    ax.text(0.5, 0.965, "QUANTUM ONCOLOGY", ha="center", va="top",
            fontsize=22, fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.925, "From Molecular Quantum Events to Cancer Progression",
            ha="center", va="top", fontsize=12, color=MUT, style="italic")

    # ---- Row 1: QUANTUM SCALE (10^-10 m) ----
    _band(ax, 0.60, "QUANTUM\nSCALE\n10^-10 m", ACCENT)
    qy, qh, qw = 0.62, 0.22, 0.185
    qx = [0.13, 0.335, 0.54, 0.745]
    _panel(ax, qx[0], qy, qw, qh, "0. QPU / VQE",
           "Electronic structure\nE(x) of A\u2013T H-bond\n\n\u2192 barrier V\u2212E \u2248 0.22 eV\n   width \u2248 1.2 \u00c5\n(Qiskit, verified)", ACCENT)
    _panel(ax, qx[1], qy, qw, qh, "1. PROTON TUNNELING",
           "WKB through QPU barrier\nA\u2026T \u2194 A*\u2026T* tautomer\n\nisotope effect\nP_H/P_D \u2248 2.7\u00d710\u2074\n(mass-only anchor)", ACCENT)
    _panel(ax, qx[2], qy, qw, qh, "2. ETC e\u207b TRANSFER",
           "Quantum graph + O\u2082 sink\nNADH\u2192I\u2192III\u2192IV\u2192O\u2082\n\ntransfer \u03b7 \u2248 0.70\nleak \u2248 0.30 \u2192 ROS source", ACCENT)
    _panel(ax, qx[3], qy, qw, qh, "3. SPIN CHEMISTRY",
           "Radical pair (8-state)\n+ hyperfine coupling\n\nB-field modulates ROS\nsmall, real dip\n(magnetoreception-like)", ACCENT)
    for i in range(3):
        _arrow(ax, qx[i] + qw, qy + qh/2, qx[i+1], qy + qh/2, ACCENT)

    # ---- Row 2: MOLECULAR / MUTATION SCALE (10^-9..10^-7 m) ----
    _band(ax, 0.30, "MOLECULAR\nSCALE\n10^-9..10^-7 m", ACCENT2)
    my, mh, mw = 0.32, 0.20, 0.30
    _panel(ax, 0.13, my, mw, mh, "4. MUTATION ACCUMULATION",
           "separable: p_mut = p_tun + p_chem\n(no fitted 0.4/0.6 weights)\n\ndriver genes:\nTP53 \u00b7 KRAS \u00b7 EGFR", ACCENT2)
    _panel(ax, 0.54, my, 0.30, mh, "5. CELL TRANSFORMATION",
           "loss of cell-cycle control\nresistance to apoptosis\naltered signaling\n\ndriver hit \u2248 3.6\u00d710\u207b\u2079 / division", ACCENT2)
    _arrow(ax, 0.26, qy, 0.28, my + mh, ACCENT)      # quantum -> molecular
    _arrow(ax, 0.72, qy, 0.70, my + mh, ACCENT)
    _arrow(ax, 0.13 + mw, my + mh/2, 0.54, my + mh/2, ACCENT2)

    # ---- Row 3: TISSUE SCALE (10^-5..10^-3 m) ----
    _band(ax, 0.02, "TISSUE\nSCALE\n10^-5..10^-3 m", ACCENT3)
    ty, th, tw = 0.04, 0.18, 0.71
    _panel(ax, 0.13, ty, tw, th, "6. TUMOR GROWTH  (cellular automaton, tissue scale)",
           "normal  \u2192  pre-malignant (mutant)  \u2192  malignant  \u2192  uncontrolled proliferation + invasion\n"
           "final counts (150 steps):  normal = 2831   \u00b7   mutant = 141   \u00b7   malignant = 628",
           ACCENT3, title_fs=11, body_fs=9)
    _arrow(ax, 0.43, my, 0.43, ty + th, ACCENT3)

    fig.savefig(os.path.join(OUT, "infographic.png"), dpi=130, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    _reset_theme()
    print("[saved] figures/infographic.png")


# ===================================================================== #
#  FIGURE 2: cell development process (state machine)                  #
# ===================================================================== #
def cell_development():
    _set_dark_theme()
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.955, "CELL DEVELOPMENT PROCESS", ha="center", va="top",
            fontsize=19, fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.915, "how a quantum-driven mutation rate becomes a tumor (CA states)",
            ha="center", va="top", fontsize=11, color=MUT, style="italic")

    # quantum inputs feeding the top
    ax.text(0.5, 0.86, "QUANTUM / BIOPHYSICAL INPUTS", ha="center",
            fontsize=10, fontweight="bold", color=ACCENT)
    inputs = [
        (0.20, "tunneling\np_tun = 1e-8"),
        (0.40, "ROS leak\n(ETC \u03b7=0.70)"),
        (0.60, "spin ROS\nB-modulation"),
        (0.80, "p_mut = p_tun + p_chem\n(driver hit 3.6e-9/div)"),
    ]
    for x, t in inputs:
        _panel(ax, x - 0.09, 0.74, 0.18, 0.10, "", t, ACCENT, title_fs=1, body_fs=8)

    # state machine: 4 states in a row
    states = [
        (0.10, "NORMAL CELL", "healthy, contact\ninhibited, p53 intact\n\nCA state = 1", "#27ae60"),
        (0.34, "PRE-MALIGNANT", "driver mutation hit\n(TP53/KRAS/EGFR)\nloss of apoptosis\n\nCA state = 2", "#f39c12"),
        (0.58, "MALIGNANT", "uncontrolled division\ncontact-inhibition\nescape, second hit\n\nCA state = 3", "#fc8181"),
        (0.82, "TUMOR", "uncontrolled\nproliferation +\nangiogenesis +\ninvasion (metastasis)", "#e53e3e"),
    ]
    sy, sh, sw = 0.30, 0.26, 0.20
    for i, (x, title, body, c) in enumerate(states):
        _panel(ax, x, sy, sw, sh, title, body, c, title_fs=11, body_fs=8.5)
        if i < 3:
            _arrow(ax, x + sw, sy + sh/2, states[i+1][0], sy + sh/2, MUT, lw=2.5)

    # feedback arrows from inputs to states
    _arrow(ax, 0.20, 0.74, 0.20, sy + sh, ACCENT, lw=1.6)
    _arrow(ax, 0.40, 0.74, 0.44, sy + sh, ACCENT, lw=1.6)
    _arrow(ax, 0.60, 0.74, 0.68, sy + sh, ACCENT, lw=1.6)

    # bottom: the rate that drives the transition (the quantum -> biology bridge)
    _panel(ax, 0.34, 0.06, 0.32, 0.14,
           "THE BRIDGE (calibrated, not fitted)",
           "P(normal \u2192 mutant) per division = p_mut\n"
           "    = p_tun (tunneling) + p_chem (ROS)\n"
           "divisions to 50% driver hit \u2248 1.9\u00d710\u2078",
           ACCENT2, title_fs=10, body_fs=8.5)
    _arrow(ax, 0.50, 0.20, 0.44, sy, ACCENT2, lw=1.8)

    fig.savefig(os.path.join(OUT, "07_cell_development.png"), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    _reset_theme()
    print("[saved] figures/07_cell_development.png")


if __name__ == "__main__":
    infographic()
    cell_development()
