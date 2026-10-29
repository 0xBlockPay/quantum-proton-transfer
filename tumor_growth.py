"""
tumor_growth.py
===============
Top (tissue) scale: a 2D cellular automaton of tumour growth driven by the
mutation rate coming from mutation_engine.py.

Model (contact-inhibited growth with a mutation-driven escape):
  * Each cell is one of:  0 empty, 1 normal, 2 mutant (pre-malignant),
                          3 malignant (uncontrolled).
  * Normal cells divide if space allows (contact inhibition).
  * A normal cell acquires a DRIVER MUTATION with probability p_mut per
    division -> becomes mutant (TP53 loss = loss of apoptosis + contact
    inhibition).
  * Mutant cells divide FASTER (loss of cell-cycle control) and do NOT
    respect contact inhibition as strongly.
  * A small fraction of mutants progress to fully malignant (second hit),
    which proliferates uncontrolled and can invade (metastasis seed).

The ONLY quantum-derived input is p_mut (per-division driver-hit rate).
Everything else is standard, well-tested tumour-growth CA behaviour. This
keeps the quantum/biophysical part (the *rate*) cleanly separated from the
classical tissue-scale dynamics (the *consequence*).
"""
import numpy as np

EMPTY, NORMAL, MUTANT, MALIGNANT = 0, 1, 2, 3


def step(grid, p_mut, mutant_div=0.8, malig_div=0.9,
         mutant_ci=0.5, invad=0.02, seed=None):
    """One CA step. Returns (new_grid, events)."""
    rng = np.random.default_rng(seed)
    g = grid.copy()
    events = {"mut": 0, "malig": 0, "div": 0}
    H, W = g.shape
    for y in range(H):
        for x in range(W):
            c = g[y, x]
            if c == EMPTY:
                continue
            # count occupied neighbours (Moore)
            y0, y1 = max(0, y - 1), min(H, y + 2)
            x0, x1 = max(0, x - 1), min(W, x + 2)
            nbrs = g[y0:y1, x0:x1]
            occupied = int((nbrs > 0).sum()) - 1
            free = 8 - occupied
            if free <= 0:
                # contact inhibition: normal cells stop; mutant/malignant escape
                if c == NORMAL:
                    continue
                p_div = mutant_ci if c == MUTANT else 0.7
            else:
                p_div = 0.5 if c == NORMAL else (mutant_div if c == MUTANT else malig_div)
            if rng.random() > p_div:
                continue
            # choose a free neighbour to divide into
            free_cells = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == EMPTY:
                        free_cells.append((ny, nx))
            if not free_cells:
                continue
            ny, nx = free_cells[rng.integers(len(free_cells))]
            # new cell state
            newc = c
            if c == NORMAL and rng.random() < p_mut:
                newc = MUTANT
                events["mut"] += 1
            elif c == MUTANT and rng.random() < 0.05:
                newc = MALIGNANT
                events["malig"] += 1
            g[ny, nx] = newc
            events["div"] += 1
    return g, events


def run(steps=120, size=60, p_mut=3.6e-6, seed=42):
    """p_mut here is PER-DIVISION (already aggregated across the genome),
    boosted for the CA so growth is visible on the simulation timescale.
    In main.py we feed in the calibrated rate and rescale for the CA clock."""
    rng = np.random.default_rng(seed)
    g = np.zeros((size, size), dtype=int)
    g[size // 2, size // 2] = NORMAL
    history = []
    for s in range(steps):
        g, ev = step(g, p_mut, seed=s)
        if s % 10 == 0:
            history.append((s, int((g == NORMAL).sum()), int((g == MUTANT).sum()),
                            int((g == MALIGNANT).sum()), ev))
    counts = {k: int((g == v).sum()) for k, v in
              [("normal", NORMAL), ("mutant", MUTANT), ("malignant", MALIGNANT)]}
    return dict(final=g, counts=counts, history=history,
                total_cells=int((g > 0).sum()))


if __name__ == "__main__":
    r = run()
    print("=== tumor_growth : cellular automaton (tissue scale) ===")
    print(f"final cell counts: {r['counts']}  (total = {r['total_cells']})")
    print("growth trajectory (step, normal, mutant, malignant):")
    for h in r["history"][::2]:
        print(f"   step {h[0]:3d}:  N={h[1]:5d}  M={h[2]:5d}  Mal={h[3]:5d}")
