# Prints the coordinate block that is pasted into the .tex of the same name.
import numpy as np, sys
def run(seed):
    rng = np.random.default_rng(seed)
    a = np.deg2rad(-28); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    C = R @ np.diag([.7**2, .3**2]) @ R.T
    m1 = np.array([3.15, 3.7]); m0 = np.array([2.5, 2.15])
    X1 = rng.multivariate_normal(m1, C, 8); X0 = rng.multivariate_normal(m0, C, 8)
    Sw = np.cov(X0.T) + np.cov(X1.T)
    w = np.linalg.solve(Sw, X1.mean(0) - X0.mean(0)); w /= np.linalg.norm(w)
    if w[0] < 0: w = -w
    return X1, X0, np.degrees(np.arctan2(w[1], w[0]))
def t(X, deg): return X @ [np.cos(np.radians(deg)), np.sin(np.radians(deg))]
if len(sys.argv) > 1:
    for seed in range(60):
        X1, X0, g = run(seed); A = np.vstack([X1, X0])
        if A.min() < 1.1 or A.max() > 4.7: continue
        gap = t(X1, g).min() - t(X0, g).max()
        ov = min(t(X1, 4).max(), t(X0, 4).max()) - max(t(X1, 4).min(), t(X0, 4).min())
        sp = np.sort(t(A, g)); mind = np.diff(sp).min()
        print(seed, round(g, 1), round(gap, 2), round(ov, 2), round(mind, 3))
    sys.exit()
SEED = 43
X1, X0, good = run(SEED)
fmt = lambda X: ", ".join(f"{x:.2f}/{y:.2f}" for x, y in X)
print("\\def\\Pa{" + fmt(X1) + "}")
print("\\def\\Pd{" + fmt(X0) + "}")
print(f"\\def\\good{{{good:.2f}}}\\def\\bad{{4.00}}")
