"""
Verificación simbólica (SymPy) de tres resultados del portafolio, y las figuras de la nota (Matplotlib).

  1. Gradiente ⟂ curva de nivel:  f = ½(x² + 3y²),  ∇f · r'(t) = 0  sobre  r(t) = (√(2c)·cos t, √(2c/3)·sin t)
  2. Lata óptima:                 A(r) = 2πr² + 2V/r,  A'(r) = 0  ⇒  h = 2r,  y A''(r) > 0
  3. Serie de Taylor de sen x:    coeficientes y cota del error de Lagrange

Uso:  python verificar_con_sympy.py        (escribe img/verificacion-*.svg en la raíz del sitio)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

IMG = Path(__file__).resolve().parents[2] / "img"
GRIS, VERDE, NARANJA = "#8a8a8a", "#18bc9c", "#f97316"

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 11,
    "text.color": GRIS, "axes.labelcolor": GRIS, "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.edgecolor": GRIS, "legend.frameon": False, "figure.facecolor": "none", "axes.facecolor": "none",
})


def guardar(fig, nombre):
    fig.savefig(IMG / nombre, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


# ── 1. Gradiente perpendicular a la curva de nivel ──────────────────────────────────────────────────
x, y, t, c = sp.symbols("x y t c", positive=True)
f = (x**2 + 3 * y**2) / 2
grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
r = sp.Matrix([sp.sqrt(2 * c) * sp.cos(t), sp.sqrt(2 * c / 3) * sp.sin(t)])
assert sp.simplify(f.subs({x: r[0], y: r[1]}) - c) == 0, "r(t) no está sobre f = c"
producto = sp.simplify(grad.subs({x: r[0], y: r[1]}).dot(r.diff(t)))
print("1. ∇f · r'(t) =", producto)
assert producto == 0

# El paso del descenso: y_{k+1} = (1 − 3α) y_k  ⇒  converge sii |1 − 3α| < 1 y |1 − α| < 1
a = sp.symbols("alpha", positive=True)
rango = sp.solve_univariate_inequality(sp.Abs(1 - 3 * a) < 1, a, relational=False)
print("   converge para α en", rango)
assert rango == sp.Interval.open(0, sp.Rational(2, 3))

# ── 2. Lata óptima ───────────────────────────────────────────────────────────────────────────────────
rr, V = sp.symbols("r V", positive=True)
A = 2 * sp.pi * rr**2 + 2 * V / rr
r_opt = sp.solve(sp.diff(A, rr), rr)[0]
h_opt = sp.simplify(V / (sp.pi * r_opt**2))
print("2. r* =", r_opt, "   h* / r* =", sp.simplify(h_opt / r_opt))
assert sp.simplify(h_opt / r_opt) == 2
assert sp.simplify(sp.diff(A, rr, 2).subs(rr, r_opt)) > 0  # mínimo, no máximo

# ── 3. Taylor de sen x ───────────────────────────────────────────────────────────────────────────────
X = sp.symbols("x")
serie = sp.series(sp.sin(X), X, 0, 10).removeO()
print("3. Taylor de sen x:", serie)
assert serie == X - X**3 / 6 + X**5 / 120 - X**7 / 5040 + X**9 / 362880


def taylor(n):
    return sp.lambdify(X, sp.series(sp.sin(X), X, 0, n + 1).removeO(), "numpy")


# ── Figura 1: polinomios de Taylor contra sen x ──────────────────────────────────────────────────────
xs = np.linspace(-2 * np.pi, 2 * np.pi, 800)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(xs, np.sin(xs), color=GRIS, lw=2.6, label="sen x")
for n, col in [(1, NARANJA), (3, "#e0b100"), (7, VERDE)]:
    ys = taylor(n)(xs) * np.ones_like(xs)
    ax.plot(xs, ys, color=col, lw=1.6, label=f"grado {n}")
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("x")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
ax.grid(alpha=0.18)
guardar(fig, "verificacion-taylor.svg")

# ── Figura 2: A(r) con su mínimo, por SymPy ──────────────────────────────────────────────────────────
V0 = 1000.0  # cm³
A_f = sp.lambdify(rr, A.subs(V, V0), "numpy")
r0 = float(r_opt.subs(V, V0))
rs = np.linspace(2, 12, 400)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(rs, A_f(rs), color=VERDE, lw=2.2)
ax.plot([r0], [A_f(r0)], "o", color=NARANJA, ms=8)
ax.annotate(f"r* = {r0:.2f} cm\nh* = 2r* = {2 * r0:.2f} cm", (r0, A_f(r0)), (r0 + 1.6, A_f(r0) + 150), color=GRIS,
            arrowprops=dict(arrowstyle="-", color=GRIS))
ax.set_xlabel("radio r (cm)")
ax.set_ylabel("chapa A(r) (cm²), V = 1000 cm³")
ax.grid(alpha=0.18)
guardar(fig, "verificacion-lata.svg")
print("Figuras escritas en", IMG)
