"""
Figuras de la nota "Taylor contra mínimos cuadrados" (escribe img/taylor-vs-ajuste-*.svg) y la tabla de errores.

Uso:  python figuras.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ajuste import A, B, ajustar, datos, errores, predecir, taylor

IMG = Path(__file__).resolve().parents[2] / "img"
GRIS, VERDE, NARANJA, AMARILLO = "#8a8a8a", "#18bc9c", "#f97316", "#e0b100"
N, RUIDO, SEMILLA = 20, 0.25, 0

sns.set_theme(style="ticks", rc={
    "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 11,
    "text.color": GRIS, "axes.labelcolor": GRIS, "xtick.color": GRIS, "ytick.color": GRIS,
    "axes.edgecolor": GRIS, "legend.frameon": False, "figure.facecolor": "none", "axes.facecolor": "none",
    "grid.color": GRIS, "grid.alpha": 0.18, "axes.grid": True,
})


def guardar(fig, nombre):
    fig.savefig(IMG / nombre, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


x, y = datos(N, RUIDO, SEMILLA)
xs = np.linspace(A, B, 600)

# ── Figura 1: los puntos, sen x y dos ajustes ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(xs, np.sin(xs), color=GRIS, lw=2.6, label="sen x")
ax.scatter(x, y, color=GRIS, s=22, alpha=0.8, label=f"{N} puntos con ruido", zorder=3)
ax.plot(xs, predecir(ajustar(x, y, 3), xs), color=VERDE, lw=1.8, label="mínimos cuadrados, grado 3")
ax.plot(xs, np.clip(predecir(ajustar(x, y, 12), xs), -2.2, 2.2), color=NARANJA, lw=1.8, label="mínimos cuadrados, grado 12")
ax.plot(xs, taylor(xs, 5), color=AMARILLO, lw=1.6, ls="--", label="Taylor, grado 5")
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("x")
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=9.5)
guardar(fig, "taylor-vs-ajuste-curvas.svg")

# ── Figura 2: error contra grado (DataFrame largo → seaborn) ────────────────────────────────────────
df = errores(x, y, max_grado=12)
fig, ax = plt.subplots(figsize=(7, 3.8))
sns.lineplot(data=df, x="grado", y="rmse", hue="conjunto", style="conjunto", markers=True, dashes=False,
             palette=[GRIS, NARANJA, AMARILLO], ax=ax)
ax.set_yscale("log")
ax.set_xticks(range(1, 13))
ax.set_ylabel("error RMSE (escala log)")
ax.legend(title=None, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, fontsize=9.5)
guardar(fig, "taylor-vs-ajuste-error.svg")

tabla = df.pivot(index="conjunto", columns="grado", values="rmse").round(3)
print(tabla.to_string())
prueba = df[df.conjunto == "Prueba (sen x exacto)"].set_index("grado").rmse
print(f"\nMejor grado de mínimos cuadrados (error de prueba): {prueba.idxmin()}  (RMSE {prueba.min():.3f})")
print(f"Error de prueba en grado 12: {prueba[12]:.3f}")
print("Figuras escritas en", IMG)
