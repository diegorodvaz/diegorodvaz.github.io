"""
Verificación numérica de senx_sobre_x.py, antes de animar.

    python verificar_senx_sobre_x.py

Chequea con la geometría del archivo de la escena (vértices, contornos, shoelace, Figura):
  - que sen x / 2 ≤ x / 2 ≤ tg x / 2 y que cos x ≤ sen x / x ≤ 1 en todo (0, π/2);
  - que T esté sobre la recta OP y sobre la tangente en A (en números y en la pantalla);
  - que los polígonos dibujados tengan las áreas sen x / 2, x / 2, tg x / 2 y estén uno dentro del otro,
    y que las dos franjas sean exactamente lo que separa una región de la siguiente;
  - que en la figura estirada (y en la reflejada) los cocientes de áreas sean los de la original,
    también medidos sobre los mobjects transformados con apply_function, y que el arco estirado sea
    la imagen del arco real (una elipse);
  - el gráfico del sándwich, la paridad y que las barras y los paneles quepan en el cuadro.
"""
import numpy as np
from manim import ValueTracker

from senx_sobre_x import (FH, FW, L_BARRA, N_ARCO, O_EST, O_FIG, O_PAR, O_REAL, R_FIG, R_PANEL, R_PAR, V_MAX, X0,
                          X_BARRA, X_FIN, Y_PANEL, Y_TIT, Figura, areas, cocientes, contorno, factor, mapa_vertical,
                          shoelace, sinc, vertices)

ok = True


def chequear(cond, msg):
    global ok
    print(("  ok   " if cond else "  FALLA ") + msg)
    ok &= bool(cond)


def adentro(pts, poli, tol=1e-12):
    """¿Todos los pts dentro (o en el borde) del polígono convexo antihorario `poli`?"""
    poli = np.asarray(poli)
    peor = np.inf
    for a, b in zip(poli, np.roll(poli, -1, axis=0)):
        if np.hypot(*(b - a)) < 1e-15:
            continue
        for p in np.asarray(pts):
            peor = min(peor, (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    return peor >= -tol, peor


def convexo(poli, tol=1e-12):
    p = np.asarray(poli)
    d1, d2 = np.roll(p, -1, axis=0) - p, np.roll(p, -2, axis=0) - np.roll(p, -1, axis=0)
    cruz = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return np.all(cruz >= -tol)


def a_figura(fig, puntos3d, k=1.0, o=None):
    """Puntos de la pantalla → coordenadas (u, v) de la figura (deshaciendo un estiramiento k con origen o)."""
    o = fig.O() if o is None else np.asarray(o)
    q = (np.asarray(puntos3d)[:, :2] - o[:2]) / fig.R()
    q[:, 1] /= k
    return q


XS = np.linspace(1e-3, np.pi / 2 - 1e-3, 3000)
MUESTRA = [X_FIN, 0.05, 0.1, 0.3, 0.5, X0, 1.2, 1.5]

print("1. las desigualdades en (0, π/2)")
s, c, t = np.sin(XS), np.cos(XS), np.tan(XS)
chequear(np.all(s / 2 < XS / 2) and np.all(XS / 2 < t / 2),
         f"sen x/2 < x/2 < tg x/2 en {len(XS)} puntos (márgenes mín. {np.min(XS - s) / 2:.1e} y {np.min(t - XS) / 2:.1e})")
chequear(np.all(c < s / XS) and np.all(s / XS < 1), "cos x < sen x / x < 1 en los mismos puntos")
chequear(np.allclose(np.cos(XS) * (XS / np.sin(XS)) <= 1 + 1e-15, True), "y al revés: 1 ≤ x/sen x ≤ 1/cos x")

print("\n2. T sobre la recta OP y sobre la tangente en A")
peor_recta = max(abs(vertices(x)["P"][0] * vertices(x)["T"][1] - vertices(x)["P"][1] * vertices(x)["T"][0])
                 for x in XS)
peor_tang = max(abs(vertices(x)["T"][0] - 1.0) for x in XS)
chequear(peor_recta < 1e-12, f"det(P, T) = 0 (O, P, T alineados): máx {peor_recta:.1e}")
chequear(peor_tang < 1e-15, f"T tiene u = 1 (la tangente en A es perpendicular a OA): máx {peor_tang:.1e}")
xt = ValueTracker(X0)
for o, r in ((O_FIG, R_FIG), (O_REAL, R_PANEL), (O_PAR, R_PAR)):
    fig = Figura(xt, o, r)
    peor = 0.0
    for x in MUESTRA:
        xt.set_value(x)
        O, A, P, T = (fig.pt(n) for n in "OAPT")
        d1, d2 = P - O, T - O
        peor = max(peor, abs(d1[0] * d2[1] - d1[1] * d2[0]) / (np.linalg.norm(d1) * np.linalg.norm(d2)),
                   abs(T[0] - A[0]), abs(np.dot(T - A, A - O)) / r ** 2)
    chequear(peor < 1e-12, f"en pantalla (O = {o}, R = {r}): máx desvío {peor:.1e}")
xt.set_value(X0)

print("\n3. áreas de los polígonos y encierro")
for x in MUESTRA:
    a = areas(x)
    err = abs(a["sector"] - x / 2) / (x / 2)
    cota = (x / (N_ARCO - 1)) ** 2 / 6 * 1.01 + 1e-14
    chequear(abs(a["chico"] - np.sin(x) / 2) < 1e-15 and abs(a["grande"] - np.tan(x) / 2) < 1e-13 and err <= cota,
             f"x = {x:<5}: {a['chico']:.6f} = sen x/2,  {a['sector']:.6f} ≈ x/2 (rel. {err:.1e} ≤ {cota:.1e}),"
             f"  {a['grande']:.6f} = tg x/2")
    chico, sector, grande = (np.array(contorno(k, x)) for k in ("chico", "sector", "grande"))
    c1, m1 = adentro(chico, sector)
    c2, m2 = adentro(sector, grande)
    chequear(convexo(sector) and convexo(grande) and c1 and c2,
             f"         chico ⊂ sector ⊂ grande (márgenes {m1:.1e}, {m2:.1e})")
    seg, cuna = shoelace(contorno("segmento", x)), shoelace(contorno("cuña", x))
    chequear(abs(seg - (a["sector"] - a["chico"])) < 1e-13 * a["sector"] and abs(cuna - (a["grande"] - a["sector"])) < 1e-13 * a["grande"],
             f"         franjas: segmento {seg:.6f} = sector − chico, cuña {cuna:.6f} = grande − sector")
fig = Figura(xt, O_FIG, R_FIG)
peor = 0.0
for k in ("chico", "sector", "grande"):
    q = a_figura(fig, fig.region(k).get_vertices())
    peor = max(peor, abs(shoelace(q) - areas(X0)[k]))
chequear(peor < 1e-12, f"los polígonos dibujados (acto 2–5) miden lo mismo en coordenadas de la figura: {peor:.1e}")

print("\n4. figura estirada y reflejada: los cocientes no cambian")
peor = 0.0
for x in MUESTRA:
    base = np.array(cocientes(x))
    for k in (1.0, 1.7, factor(x), -1.0):
        peor = max(peor, np.max(np.abs(np.array(cocientes(x, k)) - base)))
chequear(peor < 1e-12, f"cocientes con (u, v) ↦ (u, k·v), k ∈ {{1, 1.7, 1/sen x, −1}}: máx diferencia {peor:.1e}")
fig = Figura(xt, O_REAL, R_PANEL)
o2 = np.array([O_EST[0], O_EST[1], 0.0])
peor_c, peor_a, peor_arco = 0.0, 0.0, 0.0
for x in MUESTRA[:6]:
    xt.set_value(x)
    k = factor(x)
    m = mapa_vertical(fig.O(), o2, k)
    est = {r: shoelace(a_figura(fig, fig.region(r).apply_function(m).get_vertices(), o=o2)) for r in
           ("chico", "sector", "grande")}
    real = {r: shoelace(a_figura(fig, fig.region(r).get_vertices())) for r in est}
    peor_c = max(peor_c, abs(est["chico"] / est["sector"] - real["chico"] / real["sector"]),
                 abs(est["grande"] / est["sector"] - real["grande"] / real["sector"]))
    peor_a = max(peor_a, max(abs(est[r] - k * real[r]) / real[r] for r in est))
    arc = fig.circ(0.0, float(np.arcsin(min(1.0, V_MAX / k)))).apply_function(m)
    q = a_figura(fig, arc.get_anchors(), o=o2)
    peor_arco = max(peor_arco, np.max(np.abs(q[:, 0] ** 2 + (q[:, 1] / k) ** 2 - 1)))
    Pp, Tp = (a_figura(fig, [m(fig.pt(n))], o=o2)[0] for n in "PT")
    chequear(abs(Pp[1] - 1) < 1e-12 and abs(Tp[1] - 1 / np.cos(x)) < 1e-12 and abs(Pp[0] - np.cos(x)) < 1e-12,
             f"x = {x:<5}: estirada ×{k:7.3f}: P' = ({Pp[0]:.4f}, {Pp[1]:.4f}), T' = (1, {Tp[1]:.4f} = 1/cos x),"
             f"  cocientes {est['chico'] / est['sector']:.6f}, {est['grande'] / est['sector']:.6f}")
chequear(peor_c < 1e-10, f"sobre los mobjects transformados (apply_function): cocientes iguales a los reales ({peor_c:.1e})")
chequear(peor_a < 1e-10, f"y cada área queda multiplicada por el mismo factor k ({peor_a:.1e})")
chequear(peor_arco < 1e-5, f"el arco estirado es la imagen del arco: u² + (v/k)² = 1 en las anclas ({peor_arco:.1e})")
v = vertices(X0)
vn = vertices(-X0)
chequear(np.allclose([vn["P"], vn["T"]], [(v["P"][0], -v["P"][1]), (v["T"][0], -v["T"][1])], atol=1e-15),
         "la figura de −x es la de x reflejada: P(−x) = (cos x, −sen x), T(−x) = (1, −tg x)")
xt.set_value(X0)

print("\n5. se pegan: los cocientes tienden a 1")
for x in (X0, 0.5, 0.2, 0.1, X_FIN):
    r = cocientes(x)
    print(f"   x = {x:<5}  chico/sector = {r[0]:.6f} (sen x/x = {sinc(x):.6f})   grande/sector = {r[2]:.6f}"
          f" (tg x/x = {np.tan(x) / x:.6f})   cos x = {np.cos(x):.6f}   ×{factor(x):.2f}")
r = cocientes(X_FIN)
chequear(abs(r[0] - 1) < 1e-4 and abs(r[2] - 1) < 2e-4, f"en x = {X_FIN}: {r[0]:.6f} y {r[2]:.6f}")
chequear(abs(r[0] - sinc(X_FIN)) < 1e-8 and abs(r[2] - np.tan(X_FIN) / X_FIN) < 1e-8,
         "las barras (shoelace) y los números (sen x / x, tg x / x) coinciden")
print(f"   en pantalla, x = {X0}: cos x = {np.cos(X0):.4f} ≤ sen x/x = {sinc(X0):.4f} ≤ 1;"
      f"  x = {X_FIN}: {np.cos(X_FIN):.4f} ≤ {sinc(X_FIN):.4f} ≤ 1")
chequear(f"{np.cos(X_FIN):.4f}" < f"{sinc(X_FIN):.4f}" < "1.0000", "con 4 decimales el sándwich sigue viéndose estricto")
print(f"   la astilla real en x = {X_FIN}: alto {R_PANEL * np.tan(X_FIN):.3f} y ancho {R_PANEL * (1 - np.cos(X_FIN)):.5f}"
      " en pantalla")

print("\n6. el gráfico del sándwich y la paridad")
ts = np.concatenate([np.linspace(-np.pi / 2, -1e-6, 2000), np.linspace(1e-6, np.pi / 2, 2000)])
chequear(np.all(np.cos(ts) <= sinc(ts)) and np.all(sinc(ts) <= 1), "cos t ≤ sen t / t ≤ 1 en [−π/2, π/2] sin el 0")
chequear(np.max(np.abs(sinc(ts) - sinc(-ts))) < 1e-15, "sen t / t es par")
chequear(abs(sinc(1e-8) - 1) < 1e-15 and np.cos(0) == 1, "cerca de 0: sen t / t → 1 y cos 0 = 1")

print("\n7. todo cabe en el cuadro")
fin = X_BARRA + L_BARRA * max(cocientes(x)[2] for x in np.linspace(X_FIN, X0, 50))
chequear(fin < FW / 2 - 0.2, f"la barra más larga termina en {fin:.2f} (borde {FW / 2:.2f})")
alto_est = O_EST[1] + R_PANEL / np.cos(X0)
techo = O_EST[1] + R_PANEL * V_MAX
chequear(alto_est < techo < Y_PANEL[1] - 0.9, f"panel estirado: T' llega a {alto_est:.2f}, el recorte a {techo:.2f}"
                                              f" (techo del panel {Y_PANEL[1]})")
chequear(O_REAL[0] - 0.45 > -FW / 2 + 0.15 and O_EST[0] + R_PANEL + 0.45 <= 0,
         f"paneles entre {O_REAL[0] - 0.45:.2f} y {O_EST[0] + R_PANEL + 0.45:.2f}")
tope = O_FIG[1] + R_FIG * 1.42
chequear(tope < Y_TIT - 0.4, f"figura grande: la tangente llega a {tope:.2f}, T a {O_FIG[1] + R_FIG * np.tan(X0):.2f}")
chequear(O_PAR[1] + R_PAR * 1.42 < Y_TIT - 0.4 and O_PAR[1] - R_PAR * 1.42 > -FH / 2 + 0.6,
         f"figura con su reflejo: de {O_PAR[1] - R_PAR * 1.42:.2f} a {O_PAR[1] + R_PAR * 1.42:.2f}")

print("\nTODO OK" if ok else "\nHAY FALLAS")
