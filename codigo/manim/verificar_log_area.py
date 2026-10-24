"""
Verificación numérica de log_area.py, antes de animar.

    python verificar_log_area.py

Chequea con la numérica y la geometría del archivo de la escena:
  - que (t, y) ↦ (a·t, y/a) lleve puntos de la hipérbola a la hipérbola (en los números y en
    los ejes de la escena, con p2c/c2p dentro de apply_function), y que el techo de la región
    siga sobre la curva en todo el recorrido del factor;
  - que el área de a a ab sea igual a la de 1 a b (quad, y shoelace de las celdas antes y después);
  - que la grilla tape la región sin huecos ni solapamientos;
  - que el punto donde el área vale 1 sea e, y que las regiones apiladas encajen;
  - potencias, inversos y la derivada 1/x.
"""
import numpy as np

from log_area import (A, B, DX, E, E2, T_MAX, T_MIN, area, celdas, contorno_region, derivada, ejes, hiperbola,
                      ln, mapa, medidas, poligono, region, shoelace, transformada, transformar_punto)

ok = True


def chequear(cond, msg):
    global ok
    print(("  ok   " if cond else "  FALLA ") + msg)
    ok &= bool(cond)


ax = ejes()
S = np.linspace(1, A, 41)         # el recorrido del factor en el acto 3

print("1. la transformación lleva la hipérbola en sí misma")
t = np.geomspace(0.05, 20, 400)
peor = max(np.max(np.abs(np.multiply(*mapa(t, 1 / t, s)) - 1)) for s in (0.25, 0.5, A, B, E, 7.3))
chequear(peor < 1e-14, f"en números: máx |t'·y' − 1| = {peor:.1e}")
ida = max(np.max(np.abs(ax.p2c(ax.c2p(tt, yy)) - [tt, yy])) for tt, yy in [(0, 0), (1, 1), (8, 3), (2.5, 0.4)])
chequear(ida < 1e-12, f"p2c(c2p(t, y)) = (t, y): máx error {ida:.1e}")
hip = hiperbola(ax)
peor = 0.0
for s in S[1:]:
    tr = transformada(ax, hip, s)
    q = np.array([ax.p2c(p) for p in tr.get_anchors()])
    peor = max(peor, np.max(np.abs(q[:, 0] * q[:, 1] - 1)))
chequear(peor < 1e-9, f"la curva dibujada, transformada en los ejes: máx |t·y − 1| en las anclas = {peor:.1e}")
techo = poligono(ax, contorno_region(1, B))
peor, extremos = 0.0, []
for s in S:
    q = np.array([ax.p2c(p) for p in transformada(ax, techo, s).get_vertices()])
    peor = max(peor, np.max(np.abs(q[2:, 0] * q[2:, 1] - 1)))
    extremos.append((q[:, 0].min(), q[:, 0].max()))
chequear(peor < 1e-9, f"techo de la región durante toda la transformación: máx |t·y − 1| = {peor:.1e}")
chequear(np.allclose(extremos[-1], (A, A * B), atol=1e-9),
         f"la región termina entre {extremos[-1][0]:.6f} y {extremos[-1][1]:.6f} (a = {A:g}, ab = {A * B:g})")

print("\n2. el área de a a ab es la de 1 a b")
a13, a26 = area(1, B), area(A, A * B)
chequear(abs(a13 - a26) < 1e-13, f"quad: ∫₁³ dt/t = {a13:.12f},  ∫₂⁶ dt/t = {a26:.12f}")
chequear(abs(a13 - np.log(B)) < 1e-13, f"y coincide con np.log({B:g}) = {np.log(B):.12f}")
datos = celdas(1, B)
polis = [poligono(ax, pts) for _, _, pts in datos]
antes = np.array([medidas(ax, p)[2] for p in polis])
despues = np.array([medidas(ax, transformada(ax, p, A))[2] for p in polis])
chequear(np.max(np.abs(antes - despues)) < 1e-12,
         f"{len(polis)} celdas: máx |área después − área antes| = {np.max(np.abs(antes - despues)):.1e}")
chequear(np.allclose([shoelace(pts) for _, _, pts in datos], antes, atol=1e-12),
         "el área de cada celda medida en la escena (p2c) es la de sus vértices (t, y)")
reg = np.array([medidas(ax, transformada(ax, techo, s))[2] for s in S])
chequear(np.ptp(reg) < 1e-12, f"área del contorno durante la transformación: {reg[0]:.6f} (varía {np.ptp(reg):.1e})")
print(f"   el contorno tiene {reg[0]:.6f} vs ln 3 = {a13:.6f}: se muestra {reg[0]:.3f} = {a13:.3f}")
chequear(f"{reg[0]:.3f}" == f"{a13:.3f}", "con 3 decimales el polígono y la integral dicen lo mismo")
marcada = next(pts for i, k, pts in datos if (i, k) == (4, 2))
w, h, ar = medidas(ax, poligono(ax, marcada))
w2, h2, ar2 = medidas(ax, transformada(ax, poligono(ax, marcada), A))
chequear(np.allclose([w2, h2, ar2], [A * w, h / A, ar], atol=1e-12),
         f"celda marcada: {w:.4f} × {h:.4f} = {ar:.6f}  →  {w2:.4f} × {h2:.4f} = {ar2:.6f}")

print("\n3. la grilla tapa la región sin huecos ni solapamientos")
suma = antes.sum()
chequear(abs(suma - a13) < 5e-5, f"suma de las celdas = {suma:.6f}  vs  ∫ = {a13:.6f} (diferencia {suma - a13:.1e},"
                                 " la cuerda bajo la curva)")
caja = lambda pts: (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts))
choques = 0
for j, (_, _, p) in enumerate(datos):
    a0, a1, b0, b1 = caja(p)
    for _, _, q in datos[j + 1:]:
        c0, c1, d0, d1 = caja(q)
        if min(a1, c1) - max(a0, c0) > 1e-12 and min(b1, d1) - max(b0, d0) > 1e-12:
            choques += 1
chequear(choques == 0, f"cajas de celdas que se superponen: {choques}")
fuera = max(max(p[1] - 1 / p[0] for p in pts) for _, _, pts in datos)
chequear(fuera < 1e-12, f"ningún vértice por encima de la curva (máx y − 1/t = {fuera:.1e})")

print("\n4. e: donde el área vale 1, y las regiones apiladas")
chequear(abs(E - np.e) < 1e-12, f"brentq(área = 1) = {E:.15f},  np.e = {np.e:.15f}")
chequear(abs(ln(E) - 1) < 1e-13, f"∫₁ᵉ dt/t = {ln(E):.15f}")
chequear(abs(E2 - E * E) < 1e-11 and abs(E2 - np.e ** 2) < 1e-11, f"siguiente área 1: {E2:.12f};  e² = {E * E:.12f}")
r1 = region(ax, 1, E, "#ffffff")
r2 = transformada(ax, r1, E)
q1 = np.array([ax.p2c(p) for p in r1.get_vertices()])
q2 = np.array([ax.p2c(p) for p in r2.get_vertices()])
chequear(abs(q2[:, 0].min() - q1[:, 0].max()) < 1e-9 and abs(q2[:, 0].max() - E2) < 1e-9,
         f"la copia transformada va de {q2[:, 0].min():.9f} a {q2[:, 0].max():.9f}: encaja sin hueco")
chequear(abs(shoelace(q1) - shoelace(q2)) < 1e-12, f"y tiene la misma área ({shoelace(q2):.6f})")
print(f"   la animación frena en x = {E:.6f}: área mostrada {ln(E):.3f}, falta {max(0.0, 1 - ln(E)):.3f}")

print("\n5. potencias e inversos")
rA = region(ax, 1, A, "#ffffff")
r_2, r_3 = transformada(ax, rA, A), transformada(ax, transformada(ax, rA, A), A)
for nombre, r, lo, hi in (("a → a²", r_2, A, A ** 2), ("a² → a³", r_3, A ** 2, A ** 3)):
    q = np.array([ax.p2c(p) for p in r.get_vertices()])
    chequear(abs(q[:, 0].min() - lo) < 1e-9 and abs(q[:, 0].max() - hi) < 1e-9 and abs(shoelace(q) - shoelace(
        np.array([ax.p2c(p) for p in rA.get_vertices()]))) < 1e-12, f"{nombre}: de {q[:, 0].min():.6f} a "
                                                                    f"{q[:, 0].max():.6f}, área {shoelace(q):.6f}")
chequear(abs(ln(A ** 3) - 3 * ln(A)) < 1e-13, f"ln(a³) = {ln(A ** 3):.6f} = 3·ln a = {3 * ln(A):.6f}")
chequear(f"{ln(A ** 3):.3f}" == f"{3 * round(ln(A), 3):.3f}", f"en pantalla: 3 · {ln(A):.3f} = {ln(A ** 3):.3f}")
r_inv = transformada(ax, rA, 1 / A)
q = np.array([ax.p2c(p) for p in r_inv.get_vertices()])
chequear(abs(q[:, 0].min() - 1 / A) < 1e-9 and abs(q[:, 0].max() - 1) < 1e-9,
         f"inversa: de {q[:, 0].min():.6f} a {q[:, 0].max():.6f}")
chequear(abs(ln(1 / A) + ln(A)) < 1e-13, f"ln(1/a) = {ln(1 / A):.6f} = −ln a")
chequear(f"{ln(A * B):.3f}" == f"{round(ln(A), 3) + round(ln(B), 3):.3f}",
         f"en pantalla: {ln(A):.3f} + {ln(B):.3f} = {ln(A * B):.3f}")

print("\n6. derivada: la pendiente de ln es 1/x")
xs = np.linspace(0.4, 7.5, 30)
d = max(abs(derivada(ln, v) - 1 / v) for v in xs)
chequear(d < 1e-6, f"máx |pendiente numérica − 1/x| en [0.4, 7.5] = {d:.1e}")
d = max(abs(ln(v) - np.log(v)) for v in xs)
chequear(d < 1e-13, f"máx |área − np.log| (la altura del gráfico) = {d:.1e}")
print(f"   franja en x = 2: área {area(2, 2 + DX):.4f} vs dx/x = {DX / 2:.4f}")

print("\n7. la hipérbola dibujada cabe en los ejes")
chequear(abs(1 / T_MIN - 3) < 1e-12 and T_MAX == 8, f"t de {T_MIN:.4f} (y = {1 / T_MIN:g}) a {T_MAX:g}")

print("\nTODO OK" if ok else "\nHAY FALLAS")
