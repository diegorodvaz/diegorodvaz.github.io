"""
Verificaciones numéricas para lata_optima.py (entre los cilindros de volumen V, el de menos chapa).

    h = V/(πr²)        A(r) = 2πr² + 2V/r        A'(r) = 4πr − 2V/r²
    A'(r) = 0  ⇒  r³ = V/(2π)  ⇒  h = 2r

Con tapas k veces más caras por cm²: C(r) = k·2πr² + 2V/r  ⇒  r³ = V/(2πk)  ⇒  h = 2k·r.
Una lata con altura/diámetro = q gasta (1 + 2q)/(3·q^(2/3)) veces la chapa del óptimo.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def h_de(r, V):
    return V / (np.pi * r ** 2)


def tapas(r, k=1.0):
    return k * 2 * np.pi * r ** 2


def pared(r, V):
    return 2 * np.pi * r * h_de(r, V)


def costo(r, V, k=1.0):
    return tapas(r, k) + pared(r, V)


def minimo(V, k=1.0):
    """El r del mínimo, buscado numéricamente (no con la fórmula)."""
    r_f = (V / (2 * np.pi * k)) ** (1 / 3)
    res = minimize_scalar(lambda r: costo(r, V, k), bounds=(r_f / 10, r_f * 10), method="bounded",
                          options={"xatol": 1e-12})
    return res.x


def r_con_proporcion(q, V):
    """Radio del cilindro de volumen V con altura/diámetro = q: πr²·2qr = V."""
    return (V / (2 * np.pi * q)) ** (1 / 3)


if __name__ == "__main__":
    for V in (2 * np.pi, 1.0, 355.0):
        r = minimo(V)
        h = h_de(r, V)
        print(f"V = {V:8.3f}: r* = {r:.6f}  h* = {h:.6f}  h/r = {h / r:.8f}  "
              f"pared/tapas = {pared(r, V) / tapas(r):.8f}  fórmula r = {(V / (2 * np.pi)) ** (1 / 3):.6f}")
        assert abs(h / r - 2) < 1e-5
        assert abs(pared(r, V) / tapas(r) - 2) < 1e-5
        assert abs(pared(r, V) - 4 * np.pi * r ** 2) < 1e-6 * pared(r, V)

    print("\nTapas k veces más caras: h = 2k·r")
    for k in (1.0, 1.25, 1.5, 2.0, 3.0):
        for V in (2 * np.pi, 355.0):
            r = minimo(V, k)
            h = h_de(r, V)
            print(f"  k = {k:4.2f}  V = {V:7.2f}: r = {r:.5f}  h = {h:.5f}  h/(2r) = {h / (2 * r):.6f}")
            assert abs(h / (2 * r) - k) < 1e-5

    print("\nLata real (V = 355 cm³)")
    V = 355.0
    r = minimo(V)
    A0 = costo(r, V)
    print(f"  óptima: r = {r:.3f} cm  d = {2 * r:.3f} cm  h = {h_de(r, V):.3f} cm  A = {A0:.2f} cm²")
    for q in (1.8, 1.85, 1.9):
        rq = r_con_proporcion(q, V)
        hq = h_de(rq, V)
        Aq = costo(rq, V)
        extra = Aq / A0 - 1
        teo = (1 + 2 * q) / (3 * q ** (2 / 3)) - 1
        print(f"  q = {q:.2f}: r = {rq:.3f} cm  h = {hq:.3f} cm  h/d = {hq / (2 * rq):.4f}  "
              f"A = {Aq:.2f} cm²  extra = {100 * extra:.3f} %  (fórmula {100 * teo:.3f} %)")
        assert abs(hq / (2 * rq) - q) < 1e-9 and abs(extra - teo) < 1e-9
        # el mismo porcentaje en las unidades normalizadas de la animación (V = 2π)
        assert abs(costo(r_con_proporcion(q, 2 * np.pi), 2 * np.pi) / costo(1.0, 2 * np.pi) - 1 - extra) < 1e-9

    print("\nRango de la animación (V = 2π, escala fija)")
    for r in (0.62, 0.8, 1.0, 2.2):
        h = h_de(r, 2 * np.pi)
        print(f"  r = {r:.2f}: h = {h:.3f}  h/d = {h / (2 * r):.3f}  tapas = {tapas(r):6.2f}  "
              f"pared = {pared(r, 2 * np.pi):6.2f}  A = {costo(r, 2 * np.pi):6.2f}")
    print("\nTodo OK")
