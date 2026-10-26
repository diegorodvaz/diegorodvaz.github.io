"""
La lata óptima: entre todos los cilindros de volumen V fijo, el que usa menos chapa.

    V = πr²h  ⇒  h = V/(πr²)        A(r) = 2πr² + 2πrh = 2πr² + 2V/r
    A'(r) = 4πr − 2V/r² = 0  ⇒  r³ = V/(2π)  ⇒  h = 2r

En el óptimo la altura es igual al diámetro (el perfil es un cuadrado) y la pared (4πr²) usa el doble que
las dos tapas (2πr²). Si la tapa cuesta k veces más por cm², C(r) = k·2πr² + 2V/r y el óptimo da h = 2k·r.

La animación usa V = 2π (r* = 1, h* = 2); el acto de la lata real usa V = 355 cm³. Un único ValueTracker r
maneja la lata, las barras, los puntos y la línea vertical del gráfico, todos con always_redraw.

    manim -pql lata_optima.py LataOptima
    manim -pqh lata_optima.py LataOptima

`python lata_optima.py` verifica la numérica y que todo entre en cuadro (ver también verificar_lata_optima.py).
"""
from manim import *
import numpy as np

config.max_files_cached = 1000


# ---------------------------------------------------------------------------
# Numérica
# ---------------------------------------------------------------------------
V_N = 2 * np.pi             # volumen en unidades normalizadas: r* = 1, h* = 2
V_CM3 = 355.0               # una lata real, en cm³
Q_REAL = 1.85               # altura/diámetro de una lata real
R_ACTO2 = 0.8               # la lata "normal" que se desarma
R_MIN, R_MAX = 0.62, 2.2    # tubo finito y disco chato: a escala fija entran los dos
R_MEDIO = 1.45
K_MAX = 2.0                 # hasta dónde sube el costo relativo de las tapas


def h_de(r, V=V_N):
    return V / (np.pi * r ** 2)


def tapas(r, k=1.0):
    return k * 2 * np.pi * r ** 2


def pared(r, V=V_N):
    return 2 * np.pi * r * h_de(r, V)        # = 2V/r


def costo(r, k=1.0, V=V_N):
    return tapas(r, k) + pared(r, V)


def r_optimo(k=1.0, V=V_N):
    """C'(r) = 4πk·r − 2V/r² = 0."""
    return (V / (2 * np.pi * k)) ** (1 / 3)


def r_con_proporcion(q, V=V_N):
    """El cilindro de volumen V con altura/diámetro = q: πr²·(2qr) = V."""
    return (V / (2 * np.pi * q)) ** (1 / 3)


R_OPT = r_optimo()
R_REAL = r_con_proporcion(Q_REAL)
EXTRA_REAL = costo(R_REAL) / costo(R_OPT) - 1       # no depende de V


# ---------------------------------------------------------------------------
# Paleta: tapas, pared y total con el mismo color en la lata, la chapa, las curvas, las barras y las fórmulas
# ---------------------------------------------------------------------------
FONDO = "#0e1117"
config.background_color = FONDO
FW, FH = config.frame_width, config.frame_height

C_TAPA = "#f4a259"
C_TAPA_BORDE = "#ffd3a1"
C_TAPA_ARO = "#b8733a"
C_PARED = "#4cc9f0"
C_PARED_OSC = "#123d52"
C_PARED_CLARO = "#8fe3ff"
C_PARED_BORDE = "#b9efff"
C_TOTAL = "#f2f2f2"
C_OPT = "#2ee6a6"           # el óptimo: el cuadrado, "altura = diámetro"
C_REAL = "#ff6b8b"          # la lata real
C_EJE = "#7d87a3"
C_TEXTO = GREY_A
C_TENUE = "#8a93a8"


def _hex(c):
    return ManimColor(c).to_hex()[1:].upper()


PLANTILLA = TexTemplate()
PLANTILLA.add_to_preamble(
    r"\usepackage{xcolor}"
    + "".join(rf"\newcommand{{\{n}}}[1]{{\textcolor[HTML]{{{_hex(c)}}}{{#1}}}}"
              for n, c in (("cT", C_TAPA), ("cP", C_PARED), ("cO", C_OPT), ("cR", C_REAL))))
config.tex_template = PLANTILLA

# ---------------------------------------------------------------------------
# Layout (16:9, 14.2 × 8): la lata a la izquierda, el gráfico a la derecha
# ---------------------------------------------------------------------------
X_LATA, Y_BASE = -4.3, -1.75        # centro de la tapa de abajo
S_LATA, S_PERFIL = 0.9, 1.4         # unidades de pantalla por unidad de r
K_PERSP = 0.28                      # semieje menor / radio de las elipses
LUZ = -0.6                          # de dónde viene la luz (ángulo alrededor del eje de la lata)

O_GRAF = np.array([0.0, -3.2, 0.0])
R_EJE, Y_EJE = 2.4, 40.0
LX_GRAF, LY_GRAF = 6.0, 5.0
UX, UY = LX_GRAF / R_EJE, LY_GRAF / Y_EJE

X_FORM, Y_FORM = 3.3, 3.3           # la fórmula de A(r) arriba del gráfico
Y_LECT = 3.55                       # lecturas r, h, V arriba de la lata
Y_TECHO = 3.15                      # debajo de las lecturas: borde de arriba de las cuentas del acto 5
X_BARRA0, ESC_BARRA, ALTO_BARRA = -6.0, 0.12, 0.26
Y_BT, Y_BP = -2.8, -3.4

# acto 2: dónde quedan las piezas de la lata desarmada
P_TAPA1, P_TAPA2 = np.array([-5.6, 1.6, 0]), np.array([-5.6, -1.6, 0])
P_PARED = np.array([0.6, -1.2, 0])
VACIO = np.zeros((0, 3))


def g2p(r, y):
    """Coordenadas del gráfico (r, A) → pantalla."""
    return O_GRAF + np.array([r * UX, y * UY, 0.0])


# ---------------------------------------------------------------------------
# La lata: tapas como elipses, pared como franjas sombreadas que se pueden desenrollar
# ---------------------------------------------------------------------------
def elipse_pts(c, a, b, n=72):
    t = np.linspace(0, TAU, n + 1)
    return np.stack([c[0] + a * np.sin(t), c[1] - b * np.cos(t), np.zeros_like(t)], axis=1)


def poligono(pts, fill, opacidad=1.0, borde=None, ancho=0.0):
    m = VMobject()
    m.set_points_as_corners(pts)
    m.set_fill(fill, opacidad).set_stroke(borde or fill, ancho)
    return m


def poligonal(pts, color, ancho):
    m = VMobject()
    m.set_points_as_corners(pts)
    return m.set_fill(opacity=0).set_stroke(color, ancho)


def tapa(c, R, ry):
    """Disco de radio R visto con semieje menor ry (ry = R: de frente; ry = 0: de perfil, un segmento)."""
    cuerpo = poligono(elipse_pts(c, R, ry), C_TAPA, 1, C_TAPA_BORDE, 2.2)
    aro = poligonal(elipse_pts(c, 0.84 * R, 0.84 * ry), C_TAPA_ARO, 1.6)
    return VGroup(cuerpo, aro)


def pared_mob(base, R, H, k, u=0.0, n=48):
    """La pared de radio R y alto H (en pantalla) con su borde inferior delantero en base − k·R.
    u ∈ [0, 1] la desenrolla: la chapa se curva con radio R/(1 − u) manteniendo fijo el punto de adelante,
    hasta quedar plana (2πR × H) con u = 1. Franjas de atrás primero, más oscuras (se ve el interior)."""
    sig = np.linspace(-PI * R, PI * R, n + 1)
    if u > 1 - 1e-6:
        rho, phi = np.inf, np.zeros_like(sig)
        x, d = sig, np.zeros_like(sig)
        phi_fin = 0.0
    else:
        rho = R / (1 - u)
        phi = sig / rho
        x, d = rho * np.sin(phi), rho * (1 - np.cos(phi))
        phi_fin = PI * (1 - u)
    yb = -k * R + k * d                 # profundidad d → más arriba en la vista desde arriba
    yt = yb + H
    b = np.array([base[0], base[1], 0.0])
    franjas = []
    for i in range(n):
        fm = 0.5 * (phi[i] + phi[i + 1])
        delante = np.cos(fm) >= 0
        luz = max(np.cos(fm - LUZ), 0.0)
        tono = 0.30 + 0.70 * luz if delante else 0.08 + 0.12 * luz
        col = interpolate_color(ManimColor(C_PARED_OSC), ManimColor(C_PARED_CLARO), tono)
        pts = b + np.array([[x[i], yb[i], 0], [x[i + 1], yb[i + 1], 0], [x[i + 1], yt[i + 1], 0],
                            [x[i], yt[i], 0], [x[i], yb[i], 0]])
        franjas.append((0.5 * (d[i] + d[i + 1]), poligono(pts, col, 1, col, 1.0)))
    franjas.sort(key=lambda t: -t[0])
    # contorno: borde de arriba entero, borde de abajo sólo adelante, y los costados
    borde_sup = poligonal(b + np.stack([x, yt, 0 * x], axis=1), C_PARED_BORDE, 2)
    fs = min(PI / 2, phi_fin) if np.isfinite(rho) else 0.0
    if np.isfinite(rho):
        ff = np.linspace(-fs, fs, 40)
        xf, df = rho * np.sin(ff), rho * (1 - np.cos(ff))
        xs = rho * np.sin(fs)
        ds = rho * (1 - np.cos(fs))
    else:
        xf, df = sig, np.zeros_like(sig)
        xs, ds = PI * R, 0.0
    borde_inf = poligonal(b + np.stack([xf, -k * R + k * df, 0 * xf], axis=1), C_PARED_BORDE, 2)
    y0 = -k * R + k * ds
    costados = VGroup(*[poligonal(b + np.array([[s * xs, y0, 0], [s * xs, y0 + H, 0]]), C_PARED_BORDE, 2)
                        for s in (-1, 1)])
    return VGroup(*[f for _, f in franjas], borde_sup, borde_inf, costados)


def lata(r, base, s=S_LATA, k=K_PERSP, h=None):
    """La lata de radio r (y altura h, por defecto la del volumen V_N) con la tapa de abajo centrada en base."""
    h = h_de(r) if h is None else h
    R, H = r * s, h * s
    base = np.asarray(base, dtype=float)
    return VGroup(tapa(base, R, k * R), pared_mob(base, R, H, k), tapa(base + UP * H, R, k * R))


def caja_lata(r, base, s=S_LATA, k=K_PERSP):
    """x0, x1, y0, y1 de la lata dibujada (para verificar el encuadre)."""
    R, H = r * s, h_de(r) * s
    return base[0] - R, base[0] + R, base[1] - k * R, base[1] + H + k * R


# ---------------------------------------------------------------------------
# Gráfico: curvas recortadas al techo del eje y líneas discontinuas
# ---------------------------------------------------------------------------
def bezier_de(tramos):
    """Puntos bezier (rectos) de varias poligonales; cada una queda como un subcamino."""
    partes = []
    for tr in tramos:
        tr = np.asarray(tr, dtype=float)
        if len(tr) < 2:
            continue
        a, b = tr[:-1], tr[1:]
        partes.append(np.stack([a, a + (b - a) / 3, b - (b - a) / 3, b], axis=1).reshape(-1, 3))
    return np.concatenate(partes) if partes else VACIO


def tramos_curva(f, a, b, ymax=Y_EJE, n=160):
    """La curva y = f(r) en [a, b], cortada donde pasa el techo del gráfico (en pantalla)."""
    if b - a < 1e-4:
        return []
    rs = np.linspace(a, b, n)
    ys = f(rs)
    tramos, actual = [], []
    for i in range(n):
        if ys[i] <= ymax:
            if not actual and i > 0:
                t = (ymax - ys[i - 1]) / (ys[i] - ys[i - 1])
                actual.append(g2p(rs[i - 1] + t * (rs[i] - rs[i - 1]), ymax))
            actual.append(g2p(rs[i], ys[i]))
        elif actual:
            t = (ymax - ys[i - 1]) / (ys[i] - ys[i - 1])
            actual.append(g2p(rs[i - 1] + t * (rs[i] - rs[i - 1]), ymax))
            tramos.append(actual)
            actual = []
    if actual:
        tramos.append(actual)
    return tramos


def curva(f, a, b, color, ancho):
    m = VMobject()
    pts = bezier_de(tramos_curva(f, a, b))
    if len(pts):
        m.set_points(pts)
    return m.set_fill(opacity=0).set_stroke(color, ancho)


def discontinua(a, b, color, ancho=2.5, raya=0.11, hueco=0.08):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    L = np.linalg.norm(b - a)
    m = VMobject()
    if L > 1e-6:
        u = (b - a) / L
        m.set_points(bezier_de([[a + s * u, a + min(s + raya, L) * u] for s in np.arange(0, L, raya + hueco)]))
    return m.set_fill(opacity=0).set_stroke(color, ancho)


def barra(y, valor, color):
    w = max(valor * ESC_BARRA, 1e-3)
    pts = np.array([[X_BARRA0, y - ALTO_BARRA / 2, 0], [X_BARRA0 + w, y - ALTO_BARRA / 2, 0],
                    [X_BARRA0 + w, y + ALTO_BARRA / 2, 0], [X_BARRA0, y + ALTO_BARRA / 2, 0],
                    [X_BARRA0, y - ALTO_BARRA / 2, 0]])
    return poligono(pts, color, 0.85, color, 1.5)


# ---------------------------------------------------------------------------
# Texto: Text para el castellano, MathTex para la matemática, con línea de base común
# ---------------------------------------------------------------------------
DESCENDENTES = "gjpqy,;()¿?"


def tex(t, font_size, color=None):
    """MathTex de t con su línea de base anotada (se compila con una x de referencia delante)."""
    m = MathTex("x", t, font_size=font_size)
    parte = m[1]
    if color is not None:
        parte.set_color(color)
    parte.base_rel = m[0].get_bottom()[1] - parte.get_bottom()[1]
    return parte


def linea_base(mob):
    if hasattr(mob, "base_rel"):
        return mob.get_bottom()[1] + mob.base_rel
    if isinstance(mob, Text):
        chars = [c for c in mob.text if not c.isspace()]
        glifos = [g for c, g in zip(chars, mob.submobjects) if c not in DESCENDENTES]
        if glifos:
            return np.median([g.get_bottom()[1] for g in glifos])
        ref = Text("xy", font_size=mob.font_size)
        return mob.get_bottom()[1] + ref[0].get_bottom()[1] - ref[1].get_bottom()[1]
    fondos = np.array([m.get_bottom()[1] for m in mob.family_members_with_points()])
    cerca = (np.abs(fondos[:, None] - fondos[None, :]) < 0.015).sum(axis=1)
    return fondos[cerca == cerca.max()].min()


def renglon(*partes, font_size=28, color=C_TEXTO, buff=0.14):
    """Mezcla Text y MathTex en fila: las partes que empiezan con '$' son TeX (sin color propio: los
    colores van con \\cT, \\cP, …; las que no tienen, quedan blancas)."""
    mobs = [p if isinstance(p, Mobject) else
            tex(p[1:], font_size * 1.25) if p.startswith("$")
            else Text(p, font_size=font_size, color=color) for p in partes]
    g = VGroup(*mobs).arrange(RIGHT, buff=buff)
    y = linea_base(mobs[0])
    for m in mobs[1:]:
        m.shift(UP * (y - linea_base(m)))
    return g


class Numero(VMobject):
    """Número que se rehace en cada cuadro copiando los puntos de cada cifra de una plantilla, todo en un
    solo VMobject (DecimalNumber.set_value es carísimo con varios números vivos). Cifras tabulares.
    Dos puntos invisibles marcan el extremo izquierdo de la línea de base y la escala."""
    _plantillas = {}
    EPS = 0.01

    def __init__(self, valor, decimales, font_size, color=WHITE):
        super().__init__()
        self.decimales = decimales
        self.cifras = self.plantilla(font_size)
        self.add(VectorizedPoint(ORIGIN), VectorizedPoint(RIGHT * self.EPS))
        self.set_fill(color, 1).set_stroke(width=0)
        self.poner(valor)

    @classmethod
    def plantilla(cls, font_size):
        if font_size not in cls._plantillas:
            chars = "0123456789.-"
            m = MathTex(*chars, font_size=font_size)
            base = m[1].get_bottom()[1]
            paso = (m[9].get_center()[0] - m[0].get_center()[0]) / 9
            pl = {}
            for c, parte in zip(chars, m):
                pts = np.concatenate([x.points for x in parte.family_members_with_points()])
                pts = pts - np.array([parte.get_center()[0], base, 0.0])
                ancho = paso if c.isdigit() else parte.width + 0.3 * paso
                pl[c] = (pts, ancho)
            cls._plantillas[font_size] = pl
        return cls._plantillas[font_size]

    def ancla(self):
        return self.submobjects[0].get_center()

    def poner(self, valor):
        s = f"{valor:.{self.decimales}f}"
        if float(s) == 0:
            s = s.lstrip("-")
        p0 = self.ancla()
        esc = np.linalg.norm(self.submobjects[1].get_center() - p0) / self.EPS
        x, partes = 0.0, []
        for c in s:
            pts, ancho = self.cifras[c]
            partes.append(pts + np.array([x + ancho / 2, 0.0, 0.0]))
            x += ancho
        self.set_points(p0 + esc * np.concatenate(partes))
        return self


def lectura(partes, vivos, font_size=26):
    """Renglón de TeX en partes; vivos = {i: (función, decimales, color)}: la parte i trae una muestra
    con las mismas cifras y se reemplaza por un Numero que sigue a la función."""
    m = MathTex(*partes, font_size=font_size * 1.25)
    cero = Numero(0, 0, font_size * 1.25)
    sobra = cero.get_bottom()[1] - cero.ancla()[1]          # lo que el 0 baja de la línea de base
    g, corrida = VGroup(), 0.0
    for i, parte in enumerate(m):
        parte.shift(RIGHT * corrida)
        if i in vivos:
            f, d, col = vivos[i]
            n = Numero(f(), d, font_size * 1.25, col)
            n.shift(np.array([parte.get_left()[0], parte.get_bottom()[1] - sobra, 0]) - n.ancla())
            n.add_updater(lambda mm, f=f: mm.poner(f()))
            corrida += n.width - parte.width
            g.add(n)
        else:
            g.add(parte)
    return g


def llave(mob, direccion, etiqueta, buff=0.1):
    b = Brace(mob, direccion, buff=buff, color=C_TEXTO)
    etiqueta.next_to(b, direccion, buff=0.12)
    return VGroup(b, etiqueta)


def texto(t, font_size=28, color=C_TEXTO, **kw):
    return Text(t, font_size=font_size, color=color, **kw)


# ---------------------------------------------------------------------------
# La escena
# ---------------------------------------------------------------------------
class LataOptima(Scene):
    def construct(self):
        self.r = ValueTracker(R_ACTO2)
        self.k = ValueTracker(K_PERSP)      # perspectiva de las elipses (0: de perfil)
        self.s = ValueTracker(S_LATA)       # escala del dibujo de la lata
        self.x = ValueTracker(X_LATA)
        self.kc = ValueTracker(1.0)         # cuántas veces más cara es la tapa por cm²
        self.siguiendo_k = False            # en el acto de k, r sigue al óptimo de C(r)
        self.trazando = False
        self.lo = self.hi = R_ACTO2         # tramo de r ya recorrido: hasta ahí están trazadas las curvas
        self.acto_titulo()
        self.acto_chapa()
        self.acto_despeje()
        self.acto_forma()
        self.acto_minimo()
        self.acto_cuadrado()
        self.acto_real()
        self.acto_cierre()

    # -- lo que depende del tracker ------------------------------------------
    def r_act(self):
        return r_optimo(self.kc.get_value()) if self.siguiendo_k else self.r.get_value()

    def base(self):
        return np.array([self.x.get_value(), Y_BASE, 0.0])

    def rango(self):
        r = self.r_act()
        if self.trazando:
            self.lo, self.hi = min(self.lo, r), max(self.hi, r)
        return self.lo, self.hi

    def f_tapas(self, r):
        return tapas(r, self.kc.get_value())

    def f_total(self, r):
        return costo(r, self.kc.get_value())

    def lata_viva(self):
        return always_redraw(lambda: lata(self.r_act(), self.base(), self.s.get_value(), self.k.get_value()))

    # -- 1 -------------------------------------------------------------------
    def acto_titulo(self):
        t = Text("La lata óptima", font_size=64, color=WHITE)
        sub = texto("entre todos los cilindros de igual volumen, el que usa menos chapa", 28, C_TENUE)
        VGroup(t, sub).arrange(DOWN, buff=0.45)
        self.play(FadeIn(t, shift=UP * 0.2), run_time=1.2)
        self.play(FadeIn(sub), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(t), FadeOut(sub), run_time=0.8)

    # -- 2 -------------------------------------------------------------------
    def acto_chapa(self):
        self.lata = self.lata_viva()
        self.play(FadeIn(self.lata, shift=UP * 0.3), run_time=1.2)
        r, s, k = R_ACTO2, S_LATA, K_PERSP
        R, H = r * s, h_de(r) * s
        b0 = self.base()
        et_t = texto("tapas", 30, C_TAPA).move_to([-2.2, 1.9, 0])
        et_p = texto("pared", 30, C_PARED).move_to([-2.2, -0.6, 0])
        fl_t = Arrow(et_t.get_left(), b0 + np.array([R * 0.75, H + 0.05, 0]), buff=0.12, color=C_TAPA,
                     stroke_width=3, max_tip_length_to_length_ratio=0.15)
        fl_p = Arrow(et_p.get_left(), b0 + np.array([R + 0.02, 0.9, 0]), buff=0.12, color=C_PARED,
                     stroke_width=3, max_tip_length_to_length_ratio=0.15)
        self.play(FadeIn(et_t), GrowArrow(fl_t), FadeIn(et_p), GrowArrow(fl_p))
        self.wait(1.2)
        self.play(FadeOut(et_t), FadeOut(fl_t), FadeOut(et_p), FadeOut(fl_p), run_time=0.7)

        # desarmar: las tapas salen y se ponen de frente; la pared se desenrolla
        ta, tp = ValueTracker(0.0), ValueTracker(0.0)

        def desarmada():
            a, u = ta.get_value(), tp.get_value()
            c1 = interpolate(b0 + UP * H, P_TAPA1, a)
            c2 = interpolate(b0, P_TAPA2, a)
            ry = interpolate(k, 1.0, a) * R
            return VGroup(tapa(c2, R, ry), pared_mob(interpolate(b0, P_PARED, u), R, H, k, u=u), tapa(c1, R, ry))

        piezas = always_redraw(desarmada)
        self.remove(self.lata)
        self.add(piezas)
        self.play(ta.animate.set_value(1), run_time=2.0)
        self.play(tp.animate.set_value(1), run_time=3.0)

        # rótulos de la chapa
        et1 = tex(r"\cT{\pi r^2}", 40).next_to(P_TAPA1 + RIGHT * R, RIGHT, buff=0.25)
        et2 = tex(r"\cT{\pi r^2}", 40).next_to(P_TAPA2 + RIGHT * R, RIGHT, buff=0.25)
        yb = P_PARED[1] - k * R
        borde_inf = Line([P_PARED[0] - PI * R, yb, 0], [P_PARED[0] + PI * R, yb, 0])
        borde_izq = Line([P_PARED[0] - PI * R, yb, 0], [P_PARED[0] - PI * R, yb + H, 0])
        ll_ancho = llave(borde_inf, DOWN, tex(r"\cP{2\pi r}", 40))
        ll_alto = llave(borde_izq, LEFT, tex(r"\cP{h}", 40))
        self.play(FadeIn(et1), FadeIn(et2))
        self.play(GrowFromCenter(ll_ancho), GrowFromCenter(ll_alto))
        self.wait(0.8)

        self.fA = MathTex("A", "=", r"2\pi r^2", "+", r"2\pi r h", font_size=46).move_to([-2.4, 3.2, 0])
        self.fA[2].set_color(C_TAPA)
        self.fA[4].set_color(C_PARED)
        self.play(Write(self.fA[:2]))
        self.play(TransformFromCopy(VGroup(et1, et2), self.fA[2]))
        self.play(Write(self.fA[3]), TransformFromCopy(VGroup(ll_ancho[1], ll_alto[1]), self.fA[4]))
        self.wait(0.6)
        self.fV = renglon(r"$V = \pi r^2 h", "(fijo)", font_size=34, color=C_TENUE).move_to([3.6, 3.2, 0])
        self.play(FadeIn(self.fV, shift=DOWN * 0.15))
        self.wait(1.5)

        # volver a armarla
        self.play(FadeOut(et1), FadeOut(et2), FadeOut(ll_ancho), FadeOut(ll_alto), run_time=0.7)
        self.play(tp.animate.set_value(0), run_time=2.5)
        self.play(ta.animate.set_value(0), run_time=1.8)
        self.remove(piezas)
        self.add(self.lata)
        self.wait(0.5)

    # -- 3 -------------------------------------------------------------------
    def acto_despeje(self):
        fs = 42
        l1 = MathTex("V", "=", r"\pi r^2", "h", r"\;\Rightarrow\;", "h", "=", r"\frac{V}{\pi r^2}",
                     font_size=fs).move_to([3.2, 2.4, 0])
        l2 = MathTex("A", "=", r"2\pi r^2", "+", r"2\pi r", "h", font_size=fs).move_to([3.2, 1.2, 0])
        l3 = MathTex("A(r)", "=", r"2\pi r^2", "+", r"2\pi r", r"\frac{V}{\pi r^2}", font_size=fs)
        l4 = MathTex("A(r)", "=", r"2\pi r^2", "+", r"\frac{2V}{r}", font_size=fs)
        l3.move_to([3.2, 0.0, 0])
        l4.move_to([3.2, -1.25, 0])
        for l, ts, ps in ((l2, [2], [4, 5]), (l3, [2], [4, 5]), (l4, [2], [4])):
            for i in ts:
                l[i].set_color(C_TAPA)
            for i in ps:
                l[i].set_color(C_PARED)

        self.play(FadeOut(self.fV[1]), ReplacementTransform(self.fV[0], l1[:4]), run_time=1.2)
        self.play(Write(l1[4]), TransformFromCopy(l1[3], l1[5]), run_time=0.9)
        self.play(Write(l1[6:]), run_time=1.0)
        self.wait(0.8)
        self.play(ReplacementTransform(self.fA[0], l2[0]), ReplacementTransform(self.fA[1], l2[1]),
                  ReplacementTransform(self.fA[2], l2[2]), ReplacementTransform(self.fA[3], l2[3]),
                  ReplacementTransform(self.fA[4], VGroup(l2[4], l2[5])), run_time=1.3)
        self.wait(0.5)
        self.play(TransformFromCopy(l2[:5], l3[:5]), TransformFromCopy(l1[7], l3[5]), run_time=1.5)
        self.play(Indicate(l3[5], color=C_PARED, scale_factor=1.1))
        self.wait(0.5)
        self.play(TransformFromCopy(l3[:4], l4[:4]), TransformFromCopy(VGroup(l3[4], l3[5]), l4[4]),
                  run_time=1.5)
        self.wait(0.6)
        et_t = texto("tapas", 24, C_TAPA).next_to(l4[2], DOWN, buff=0.22)
        et_p = texto("pared", 24, C_PARED).next_to(l4[4], DOWN, buff=0.22).align_to(et_t, UP)
        self.play(FadeIn(et_t), FadeIn(et_p))
        self.wait(1.5)
        self.fA = l4
        self.play(FadeOut(VGroup(l1, l2, l3, et_t, et_p)),
                  l4.animate.scale(0.95).move_to([X_FORM, Y_FORM, 0]), run_time=1.2)

    # -- 4 -------------------------------------------------------------------
    def armar_grafico(self):
        ax = Axes(x_range=[0, R_EJE, 0.5], y_range=[0, Y_EJE, 10], x_length=LX_GRAF, y_length=LY_GRAF,
                  axis_config={"color": C_EJE, "stroke_width": 2, "include_tip": True,
                               "tip_width": 0.18, "tip_height": 0.18})
        ax.shift(O_GRAF - ax.c2p(0, 0))
        nums = VGroup()
        for x in (0.5, 1, 1.5, 2):
            nums.add(MathTex(f"{x:g}", font_size=24, color=C_TENUE).next_to(g2p(x, 0), DOWN, buff=0.15))
        for y in (10, 20, 30, 40):
            nums.add(MathTex(f"{y}", font_size=24, color=C_TENUE).next_to(g2p(0, y), LEFT, buff=0.15))
        et_r = MathTex("r", font_size=34, color=C_TEXTO).next_to(g2p(R_EJE, 0), DOWN + RIGHT * 0.2, buff=0.1)
        et_A = MathTex("A", font_size=34, color=C_TEXTO).next_to(g2p(0, Y_EJE), LEFT, buff=0.2).shift(UP * 0.3)
        self.ejes = VGroup(ax, nums, et_r, et_A)

        def curvas():
            lo, hi = self.rango()
            return VGroup(curva(self.f_tapas, lo, hi, C_TAPA, 3.5),
                          curva(pared, lo, hi, C_PARED, 3.5),
                          curva(self.f_total, lo, hi, C_TOTAL, 6))

        def marcas():
            r = self.r_act()
            ys = (self.f_tapas(r), pared(r), self.f_total(r))
            linea = discontinua(g2p(r, 0), g2p(r, min(Y_EJE, ys[2] + 4)), C_TENUE, 2.2)
            puntos = [Dot(g2p(r, y), radius=0.075 if i < 2 else 0.095, color=c)
                      for i, (y, c) in enumerate(zip(ys, (C_TAPA, C_PARED, C_TOTAL)))]
            return VGroup(linea, *puntos)

        self.curvas = always_redraw(curvas)
        self.marcas = always_redraw(marcas)

    def armar_lecturas(self):
        r_vivo = lambda: self.r_act()
        lr = lectura(["r", "=", "0.00"], {2: (r_vivo, 2, WHITE)})
        lh = lectura(["h", "=", "0.00"], {2: (lambda: h_de(self.r_act()), 2, WHITE)})
        lv = renglon(r"$V = \pi r^2 h =", font_size=26)
        self.V_num = DecimalNumber(np.pi * r_vivo() ** 2 * h_de(r_vivo()), num_decimal_places=2,
                                   font_size=26 * 1.25, color=C_OPT)
        fila = VGroup(lr, lh, lv).arrange(RIGHT, buff=0.55, aligned_edge=DOWN)
        fila.to_edge(LEFT, buff=0.35).set_y(Y_LECT)
        for g in (lh, lv):          # misma línea de base que la primera
            g.shift(UP * (linea_base(lr[0]) - linea_base(g[0])))
        self.V_num.next_to(lv, RIGHT, buff=0.15)
        self.V_num.shift(UP * (linea_base(lv[0]) - self.V_num[0].get_bottom()[1]))
        self.V_num.add_updater(lambda d: d.set_value(np.pi * r_vivo() ** 2 * h_de(r_vivo())))
        self.lecturas = VGroup(lr, lh, lv, self.V_num)

        # barras de chapa debajo de la lata
        et_t = texto("tapas", 24, C_TAPA)
        et_p = texto("pared", 24, C_PARED)
        for et, y in ((et_t, Y_BT), (et_p, Y_BP)):
            et.next_to([X_BARRA0, y, 0], LEFT, buff=0.15)
        alto = Numero(0, 0, 30).height

        def numero_barra(f, y, col):
            n = Numero(f(), 2, 30, col)

            def seguir(m):
                m.poner(f())
                m.shift(np.array([X_BARRA0 + f() * ESC_BARRA + 0.12, y - alto / 2, 0]) - m.ancla())
            n.add_updater(seguir)
            seguir(n)
            return n

        barras = always_redraw(lambda: VGroup(barra(Y_BT, self.f_tapas(self.r_act()), C_TAPA),
                                              barra(Y_BP, pared(self.r_act()), C_PARED)))
        self.barras = VGroup(et_t, et_p, barras, numero_barra(lambda: self.f_tapas(self.r_act()), Y_BT, C_TAPA),
                             numero_barra(lambda: pared(self.r_act()), Y_BP, C_PARED))

    def acto_forma(self):
        self.armar_grafico()
        self.armar_lecturas()
        self.play(Create(self.ejes[0]), FadeIn(self.ejes[1:]), run_time=1.5)
        self.play(FadeIn(self.lecturas), FadeIn(self.barras), run_time=1.0)
        self.trazando = True
        self.add(self.curvas, self.marcas)
        self.wait(0.8)

        # hacia el disco chato
        self.play(self.r.animate.set_value(R_MAX), run_time=5, rate_func=smooth)
        sobran = texto("sobran las tapas", 30, C_TAPA).move_to([X_LATA, 0.3, 0])
        self.play(FadeIn(sobran, shift=UP * 0.15), Indicate(self.barras[0], color=C_TAPA_BORDE))
        self.et_curvas = VGroup(
            texto("tapas", 22, C_TAPA).next_to(g2p(R_MAX, tapas(R_MAX)), RIGHT, buff=0.12),
            texto("pared", 22, C_PARED).next_to(g2p(R_MAX, pared(R_MAX)), RIGHT, buff=0.12),
            MathTex("A(r)", font_size=30, color=C_TOTAL).next_to(g2p(R_MAX, costo(R_MAX)), RIGHT, buff=0.12))
        self.play(FadeIn(self.et_curvas))
        self.wait(1.2)
        self.play(FadeOut(sobran), run_time=0.6)

        # hacia el tubo finito
        self.play(self.r.animate.set_value(R_MIN), run_time=7, rate_func=smooth)
        sobra = texto("sobra la pared", 30, C_PARED)
        sobra.move_to([caja_lata(R_MIN, self.base())[1] + 0.25 + sobra.width / 2, 1.0, 0])
        self.play(FadeIn(sobra, shift=UP * 0.15), Indicate(self.barras[1], color=C_PARED_BORDE))
        self.wait(1.5)
        self.play(FadeOut(sobra), run_time=0.6)

        # de vuelta hacia el medio
        self.play(self.r.animate.set_value(R_MEDIO), run_time=3.5, rate_func=smooth)
        self.wait(0.5)

    # -- 5 -------------------------------------------------------------------
    def acto_minimo(self):
        self.play(self.r.animate.set_value(R_OPT), run_time=3, rate_func=smooth)
        self.trazando = False
        A_opt = costo(R_OPT)
        self.play(Flash(g2p(R_OPT, A_opt), color=C_OPT, line_length=0.25, flash_radius=0.3))
        tg = Line(g2p(R_OPT - 0.4, A_opt), g2p(R_OPT + 0.4, A_opt), color=C_OPT, stroke_width=4)
        et_tg = MathTex(r"A'(r) = 0", font_size=28, color=C_OPT)
        et_tg.move_to(g2p(R_OPT, A_opt) + np.array([0.2 + et_tg.width / 2, -0.32, 0]))  # entre A y las curvas de abajo
        self.play(Create(tg), FadeIn(et_tg))
        self.wait(0.8)

        # derivación corta, arriba de la lata
        fs = 34
        d1 = MathTex(r"A'(r)", "=", r"4\pi r", "-", r"\frac{2V}{r^2}", "=", "0", font_size=fs)
        d1[2].set_color(C_TAPA)
        d1[4].set_color(C_PARED)
        d2 = MathTex(r"\Rightarrow\;", "r^3", "=", r"\frac{V}{2\pi}", font_size=fs)
        d3 = MathTex(r"\Rightarrow\;", "h", "=", r"\frac{V}{\pi r^2}", "=", r"\frac{2\pi r^3}{\pi r^2}", "=",
                     "2r", font_size=fs)
        d3[7].set_color(C_OPT)
        der = VGroup(d1, d2, d3).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        der.move_to([X_LATA + 0.2, 0, 0]).shift(UP * (Y_TECHO - der.get_top()[1]))
        self.play(Write(d1), run_time=1.5)
        self.wait(0.6)
        self.play(Write(d2), run_time=1.2)
        self.wait(0.6)
        self.play(Write(d3), run_time=1.8)
        self.play(Indicate(d3[7], color=C_OPT))
        self.wait(1.2)

        # interpretación: pendientes opuestas de tapas y pared
        m_t, m_p = 4 * np.pi * R_OPT, -2 * V_N / R_OPT ** 2        # derivadas de 2πr² y de 2V/r en r*
        self.play(FadeOut(d1), FadeOut(d2), FadeOut(et_tg),
                  d3.animate.shift(UP * (Y_TECHO - d3.get_top()[1])).set_x(X_LATA + 0.2))
        enc = texto("al agrandar un poco r:", 26, C_TEXTO)
        f_t = renglon("tapas", rf"$\cT{{{m_t:+.2f}\,\Delta r}}", font_size=26, color=C_TAPA)
        f_p = renglon("pared", rf"$\cP{{{m_p:+.2f}\,\Delta r}}", font_size=26, color=C_PARED)
        fila_tp = VGroup(f_t, f_p).arrange(RIGHT, buff=0.5)
        f_p.shift(UP * (linea_base(f_t[0]) - linea_base(f_p[0])))
        nota = texto("se compensan justo", 26, C_OPT)
        interp = VGroup(enc, fila_tp, nota).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        interp.next_to(d3, DOWN, buff=0.3).set_x(X_LATA + 0.2)
        dr = 0.4

        def tangente(f, m, col):
            y0 = f(R_OPT)
            return Line(g2p(R_OPT - dr, y0 - m * dr), g2p(R_OPT + dr, y0 + m * dr), color=col, stroke_width=4)

        tg_t, tg_p = tangente(tapas, m_t, C_TAPA_BORDE), tangente(pared, m_p, C_PARED_BORDE)
        et_mt = MathTex(f"{m_t:+.2f}", font_size=26, color=C_TAPA).next_to(tg_t.get_end(), RIGHT, buff=0.1)
        et_mp = MathTex(f"{m_p:+.2f}", font_size=26, color=C_PARED).next_to(tg_p.get_start(), LEFT, buff=0.1)
        self.play(FadeIn(enc))
        self.play(Create(tg_t), FadeIn(et_mt), FadeIn(f_t))
        self.play(Create(tg_p), FadeIn(et_mp), FadeIn(f_p))
        self.wait(0.5)
        self.play(FadeIn(nota))
        self.wait(2.0)
        self.tg = tg
        self.play(FadeOut(VGroup(interp, nota, tg_t, tg_p, et_mt, et_mp, d3)))

    # -- 6 -------------------------------------------------------------------
    def acto_cuadrado(self):
        self.play(self.s.animate.set_value(S_PERFIL), run_time=1.5)
        self.play(self.k.animate.set_value(0.0), run_time=2.5)
        R, H = R_OPT * S_PERFIL, h_de(R_OPT) * S_PERFIL
        b = self.base()
        lado = 2 * R_OPT * S_PERFIL                     # el cuadrado se arma con 2r, la lata con h(r)
        cuad = Square(lado, color=C_OPT, stroke_width=5).move_to(b + UP * H / 2)
        self.play(Create(cuad), run_time=1.5)
        self.play(Indicate(cuad, color=C_OPT, scale_factor=1.04))
        ll_h = llave(Line(b + LEFT * R, b + LEFT * R + UP * H), LEFT, tex("h", 40))
        ll_d = llave(Line(b + LEFT * R + UP * H, b + RIGHT * R + UP * H), UP,
                     renglon("$2r =", "diámetro", font_size=28))
        self.play(GrowFromCenter(ll_h), GrowFromCenter(ll_d))
        alt = texto("altura = diámetro", 34, C_OPT).move_to([X_LATA, 2.55, 0])
        self.play(Write(alt))
        self.wait(1.2)

        # la pared usa el doble que las tapas: dos copias de la barra de tapas llenan la de pared
        T, P = tapas(R_OPT), pared(R_OPT)
        copias = VGroup(*[barra(Y_BT, T, C_TAPA).set_fill(C_TAPA, 0.35).set_stroke(C_TAPA_BORDE, 2)
                          for _ in range(2)])
        self.add(copias)
        self.play(copias[0].animate.move_to([X_BARRA0 + T * ESC_BARRA / 2, Y_BP, 0]),
                  copias[1].animate.move_to([X_BARRA0 + 3 * T * ESC_BARRA / 2, Y_BP, 0]), run_time=1.5)
        doble = MathTex(r"=\,2\times", f"{T:.2f}", font_size=34)
        doble[1].set_color(C_TAPA)
        num_p = self.barras[4]
        doble.next_to(num_p, RIGHT, buff=0.15)
        doble.shift(UP * (num_p.ancla()[1] - linea_base(doble)))
        formula = MathTex(r"2\pi r h", "=", r"4\pi r^2", "=", r"2\cdot", r"2\pi r^2", font_size=34)
        formula[0].set_color(C_PARED)
        formula[2].set_color(C_PARED)
        formula[5].set_color(C_TAPA)
        formula.next_to(self.barras[3], RIGHT, buff=0.45)
        formula.shift(UP * (self.barras[3].ancla()[1] - linea_base(formula)))
        self.play(Write(doble))
        self.play(Write(formula), run_time=1.5)
        self.wait(2.5)
        self.cuadrado_extra = VGroup(cuad, ll_h, ll_d, alt, copias, doble, formula)

    # -- 7 -------------------------------------------------------------------
    def acto_real(self):
        self.play(FadeOut(self.cuadrado_extra), FadeOut(self.barras), FadeOut(self.lecturas), run_time=1.0)
        X_OPT, X_REAL = -5.5, -3.0
        self.play(self.s.animate.set_value(S_LATA), self.k.animate.set_value(K_PERSP),
                  self.x.animate.set_value(X_OPT), run_time=2.0)
        real = lata(R_REAL, [X_REAL, Y_BASE, 0])
        self.play(FadeIn(real, shift=UP * 0.2))

        def rotulo(nombre, col, r_cm, x):
            h_cm = h_de(r_cm, V_CM3)
            filas = VGroup(texto(nombre, 22, col),
                           MathTex(rf"d = {2 * r_cm:.2f}\ \mathrm{{cm}}", font_size=28),
                           MathTex(rf"h = {h_cm:.2f}\ \mathrm{{cm}}", font_size=28),
                           MathTex(rf"A = {costo(r_cm, V=V_CM3):.1f}\ \mathrm{{cm^2}}", font_size=28))
            filas.arrange(DOWN, buff=0.16).move_to([x, -2.2, 0], aligned_edge=UP)
            return filas

        r_opt_cm, r_real_cm = r_optimo(V=V_CM3), r_con_proporcion(Q_REAL, V_CM3)
        rot_o = rotulo("óptima", C_OPT, r_opt_cm, X_OPT)
        rot_r = rotulo(f"real, h/d = {h_de(r_real_cm, V_CM3) / (2 * r_real_cm):.2f}", C_REAL, r_real_cm, X_REAL)
        vol = renglon("las dos con", rf"$V = {V_CM3:.0f}\ \mathrm{{cm^3}}", font_size=26)
        vol.to_edge(LEFT, buff=0.35).set_y(Y_LECT)
        self.play(FadeIn(vol), FadeIn(rot_o), FadeIn(rot_r), run_time=1.2)
        self.wait(1.5)

        # en el gráfico: los dos puntos, casi a la misma altura
        A_opt, A_real = costo(R_OPT), costo(R_REAL)
        p_real = Dot(g2p(R_REAL, A_real), radius=0.095, color=C_REAL)
        piso = discontinua(g2p(0.45, A_opt), g2p(1.65, A_opt), C_OPT, 2)
        et_real = renglon("real:", rf"$\cR{{+{100 * EXTRA_REAL:.2f}\,\%}}", font_size=24, color=C_REAL)
        et_real.next_to(p_real, UP, buff=0.5)
        et_opt = texto("óptima", 22, C_OPT).next_to(g2p(R_OPT, A_opt), DR, buff=0.18)
        self.play(FadeIn(p_real, scale=0.5), Create(piso), FadeIn(et_opt))
        self.play(FadeIn(et_real, shift=DOWN * 0.1))
        cerca = VGroup(texto("cerca del mínimo,", 26, WHITE), texto("apartarse cuesta muy poco", 26, WHITE))
        cerca.arrange(DOWN, buff=0.15).move_to([2.05, 1.2, 0])
        self.play(FadeIn(cerca))
        self.wait(3.0)

        # ¿por qué entonces? 1) las tapas son más gruesas
        self.play(FadeOut(VGroup(real, rot_o, rot_r, vol, p_real, piso, et_real, et_opt, cerca, self.tg)),
                  run_time=1.0)
        self.play(self.x.animate.set_value(X_LATA), run_time=1.2)
        self.siguiendo_k = True
        titulo = texto("¿Por qué las latas reales no son así?", 26, WHITE).to_edge(LEFT, buff=0.35).set_y(Y_LECT)
        razon = texto("1. las tapas son más gruesas que la pared", 24, C_TAPA)
        razon.next_to(titulo, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(titulo), FadeIn(razon))
        fC = MathTex("C(r)", "=", r"k\cdot", r"2\pi r^2", "+", r"\frac{2V}{r}", r"\;\Rightarrow\;", "h", "=",
                     r"2k\,r", font_size=36)
        fC[2:4].set_color(C_TAPA)
        fC[5].set_color(C_PARED)
        fC[7:].set_color(C_OPT)
        fC.move_to([X_FORM, Y_FORM, 0])
        if fC.get_right()[0] > FW / 2 - 0.25:
            fC.shift(LEFT * (fC.get_right()[0] - FW / 2 + 0.25))
        assert fC.get_left()[0] > titulo.get_right()[0] + 0.3, "la fórmula de C(r) pisa el título"
        self.play(ReplacementTransform(self.fA[0], fC[0]), ReplacementTransform(self.fA[1], fC[1]),
                  FadeIn(fC[2]), ReplacementTransform(self.fA[2], fC[3]),
                  ReplacementTransform(self.fA[3], fC[4]), ReplacementTransform(self.fA[4], fC[5]),
                  FadeOut(self.et_curvas), run_time=1.5)
        self.play(Write(fC[6:]))
        self.fA = fC
        lk = lectura(["k", "=", "0.00"], {2: (lambda: self.kc.get_value(), 2, C_TAPA)}, font_size=28)
        lq = lectura([r"h/d", "=", "0.00"],
                     {2: (lambda: h_de(self.r_act()) / (2 * self.r_act()), 2, C_OPT)}, font_size=28)
        lecturas_k = VGroup(lk, lq).arrange(RIGHT, buff=0.7, aligned_edge=DOWN)
        lq.shift(UP * (linea_base(lk[0]) - linea_base(lq[0])))
        lecturas_k.next_to(razon, DOWN, buff=0.4, aligned_edge=LEFT)
        nota_k = texto("cada cm² de tapa cuesta k veces más", 22, C_TENUE)
        nota_k.next_to(lecturas_k, DOWN, buff=0.22, aligned_edge=LEFT)
        tg_viva = always_redraw(lambda: Line(g2p(self.r_act() - 0.35, self.f_total(self.r_act())),
                                             g2p(self.r_act() + 0.35, self.f_total(self.r_act())),
                                             color=C_OPT, stroke_width=4))
        self.play(FadeIn(lecturas_k), FadeIn(nota_k), FadeIn(tg_viva))
        self.wait(0.8)
        self.play(self.kc.animate.set_value(K_MAX), run_time=6, rate_func=smooth)
        r_k = r_optimo(K_MAX)
        fin_t = tramos_curva(self.f_tapas, self.lo, self.hi)[0][-1]
        fin_c = tramos_curva(self.f_total, self.lo, self.hi)[0][-1]
        et_k = VGroup(MathTex(r"k\cdot 2\pi r^2", font_size=28, color=C_TAPA).next_to(fin_t, RIGHT, buff=0.1),
                      MathTex("C(r)", font_size=30, color=C_TOTAL).next_to(fin_c, LEFT, buff=0.15))
        self.play(FadeIn(et_k), Flash(g2p(r_k, costo(r_k, K_MAX)), color=C_OPT, line_length=0.2,
                                      flash_radius=0.28))
        self.wait(2.5)

        # las otras razones
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.2)
        self.siguiendo_k = False
        self.razones()

    def razones(self):
        titulo = Text("¿Por qué las latas reales no son así?", font_size=38, color=WHITE).set_y(3.45)
        self.play(FadeIn(titulo))
        datos = [
            ("Las tapas son más gruesas",
             ["si el cm² de tapa cuesta k veces más,", "el óptimo da altura/diámetro = k"], self.icono_gruesas()),
            ("El cuello",
             ["arriba se angosta, así la tapa,", "la pieza cara, queda más chica"], self.icono_cuello()),
            ("Costuras y bordes",
             ["las uniones tapa–pared gastan", "chapa extra, según el perímetro"], self.icono_costuras()),
            ("La mano",
             ["se tiene que agarrar con una mano,", "apilar y entrar en heladeras"], self.icono_mano()),
        ]
        centros = [(-3.55, 1.25), (3.55, 1.25), (-3.55, -2.0), (3.55, -2.0)]
        ANCHO_MARCO, ANCHO_CUERPO = 6.8, 4.3
        cuerpos = []
        for nom, lineas, _ in datos:
            c = VGroup(Text(nom, font_size=30, color=WHITE, weight=BOLD),
                       *[texto(l, 24, C_TEXTO) for l in lineas]).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
            c[1:].shift(DOWN * 0.1)
            cuerpos.append(c)
        esc = min(1.0, min(ANCHO_CUERPO / c.width for c in cuerpos))     # la misma letra en los cuatro
        for (_, _, icono), cuerpo, (cx, cy) in zip(datos, cuerpos, centros):
            marco = RoundedRectangle(corner_radius=0.2, width=ANCHO_MARCO, height=2.95, stroke_color="#3c4459",
                                     stroke_width=2, fill_color="#161b26", fill_opacity=1).move_to([cx, cy, 0])
            icono.scale_to_fit_height(2.1)
            if icono.width > 1.6:
                icono.scale_to_fit_width(1.6)
            icono.move_to([cx - 2.45, cy, 0])
            cuerpo.scale(esc)
            cuerpo.move_to([cx - 1.35 + cuerpo.width / 2, cy, 0])
            assert cuerpo.get_right()[0] < cx + ANCHO_MARCO / 2 - 0.15
            self.play(FadeIn(marco), FadeIn(icono, shift=UP * 0.15), FadeIn(cuerpo), run_time=1.0)
            self.wait(2.2)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)

    # -- íconos de las razones ------------------------------------------------
    @staticmethod
    def perfil(ancho, alto, centro=ORIGIN, grosor_tapa=3.0):
        """Perfil de una lata: pared en su color y tapas como segmentos."""
        c = np.asarray(centro, dtype=float)
        x0, x1, y0, y1 = c[0] - ancho / 2, c[0] + ancho / 2, c[1] - alto / 2, c[1] + alto / 2
        cuerpo = Rectangle(width=ancho, height=alto, stroke_width=0, fill_color=C_PARED_OSC, fill_opacity=1)
        cuerpo.move_to(c)
        lados = VGroup(Line([x0, y0, 0], [x0, y1, 0]), Line([x1, y0, 0], [x1, y1, 0])).set_stroke(C_PARED, 3)
        tapas_ = VGroup(Line([x0, y0, 0], [x1, y0, 0]), Line([x0, y1, 0], [x1, y1, 0])).set_stroke(
            C_TAPA, grosor_tapa)
        return VGroup(cuerpo, lados, tapas_)

    def icono_gruesas(self):
        p = self.perfil(1.0, 2.0, grosor_tapa=12)
        q = MathTex(r"h/d = k", font_size=30, color=C_OPT).next_to(p, DOWN, buff=0.2)
        return VGroup(p, q)

    def icono_cuello(self):
        w, h, w2, hc = 1.1, 2.1, 0.78, 0.32         # ancho, alto, ancho de la tapa, alto del cuello
        x0, x1, y0, y1 = -w / 2, w / 2, -h / 2, h / 2
        pts = [[x0, y0, 0], [x1, y0, 0], [x1, y1 - hc, 0], [w2 / 2, y1, 0], [-w2 / 2, y1, 0], [x0, y1 - hc, 0],
               [x0, y0, 0]]
        cuerpo = Polygon(*pts, stroke_color=C_PARED, stroke_width=3, fill_color=C_PARED_OSC, fill_opacity=1)
        tapa_ = Line([-w2 / 2, y1, 0], [w2 / 2, y1, 0], color=C_TAPA, stroke_width=6)
        fondo = Line([x0 + 0.1, y0, 0], [x1 - 0.1, y0, 0], color=C_TAPA, stroke_width=6)
        flechas = VGroup(Arrow([x0 - 0.35, y1 - 0.05, 0], [-w2 / 2 - 0.02, y1 - 0.05, 0], buff=0, stroke_width=3,
                               color=C_TEXTO, max_tip_length_to_length_ratio=0.4),
                         Arrow([x1 + 0.35, y1 - 0.05, 0], [w2 / 2 + 0.02, y1 - 0.05, 0], buff=0, stroke_width=3,
                               color=C_TEXTO, max_tip_length_to_length_ratio=0.4))
        return VGroup(cuerpo, tapa_, fondo, flechas)

    def icono_costuras(self):
        p = self.perfil(1.1, 2.0, grosor_tapa=4)
        x0, x1, y0, y1 = -0.55, 0.55, -1.0, 1.0
        rulos = VGroup(*[Circle(radius=0.09, color=C_OPT, stroke_width=4).move_to([x, y, 0])
                         for x in (x0, x1) for y in (y0, y1)])
        lupa = Circle(radius=0.42, color=C_TEXTO, stroke_width=2).move_to([1.05, 0.95, 0])
        # la doble costura: tapa y pared enganchadas, vistas en corte
        gancho_t = VMobject().set_points_smoothly([[0.95, 0.7, 0], [0.95, 1.12, 0], [1.13, 1.18, 0],
                                                   [1.18, 1.0, 0]]).set_stroke(C_TAPA, 3)
        gancho_p = VMobject().set_points_smoothly([[1.06, 0.62, 0], [1.06, 1.02, 0], [1.0, 1.1, 0]]).set_stroke(
            C_PARED, 3)
        guia = Line([x1 + 0.08, y1 + 0.05, 0], lupa.point_at_angle(PI * 1.1), color=C_TEXTO, stroke_width=1.5)
        return VGroup(p, rulos, guia, lupa, gancho_t, gancho_p)

    def icono_mano(self):
        center = np.array([-0.25, 0, 0])
        p = self.perfil(0.85, 1.8, center, grosor_tapa=4)
        # dedos que abrazan la lata, y el pulgar del otro lado
        dedos = VGroup(*[RoundedRectangle(corner_radius=0.09, width=1.0, height=0.2, stroke_width=1.5,
                                          stroke_color="#d9a07c", fill_color="#f2c4a0", fill_opacity=1)
                         .move_to(center + np.array([0.12, y, 0])) for y in (0.32, 0.08, -0.16, -0.40)])
        pulgar = RoundedRectangle(corner_radius=0.09, width=0.2, height=0.75, stroke_width=1.5,
                                  stroke_color="#d9a07c", fill_color="#f2c4a0", fill_opacity=1)
        pulgar.move_to(center + np.array([-0.52, 0.15, 0])).rotate(-0.25)
        # una pila de latas al lado
        pila = VGroup(*[self.perfil(0.32, 0.58, [0.75 + dx, -0.62 + dy, 0], 2.5)
                        for dx, dy in ((0, 0), (0.36, 0), (0.18, 0.62))])
        return VGroup(p, dedos, pulgar, pila)

    # -- 8 -------------------------------------------------------------------
    def acto_cierre(self):
        f = MathTex("A(r)", "=", r"2\pi r^2", "+", r"\frac{2V}{r}", r"\;\Longrightarrow\;", "h", "=", "2r",
                    font_size=56)
        f[2].set_color(C_TAPA)
        f[4].set_color(C_PARED)
        f[6:].set_color(C_OPT)
        f.move_to([-2.2, 0.3, 0])
        caja = SurroundingRectangle(f, color=C_OPT, buff=0.3, corner_radius=0.12, stroke_width=4)
        r, s = R_OPT, 1.3
        base = np.array([4.5, -1.2, 0])
        perfil = lata(r, base, s, 0.0)
        cuad = Square(2 * r * s, color=C_OPT, stroke_width=5).move_to(base + UP * h_de(r) * s / 2)
        pie = texto("altura = diámetro", 30, C_OPT).next_to(cuad, DOWN, buff=0.35)
        self.play(Write(f), run_time=2.0)
        self.play(Create(caja))
        self.play(FadeIn(perfil), run_time=1.0)
        self.play(Create(cuad), FadeIn(pie))
        self.wait(4.0)


# ---------------------------------------------------------------------------
# Verificación: python lata_optima.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.optimize import minimize_scalar

    def minimo(k=1.0, V=V_N):
        return minimize_scalar(lambda r: costo(r, k, V), bounds=(0.05, 20), method="bounded",
                               options={"xatol": 1e-12}).x

    for V in (V_N, V_CM3):
        r = minimo(V=V)
        assert abs(r - r_optimo(V=V)) < 1e-5 * r and abs(h_de(r, V) / r - 2) < 1e-5
        assert abs(pared(r, V) / tapas(r) - 2) < 1e-5
    for k in (1.0, 1.5, K_MAX):
        r = minimo(k)
        assert abs(r - r_optimo(k)) < 1e-5 and abs(h_de(r) / (2 * r) - k) < 1e-5
    rr = r_con_proporcion(Q_REAL, V_CM3)
    assert abs(costo(rr, V=V_CM3) / costo(r_optimo(V=V_CM3), V=V_CM3) - 1 - EXTRA_REAL) < 1e-12
    print(f"r* = {R_OPT:.6f}, h* = {h_de(R_OPT):.6f}, lata real (q = {Q_REAL}): +{100 * EXTRA_REAL:.3f} %")
    print(f"V = 355 cm³: r* = {r_optimo(V=V_CM3):.3f} cm, h* = {h_de(r_optimo(V=V_CM3), V_CM3):.3f} cm; "
          f"real r = {rr:.3f} cm, h = {h_de(rr, V_CM3):.3f} cm")

    # encuadre de la lata en los extremos del barrido y en el perfil
    for r, s, k in ((R_MIN, S_LATA, K_PERSP), (R_MAX, S_LATA, K_PERSP), (R_OPT, S_PERFIL, 0.0),
                    (r_optimo(K_MAX), S_LATA, K_PERSP)):
        x0, x1, y0, y1 = caja_lata(r, [X_LATA, Y_BASE], s, k)
        print(f"  r = {r:.3f}, s = {s}: x ∈ [{x0:.2f}, {x1:.2f}], y ∈ [{y0:.2f}, {y1:.2f}]")
        assert -FW / 2 + 0.2 < x0 and x1 < -0.8 and y0 > Y_BT + 0.3 and y1 < Y_LECT - 0.3
    print("Barras: máximo", max(tapas(R_MAX), pared(R_MIN)) * ESC_BARRA + X_BARRA0)
    print("Todo OK")
