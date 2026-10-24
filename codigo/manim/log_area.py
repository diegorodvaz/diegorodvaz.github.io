"""
El logaritmo como área.

ln x = ∫₁ˣ dt/t es el área bajo la hipérbola y = 1/t entre 1 y x (negativa si x < 1).
La transformación (t, y) ↦ (a·t, y/a) estira por a en t y comprime por a en y: conserva
áreas y manda la hipérbola en sí misma, así que la región de 1 a b pasa a ser la de a a ab,
con la misma área. Sumando, ln(ab) = ln a + ln b; y e es donde el área desde 1 vale 1.

    manim -pql log_area.py LogaritmoArea
    manim -pqh log_area.py LogaritmoArea

Todos los números en pantalla salen de la numérica de abajo (quad, brentq, shoelace).
`python verificar_log_area.py` la verifica.
"""
from manim import *
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Numérica: todo lo que aparece en pantalla sale de acá
# ---------------------------------------------------------------------------
A, B = 2.0, 3.0             # a y b (ab = 6)
T_MIN, T_MAX = 1 / 3, 8.0   # tramo dibujado de la hipérbola: desde y = 3 (el techo) hasta t = 8
W_CELDA = H_CELDA = 0.125   # celdas de la grilla del acto 3, en unidades de t y de y
CELDA_MARCADA = (4, 2)      # (columna, fila) de la celda de la lupa: [1.5, 1.625] × [0.25, 0.375]
DX = 0.4                    # ancho de la franja del acto 7


def area(x0, x1):
    """∫ dt/t de x0 a x1, con signo (negativa si x1 < x0)."""
    return quad(lambda t: 1.0 / t, x0, x1, epsabs=1e-14, epsrel=1e-13, limit=200)[0]


def ln(x):
    """ln x por definición: el área de 1 a x."""
    return area(1.0, x)


def derivada(f, x, h=1e-4):
    """Cociente centrado: la pendiente numérica de f en x."""
    return (f(x + h) - f(x - h)) / (2 * h)


E = brentq(lambda x: ln(x) - 1.0, 1.0, 4.0, xtol=1e-15)        # donde el área desde 1 vale 1
E2 = brentq(lambda x: area(E, x) - 1.0, E, 10.0, xtol=1e-14)   # la región de área 1 que sigue


def mapa(t, y, s):
    """(t, y) ↦ (s·t, y/s): estira por s en t y comprime por s en y."""
    return s * t, y / s


def shoelace(pts):
    """Área del polígono de vértices pts = [(t, y), …] (positiva si es antihorario)."""
    x, y = np.asarray(pts, dtype=float).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def contorno_region(t0, t1, n=160):
    """Vértices (t, y), en sentido antihorario, de la región bajo 1/t entre t0 < t1.
    El techo va muestreado en progresión geométrica (más denso donde la curva dobla más)."""
    ts = np.geomspace(t1, t0, n)
    return [(t0, 0.0), (t1, 0.0)] + [(t, 1.0 / t) for t in ts]


def celdas(t0, t1, w=W_CELDA, h=H_CELDA, m=12):
    """La región de t0 a t1 cortada en columnas de ancho w y filas de alto h, recortadas por la
    curva: [(i, k, vértices)], i la columna y k la fila. Las enteras son rectángulos; las de
    arriba tienen el techo min((k+1)·h, 1/t), muestreado en m puntos (más el quiebre)."""
    res = []
    for i in range(int(round((t1 - t0) / w))):
        a, b = t0 + i * w, t0 + (i + 1) * w
        k = 0
        while k * h < 1 / a - 1e-12:
            y0, y1 = k * h, (k + 1) * h
            if y1 <= 1 / b:
                pts = [(a, y0), (b, y0), (b, y1), (a, y1)]
            else:
                fin = min(b, 1 / y0) if y0 > 0 else b       # donde la curva corta el piso de la celda
                ts = np.linspace(a, fin, m)
                if a < 1 / y1 < fin:                         # donde la curva corta el techo
                    ts = np.sort(np.append(ts, 1 / y1))
                pts = [(a, y0), (fin, y0)] + [(t, min(y1, 1 / t)) for t in ts[::-1]]
                pts = [p for j, p in enumerate(pts) if j == 0 or np.hypot(*np.subtract(p, pts[j - 1])) > 1e-12]
            res.append((i, k, pts))
            k += 1
    return res


# ---------------------------------------------------------------------------
# Paleta: hipérbola blanca; el mismo color en cada región y en su término de las fórmulas
# ---------------------------------------------------------------------------
FONDO = "#0e1117"
config.background_color = FONDO
FW, FH = config.frame_width, config.frame_height

C_HIP = WHITE
C_LN = "#b39dff"            # ln x en general (área positiva) y su gráfico
C_NEG = "#ff5c7a"           # áreas negativas (a la izquierda de 1)
C_A = "#4cc9f0"             # ln a
C_B = "#ff9f43"             # ln b y su transformada
C_UNO = ["#80ed99", "#2ec4b6"]      # regiones de área 1: de 1 a e y de e a e²
C_RES = "#ffe66d"           # resaltados: la celda de la lupa, los puntos, la franja dx
C_EJE = "#6b7591"
C_TEXTO = GREY_A


def _hex(c):
    return ManimColor(c).to_hex()[1:].upper()


PLANTILLA = TexTemplate()
PLANTILLA.add_to_preamble(
    r"\usepackage{xcolor}"
    + "".join(rf"\newcommand{{\{n}}}[1]{{\textcolor[HTML]{{{_hex(c)}}}{{#1}}}}"
              for n, c in (("cL", C_LN), ("cN", C_NEG), ("cA", C_A), ("cB", C_B),
                           ("cU", C_UNO[0]), ("cV", C_UNO[1]), ("cR", C_RES))))
config.tex_template = PLANTILLA

# ---------------------------------------------------------------------------
# Layout (16:9, 14.2 × 8)
# ---------------------------------------------------------------------------
O_EJES = np.array([-6.5, -2.1, 0.0])    # dónde cae el (0, 0) de los ejes principales
X_COL, ANCHO_COL = 4.6, 4.5             # columna derecha de fórmulas
Y_TIT, Y_LEY = 3.5, -3.5
LUPA, M_LUPA = np.array([X_COL, 0.5, 0.0]), 7.0
O_IZQ, O_LN = np.array([-6.55, -1.3, 0.0]), np.array([0.95, -1.3, 0.0])     # acto 7
O_FIN = np.array([-6.5, -1.85, 0.0])                                         # acto 8


def ejes(x_length=8.4, y_length=4.5, origen=O_EJES, numeros=24, y_range=(0, 3, 1), y_numeros=(1, 2, 3)):
    ax = Axes(x_range=[0, 8, 1], y_range=list(y_range), x_length=x_length, y_length=y_length,
              axis_config=dict(color=C_EJE, stroke_width=2, include_tip=True, tip_width=0.15,
                               tip_height=0.15, font_size=numeros),
              x_axis_config=dict(numbers_to_include=range(1, 9)),
              y_axis_config=dict(numbers_to_include=list(y_numeros)))
    ax.shift(np.asarray(origen) - ax.c2p(0, 0))
    for eje in ax.get_axes():
        eje.numbers.set_color(C_TEXTO)
    return ax.set_z_index(2)


def rotulos_ejes(ax, x="t", y="y", font_size=32):
    return VGroup(MathTex(x, font_size=font_size, color=C_TEXTO).next_to(ax.x_axis.get_end(), RIGHT, buff=0.12),
                  MathTex(y, font_size=font_size, color=C_TEXTO).next_to(ax.y_axis.get_end(), UP, buff=0.1))


def hiperbola(ax, ancho=4):
    return ax.plot(lambda t: 1 / t, x_range=[T_MIN, T_MAX, 0.01], color=C_HIP, stroke_width=ancho).set_z_index(3)


def linea_uno(ax):
    return DashedLine(ax.c2p(1, 0), ax.c2p(1, 3), dash_length=0.08).set_stroke(GREY_B, 2, opacity=0.8).set_z_index(2)


def poligono(ax, pts, color=None, opacidad=0.8, borde=None, ancho=2.0):
    """Polígono de vértices (t, y) en coordenadas de los ejes."""
    P = Polygon(*[ax.c2p(t, y) for t, y in pts])
    P.set_fill(color or BLACK, opacidad if color else 0)
    P.set_stroke(borde or color or BLACK, ancho if (borde or color) else 0)
    return P


def region(ax, t0, t1, color, **kw):
    a, b = sorted((t0, t1))
    return poligono(ax, contorno_region(a, max(b, a + 1e-4)), color, **kw)


def transformar_punto(ax, p, s):
    """El punto de la escena p, llevado por (t, y) ↦ (s·t, y/s) en coordenadas de los ejes."""
    return ax.c2p(*mapa(*ax.p2c(p), s))


def transformada(ax, mob, s):
    """Copia de mob con la transformación aplicada punto a punto (anclas y manijas)."""
    return mob.copy().apply_function(lambda p: transformar_punto(ax, p, s))


def medidas(ax, poli):
    """Ancho, alto y área (shoelace) de un polígono de la escena, medidos en coordenadas de los ejes."""
    pts = np.array([ax.p2c(v) for v in poli.get_vertices()])
    return np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), shoelace(pts)


# ---------------------------------------------------------------------------
# Texto: Text para el castellano, MathTex para la matemática, con línea de base común
# ---------------------------------------------------------------------------
DESCENDENTES = "gjpqy,;()¿?"


def tex(t, font_size, color):
    """MathTex de t con su línea de base anotada: se compila con una x de referencia delante
    (así "w," o "a^2" no se apoyan en la coma ni en el exponente) y se devuelve solo la parte t."""
    m = MathTex("x", t, font_size=font_size, color=color)
    parte = m[1]
    parte.base_rel = m[0].get_bottom()[1] - parte.get_bottom()[1]
    return parte


def linea_base(mob):
    """Altura de la línea de base. Con `tex`: la de la x de referencia. En un Text: mediana de los
    bordes inferiores de los glifos sin descendente. En otro MathTex: el borde inferior más
    repetido (el más bajo si hay empate; sirve para los renglones de números)."""
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
    """Mezcla Text y MathTex en fila: las partes que empiezan con '$' son TeX."""
    mobs = [p if isinstance(p, Mobject) else
            tex(p[1:], font_size * 1.25, color) if p.startswith("$")
            else Text(p, font_size=font_size, color=color) for p in partes]
    g = VGroup(*mobs).arrange(RIGHT, buff=buff)
    y = linea_base(mobs[0])
    for m in mobs[1:]:
        m.shift(UP * (y - linea_base(m)))
    return g


def ajustar(mob, ancho):
    if mob.width > ancho:
        mob.scale_to_fit_width(ancho)
    return mob


def leyenda(*partes, font_size=28, color=C_TEXTO):
    return ajustar(renglon(*partes, font_size=font_size, color=color), FW - 1.0).move_to([0, Y_LEY, 0])


def fila(texto, partes, vivos=None, font_size=30, color=WHITE):
    """Renglón 'texto partes…' (texto en Text, partes en un MathTex). `vivos` = {i: (función, decimales)}:
    la parte i trae una muestra con las mismas cifras (para la línea de base) y se reemplaza por un
    DecimalNumber que sigue a la función; lo que viene después se corre según su ancho real."""
    vivos = vivos or {}
    m = MathTex(*partes, font_size=font_size * 1.25, color=color)
    mobs = ([Text(texto, font_size=font_size, color=color)] if texto else []) + [m]
    renglon(*mobs)
    g = VGroup(*mobs[:-1])
    g.numeros = {}
    corrida = 0.0
    for i, parte in enumerate(m):
        parte.shift(RIGHT * corrida)
        if i in vivos:
            f, d = vivos[i]
            n = DecimalNumber(f(), num_decimal_places=d, font_size=font_size * 1.25, color=color)
            n.move_to(parte, aligned_edge=LEFT)
            n.add_updater(lambda mm, f=f: mm.set_value(f()))
            corrida += n.width - parte.width
            g.add(n)
            g.numeros[i] = n
        else:
            g.add(parte)
    return g


def meter(*mobs, x=X_COL, ancho=ANCHO_COL):
    """Corre (y si hace falta achica) el grupo para que quepa en la columna derecha."""
    g = VGroup(*mobs)
    if g.width > ancho:
        g.scale_to_fit_width(ancho)
    izq, der = x - ancho / 2, x + ancho / 2
    if g.get_right()[0] > der:
        g.shift(LEFT * (g.get_right()[0] - der))
    if g.get_left()[0] < izq:
        g.shift(RIGHT * (izq - g.get_left()[0]))
    return g


def alinear_igual(mt, x, y, i=1):
    """Corre `mt` para que su parte i quede centrada en (x, y)."""
    return mt.shift(np.array([x, y, 0]) - mt[i].get_center())


def caja(mob, color=GREY_B):
    return SurroundingRectangle(mob, color=color, buff=0.2, corner_radius=0.12, stroke_width=2.5)


# ---------------------------------------------------------------------------
# La escena
# ---------------------------------------------------------------------------
class LogaritmoArea(Scene):
    def construct(self):
        self.caption = None
        self.titulo = None
        self.ax = ejes()
        self.rot_ejes = rotulos_ejes(self.ax)
        self.hip = hiperbola(self.ax)
        self.linea1 = linea_uno(self.ax)
        self.rot_hip = MathTex(r"y = \frac{1}{t}", font_size=36).next_to(self.ax.c2p(0.56, 2.5), RIGHT, buff=0)
        self.acto_titulo()
        self.acto_definicion()
        self.acto_transformacion()
        self.acto_suma()
        self.acto_consecuencias()
        self.acto_e()
        self.acto_grafico()
        self.acto_cierre()

    # -- utilidades ---------------------------------------------------------
    def cambiar_leyenda(self, nueva, *extra, run_time=0.8):
        """La leyenda vieja sale antes de que entre la nueva, para que no se pisen."""
        cambio = []
        if self.caption is not None:
            cambio.append(FadeOut(self.caption, shift=UP * 0.15))
        if nueva is not None:
            cambio.append(FadeIn(nueva, shift=UP * 0.15))
        self.play(*extra, LaggedStart(*cambio, lag_ratio=0.6), run_time=run_time)
        self.caption = nueva

    def seccion(self, *partes):
        nuevo = renglon(*partes, font_size=34, color=WHITE).move_to([0, Y_TIT, 0])
        anims = ([FadeOut(self.titulo, shift=UP * 0.2)] if self.titulo is not None else []) + [
            FadeIn(nuevo, shift=DOWN * 0.2)]
        self.play(LaggedStart(*anims, lag_ratio=0.5), run_time=1.0)
        self.titulo = nuevo

    def limpiar(self, *conservar, run_time=0.9):
        """Saca todo lo que no esté en `conservar` (título y leyenda incluidos)."""
        quedan = {id(x) for m in conservar for x in m.get_family()}
        salen = [m for m in self.mobjects if id(m) not in quedan and any(x.has_points() for x in m.get_family())]
        for m in salen:
            for x in m.get_family():
                x.clear_updaters()
        if salen:
            self.play(*[FadeOut(m) for m in salen], run_time=run_time)
        self.remove(*[m for m in self.mobjects if id(m) not in quedan])
        self.caption = self.titulo = None

    @staticmethod
    def congelar(*mobs):
        for m in mobs:
            for x in m.get_family():
                x.clear_updaters()

    def transformada(self, mob, s):
        return transformada(self.ax, mob, s)

    def region(self, t0, t1, color, **kw):
        return region(self.ax, t0, t1, color, **kw)

    def region_viva(self, x_fn, color=C_LN):
        """Región entre 1 y x_fn(): en `color` si x ≥ 1 y en C_NEG si x < 1 (el área resta)."""
        def dibujar():
            x = x_fn()
            return self.region(1.0, x, color if x >= 1 else C_NEG)
        reg = dibujar()
        reg.add_updater(lambda m: m.become(dibujar()))
        return reg

    def marca_x(self, x_fn):
        """Punto sobre el eje t en x, con su rótulo arriba a la derecha."""
        p = Dot(radius=0.055, color=WHITE).set_z_index(5)
        lab = MathTex("x", font_size=32).set_z_index(5)
        p.add_updater(lambda m: m.move_to(self.ax.c2p(x_fn(), 0)))
        lab.add_updater(lambda m: m.next_to(self.ax.c2p(x_fn(), 0), UR, buff=0.08))
        return VGroup(p, lab)

    def rotulo_t(self, t, tex, color=WHITE, marca=False, font_size=30):
        """Rótulo simbólico en una segunda fila debajo de los números del eje t."""
        p = self.ax.c2p(t, 0)
        lab = MathTex(tex, font_size=font_size, color=color).move_to(p + DOWN * 0.8)
        if not marca:
            return lab
        return VGroup(lab, Line(p + DOWN * 0.1, p + UP * 0.1).set_stroke(color, 3).set_z_index(3))

    def deformar(self, mob0, s_fin, *extra, color=None, run_time=2.5):
        """Copia de mob0 que se transforma con (t, y) ↦ (s·t, y/s), s de 1 a s_fin;
        color(s) da el color de cada momento."""
        s = ValueTracker(1.0)

        def upd(m):
            m.become(self.transformada(mob0, s.get_value()))
            if color is not None:
                m.set_fill(color(s.get_value())).set_stroke(color(s.get_value()))

        copia = self.transformada(mob0, 1.0)
        copia.add_updater(upd)
        self.add(copia)
        self.play(s.animate.set_value(s_fin), *extra, run_time=run_time, rate_func=smooth)
        copia.clear_updaters()
        return copia

    # -- 1. título ----------------------------------------------------------
    def acto_titulo(self):
        titulo = Text("El logaritmo como área", font_size=60)
        self.play(FadeIn(titulo, shift=UP * 0.3), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(titulo), run_time=0.8)

    # -- 2. la hipérbola y la definición ------------------------------------
    def acto_definicion(self):
        self.seccion("El área bajo la hipérbola")
        self.play(Create(self.ax), FadeIn(self.rot_ejes), run_time=1.6)
        self.play(Create(self.hip), run_time=1.6)
        self.play(FadeIn(self.rot_hip), Create(self.linea1), run_time=1.0)
        self.cambiar_leyenda(leyenda("la hipérbola", "$y = 1/t", "y una línea vertical en", "$t = 1"))
        self.wait(1.2)

        x = ValueTracker(1.0)
        reg = self.region_viva(x.get_value)
        marca = self.marca_x(x.get_value)
        fx = fila(None, ["x", "=", "0.000"], {2: (x.get_value, 3)}).move_to([X_COL, 1.0, 0])
        fa = fila("área", ["=", "0.000"], {1: (lambda: ln(x.get_value()), 3)})
        alinear_igual(fa, fx[1].get_x(), 0.3)
        fa.numeros[1].add_updater(lambda m: m.set_color(WHITE if x.get_value() >= 1 else C_NEG))
        self.add(reg)
        self.play(FadeIn(marca), FadeIn(fx), FadeIn(fa))
        self.cambiar_leyenda(leyenda("movemos", "$x", "desde", "$1", "y se pinta el área bajo la curva entre",
                                     "$1", "y", "$x"))
        self.play(x.animate.set_value(4.0), run_time=4, rate_func=smooth)
        self.wait(0.8)

        defi = MathTex(r"\cL{\ln x}", "=", r"\int_1^x \frac{dt}{t}", font_size=52).move_to([X_COL, 2.35, 0])
        self.play(Write(defi), run_time=1.5)
        self.cambiar_leyenda(leyenda("esa área es, por definición, el logaritmo natural de", "$x"))
        self.wait(2.5)

        neg = MathTex(r"x<1:\quad \ln x = -\int_x^1 \frac{dt}{t} < 0", color=C_NEG, font_size=38)
        neg.move_to([X_COL, -1.0, 0])
        self.play(x.animate.set_value(0.45), run_time=4.5, rate_func=smooth)
        self.cambiar_leyenda(leyenda("si vas para la izquierda, el área resta"))
        self.play(Write(neg))
        self.wait(2.5)
        self.play(x.animate.set_value(B), run_time=3.5, rate_func=smooth)
        self.wait(1)
        self.congelar(reg, marca, fx, fa)
        self.reg = reg
        self.play(FadeOut(VGroup(marca, fx, fa, neg, defi)))

    # -- 3. la transformación que conserva áreas ----------------------------
    def acto_transformacion(self):
        ax = self.ax
        self.seccion("Una transformación que conserva áreas")
        claro = interpolate_color(ManimColor(FONDO), ManimColor(C_B), 0.92)
        oscuro = interpolate_color(ManimColor(FONDO), ManimColor(C_B), 0.55)
        datos = sorted(celdas(1.0, B), key=lambda d: (d[0], d[1]) == CELDA_MARCADA)      # la marcada, última
        celdas0 = VGroup(*[poligono(ax, pts, claro if (i + k) % 2 == 0 else oscuro, opacidad=1,
                                    borde=FONDO, ancho=0.8) for i, k, pts in datos])
        contorno0 = poligono(ax, contorno_region(1.0, B), None, borde=C_B, ancho=3).set_z_index(1)
        celdas_ = celdas0.copy()
        contorno = contorno0.copy()
        rb = self.rotulo_t(B, "b")
        self.play(FadeOut(self.reg), FadeIn(celdas_), Create(contorno), FadeIn(rb), run_time=1.5)
        self.cambiar_leyenda(leyenda("cortamos en celdas chicas la región de", "$1", "a", f"$b = {B:g}"))
        self.wait(1.5)

        T = MathTex(r"(t,\ y)", r"\ \longmapsto\ ", r"(a\,t,\ y/a)", font_size=46).move_to([X_COL, 2.65, 0])
        self.play(Write(T))
        self.cambiar_leyenda(leyenda("la transformación estira por", "$a", "en", "$t", "y comprime por", "$a",
                                     "en", "$y"))
        self.wait(2)

        # la celda marcada, ampliada en la lupa
        marcada0 = celdas0[-1]
        self.play(celdas_[-1].animate.set_fill(C_RES).set_stroke(WHITE, 1.5), run_time=0.8)
        marcada0.set_fill(C_RES).set_stroke(WHITE, 1.5)
        s = ValueTracker(1.0)

        def ampliada(sv):
            return self.transformada(marcada0, sv).scale(M_LUPA).move_to(LUPA).set_stroke(WHITE, 2)

        lupa = ampliada(1.0)
        br_w = Brace(lupa, DOWN, buff=0.08, color=C_TEXTO)
        br_h = Brace(lupa, LEFT, buff=0.08, color=C_TEXTO)
        lw = MathTex("w", font_size=36).next_to(br_w, DOWN, buff=0.08)
        lh = MathTex("h", font_size=36).next_to(br_h, LEFT, buff=0.08)

        def upd_lupa(g):
            c = ampliada(s.get_value())
            lupa.become(c)
            br_w.become(Brace(c, DOWN, buff=0.08, color=C_TEXTO))
            br_h.become(Brace(c, LEFT, buff=0.08, color=C_TEXTO))

        g_lupa = VGroup(lupa, br_w, br_h)

        def medida(j):
            return lambda: medidas(ax, self.transformada(marcada0, s.get_value()))[j]

        fc = fila("celda:", ["0.0000", r"\times", "0.0000", "=", "0.0000"],
                  {0: (medida(0), 4), 2: (medida(1), 4), 4: (medida(2), 4)}, font_size=28)
        meter(fc.move_to([X_COL, -1.8, 0]))
        fr = fila("área de la región:", ["0.000"],
                  {0: (lambda: medidas(ax, self.transformada(contorno0, s.get_value()))[2], 3)}, font_size=28)
        meter(fr.move_to([X_COL, -2.45, 0]))
        self.play(TransformFromCopy(celdas_[-1], lupa), FadeIn(br_w), FadeIn(br_h), FadeIn(lw), FadeIn(lh),
                  run_time=1.3)
        self.play(FadeIn(fc), FadeIn(fr))
        self.cambiar_leyenda(leyenda("una celda, ampliada: ancho", "$w{,}", "alto", "$h"))
        self.wait(2)

        fs = fila(None, ["a", "=", "0.00"], {2: (s.get_value, 2)}).move_to([X_COL, 2.0, 0])
        fantasma = DashedVMobject(contorno0.copy().set_stroke(GREY_B, 2), num_dashes=70).set_z_index(1)
        self.play(FadeIn(fs), FadeOut(lw), FadeOut(lh), FadeIn(fantasma))
        celdas_.add_updater(lambda m: m.become(self.transformada(celdas0, s.get_value())))
        contorno.add_updater(lambda m: m.become(self.transformada(contorno0, s.get_value())))
        g_lupa.add_updater(upd_lupa)
        self.add(g_lupa)
        self.cambiar_leyenda(leyenda("$a", "crece de", "$1", "a", f"${A:g}"))
        self.play(s.animate.set_value(A), FadeOut(rb), run_time=5, rate_func=smooth)
        self.congelar(celdas_, contorno, g_lupa, fs, fc, fr)

        lw2 = MathTex(r"a\,w", font_size=36).next_to(br_w, DOWN, buff=0.08)
        lh2 = MathTex("h/a", font_size=36).next_to(br_h, LEFT, buff=0.08)
        prod = MathTex(r"w\cdot h", "=", r"(a\,w)\cdot(h/a)", font_size=40).move_to([X_COL, -1.15, 0])
        self.play(FadeIn(lw2), FadeIn(lh2), Write(prod))
        self.cambiar_leyenda(leyenda("estirar por", "$a", "y comprimir por", "$a", "no cambia el área"))
        self.wait(3)

        # la hipérbola va a parar a sí misma: puntos que se deslizan por ella
        self.play(FadeOut(VGroup(lupa, br_w, br_h, lw2, lh2, prod, fc)))
        inv = MathTex(r"y = \frac{1}{t}", r"\ \Longrightarrow\ ", r"\frac{y}{a} = \frac{1}{a\,t}",
                      font_size=44).move_to(LUPA)
        self.play(Write(inv))
        self.cambiar_leyenda(leyenda("y la hipérbola se transforma en sí misma"))
        s2 = ValueTracker(1.0)
        puntos = VGroup(*[Dot(ax.c2p(t, 1 / t), radius=0.065, color=C_RES).set_z_index(6)
                          for t in (0.45, 0.7, 1.0, 1.5, 2.2, 3.3)])
        for p in puntos:
            c0 = p.get_center()
            p.add_updater(lambda m, c0=c0: m.move_to(transformar_punto(ax, c0, s2.get_value())))
        estelas = VGroup(*[TracedPath(p.get_center, stroke_color=C_RES, stroke_width=4).set_z_index(5)
                           for p in puntos])
        self.play(FadeIn(puntos))
        self.add(estelas)
        self.play(s2.animate.set_value(A), run_time=3, rate_func=smooth)
        self.wait(1.5)
        self.congelar(puntos, estelas)
        self.play(FadeOut(puntos), FadeOut(estelas))

        igual = MathTex(rf"\int_1^{{{B:g}}} \frac{{dt}}{{t}}", "=", f"{area(1, B):.3f}", "=",
                        rf"\int_{{{A:g}}}^{{{A * B:g}}} \frac{{dt}}{{t}}", color=C_B, font_size=44).move_to(LUPA)
        meter(igual)
        self.r_a, self.r_ab = self.rotulo_t(A, "a"), self.rotulo_t(A * B, "ab")
        self.play(FadeOut(inv), FadeTransform(fr, igual), FadeIn(self.r_a), FadeIn(self.r_ab), run_time=1.2)
        self.cambiar_leyenda(leyenda("la región de", "$1", "a", "$b", "pasó a ser la de", "$a", "a", "$ab{,}",
                                     "con la misma área"))
        self.wait(3)
        self.play(FadeOut(VGroup(T, fs, igual, fantasma)))
        self.celdas, self.contorno = celdas_, contorno

    # -- 4. ln(ab) = ln a + ln b --------------------------------------------
    def acto_suma(self):
        ax = self.ax
        self.seccion("El producto se vuelve suma")
        self.regB = self.region(A, A * B, C_B)
        self.play(FadeOut(self.celdas), FadeOut(self.contorno), FadeIn(self.regB), run_time=1.2)
        self.lab_b = MathTex(r"\ln b", font_size=30).move_to(ax.c2p(3.3, 0.15)).set_z_index(4)
        self.play(Write(self.lab_b))
        self.cambiar_leyenda(leyenda("la región de", "$a", "a", "$ab", "es la de", "$1", "a", "$b",
                                     "deformada: su área es", "$\ln b"))
        self.wait(2.5)

        self.regA = self.region(1, A, C_A)
        self.lab_a = MathTex(r"\ln a", font_size=30).move_to(ax.c2p(1.45, 0.3)).set_z_index(4)
        self.play(FadeIn(self.regA), Write(self.lab_a))
        self.cambiar_leyenda(leyenda("y la de", "$1", "a", "$a", "tiene área", "$\ln a"))
        self.wait(2)

        Y = 1.15
        llave = Brace(Line(ax.c2p(1, Y), ax.c2p(A * B, Y)), UP, buff=0.05, color=WHITE)
        lab_ab = MathTex(r"\ln(ab)", font_size=34).next_to(llave, UP, buff=0.1)
        borde = poligono(ax, contorno_region(1, A * B), None, borde=WHITE, ancho=3).set_z_index(1)
        self.play(Create(borde), GrowFromCenter(llave), Write(lab_ab), run_time=1.5)
        self.cambiar_leyenda(leyenda("juntas cubren de", "$1", "a", "$ab", "y su área es", "$\ln(ab)"))
        self.wait(1.5)

        eq1 = MathTex(r"\ln(ab)", "=", r"\cA{\ln a}", "+", r"\cB{\ln b}", font_size=52).move_to([X_COL, 2.3, 0])
        eq2 = MathTex(rf"\ln {A * B:g}", "=", rf"\cA{{\ln {A:g}}}", "+", rf"\cB{{\ln {B:g}}}", font_size=44)
        eq3 = MathTex(f"{ln(A * B):.3f}", "=", rf"\cA{{{ln(A):.3f}}}", "+", rf"\cB{{{ln(B):.3f}}}", font_size=44)
        alinear_igual(eq2, eq1[1].get_x(), 1.25)
        alinear_igual(eq3, eq1[1].get_x(), 0.45)
        meter(eq1, eq2, eq3)
        self.play(TransformFromCopy(lab_ab, eq1[0]), Write(eq1[1]), TransformFromCopy(self.lab_a, eq1[2]),
                  Write(eq1[3]), TransformFromCopy(self.lab_b, eq1[4]), run_time=2)
        self.play(Write(eq2))
        self.play(Write(eq3))
        self.cambiar_leyenda(leyenda("el eje", "$t", "multiplica, el área suma"))
        self.wait(3.5)
        self.suma = VGroup(self.regB, self.lab_b, borde, llave, lab_ab, eq1, eq2, eq3, self.r_ab)

    # -- 5. potencias e inversos --------------------------------------------
    def acto_consecuencias(self):
        ax = self.ax
        self.seccion("Potencias e inversos")
        self.play(FadeOut(self.suma), FadeOut(self.caption))
        self.caption = None
        tonos = [C_A, interpolate_color(ManimColor(C_A), ManimColor(WHITE), 0.45),
                 interpolate_color(ManimColor(C_A), ManimColor("#123c69"), 0.45)]

        def pasar(c0, c1, s0, s1):
            return lambda s: interpolate_color(ManimColor(c0), ManimColor(c1), (s - s0) / (s1 - s0))

        self.cambiar_leyenda(leyenda("la región de", "$1", "a", "$a", "transformada por", "$a", "va de", "$a",
                                     "a", "$a^2"))
        r2 = self.deformar(self.regA, A, color=pasar(tonos[0], tonos[1], 1, A))
        l2 = MathTex(r"\ln a", font_size=30).move_to(ax.c2p(2.8, 0.15)).set_z_index(4)
        ra2 = self.rotulo_t(A ** 2, "a^2")
        self.play(FadeIn(l2), FadeIn(ra2))
        self.wait(0.8)
        self.cambiar_leyenda(leyenda("y otra vez: de", "$a^2", "a", "$a^3{,}", "siempre con área", "$\ln a"))
        r3 = self.deformar(r2, A, color=pasar(tonos[1], tonos[2], 1, A))
        l3 = MathTex(r"\ln a", font_size=30, color=tonos[1]).move_to(ax.c2p(6.0, 0.45)).set_z_index(4)
        ra3 = self.rotulo_t(A ** 3, "a^3")
        self.play(FadeIn(l3), FadeIn(ra3))

        p1 = MathTex(r"\ln(a^3)", "=", r"\cA{\ln a}+\cA{\ln a}+\cA{\ln a}", font_size=40)
        ajustar(p1, ANCHO_COL).move_to([X_COL, 2.4, 0])
        p2 = MathTex("=", r"3\,\cA{\ln a}", font_size=40)
        alinear_igual(p2, p1[1].get_x(), 1.65, i=0)
        p4 = MathTex(rf"\ln {A ** 3:g}", "=", f"{ln(A ** 3):.3f}", "=", rf"3\cdot\cA{{{ln(A):.3f}}}", font_size=38)
        alinear_igual(p4, p1[1].get_x(), 0.9)
        meter(p1, p2, p4)
        p3 = MathTex(r"\ln(a^n) = n\,\cA{\ln a}", font_size=48).move_to([X_COL, -0.6, 0])
        k3 = caja(p3)
        self.play(Write(p1))
        self.play(Write(p2))
        self.play(Write(p4))
        self.cambiar_leyenda(leyenda("cada copia es la anterior transformada por", "$a{:}", "todas tienen área",
                                     "$\ln a"))
        self.wait(1.5)
        self.play(Write(p3), Create(k3))
        self.wait(2.5)

        # inversos
        self.play(FadeOut(VGroup(r2, r3, l2, l3, ra2, ra3, p1, p2, p3, k3, p4)))
        self.cambiar_leyenda(leyenda("al revés: transformamos la región de", "$1", "a", "$a", "por", "$1/a"))
        r0 = self.deformar(self.regA, 1 / A, color=pasar(C_A, C_NEG, 1, 1 / A), run_time=3)
        l0 = MathTex(r"-\ln a", font_size=28).rotate(PI / 2).move_to(ax.c2p(0.75, 0.6)).set_z_index(4)
        r1a = self.rotulo_t(1 / A, "1/a")
        self.play(FadeIn(l0), FadeIn(r1a))
        self.cambiar_leyenda(leyenda("queda de", "$1/a", "a", "$1", "con la misma área, pero a la izquierda de",
                                     "$1", "resta"))
        q1 = MathTex(r"\ln(1/a)", "=", r"\cN{-\ln a}", font_size=50).move_to([X_COL, 2.2, 0])
        q2 = MathTex(rf"\ln {1 / A:g}", "=", rf"\cN{{{ln(1 / A):.3f}}}", "=", rf"-\cA{{\ln {A:g}}}", font_size=42)
        alinear_igual(q2, q1[1].get_x(), 1.2)
        meter(q1, q2)
        self.play(Write(q1))
        self.play(Write(q2))
        self.wait(3.5)

    # -- 6. e: donde el área vale 1 -----------------------------------------
    def acto_e(self):
        ax = self.ax
        self.limpiar(ax, self.rot_ejes, self.hip, self.linea1, self.rot_hip)
        self.seccion("$e{:}", "donde el área vale", "$1")
        x = ValueTracker(1.0)
        reg = self.region_viva(x.get_value)
        marca = self.marca_x(x.get_value)
        fx = fila(None, ["x", "=", "0.000"], {2: (x.get_value, 3)}).move_to([X_COL, 2.3, 0])
        fa = fila("área", ["=", "0.000"], {1: (lambda: ln(x.get_value()), 3)})
        alinear_igual(fa, fx[1].get_x(), 1.6)
        L = 3.6
        pista = Rectangle(width=L, height=0.32).set_stroke(GREY_B, 2).move_to([X_COL, 0.75, 0])

        def dib_barra():
            v = min(1.0, max(ln(x.get_value()), 0.0))
            r = Rectangle(width=max(L * v, 1e-3), height=0.32).set_stroke(width=0)
            r.set_fill(C_LN if v < 1 - 1e-9 else C_UNO[0], 1)
            return r.align_to(pista, LEFT).set_y(pista.get_y())

        barra = always_redraw(dib_barra)
        tope = Line(pista.get_corner(UR) + UP * 0.12, pista.get_corner(DR) + DOWN * 0.12).set_stroke(WHITE, 3)
        m0 = MathTex("0", font_size=28, color=C_TEXTO).next_to(pista.get_corner(DL), DOWN, buff=0.15)
        m1 = MathTex("1", font_size=28).next_to(tope, DOWN, buff=0.05)
        ff = fila("falta", ["0.000"], {0: (lambda: max(0.0, 1 - ln(x.get_value())), 3)}).move_to([X_COL, -0.3, 0])
        self.add(reg)
        self.play(FadeIn(VGroup(marca, fx, fa, pista, tope, m0, m1, ff)), FadeIn(barra))
        self.cambiar_leyenda(leyenda("¿hasta dónde hay que llevar", "$x", "para que el área valga", "$1?"))
        self.wait(0.5)
        self.play(x.animate.set_value(E), run_time=7, rate_func=rate_functions.ease_out_cubic)
        self.congelar(reg, marca, fx, fa, barra, ff)

        lab_e = MathTex(r"x = \cU{e} \approx " + f"{E:.3f}", font_size=36).next_to(ax.c2p(E, 1 / E), UR, buff=0.12)
        re = self.rotulo_t(E, "e", color=C_UNO[0], marca=True)
        self.play(Flash(ax.c2p(E, 0), color=C_UNO[0], line_length=0.3, flash_radius=0.3, num_lines=14),
                  reg.animate.set_fill(C_UNO[0]).set_stroke(C_UNO[0]), barra.animate.set_fill(C_UNO[0]), run_time=1.2)
        self.play(Write(lab_e), FadeIn(re))
        self.cambiar_leyenda(leyenda("frena justo en", "$x = e{:}", "el número donde el área desde", "$1",
                                     "vale", "$1"))
        self.wait(2.5)

        self.play(FadeOut(VGroup(fx, fa, pista, barra, tope, m0, m1, ff, marca)))
        f1 = MathTex(r"\int_1^{\cU{e}} \frac{dt}{t}", "=", "1", font_size=44).move_to([X_COL, 2.3, 0])
        f2 = MathTex(r"\cU{e}", r"\approx", f"{E:.3f}", font_size=44)
        alinear_igual(f2, f1[1].get_x(), 1.35)
        uno1 = MathTex("1", font_size=34).move_to(ax.c2p(1.7, 0.28)).set_z_index(4)
        self.play(Write(f1), Write(f2), FadeOut(lab_e), FadeIn(uno1))
        self.wait(1)

        self.cambiar_leyenda(leyenda("transformada por", "$e{,}", "la región de", "$1", "a", "$e", "se apila de",
                                     "$e", "a", "$e^2"))
        r2 = self.deformar(reg, E, color=lambda s: interpolate_color(ManimColor(C_UNO[0]), ManimColor(C_UNO[1]),
                                                                     (s - 1) / (E - 1)), run_time=3.5)
        uno2 = MathTex("1", font_size=30).move_to(ax.c2p(3.9, 0.12)).set_z_index(4)
        re2 = self.rotulo_t(E2, "e^2", color=C_UNO[1], marca=True)
        f3 = MathTex(r"\int_{\cU{e}}^{\cV{e^2}} \frac{dt}{t}", "=", "1", font_size=44)
        alinear_igual(f3, f1[1].get_x(), 0.2)
        f4 = MathTex(r"\cV{e^2}", r"\approx", f"{E2:.3f}", font_size=44)
        alinear_igual(f4, f1[1].get_x(), -0.75)
        self.play(FadeIn(uno2), FadeIn(re2), Write(f3))
        self.play(Write(f4))
        self.cambiar_leyenda(leyenda("espaciadas multiplicando en", "$t{,}", "pero cada una con área", "$1"))
        self.wait(2)
        f5 = MathTex(r"\ln e^n = n", font_size=50).move_to([X_COL, -2.0, 0])
        self.play(Write(f5), Create(caja(f5)))
        self.cambiar_leyenda(leyenda(r"$\ln e^n = n{:}", "el logaritmo cuenta cuántas veces cabe", "$e"))
        self.wait(3.5)

    # -- 7. el gráfico de ln y su derivada ----------------------------------
    def acto_grafico(self):
        self.limpiar(self.ax, self.rot_ejes, self.hip, self.linea1)
        self.seccion("El gráfico de", r"$\ln x", "y su derivada")
        ax = ejes(5.4, 3.0, O_IZQ, numeros=20)
        nuevos = VGroup(ax, rotulos_ejes(ax, font_size=28), hiperbola(ax, 3.5), linea_uno(ax))
        self.play(ReplacementTransform(VGroup(self.ax, self.rot_ejes, self.hip, self.linea1), nuevos), run_time=1.5)
        self.remove(nuevos)
        self.add(*nuevos)
        self.ax, self.rot_ejes, self.hip, self.linea1 = nuevos
        axl = ejes(5.4, 3.5, O_LN, numeros=20, y_range=(-1, 2.5, 1), y_numeros=(-1, 1, 2))
        rotl = rotulos_ejes(axl, "x", "y", font_size=28)
        sub_i = renglon("área bajo", "$1/t", "de", "$1", "a", "$x", font_size=26).move_to([ax.c2p(4, 0)[0], 2.45, 0])
        sub_d = renglon("gráfico de", r"$y = \ln x", font_size=26).move_to([axl.c2p(4, 0)[0], 2.45, 0])
        self.play(Create(axl), FadeIn(rotl), FadeIn(sub_i), FadeIn(sub_d), run_time=1.5)

        x = ValueTracker(1.0)
        self.lo = self.hi = 1.0

        def visitado():
            v = x.get_value()
            self.lo, self.hi = min(self.lo, v), max(self.hi, v)
            return [self.lo, max(self.hi, self.lo + 1e-3), 0.01]

        def color_signo():
            return C_LN if x.get_value() >= 1 else C_NEG

        reg = self.region_viva(x.get_value)
        marca = self.marca_x(x.get_value)
        graf = always_redraw(lambda: axl.plot(np.log, x_range=visitado(), color=C_LN, stroke_width=4).set_z_index(3))
        seg = always_redraw(lambda: Line(axl.c2p(x.get_value(), 0), axl.c2p(x.get_value(), ln(x.get_value()) + 1e-4))
                            .set_stroke(color_signo(), 6).set_z_index(4))
        pto = always_redraw(lambda: Dot(axl.c2p(x.get_value(), ln(x.get_value())), radius=0.06,
                                        color=WHITE).set_z_index(5))
        fi = fila("área", ["=", "0.000"], {1: (lambda: ln(x.get_value()), 3)}).move_to([ax.c2p(4, 0)[0], -2.75, 0])
        fd = fila("altura", ["=", "0.000"], {1: (lambda: float(np.log(x.get_value())), 3)})
        fd.move_to([axl.c2p(4, 0)[0], -2.75, 0])
        for f in (fi, fd):
            f.numeros[1].add_updater(lambda m: m.set_color(WHITE if x.get_value() >= 1 else C_NEG))
        self.add(reg, graf, seg, pto)
        self.play(FadeIn(marca), FadeIn(fi), FadeIn(fd))
        self.cambiar_leyenda(leyenda("la altura del gráfico de", r"$\ln", "es el área de la izquierda"))
        self.play(x.animate.set_value(7.5), run_time=5, rate_func=smooth)
        self.play(x.animate.set_value(0.4), run_time=5, rate_func=smooth)
        self.play(x.animate.set_value(2.0), run_time=2.5, rate_func=smooth)
        self.wait(0.5)

        # la franja de ancho dx y lo que sube el gráfico
        franja = always_redraw(lambda: region(ax, x.get_value(), x.get_value() + DX, C_RES, opacidad=0.9)
                               .set_z_index(1))

        def esquina():
            v = x.get_value()
            return axl.c2p(v, ln(v)), axl.c2p(v + DX, ln(v)), axl.c2p(v + DX, ln(v + DX))

        corre = always_redraw(lambda: DashedLine(*esquina()[:2], dash_length=0.05).set_stroke(GREY_B, 2.5)
                              .set_z_index(4))
        sube = always_redraw(lambda: Line(*esquina()[1:]).set_stroke(C_RES, 5).set_z_index(4))
        br_dx = Brace(Line(ax.c2p(2, 0.55), ax.c2p(2 + DX, 0.55)), UP, buff=0.02, color=C_RES)
        lab_dx = MathTex("dx", font_size=26, color=C_RES).next_to(br_dx, UP, buff=0.05)
        br_h = Brace(franja, LEFT, buff=0.04, color=WHITE).set_z_index(4)
        lab_h = MathTex("1/x", font_size=26).next_to(br_h, LEFT, buff=0.05).set_z_index(4)
        lab_s = MathTex(r"\approx dx/x", font_size=26, color=C_RES).next_to(sube, RIGHT, buff=0.1)
        self.congelar(marca)
        self.play(FadeOut(marca), FadeIn(franja), GrowFromCenter(br_dx), FadeIn(lab_dx), GrowFromCenter(br_h),
                  FadeIn(lab_h))
        self.cambiar_leyenda(leyenda("una franja fina de ancho", "$dx", "y altura", "$1/x", "agrega un área",
                                     r"$\approx dx/x"))
        self.wait(1.5)
        self.play(Create(corre), Create(sube), FadeIn(lab_s))
        self.cambiar_leyenda(leyenda("y el gráfico de", r"$\ln", "sube eso mismo cuando", "$x", "avanza", "$dx"))
        self.wait(2)
        der = MathTex(r"\frac{d}{dx}\ln x = \frac{1}{x}", font_size=44).move_to([2.75, 0.95, 0])
        k_der = caja(der, C_RES)
        self.play(Write(der), Create(k_der))
        self.cambiar_leyenda(leyenda("la pendiente de", r"$\ln", "en", "$x", "es la altura", "$1/x", "de la franja"))
        self.wait(2)

        # la tangente sigue a x
        tang = always_redraw(lambda: self.tangente(axl, x.get_value()))
        fi2 = fila(None, ["1/x", "=", "0.000"], {2: (lambda: 1 / x.get_value(), 3)}).move_to(fi)
        fd2 = fila("pendiente", ["=", "0.000"], {1: (lambda: derivada(ln, x.get_value()), 3)}).move_to(fd)
        self.play(FadeOut(VGroup(br_dx, lab_dx, br_h, lab_h, lab_s)), FadeOut(fi), FadeOut(fd), FadeIn(fi2),
                  FadeIn(fd2), Create(tang))
        self.play(x.animate.set_value(5.0), run_time=3.5, rate_func=smooth)
        self.play(x.animate.set_value(1.2), run_time=3.5, rate_func=smooth)
        self.play(x.animate.set_value(2.5), run_time=2, rate_func=smooth)
        self.wait(2)

    @staticmethod
    def tangente(axl, v, medio=0.9):
        m = derivada(ln, v)
        t0, t1 = max(v - medio, 0.15), v + medio
        return Line(axl.c2p(t0, ln(v) + m * (t0 - v)), axl.c2p(t1, ln(v) + m * (t1 - v))).set_stroke(C_RES, 3.5) \
            .set_z_index(4)

    # -- 8. cierre ----------------------------------------------------------
    def acto_cierre(self):
        self.limpiar(self.ax, self.rot_ejes, self.hip, self.linea1)
        self.seccion("En resumen")
        izq = VGroup(self.ax, self.rot_ejes, self.hip, self.linea1)
        o0 = self.ax.c2p(0, 0)
        self.play(izq.animate.scale(1.2, about_point=o0).shift(O_FIN - o0), run_time=1.2)
        ax = self.ax
        regA, regB = self.region(1, A, C_A), self.region(A, A * B, C_B)
        lab_a = MathTex(r"\ln a", font_size=28).move_to(ax.c2p(1.45, 0.3)).set_z_index(4)
        lab_b = MathTex(r"\ln b", font_size=28).move_to(ax.c2p(3.3, 0.14)).set_z_index(4)
        rot = VGroup(self.rotulo_t(A, "a"), self.rotulo_t(A * B, "ab"))
        rot_hip = MathTex(r"y = \frac{1}{t}", font_size=34).next_to(ax.c2p(0.6, 2.5), RIGHT, buff=0)
        self.play(FadeIn(regA), FadeIn(regB), FadeIn(lab_a), FadeIn(lab_b), FadeIn(rot), FadeIn(rot_hip))

        c1 = MathTex(r"\cL{\ln x} = \int_1^x \frac{dt}{t}", font_size=46)
        c2 = MathTex(r"\ln(ab) = \cA{\ln a} + \cB{\ln b}", font_size=46)
        c3 = MathTex(r"\int_1^{\cU{e}} \frac{dt}{t} = 1", font_size=46)
        VGroup(c1, c2, c3).arrange(DOWN, buff=0.85).move_to([3.75, -0.1, 0])
        for c in (c1, c2, c3):
            self.play(Write(c), Create(caja(c)), run_time=1.3)
            self.wait(0.8)
        self.wait(4)
