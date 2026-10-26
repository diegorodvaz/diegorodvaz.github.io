"""
sen x / x con áreas.

En la circunferencia unidad, con 0 < x < π/2, el triángulo OAP (área sen x / 2) está dentro del
sector OAP (área x / 2), que está dentro del triángulo OAT (área tg x / 2). De
sen x / 2 ≤ x / 2 ≤ tg x / 2 sale cos x ≤ sen x / x ≤ 1, y como cos x → 1 el sándwich da
lim sen x / x = 1. Las tres áreas tienden a 0; lo que importa es que sus cocientes tienden a 1,
y eso se ve estirando la figura verticalmente por 1/sen x: una transformación lineal multiplica
todas las áreas por lo mismo, así que no cambia los cocientes.

    manim -pql senx_sobre_x.py SenXSobreX
    manim -pqh senx_sobre_x.py SenXSobreX

Todo lo que aparece en pantalla sale de la geometría de abajo (vértices en función de x, shoelace)
y de un único ValueTracker de x. `python verificar_senx_sobre_x.py` la verifica.
"""
from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Geometría: todo lo que aparece en pantalla sale de acá
# ---------------------------------------------------------------------------
X0 = 0.9            # el ángulo de la figura (rad)
X_FIN = 0.02        # hasta dónde se achica
N_ARCO = 120        # puntos sobre el arco en los polígonos del sector y las franjas


def vertices(x):
    """O, A, P, T y H (el pie de la altura de P) en coordenadas de la figura (radio 1).
    T = (1, tg x) es donde la recta OP (v = u·tg x) corta a la tangente en A (u = 1)."""
    c, s = np.cos(x), np.sin(x)
    return {"O": (0.0, 0.0), "A": (1.0, 0.0), "P": (c, s), "T": (1.0, s / c), "H": (c, 0.0)}


def arco(t0, t1, n=N_ARCO):
    t = np.linspace(t0, t1, n)
    return list(zip(np.cos(t), np.sin(t)))


def contorno(cual, x, n=N_ARCO):
    """Vértices (u, v), en sentido antihorario, de cada región:
    'chico' = triángulo OAP, 'sector' = sector OAP, 'grande' = triángulo OAT,
    'segmento' = entre la cuerda AP y el arco, 'cuña' = entre el arco y los lados AT y TP."""
    v = vertices(x)
    if cual == "chico":
        return [v["O"], v["A"], v["P"]]
    if cual == "sector":
        return [v["O"]] + arco(0.0, x, n)
    if cual == "grande":
        return [v["O"], v["A"], v["T"]]
    if cual == "segmento":
        return arco(0.0, x, n)                          # se cierra por la cuerda PA
    if cual == "cuña":
        return [v["A"], v["T"]] + arco(x, 0.0, n)[:-1]  # A → T → P y por el arco de vuelta a A
    raise ValueError(cual)


def shoelace(pts):
    """Área del polígono de vértices pts = [(u, v), …] (positiva si es antihorario)."""
    x, y = np.asarray(pts, dtype=float).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def estirar(pts, k):
    """(u, v) ↦ (u, k·v): estira verticalmente por k (k = −1 refleja)."""
    return [(u, k * v) for u, v in pts]


def areas(x, k=1.0):
    return {c: shoelace(estirar(contorno(c, x), k)) for c in ("chico", "sector", "grande")}


def cocientes(x, k=1.0):
    """Área de cada región sobre el área del sector, medidas en la figura estirada por k."""
    a = areas(x, k)
    return a["chico"] / a["sector"], a["sector"] / a["sector"], a["grande"] / a["sector"]


def factor(x):
    """El estiramiento que deja al triángulo chico con altura 1."""
    return 1.0 / np.sin(x)


def sinc(t):
    return np.sin(t) / t


def mapa_vertical(o_ori, o_dest, k):
    """p ↦ o_dest + (Δu, k·Δv) con Δ = p − o_ori: estira (k > 1) o refleja (k = −1) la figura
    respecto del eje horizontal que pasa por o_ori, y la lleva a o_dest. Es afín: manda rectas en
    rectas y curvas de Bézier en las curvas de Bézier de los puntos de control transformados."""
    def f(p):
        return np.array([o_dest[0] + p[0] - o_ori[0], o_dest[1] + k * (p[1] - o_ori[1]), 0.0])
    return f


# ---------------------------------------------------------------------------
# Paleta: cada región con su color, usado también en su término, su barra y su curva
# ---------------------------------------------------------------------------
FONDO = "#0e1117"
config.background_color = FONDO
config.max_files_cached = 1000      # que la limpieza de caché no borre parciales de este mismo render
FW, FH = config.frame_width, config.frame_height

C_P = "#4cc9f0"             # triángulo chico OAP: sen x / 2, y el 1 del sándwich
C_S = "#ffd166"             # sector OAP: x / 2, y sen x / x
C_T = "#ff6b8b"             # triángulo grande OAT: tg x / 2, y cos x
C_RES = "#fff4d6"           # resaltados (franjas, semejanza)
C_EJE = "#6b7591"
C_CIRC = "#c3c9d6"
C_TEXTO = GREY_A
C_BANDA = "#2a3554"
COLOR = {"chico": C_P, "sector": C_S, "grande": C_T}
RELLENO = {k: interpolate_color(ManimColor(FONDO), ManimColor(c), 0.42) for k, c in COLOR.items()}


def _hex(c):
    return ManimColor(c).to_hex()[1:].upper()


PLANTILLA = TexTemplate()
PLANTILLA.add_to_preamble(
    r"\usepackage{xcolor}"
    r"\newcommand{\sen}{\operatorname{sen}}"
    r"\newcommand{\tg}{\operatorname{tg}}"
    + "".join(rf"\newcommand{{\{n}}}[1]{{\textcolor[HTML]{{{_hex(c)}}}{{#1}}}}"
              for n, c in (("cP", C_P), ("cS", C_S), ("cT", C_T))))
config.tex_template = PLANTILLA

# ---------------------------------------------------------------------------
# Layout (16:9, 14.2 × 8)
# ---------------------------------------------------------------------------
Y_TIT, Y_LEY = 3.5, -3.55
O_FIG, R_FIG = (-4.8, -1.75), 3.0           # actos 2–5: la figura grande a la izquierda
X_COL, ANCHO_COL = 3.6, 6.4                 # columna derecha de fórmulas
X_EQ = 2.5                                  # los "=" de las áreas (acto 3)
X_LAD, X_NOTA = 1.45, 3.7                   # escalera del álgebra y sus notas (acto 5)
Y_LAD = (2.55, 1.35, 0.1, -1.15, -2.4)
R_PANEL = 2.5                               # acto 6: los dos paneles
O_REAL, O_EST = (-6.5, -2.45), (-2.95, -2.45)
Y_PANEL = (-3.0, 2.85)
V_MAX = 1.7                                 # hasta dónde se dibuja (en v) dentro del panel estirado
V_ABAJO = -0.04                             # cuánto bajan del eje horizontal el eje vertical y la tangente
X_NUM, Y_NUM = 3.6, (-1.3, -2.1, -2.88)     # bloque de números en vivo
X_BARRA, L_BARRA, ALTO_BARRA = 2.65, 2.55, 0.3
Y_BARRAS = (1.95, 1.25, 0.55)
O_GRAF = (3.75, -0.35)                      # acto 7: gráfico
O_PAR, R_PAR = (-4.9, 0.0), 2.1             # acto 8: la figura y su reflejo
O_FIN, R_FIN = (-6.0, -1.75), 2.7           # acto 9


def direccion(t):
    return np.array([np.cos(t), np.sin(t), 0.0])


def esquina(vertice, hacia1, hacia2, lado=0.17):
    """Marca de ángulo recto en `vertice`, entre las direcciones unitarias hacia1 y hacia2."""
    v = np.asarray(vertice)
    a, b, c = v + lado * hacia1, v + lado * (hacia1 + hacia2), v + lado * hacia2
    return VMobject().set_points_as_corners([a, b, c]).set_stroke(C_TEXTO, 2).set_z_index(6)


def caja_panel(o, r):
    izq, der = o[0] - 0.45, o[0] + r + 0.45
    rect = RoundedRectangle(width=der - izq, height=Y_PANEL[1] - Y_PANEL[0], corner_radius=0.15)
    return rect.move_to([(izq + der) / 2, sum(Y_PANEL) / 2, 0]).set_stroke(GREY_B, 1.5, opacity=0.55)


class Figura:
    """La figura en pantalla: se rearma en cada cuadro desde el tracker de x y desde su origen
    y su radio (también trackers, para poder moverla y achicarla)."""

    def __init__(self, x, o, r):
        self.x = x
        self.ox, self.oy, self.r = ValueTracker(o[0]), ValueTracker(o[1]), ValueTracker(r)

    def O(self):
        return np.array([self.ox.get_value(), self.oy.get_value(), 0.0])

    def R(self):
        return self.r.get_value()

    def p(self, u, v):
        return self.O() + self.R() * np.array([u, v, 0.0])

    def pt(self, nombre):
        return self.p(*vertices(self.x.get_value())[nombre])

    def mover(self, o, r):
        return [self.ox.animate.set_value(o[0]), self.oy.animate.set_value(o[1]), self.r.animate.set_value(r)]

    # -- piezas ---------------------------------------------------------------
    def region(self, cual, relleno=True, color=None):
        """El polígono de la región: relleno opaco (premezclado con el fondo, así las regiones apiladas muestran
        cada franja en su color) o solo el borde."""
        P = Polygon(*[self.p(u, v) for u, v in contorno(cual, self.x.get_value())])
        if relleno:
            return P.set_fill(color or RELLENO[cual], 1).set_stroke(width=0)
        return P.set_fill(opacity=0).set_stroke(color or COLOR[cual], 3.5)

    def ejes(self, vmin=V_ABAJO, vmax=1.15):
        return VGroup(Line(self.p(-0.14, 0), self.p(1.24, 0)), Line(self.p(0, vmin), self.p(0, vmax))) \
            .set_stroke(C_EJE, 2)

    def circ(self, t0=0.0, t1=PI / 2):
        return Arc(radius=self.R(), start_angle=t0, angle=t1 - t0, arc_center=self.O(), num_components=24) \
            .set_stroke(C_CIRC, 2.5)

    def tangente(self, vmin=V_ABAJO, vmax=1.42):
        return DashedLine(self.p(1, vmin), self.p(1, vmax), dash_length=0.08).set_stroke(C_EJE, 2)

    def segmento(self, a, b, color=C_TEXTO, ancho=3):
        return Line(self.pt(a), self.pt(b)).set_stroke(color, ancho)

    def punto(self, nombre):
        return Dot(self.pt(nombre), radius=0.05, color=WHITE)


# ---------------------------------------------------------------------------
# Texto: Text para el castellano, MathTex para la matemática, con línea de base común
# ---------------------------------------------------------------------------
DESCENDENTES = "gjpqy,;()¿?"


def tex(t, font_size, color):
    """MathTex de t con su línea de base anotada: se compila con una x de referencia delante
    (así "a^2" o "x/2," no se apoyan en el exponente ni en la coma) y se devuelve solo la parte t."""
    m = MathTex("x", t, font_size=font_size, color=color)
    parte = m[1]
    parte.base_rel = m[0].get_bottom()[1] - parte.get_bottom()[1]
    return parte


def linea_base(mob):
    """Altura de la línea de base. Con `tex`: la de la x de referencia. En un Text: mediana de los
    bordes inferiores de los glifos sin descendente. En otro MathTex: el borde inferior más repetido."""
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


def caja(mob, color=GREY_B, buff=0.2):
    return SurroundingRectangle(mob, color=color, buff=buff, corner_radius=0.12, stroke_width=2.5)


def colorear(mt, colores):
    """colores = {color: [índices de partes]}."""
    for c, idx in colores.items():
        for i in idx:
            mt[i].set_color(c)
    return mt


def centrar_en(mt, partes, x, y):
    """Corre mt para que el grupo de las partes dadas quede centrado en x y la ecuación en y."""
    mt.shift(RIGHT * (x - VGroup(*[mt[i] for i in partes]).get_x()))
    return mt.set_y(y)


# Las cuatro líneas del álgebra. Las fracciones van partidas (\frac{ | numerador | }{ | denominador | })
# para que TransformMatchingTex mueva numeradores y denominadores por separado. Los \relax y el
# \sen{x} no cambian lo que se ve: solo distinguen las claves de partes que no tienen que juntarse.
L0_PARTES = [r"\frac{", r"\sen x", r"}{2}", r"\le", r"\frac{", "x", "}{2}", r"\le", r"\frac{", r"\tg x", "}{2}"]
L1_PARTES = [r"\sen x", r"\le", "x", r"\le", r"\tg x"]
L1B_PARTES = [r"\frac{", r"\sen x", r"}{\relax", r"\sen{x}\relax", "}", r"\le",
              r"\frac{", "x", "}{", r"\sen{x}", "}", r"\le",
              r"\frac{", r"\tg x", r"}{\relax\relax", r"\sen{x}\relax\relax", "}"]
L2_PARTES = ["1", r"\le", r"\frac{", "x", "}{", r"\sen{x}", "}", r"\le", r"\frac{1}{\cos x}"]
L3_PARTES = [r"\cos x", r"\le", r"\frac{", r"\sen{x}", "}{", "x", "}", r"\le", "1"]
MEDIO = {"L0": [4, 5, 6], "L1": [2], "L1B": [6, 7, 8, 9, 10], "L2": [2, 3, 4, 5, 6], "L3": [2, 3, 4, 5, 6]}
COLORES = {"L0": {C_P: [0, 1, 2], C_S: [4, 5, 6], C_T: [8, 9, 10]},
           "L1": {C_P: [0], C_S: [2], C_T: [4]},
           "L1B": {C_P: [0, 1, 2, 3, 4], C_S: [6, 7, 8, 9, 10], C_T: [12, 13, 14, 15, 16]},
           "L2": {C_P: [0], C_S: [2, 3, 4, 5, 6], C_T: [8]},
           "L3": {C_T: [0], C_S: [2, 3, 4, 5, 6], C_P: [8]}}


def linea(nombre, partes, font_size=44):
    return colorear(MathTex(*partes, font_size=font_size), COLORES[nombre])


class TransformarPorClave(TransformMatchingTex):
    """TransformMatchingTex que empareja cada parte por su atributo `clave` si lo tiene (si no, por su TeX).
    Reemplaza a key_map, que en 0.22 pasa por FadeTransformPieces y falla si las familias difieren."""

    @staticmethod
    def get_mobject_key(mobject):
        return getattr(mobject, "clave", mobject.tex_string)


# ---------------------------------------------------------------------------
# La escena
# ---------------------------------------------------------------------------
class SenXSobreX(Scene):
    def construct(self):
        self.caption = None
        self.titulo = None
        self.x = ValueTracker(X0)
        self.fig = Figura(self.x, O_FIG, R_FIG)
        self.acto_titulo()
        self.acto_figura()
        self.acto_areas()
        self.acto_encierro()
        self.acto_algebra()
        self.acto_pegan()
        self.acto_sandwich()
        self.acto_negativo()
        self.acto_cierre()

    # -- utilidades ---------------------------------------------------------
    def xv(self):
        return self.x.get_value()

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
        self.congelar(*salen)
        if salen:
            self.play(*[FadeOut(m) for m in salen], run_time=run_time)
        self.remove(*[m for m in self.mobjects if id(m) not in quedan])
        self.caption = self.titulo = None

    @staticmethod
    def congelar(*mobs):
        for m in mobs:
            for x in m.get_family():
                x.clear_updaters()

    def sacar(self, *mobs, run_time=0.8, **kw):
        self.congelar(*mobs)
        self.play(*[FadeOut(m) for m in mobs], *kw.pop("extra", []), run_time=run_time)

    def rotulo(self, tex_, donde, desplazamiento, font_size=36, color=WHITE):
        """Rótulo que sigue a un punto de la figura: centro en donde() + desplazamiento·R/R_FIG, y se
        escala con el radio (así, al achicar la figura, todo se achica junto)."""
        fig = self.fig
        m = MathTex(tex_, font_size=font_size, color=color).set_z_index(9)
        m.r_act = R_FIG
        d = np.array([desplazamiento[0], desplazamiento[1], 0.0])

        def upd(mm):
            r = fig.R()
            mm.scale(r / mm.r_act)
            mm.r_act = r
            mm.move_to(donde() + d * r / R_FIG)

        upd(m)
        m.add_updater(upd)
        return m

    def rotulo_fijo(self, tex_, punto, desplazamiento, r, font_size=36, color=WHITE):
        return MathTex(tex_, font_size=font_size * r / R_FIG, color=color).set_z_index(9) \
            .move_to(np.asarray(punto) + np.array([desplazamiento[0], desplazamiento[1], 0.0]) * r / R_FIG)

    def angulo(self, o, r, x, signo=1, nombre="x"):
        """Arco del ángulo en O (de 0 a signo·x) y su rótulo, quietos."""
        arc = Arc(radius=0.25 * r, start_angle=0, angle=signo * x, arc_center=o).set_stroke(WHITE, 2.5)
        lab = MathTex(nombre, font_size=36 * r / R_FIG).move_to(o + 0.37 * r * direccion(signo * x / 2))
        return VGroup(arc.set_z_index(6), lab.set_z_index(9))

    def fila_area(self, nombre, color, partes, y):
        """'nombre = fórmula', con el primer '=' en X_EQ (nombre = partes de `renglon`)."""
        m = MathTex(*partes, font_size=44)
        m.shift([X_EQ - m[0].get_x(), y - m.get_y(), 0])
        # la línea de base de la fórmula, medida desde el "=" con una x de referencia
        ref = MathTex("x", "=", font_size=44)
        base = m[0].get_bottom()[1] - (ref[1].get_bottom()[1] - ref[0].get_bottom()[1])
        t = renglon(*nombre, font_size=30, color=color, buff=0.12).next_to(m, LEFT, buff=0.25)
        t.shift(UP * (base - linea_base(t[0])))
        return VGroup(t, m), t, m

    # -- 1. título ----------------------------------------------------------
    def acto_titulo(self):
        titulo = renglon(r"$\frac{\sen x}{x}", "con áreas", font_size=60, color=WHITE, buff=0.35)
        self.play(FadeIn(titulo, shift=UP * 0.3), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(titulo), run_time=0.8)

    # -- 2. la figura -------------------------------------------------------
    def acto_figura(self):
        fig = self.fig
        self.seccion("La figura")
        self.ejes = always_redraw(fig.ejes).set_z_index(0)
        self.circ = always_redraw(fig.circ).set_z_index(1)
        self.dO = always_redraw(lambda: fig.punto("O")).set_z_index(8)
        self.dA = always_redraw(lambda: fig.punto("A")).set_z_index(8)
        self.lO = self.rotulo("O", lambda: fig.pt("O"), (-0.24, -0.26))
        self.lA = self.rotulo("A", lambda: fig.pt("A"), (0.22, -0.28))
        self.play(Create(self.ejes), run_time=1.2)
        self.play(Create(self.circ), FadeIn(self.dO), FadeIn(self.lO), FadeIn(self.dA), FadeIn(self.lA), run_time=1.6)
        self.cambiar_leyenda(leyenda("la circunferencia de radio", "$1{,}", "con centro en", "$O"))
        self.wait(1.5)

        # el ángulo x y el punto P
        self.rOP = always_redraw(lambda: fig.segmento("O", "P")).set_z_index(4)
        self.dP = always_redraw(lambda: fig.punto("P")).set_z_index(8)
        dP = 0.38 * direccion(X0 + PI / 4)
        self.lP = self.rotulo("P", lambda: fig.pt("P"), (dP[0], dP[1]))
        self.ang = self.angulo(fig.O(), fig.R(), X0)
        self.play(Create(self.rOP), run_time=1.0)
        self.play(Create(self.ang[0]), FadeIn(self.ang[1]), FadeIn(self.dP), FadeIn(self.lP))
        self.cambiar_leyenda(leyenda("un ángulo", "$x", "entre", "$0", "y", r"$\pi/2{,}", "medido en radianes"))
        self.wait(2)

        # OA = 1
        O, A, P, H, T = (fig.pt(n) for n in "OAPHT")
        self.lsen = MathTex(r"\sen x", font_size=32).move_to([H[0] - 0.46, (P[1] + H[1]) / 2, 0]).set_z_index(9)
        self.br_cos = Brace(Line(O, H), DOWN, buff=0.1, color=C_TEXTO)
        self.lcos = MathTex(r"\cos x", font_size=32).next_to(self.br_cos, DOWN, buff=0.06)
        y1 = self.lcos.get_bottom()[1] - 0.06
        self.br_1 = Brace(Line([O[0], y1, 0], [A[0], y1, 0]), DOWN, buff=0.0, color=C_TEXTO)
        self.l1 = MathTex("1", font_size=32).next_to(self.br_1, DOWN, buff=0.06)
        self.play(GrowFromCenter(self.br_1), FadeIn(self.l1))
        self.cambiar_leyenda(leyenda("$OA", "es un radio: mide", "$1"))
        self.wait(1.5)

        # la altura de P y su proyección
        self.alt = DashedLine(P, H, dash_length=0.07).set_stroke(C_TEXTO, 2.5).set_z_index(6)
        self.proy = Line(O, H).set_stroke(WHITE, 5).set_z_index(6)
        self.recto_H = esquina(H, LEFT, UP)
        self.play(Create(self.alt), Create(self.recto_H), FadeIn(self.lsen))
        self.play(Create(self.proy), GrowFromCenter(self.br_cos), FadeIn(self.lcos))
        self.cambiar_leyenda(leyenda(r"$P = (\cos x,\ \sen x){:}", "la altura de", "$P", "es", r"$\sen x", "y su proyección,",
                                     r"$\cos x"))
        self.wait(2.5)

        # la tangente en A y el punto T
        self.tang = always_redraw(fig.tangente).set_z_index(0)
        self.recto_A = esquina(A, LEFT, UP)
        self.play(Create(self.tang), Create(self.recto_A), run_time=1.2)
        self.cambiar_leyenda(leyenda("la tangente por", "$A", "es perpendicular al radio", "$OA"))
        self.wait(1.2)
        self.rPT = always_redraw(lambda: fig.segmento("P", "T")).set_z_index(4)
        self.rAT = always_redraw(lambda: fig.segmento("A", "T")).set_z_index(4)
        self.dT = always_redraw(lambda: fig.punto("T")).set_z_index(8)
        self.lT = self.rotulo("T", lambda: fig.pt("T"), (0.32, 0.24))
        self.play(Create(self.rPT), run_time=1.0)
        self.play(FadeIn(self.dT), FadeIn(self.lT), Create(self.rAT))
        self.cambiar_leyenda(leyenda("la recta", "$OP", "corta a la tangente en", "$T"))
        self.wait(1.5)

        # semejanza: el triángulo de catetos cos x, sen x, agrandado desde O por 1/cos x, es OAT
        t1 = Polygon(O, H, P).set_stroke(C_RES, 4).set_fill(C_RES, 0.12).set_z_index(7)
        self.play(Create(t1))
        self.cambiar_leyenda(leyenda("el triángulo de catetos", r"$\cos x", "y", r"$\sen x", "y el", "$OAT",
                                     "comparten el ángulo", "$x", "y tienen un ángulo recto"))
        t2 = t1.copy()
        self.add(t2)
        self.play(t2.animate.scale(1 / np.cos(X0), about_point=O), run_time=2)
        self.cambiar_leyenda(leyenda("son semejantes: uno es el otro agrandado desde", "$O"))
        sem = MathTex(r"\frac{AT}{OA}", "=", r"\frac{\sen x}{\cos x}", "=", r"\tg x", font_size=46)
        sem.move_to([X_COL, 1.2, 0])
        sem2 = MathTex(r"OA = 1", r"\quad\Longrightarrow\quad", r"AT = \tg x", font_size=46).move_to([X_COL, -0.1, 0])
        meter(sem, sem2)
        self.play(Write(sem), run_time=1.6)
        self.wait(1.5)
        self.br_tg = Brace(Line(A, T), RIGHT, buff=0.08, color=C_TEXTO)
        self.ltg = MathTex(r"\tg x", font_size=32).next_to(self.br_tg, RIGHT, buff=0.08)
        self.play(Write(sem2), GrowFromCenter(self.br_tg), FadeIn(self.ltg), run_time=1.4)
        self.cambiar_leyenda(leyenda("como", "$OA = 1{,}", "el segmento", "$AT", "mide", r"$\tg x"))
        self.wait(2.5)
        self.play(FadeOut(VGroup(t1, t2, sem, sem2)))

    # -- 3. las tres áreas --------------------------------------------------
    def acto_areas(self):
        fig = self.fig
        self.seccion("Tres áreas")
        self.rel = {k: always_redraw(lambda k=k: fig.region(k)).set_z_index(z)
                    for k, z in (("grande", 2), ("sector", 3), ("chico", 3.5))}
        self.bor = {k: always_redraw(lambda k=k: fig.region(k, relleno=False)).set_z_index(z)
                    for k, z in (("grande", 5), ("sector", 5.1), ("chico", 5.2))}
        f1, n1, m1 = self.fila_area(("triángulo", "$OAP"), C_P, ["=", r"\frac{1\cdot\sen x}{2}", "=", r"\frac{",
                                                          r"\sen x", "}{2}"], 2.0)
        f2, n2, m2 = self.fila_area(("sector", "$OAP"), C_S, ["=", r"\frac{x}{2\pi}\cdot\pi", "=", r"\frac{", "x", "}{2}"],
                                    0.55)
        f3, n3, m3 = self.fila_area(("triángulo", "$OAT"), C_T, ["=", r"\frac{1\cdot\tg x}{2}", "=", r"\frac{", r"\tg x",
                                                          "}{2}"], -1.15)
        chica = renglon("fracción", r"$x/2\pi", "del círculo, que tiene área", r"$\pi", font_size=22)
        chica.next_to(m2, DOWN, buff=0.22).align_to(m2, LEFT)
        meter(f1, f2, f3, chica)
        colorear(m1, {C_P: [3, 4, 5]})
        colorear(m2, {C_S: [3, 4, 5]})
        colorear(m3, {C_T: [3, 4, 5]})

        # triángulo chico: base 1, altura sen x
        self.play(FadeIn(self.rel["chico"]), Create(self.bor["chico"]), run_time=1.3)
        self.play(FadeIn(n1, shift=RIGHT * 0.2))
        self.cambiar_leyenda(leyenda("triángulo", "$OAP{:}", "base", "$OA = 1{,}", "altura", r"$\sen x"))
        self.play(Indicate(VGroup(self.br_1, self.l1), color=C_P), run_time=1.0)
        self.play(Indicate(VGroup(self.alt, self.lsen), color=C_P), run_time=1.0)
        self.play(Write(m1), run_time=1.5)
        self.wait(1.5)

        # sector: la fracción x/2π de un círculo de área π
        self.play(FadeOut(self.rel["chico"]), FadeIn(self.rel["sector"]), Create(self.bor["sector"]), run_time=1.3)
        self.play(FadeIn(n2, shift=RIGHT * 0.2))
        self.cambiar_leyenda(leyenda("sector", "$OAP{:}", "el ángulo", "$x", "es la fracción", r"$x/2\pi", "de la vuelta"))
        self.play(Indicate(self.ang, color=C_S, scale_factor=1.3), run_time=1.0)
        self.play(Write(m2), FadeIn(chica), run_time=1.5)
        self.wait(2.5)

        # triángulo grande: base 1, altura tg x
        self.play(FadeOut(self.rel["sector"]), FadeIn(self.rel["grande"]), Create(self.bor["grande"]), run_time=1.3)
        self.play(FadeIn(n3, shift=RIGHT * 0.2))
        self.cambiar_leyenda(leyenda("triángulo", "$OAT{:}", "base", "$OA = 1{,}", "altura", r"$AT = \tg x"))
        self.play(Indicate(VGroup(self.br_1, self.l1), color=C_T), run_time=1.0)
        self.play(Indicate(VGroup(self.br_tg, self.ltg), color=C_T), run_time=1.0)
        self.play(Write(m3), run_time=1.5)
        self.wait(2)
        self.filas = VGroup(f1, f2, f3, chica)
        self.resultados = [VGroup(*m[3:6]) for m in (m1, m2, m3)]

    # -- 4. el encierro -----------------------------------------------------
    def acto_encierro(self):
        fig = self.fig
        self.seccion("Una dentro de otra")
        self.play(FadeOut(VGroup(self.br_1, self.l1, self.br_cos, self.lcos, self.lsen, self.alt, self.proy,
                                 self.br_tg, self.ltg, self.recto_H, self.recto_A)), run_time=1.0)
        self.play(FadeIn(self.rel["sector"]), run_time=1.0)
        self.play(FadeIn(self.rel["chico"]), run_time=1.0)
        self.cambiar_leyenda(leyenda("el triángulo chico está dentro del sector, y el sector dentro del triángulo grande"))
        destellos = [ShowPassingFlash(fig.region(k, relleno=False).set_stroke(WHITE, 7).set_z_index(7), time_width=0.6)
                     for k in ("chico", "sector", "grande")]
        self.play(LaggedStart(*destellos, lag_ratio=0.55), run_time=3.2)
        self.wait(1)

        # las dos franjas que separan una región de la siguiente
        seg = fig.region("segmento", color=RELLENO["sector"]).set_z_index(3.6)
        cuna = fig.region("cuña", color=RELLENO["grande"]).set_z_index(3.6)
        self.add(seg, cuna)
        self.cambiar_leyenda(leyenda("al sector le sobra el segmento entre la cuerda", "$AP", "y el arco"))
        self.play(Indicate(seg, color=C_RES, scale_factor=1.0), run_time=1.6)
        self.play(Indicate(seg, color=C_RES, scale_factor=1.0), run_time=1.2)
        self.cambiar_leyenda(leyenda("y al triángulo grande, la zona entre el arco y los lados", "$AT", "y", "$TP"))
        self.play(Indicate(cuna, color=C_RES, scale_factor=1.0), run_time=1.6)
        self.play(Indicate(cuna, color=C_RES, scale_factor=1.0), run_time=1.2)
        self.remove(seg, cuna)

        # la desigualdad, con cada término en el color de su región
        L0 = linea("L0", L0_PARTES, font_size=48).move_to([X_COL, -2.4, 0])
        meter(L0)
        self.cambiar_leyenda(leyenda("una región dentro de otra tiene menos área:"))
        grupos = [VGroup(*L0[i:i + 3]) for i in (0, 4, 8)]
        for j, (res, g) in enumerate(zip(self.resultados, grupos)):
            self.play(TransformFromCopy(res, g), run_time=1.2)
            if j < 2:
                self.play(Write(L0[4 * j + 3]), run_time=0.5)
        self.remove(*grupos, L0[3], L0[7])
        self.add(L0)
        self.cambiar_leyenda(leyenda("área del triángulo chico", r"$\le", "área del sector", r"$\le",
                                     "área del triángulo grande"))
        self.wait(3)
        self.L0 = L0

    # -- 5. el álgebra ------------------------------------------------------
    def acto_algebra(self):
        self.seccion("El álgebra")
        self.play(FadeOut(self.filas), run_time=0.8)
        L0 = self.L0
        destino = L0.copy().scale(40 / 48)
        centrar_en(destino, MEDIO["L0"], X_LAD, Y_LAD[0])
        self.play(Transform(L0, destino), run_time=1.2)
        self.cambiar_leyenda(leyenda("vale para todo", "$x", "entre", "$0", "y", r"$\pi/2"))

        def nota(*renglones):
            return VGroup(*[renglon(*r, font_size=21) if isinstance(r, tuple) else Text(r, font_size=21, color=C_TEXTO)
                            for r in renglones]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        # claves: de qué parte de la línea anterior viene cada parte que cambia de forma
        # (sen x / sen x → 1, tg x / sen x → 1/cos x, y al invertir 1/cos x → cos x)
        pasos = [("L1", L1_PARTES, nota(("multiplicamos por", "$2")), {}),
                 ("L1B", L1B_PARTES, nota(("dividimos por", r"$\sen x > 0"),
                                          ("(es positivo para", r"$0 < x < \pi/2)")), {}),
                 ("L2", L2_PARTES, nota(("simplificamos:",), (r"$\tg x / \sen x = 1/\cos x",)),
                  {0: r"\sen x", 8: r"\tg x"}),
                 ("L3", L3_PARTES, nota("invertimos: como todo es positivo,", "se dan vuelta las desigualdades"),
                  {0: r"\frac{1}{\cos x}"})]
        # todas las notas con la misma escala, la que haga entrar a la más ancha
        esc = min(1.0, (FW / 2 - 0.15 - X_NOTA) / max(p[2].width for p in pasos))
        for p in pasos:
            p[2].scale(esc)
        previa = L0
        self.lineas, self.notas = [L0], []
        for j, (nombre, partes, nt, claves) in enumerate(pasos, start=1):
            nueva = centrar_en(linea(nombre, partes, font_size=40), MEDIO[nombre], X_LAD, Y_LAD[j])
            nt.move_to([0, Y_LAD[j], 0]).align_to([X_NOTA, 0, 0], LEFT)
            for i, c in claves.items():
                nueva[i].clave = c
            self.play(TransformarPorClave(previa.copy(), nueva), FadeIn(nt, shift=LEFT * 0.2), run_time=1.8)
            for i in claves:
                del nueva[i].clave
            self.wait(2.2)
            previa = nueva
            self.lineas.append(nueva)
            self.notas.append(nt)
        self.caja_L3 = caja(previa, color=WHITE)
        self.play(Create(self.caja_L3))
        self.cambiar_leyenda(leyenda(r"$\sen x / x", "queda encerrado entre", r"$\cos x", "y", "$1"))
        self.wait(3)
        self.L3 = previa

    # -- 6. se pegan ----------------------------------------------------------
    def s_est(self):
        return 1.0 + self.lam.get_value() * (factor(self.xv()) - 1.0)

    def O_est(self):
        return self.fig.O() + self.mu.get_value() * (np.array([O_EST[0], O_EST[1], 0.0]) - self.fig.O())

    def mapa_est(self):
        return mapa_vertical(self.fig.O(), self.O_est(), self.s_est())

    def piezas_estiradas(self):
        """La figura del panel derecho: la imagen de la real por (u, v) ↦ (u, v/sen x). Regiones, bordes y
        arco son la imagen punto a punto (apply_function) de las piezas reales; ejes y tangente se recortan
        al panel antes de transformarlos, y los puntos se ubican en la imagen de su centro."""
        def recorte(tope):
            return min(tope, V_MAX / self.s_est())

        def ejes():
            s = self.s_est()
            return self.fig.ejes(V_ABAJO / s, recorte(1.15)).apply_function(self.mapa_est())

        def circ():
            return self.fig.circ(0.0, float(np.arcsin(min(1.0, V_MAX / self.s_est())))).apply_function(self.mapa_est())

        def tangente():
            m, s = self.mapa_est(), self.s_est()
            return DashedLine(m(self.fig.p(1, V_ABAJO / s)), m(self.fig.p(1, recorte(1.42))), dash_length=0.08) \
                .set_stroke(C_EJE, 2)

        piezas = [always_redraw(ejes).set_z_index(0), always_redraw(tangente).set_z_index(0),
                  always_redraw(circ).set_z_index(1)]
        for k, z in (("grande", 2), ("sector", 3), ("chico", 3.5)):
            piezas.append(always_redraw(lambda k=k: self.fig.region(k).apply_function(self.mapa_est())).set_z_index(z))
        for k, z in (("grande", 5), ("sector", 5.1), ("chico", 5.2)):
            piezas.append(always_redraw(lambda k=k: self.fig.region(k, relleno=False).apply_function(self.mapa_est()))
                          .set_z_index(z))
        for n in "OAPT":
            piezas.append(always_redraw(lambda n=n: Dot(self.mapa_est()(self.fig.pt(n)), radius=0.05, color=WHITE))
                          .set_z_index(8))
        return piezas

    def barra(self, i, k):
        r = cocientes(self.xv(), self.s_est())[i]
        b = Rectangle(width=max(L_BARRA * r, 1e-3), height=ALTO_BARRA).set_fill(COLOR[k], 0.85).set_stroke(width=0)
        return b.move_to([X_BARRA + L_BARRA * r / 2, Y_BARRAS[i], 0]).set_z_index(2)

    def icono(self, k, esc=0.36):
        P = Polygon(*[np.array([esc * u, esc * v, 0.0]) for u, v in contorno(k, X0)])
        return P.set_fill(RELLENO[k], 1).set_stroke(COLOR[k], 2)

    def acto_pegan(self):
        fig = self.fig
        self.seccion("Las regiones se pegan")
        self.sacar(*self.lineas[:-1], *self.notas, self.caja_L3, self.ang, self.lP, self.lT, run_time=1.0)

        # la desigualdad final pasa al bloque de números; la figura, al panel izquierdo
        filaB = colorear(MathTex(r"\cos x", r"\quad", r"\le", r"\quad", r"\frac{", r"\sen{x}", "}{", "x", "}",
                                 r"\quad", r"\le", r"\quad", "1", font_size=40), {C_T: [0], C_S: [4, 5, 6, 7, 8], C_P: [12]})
        filaB.move_to([X_NUM, Y_NUM[1], 0])
        self.play(*fig.mover(O_REAL, R_PANEL), TransformMatchingTex(self.L3, filaB), run_time=2.2)
        self.caja1 = caja_panel(O_REAL, R_PANEL)
        self.cap1 = Text("la figura", font_size=24, color=C_TEXTO).move_to([self.caja1.get_x(), 2.5, 0])
        self.play(Create(self.caja1), FadeIn(self.cap1), run_time=1.0)

        # números en vivo: x arriba; cos x y sen x / x debajo de su término
        fx = fila(None, ["x", "=", "0.000"], {2: (self.xv, 3)}).move_to([X_NUM, Y_NUM[0], 0])
        muestra = MathTex("0.0000", r"\le", "0.0000", r"\le", "1", font_size=36).move_to([X_NUM, Y_NUM[2], 0])
        refs = [filaB[0], filaB[2], VGroup(*filaB[4:9]), filaB[10], filaB[12]]
        colores = [C_T, C_TEXTO, C_S, C_TEXTO, C_P]
        vivos = {0: lambda: np.cos(self.xv()), 2: lambda: sinc(self.xv())}
        filaC = VGroup()
        for i, (parte, ref) in enumerate(zip(muestra, refs)):
            cx = ref.get_x()
            if i in vivos:
                n = DecimalNumber(vivos[i](), num_decimal_places=4, font_size=36, color=colores[i]).move_to(parte)
                n.set_x(cx)
                n.add_updater(lambda mm, f=vivos[i], cx=cx: mm.set_value(f()).set_x(cx))
                filaC.add(n)
            else:
                filaC.add(parte.set_x(cx).set_color(colores[i]))
        self.bloque = VGroup(fx, filaB, filaC)
        self.play(FadeIn(fx), FadeIn(filaC), run_time=1.0)
        self.cambiar_leyenda(leyenda("achicamos", "$x{:}", "las tres áreas tienden a", "$0"))
        self.play(self.x.animate.set_value(X_FIN), run_time=5, rate_func=smooth)
        self.cambiar_leyenda(leyenda("en la figura real todo se vuelve una astilla junto a", "$A{:}",
                                     "no se ve el cociente"))
        self.wait(2.5)
        self.play(self.x.animate.set_value(X0), run_time=2.5, rate_func=smooth)

        # la misma figura, estirada verticalmente por 1/sen x
        self.mu, self.lam = ValueTracker(0.0), ValueTracker(0.0)
        self.estiradas = self.piezas_estiradas()
        self.add(*self.estiradas)
        self.caja2 = caja_panel(O_EST, R_PANEL)
        self.cap2 = renglon("estirada por", r"$1/\sen x", font_size=24).move_to([self.caja2.get_x(), 2.5, 0])
        self.cambiar_leyenda(leyenda("una copia de la figura"))
        self.play(self.mu.animate.set_value(1.0), Create(self.caja2), run_time=1.8)
        self.cambiar_leyenda(leyenda("la estiramos verticalmente por", r"$1/\sen x{:}",
                                     "el triángulo chico queda con altura", "$1"))
        self.play(self.lam.animate.set_value(1.0), FadeIn(self.cap2), run_time=2.5, rate_func=smooth)
        self.fac = fila(None, [r"\times", "00.00"], {1: (self.s_est, 2)}, font_size=24).move_to(
            [self.caja2.get_x(), 2.07, 0])
        self.play(FadeIn(self.fac))
        self.cambiar_leyenda(leyenda("una transformación lineal multiplica todas las áreas por lo mismo:",
                                     "los cocientes no cambian"))
        self.wait(3)

        # barras: el área de cada región sobre la del sector
        titulo_b = Text("área de cada región ÷ área del sector", font_size=24, color=C_TEXTO).move_to([X_NUM, 2.65, 0])
        filas_b = VGroup()
        self.barras = VGroup()
        for i, (k, form) in enumerate((("chico", r"\frac{\sen x}{x}"), ("sector", "1"), ("grande", r"\frac{\tg x}{x}"))):
            lab = VGroup(self.icono(k), MathTex("/", font_size=34), self.icono("sector"), MathTex("=", font_size=34),
                         MathTex(form, font_size=30)).arrange(RIGHT, buff=0.09)
            lab.move_to([X_BARRA - 0.2 - lab.width / 2, Y_BARRAS[i], 0])
            filas_b.add(lab)
            self.barras.add(always_redraw(lambda i=i, k=k: self.barra(i, k)))
        borde_izq = Line([X_BARRA, Y_BARRAS[0] + 0.3, 0], [X_BARRA, Y_BARRAS[-1] - 0.3, 0]).set_stroke(C_EJE, 2)
        uno = DashedLine([X_BARRA + L_BARRA, Y_BARRAS[0] + 0.3, 0], [X_BARRA + L_BARRA, Y_BARRAS[-1] - 0.3, 0],
                         dash_length=0.07).set_stroke(WHITE, 2).set_z_index(3)
        m0 = MathTex("0", font_size=26, color=C_TEXTO).next_to(borde_izq, DOWN, buff=0.08)
        m1 = MathTex("1", font_size=26, color=C_TEXTO).next_to(uno, DOWN, buff=0.08)
        self.panel_barras = VGroup(titulo_b, filas_b, borde_izq, uno, m0, m1, self.barras)
        self.play(FadeIn(titulo_b), FadeIn(filas_b), Create(borde_izq), Create(uno), FadeIn(m0), FadeIn(m1),
                  run_time=1.2)
        self.play(*[GrowFromEdge(b, LEFT) for b in self.barras], run_time=1.2)
        self.cambiar_leyenda(leyenda("las barras miden los cocientes:", r"$\tfrac{\sen x}{x}{,}", "$1", "y",
                                     r"$\tfrac{\tg x}{x}"))
        self.wait(2.5)

        self.cambiar_leyenda(leyenda("ahora achicamos", "$x", "mirando la figura estirada"))
        self.play(self.x.animate.set_value(X_FIN), run_time=10, rate_func=smooth)
        self.cambiar_leyenda(leyenda("el arco se endereza,", "$T", "se acerca a", "$P", "y las tres regiones se vuelven la misma"))
        self.wait(3)
        self.cambiar_leyenda(leyenda("los cocientes tienden a", "$1{:}", r"$\tfrac{\sen x}{x} \to 1", "y",
                                     r"$\tfrac{\tg x}{x} \to 1"))
        self.wait(3)

    # -- 7. el sándwich en un gráfico ---------------------------------------
    def punto_curva(self, f, color, signo=1):
        ax = self.ax
        return always_redraw(lambda: Dot(ax.c2p(signo * self.xv(), f(signo * self.xv())), radius=0.07, color=color)
                             .set_stroke(FONDO, 1.5).set_z_index(7))

    def guia(self, signo=1):
        ax = self.ax
        return always_redraw(lambda: DashedLine(ax.c2p(signo * self.xv(), 0), ax.c2p(signo * self.xv(), 1),
                                                dash_length=0.06).set_stroke(GREY_B, 1.5, opacity=0.7).set_z_index(2))

    def acto_sandwich(self):
        self.seccion("El sándwich")
        self.sacar(self.panel_barras, run_time=0.8)
        ax = Axes(x_range=[-PI / 2, PI / 2, PI / 4], y_range=[0, 1.2, 0.2], x_length=5.8, y_length=3.0,
                  axis_config=dict(color=C_EJE, stroke_width=2, include_tip=True, tip_width=0.13, tip_height=0.13,
                                   include_ticks=False))
        ax.shift(np.array([O_GRAF[0], O_GRAF[1], 0]) - ax.c2p(0, 0)).set_z_index(1)
        self.ax = ax
        marcas = VGroup(*[Line(ax.c2p(t, 0) + DOWN * 0.07, ax.c2p(t, 0) + UP * 0.07).set_stroke(C_EJE, 2)
                          for t in (-PI / 2, PI / 2)])
        nums = VGroup(MathTex(r"-\pi/2", font_size=26, color=C_TEXTO).next_to(ax.c2p(-PI / 2, 0), DOWN, buff=0.14),
                      MathTex("0", font_size=26, color=C_TEXTO).next_to(ax.c2p(0, 0), DOWN, buff=0.14),
                      MathTex(r"\pi/2", font_size=26, color=C_TEXTO).next_to(ax.c2p(PI / 2, 0), DOWN, buff=0.14))
        rot = VGroup(MathTex("x", font_size=30, color=C_TEXTO).next_to(ax.x_axis.get_end(), RIGHT, buff=0.1),
                     MathTex("y", font_size=30, color=C_TEXTO).next_to(ax.y_axis.get_end(), UP, buff=0.08))
        ts = np.linspace(-PI / 2, PI / 2, 181)
        banda = Polygon(*[ax.c2p(t, np.cos(t)) for t in ts] + [ax.c2p(t, 1.0) for t in ts[::-1]])
        banda.set_fill(C_BANDA, 0.9).set_stroke(width=0).set_z_index(0)
        cos_c = ax.plot(np.cos, x_range=[-PI / 2, PI / 2, 0.01], color=C_T, stroke_width=4).set_z_index(3)
        uno_c = ax.plot(lambda t: 1.0, x_range=[-PI / 2, PI / 2, 0.05], color=C_P, stroke_width=4).set_z_index(3)
        d = 0.1 / ax.x_axis.get_unit_size()
        sinc_c = VGroup(ax.plot(sinc, x_range=[-PI / 2, -d, 0.01]), ax.plot(sinc, x_range=[d, PI / 2, 0.01])) \
            .set_stroke(C_S, 4).set_z_index(4)
        hueco = Circle(radius=0.1).move_to(ax.c2p(0, 1)).set_stroke(C_S, 3).set_fill(opacity=0).set_z_index(8)
        lleno = Dot(ax.c2p(0, 1), radius=0.045, color=WHITE).set_z_index(8)
        l_uno = MathTex("1", font_size=32, color=C_P).next_to(ax.c2p(-PI / 2, 1), LEFT, buff=0.15)
        l_sinc = MathTex(r"\sen x / x", font_size=32, color=C_S).next_to(ax.c2p(-PI / 2, sinc(PI / 2)), LEFT, buff=0.15)
        l_cos = MathTex(r"\cos x", font_size=32, color=C_T).next_to(ax.c2p(-PI / 2, 0.12), LEFT, buff=0.15)
        self.graf = VGroup(ax, marcas, nums, rot, banda, cos_c, uno_c, sinc_c, hueco, lleno, l_uno, l_sinc, l_cos)

        self.play(Create(ax), FadeIn(marcas), FadeIn(nums), FadeIn(rot), run_time=1.2)
        self.play(Create(uno_c), Create(cos_c), FadeIn(l_uno), FadeIn(l_cos), run_time=1.5)
        self.play(FadeIn(banda), run_time=0.8)
        self.cambiar_leyenda(leyenda("la banda entre", r"$\cos x", "y", "$1"))
        self.play(Create(sinc_c), FadeIn(l_sinc), run_time=1.5)
        self.play(Create(hueco), FadeIn(lleno))
        self.cambiar_leyenda(leyenda(r"$\sen x / x", "va adentro de la banda; en", "$x = 0", "no está definida"))
        self.wait(2)

        self.puntos = VGroup(self.punto_curva(np.cos, C_T), self.punto_curva(lambda t: 1.0, C_P),
                             self.punto_curva(sinc, C_S))
        self.guia1 = self.guia()
        self.play(FadeIn(self.puntos), FadeIn(self.guia1))
        self.cambiar_leyenda(leyenda("el mismo", "$x", "de las figuras, en cada curva"),
                             self.x.animate.set_value(X0), run_time=3)
        self.wait(1)
        self.cambiar_leyenda(leyenda("si", r"$\cos x \to 1", "y", r"$1 \to 1{,}", "lo que está en el medio también"))
        self.play(self.x.animate.set_value(X_FIN), run_time=8, rate_func=smooth)
        self.wait(1)
        self.cambiar_leyenda(leyenda(r"$\cos x", "y", "$1", "valen", "$1", "en", "$x = 0{;}", r"$\sen x / x", "no está definida ahí,",
                                     "pero su límite es", "$1"))
        self.play(Indicate(hueco, color=WHITE, scale_factor=1.6), run_time=1.2)
        self.wait(2.5)

    # -- 8. x negativo ------------------------------------------------------
    def acto_negativo(self):
        fig = self.fig
        self.seccion("$x", "negativo")
        self.sacar(*self.estiradas, self.caja1, self.caja2, self.cap1, self.cap2, self.fac, self.bloque, self.caption,
                   run_time=1.0)
        self.caption = None
        self.play(*fig.mover(O_PAR, R_PAR), self.x.animate.set_value(X0), run_time=2.5)
        O, R = fig.O(), fig.R()
        abajo = VGroup(Line(fig.p(0, V_ABAJO), fig.p(0, -1.15)).set_stroke(C_EJE, 2).set_z_index(0),
                       DashedLine(fig.p(1, V_ABAJO), fig.p(1, -1.42), dash_length=0.08).set_stroke(C_EJE, 2).set_z_index(0),
                       Arc(radius=R, start_angle=-PI / 2, angle=PI / 2, arc_center=O).set_stroke(C_CIRC, 2.5).set_z_index(1))
        self.play(Create(abajo), run_time=1.2)

        # el reflejo respecto del eje horizontal: (u, v) ↦ (u, ρ·v), ρ de 1 a −1
        rho = ValueTracker(1.0)

        def m():
            return mapa_vertical(fig.O(), fig.O(), rho.get_value())

        reflejo = [always_redraw(lambda k=k: fig.region(k).apply_function(m())).set_z_index(z)
                   for k, z in (("grande", 2), ("sector", 3), ("chico", 3.5))]
        reflejo += [always_redraw(lambda k=k: fig.region(k, relleno=False).apply_function(m())).set_z_index(z)
                    for k, z in (("grande", 5), ("sector", 5.1), ("chico", 5.2))]
        reflejo += [always_redraw(lambda n=n: Dot(m()(fig.pt(n)), radius=0.05, color=WHITE)).set_z_index(8) for n in "PT"]
        self.add(*reflejo)
        self.cambiar_leyenda(leyenda("para el ángulo", "$-x{,}", "la figura se refleja respecto del eje horizontal"))
        self.play(rho.animate.set_value(-1.0), run_time=2.5, rate_func=smooth)
        P, T = fig.pt("P"), fig.pt("T")
        espejo = mapa_vertical(O, O, -1.0)
        dP = 0.38 * direccion(X0 + PI / 4)
        rotulos = VGroup(self.rotulo_fijo("P", P, (dP[0], dP[1]), R), self.rotulo_fijo("T", T, (0.27, 0.04), R),
                         self.rotulo_fijo("P'", espejo(P), (dP[0], -dP[1]), R),
                         self.rotulo_fijo("T'", espejo(T), (0.3, -0.04), R),
                         self.angulo(O, R, X0), self.angulo(O, R, X0, signo=-1, nombre="-x"))
        self.play(FadeIn(rotulos))
        self.wait(1)

        # en el gráfico: el punto de −x, a la misma altura
        ax = self.ax
        espejos = VGroup(self.punto_curva(np.cos, C_T, -1), self.punto_curva(lambda t: 1.0, C_P, -1),
                         self.punto_curva(sinc, C_S, -1))
        guia2 = self.guia(-1)
        horiz = always_redraw(lambda: DashedLine(ax.c2p(-self.xv(), sinc(self.xv())), ax.c2p(self.xv(), sinc(self.xv())),
                                                 dash_length=0.06).set_stroke(C_S, 2, opacity=0.85).set_z_index(5))
        p1 = MathTex(r"\sen(-x) = -\sen x", font_size=40).move_to([O_GRAF[0], -1.35, 0])
        p2 = MathTex(r"\frac{\sen(-x)}{-x}", "=", r"\frac{-\sen x}{-x}", "=", r"\frac{\sen x}{x}", font_size=40)
        p2.move_to([O_GRAF[0], -2.45, 0])
        meter(p1, p2, x=O_GRAF[0], ancho=6.2)
        self.play(Write(p1))
        self.cambiar_leyenda(leyenda("la altura cambia de signo y el ángulo también; el área no cambia"))
        self.wait(1.5)
        self.play(Write(p2), run_time=1.6)
        self.play(FadeIn(espejos), FadeIn(guia2), Create(horiz))
        self.cambiar_leyenda(leyenda("el cociente es el mismo: la función es par, su gráfico es simétrico"))
        self.wait(3)

        self.play(FadeOut(rotulos))
        self.cambiar_leyenda(leyenda("los dos límites laterales son iguales"))
        self.play(self.x.animate.set_value(X_FIN), run_time=7, rate_func=smooth)
        p3 = MathTex(r"\lim_{x\to 0^-}\frac{\sen x}{x}", "=", r"\lim_{x\to 0^+}\frac{\sen x}{x}", "=", "1", font_size=40)
        p3.move_to([O_GRAF[0], -1.95, 0])
        meter(p3, x=O_GRAF[0], ancho=6.2)
        self.play(FadeOut(p1), FadeOut(p2))
        self.play(Write(p3), run_time=1.6)
        self.wait(3)

    # -- 9. cierre ----------------------------------------------------------
    def acto_cierre(self):
        fig = self.fig
        quedan = [self.ejes, self.circ, self.tang, *self.rel.values(), *self.bor.values(), self.rOP, self.rPT, self.rAT,
                  self.dO, self.dA, self.dP, self.dT, self.lO, self.lA]
        self.limpiar(*quedan)
        self.play(*fig.mover(O_FIN, R_FIN), self.x.animate.set_value(X0), run_time=2.5)
        lim = MathTex(r"\lim_{x\to 0}\frac{\sen x}{x} = 1", font_size=84).move_to([2.6, 1.0, 0])
        k = SurroundingRectangle(lim, color=WHITE, buff=0.3, corner_radius=0.15, stroke_width=3)
        ar = MathTex(r"\cP{\frac{\sen x}{2}} \le \cS{\frac{x}{2}} \le \cT{\frac{\tg x}{2}}", font_size=40)
        des = MathTex(r"\cT{\cos x} \le \cS{\frac{\sen x}{x}} \le \cP{1}", font_size=40)
        VGroup(ar, des).arrange(RIGHT, buff=1.2).move_to([2.6, -1.1, 0])
        flecha = MathTex(r"\Longrightarrow", font_size=40, color=C_TEXTO).move_to(VGroup(ar, des).get_center())
        flecha.set_x((ar.get_right()[0] + des.get_left()[0]) / 2)
        nota = renglon("usamos que el área del sector es", "$x/2{,}", "o sea que", "$x", "se mide en radianes, y que",
                       r"$\cos x \to 1", font_size=22).move_to([0, -3.45, 0])
        self.play(Write(lim), run_time=1.8)
        self.play(Create(k), run_time=1.0)
        self.play(FadeIn(ar), FadeIn(flecha), FadeIn(des), run_time=1.2)
        self.play(FadeIn(nota), run_time=1.0)
        self.wait(5)
