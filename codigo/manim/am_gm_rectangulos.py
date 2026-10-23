"""
AM–GM con rectángulos.

Cuatro rectángulos de a×b en "molinete" dentro de un cuadrado de lado a+b dejan un
hueco cuadrado de lado |a−b|. Contando áreas, (a+b)² = 4ab + (a−b)² ≥ 4ab, y de ahí
(a+b)/2 ≥ √(ab). El hueco desaparece exactamente cuando a = b.

    manim -pql am_gm_rectangulos.py AMGMRectangulos
"""
from manim import *
import numpy as np

config.background_color = "#14161c"

# ---------------------------------------------------------------------------
# Proporciones concretas (las etiquetas son siempre simbólicas)
# ---------------------------------------------------------------------------
A = 3.0                     # lado largo del rectángulo
B = 1.5                     # lado corto
S = A + B                   # lado del cuadrado exterior (fijo en el acto 8)
U = 1.1                     # tamaño en pantalla de una unidad (actos 2–5)

# Paleta: el mismo color en la figura y en su término de cada MathTex
C_RECTS = ["#7cc4ff", "#4fa3f7", "#2f80ed", "#1c5fc4"]   # R1..R4, familia de azules
C_RECT = C_RECTS[1]         # ab, 4ab, √(ab)
C_HOLE = "#ff9f43"          # el hueco y (a−b)²
C_BIG = WHITE               # el cuadrado exterior y (a+b)²
C_EQUAL = YELLOW            # momento de igualdad y recuadro final
C_CAPTION = GRAY_A

# ---------------------------------------------------------------------------
# Layout: P0 es la esquina inferior izquierda del cuadrado exterior en los actos 2–5.
# Desde el acto 6 la figura se achica a FIG_CENTER y las ecuaciones usan la zona EQ_*.
# ---------------------------------------------------------------------------
P0 = np.array([-S * U / 2, -S * U / 2 - 0.25, 0])
TOP_CAPTION = UP * 3.4
FIG_SCALE = 0.8
FIG_CENTER = np.array([-3.9, -0.2, 0])
EQ_X, EQ_TOP, EQ_W = 2.9, 2.9, 7.0
CAPTION_POS = np.array([2.9, -3.4, 0])
FIG8_CENTER = np.array([-3.4, 0.1, 0])     # centro del cuadrado exterior en el acto 8
FIG8_SIDE = 4.4
FINAL_FIG_POS = np.array([-4.2, 0.45, 0])
FINAL_FORMULA_POS = np.array([2.4, 0.6, 0])


def punto(P, Uf, x, y):
    """Punto en coordenadas de la figura (unidades de a y b) → escena."""
    return P + Uf * np.array([x, y, 0])


def u(x, y):
    return punto(P0, U, x, y)


def molinete(a, b):
    """Los cuatro rectángulos como (x0, x1, y0, y1); valen para a > b y para a < b."""
    s = a + b
    return [(0, a, 0, b), (a, s, 0, a), (b, s, a, s), (0, b, b, s)]


def caja(P, Uf, x0, x1, y0, y1, color):
    return Rectangle(width=(x1 - x0) * Uf, height=(y1 - y0) * Uf, stroke_color=color,
                     stroke_width=3, fill_color=color, fill_opacity=0.5
                     ).move_to(punto(P, Uf, (x0 + x1) / 2, (y0 + y1) / 2))


def cuadrado_hueco(P, Uf, a, b):
    m, M = min(a, b), max(a, b)
    return Square(side_length=max(M - m, 1e-3) * Uf).move_to(punto(P, Uf, (m + M) / 2, (m + M) / 2))


def contorno_hueco(P, Uf, a, b):
    lado = abs(a - b)
    sq = cuadrado_hueco(P, Uf, a, b).set_stroke(C_HOLE, 4)
    if lado < 0.02:
        return sq.set_stroke(opacity=0)
    return DashedVMobject(sq, num_dashes=max(4, int(round(10 * lado))))


def llave(mob, direction, tex, font_size=34, color=WHITE):
    br = Brace(mob, direction, buff=0.08, color=color)
    lab = MathTex(tex, font_size=font_size, color=color).next_to(br, direction, buff=0.1)
    return VGroup(br, lab)


def llaves_hueco(P, Uf, a, b, font_size=32):
    """Braces del hueco por dentro (lados inferior e izquierdo). Si a < b dicen |a−b|."""
    lado = abs(a - b)
    if lado < 0.12:
        return VGroup()
    sq = cuadrado_hueco(P, Uf, a, b)
    tex = "a-b" if a >= b else "|a-b|"
    g = VGroup(llave(sq, DOWN, tex, font_size, C_HOLE), llave(sq, LEFT, tex, font_size, C_HOLE))
    return g.set_opacity(min(1, (lado - 0.12) / 0.3))


def ajustar(mob, ancho=EQ_W):
    if mob.width > ancho:
        mob.scale_to_fit_width(ancho)
    return mob


DESCENDENTES = "gjpqy,;()"


def linea_base(mob):
    """Altura de la línea de base: mediana de los bordes inferiores de los glifos.
    En un Text se ignoran las letras con descendente (un "y" suelto se corrige aparte)."""
    if isinstance(mob, Text):
        chars = [c for c in mob.text if not c.isspace()]
        glifos = [g for c, g in zip(chars, mob.submobjects) if c not in DESCENDENTES]
        if glifos:
            return np.median([g.get_bottom()[1] for g in glifos])
        ref = Text("xy", font_size=mob.font_size)
        return mob.get_bottom()[1] + ref[0].get_bottom()[1] - ref[1].get_bottom()[1]
    return np.median([m.get_bottom()[1] for m in mob.family_members_with_points()])


def renglon(*mobs, buff=0.14):
    """Pone los mobjects en fila con una línea de base común (Text y MathTex mezclados)."""
    g = VGroup(*mobs).arrange(RIGHT, buff=buff)
    y = linea_base(mobs[0])
    for m in mobs[1:]:
        m.shift(UP * (y - linea_base(m)))
    return g


def leyenda(*partes, font_size=30, color=C_CAPTION, pos=CAPTION_POS, ancho=EQ_W):
    """Mezcla Text (castellano) y MathTex: las partes que empiezan con '$' son TeX."""
    mobs = [MathTex(p[1:], font_size=font_size * 1.35, color=color) if p.startswith("$")
            else Text(p, font_size=font_size, color=color) for p in partes]
    return ajustar(renglon(*mobs), ancho).move_to(pos)


def arriba(*partes):
    return leyenda(*partes, pos=TOP_CAPTION, ancho=config.frame_width - 1)


def colorear(mt, colores):
    for i, c in colores.items():
        mt[i].set_color(c)
    return mt


def parpadeo(mob, opacidad, veces=2):
    """Sube y baja la opacidad del relleno `veces` veces; termina como empezó."""
    return mob.animate(rate_func=lambda t: there_and_back((veces * t) % 1)).set_fill(opacity=opacidad)


class AMGMRectangulos(Scene):
    def construct(self):
        self.caption = None
        self.acto_titulo()
        self.acto_rectangulo()
        self.acto_copias()
        self.acto_cuadrado()
        self.acto_hueco()
        self.acto_areas()
        self.acto_desigualdad()
        self.acto_igualdad()
        self.acto_cierre()

    # -- utilidades ---------------------------------------------------------
    def cambiar_leyenda(self, nueva):
        anim = [FadeIn(nueva, shift=UP * 0.15)]
        if self.caption is not None:
            anim.append(FadeOut(self.caption, shift=UP * 0.15))
        self.play(*anim, run_time=0.7)
        self.caption = nueva

    def quitar_leyenda(self):
        anim = FadeOut(self.caption) if self.caption is not None else Wait(0)
        self.caption = None
        return anim

    def debajo(self, prev, mob, buff=0.45):
        ajustar(mob).next_to(prev, DOWN, buff=buff).set_x(EQ_X)
        return mob

    # -- 1. título ----------------------------------------------------------
    def acto_titulo(self):
        titulo = Text("AM–GM con rectángulos", font_size=56)
        self.play(FadeIn(titulo, shift=UP * 0.3), run_time=1.2)
        self.wait(1.3)
        self.play(FadeOut(titulo), run_time=0.8)

    # -- 2. un rectángulo ---------------------------------------------------
    def acto_rectangulo(self):
        self.r1 = caja(P0, U, 0, A, 0, B, C_RECTS[0]).move_to(UP * 0.2)
        self.r1_llaves = VGroup(llave(self.r1, DOWN, "a"), llave(self.r1, LEFT, "b"))
        self.r1_lab = MathTex("ab").move_to(self.r1)

        self.cambiar_leyenda(arriba("un rectángulo de lados", "$a", "y", "$b"))
        self.play(DrawBorderThenFill(self.r1), run_time=1.2)
        self.play(FadeIn(self.r1_llaves), Write(self.r1_lab))
        self.wait(1.5)

    # -- 3. cuatro copias en molinete ---------------------------------------
    def acto_copias(self):
        self.centro = u(S / 2, S / 2)
        guia = DashedVMobject(Square(side_length=S * U).set_stroke(GRAY_C, 2).move_to(self.centro),
                              num_dashes=60)
        # la etiqueta va a la esquina "libre" de cada rectángulo, lejos del hueco
        self.play(FadeOut(self.r1_llaves), run_time=0.6)
        self.play(self.r1.animate.move_to(u(A / 2, B / 2)), self.r1_lab.animate.move_to(u(B / 2, B / 2)),
                  Create(guia), run_time=1.5)
        self.guia = guia

        self.cambiar_leyenda(arriba("tres copias, cada una girada", "$90^\\circ",
                                    "alrededor del centro"))
        self.rects, self.labs = [self.r1], [self.r1_lab]
        for k in range(1, 4):
            nuevo = self.rects[-1].copy()
            lab = self.labs[-1].copy()
            destino = rotate_vector(lab.get_center() - self.centro, PI / 2) + self.centro
            self.add(nuevo, lab)
            # path_arc = ángulo de giro → cada punto orbita el centro: movimiento rígido
            self.play(nuevo.animate(path_arc=PI / 2).rotate(PI / 2, about_point=self.centro)
                      .set_fill(C_RECTS[k]).set_stroke(C_RECTS[k]),
                      lab.animate(path_arc=PI / 2).move_to(destino), run_time=1.4)
            self.rects.append(nuevo)
            self.labs.append(lab)
            self.wait(0.3)
        self.rects = VGroup(*self.rects)
        self.labs = VGroup(*self.labs)
        self.wait(0.6)

    # -- 4. el cuadrado grande ----------------------------------------------
    def acto_cuadrado(self):
        self.grande = Square(side_length=S * U).set_stroke(C_BIG, 5).move_to(self.centro)
        self.grande_relleno = self.grande.copy().set_stroke(width=0).set_fill(C_BIG, 0)
        self.cambiar_leyenda(arriba("forman un cuadrado de lado", "$a+b"))
        self.play(Create(self.grande), FadeOut(self.guia), run_time=1.3)

        self.grande_llaves = VGroup(llave(self.grande, LEFT, "a+b"), llave(self.grande, DOWN, "a+b"))
        self.area_grande = MathTex("(a+b)^2", color=C_BIG).next_to(self.grande, RIGHT, buff=0.6)
        self.play(FadeIn(self.grande_llaves))
        self.play(Write(self.area_grande), parpadeo(self.grande_relleno, 0.25), run_time=1.4)
        self.wait(1)

    # -- 5. el hueco --------------------------------------------------------
    def acto_hueco(self):
        self.hueco = cuadrado_hueco(P0, U, A, B)
        self.hueco_relleno = self.hueco.copy().set_stroke(width=0).set_fill(C_HOLE, 0)
        self.hueco_contorno = contorno_hueco(P0, U, A, B)
        self.add(self.grande_relleno, self.hueco_relleno)

        self.cambiar_leyenda(arriba("en el centro queda un hueco cuadrado"))
        self.play(Create(self.hueco_contorno))
        self.play(Indicate(self.hueco_contorno, color=C_HOLE, scale_factor=1.25))

        self.hueco_llaves = llaves_hueco(P0, U, A, B)
        self.hueco_lab = MathTex("(a-b)^2", font_size=34, color=C_HOLE).move_to(self.hueco)
        self.cambiar_leyenda(arriba("de lado", "$a-b", "y área", "$(a-b)^2"))
        self.play(FadeIn(self.hueco_llaves), Write(self.hueco_lab))
        self.wait(1.5)

    # -- 6. contar áreas ----------------------------------------------------
    def acto_areas(self):
        self.fig = VGroup(self.rects, self.labs, self.grande_relleno, self.hueco_relleno, self.grande,
                          self.grande_llaves, self.hueco_contorno, self.hueco_lab)
        self.play(FadeOut(self.hueco_llaves), self.quitar_leyenda(),
                  self.fig.animate.scale(FIG_SCALE).move_to(FIG_CENTER), run_time=1.3)

        self.eq = MathTex("(a+b)^2", "=", "4ab", "+", "(a-b)^2")
        colorear(self.eq, {2: C_RECT, 4: C_HOLE}).move_to([EQ_X, EQ_TOP, 0])
        self.cambiar_leyenda(leyenda("contamos el área de dos maneras"))
        self.play(ReplacementTransform(self.area_grande, self.eq[0]),
                  parpadeo(self.grande_relleno, 0.25), run_time=1.4)
        self.play(Write(self.eq[1]), run_time=0.6)
        self.play(TransformFromCopy(self.labs, self.eq[2]), parpadeo(self.rects, 0.95), run_time=1.4)
        self.play(Write(self.eq[3]), run_time=0.6)
        self.play(TransformFromCopy(self.hueco_lab, self.eq[4]), parpadeo(self.hueco_relleno, 0.7),
                  run_time=1.4)
        self.wait(1.5)

    # -- 7. la desigualdad --------------------------------------------------
    def acto_desigualdad(self):
        eq = self.eq
        self.cambiar_leyenda(leyenda("el hueco nunca tiene área negativa"))
        pos = colorear(MathTex("(a-b)^2", r"\ge", "0"), {0: C_HOLE})
        self.debajo(eq, pos)
        self.play(Write(pos), parpadeo(self.hueco_relleno, 0.7), run_time=1.4)
        self.wait(0.8)

        cruz = Cross(VGroup(eq[3], eq[4]), stroke_color=RED, stroke_width=5)
        self.play(Create(cruz))
        self.l2 = colorear(MathTex("(a+b)^2", r"\ge", "4ab"), {2: C_RECT}).move_to([EQ_X, EQ_TOP, 0])
        self.cambiar_leyenda(leyenda("sin el hueco, el igual pasa a ser", r"$\ge"))
        self.play(FadeOut(VGroup(eq[3], eq[4], cruz)), FadeOut(pos),
                  *[ReplacementTransform(eq[i], self.l2[i]) for i in range(3)], run_time=1.4)
        self.wait(1.2)

        l3 = colorear(MathTex(r"\frac{(a+b)^2}{4}", r"\ge", "ab"), {2: C_RECT})
        self.debajo(self.l2, l3)
        self.cambiar_leyenda(leyenda("dividimos por", "$4"))
        self.play(*[TransformFromCopy(self.l2[i], l3[i]) for i in range(3)], run_time=1.3)
        self.wait(0.8)

        self.l4 = colorear(MathTex(r"\left(\frac{a+b}{2}\right)^2", r"\ge", "ab"), {2: C_RECT}).move_to(l3)
        self.cambiar_leyenda(leyenda("como", "$4=2^2:", "el 2 entra al cuadrado"))
        self.play(TransformMatchingShapes(l3, self.l4), run_time=1.3)
        self.wait(0.8)

        self.l5 = colorear(MathTex(r"\frac{a+b}{2}", r"\ge", r"\sqrt{ab}", font_size=56), {2: C_RECT})
        self.debajo(self.l4, self.l5, buff=0.55)
        self.cambiar_leyenda(leyenda("con", r"$a,\,b>0", "sacamos raíz cuadrada"))
        self.play(TransformMatchingShapes(self.l4.copy(), self.l5), run_time=1.4)
        self.wait(0.6)

        am = Text("media aritmética", font_size=26, color=C_BIG)
        gm = Text("media geométrica", font_size=26, color=C_RECT)
        am.move_to(self.l5[0].get_bottom() + DOWN * 1.15 + LEFT * 0.9)
        gm.move_to(self.l5[2].get_bottom() + DOWN * 1.15 + RIGHT * 0.9)
        fa = Arrow(am.get_top(), self.l5[0].get_bottom(), buff=0.12, stroke_width=3, color=C_BIG,
                   max_tip_length_to_length_ratio=0.2)
        fg = Arrow(gm.get_top(), self.l5[2].get_bottom(), buff=0.12, stroke_width=3, color=C_RECT,
                   max_tip_length_to_length_ratio=0.2)
        self.medias = VGroup(am, gm, fa, fg)
        self.play(FadeIn(am, shift=UP * 0.2), GrowArrow(fa))
        self.play(FadeIn(gm, shift=UP * 0.2), GrowArrow(fg))
        self.wait(2)

    # -- 8. igualdad: el hueco desaparece -----------------------------------
    def acto_igualdad(self):
        # la figura vuelve a crecer; quedan solo las piezas que se van a mover
        g = self.grande
        abajo = self.grande_llaves[1]
        piezas = VGroup(self.rects, g, self.grande_llaves[0], self.hueco_contorno)
        k = FIG8_SIDE / g.width
        self.play(FadeOut(VGroup(self.labs, self.hueco_lab, abajo, self.l2, self.l4, self.medias)),
                  self.quitar_leyenda(), self.l5.animate.move_to([EQ_X, EQ_TOP - 0.1, 0]),
                  piezas.animate.scale(k, about_point=g.get_center()).shift(FIG8_CENTER - g.get_center()),
                  run_time=1.5)

        P, Uf = g.get_corner(DL), g.width / S
        self.ta, self.tb = ValueTracker(A), ValueTracker(B)
        a_, b_ = self.ta.get_value, self.tb.get_value

        rects = always_redraw(lambda: VGroup(*[caja(P, Uf, *c, C_RECTS[i])
                                               for i, c in enumerate(molinete(a_(), b_()))]))
        hueco = always_redraw(lambda: contorno_hueco(P, Uf, a_(), b_()))
        hueco_llaves = always_redraw(lambda: llaves_hueco(P, Uf, a_(), b_()))
        ab_llaves = always_redraw(lambda: VGroup(
            llave(Line(punto(P, Uf, 0, 0), punto(P, Uf, a_(), 0)), DOWN, "a"),
            llave(Line(punto(P, Uf, a_(), 0), punto(P, Uf, a_() + b_(), 0)), DOWN, "b")))
        # cambio invisible: las piezas estáticas pasan a ser las dinámicas
        self.remove(self.rects, self.hueco_contorno, self.grande_relleno, self.hueco_relleno)
        self.add(rects, g, hueco)
        self.play(FadeIn(hueco_llaves), FadeIn(ab_llaves))
        self.dinamicos = VGroup(rects, hueco, hueco_llaves, ab_llaves)
        self.piezas = VGroup(rects, g, self.grande_llaves[0], hueco, hueco_llaves, ab_llaves)

        # tablero de valores: los "=" alineados en EQ_X
        filas = [("a", a_, WHITE, 1.45), ("b", b_, WHITE, 0.75),
                 (r"\frac{a+b}{2}", lambda: (a_() + b_()) / 2, C_BIG, -0.35),
                 (r"\sqrt{ab}", lambda: np.sqrt(a_() * b_()), C_RECT, -1.35)]
        self.tablero = VGroup()
        for tex, f, color, y in filas:
            igual = MathTex("=").move_to([EQ_X, y, 0])
            lab = MathTex(tex, color=color).next_to(igual, LEFT, buff=0.2)
            num = DecimalNumber(f(), num_decimal_places=2, color=color).next_to(igual, RIGHT, buff=0.2)
            num.add_updater(lambda m, f=f: m.set_value(f()))
            self.tablero.add(VGroup(lab, igual, num))
        self.play(LaggedStart(*[FadeIn(f, shift=LEFT * 0.2) for f in self.tablero], lag_ratio=0.15))

        self.cambiar_leyenda(leyenda("movemos", "$a", "y", "$b", "con", "$a+b", "fijo"))
        self.wait(0.5)
        self.play(self.ta.animate.set_value(S / 2), self.tb.animate.set_value(S / 2), run_time=4)

        self.igualdad = renglon(Text("igualdad", font_size=34, color=C_EQUAL),
                                MathTex(r"\iff a=b", color=C_EQUAL), buff=0.2)
        self.igualdad.move_to([EQ_X, -2.4, 0])
        self.play(Flash(punto(P, Uf, S / 2, S / 2), color=C_EQUAL, flash_radius=0.9, line_length=0.35),
                  Flash(self.tablero[2][2].get_center() + RIGHT * 0.3, color=C_EQUAL, flash_radius=0.7),
                  Flash(self.tablero[3][2].get_center() + RIGHT * 0.3, color=C_EQUAL, flash_radius=0.7),
                  FadeIn(self.igualdad, scale=1.2), run_time=1.2)
        self.cambiar_leyenda(leyenda("sin hueco: las dos medias coinciden"))
        self.wait(1.5)

        self.cambiar_leyenda(leyenda("si", "$a<b,", "el lado del hueco es", "$b-a=|a-b|"))
        self.play(self.ta.animate.set_value(B), self.tb.animate.set_value(A), run_time=3)
        self.play(Flash(punto(P, Uf, S / 2, S / 2), color=C_HOLE, flash_radius=1.1, line_length=0.3))
        self.wait(2)

    # -- 9. cierre ----------------------------------------------------------
    def acto_cierre(self):
        for m in [*self.dinamicos, *[f[2] for f in self.tablero]]:
            m.clear_updaters()
        self.play(FadeOut(self.tablero), FadeOut(self.igualdad), self.quitar_leyenda(),
                  self.piezas.animate.scale(0.75).move_to(FINAL_FIG_POS),
                  self.l5.animate.scale(1.2).move_to(FINAL_FORMULA_POS), run_time=1.5)

        cond = renglon(Text("con igualdad sii", font_size=32), MathTex("a=b"), buff=0.2)
        cond.next_to(self.l5, DOWN, buff=0.45)
        caja_final = SurroundingRectangle(VGroup(self.l5, cond), color=C_EQUAL, buff=0.35,
                                         corner_radius=0.1)
        self.play(FadeIn(cond, shift=UP * 0.15))
        self.play(Create(caja_final))

        interp = Text("con perímetro fijo, el rectángulo de mayor área es el cuadrado",
                      font_size=30, color=C_CAPTION)
        ajustar(interp, config.frame_width - 1).move_to(DOWN * 3.1)
        self.play(FadeIn(interp, shift=UP * 0.2))
        self.wait(3.5)
