"""
Completar cuadrados, literalmente.

x² + bx es un cuadrado más un rectángulo. Se corta el rectángulo en dos tiras de
ancho b/2, se pegan a dos lados del cuadrado y queda una "L" (gnomon). Falta la
esquinita (b/2)²: sumándola aparece el cuadrado de lado x + b/2, y leyendo las áreas
sale la fórmula resolvente.

    manim -pql completar_cuadrados.py CompletarCuadrados

Para la versión vertical (9:16, redes) poner VERTICAL = True.
"""
from manim import *
import numpy as np

VERTICAL = False            # True: frame 9:16 en lugar de 16:9

if VERTICAL:
    config.frame_width = 8.0
    config.frame_height = 8.0 * 16 / 9
    if config.pixel_width > config.pixel_height:
        config.pixel_width, config.pixel_height = config.pixel_height, config.pixel_width

config.background_color = "#14161c"

# ---------------------------------------------------------------------------
# Proporciones concretas (las etiquetas son siempre simbólicas)
# ---------------------------------------------------------------------------
X = 3.0                     # lado del cuadrado x²
B = 2.0                     # ancho del rectángulo bx
H = B / 2                   # ancho de cada tira
GAP = 0.8                   # separación inicial entre cuadrado y rectángulo
SEP = 0.25                  # cuánto se aleja cada tira al cortar
U = 1.0 if VERTICAL else 1.2   # tamaño en pantalla de una unidad de la figura

# Paleta: el mismo color en la figura y en su término de cada MathTex
C_SQ = BLUE                 # x²
C_RECT = YELLOW             # bx y las tiras (b/2)x
C_CORNER = GREEN            # (b/2)²
C_SUB = ORANGE              # sustitución b → b/a, c → c/a
C_CAPTION = GRAY_A
C_NOTE = GRAY_B

# ---------------------------------------------------------------------------
# Layout: P0 es la esquina inferior izquierda del cuadrado x² en los actos 2–5.
# Desde el acto 6 la figura se achica a FIG_CENTER y las ecuaciones usan la zona EQ_*.
# ---------------------------------------------------------------------------
if VERTICAL:
    P0 = np.array([-2.8, -1.0, 0])
    TOP_EQ = UP * 4.4
    FIG_CENTER = DOWN * 4.0
    FIG_SCALE = 0.9
    EQ_X, EQ_TOP, EQ_W = 0.0, 6.2, 7.4
    CAPTION_POS = DOWN * 1.3
    FINAL_FORMULA_POS = UP * 2.5
    FINAL_FIG_POS = DOWN * 3.2
else:
    P0 = np.array([-3.6, -1.0, 0])
    TOP_EQ = UP * 3.2
    FIG_CENTER = np.array([-3.5, -0.3, 0])
    FIG_SCALE = 0.75
    EQ_X, EQ_TOP, EQ_W = 2.9, 3.3, 7.4
    CAPTION_POS = np.array([2.9, -3.45, 0])
    FINAL_FORMULA_POS = RIGHT * 2.2
    FINAL_FIG_POS = LEFT * 4.6


def u(x, y):
    """Punto en coordenadas de la figura (en unidades de x y b) → escena."""
    return P0 + U * np.array([x, y, 0])


def pieza(w, h, color):
    return Rectangle(width=w * U, height=h * U, stroke_color=color, stroke_width=3,
                     fill_color=color, fill_opacity=0.45)


def llave(mob, direction, tex, font_size=34):
    br = Brace(mob, direction, buff=0.08)
    lab = MathTex(tex, font_size=font_size).next_to(br, direction, buff=0.1)
    return VGroup(br, lab)


def ajustar(mob, ancho=EQ_W):
    if mob.width > ancho:
        mob.scale_to_fit_width(ancho)
    return mob


def leyenda(*partes, font_size=28, color=C_CAPTION):
    """Mezcla Text (castellano) y MathTex: las partes que empiezan con '$' son TeX."""
    mobs = [MathTex(p[1:], font_size=font_size * 1.2, color=color) if p.startswith("$")
            else Text(p, font_size=font_size, color=color) for p in partes]
    return ajustar(VGroup(*mobs).arrange(RIGHT, buff=0.14)).move_to(CAPTION_POS)


def colorear(mt, colores):
    for i, c in colores.items():
        mt[i].set_color(c)
    return mt


class CompletarCuadrados(Scene):
    def construct(self):
        self.acto_titulo()
        self.acto_piezas()
        self.acto_corte()
        self.acto_pegar()
        self.acto_esquina()
        self.acto_completar()
        self.acto_monico()
        self.acto_general()

    # -- utilidades ---------------------------------------------------------
    def cambiar_leyenda(self, nueva):
        anim = [FadeIn(nueva, shift=UP * 0.15)]
        if getattr(self, "caption", None) is not None:
            anim.append(FadeOut(self.caption, shift=UP * 0.15))
        self.play(*anim, run_time=0.7)
        self.caption = nueva

    def debajo(self, prev, mob, buff=0.38):
        ajustar(mob).next_to(prev, DOWN, buff=buff).set_x(EQ_X)
        return mob

    # -- 1. título ----------------------------------------------------------
    def acto_titulo(self):
        titulo = Text("Completar cuadrados, literalmente", font_size=48)
        ajustar(titulo, config.frame_width - 1)
        self.play(FadeIn(titulo, shift=UP * 0.3), run_time=1.2)
        self.wait(1.3)
        self.play(FadeOut(titulo), run_time=0.8)

    # -- 2. las piezas ------------------------------------------------------
    def acto_piezas(self):
        self.sq = pieza(X, X, C_SQ).move_to(u(X / 2, X / 2))
        self.sq_lab = MathTex("x^2").move_to(self.sq)
        self.sq_llaves = VGroup(llave(self.sq, LEFT, "x"), llave(self.sq, DOWN, "x"))

        self.rect = pieza(B, X, C_RECT).move_to(u(X + GAP + B / 2, X / 2))
        self.rect_lab = MathTex("bx").move_to(self.rect)
        self.rect_llave = llave(self.rect, DOWN, "b")

        self.top = colorear(MathTex("x^2", "+", "bx").move_to(TOP_EQ), {0: C_SQ, 2: C_RECT})

        self.play(DrawBorderThenFill(self.sq), run_time=1.2)
        self.play(FadeIn(self.sq_llaves), Write(self.sq_lab))
        self.play(DrawBorderThenFill(self.rect), run_time=1.2)
        self.play(FadeIn(self.rect_llave), Write(self.rect_lab))
        self.play(Write(self.top))
        self.wait(1.5)

    # -- 3. cortar el rectángulo --------------------------------------------
    def acto_corte(self):
        r = self.rect
        corte = DashedLine(r.get_top(), r.get_bottom(), dash_length=0.12, stroke_width=3)
        self.play(Create(corte))
        self.wait(0.4)

        self.tira_a = pieza(H, X, C_RECT).move_to(r.get_center() + LEFT * H * U / 2)
        self.tira_b = pieza(H, X, C_RECT).move_to(r.get_center() + RIGHT * H * U / 2)
        self.remove(r)
        self.add(self.tira_a, self.tira_b, self.rect_lab)
        self.play(self.tira_a.animate.shift(LEFT * SEP * U),
                  self.tira_b.animate.shift(RIGHT * SEP * U),
                  FadeOut(corte), FadeOut(self.rect_llave))

        self.lab_a = MathTex(r"\tfrac{b}{2}x", font_size=34).move_to(self.tira_a)
        self.lab_b = MathTex(r"\tfrac{b}{2}x", font_size=34).move_to(self.tira_b)
        self.tiras_llaves = VGroup(llave(self.tira_a, DOWN, r"\tfrac{b}{2}"),
                                   llave(self.tira_b, DOWN, r"\tfrac{b}{2}"))
        self.play(ReplacementTransform(self.rect_lab, VGroup(self.lab_a, self.lab_b)),
                  FadeIn(self.tiras_llaves))
        self.wait(1.5)

    # -- 4. pegar las tiras -------------------------------------------------
    def acto_pegar(self):
        self.play(FadeOut(self.sq_llaves), FadeOut(self.tiras_llaves), run_time=0.7)

        # tira A: al lado derecho del cuadrado
        destino_a = u(X + H / 2, X / 2)
        self.play(self.tira_a.animate.move_to(destino_a),
                  self.lab_a.animate.move_to(destino_a), run_time=1.2)
        self.wait(0.3)

        # tira B: gira 90° orbitando alrededor del cuadrado (movimiento rígido gracias a
        # path_arc = ángulo de giro) y queda justo debajo; después sube hasta encajar.
        abajo = u(X / 2, -H / 2 - 0.7)
        self.play(self.tira_b.animate(path_arc=-PI / 2).rotate(-PI / 2).move_to(abajo),
                  self.lab_b.animate(path_arc=-PI / 2).move_to(abajo), run_time=2)
        destino_b = u(X / 2, -H / 2)
        self.play(self.tira_b.animate.move_to(destino_b),
                  self.lab_b.animate.move_to(destino_b), run_time=0.9)
        self.wait(1.2)

    # -- 5. la esquina que falta --------------------------------------------
    def acto_esquina(self):
        self.hueco = Square(side_length=H * U).move_to(u(X + H / 2, -H / 2))
        self.contorno = DashedVMobject(self.hueco.copy().set_stroke(C_CORNER, 4), num_dashes=16)
        self.play(Create(self.contorno))
        self.play(Flash(self.hueco.get_center(), color=C_CORNER, flash_radius=0.75,
                        line_length=0.3), Indicate(self.contorno, color=C_CORNER, scale_factor=1.25))

        self.esq_lab = MathTex(r"\left(\tfrac{b}{2}\right)^2", font_size=30)
        ajustar(self.esq_lab, 0.82 * H * U).move_to(self.hueco)
        self.esq_llaves = VGroup(llave(self.hueco, RIGHT, r"\tfrac{b}{2}"),
                                 llave(self.hueco, DOWN, r"\tfrac{b}{2}"))
        self.msg = Text("Falta esta esquinita", font_size=30, color=C_CORNER)
        if VERTICAL:
            self.msg.next_to(self.esq_llaves, DOWN, buff=0.35).set_x(0)
        else:
            self.msg.next_to(self.esq_llaves[0], RIGHT, buff=0.35)
        self.play(FadeIn(self.esq_llaves), Write(self.esq_lab), FadeIn(self.msg, shift=LEFT * 0.2))
        self.play(Indicate(self.contorno, color=C_CORNER, scale_factor=1.25))
        self.wait(1.5)

    # -- 6. completar el cuadrado -------------------------------------------
    def acto_completar(self):
        self.esquina = pieza(H, H, C_CORNER).move_to(self.hueco)
        self.play(FadeIn(self.esquina), FadeOut(self.contorno), FadeOut(self.msg))
        self.add(self.esq_lab)          # la etiqueta queda encima del relleno
        self.play(FadeOut(self.esq_llaves), run_time=0.6)

        self.grande = Square(side_length=(X + H) * U, stroke_color=WHITE, stroke_width=5)
        self.grande.move_to(u((X + H) / 2, (X - H) / 2))
        self.play(Create(self.grande))

        self.fig = VGroup(self.sq, self.tira_a, self.tira_b, self.esquina, self.grande,
                          self.sq_lab, self.lab_a, self.lab_b, self.esq_lab)
        self.play(self.fig.animate.scale(FIG_SCALE).move_to(FIG_CENTER), run_time=1.3)

        self.grandes_llaves = VGroup(llave(self.grande, LEFT, r"x+\tfrac{b}{2}", font_size=30),
                                     llave(self.grande, DOWN, r"x+\tfrac{b}{2}", font_size=30))
        self.fig.add(self.grandes_llaves)
        self.play(FadeIn(self.grandes_llaves))

        self.identidad = MathTex("x^2", "+", "bx", "+", r"\left(\tfrac{b}{2}\right)^2", "=",
                                 r"\left(x+\tfrac{b}{2}\right)^2")
        colorear(self.identidad, {0: C_SQ, 2: C_RECT, 4: C_CORNER})
        ajustar(self.identidad).move_to([EQ_X, EQ_TOP, 0])
        self.play(TransformMatchingTex(self.top, self.identidad), run_time=1.5)
        self.play(Indicate(self.identidad[4], color=C_CORNER), Indicate(self.esquina, color=C_CORNER))
        self.play(Indicate(self.identidad[6]), Indicate(self.grande))
        self.wait(1.5)

    # -- 7. caso mónico x² + bx + c = 0 ---------------------------------------
    def acto_monico(self):
        l1 = self.debajo(self.identidad, MathTex("x^2", "+", "bx", "+", "c", "=", "0"), buff=0.55)
        colorear(l1, {0: C_SQ, 2: C_RECT})
        self.cambiar_leyenda(leyenda("Ecuación mónica:", "$x^2+bx+c=0"))
        self.play(Write(l1))
        self.wait(0.8)

        # el área de la L vale −c
        l2 = colorear(self.debajo(l1, MathTex("x^2", "+", "bx", "=", "-c")), {0: C_SQ, 2: C_RECT})
        ese = Polygon(self.sq.get_corner(UL), self.tira_a.get_corner(UR), self.tira_a.get_corner(DR),
                      self.esquina.get_corner(UL), self.tira_b.get_corner(DR), self.tira_b.get_corner(DL),
                      stroke_color=WHITE, stroke_width=6)
        self.cambiar_leyenda(leyenda("el área de la L vale", "$-c"))
        self.play(TransformMatchingTex(l1.copy(), l2), Create(ese), run_time=1.3)
        self.play(Indicate(ese, color=WHITE, scale_factor=1.03))
        self.play(FadeOut(ese), run_time=0.5)
        self.wait(1)

        # sumar la esquina a ambos lados
        l3 = MathTex(r"\left(x+\tfrac{b}{2}\right)^2", "=", r"\left(\tfrac{b}{2}\right)^2", "-c")
        colorear(self.debajo(l2, l3), {2: C_CORNER})
        self.cambiar_leyenda(leyenda("sumamos la esquina", r"$\left(\tfrac{b}{2}\right)^2",
                                     "a ambos lados"))
        self.play(TransformMatchingTex(l2.copy(), l3), Indicate(self.esquina, color=C_CORNER),
                  run_time=1.3)
        self.wait(1.5)

        # el lado es la raíz del área
        l4 = MathTex(r"x+\tfrac{b}{2}", "=", r"\pm", r"\sqrt{\left(\tfrac{b}{2}\right)^2-c}")
        self.debajo(l3, l4)
        self.cambiar_leyenda(leyenda("el lado del cuadrado es la raíz de su área"))
        self.play(TransformMatchingTex(l3.copy(), l4), Indicate(self.grandes_llaves[0]),
                  Indicate(self.grandes_llaves[1]), run_time=1.3)
        self.wait(1.5)

        # despejar x
        self.formula = MathTex("x", "=", r"-\frac{", "b", "}{2}", r"\pm", r"\sqrt{\frac{", "b",
                               "^2}{4}", "-", "c", "}")
        self.debajo(l4, self.formula)
        self.cambiar_leyenda(leyenda("despejamos", "$x"))
        self.play(TransformMatchingShapes(l4.copy(), self.formula), run_time=1.3)
        self.wait(0.8)

        nota = VGroup(
            leyenda("el", r"$\pm", "aparece porque el dibujo solo muestra la raíz positiva;",
                    font_size=25, color=C_NOTE),
            Text("el álgebra recupera la otra", font_size=25, color=C_NOTE),
        ).arrange(DOWN, buff=0.12).move_to(CAPTION_POS)
        self.cambiar_leyenda(nota)
        self.wait(3)
        self.lineas_monico = VGroup(self.identidad, l1, l2, l3, l4)

    # -- 8. caso general ax² + bx + c = 0 -------------------------------------
    def acto_general(self):
        self.play(FadeOut(self.lineas_monico), FadeOut(self.caption))
        self.caption = None

        g1 = MathTex("a", "x^2", "+", "b", "x", "+", "c", "=", "0").move_to([EQ_X, EQ_TOP, 0])
        self.play(self.formula.animate.move_to([EQ_X, EQ_TOP - 1.6, 0]), Write(g1))
        self.cambiar_leyenda(leyenda("dividimos todo por", "$a"))
        g2 = MathTex("x^2", "+", r"\frac{b}{a}", "x", "+", r"\frac{c}{a}", "=", "0")
        colorear(g2, {2: C_SUB, 5: C_SUB}).move_to(g1)
        self.play(TransformMatchingShapes(g1, g2), run_time=1.3)
        self.wait(1)

        # b → b/a y c → c/a dentro de la fórmula del caso mónico
        f = self.formula
        self.cambiar_leyenda(leyenda("en la fórmula anterior:", r"$b\to\tfrac{b}{a}", ",",
                                     r"$c\to\tfrac{c}{a}"))
        self.play(*[Indicate(f[i], color=C_SUB, scale_factor=1.5) for i in (3, 7, 10)])
        s1 = MathTex("x", "=", r"-\frac{", "b/a", "}{2}", r"\pm", r"\sqrt{\frac{", "(b/a)",
                     "^2}{4}", "-", r"\frac{c}{a}", "}")
        colorear(s1, {3: C_SUB, 7: C_SUB, 10: C_SUB}).move_to(f)
        self.play(*[ReplacementTransform(f[i], s1[i]) for i in range(len(f))], run_time=1.5)
        self.wait(1.5)

        # simplificar en dos pasos
        s2 = MathTex("x", "=", r"-\frac{b}{2a}", r"\pm", r"\sqrt{\frac{b^2-4ac}{4a^2}}").move_to(s1)
        self.cambiar_leyenda(leyenda("ordenamos las fracciones"))
        self.play(TransformMatchingShapes(s1, s2), run_time=1.5)
        self.wait(1.5)

        s3 = MathTex("x", "=", r"\frac{-b\pm\sqrt{b^2-4ac}}{2a}").move_to(s2)
        self.cambiar_leyenda(leyenda("sacamos", r"$\sqrt{4a^2}", "de la raíz (el", r"$\pm",
                                     "cubre el signo)"))
        self.play(TransformMatchingShapes(s2, s3), run_time=1.5)
        self.wait(1.5)

        # cierre: fórmula grande encuadrada y la figura achicada al costado
        self.play(FadeOut(g2), FadeOut(self.caption),
                  self.fig.animate.scale(0.85).move_to(FINAL_FIG_POS),
                  s3.animate.scale(1.35).move_to(FINAL_FORMULA_POS), run_time=1.5)
        caja = SurroundingRectangle(s3, color=C_RECT, buff=0.3, corner_radius=0.1)
        self.play(Create(caja))
        self.wait(3)
