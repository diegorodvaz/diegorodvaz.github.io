"""
La Serie de Taylor como una Oruga Elástica.

El polinomio de Taylor centrado en x0 = 0 va sumando términos de grado 1, 2, 3, ...
y en cada paso se estira (con un pequeño "encogerse y estirarse" de oruga) hasta
abrazar a f(x) cerca del origen, mientras que lejos del centro se sigue escapando.

    manim -pql Taylor.py TaylorCaterpillarAnimation
"""
from math import factorial

from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Elección de la función: cambiar FUNCION por "cos", "exp" o "sincos" para probar otras.
#   deriv(k) = f^(k)(0), que es lo único que necesita el polinomio de Taylor en 0.
# ---------------------------------------------------------------------------
FUNCIONES = {
    "sin": dict(f=np.sin, deriv=lambda k: [0, 1, 0, -1][k % 4], tex=r"\sin x"),
    "cos": dict(f=np.cos, deriv=lambda k: [1, 0, -1, 0][k % 4], tex=r"\cos x"),
    "exp": dict(f=np.exp, deriv=lambda k: 1, tex=r"e^{x}"),
    "sincos": dict(f=lambda x: np.sin(x) + np.cos(x),
                   deriv=lambda k: [1, 1, -1, -1][k % 4], tex=r"\sin x + \cos x"),
}
FUNCION = "sin"

N_MAX = 15                  # grado máximo que alcanza la oruga
X_MAX = 7.5                 # ventana horizontal [-X_MAX, X_MAX]
Y_MAX = 2.5                 # ventana vertical [-Y_MAX, Y_MAX]: el polinomio se recorta acá
EPS = 0.1                   # tolerancia que define la "zona de buen ajuste" |f - P_n| < EPS
MUESTRAS = 1501             # resolución con la que se evalúan f y P_n

FONDO = "#0b0f1a"
COLOR_F = BLUE_C
COLOR_P = ORANGE
COLOR_ZONA = YELLOW_C

F = FUNCIONES[FUNCION]
COEF = np.array([F["deriv"](k) / factorial(k) for k in range(N_MAX + 1)], dtype=float)
XS = np.linspace(-X_MAX, X_MAX, MUESTRAS)
FXS = F["f"](XS)


def termino_tex(k):
    # monomio c_k x^k escrito como f^(k)(0)/k! x^k (sin signo; el signo lo pone polinomio_tex)
    d = abs(F["deriv"](k))
    num = "" if d == 1 else f"{d}"
    if k == 0:
        return f"{d}"
    if k == 1:
        return f"{num}x"
    return rf"\frac{{{num}x^{{{k}}}}}{{{k}!}}"


def polinomio_tex(n):
    # P_n(x) explícito; si hay muchos términos se muestran los 3 primeros, "⋯" y el último
    terminos = [(k, COEF[k]) for k in range(n + 1) if COEF[k] != 0]
    if len(terminos) > 4:
        terminos = terminos[:3] + [None] + terminos[-1:]
    partes = []
    for i, t in enumerate(terminos):
        if t is None:
            partes.append(r"+\cdots")
            continue
        k, c = t
        signo = "-" if c < 0 else ("+" if i > 0 else "")
        partes.append(signo + termino_tex(k))
    return rf"P_{{{n}}}(x) = " + " ".join(partes)


class TaylorCaterpillarAnimation(Scene):
    def construct(self):
        self.camera.background_color = FONDO

        # -------------------------------------------------------------------
        # 1. Título, ejes y función base f(x)
        # -------------------------------------------------------------------
        titulo = Text("La Serie de Taylor", font_size=36, weight=BOLD)
        titulo.to_edge(UP, buff=0.3)

        ejes = Axes(
            x_range=[-X_MAX, X_MAX, 1],
            y_range=[-Y_MAX, Y_MAX, 1],
            x_length=13,
            y_length=4.4,
            axis_config={"color": GREY_B, "stroke_width": 2, "include_ticks": True,
                         "tick_size": 0.05},
            tips=False,
        ).to_edge(DOWN, buff=0.35)
        rotulos_x = VGroup(*[
            MathTex(rf"{k}\pi" if abs(k) != 1 else ("-" if k < 0 else "") + r"\pi",
                    font_size=22, color=GREY_B).next_to(ejes.c2p(k * PI, 0), DOWN, buff=0.12)
            for k in (-2, -1, 1, 2)
        ])

        grafica_f = ejes.plot(F["f"], x_range=[-X_MAX, X_MAX, 0.02], color=COLOR_F, stroke_width=5)
        etiqueta_f = MathTex(rf"f(x) = {F['tex']}", font_size=32, color=COLOR_F)

        self.play(Write(titulo), run_time=1.5)
        self.play(Create(ejes), FadeIn(rotulos_x), run_time=1.5)
        self.play(Create(grafica_f), run_time=2.5, rate_func=linear)

        # -------------------------------------------------------------------
        # 2. Estado de la oruga: P(x) = P_{n-1}(x) + s · c_n x^n
        #    - n: grado que se está agregando (entero)
        #    - s: ValueTracker que va de 0 a 1; en s = 1 se tiene exactamente P_n.
        #    Con un rate_func que primero retrocede y después se pasa de largo,
        #    el polinomio se encoge y se estira como una oruga antes de asentarse.
        # -------------------------------------------------------------------
        estado = {"n": 1}
        s = ValueTracker(1.0)

        def coef_actuales():
            n = estado["n"]
            c = np.zeros(n + 1)
            c[:n] = COEF[:n]
            c[n] = s.get_value() * COEF[n]
            return c

        def evaluar_P():
            return np.polynomial.polynomial.polyval(XS, coef_actuales())

        def curva_P():
            # el polinomio se dibuja solo donde |P| <= Y_MAX; cada tramo termina
            # exactamente en el borde de la ventana (interpolando el cruce)
            ys = evaluar_P()
            dentro = np.abs(ys) <= Y_MAX
            tramos = VGroup()
            i = 0
            while i < len(XS):
                if not dentro[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(XS) and dentro[j + 1]:
                    j += 1
                xs, ys_t = list(XS[i:j + 1]), list(ys[i:j + 1])
                for borde, vecino in ((i, i - 1), (j, j + 1)):
                    if 0 <= vecino < len(XS):
                        yb = np.sign(ys[vecino]) * Y_MAX
                        t = (yb - ys[borde]) / (ys[vecino] - ys[borde])
                        xb = XS[borde] + t * (XS[vecino] - XS[borde])
                        if vecino < borde:
                            xs.insert(0, xb), ys_t.insert(0, yb)
                        else:
                            xs.append(xb), ys_t.append(yb)
                if len(xs) >= 2:
                    tramo = VMobject(stroke_color=COLOR_P, stroke_width=5)
                    tramo.set_points_as_corners([ejes.c2p(x, y) for x, y in zip(xs, ys_t)])
                    tramos.add(tramo)
                i = j + 1
            return tramos

        def zona_ajuste():
            # mayor intervalo [x_izq, x_der] alrededor de 0 donde |f - P| < EPS
            err = np.abs(FXS - evaluar_P())
            c = MUESTRAS // 2
            der = c
            while der + 1 < MUESTRAS and err[der + 1] < EPS:
                der += 1
            izq = c
            while izq - 1 >= 0 and err[izq - 1] < EPS:
                izq -= 1
            return XS[izq], XS[der]

        # -------------------------------------------------------------------
        # 3. Elementos que se redibujan solos en cada frame (always_redraw)
        # -------------------------------------------------------------------
        polinomio = always_redraw(curva_P)

        def banda():
            a, b = zona_ajuste()
            rect = Rectangle(
                width=max(ejes.c2p(b, 0)[0] - ejes.c2p(a, 0)[0], 1e-3),
                height=ejes.y_length,
                fill_color=COLOR_ZONA, fill_opacity=0.10, stroke_width=0,
            ).move_to(ejes.c2p((a + b) / 2, 0))
            segmento = Line(ejes.c2p(a, 0), ejes.c2p(b, 0), color=COLOR_ZONA, stroke_width=7)
            bordes = VGroup(*[
                DashedLine(ejes.c2p(x, -Y_MAX), ejes.c2p(x, Y_MAX), color=COLOR_ZONA,
                           stroke_width=2, dash_length=0.08).set_opacity(0.7)
                for x in (a, b)
            ])
            return VGroup(rect, segmento, bordes)

        zona = always_redraw(banda)

        def patitas():
            # "segmentos del cuerpo" de la oruga: puntos sobre P_n coloreados según el error
            # (verde = pegado a f, rojo = lejos de f)
            xs = np.arange(-7.0, 7.01, 0.5)
            ys = np.polynomial.polynomial.polyval(xs, coef_actuales())
            err = np.abs(F["f"](xs) - ys)
            puntos = VGroup()
            for x, y, e in zip(xs, ys, err):
                if abs(y) <= Y_MAX:
                    color = interpolate_color(GREEN_C, RED_C, min(e / 0.6, 1.0))
                    puntos.add(Dot(ejes.c2p(x, y), radius=0.055, color=color))
            return puntos

        cuerpo = always_redraw(patitas)

        # -------------------------------------------------------------------
        # 4. Anotaciones: fórmula general, P_n explícito, contador n y radio de ajuste
        # -------------------------------------------------------------------
        formula = MathTex(
            r"P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!}\,x^k", font_size=34
        ).to_corner(UL, buff=0.4).shift(DOWN * 0.75)

        n_rotulo = MathTex(r"n =", font_size=44)
        n_valor = Integer(1, font_size=44, color=COLOR_P)
        n_grupo = VGroup(n_rotulo, n_valor).arrange(RIGHT, buff=0.2)
        n_valor.add_updater(lambda m: m.next_to(n_rotulo, RIGHT, buff=0.2))

        r_rotulo = MathTex(rf"|f - P_n| < {EPS}\ \text{{en}}", font_size=28, color=COLOR_ZONA)
        r_intervalo = VGroup(
            MathTex(r"[", font_size=28), DecimalNumber(0, num_decimal_places=2, font_size=28),
            MathTex(r",", font_size=28), DecimalNumber(0, num_decimal_places=2, font_size=28),
            MathTex(r"]", font_size=28),
        ).set_color(COLOR_ZONA)

        def actualizar_intervalo(grupo):
            a, b = zona_ajuste()
            grupo[1].set_value(a)
            grupo[3].set_value(b)
            grupo.arrange(RIGHT, buff=0.06).next_to(r_rotulo, RIGHT, buff=0.15)
            grupo[2].align_to(grupo[1], DOWN).shift(DOWN * 0.06)  # la coma va a la altura de la base

        r_intervalo.add_updater(actualizar_intervalo)
        actualizar_intervalo(r_intervalo)

        panel_der = VGroup(n_grupo, VGroup(r_rotulo, r_intervalo)).arrange(DOWN, aligned_edge=RIGHT, buff=0.25)
        panel_der.to_corner(UR, buff=0.4).shift(DOWN * 0.7)

        explicito = MathTex(polinomio_tex(1), font_size=34, color=COLOR_P)
        explicito.next_to(formula, DOWN, aligned_edge=LEFT, buff=0.3)

        etiqueta_f.next_to(ejes.c2p(X_MAX, 0), UP + LEFT, buff=0.1).shift(UP * 1.2)

        # -------------------------------------------------------------------
        # 5. Aproximación inicial: la recta tangente en x0 = 0 (grado 1)
        # -------------------------------------------------------------------
        origen = Dot(ejes.c2p(0, F["f"](0)), color=WHITE, radius=0.08)
        x0 = MathTex(r"x_0 = 0", font_size=28).next_to(origen, DR, buff=0.12)
        self.play(FadeIn(origen, scale=2), Write(x0), Write(etiqueta_f))
        self.play(Flash(origen, color=WHITE, flash_radius=0.35))

        tangente_txt = Text("Grado 1: la recta tangente", font_size=26, color=COLOR_P)
        tangente_txt.next_to(ejes, UP, buff=0.15)
        self.play(Write(formula), FadeIn(n_grupo), FadeIn(tangente_txt, shift=DOWN * 0.2))
        self.play(Create(polinomio), Write(explicito), run_time=2)
        self.add(polinomio)  # se asegura que siga redibujándose tras Create
        self.wait(0.5)

        self.play(FadeIn(zona), FadeIn(cuerpo), FadeIn(r_rotulo), FadeIn(r_intervalo),
                  FadeOut(tangente_txt), run_time=1.2)
        # la banda de ajuste queda detrás de las curvas
        self.bring_to_back(zona)
        self.wait(0.8)

        # -------------------------------------------------------------------
        # 6. Crecimiento de la oruga: se agregan los términos de grado 2, 3, ..., N_MAX
        # -------------------------------------------------------------------
        for n in range(2, N_MAX + 1):
            estado["n"] = n
            s.set_value(0)                   # P = P_{n-1}: no hay salto visual
            n_valor.set_value(n)
            nuevo = MathTex(polinomio_tex(n), font_size=34, color=COLOR_P)
            nuevo.move_to(explicito, aligned_edge=LEFT)

            if COEF[n] == 0:
                # término nulo (por ejemplo, los grados pares de sin x): el contador avanza
                # pero la curva no cambia
                aviso = VGroup(
                    MathTex(rf"f^{{({n})}}(0) = 0\ \Rightarrow", font_size=26),
                    Text("término nulo", font_size=22),
                ).arrange(RIGHT, buff=0.15).set_color(GREY_B)
                aviso.next_to(panel_der, DOWN, aligned_edge=RIGHT)
                self.play(Indicate(n_valor, color=GREY_B, scale_factor=1.3), FadeIn(aviso),
                          run_time=0.5)
                self.play(FadeOut(aviso), s.animate.set_value(1), run_time=0.35)
                self.remove(explicito)
                explicito = nuevo
                self.add(explicito)
                continue

            # paso elástico: se encoge un poco, se estira de más y se asienta en P_n
            duracion = 2.0 if n <= 7 else 1.4
            self.play(
                s.animate(rate_func=rate_functions.ease_in_out_back).set_value(1),
                Indicate(n_valor, color=YELLOW, scale_factor=1.4),
                ReplacementTransform(explicito, nuevo),
                run_time=duracion,
            )
            explicito = nuevo
            self.wait(0.3)

        # -------------------------------------------------------------------
        # 7. Cierre: cerca del centro P_n abraza a f; lejos, todavía se escapa
        # -------------------------------------------------------------------
        self.wait(0.5)
        cierre = VGroup(
            Text("Cerca del centro el polinomio abraza a f,", font_size=26),
            Text("lejos del centro todavía se escapa.", font_size=26),
        ).arrange(DOWN, buff=0.12).move_to(ejes.get_bottom() + UP * 0.55)
        fondo_cierre = BackgroundRectangle(cierre, color=FONDO, fill_opacity=0.85, buff=0.15)
        self.play(FadeIn(fondo_cierre), Write(cierre), run_time=2)
        self.play(Indicate(r_intervalo, color=COLOR_ZONA, scale_factor=1.2))
        self.wait(2.5)
