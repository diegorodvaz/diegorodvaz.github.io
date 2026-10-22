from manim import *
import numpy as np

# A = [0, 2] x [0, 2): cuadrado con el lado superior abierto
L = 2

VERDE = GREEN_C
ROJO = RED_C
AMARILLO = YELLOW_C
FONDO = "#0E1116"


def dist_frontera(p):
    # distancia de un punto del cuadrado a su borde
    return min(p[0], L - p[0], p[1], L - p[1])


class TopologiaCuadrado(Scene):
    def construct(self):
        self.camera.background_color = FONDO

        # ---------- Título ----------
        titulo = Text("Topología de puntos en el plano", font_size=32).to_edge(UP, buff=0.3)
        self.play(Write(titulo), run_time=1.5)

        # ---------- Ejes ----------
        ejes = Axes(
            x_range=[-1, 3, 1], y_range=[-1, 3, 1],
            x_length=6, y_length=6,
            axis_config={"color": GREY_B, "stroke_width": 1.5, "include_tip": False,
                         "include_numbers": True, "font_size": 24},
        ).to_edge(LEFT, buff=0.6).shift(DOWN * 0.35)
        u = ejes.x_axis.unit_size

        def c(x, y):
            return ejes.c2p(x, y)

        self.play(Create(ejes), run_time=1.5)

        # ---------- El conjunto A ----------
        esquinas = [c(0, 0), c(L, 0), c(L, L), c(0, L)]
        region = Polygon(*esquinas, stroke_width=0, fill_color=BLUE, fill_opacity=0.3)
        lado_inf = Line(c(0, 0), c(L, 0), color=BLUE_B, stroke_width=5)
        lado_der = Line(c(L, 0), c(L, L), color=BLUE_B, stroke_width=5)
        lado_izq = Line(c(0, 0), c(0, L), color=BLUE_B, stroke_width=5)
        lado_sup = DashedLine(c(0, L), c(L, L), color=BLUE_B, stroke_width=4, dash_length=0.15)
        esq_abiertas = VGroup(*[
            Circle(radius=0.08, stroke_color=BLUE_B, stroke_width=3,
                   fill_color=FONDO, fill_opacity=1).move_to(p)
            for p in (c(0, L), c(L, L))
        ])
        nombre_A = MathTex("A", font_size=44, color=BLUE_B).move_to(c(0.3, 1.65))

        def panel(*items):
            g = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
            if g.width > 6.2:
                g.scale_to_fit_width(6.2)
            g.move_to(UP * 0.3)
            return g.align_to(RIGHT * 0.5, LEFT)

        def_A = panel(
            MathTex(r"A = [0,2]\times[0,2)", font_size=42),
            Text("Lados inferior, izquierdo y derecho:", font_size=24),
            Text("pertenecen a A (trazo continuo)", font_size=24, color=BLUE_B),
            Text("Lado superior:", font_size=24),
            Text("NO pertenece a A (trazo punteado)", font_size=24, color=BLUE_B),
        )

        def_A[2].shift(RIGHT * 0.4)
        def_A[4].shift(RIGHT * 0.4)

        self.play(FadeIn(region), Create(lado_inf), Create(lado_der), Create(lado_izq), run_time=2)
        self.play(Create(lado_sup), FadeIn(esq_abiertas), Write(nombre_A), run_time=1.5)
        self.play(Write(def_A[0]))
        self.play(FadeIn(def_A[1:3], shift=UP * 0.2))
        self.play(FadeIn(def_A[3:], shift=UP * 0.2), Indicate(lado_sup, color=WHITE))
        self.wait(2)

        # ---------- Bola abierta ----------
        def_bola = panel(
            Text("Bola abierta de centro p y radio ε", font_size=28),
            MathTex(r"B(p,\varepsilon)=\{x\in\mathbb{R}^2 : \|x-p\|<\varepsilon\}", font_size=36),
            Text("Clasificamos cada punto según cómo", font_size=24, color=GREY_B),
            Text("sus bolas cortan a A y a su complemento.", font_size=24, color=GREY_B),
        )
        self.play(FadeOut(def_A), FadeIn(def_bola))
        self.wait(2.5)

        # ---------- Bola coloreada: verde dentro de A, rojo fuera ----------
        cerrado = Polygon(*esquinas)

        def bola(centro, r, con_radio=False):
            disco = Circle(radius=r * u).move_to(c(*centro))
            dentro = Intersection(disco, cerrado, fill_color=VERDE, fill_opacity=0.55, stroke_width=0)
            fuera = Difference(disco, cerrado, fill_color=ROJO, fill_opacity=0.55, stroke_width=0)
            borde = DashedVMobject(disco.set_stroke(WHITE, 2), num_dashes=32)
            g = VGroup(dentro, fuera, borde)
            if con_radio:
                radio = Line(c(*centro), c(centro[0] + r, centro[1]), color=WHITE, stroke_width=2)
                eps = MathTex(r"\varepsilon", font_size=28).next_to(radio, UP, buff=0.05)
                if r * u < 0.35:
                    eps.next_to(radio, RIGHT, buff=0.08)
                g.add(radio, eps)
            return g

        def punto(centro, nombre, en_A=True, direccion=DL):
            if en_A:
                d = Dot(c(*centro), radius=0.07, color=WHITE)
            else:
                d = Circle(radius=0.07, stroke_color=WHITE, stroke_width=3,
                           fill_color=FONDO, fill_opacity=1).move_to(c(*centro))
            etiqueta = MathTex(nombre, font_size=34).next_to(d, direccion, buff=0.08)
            return VGroup(d, etiqueta)

        # ---------- Punto interior ----------
        p0 = np.array([1.0, 1.0])
        t = ValueTracker(0)
        destino = np.array([1.75, 0.25])

        def pos_p():
            return p0 + t.get_value() * (destino - p0)

        p_mob = always_redraw(lambda: punto(pos_p(), "p"))
        bola_p = always_redraw(lambda: bola(pos_p(), 0.85 * dist_frontera(pos_p()), con_radio=True))

        txt_int = panel(
            Text("Punto interior", font_size=34, color=VERDE),
            MathTex(r"\exists\,\varepsilon>0 \;:\; B(p,\varepsilon)\subset A", font_size=40),
            Text("Alguna bola alrededor de p", font_size=24, color=GREY_B),
            Text("queda completamente dentro de A.", font_size=24, color=GREY_B),
        )
        self.play(FadeOut(def_bola), FadeIn(txt_int[:2]))
        self.play(FadeIn(p_mob, scale=0.5))
        self.play(GrowFromCenter(bola_p))
        self.play(FadeIn(txt_int[2:]))
        self.wait(1.5)

        nota_eps = Text("Cerca del borde ε debe ser más chico,\npero siempre existe uno > 0.",
                        font_size=24, color=AMARILLO, line_spacing=1.2)
        nota_eps.next_to(txt_int, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(nota_eps))
        self.play(t.animate.set_value(1), run_time=4, rate_func=there_and_back_with_pause)
        self.wait(1)
        bola_p.clear_updaters()
        p_mob.clear_updaters()
        self.play(FadeOut(bola_p), FadeOut(p_mob), FadeOut(txt_int), FadeOut(nota_eps))

        # ---------- Punto exterior ----------
        q = (2.55, 1.2)
        q_mob = punto(q, "q", direccion=DR)
        bola_q = bola(q, 0.38, con_radio=True)
        txt_ext = panel(
            Text("Punto exterior", font_size=34, color=ROJO),
            MathTex(r"\exists\,\varepsilon>0 \;:\; B(q,\varepsilon)\subset \mathbb{R}^2\setminus A",
                    font_size=40),
            Text("Alguna bola alrededor de q", font_size=24, color=GREY_B),
            Text("no toca a A en absoluto.", font_size=24, color=GREY_B),
        )
        self.play(FadeIn(txt_ext[:2]))
        self.play(FadeIn(q_mob, scale=0.5))
        self.play(GrowFromCenter(bola_q))
        self.play(FadeIn(txt_ext[2:]))
        self.wait(2)
        self.play(FadeOut(bola_q), FadeOut(q_mob), FadeOut(txt_ext))

        # ---------- Punto frontera que pertenece a A ----------
        txt_fr = panel(
            Text("Punto frontera", font_size=34, color=AMARILLO),
            MathTex(r"\forall\,\varepsilon>0:\quad &B(r,\varepsilon)\cap A\neq\emptyset\\"
                    r"&B(r,\varepsilon)\cap A^{c}\neq\emptyset", font_size=38),
            Text("Toda bola, por chica que sea,", font_size=24, color=GREY_B),
            Text("tiene parte verde y parte roja.", font_size=24, color=GREY_B),
        )
        r_pt = (1.0, 0.0)
        radio_r = ValueTracker(0.6)
        r_mob = punto(r_pt, "r", direccion=DOWN)
        bola_r = always_redraw(lambda: bola(r_pt, radio_r.get_value()))
        self.play(FadeIn(txt_fr[:2]))
        self.play(FadeIn(r_mob, scale=0.5))
        self.play(GrowFromCenter(bola_r))
        self.play(FadeIn(txt_fr[2:]))
        self.play(radio_r.animate.set_value(0.12), run_time=3)
        self.play(radio_r.animate.set_value(0.6), run_time=1.5)

        en_A = MathTex(r"r\in A", font_size=40, color=BLUE_B).next_to(txt_fr, DOWN, buff=0.5, aligned_edge=LEFT)
        en_A_txt = Text("(el lado inferior es cerrado)", font_size=22, color=GREY_B).next_to(en_A, RIGHT, buff=0.3)
        self.play(Write(en_A), FadeIn(en_A_txt))
        self.wait(2)
        bola_r.clear_updaters()
        self.play(FadeOut(bola_r), FadeOut(r_mob), FadeOut(en_A), FadeOut(en_A_txt))

        # ---------- Punto frontera que NO pertenece a A ----------
        s_pt = (1.0, 2.0)
        radio_s = ValueTracker(0.6)
        s_mob = punto(s_pt, "s", en_A=False, direccion=UP)
        bola_s = always_redraw(lambda: bola(s_pt, radio_s.get_value()))
        self.play(FadeIn(s_mob, scale=0.5))
        self.play(GrowFromCenter(bola_s))
        self.play(radio_s.animate.set_value(0.12), run_time=3)
        self.play(radio_s.animate.set_value(0.6), run_time=1.5)

        no_en_A = MathTex(r"s\notin A", font_size=40, color=BLUE_B).next_to(txt_fr, DOWN, buff=0.5, aligned_edge=LEFT)
        no_en_A_txt = Text("(el lado superior es abierto)", font_size=22, color=GREY_B).next_to(no_en_A, RIGHT, buff=0.3)
        self.play(Write(no_en_A), FadeIn(no_en_A_txt))
        aviso = Text("¡Un punto frontera puede estar o no en A!", font_size=24, color=AMARILLO)
        aviso.next_to(no_en_A, DOWN, buff=0.4, aligned_edge=LEFT)
        self.play(Write(aviso))
        self.wait(2.5)
        bola_s.clear_updaters()
        self.play(FadeOut(bola_s), FadeOut(s_mob), FadeOut(no_en_A), FadeOut(no_en_A_txt),
                  FadeOut(aviso), FadeOut(txt_fr))

        # ---------- Resumen: interior, exterior y frontera ----------
        fondo_ext = Difference(
            Polygon(c(-1, -1), c(3, -1), c(3, 3), c(-1, 3)), cerrado,
            fill_color=ROJO, fill_opacity=0.25, stroke_width=0,
        )
        interior = Polygon(*esquinas, stroke_width=0, fill_color=VERDE, fill_opacity=0.45)
        frontera = VGroup(
            Line(c(0, 0), c(L, 0)), Line(c(L, 0), c(L, L)),
            Line(c(L, L), c(0, L)), Line(c(0, L), c(0, 0)),
        ).set_stroke(AMARILLO, 6)

        res = panel(
            MathTex(r"\operatorname{int} A = (0,2)\times(0,2)", font_size=38, color=VERDE),
            MathTex(r"\operatorname{ext} A = \mathbb{R}^2\setminus[0,2]^2", font_size=38, color=ROJO),
            MathTex(r"\partial A = [0,2]^2\setminus(0,2)^2", font_size=38, color=AMARILLO),
        )
        conjunto_A = VGroup(region, lado_inf, lado_der, lado_izq, lado_sup, esq_abiertas, nombre_A)

        self.play(FadeOut(conjunto_A), FadeIn(interior), Write(res[0]), run_time=1.5)
        self.play(FadeIn(fondo_ext), Write(res[1]), run_time=1.5)
        self.play(Create(frontera), Write(res[2]), run_time=2)
        self.wait(1.5)

        # ---------- Conclusión: A no es abierto ni cerrado ----------
        concl = VGroup(
            Text("A no es abierto:", font_size=26, color=WHITE),
            Text("contiene puntos de su frontera (como r).", font_size=22, color=GREY_B),
            Text("A no es cerrado:", font_size=26, color=WHITE),
            Text("no contiene toda su frontera (como s).", font_size=22, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        concl[1].shift(RIGHT * 0.4)
        concl[3].shift(RIGHT * 0.4)
        concl.next_to(res, DOWN, buff=0.6, aligned_edge=LEFT)

        lados_en_A = VGroup(frontera[0], frontera[1], frontera[3])
        self.play(FadeIn(concl[:2], shift=UP * 0.2), Indicate(lados_en_A, color=BLUE_B, scale_factor=1.05))
        self.play(FadeIn(concl[2:], shift=UP * 0.2), Indicate(frontera[2], color=BLUE_B, scale_factor=1.05))
        self.wait(3)
