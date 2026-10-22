"""
Teorema de Green: los remolinos del interior se cancelan y solo sobrevive el borde.

    manim -pql Green.py GreensTheoremAnimation
"""
from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Geometría de la región D y de su frontera C
# ---------------------------------------------------------------------------
CENTRO = np.array([-2.3, -0.55, 0.0])
SX, SY = 3.2, 2.35          # semiejes de la "elipse deformada"
H = 0.4                     # lado de las celdas de la malla principal
INSET = 0.06               # separación de las flechas respecto de la arista
FONDO = "#0b0f1a"
PALETA = [BLUE_D, TEAL_C, GREEN_C, YELLOW_C, ORANGE]


def radio(theta):
    # perturbación suave de un círculo (sigue siendo convexa)
    return 1 + 0.07 * np.cos(3 * theta) + 0.04 * np.sin(2 * theta)


def punto_curva(t):
    r = radio(t)
    return CENTRO + np.array([SX * r * np.cos(t), SY * r * np.sin(t), 0.0])


def dentro(p):
    u, v = (p[0] - CENTRO[0]) / SX, (p[1] - CENTRO[1]) / SY
    return np.hypot(u, v) < radio(np.arctan2(v, u))


def rotor(p):
    # densidad de rotor (∂Q/∂x − ∂P/∂y), siempre positiva: todas las celdas giran antihorario
    u, v = (p[0] - CENTRO[0]) / SX, (p[1] - CENTRO[1]) / SY
    return 1 + 0.5 * np.sin(2.2 * u + 0.5) * np.cos(1.8 * v)


def color_rotor(w):
    alpha = np.clip((w - 0.5) / 1.0, 0, 1)
    return color_gradient(PALETA, 101)[int(round(alpha * 100))]


def vertice(h, a, b):
    # vértice entero (a, b) de la malla de lado h -> punto del plano
    return CENTRO + np.array([a * h, b * h, 0.0])


def lados_ccw(i, j):
    # aristas de la celda (i, j) recorridas antihorario, con la celda vecina que la comparte
    return [
        ((i, j), (i + 1, j), (i, j - 1)),              # abajo   ->
        ((i + 1, j), (i + 1, j + 1), (i + 1, j)),      # derecha ^
        ((i + 1, j + 1), (i, j + 1), (i, j + 1)),      # arriba  <-
        ((i, j + 1), (i, j), (i - 1, j)),              # izquierda v
    ]


def celdas(h):
    # celdas cuyas cuatro esquinas caen dentro de la curva
    n = int(4.5 / h) + 1
    return {
        (i, j)
        for i in range(-n, n)
        for j in range(-n, n)
        if all(dentro(vertice(h, *a)) for a, _, _ in lados_ccw(i, j))
    }


def contorno_escalera(h, **estilo):
    """Frontera de la unión de celdas: las aristas que no se cancelan, encadenadas en orden."""
    cs = celdas(h)
    siguiente = {a: b for (i, j) in cs for a, b, vecina in lados_ccw(i, j) if vecina not in cs}
    # arrancamos en el vértice más cercano al ángulo 0, igual que la curva paramétrica
    inicio = min(siguiente, key=lambda v: abs(np.arctan2(v[1] * h / SY, v[0] * h / SX)) - 1e-3 * v[0])
    recorrido = [inicio]
    while (v := siguiente[recorrido[-1]]) != inicio:
        recorrido.append(v)
    pts = [vertice(h, a, b) for a, b in recorrido]
    return VMobject(**estilo).set_points_as_corners(pts + [pts[0]])


class GreensTheoremAnimation(Scene):
    def construct(self):
        self.camera.background_color = FONDO
        self.rotulo_actual = None

        # -------------------------------------------------------------------
        # 0. Fórmula y curva cerrada C
        # -------------------------------------------------------------------
        formula = MathTex(
            r"\oint_{C} P\,dx + Q\,dy",
            r"=",
            r"\iint_{D} \left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right) dA",
            font_size=40,
        ).to_edge(UP, buff=0.3)

        curva = ParametricFunction(punto_curva, t_range=[0, TAU], color=WHITE, stroke_width=4)
        etiqueta_C = MathTex("C", font_size=40).move_to(punto_curva(0.75) + 0.35 * UR)
        etiqueta_D = MathTex("D", font_size=48, color=GREY_B).move_to(CENTRO)

        self.play(Write(formula), run_time=1.5)
        self.play(Create(curva), FadeIn(etiqueta_C), run_time=2)
        self.play(FadeIn(etiqueta_D))
        self.cambiar_rotulo("Una curva cerrada C", "encierra una región D.")
        self.wait(0.5)

        # -------------------------------------------------------------------
        # 1. Malla de celdas coloreadas por el rotor + remolinos microscópicos
        # -------------------------------------------------------------------
        cs = sorted(celdas(H), key=lambda c: np.linalg.norm(vertice(H, c[0] + 0.5, c[1] + 0.5) - CENTRO))
        cuadros, remolinos, flechas_celda = VGroup(), VGroup(), VGroup()
        aristas = {}   # arista (par de vértices) -> lista de (flecha, desplazamiento hacia la arista)

        for i, j in cs:
            centro = vertice(H, i + 0.5, j + 0.5)
            w = rotor(centro)
            col = color_rotor(w)

            cuadros.add(
                Square(side_length=H)
                .move_to(centro)
                .set_fill(col, opacity=0.55)
                .set_stroke(WHITE, width=1, opacity=0.25)
            )

            # remolino: arco con punta que gira más rápido donde el rotor es mayor
            remolino = Arc(radius=0.12, start_angle=0, angle=1.5 * PI, arc_center=centro,
                           color=interpolate_color(col, WHITE, 0.4), stroke_width=3)
            remolino.add_tip(tip_length=0.08, tip_width=0.08)
            remolino.centro, remolino.omega = centro, w
            remolinos.add(remolino)

            # circulación de la celda: cuatro flechas antihorario, levemente metidas hacia dentro
            flechas = VGroup()
            for a, b, _ in lados_ccw(i, j):
                pa, pb = vertice(H, *a), vertice(H, *b)
                medio, d = (pa + pb) / 2, normalize(pb - pa)
                hacia_dentro = normalize(centro - medio) * INSET
                flecha = Arrow(
                    medio - d * 0.275 * H + hacia_dentro, medio + d * 0.275 * H + hacia_dentro,
                    buff=0, stroke_width=3, tip_length=0.09,
                    max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=12,
                    color=interpolate_color(col, WHITE, 0.55),
                )
                flechas.add(flecha)
                aristas.setdefault(frozenset((a, b)), []).append((flecha, -hacia_dentro, medio))
            flechas_celda.add(flechas)

        self.play(
            FadeOut(etiqueta_D),
            LaggedStart(*[FadeIn(c, scale=0.4) for c in cuadros], lag_ratio=0.015),
            curva.animate.set_stroke(opacity=0.5),
            run_time=2.5,
        )
        self.cambiar_rotulo(
            "Cada celda aporta su", "rotor microscópico:",
            MathTex(r"\left(\tfrac{\partial Q}{\partial x}-\tfrac{\partial P}{\partial y}\right)\Delta A", font_size=34),
        )
        self.play(LaggedStart(*[GrowFromCenter(r) for r in remolinos], lag_ratio=0.01), run_time=1.5)

        # -------------------------------------------------------------------
        # 2. Los remolinos giran (velocidad proporcional al rotor local)
        # -------------------------------------------------------------------
        for r in remolinos:
            r.add_updater(lambda m, dt: m.rotate(2.5 * m.omega * dt, about_point=m.centro))
        self.wait(3)
        for r in remolinos:
            r.clear_updaters()

        # Cada remolino se convierte en la circulación alrededor del borde de su celda
        self.cambiar_rotulo("Cada remolino es una", "pequeña circulación", "antihoraria en su celda.")
        self.play(
            *[ReplacementTransform(r, f) for r, f in zip(remolinos, flechas_celda)],
            run_time=1.8,
        )
        self.wait(0.5)

        # -------------------------------------------------------------------
        # 2b. Zoom conceptual en dos celdas vecinas: la arista común se recorre dos veces
        # -------------------------------------------------------------------
        par = min(
            (k for k, v in aristas.items() if len(v) == 2 and len({a[1] for a in k}) == 2),
            key=lambda k: np.linalg.norm(np.mean([vertice(H, *a) for a in k], axis=0) - CENTRO),
        )
        (fa, da, medio_par), (fb, db, _) = aristas.pop(par)
        idx_par = [n for n, f in enumerate(flechas_celda) if fa in f or fb in f]
        velo = FullScreenRectangle(fill_color=FONDO, fill_opacity=0.75, stroke_width=0).set_z_index(1)
        for n in idx_par:
            cuadros[n].set_z_index(2)
            flechas_celda[n].set_z_index(3)
        arista_comun = Line(*[vertice(H, *a) for a in par], color=YELLOW, stroke_width=7).set_z_index(2)

        self.play(FadeIn(velo), run_time=0.8)
        self.cambiar_rotulo(
            "En la arista común,", "las dos celdas la recorren", "en sentidos opuestos:",
            MathTex(r"+\int \mathbf F\cdot d\mathbf r \;\;\text{y}\;\; -\int \mathbf F\cdot d\mathbf r", font_size=34),
        )
        self.play(Create(arista_comun), Indicate(fa, scale_factor=1.6), Indicate(fb, scale_factor=1.6), run_time=1.5)

        # choque: ambas flechas se encuentran sobre la arista y se aniquilan
        chispa = self.chispa(medio_par)
        self.play(
            fa.animate.shift(da).set_opacity(0),
            fb.animate.shift(db).set_opacity(0),
            arista_comun.animate.set_stroke(opacity=0),
            chispa,
            run_time=1.2,
        )
        self.remove(chispa.mobject, arista_comun)
        self.cambiar_rotulo("Se cancelan:", "su aporte neto es 0.", color=YELLOW_B)
        self.wait(0.6)
        self.play(FadeOut(velo), run_time=0.8)

        # -------------------------------------------------------------------
        # 3. Cancelación masiva: una onda de choques que se propaga desde el centro
        # -------------------------------------------------------------------
        internas = sorted(
            (v for v in aristas.values() if len(v) == 2),
            key=lambda v: np.linalg.norm(v[0][2] - CENTRO),
        )
        borde = VGroup(*[v[0][0] for v in aristas.values() if len(v) == 1])

        self.cambiar_rotulo("Lo mismo ocurre en", "TODAS las aristas", "interiores...")
        choques, chispas = [], []
        for (f1, d1, medio), (f2, d2, _) in internas:
            ch = self.chispa(medio)
            chispas.append(ch.mobject)
            choques.append(AnimationGroup(
                f1.animate.shift(d1).set_opacity(0),
                f2.animate.shift(d2).set_opacity(0),
                ch,
            ))
        self.play(
            LaggedStart(*choques, lag_ratio=0.03),
            cuadros.animate.set_fill(opacity=0.2).set_stroke(opacity=0.1),
            run_time=5,
        )
        self.remove(*chispas)

        # -------------------------------------------------------------------
        # 4. El interior se desvanece: solo sobrevive la circulación del borde
        # -------------------------------------------------------------------
        self.cambiar_rotulo("El interior se anula.", "Solo sobrevive", "la circulación del borde.", color=YELLOW_B)
        self.play(FadeOut(cuadros), borde.animate.set_color(YELLOW), run_time=1.5)
        self.play(LaggedStart(*[Indicate(f, color=WHITE, scale_factor=1.4) for f in borde], lag_ratio=0.02),
                  run_time=2)

        # las flechas del borde se funden en un contorno escalonado
        estilo = dict(stroke_color=YELLOW, stroke_width=5)
        escalera = contorno_escalera(H, **estilo)
        self.play(Create(escalera), borde.animate.set_opacity(0), run_time=1.5)
        self.remove(borde)

        # refinamos la malla: la escalera tiende a la curva C
        self.cambiar_rotulo("Con celdas más finas,", "el borde escalonado", "tiende a la curva C.",
                            MathTex(r"\Delta A \to 0", font_size=36, color=YELLOW_B))
        for h in (H / 2, H / 4, H / 8):
            self.play(Transform(escalera, contorno_escalera(h, **estilo)), run_time=1.1)
        curva_final = curva.copy().set_stroke(YELLOW, width=5, opacity=1)
        self.play(Transform(escalera, curva_final), FadeOut(curva), run_time=1.2)

        # -------------------------------------------------------------------
        # 5. El borde brilla y una corriente de partículas recorre C (integral de línea)
        # -------------------------------------------------------------------
        brillo = VGroup(*[
            curva_final.copy().set_stroke(YELLOW, width=wd, opacity=op)
            for wd, op in [(28, 0.05), (20, 0.08), (13, 0.14), (8, 0.25)]
        ])
        fase = ValueTracker(0)
        n_part, n_cola = 14, 7
        particulas = VGroup(*[
            Dot(radius=0.07 * (1 - c / n_cola) + 0.015, color=interpolate_color(WHITE, YELLOW, c / n_cola))
            .set_opacity(1 - c / n_cola)
            for _ in range(n_part) for c in range(n_cola)
        ])

        def mover_particulas(grupo):
            for k in range(n_part):
                for c in range(n_cola):
                    s = (fase.get_value() + k / n_part - 0.006 * c) % 1
                    grupo[k * n_cola + c].move_to(punto_curva(TAU * s))

        particulas.add_updater(mover_particulas)
        mover_particulas(particulas)

        self.cambiar_rotulo("Circulación neta:", MathTex(r"\oint_C \mathbf F\cdot d\mathbf r", font_size=40))
        self.play(FadeIn(brillo), FadeIn(particulas), etiqueta_C.animate.set_color(YELLOW), run_time=1)
        self.play(fase.animate.set_value(0.6), run_time=4, rate_func=linear)

        # Conexión final con la fórmula
        caja_borde = SurroundingRectangle(formula[0], color=YELLOW, buff=0.1)
        caja_interior = SurroundingRectangle(formula[2], color=TEAL_C, buff=0.1)
        self.cambiar_rotulo("Lo que gira dentro", "= lo que circula en el borde.", color=WHITE)
        self.play(
            Create(caja_borde), Create(caja_interior),
            fase.animate.set_value(1.2),
            run_time=4, rate_func=linear,
        )
        self.play(fase.animate.set_value(1.6), run_time=2.5, rate_func=linear)
        particulas.clear_updaters()
        self.wait(1)

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------
    def cambiar_rotulo(self, *lineas, color=WHITE):
        """Reemplaza el texto explicativo del panel derecho."""
        nuevo = VGroup(*[
            Text(l, font_size=24, color=color) if isinstance(l, str) else l for l in lineas
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        if nuevo.width > 4.4:
            nuevo.scale_to_fit_width(4.4)
        nuevo.move_to([4.5, 0.0, 0]).set_z_index(5)   # siempre por encima del velo
        anims = [FadeIn(nuevo, shift=0.2 * UP)]
        if self.rotulo_actual is not None:
            anims.append(FadeOut(self.rotulo_actual, shift=0.2 * UP))
        self.play(*anims, run_time=0.7)
        self.rotulo_actual = nuevo

    @staticmethod
    def chispa(punto):
        """Destello breve en el punto donde dos flechas opuestas se aniquilan."""
        punto_luz = Dot(punto, radius=0.03, color=WHITE).set_opacity(0).set_z_index(4)

        def destello(m, a):
            m.set_width(0.06 + 0.3 * a).set_opacity(np.sin(PI * a))

        return UpdateFromAlphaFunc(punto_luz, destello)
