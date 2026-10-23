"""
Partícula cargada en un campo magnético: círculo → hélice → selector de velocidades.

    manim -pql particula_campo_B.py Circulo
    manim -pql particula_campo_B.py Helice
    manim -pql particula_campo_B.py SelectorVelocidades

Unidades adimensionales (q = m = 1). Todas las trayectorias se integran con el
método de Boris a partir de dv/dt = (q/m)(E + v × B) y después se recorren con
un ValueTracker de tiempo. `python particula_campo_B.py` corre las verificaciones.
"""
from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Paleta (la misma en las tres escenas)
# ---------------------------------------------------------------------------
FONDO = "#0b0f1a"
C_V = YELLOW_C            # velocidad
C_V_COMP = "#d9a520"      # componentes de la velocidad
C_FB = "#c77dff"          # fuerza magnética
C_FE = GREEN_C            # fuerza eléctrica
C_CAMPO_B = GREY_B        # campo B (tenue)
C_CAMPO_E = "#6f9f5e"     # campo E (tenue)
C_POS = "#ff8c42"         # carga positiva
C_NEG = "#5dade2"         # carga negativa
C_TEXTO = GREY_A

config.background_color = FONDO
R_CARGA = 0.13
CERO = np.zeros(3)


# ---------------------------------------------------------------------------
# Integración numérica (Boris) y trayectorias
# ---------------------------------------------------------------------------
def cruz(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


class Trayectoria:
    def __init__(self, ts, rs, vs):
        self.ts, self.rs, self.vs = ts, rs, vs
        self.t_fin = ts[-1]

    def _interp(self, datos, t):
        t = np.clip(t, 0, self.t_fin)
        return np.array([np.interp(t, self.ts, datos[:, k]) for k in range(3)])

    def pos(self, t):
        return self._interp(self.rs, t)

    def vel(self, t):
        return self._interp(self.vs, t)

    def tiempo_en_x(self, x):
        # primer instante en que la partícula alcanza la abscisa x
        i = np.argmax(self.rs[:, 0] >= x)
        return self.ts[i]


def integrar(r0, v0, campos, t_max, dt=1e-3, q=1.0, m=1.0, parar=None):
    """Método de Boris para dv/dt = (q/m)(E + v × B)."""
    r, v = np.array(r0, float), np.array(v0, float)
    ts, rs, vs = [0.0], [r.copy()], [v.copy()]
    k = q / m * dt / 2
    for i in range(int(round(t_max / dt))):
        E, B = campos(r)
        v_menos = v + k * E
        tb = k * B
        s = 2 * tb / (1 + tb @ tb)
        v_prima = v_menos + cruz(v_menos, tb)
        v = v_menos + cruz(v_prima, s) + k * E
        r = r + v * dt
        ts.append((i + 1) * dt)
        rs.append(r.copy())
        vs.append(v.copy())
        if parar is not None and parar(r):
            break
    return Trayectoria(np.array(ts), np.array(rs), np.array(vs))


def inicio_para_centro(centro, v0, B, q=1.0, m=1.0):
    # el centro de giro está en r + m (v × B) / (q B²): despejamos r
    return np.array(centro, float) - m * cruz(v0, B) / (q * (B @ B))


def campo_uniforme(E, B):
    E, B = np.array(E, float), np.array(B, float)
    return lambda r: (E, B)


# ---------------------------------------------------------------------------
# Piezas gráficas
# ---------------------------------------------------------------------------
def otimes(radio=0.09, color=C_CAMPO_B, opacidad=0.35, grosor=1.5):
    d = radio / np.sqrt(2)
    return VGroup(
        Circle(radius=radio),
        Line([-d, -d, 0], [d, d, 0]),
        Line([-d, d, 0], [d, -d, 0]),
    ).set_stroke(color, grosor, opacidad)


def grilla_B(xs, ys, **kw):
    return VGroup(*[otimes(**kw).move_to([x, y, 0]) for x in xs for y in ys])


def carga(signo=+1, radio=R_CARGA):
    color = C_POS if signo > 0 else C_NEG
    bola = Circle(radius=radio, fill_color=color, fill_opacity=1,
                  stroke_color=WHITE, stroke_width=1.5)
    s = MathTex("+" if signo > 0 else "-", color=BLACK, font_size=26).move_to(bola)
    return VGroup(bola, s)


def flecha(p, vec, color, grosor=5):
    return Arrow(p, p + vec, buff=0, color=color, stroke_width=grosor,
                 max_tip_length_to_length_ratio=0.22, max_stroke_width_to_length_ratio=8)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def flecha3d(a, b, color, grosor=4, alto=0.25, radio=0.08, opacidad=1.0):
    # Arrow3D tiene ~1000 caras y vuelve lentísimo el renderer de Cairo:
    # alcanza con una línea y un cono de 8 caras
    a, b = np.array(a, float), np.array(b, float)
    w = unit(b - a)
    e1 = unit(cruz(w, RIGHT if abs(w[0]) < 0.9 else UP))
    e2 = cruz(w, e1)
    linea = Line(a, b - alto * w, color=color, stroke_width=grosor, stroke_opacity=opacidad)
    punta = Surface(
        lambda u, v: b - u * alto * w + u * radio * (np.cos(v) * e1 + np.sin(v) * e2),
        u_range=[0, 1], v_range=[0, TAU], resolution=(1, 8), checkerboard_colors=False,
        fill_color=color, fill_opacity=opacidad, stroke_width=0)
    return VGroup(linea, punta)


def con_fondo(m, buff=0.1):
    return m.add_background_rectangle(color=FONDO, opacity=0.85, buff=buff)


# ---------------------------------------------------------------------------
# Escena 1: círculo
# ---------------------------------------------------------------------------
B1 = np.array([0.0, 0.0, -1.0])        # entrando a la pantalla
CAMPOS_1 = campo_uniforme(CERO, B1)
CENTRO_1 = np.array([-3.4, -0.3, 0.0])
V_1 = 1.5                              # r = mv/(qB) = 1.5
T_1 = 2 * PI                           # período 2πm/(qB)
VS, FS = 0.8, 0.6                      # escalas de las flechas v y F
INICIO_PAR = np.array([-3.4, -2.6, 0.0])


def tray_circulo(v, centro=None, inicio=None, q=1.0, vueltas=2.1):
    v0 = np.array([v, 0.0, 0.0])
    if inicio is None:
        inicio = inicio_para_centro(centro, v0, B1, q=q)
    return integrar(inicio, v0, CAMPOS_1, vueltas * T_1, q=q)


class Circulo(Scene):
    def construct(self):
        titulo = Text("Partícula en un campo B", font_size=42).to_edge(UP, buff=0.3)
        self.play(Write(titulo))
        self.wait(0.5)

        # --- campo B entrando ---
        grilla = grilla_B(np.arange(-6.6, 6.7, 0.8), np.arange(-3.6, 2.9, 0.8))
        etiq_B = VGroup(otimes(radio=0.16, color=GREY_A, opacidad=1, grosor=2.5),
                        MathTex(r"\vec B", color=GREY_A),
                        Text("entra a la pantalla", font_size=22, color=GREY_B)
                        ).arrange(RIGHT, buff=0.15).move_to([-4.6, 2.75, 0])
        con_fondo(etiq_B)
        self.play(LaggedStartMap(FadeIn, grilla, lag_ratio=0.004, run_time=1.5))
        self.play(FadeIn(etiq_B))
        self.wait(0.5)

        # --- carga, velocidad y fuerza ---
        tray = tray_circulo(V_1, centro=CENTRO_1)
        t = ValueTracker(0)

        def p():
            return tray.pos(t.get_value())

        def v():
            return tray.vel(t.get_value())

        def F():
            return cruz(v(), B1)          # q = 1

        particula = carga(+1).move_to(p())
        fl_v = flecha(p(), VS * v(), C_V)
        fl_F = flecha(p(), FS * F(), C_FB)
        lab_v = MathTex(r"\vec v", color=C_V).move_to(p() + VS * v() + 0.3 * unit(v()))
        lab_F = MathTex(r"\vec F", color=C_FB, font_size=40).move_to(
            p() + FS * F() + 0.28 * unit(F()))

        self.play(FadeIn(particula, scale=0.5))
        self.play(GrowArrow(fl_v), FadeIn(lab_v))
        self.wait(0.5)

        f_F = MathTex(r"\vec F", "=", "q", r"\,\vec v", r"\times", r"\vec B", font_size=46)
        f_F[0].set_color(C_FB)
        f_F[3].set_color(C_V)
        f_F[5].set_color(GREY_A)
        f_F.move_to([3.4, 2.0, 0])
        con_fondo(f_F)
        self.play(GrowArrow(fl_F), FadeIn(lab_F), Write(f_F))
        self.wait(1)

        # versiones dinámicas: idénticas en t = 0
        fl_v_din = always_redraw(lambda: flecha(p(), VS * v(), C_V))
        fl_F_din = always_redraw(lambda: flecha(p(), FS * F(), C_FB))
        self.remove(fl_v, fl_F)
        self.add(fl_v_din, fl_F_din)
        particula.add_updater(lambda m: m.move_to(p()))
        lab_v.add_updater(lambda m: m.move_to(p() + VS * v() + 0.3 * unit(v())))
        lab_F.add_updater(lambda m: m.move_to(p() + FS * F() + 0.28 * unit(F())))
        rastro = TracedPath(lambda: p(), stroke_color=C_POS, stroke_width=3, stroke_opacity=0.8)
        self.add(rastro, fl_v_din, fl_F_din, particula, lab_v, lab_F)

        self.play(t.animate.set_value(T_1), run_time=6, rate_func=linear)

        texto_perp = VGroup(
            MathTex(r"\vec F \perp \vec v\ \ \rightarrow", font_size=44),
            Text("cambia la dirección,\nno la rapidez", font_size=28, line_spacing=0.8),
        ).arrange(RIGHT, buff=0.25).move_to([3.6, 0.6, 0])
        texto_perp[0][0][0:2].set_color(C_FB)
        texto_perp[0][0][3:5].set_color(C_V)
        con_fondo(texto_perp)
        self.play(FadeIn(texto_perp, shift=0.2 * UP),
                  t.animate.set_value(2 * T_1), run_time=6, rate_func=linear)
        self.wait(0.5)

        # --- radio ---
        linea_r = DashedLine(CENTRO_1, CENTRO_1 + V_1 * LEFT, color=WHITE, stroke_width=2.5)
        lab_r = MathTex("r", font_size=38).next_to(linea_r, UP, buff=0.1)
        centro_pt = Dot(CENTRO_1, radius=0.04, color=WHITE)
        f_r = MathTex(r"r = \frac{m v}{q B}", font_size=50).move_to([3.4, -1.1, 0])
        con_fondo(f_r)
        self.play(Create(linea_r), FadeIn(centro_pt), FadeIn(lab_r))
        self.play(Write(f_r))
        self.wait(2)

        for m in (particula, lab_v, lab_F):
            m.clear_updaters()
        self.play(FadeOut(VGroup(particula, lab_v, lab_F, fl_v_din, fl_F_din, rastro,
                                 linea_r, lab_r, centro_pt, texto_perp, f_F)),
                  f_r.animate.move_to([3.4, 2.0, 0]))

        # --- dos partículas: v y 2v ---
        t2 = ValueTracker(0)
        tr1 = tray_circulo(1.0, inicio=INICIO_PAR)
        tr2 = tray_circulo(2.0, inicio=INICIO_PAR)
        VS2 = 0.6

        def par(tr, nombre, color_rastro):
            q_ = carga(+1).move_to(tr.pos(0))
            fl = always_redraw(lambda: flecha(tr.pos(t2.get_value()),
                                              VS2 * tr.vel(t2.get_value()), C_V, grosor=4))
            lab = MathTex(nombre, color=C_V, font_size=36)
            lab.add_updater(lambda m: m.move_to(
                tr.pos(t2.get_value()) + VS2 * tr.vel(t2.get_value())
                + 0.3 * unit(tr.vel(t2.get_value()))))
            q_.add_updater(lambda m: m.move_to(tr.pos(t2.get_value())))
            ras = TracedPath(q_.get_center, stroke_color=color_rastro, stroke_width=3,
                             stroke_opacity=0.85)
            return q_, fl, lab, ras

        q1, fl1, lab1, ras1 = par(tr1, r"\vec v", TEAL_C)
        q2, fl2, lab2, ras2 = par(tr2, r"2\vec v", C_POS)
        # la etiqueta "v" arranca encima de la flecha de 2v: la corro hacia abajo al inicio
        lab1.clear_updaters()
        lab1.next_to(INICIO_PAR + VS2 * np.array([1.0, 0, 0]), DOWN, buff=0.12)
        lab2.update()
        self.add(ras1, ras2)
        self.play(FadeIn(q1), FadeIn(q2), FadeIn(fl2), FadeIn(fl1), FadeIn(lab1), FadeIn(lab2))
        self.add(fl2, fl1, q1, q2)
        self.wait(1)
        self.play(FadeOut(lab1, run_time=0.3))
        lab1.add_updater(lambda m: m.move_to(
            tr1.pos(t2.get_value()) + VS2 * tr1.vel(t2.get_value())
            + 0.3 * unit(tr1.vel(t2.get_value()))))
        self.add(lab1)

        self.play(t2.animate.set_value(T_1), run_time=5, rate_func=linear)
        self.play(Flash(INICIO_PAR, color=WHITE, line_length=0.3, flash_radius=0.35))

        mismo_t = con_fondo(Text("¡llegan juntas!", font_size=26, color=WHITE).move_to(
            INICIO_PAR + 0.6 * DOWN), buff=0.06)
        self.play(FadeIn(mismo_t))
        self.wait(0.5)

        # radios r y 2r
        c1, c2 = INICIO_PAR + UP * 1.0, INICIO_PAR + UP * 2.0
        rad1 = DashedLine(c1, c1 + LEFT * 1.0, color=TEAL_C, stroke_width=2.5)
        rad2 = DashedLine(c2, c2 + LEFT * 2.0, color=C_POS, stroke_width=2.5)
        l_r1 = MathTex("r", font_size=34, color=TEAL_C).next_to(rad1, UP, buff=0.08)
        l_r2 = MathTex("2r", font_size=34, color=C_POS).next_to(rad2, UP, buff=0.08)
        self.play(Create(rad1), Create(rad2), FadeIn(l_r1), FadeIn(l_r2))

        f_w = MathTex(r"\omega = \frac{qB}{m}", font_size=50)
        f_T = MathTex(r"T = \frac{2\pi m}{qB}", font_size=50)
        txt_T = Text("el período no depende de v", font_size=28, color=WHITE)
        bloque = VGroup(VGroup(f_w, f_T).arrange(RIGHT, buff=0.8), txt_T).arrange(
            DOWN, buff=0.35).move_to([3.4, 0.0, 0])
        con_fondo(bloque)
        self.play(Write(f_w), FadeOut(mismo_t))
        self.play(Write(f_T))
        self.play(FadeIn(txt_T, shift=0.2 * UP),
                  t2.animate.set_value(2 * T_1), run_time=5, rate_func=linear)
        self.play(Flash(INICIO_PAR, color=WHITE, line_length=0.3, flash_radius=0.35))
        self.wait(2)

        for m in (q1, q2, lab1, lab2):
            m.clear_updaters()
        self.play(FadeOut(VGroup(q1, q2, fl1, fl2, lab1, lab2, ras1, ras2, rad1, rad2,
                                 l_r1, l_r2, bloque, f_r)))

        # --- carga negativa: gira al revés ---
        t3 = ValueTracker(0)
        tr_pos = tray_circulo(1.2, centro=np.array([-4.6, -0.6, 0]), vueltas=1.05)
        tr_neg = tray_circulo(1.2, centro=np.array([-1.6, -0.6, 0]), q=-1.0, vueltas=1.05)
        objs, rastros = VGroup(), VGroup()
        for tr, s, col in ((tr_pos, +1, C_POS), (tr_neg, -1, C_NEG)):
            qq = carga(s).move_to(tr.pos(0))
            qq.add_updater(lambda m, tr=tr: m.move_to(tr.pos(t3.get_value())))
            fl = always_redraw(lambda tr=tr: flecha(tr.pos(t3.get_value()),
                                                    VS * tr.vel(t3.get_value()), C_V, grosor=4))
            ras = TracedPath(qq.get_center, stroke_color=col, stroke_width=3)
            self.add(ras)
            rastros.add(ras)
            objs.add(fl, qq)
        txt_pos = Text("q > 0: antihorario", font_size=24, color=C_POS).move_to([-4.6, -2.6, 0])
        txt_neg = Text("q < 0: horario", font_size=24, color=C_NEG).move_to([-1.6, -2.6, 0])
        txt_neg_der = VGroup(Text("Con carga negativa", font_size=30),
                             Text("gira en sentido contrario", font_size=30)
                             ).arrange(DOWN, buff=0.15).move_to([3.4, 0.6, 0])
        con_fondo(txt_neg_der)
        con_fondo(txt_pos)
        con_fondo(txt_neg)
        self.play(FadeIn(objs), FadeIn(txt_neg_der))
        self.play(t3.animate.set_value(1.05 * T_1), FadeIn(txt_pos), FadeIn(txt_neg),
                  run_time=5, rate_func=linear)
        self.wait(2)
        for m in (*objs, *rastros):
            m.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)))


# ---------------------------------------------------------------------------
# Escena 2: hélice
# ---------------------------------------------------------------------------
B2 = np.array([0.0, 0.0, 1.0])         # a lo largo de +z
V_PERP, V_PAR = 1.4, 0.42
Z0 = 0.3
ANG_2 = 20 * DEGREES                 # dirección de v⊥ (así el inicio no cae sobre un eje)
V0_2 = np.array([V_PERP * np.cos(ANG_2), V_PERP * np.sin(ANG_2), V_PAR])
R0_2 = inicio_para_centro([0, 0, Z0], V0_2, B2)
# giro horario visto desde +z: llega a (r, 0) tras recorrer su ángulo polar inicial
T_X_R = (np.arctan2(R0_2[1], R0_2[0]) % TAU) / TAU * 2 * PI
T_2 = 2 * PI
PASO = 2 * PI * V_PAR                  # p = 2π m v∥/(qB), solo para comparar


def tray_helice():
    return integrar(R0_2, V0_2, campo_uniforme(CERO, B2), 2.05 * T_2)


class Helice(ThreeDScene):
    def fijo(self, *mobs):
        self.add_fixed_in_frame_mobjects(*mobs)
        self.remove(*mobs)

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-75 * DEGREES, zoom=0.78)

        ejes = ThreeDAxes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[0, 6, 1],
                          x_length=6, y_length=6, z_length=6,
                          axis_config={"color": GREY_B, "stroke_width": 2})
        ejes.shift(-ejes.c2p(0, 0, 3))      # la mitad de la hélice queda en el centro
        c2p = lambda q: ejes.c2p(*q)
        etiq_ejes = VGroup(MathTex("x").move_to(c2p([3.4, 0, 0])),
                           MathTex("y").move_to(c2p([0, 3.4, 0])),
                           MathTex("z").move_to(c2p([0, 0, 6.4]))).set_color(GREY_B)

        flechas_B = VGroup(*[
            flecha3d(c2p([x, y, 0.2]), c2p([x, y, 5.6]), C_CAMPO_B, grosor=2.5,
                     radio=0.06, opacidad=0.4)
            for x, y in ((2.4, 2.4), (-2.4, 2.4), (2.4, -2.4), (-2.4, -2.4))
        ])
        lab_B = MathTex(r"\vec B", color=GREY_A).move_to(c2p([-2.4, 2.4, 6.0]))

        titulo = Text("Velocidad inclinada respecto de B", font_size=34).to_corner(UL, buff=0.35)
        self.fijo(titulo)
        self.play(Write(titulo))
        self.add_fixed_orientation_mobjects(*etiq_ejes, lab_B)
        self.play(Create(ejes), FadeIn(etiq_ejes))
        self.play(FadeIn(flechas_B), FadeIn(lab_B))
        txt_B = MathTex(r"\vec B \parallel \hat z", font_size=40, color=GREY_A).next_to(
            titulo, DOWN, buff=0.25, aligned_edge=LEFT)
        self.fijo(txt_B)
        self.play(FadeIn(txt_B))

        # --- velocidad y su descomposición ---
        tray = tray_helice()
        t = ValueTracker(0)
        VS3 = 2.0
        P0 = c2p(R0_2)
        particula = Dot3D(P0, radius=0.09, color=C_POS)
        v_vec = VS3 * V0_2
        v_perp = VS3 * np.array([V0_2[0], V0_2[1], 0])
        v_par = VS3 * np.array([0, 0, V0_2[2]])
        a_v = flecha3d(P0, P0 + v_vec, C_V, grosor=5)
        a_perp = flecha3d(P0, P0 + v_perp, C_V_COMP, grosor=4, alto=0.2, radio=0.065)
        a_par = flecha3d(P0, P0 + v_par, C_V_COMP, grosor=4, alto=0.2, radio=0.065)
        punt = VGroup(DashedLine(P0 + v_vec, P0 + v_perp, color=C_V_COMP, stroke_width=2),
                      DashedLine(P0 + v_vec, P0 + v_par, color=C_V_COMP, stroke_width=2))
        l_v = MathTex(r"\vec v", color=C_V).move_to(P0 + v_vec + 0.4 * RIGHT + 0.15 * OUT)
        l_perp = MathTex(r"v_\perp", color=C_V_COMP).move_to(P0 + v_perp + 0.45 * RIGHT + 0.2 * IN)
        l_par = MathTex(r"v_\parallel", color=C_V_COMP).move_to(P0 + v_par + 0.45 * LEFT + 0.1 * OUT)
        self.add_fixed_orientation_mobjects(l_v, l_perp, l_par)
        self.remove(l_v, l_perp, l_par)

        self.play(FadeIn(particula))
        # la etiqueta del eje y queda detrás de la flecha v: la escondo mientras tanto
        self.play(FadeIn(a_v), FadeIn(l_v), FadeOut(etiq_ejes[1]))
        self.wait(0.5)
        self.play(Create(punt), FadeIn(a_perp), FadeIn(a_par), FadeIn(l_perp), FadeIn(l_par))

        f_desc = VGroup(
            VGroup(MathTex(r"v_\perp", color=C_V_COMP, font_size=38),
                   Text("en el plano xy", font_size=24)).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"v_\parallel", color=C_V_COMP, font_size=38),
                   Text("a lo largo de B", font_size=24)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UR, buff=0.5)
        self.fijo(f_desc)
        self.play(FadeIn(f_desc))
        self.wait(2)

        # --- hélice y sombra ---
        def p():
            return c2p(tray.pos(t.get_value()))

        def sombra():
            x, y, _ = tray.pos(t.get_value())
            return c2p([x, y, 0])

        punto_sombra = Dot3D(sombra(), radius=0.06, color=GREY_A)
        particula.add_updater(lambda m: m.move_to(p()))
        punto_sombra.add_updater(lambda m: m.move_to(sombra()))
        rastro = TracedPath(p, stroke_color=C_POS, stroke_width=3.5)
        rastro_sombra = TracedPath(sombra, stroke_color=GREY_A, stroke_width=2.5,
                                   stroke_opacity=0.8)
        self.play(FadeOut(VGroup(a_v, a_perp, a_par, punt)),
                  FadeOut(VGroup(l_v, l_perp, l_par)), FadeIn(punto_sombra),
                  FadeIn(etiq_ejes[1]))
        self.add(rastro, rastro_sombra, particula, punto_sombra)

        self.begin_ambient_camera_rotation(rate=0.12)
        self.play(t.animate.set_value(2 * T_2), run_time=11, rate_func=linear)

        circ_ref = DashedVMobject(Circle(radius=V_PERP, color=WHITE, stroke_width=2),
                                  num_dashes=40).move_to(c2p([0, 0, 0]))
        txt_sombra = VGroup(
            Text("La sombra en el plano xy es un círculo:", font_size=26),
            VGroup(MathTex(r"v_\perp", color=C_V_COMP, font_size=38),
                   Text("hace el círculo,", font_size=26),
                   MathTex(r"v_\parallel", color=C_V_COMP, font_size=38),
                   Text("lo estira", font_size=26)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).to_corner(DL, buff=0.4)
        con_fondo(txt_sombra)
        self.fijo(txt_sombra)
        self.play(Create(circ_ref), FadeIn(txt_sombra))
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # --- paso de la hélice ---
        particula.clear_updaters()
        punto_sombra.clear_updaters()
        self.play(FadeOut(f_desc), FadeOut(txt_sombra))
        self.move_camera(phi=82 * DEGREES, theta=-90 * DEGREES, zoom=0.78, run_time=3)

        t_a = T_X_R                           # ahí la hélice pasa por x = r, y = 0
        z_a = tray.pos(t_a)[2]
        z_b = tray.pos(t_a + T_2)[2]         # misma fase, una vuelta después
        xm = V_PERP + 0.45
        A, Bp = c2p([V_PERP, 0, z_a]), c2p([V_PERP, 0, z_b])
        Am, Bm = c2p([xm, 0, z_a]), c2p([xm, 0, z_b])
        marca = VGroup(
            Line(Am, Bm, color=WHITE, stroke_width=3),
            Line(Am + 0.12 * LEFT, Am + 0.12 * RIGHT, color=WHITE, stroke_width=3),
            Line(Bm + 0.12 * LEFT, Bm + 0.12 * RIGHT, color=WHITE, stroke_width=3),
            DashedLine(A, Am, color=WHITE, stroke_width=2),
            DashedLine(Bp, Bm, color=WHITE, stroke_width=2),
        )
        puntos = VGroup(Dot3D(A, radius=0.07, color=WHITE), Dot3D(Bp, radius=0.07, color=WHITE))
        l_p = MathTex("p", font_size=44).move_to(c2p([xm + 0.35, 0, (z_a + z_b) / 2]))
        self.add_fixed_orientation_mobjects(l_p)
        self.remove(l_p)
        self.play(FadeIn(puntos), Create(marca))
        self.play(FadeIn(l_p))

        f_p = VGroup(
            MathTex(r"p = \frac{2\pi m\, v_\parallel}{qB}", font_size=48),
            MathTex(r"= v_\parallel\, T", font_size=40, color=GREY_A),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        txt_p = Text("paso: avance en una vuelta", font_size=24, color=GREY_A).next_to(
            f_p, DOWN, buff=0.3, aligned_edge=LEFT)
        VGroup(f_p, txt_p).to_edge(RIGHT, buff=0.5).set_y(1.2)
        self.fijo(f_p, txt_p)
        self.play(Write(f_p[0]))
        self.play(FadeIn(f_p[1]), FadeIn(txt_p))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------------------------------------
# Escena 3: selector de velocidades
# ---------------------------------------------------------------------------
X_IN, X_PANT = -4.5, 4.6               # región con campos: entre las placas
Y_PLACA = 1.5
RENDIJA = 0.2                          # semiancho de la rendija
X_INICIO, X_FIN = -6.6, 6.6
B3, E3 = 0.25, 0.5
V_D = E3 / B3                          # velocidad seleccionada = 2
VELOCIDADES = {"lenta": 1.6, "justa": V_D, "rapida": 2.6}
COLORES_3 = {"lenta": "#4dd0e1", "justa": C_POS, "rapida": "#ff6b8b"}


def campos_3(r):
    if X_IN <= r[0] <= X_PANT and abs(r[1]) < Y_PLACA:
        return np.array([0.0, -E3, 0.0]), np.array([0.0, 0.0, -B3])
    return CERO, CERO


def choca_3(r):
    if X_IN <= r[0] <= X_PANT and abs(r[1]) >= Y_PLACA - R_CARGA:
        return True                    # placa
    if X_PANT - R_CARGA <= r[0] <= X_PANT + 0.1 and abs(r[1]) > RENDIJA:
        return True                    # pantalla
    return r[0] >= X_FIN


def tray_selector(v):
    return integrar([X_INICIO, 0, 0], [v, 0, 0], campos_3, 20, dt=1e-3, parar=choca_3)


class SelectorVelocidades(Scene):
    def construct(self):
        titulo = Text("Selector de velocidades", font_size=42).to_edge(UP, buff=0.3)
        self.play(Write(titulo))

        # --- aparato ---
        largo = X_PANT - X_IN
        placa_sup = Rectangle(width=largo, height=0.1, fill_color=GREY_C, fill_opacity=1,
                              stroke_width=0).move_to([(X_IN + X_PANT) / 2, Y_PLACA + 0.05, 0])
        placa_inf = placa_sup.copy().move_to([(X_IN + X_PANT) / 2, -Y_PLACA - 0.05, 0])
        xs_signos = np.linspace(X_IN + 0.4, X_PANT - 0.4, 9)
        mas = VGroup(*[MathTex("+", color=C_POS, font_size=34).move_to([x, Y_PLACA + 0.35, 0])
                       for x in xs_signos])
        menos = VGroup(*[MathTex("-", color=C_NEG, font_size=34).move_to([x, -Y_PLACA - 0.35, 0])
                         for x in xs_signos])
        alto_p = Y_PLACA - RENDIJA
        pantalla = VGroup(
            Rectangle(width=0.1, height=alto_p + 0.1, fill_color=GREY_B, fill_opacity=1,
                      stroke_width=0).move_to([X_PANT + 0.05, RENDIJA + alto_p / 2 + 0.05, 0]),
            Rectangle(width=0.1, height=alto_p + 0.1, fill_color=GREY_B, fill_opacity=1,
                      stroke_width=0).move_to([X_PANT + 0.05, -RENDIJA - alto_p / 2 - 0.05, 0]),
        )
        txt_pant = Text("rendija", font_size=20, color=GREY_B).move_to(
            [X_PANT + 0.75, RENDIJA + 0.25, 0])

        xs_E = np.linspace(X_IN + 0.6, X_PANT - 0.6, 6)
        campo_E = VGroup(*[Arrow([x, Y_PLACA - 0.1, 0], [x, -Y_PLACA + 0.1, 0], buff=0,
                                 color=C_CAMPO_E, stroke_width=2.5,
                                 max_tip_length_to_length_ratio=0.07,
                                 stroke_opacity=0.6).set_opacity(0.6)
                           for x in xs_E])
        xs_B = (xs_E[:-1] + xs_E[1:]) / 2
        campo_Bg = grilla_B(xs_B, [-0.9, 0.9], opacidad=0.45)

        leyenda = VGroup(
            VGroup(Arrow(UP * 0.3, DOWN * 0.3, buff=0, color=C_CAMPO_E, stroke_width=3,
                         max_tip_length_to_length_ratio=0.3),
                   MathTex(r"\vec E", color=C_CAMPO_E, font_size=36)).arrange(RIGHT, buff=0.12),
            VGroup(otimes(radio=0.13, opacidad=0.9, grosor=2),
                   MathTex(r"\vec B", color=GREY_A, font_size=36)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([-5.7, -2.6, 0])

        self.play(FadeIn(placa_sup), FadeIn(placa_inf), FadeIn(pantalla))
        self.play(LaggedStartMap(FadeIn, mas, lag_ratio=0.05),
                  LaggedStartMap(FadeIn, menos, lag_ratio=0.05), FadeIn(txt_pant))
        self.play(LaggedStartMap(GrowArrow, campo_E, lag_ratio=0.1), FadeIn(leyenda[0]))
        self.play(FadeIn(campo_Bg), FadeIn(leyenda[1]))
        self.wait(0.5)

        # --- una partícula con v = E/B: diagrama de fuerzas ---
        tray = tray_selector(V_D)
        t = ValueTracker(0)
        q = carga(+1).move_to(tray.pos(0))
        q.add_updater(lambda m: m.move_to(tray.pos(t.get_value())))
        ras = TracedPath(q.get_center, stroke_color=COLORES_3["justa"], stroke_width=3)
        self.add(ras, q)
        self.play(FadeIn(q, scale=0.5))

        x_alto = -1.0
        self.play(t.animate.set_value(tray.tiempo_en_x(x_alto)), run_time=3, rate_func=linear)
        P = q.get_center()
        FUERZA = 1.0 / (E3)                   # escala: qE se dibuja de largo 1
        f_E = flecha(P, FUERZA * E3 * DOWN, C_FE)
        f_B = flecha(P, FUERZA * V_D * B3 * UP, C_FB)
        f_v = flecha(P, 0.45 * V_D * RIGHT, C_V)
        l_E = con_fondo(MathTex(r"q\vec E", color=C_FE, font_size=36).next_to(
            f_E.get_end(), RIGHT, buff=0.15), buff=0.05)
        l_B = con_fondo(MathTex(r"q\,\vec v\times\vec B", color=C_FB, font_size=36).next_to(
            f_B.get_end(), RIGHT, buff=0.15), buff=0.05)
        l_v = con_fondo(MathTex(r"\vec v", color=C_V, font_size=36).next_to(
            f_v.get_end(), UR, buff=0.05), buff=0.05)
        self.play(GrowArrow(f_v), FadeIn(l_v))
        self.play(GrowArrow(f_E), FadeIn(l_E))
        self.play(GrowArrow(f_B), FadeIn(l_B))
        self.wait(1)

        ec = MathTex("qE", "=", "qvB", r"\;\;\Rightarrow\;\;", r"v = \frac{E}{B}", font_size=50)
        ec[0].set_color(C_FE)
        ec[2].set_color(C_FB)
        ec.move_to([-0.6, -2.65, 0])
        self.play(Write(ec[:3]))
        self.wait(0.5)
        self.play(Write(ec[3:]))
        self.wait(1.5)

        self.play(FadeOut(VGroup(f_E, f_B, f_v, l_E, l_B, l_v)))
        t_rend = tray.tiempo_en_x(X_PANT)
        self.play(t.animate.set_value(t_rend), run_time=2.5, rate_func=linear)
        self.play(Flash(q.get_center(), color=COLORES_3["justa"], line_length=0.25,
                        flash_radius=0.3),
                  t.animate.set_value(tray.t_fin), run_time=1.2, rate_func=linear)
        self.wait(0.5)
        q.clear_updaters()
        self.play(FadeOut(q), FadeOut(ras), FadeOut(txt_pant))

        # --- tres velocidades a la vez ---
        trays = {k: tray_selector(v) for k, v in VELOCIDADES.items()}
        t3 = ValueTracker(0)
        cargas, rastros = VGroup(), VGroup()
        for k, tr in trays.items():
            qq = carga(+1).move_to(tr.pos(0))
            qq[0].set_fill(COLORES_3[k])
            qq.add_updater(lambda m, tr=tr: m.move_to(tr.pos(t3.get_value())))
            rr = TracedPath(qq.get_center, stroke_color=COLORES_3[k], stroke_width=3)
            cargas.add(qq)
            rastros.add(rr)
        self.add(rastros, cargas)
        self.play(FadeIn(cargas))

        t_paso = trays["justa"].tiempo_en_x(X_PANT)
        t_total = max(tr.t_fin for tr in trays.values())
        self.play(t3.animate.set_value(t_paso), run_time=t_paso * 0.9, rate_func=linear)
        self.play(Flash(trays["justa"].pos(t_paso), color=COLORES_3["justa"],
                        line_length=0.3, flash_radius=0.35),
                  t3.animate.set_value(t_total), run_time=(t_total - t_paso) * 0.9,
                  rate_func=linear)

        etiquetas = VGroup()
        for k, txt in (("rapida", r"v > E/B"), ("justa", r"v = E/B"), ("lenta", r"v < E/B")):
            fin = trays[k].pos(trays[k].t_fin)
            e = MathTex(txt, color=COLORES_3[k], font_size=34)
            if k == "justa":
                e.next_to(fin, DOWN, buff=0.25).shift(0.4 * LEFT)
            else:
                e.next_to(pantalla, RIGHT, buff=0.25).set_y(fin[1])
            etiquetas.add(e)
        self.play(LaggedStartMap(FadeIn, etiquetas, lag_ratio=0.3))
        self.wait(1.5)

        caja = SurroundingRectangle(ec[4], color=YELLOW, buff=0.15)
        txt_fin = Text("solo pasa una velocidad", font_size=30).next_to(caja, RIGHT, buff=0.4)
        self.play(Create(caja), FadeIn(txt_fin, shift=0.2 * LEFT))
        self.wait(3)
        for m in cargas:
            m.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)))


# ---------------------------------------------------------------------------
# Verificaciones (python particula_campo_B.py)
# ---------------------------------------------------------------------------
def verificar():
    # sentido de la fuerza magnética: +x × (−z) = +y
    assert np.allclose(cruz([1, 0, 0], [0, 0, -1]), [0, 1, 0])

    # escena 1: F ⊥ v, F apunta al centro, giro antihorario, radio mv/(qB)
    tr = tray_circulo(V_1, centro=CENTRO_1)
    for t in np.linspace(0, 2 * T_1, 50):
        p, v = tr.pos(t), tr.vel(t)
        F = cruz(v, B1)
        assert abs(F @ v) < 1e-9 * (1 + F @ F)
        cos = unit(F) @ unit(CENTRO_1 - p)
        assert cos > 0.999, (t, cos)
        assert abs(np.linalg.norm(p - CENTRO_1) - V_1) < 5e-3
    giro = cruz(tr.pos(0) - CENTRO_1, tr.vel(0))[2]
    assert giro > 0, "debería girar antihorario"
    print("escena 1: F ⊥ v, apunta al centro, antihorario, r = %.4f" %
          np.linalg.norm(tr.pos(1.0) - CENTRO_1))

    # dos partículas: vuelven juntas al punto de partida
    tr1, tr2 = tray_circulo(1.0, inicio=INICIO_PAR), tray_circulo(2.0, inicio=INICIO_PAR)
    d1 = np.linalg.norm(tr1.pos(T_1) - INICIO_PAR)
    d2 = np.linalg.norm(tr2.pos(T_1) - INICIO_PAR)
    print("par v / 2v: distancia al inicio tras T = %.2e / %.2e" % (d1, d2))
    assert d1 < 1e-2 and d2 < 1e-2

    trn = tray_circulo(1.2, centro=np.array([-1.6, -0.6, 0]), q=-1.0)
    assert cruz(trn.pos(0) - np.array([-1.6, -0.6, 0]), trn.vel(0))[2] < 0

    # escena 2: la sombra es un círculo de radio v⊥ y el paso es 2π v∥
    th = tray_helice()
    rad = np.hypot(th.rs[:, 0], th.rs[:, 1])
    paso = th.pos(T_X_R + T_2)[2] - th.pos(T_X_R)[2]
    assert np.allclose(th.pos(T_X_R)[:2], [V_PERP, 0], atol=1e-2)
    print("hélice: radio sombra %.4f ± %.1e, paso %.4f (2πv∥ = %.4f)" %
          (rad.mean(), rad.std(), paso, PASO))
    assert rad.std() < 1e-3 and abs(paso - PASO) < 1e-2

    # escena 3: la justa pasa, la lenta baja, la rápida sube, ambas contra la pantalla
    for k, v in VELOCIDADES.items():
        trs = tray_selector(v)
        fin = trs.pos(trs.t_fin)
        print("selector %-6s v=%.2f  fin=(%.2f, %.2f)  ymax=%.2f ymin=%.2f" %
              (k, v, fin[0], fin[1], trs.rs[:, 1].max(), trs.rs[:, 1].min()))
        if k == "justa":
            assert fin[0] >= X_FIN and abs(trs.rs[:, 1]).max() < 1e-6
        else:
            assert abs(fin[0] - X_PANT) < 0.2, "debería chocar con la pantalla"
            assert (fin[1] < -RENDIJA) if k == "lenta" else (fin[1] > RENDIJA)
            assert abs(trs.rs[:, 1]).max() < Y_PLACA - R_CARGA
    print("todo ok")


if __name__ == "__main__":
    verificar()
