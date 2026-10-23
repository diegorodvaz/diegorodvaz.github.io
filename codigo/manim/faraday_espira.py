"""
Ley de Faraday: una espira que gira en un campo B uniforme (el generador).

    manim -pql faraday_espira.py FaradayEspira

Unidades adimensionales: en los gráficos B = A = 1 y ω₀ = 1, así que
Φ(t) = cos(ωt) y ε(t) = ω·sin(ωt). Un único ValueTracker de tiempo `t`
mueve la espira, decide qué líneas de campo la atraviesan (intersección
línea-plano en cada frame) y traza ambos gráficos.
`python faraday_espira.py` corre las verificaciones numéricas.
"""
from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
FONDO = "#0b0f1a"
C_B = "#6b8db5"           # campo B (tenue)
C_FLUJO = "#4fd1c5"       # líneas que atraviesan y Φ
C_FEM = "#ff8c42"         # ε y corriente
C_ESPIRA = "#5b7cfa"      # superficie de la espira
C_BORDE = GREY_A
C_NORMAL = YELLOW_C       # vector normal
C_ANGULO = YELLOW_B
C_TEXTO = GREY_A

config.background_color = FONDO

# ---------------------------------------------------------------------------
# Física (adimensional)
# ---------------------------------------------------------------------------
B0, A0, OMEGA0 = 1.0, 1.0, 1.0
T1, T2 = 7 * PI, 9 * PI    # tramo con ω duplicada (empieza y termina con ε = 0)
T_GRAF = T2                # los gráficos llegan hasta acá
SEG_POR_T = 2.0 / PI       # segundos reales por unidad de t (1 vuelta = 4 s a ω₀)


def theta_de(t):
    if t <= T1:
        return OMEGA0 * t
    if t <= T2:
        return OMEGA0 * T1 + 2 * OMEGA0 * (t - T1)
    return OMEGA0 * T1 + 2 * OMEGA0 * (T2 - T1) + OMEGA0 * (t - T2)


def omega_de(t):
    return 2 * OMEGA0 if T1 < t <= T2 else OMEGA0


def flujo(t):
    return B0 * A0 * np.cos(theta_de(t))


def fem(t):
    # ε = −dΦ/dt = B·A·ω·sin(θ)
    return B0 * A0 * omega_de(t) * np.sin(theta_de(t))


# ---------------------------------------------------------------------------
# Geometría 3D: espira, líneas de campo, cámara
# ---------------------------------------------------------------------------
ESP_W, ESP_H = 2.4, 1.8                      # ancho (horizontal) y alto (eje z)
L_CAMPO = 2.6                                # las líneas van de x = −L a x = +L
YS_CAMPO = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
ZS_CAMPO = [-1.25, -0.42, 0.42, 1.25]
X_SOMBRA = -2.45                             # plano ⟂ B donde se proyecta la espira

PHI_CAM, THETA_CAM, ZOOM = 66 * DEGREES, -40 * DEGREES, 0.9
CENTRO_3D = np.array([-3.45, -0.25])         # dónde cae el origen en pantalla

DIR_CAM = np.array([np.sin(PHI_CAM) * np.cos(THETA_CAM),
                    np.sin(PHI_CAM) * np.sin(THETA_CAM),
                    np.cos(PHI_CAM)])
DER_CAM = np.array([-np.sin(THETA_CAM), np.cos(THETA_CAM), 0])
ARR_CAM = np.cross(DIR_CAM, DER_CAM)
# La física se arma con la espira en el origen y se dibuja corrida a O3, que
# la cámara proyecta en CENTRO_3D. (Mover frame_center no sirve: también corre
# los objetos fijos en pantalla.)
O3 = (CENTRO_3D[0] * DER_CAM + CENTRO_3D[1] * ARR_CAM) / ZOOM


def base_espira(th):
    """Normal n̂ y versores del plano de la espira (e_u horizontal, e_v = z)."""
    n = np.array([np.cos(th), np.sin(th), 0.0])
    eu = np.array([-np.sin(th), np.cos(th), 0.0])
    return n, eu, OUT


def esquinas_espira(th):
    _, eu, ev = base_espira(th)
    w, h = ESP_W / 2, ESP_H / 2
    return [-w * eu - h * ev, w * eu - h * ev, w * eu + h * ev, -w * eu + h * ev]


def mundo(p):
    return np.asarray(p, dtype=float) + O3


def esquinas_mundo(th):
    return [mundo(c) for c in esquinas_espira(th)]


def lineas_campo():
    """Cada línea: (punto inicial, dirección unitaria, largo)."""
    return [(np.array([-L_CAMPO, y, z]), RIGHT.astype(float), 2 * L_CAMPO)
            for z in ZS_CAMPO for y in YS_CAMPO]


def perforacion(p0, d, largo, th):
    """Intersección exacta de la línea p0 + s·d con el plano de la espira:
    el punto si cae dentro del rectángulo, None si no."""
    n, eu, ev = base_espira(th)
    den = d @ n
    if abs(den) < 1e-9:                      # línea paralela al plano
        return None
    s = -(p0 @ n) / den                      # el plano pasa por el origen
    if not 0 <= s <= largo:
        return None
    p = p0 + s * d
    if abs(p @ eu) < ESP_W / 2 and abs(p @ ev) < ESP_H / 2:
        return p
    return None


def atraviesa(p0, d, largo, th):
    return perforacion(p0, d, largo, th) is not None


def contar_atraviesan(th):
    return sum(atraviesa(p0, d, l, th) for p0, d, l in lineas_campo())


def angulo_con_signo(th):
    return (th + PI) % TAU - PI


def cartel(mob):
    """Orienta un mobject plano para que mire a la cámara (fija)."""
    return mob.rotate(PHI_CAM, RIGHT, about_point=ORIGIN).rotate(
        THETA_CAM + PI / 2, OUT, about_point=ORIGIN)


def punta(pos, direccion, largo=0.22, color=WHITE, opacidad=1.0):
    """Punta de flecha triangular siempre de cara a la cámara."""
    d = direccion / np.linalg.norm(direccion)
    p = np.cross(d, DIR_CAM)
    p = p / (np.linalg.norm(p) + 1e-9)
    base = pos - largo * d
    return Polygon(pos, base + 0.45 * largo * p, base - 0.45 * largo * p,
                   stroke_width=0, fill_color=color, fill_opacity=opacidad)


def flecha_3d(ini, fin, color, grosor=4, largo_punta=0.22, opacidad=1.0):
    d = fin - ini
    u = d / np.linalg.norm(d)
    return VGroup(
        Line(ini, fin - 0.8 * largo_punta * u, stroke_color=color,
             stroke_width=grosor, stroke_opacity=opacidad),
        punta(fin, d, largo_punta, color, opacidad),
    )


def formula(*partes, font_size=36):
    """MathTex coloreado: partes = (tex, color) o tex suelto."""
    texs = [p[0] if isinstance(p, tuple) else p for p in partes]
    m = MathTex(*texs, font_size=font_size, color=C_TEXTO)
    for sub, p in zip(m, partes):
        if isinstance(p, tuple):
            sub.set_color(p[1])
    return m


def texto(s, font_size=26, color=C_TEXTO, **kw):
    return Text(s, font_size=font_size, color=color, **kw)


# ---------------------------------------------------------------------------
# Escena
# ---------------------------------------------------------------------------
class FaradayEspira(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=PHI_CAM, theta=THETA_CAM, zoom=ZOOM)
        self.t = ValueTracker(-TAU)
        self.ilumina = ValueTracker(0)       # 0: todas tenues, 1: resalta las que atraviesan
        self.corriente = ValueTracker(0)     # visibilidad de las flechas de corriente

        self.acto_titulo()
        self.acto_campo()
        self.acto_espira()
        self.acto_flujo()
        self.acto_graficos()
        self.acto_desfase()
        self.acto_corriente()
        self.acto_mas_rapido()
        self.acto_cierre()

    # ----- utilidades -----------------------------------------------------
    def th(self):
        return theta_de(self.t.get_value())

    def fijo(self, *mobs):
        # solo en la cámara: ThreeDScene.add_fixed_in_frame_mobjects también los
        # agrega a la escena y aparecerían antes de su animación de entrada
        self.camera.add_fixed_in_frame_mobjects(*mobs)
        return mobs[0] if len(mobs) == 1 else mobs

    def fijo_redibujado(self, func):
        """always_redraw para mobjects fijos en pantalla: al redibujar pueden
        aparecer submobjects nuevos, que también hay que fijar."""
        m = func()

        def actualizar(mob):
            mob.become(func())
            self.camera.fixed_in_frame_mobjects.update(mob.get_family())

        m.add_updater(actualizar)
        self.fijo(m)
        return m

    def girar(self, t_fin, rate_func=linear, extra=(), run_time=None):
        dt = abs(t_fin - self.t.get_value())
        self.play(self.t.animate.set_value(t_fin), *extra,
                  run_time=run_time or dt * SEG_POR_T, rate_func=rate_func)

    def leyenda(self, *lineas, color=C_TEXTO):
        g = VGroup(*[texto(s, color=color) for s in lineas]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12)
        g.to_corner(DL, buff=0.4)
        return self.fijo(g)

    # ----- 1. título ------------------------------------------------------
    def acto_titulo(self):
        titulo = texto("Faraday: el generador en una escena", font_size=48,
                       color=WHITE)
        sub = texto("una espira que gira dentro de un campo magnético uniforme",
                    font_size=26, color=GREY_B).next_to(titulo, DOWN, buff=0.35)
        self.fijo(titulo, sub)
        self.play(FadeIn(titulo, shift=0.2 * UP), FadeIn(sub), run_time=1.2)
        self.wait(1.4)
        self.play(FadeOut(titulo), FadeOut(sub), run_time=0.8)

    # ----- 2. el campo ----------------------------------------------------
    def acto_campo(self):
        datos = lineas_campo()
        self.lineas = VGroup(*[
            flecha_3d(mundo(p0), mundo(p0 + l * d), C_B, grosor=2, largo_punta=0.2,
                      opacidad=0.35)
            for p0, d, l in datos])

        def actualizar_lineas(grupo):
            th, k = self.th(), self.ilumina.get_value()
            for linea, (p0, d, l) in zip(grupo, datos):
                if k > 0 and atraviesa(p0, d, l, th):
                    color = interpolate_color(ManimColor(C_B), ManimColor(C_FLUJO), k)
                    op, gr = 0.3 + 0.7 * k, 2 + 2 * k
                else:
                    color, op, gr = C_B, 0.3, 2
                linea[0].set_stroke(color, gr, op)
                linea[1].set_fill(color, op)

        etiqueta_b = cartel(MathTex(r"\vec B", font_size=44, color=C_B))
        etiqueta_b.move_to(mundo([L_CAMPO + 0.25, YS_CAMPO[-1] + 0.15, ZS_CAMPO[-1] + 0.35]))
        self.etiqueta_b = etiqueta_b

        nota = self.leyenda("campo magnético uniforme: líneas paralelas", color=C_B)
        self.play(LaggedStart(*[Create(m) for m in self.lineas], lag_ratio=0.03),
                  run_time=2.2)
        # Create agregó cada línea suelta: el grupo tiene que estar en la escena
        # para que corra su updater
        self.add(self.lineas)
        self.lineas.add_updater(actualizar_lineas)

        def perforaciones():
            k, th = self.ilumina.get_value(), self.th()
            puntos = [perforacion(p0, d, l, th) for p0, d, l in datos]
            return VGroup(*[cartel(Dot(radius=0.055, color=WHITE, fill_opacity=k))
                            .move_to(mundo(p)) for p in puntos
                            if p is not None and k > 0])

        self.perforaciones = always_redraw(perforaciones)
        self.play(Write(etiqueta_b), FadeIn(nota), run_time=1)
        self.wait(0.8)
        self.play(FadeOut(nota), run_time=0.5)

    # ----- 3. la espira ---------------------------------------------------
    def acto_espira(self):
        superficie = Polygon(*esquinas_mundo(self.th()), stroke_width=0,
                             fill_color=C_ESPIRA, fill_opacity=0.28)
        borde = Polygon(*esquinas_mundo(self.th()), stroke_color=C_BORDE,
                        stroke_width=4.5)

        def girar_poligono(p):
            esq = esquinas_mundo(self.th())
            p.set_points_as_corners([*esq, esq[0]])

        superficie.add_updater(girar_poligono)
        borde.add_updater(girar_poligono)
        self.add(superficie)
        self.bring_to_back(superficie)       # las líneas se ven por encima del área
        self.superficie, self.borde = superficie, borde

        eje = DashedLine(mundo(1.4 * IN), mundo(1.4 * OUT), stroke_color=GREY_B,
                         stroke_width=2, dash_length=0.08)

        normal = always_redraw(lambda: flecha_3d(
            O3, mundo(1.45 * base_espira(self.th())[0]), C_NORMAL, grosor=4))
        etiqueta_n = cartel(MathTex(r"\hat n", font_size=38, color=C_NORMAL))
        etiqueta_n.add_updater(lambda m: m.move_to(
            mundo(1.75 * base_espira(self.th())[0] + 0.15 * OUT)))

        ref_b = flecha_3d(O3, mundo(1.1 * RIGHT), C_B, grosor=3, largo_punta=0.18)

        def arco():
            a = angulo_con_signo(self.th())
            if abs(a) < 0.06:
                return VMobject()
            return Arc(radius=0.75, start_angle=0, angle=a, arc_center=O3, stroke_color=C_ANGULO,
                       stroke_width=3.5)

        arco_th = always_redraw(arco)
        etiqueta_th = cartel(MathTex(r"\theta", font_size=36, color=C_ANGULO))

        def mover_th(m):
            a = angulo_con_signo(self.th())
            m.move_to(mundo(1.02 * np.array([np.cos(a / 2), np.sin(a / 2), 0]) + 0.12 * OUT))
            m.set_opacity(1 if abs(a) > 0.25 else 0)

        etiqueta_th.add_updater(mover_th)
        self.normal, self.etiqueta_n = normal, etiqueta_n
        self.arco_th, self.etiqueta_th, self.ref_b = arco_th, etiqueta_th, ref_b

        nota = self.leyenda("espira de área A: gira alrededor del eje z,",
                            "perpendicular a B")
        self.play(Create(borde), FadeIn(superficie), Create(eje), FadeIn(nota),
                  run_time=1.5)
        self.play(GrowFromPoint(normal, O3), FadeIn(etiqueta_n), FadeIn(ref_b),
                  run_time=1)
        self.add(arco_th, etiqueta_th)
        self.girar(-TAU + PI / 4, rate_func=smooth, run_time=1.6)
        self.play(FadeOut(nota), run_time=0.5)
        self.eje = eje

    # ----- 4. flujo = líneas que atraviesan ----------------------------------
    def acto_flujo(self):
        sombra = Polygon(*self._esquinas_sombra(), stroke_color=C_FLUJO,
                         stroke_width=1.5, fill_color=C_FLUJO, fill_opacity=0.22)
        sombra.add_updater(lambda p: p.set_points_as_corners(
            [*self._esquinas_sombra(), self._esquinas_sombra()[0]]))
        marco = Polygon(*[mundo([X_SOMBRA, y, z]) for y, z in
                          [(-1.8, -1.5), (1.8, -1.5), (1.8, 1.5), (-1.8, 1.5)]],
                        stroke_color=GREY_B, stroke_width=1.2, stroke_opacity=0.5)
        etiqueta_sombra = cartel(MathTex(r"A\cos\theta", font_size=30, color=C_FLUJO))
        etiqueta_sombra.move_to(mundo([X_SOMBRA, 0, -1.5 - 0.3]))

        self.f_flujo = formula((r"\Phi", C_FLUJO), "=", (r"\vec B", C_B), r"\cdot",
                               (r"\vec A", C_ESPIRA), "=", (r"B", C_B), r"\,",
                               (r"A", C_ESPIRA), r"\cos", (r"\theta", C_ANGULO))
        self.f_flujo.to_corner(UL, buff=0.4)
        self.fijo(self.f_flujo)

        rotulo = texto("líneas que atraviesan:", font_size=24, color=C_FLUJO)
        rotulo.next_to(self.f_flujo, DOWN, aligned_edge=LEFT, buff=0.3)
        contador = Integer(contar_atraviesan(self.th()), font_size=32, color=C_FLUJO)

        def actualizar_contador(m):
            m.set_value(contar_atraviesan(self.th()))
            m.next_to(rotulo, RIGHT, buff=0.15)
            self.camera.fixed_in_frame_mobjects.update(m.get_family())

        actualizar_contador(contador)
        contador.add_updater(actualizar_contador)
        self.fijo(rotulo, contador)
        self.contador = VGroup(rotulo, contador)

        nota = self.leyenda("flujo Φ: cuántas líneas de campo atraviesan la espira",
                            color=C_FLUJO)
        self.add(self.perforaciones)
        self.play(self.ilumina.animate.set_value(1), FadeIn(nota), run_time=1.2)
        self.play(Write(self.f_flujo), FadeIn(self.contador), run_time=1.5)
        self.play(FadeIn(marco), FadeIn(sombra), FadeIn(etiqueta_sombra), run_time=1)
        self.bring_to_back(marco, sombra)
        self.wait(0.6)

        self.play(FadeOut(nota), run_time=0.3)
        nota = self.leyenda("de canto: ninguna línea la atraviesa  →  Φ = 0")
        self.girar(-TAU + PI / 2, rate_func=smooth, run_time=1.4, extra=[FadeIn(nota)])
        self.wait(1)
        self.play(FadeOut(nota), run_time=0.3)
        nota = self.leyenda("de frente: atraviesan todas las que puede  →  Φ máximo")
        self.girar(0, rate_func=smooth, run_time=3.2, extra=[FadeIn(nota)])
        self.wait(1)
        self.play(FadeOut(nota), run_time=0.4)
        self.sombra = VGroup(marco, sombra, etiqueta_sombra)

    def _esquinas_sombra(self):
        # proyección de la espira a lo largo de B sobre el plano x = X_SOMBRA
        return [mundo([X_SOMBRA, p[1], p[2]]) for p in esquinas_espira(self.th())]

    # ----- 5. el giro y los gráficos -------------------------------------
    def acto_graficos(self):
        estilo = dict(stroke_color=GREY_B, stroke_width=2)
        self.ax_phi = Axes(x_range=[0, T_GRAF, PI], y_range=[-1.3, 1.3, 1],
                           x_length=5.6, y_length=2.3, tips=False,
                           axis_config=estilo).move_to([3.95, 1.45, 0])
        self.ax_eps = Axes(x_range=[0, T_GRAF, PI], y_range=[-2.4, 2.4, 1],
                           x_length=5.6, y_length=2.75, tips=False,
                           axis_config=estilo).move_to([3.95, -1.95, 0])
        ax_phi, ax_eps = self.ax_phi, self.ax_eps

        lab_phi = MathTex(r"\Phi(t)", font_size=34, color=C_FLUJO).next_to(
            ax_phi.c2p(0, 1.3), UP, buff=0.12)
        lab_eps = MathTex(r"\varepsilon(t)", font_size=34, color=C_FEM).next_to(
            ax_eps.c2p(0, 2.4), UP, buff=0.12)
        lab_t1 = MathTex("t", font_size=30, color=GREY_B).next_to(
            ax_phi.x_axis.get_end(), RIGHT, buff=0.1)
        lab_t2 = MathTex("t", font_size=30, color=GREY_B).next_to(
            ax_eps.x_axis.get_end(), RIGHT, buff=0.1)
        self.ejes = VGroup(ax_phi, ax_eps, lab_phi, lab_eps, lab_t1, lab_t2)
        self.fijo(self.ejes)

        def tn():
            return np.clip(self.t.get_value(), 0, T_GRAF)

        def traza(ax, f, color):
            tf = tn()
            if tf < 0.02:
                return VMobject()
            return ax.plot(f, x_range=[0, tf, 0.03], stroke_color=color,
                           stroke_width=3.5, use_smoothing=False)

        self.traza_phi = self.fijo_redibujado(lambda: traza(ax_phi, flujo, C_FLUJO))
        self.traza_eps = self.fijo_redibujado(lambda: traza(ax_eps, fem, C_FEM))
        self.punto_phi = self.fijo_redibujado(lambda: Dot(
            ax_phi.c2p(tn(), flujo(tn())), radius=0.07, color=C_FLUJO))
        self.punto_eps = self.fijo_redibujado(lambda: Dot(
            ax_eps.c2p(tn(), fem(tn())), radius=0.07, color=C_FEM))
        self.cursor = self.fijo_redibujado(lambda: DashedLine(
            ax_phi.c2p(tn(), 1.3), ax_eps.c2p(tn(), -2.4), stroke_color=GREY_B,
            stroke_width=1.8, dash_length=0.08))

        self.f_omega = formula(r"\omega", "=", r"\omega_0", font_size=34)
        self.f_omega.next_to(self.f_flujo, RIGHT, buff=0.9)
        self.fijo(self.f_omega)

        self.play(FadeOut(self.contador), Create(ax_phi), Create(ax_eps),
                  FadeIn(VGroup(lab_phi, lab_eps, lab_t1, lab_t2)),
                  FadeIn(self.punto_phi), FadeIn(self.punto_eps), FadeIn(self.cursor),
                  FadeIn(self.f_omega), run_time=1.5)
        self.add(self.traza_phi, self.traza_eps)
        self.girar(4 * PI)
        self.girar(4.5 * PI, rate_func=rate_functions.ease_out_sine, run_time=2)

    # ----- 6. el desfase ---------------------------------------------------
    def acto_desfase(self):
        tp = self.t.get_value()                # espira de canto: Φ = 0, ε = máx
        ax = self.ax_phi
        pendiente = -omega_de(tp) * np.sin(theta_de(tp))      # dΦ/dt
        dt = 1.4
        tangente = Line(ax.c2p(tp - dt, flujo(tp) - pendiente * dt),
                        ax.c2p(tp + dt, flujo(tp) + pendiente * dt),
                        stroke_color=WHITE, stroke_width=3)
        self.fijo(tangente)

        f_eps = formula((r"\varepsilon", C_FEM), "=", "-", r"\frac{d\Phi}{dt}")
        f_eps[3][1].set_color(C_FLUJO)        # la Φ dentro de la derivada
        f_eps.next_to(self.f_flujo, DOWN, aligned_edge=LEFT, buff=0.3)
        self.fijo(f_eps)
        self.f_eps = f_eps

        nota = self.leyenda("la fem no depende de cuánto flujo hay,",
                            "sino de qué tan rápido cambia")
        nota[1].set_color(C_FEM)

        self.play(Indicate(self.punto_phi, scale_factor=1.8, color=WHITE),
                  Indicate(self.punto_eps, scale_factor=1.8, color=WHITE),
                  Indicate(self.borde, scale_factor=1.12, color=C_FEM),
                  run_time=1.4)
        self.play(FadeIn(nota[0]), run_time=0.8)
        self.play(FadeIn(nota[1]), run_time=0.8)
        self.play(Create(tangente), Write(f_eps), run_time=1.5)
        self.play(Indicate(tangente, color=C_FEM), Indicate(self.punto_eps,
                  scale_factor=1.6, color=WHITE), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(nota), FadeOut(tangente), run_time=0.6)

    # ----- 7. la corriente -------------------------------------------------
    def acto_corriente(self):
        def flechas():
            k = self.corriente.get_value()
            th, tt = self.th(), self.t.get_value()
            e = fem(tt) / omega_de(tt)            # sin θ: el signo marca el sentido
            op = k * min(1.0, 3 * abs(e))
            if op < 0.02:
                return VGroup()
            n, eu, ev = base_espira(th)
            w, h, s = ESP_W / 2, ESP_H / 2, np.sign(e)
            # circulación positiva = antihoraria vista desde n̂ (regla de la mano derecha)
            tramos = [(-h * ev, eu), (w * eu, ev), (h * ev, -eu), (-w * eu, -ev)]
            return VGroup(*[punta(mundo(c + s * 0.15 * d), s * d, largo=0.3, color=C_FEM,
                                  opacidad=op) for c, d in tramos])

        self.flechas_i = always_redraw(flechas)
        self.borde.add_updater(lambda b: b.set_stroke(interpolate_color(
            ManimColor(C_BORDE), ManimColor(C_FEM),
            self.corriente.get_value() * abs(fem(self.t.get_value())) /
            (B0 * A0 * omega_de(self.t.get_value())))))
        self.add(self.flechas_i)

        nota = self.leyenda("corriente alterna: cambia de sentido cada media vuelta",
                            color=C_FEM)
        self.play(self.corriente.animate.set_value(1), FadeIn(nota), run_time=0.8)
        self.girar(T1, run_time=6.5)
        self.play(FadeOut(nota), run_time=0.4)

    # ----- 8. más rápido ---------------------------------------------------
    def acto_mas_rapido(self):
        f_omega2 = formula(r"\omega", "=", r"2\omega_0", font_size=34).move_to(
            self.f_omega, aligned_edge=LEFT)
        self.fijo(f_omega2)
        nota = self.leyenda("duplicamos ω: la fem crece y oscila más rápido")
        self.play(ReplacementTransform(self.f_omega, f_omega2), FadeIn(nota),
                  run_time=0.8)
        self.f_omega = f_omega2
        self.girar(T2)

        ax = self.ax_eps
        guias = VGroup(*[DashedLine(ax.c2p(0, y), ax.c2p(T_GRAF, y),
                                    stroke_color=C_FEM, stroke_width=1.5,
                                    stroke_opacity=0.6, dash_length=0.07)
                         for y in (1, 2)])
        rot = VGroup(
            MathTex(r"\varepsilon_0", font_size=28, color=C_FEM).next_to(
                ax.c2p(0, 1), LEFT, buff=0.1),
            MathTex(r"2\varepsilon_0", font_size=28, color=C_FEM).next_to(
                ax.c2p(0, 2), LEFT, buff=0.1))
        f_amp = formula((r"\varepsilon_0", C_FEM), "=", (r"B", C_B), r"\,",
                        (r"A", C_ESPIRA), r"\,", r"\omega")
        f_amp.next_to(self.f_eps, DOWN, aligned_edge=LEFT, buff=0.3)
        self.fijo(guias, rot, f_amp)

        self.play(FadeOut(nota), run_time=0.3)
        nota = self.leyenda("amplitud y frecuencia se duplican a la vez")
        self.play(Create(guias), FadeIn(rot), FadeIn(nota), Write(f_amp), run_time=1.5)
        self.wait(2.2)
        self.play(FadeOut(nota), run_time=0.4)
        self.guias = VGroup(guias, rot, f_amp)

    # ----- 9. cierre -------------------------------------------------------
    def acto_cierre(self):
        f = formula((r"\varepsilon(t)", C_FEM), "=", (r"B", C_B), r"\cdot",
                    (r"A", C_ESPIRA), r"\cdot", r"\omega", r"\cdot",
                    r"\sin(\omega t)", font_size=46)
        f.move_to([3.6, 0.4, 0])
        caja = SurroundingRectangle(f, color=C_FEM, buff=0.3, corner_radius=0.1)
        sub = texto("generador: la fem alterna sale del giro", font_size=26,
                    color=GREY_B).next_to(caja, DOWN, buff=0.4)
        self.fijo(f, caja, sub)

        fuera = [self.ejes, self.traza_phi, self.traza_eps, self.punto_phi,
                 self.punto_eps, self.cursor, self.guias, self.f_omega]
        for m in fuera:
            m.clear_updaters()
        self.play(*[FadeOut(m) for m in fuera], run_time=1,
                  rate_func=smooth)
        t0 = self.t.get_value()
        # la espira sigue girando de fondo mientras aparece y queda la fórmula
        self.play(self.t.animate(run_time=4 * PI * SEG_POR_T, rate_func=linear)
                  .set_value(t0 + 4 * PI),
                  Write(f, run_time=2.5), Create(caja, run_time=2.5),
                  FadeIn(sub, run_time=2.5))


# ---------------------------------------------------------------------------
# Verificaciones: python faraday_espira.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    total = len(lineas_campo())
    for grados in (0, 45, 90, 135, 180, 270):
        th = np.radians(grados)
        print(f"θ = {grados:3d}°  líneas que atraviesan: {contar_atraviesan(th):2d}/{total}")
    assert contar_atraviesan(0) == contar_atraviesan(PI) > 0
    assert contar_atraviesan(PI / 2) == contar_atraviesan(3 * PI / 2) == 0
    # ε = −dΦ/dt numéricamente
    for t in np.linspace(0.3, T2 - 0.3, 40):
        if abs(t - T1) < 0.01:
            continue
        d = (flujo(t + 1e-5) - flujo(t - 1e-5)) / 2e-5
        assert abs(fem(t) + d) < 1e-4, t
    # en el tramo rápido la amplitud es 2·B·A·ω₀
    ts = np.linspace(T1 + 0.01, T2, 2000)
    assert abs(max(fem(t) for t in ts) - 2 * B0 * A0 * OMEGA0) < 1e-3
    print("ok")
