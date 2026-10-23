"""
El potencial eléctrico como relieve: z = V(x, y), E = −∇V y las equipotenciales
como curvas de nivel (el gradiente de Análisis Matemático II).

    manim -pql potencial_relieve.py PotencialRelieve
    manim -pqh potencial_relieve.py PotencialRelieve

Unidades adimensionales (k = 1). El potencial se regulariza con un softening,
rᵢ = √((x−xᵢ)² + (y−yᵢ)² + ε²), y la altura dibujada se recorta suavemente a
(−Z_MAX, Z_MAX) con z = Z_MAX·tanh(V / V_SAT). Como es una función creciente de V,
las curvas de nivel y la dirección del gradiente son exactamente las de V.
∇V es analítico; los contornos salen de un marching squares sobre una grilla fina
y las líneas de campo de integrar dr/ds = −∇V/|∇V| con RK4.
`python potencial_relieve.py` corre las verificaciones numéricas.
"""
from collections import defaultdict

import cairo
from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
FONDO = "#0b0f1a"
C_POS = "#ff5a3c"         # carga positiva (montaña)
C_NEG = "#3d8bff"         # carga negativa (pozo)
C_GRAD = "#ffd23f"        # ∇V
C_E = "#c77dff"           # E = −∇V
C_EQUI = "#f4f1de"        # equipotenciales
C_LINEA = "#2ee6a6"       # líneas de campo
C_CORTE = "#fff27a"       # curva de intersección plano-relieve
C_PLANO = "#9ec5ff"       # plano de corte
C_PISO = "#111a2c"        # plano xy
C_GRILLA = "#4a5a78"
C_BOLA = "#ffb39e"
C_TEXTO = GREY_A

# alturas normalizadas (−1 … 1) → color: fríos abajo, cálidos arriba
MAPA = [(-1.00, "#1d3fa0"), (-0.55, "#2f7de1"), (-0.20, "#48b3d6"),
        (0.00, "#7f8ea3"),
        (0.20, "#e3c76a"), (0.55, "#f28a36"), (1.00, "#e8402c")]
MAPA_T = np.array([t for t, _ in MAPA])
MAPA_RGB = np.array([ManimColor(c).to_rgb() for _, c in MAPA])
LUZ = np.array([-0.45, -0.7, 1.2]) / np.linalg.norm([-0.45, -0.7, 1.2])

config.background_color = FONDO

# ---------------------------------------------------------------------------
# Física (adimensional, k = 1)
# ---------------------------------------------------------------------------
EPS = 0.3                  # softening
V_SAT = 1.5                # escala del recorte suave
Z_MAX = 1.8                # |altura dibujada| < Z_MAX
D = 1.5                    # el dipolo: +1 en (−D, 0), −1 en (D, 0)
CARGA_3 = (0.7, 1.9)       # tercera carga (+1), solo en la superposición
XMAX, YMAX = 4.0, 3.0      # dominio [−XMAX, XMAX] × [−YMAX, YMAX]

DIPOLO = [(-D, 0.0, 1.0), (D, 0.0, -1.0)]
NIVELES = [1.5, 0.9, 0.55, 0.3, -0.3, -0.55, -0.9, -1.5]
ANGULOS_LINEAS = np.radians(np.arange(0, 360, 30))
R_INICIO, R_FIN = 0.12, 0.15   # las líneas arrancan / terminan cerca de las cargas

XS_F = np.linspace(-XMAX, XMAX, 321)     # grilla fina para los contornos
YS_F = np.linspace(-YMAX, YMAX, 241)
XF, YF = np.meshgrid(XS_F, YS_F)


def potencial(x, y, cargas):
    V = np.zeros_like(np.asarray(x, dtype=float))
    for xq, yq, q in cargas:
        V = V + q / np.sqrt((x - xq) ** 2 + (y - yq) ** 2 + EPS ** 2)
    return V


def gradiente(x, y, cargas):
    """∇V analítico (con el mismo softening)."""
    gx = np.zeros_like(np.asarray(x, dtype=float))
    gy = np.zeros_like(gx)
    for xq, yq, q in cargas:
        dx, dy = x - xq, y - yq
        r3 = (dx * dx + dy * dy + EPS ** 2) ** 1.5
        gx = gx - q * dx / r3
        gy = gy - q * dy / r3
    return gx, gy


def altura(V):
    return Z_MAX * np.tanh(V / V_SAT)


def nivel_de_altura(z):
    return V_SAT * np.arctanh(np.clip(z / Z_MAX, -0.999999, 0.999999))


def color_altura(t):
    t = np.clip(t, -1, 1)
    return np.stack([np.interp(t, MAPA_T, MAPA_RGB[:, k]) for k in range(3)], axis=-1)


# ---------------------------------------------------------------------------
# Numérica: contornos (marching squares) y líneas de campo (RK4)
# ---------------------------------------------------------------------------
def contornos(F, xs, ys, c):
    """Curvas F = c sobre la grilla F[j, i] = F(xs[i], ys[j]).
    Devuelve una lista de polilíneas (N, 2); las cerradas repiten el primer punto."""
    nx = len(xs)
    A = F > c
    caso = (A[:-1, :-1].astype(int) | (A[:-1, 1:] << 1)
            | (A[1:, 1:] << 2) | (A[1:, :-1] << 3))
    NH = len(ys) * (nx - 1)          # aristas horizontales primero, después verticales

    def punto(e):
        if e < NH:
            j, i = divmod(e, nx - 1)
            f0, f1 = F[j, i], F[j, i + 1]
            t = (c - f0) / (f1 - f0)
            return xs[i] + t * (xs[i + 1] - xs[i]), ys[j]
        j, i = divmod(e - NH, nx)
        f0, f1 = F[j, i], F[j + 1, i]
        t = (c - f0) / (f1 - f0)
        return xs[i], ys[j] + t * (ys[j + 1] - ys[j])

    vecinos = defaultdict(list)
    js, is_ = np.nonzero((caso > 0) & (caso < 15))
    for j, i in zip(js, is_):
        k = caso[j, i]
        B, T = j * (nx - 1) + i, (j + 1) * (nx - 1) + i
        L, R = NH + j * nx + i, NH + j * nx + i + 1
        b0, b1, b2, b3 = k & 1, (k >> 1) & 1, (k >> 2) & 1, (k >> 3) & 1
        if k in (5, 10):             # silla: decide el valor en el centro
            arriba = 0.25 * (F[j, i] + F[j, i + 1] + F[j + 1, i + 1] + F[j + 1, i]) > c
            if (k == 5) == arriba:
                segs = [(B, R), (T, L)]
            else:
                segs = [(B, L), (T, R)]
        else:
            cruces = [e for e, corta in ((B, b0 != b1), (R, b1 != b2),
                                         (T, b3 != b2), (L, b0 != b3)) if corta]
            segs = [tuple(cruces)]
        for a, b in segs:
            vecinos[a].append(b)
            vecinos[b].append(a)

    curvas, visto = [], set()
    extremos = [n for n, v in vecinos.items() if len(v) == 1]
    for inicio in extremos + list(vecinos):
        if inicio in visto:
            continue
        cadena, previo, actual = [inicio], None, inicio
        visto.add(inicio)
        while True:
            sig = [n for n in vecinos[actual] if n != previo and n not in visto]
            if not sig:
                break
            previo, actual = actual, sig[0]
            cadena.append(actual)
            visto.add(actual)
        if len(cadena) > 2 and inicio in vecinos[actual]:
            cadena.append(inicio)
        if len(cadena) >= 2:
            curvas.append(np.array([punto(e) for e in cadena]))
    return curvas


def linea_de_campo(p0, cargas, h=0.01, n_max=4000):
    """Máximo descenso dr/ds = −∇V/|∇V| (RK4, paso h en longitud de arco).
    Corta cerca de una carga negativa o al salir del dominio."""
    def f(p):
        gx, gy = gradiente(p[0], p[1], cargas)
        n = np.hypot(gx, gy)
        return -np.array([gx, gy]) / n

    negativas = [np.array([xq, yq]) for xq, yq, q in cargas if q < 0]
    p = np.array(p0, dtype=float)
    pts = [p.copy()]
    for _ in range(n_max):
        k1 = f(p)
        k2 = f(p + h / 2 * k1)
        k3 = f(p + h / 2 * k2)
        k4 = f(p + h * k3)
        nuevo = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if abs(nuevo[0]) > XMAX or abs(nuevo[1]) > YMAX:
            # recorta el último paso justo en el borde del dominio
            d = nuevo - p
            ts = [(np.sign(d[k]) * lim - p[k]) / d[k]
                  for k, lim in ((0, XMAX), (1, YMAX)) if d[k] != 0]
            pts.append(p + min(t for t in ts if t >= 0) * d)
            break
        p = nuevo
        pts.append(p.copy())
        if any(np.linalg.norm(p - c) < R_FIN for c in negativas):
            break
    return np.array(pts)


def lineas_del_dipolo():
    xq, yq, _ = DIPOLO[0]
    return [linea_de_campo((xq + R_INICIO * np.cos(a), yq + R_INICIO * np.sin(a)), DIPOLO)
            for a in ANGULOS_LINEAS]


def punto_en_contorno(nivel, centro, angulo):
    """El punto de la equipotencial `nivel` en la dirección `angulo` vista desde `centro`."""
    curvas = contornos(potencial(XF, YF, DIPOLO), XS_F, YS_F, nivel)
    P = np.concatenate(curvas)
    d = P - np.array(centro)
    dif = np.abs((np.arctan2(d[:, 1], d[:, 0]) - angulo + PI) % TAU - PI)
    return P[np.argmin(dif)]


# ---------------------------------------------------------------------------
# Geometría en el mundo (lo que ve la cámara)
# ---------------------------------------------------------------------------
Z_CERO = 0.55                      # altura del nivel V = 0
Z_PISO = Z_CERO - Z_MAX - 0.45     # el plano xy, por debajo de todo el relieve
OPAC_SOLIDA, OPAC_VIDRIO, OPAC_TENUE = 0.95, 0.62, 0.10
ANCHO_MALLA = 0.5
ALZAR_SUP = 0.25                   # curvas sobre el relieve: un poco "adelante"
CAPA_PISO_FONDO, CAPA_PISO_GRILLA, CAPA_PISO_CURVAS = -3000, -2000, -1000

PHI_OBL, THETA_OBL, ZOOM_OBL = 62 * DEGREES, -60 * DEGREES, 0.84
ZOOM_TOP = 0.78
FOCAL = 100                        # casi ortográfica: la vista cenital no se deforma
# En las vistas cenitales del análisis el mapa va a la izquierda y las fórmulas a
# la derecha: con frame_center = c el mundo se dibuja en zoom·(x − c) y los textos
# fijos no se mueven (ver Camara.get_cairo_context).
X_MAPA = -2.55                     # dónde cae el origen del mundo en pantalla
FC_TOP = np.array([-X_MAPA / ZOOM_TOP, 0, 0])
X_PANEL = 1.2                      # borde izquierdo del panel de fórmulas


def z_mundo(V):
    return Z_CERO + altura(V)


class Camara(ThreeDCamera):
    """Pintor por profundidad con capas (`alzar`): el piso siempre primero y las
    curvas sobre el relieve un poco adelante de la cara que pisan. Las piezas con
    `sin_sombra` ya traen el color iluminado, así que no se vuelven a sombrear."""

    def get_mobjects_to_display(self, *args, **kwargs):
        mobjects = Camera.get_mobjects_to_display(self, *args, **kwargs)
        eje = self.get_rotation_matrix()[2]

        def clave(m):
            if not getattr(m, "shade_in_3d", False):
                return np.inf
            c = getattr(m, "centro", None)
            if c is None:
                c = m.get_center()
            return eje @ c + getattr(m, "alzar", 0.0)

        return sorted(mobjects, key=clave)

    def modified_rgbas(self, vmobject, rgbas):
        if getattr(vmobject, "sin_sombra", False):
            return rgbas
        return super().modified_rgbas(vmobject, rgbas)

    def get_cairo_context(self, pixel_array):
        # Camera arma (y cachea) la matriz de Cairo con el frame_center del primer
        # cuadro. Acá va siempre centrada en el origen: así frame_center solo corre
        # la proyección 3D y nunca los textos fijos, se renderice desde donde sea.
        ctx = self.get_cached_cairo_context(pixel_array)
        if ctx:
            return ctx
        pw, ph = self.pixel_width, self.pixel_height
        fw, fh = self.frame_width, self.frame_height
        superficie = cairo.ImageSurface.create_for_data(
            pixel_array.data, cairo.FORMAT_ARGB32, pw, ph)
        ctx = cairo.Context(superficie)
        ctx.scale(pw, ph)
        ctx.set_matrix(cairo.Matrix(pw / fw, 0, 0, -(ph / fh), pw / 2, ph / 2))
        self.cache_cairo_context(pixel_array, ctx)
        return ctx


class Pieza(VMobject):
    """VMobject que la cámara ordena en 3D sin sombrearlo."""

    def __init__(self, alzar=0.0, **kw):
        super().__init__(shade_in_3d=True, **kw)
        self.sin_sombra = True
        self.alzar = alzar


class Relieve(VGroup):
    """Superficie z(x, y) como caras cuadradas que se recalculan en el lugar."""

    def __init__(self, nx, ny):
        super().__init__()
        self.X, self.Y = np.meshgrid(np.linspace(-XMAX, XMAX, nx + 1),
                                     np.linspace(-YMAX, YMAX, ny + 1))
        for _ in range(nx * ny):
            cara = Pieza()
            cara.set_stroke(width=ANCHO_MALLA)
            self.add(cara)

    def poner(self, Z, T, opacidad):
        """Z: alturas en los vértices; T: altura normalizada (−1…1) para el color."""
        P = np.stack([self.X, self.Y, Z], axis=-1)
        c00, c10, c11, c01 = P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]
        esq = np.stack([c00, c10, c11, c01, c00], axis=2).reshape(-1, 5, 3)
        a, b = esq[:, :-1], esq[:, 1:]
        tercio = (b - a) / 3
        pts = np.stack([a, a + tercio, b - tercio, b], axis=2).reshape(-1, 16, 3)
        n = np.cross(c11 - c00, c01 - c10).reshape(-1, 3)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        luz = 0.45 + 0.65 * np.clip(n @ LUZ, 0, None)
        Tc = 0.25 * (T[:-1, :-1] + T[:-1, 1:] + T[1:, 1:] + T[1:, :-1]).reshape(-1)
        rgb = np.clip(color_altura(Tc) * luz[:, None], 0, 1)
        borde = rgb * 0.6
        centros = esq[:, :4].mean(axis=1)
        for k, cara in enumerate(self.submobjects):
            cara.points = pts[k]
            cara.fill_rgbas = np.array([[*rgb[k], opacidad]])
            cara.stroke_rgbas = np.array([[*borde[k], opacidad]])
            cara.centro = centros[k]


def plano_de_corte(nx, ny):
    """Plano horizontal en z = 0 con las mismas celdas que el relieve: así cada
    baldosa se ordena contra la cara que tiene justo abajo o arriba."""
    xs, ys = np.linspace(-XMAX, XMAX, nx + 1), np.linspace(-YMAX, YMAX, ny + 1)
    g = VGroup()
    for j in range(ny):
        for i in range(nx):
            b = Pieza()
            b.set_points_as_corners([[xs[i], ys[j], 0], [xs[i + 1], ys[j], 0],
                                     [xs[i + 1], ys[j + 1], 0], [xs[i], ys[j + 1], 0],
                                     [xs[i], ys[j], 0]])
            b.set_fill(C_PLANO, 0.2).set_stroke(C_PLANO, 0.4, 0.2)
            g.add(b)
    g.z = 0.0
    return g


def trozos(P, color, ancho, alzar=ALZAR_SUP, largo=0.22, opacidad=1.0):
    """Parte una polilínea 3D en trozos cortos para que el pintor los ordene bien
    contra las caras del relieve."""
    P = np.asarray(P, dtype=float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(1, int(np.ceil(s[-1] / largo)))
    idx = [0, *np.searchsorted(s, np.linspace(0, s[-1], n + 1)[1:-1]), len(P) - 1]
    g = VGroup()
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a:
            p = Pieza(alzar=alzar)
            p.set_points_as_corners(P[a:b + 1]).set_stroke(color, ancho, opacidad)
            g.add(p)
    return g


def en_piso(P2, color, ancho, capa=CAPA_PISO_CURVAS, opacidad=1.0):
    p = Pieza(alzar=capa)
    P2 = np.asarray(P2)
    p.set_points_as_corners(np.c_[P2, np.full(len(P2), Z_PISO + 0.005)])
    return p.set_stroke(color, ancho, opacidad)


def sobre_relieve(P2, cargas, dz=0.015):
    P2 = np.asarray(P2)
    return np.c_[P2, z_mundo(potencial(P2[:, 0], P2[:, 1], cargas)) + dz]


def piso():
    fondo = Pieza(alzar=CAPA_PISO_FONDO)
    fondo.set_points_as_corners([[-XMAX, -YMAX, Z_PISO], [XMAX, -YMAX, Z_PISO],
                                 [XMAX, YMAX, Z_PISO], [-XMAX, YMAX, Z_PISO],
                                 [-XMAX, -YMAX, Z_PISO]])
    fondo.set_fill(C_PISO, 1).set_stroke(C_GRILLA, 1.6, 0.9)
    grilla = VGroup()
    for x in np.arange(-XMAX + 1, XMAX):
        grilla.add(Pieza(alzar=CAPA_PISO_GRILLA).set_points_as_corners(
            [[x, -YMAX, Z_PISO], [x, YMAX, Z_PISO]]))
    for y in np.arange(-YMAX + 1, YMAX):
        grilla.add(Pieza(alzar=CAPA_PISO_GRILLA).set_points_as_corners(
            [[-XMAX, y, Z_PISO], [XMAX, y, Z_PISO]]))
    grilla.set_stroke(C_GRILLA, 1, 0.45)
    return VGroup(fondo, grilla)


def marcador(signo):
    color = C_POS if signo > 0 else C_NEG
    disco = Circle(radius=0.16, fill_color=color, fill_opacity=1,
                   stroke_color=WHITE, stroke_width=2)
    s = MathTex("+" if signo > 0 else "-", color=WHITE, font_size=34).move_to(disco)
    return VGroup(disco, s)


def bolita():
    cuerpo = Circle(radius=0.12, fill_color=C_BOLA, fill_opacity=1,
                    stroke_color=WHITE, stroke_width=1.2)
    cuerpo.set_sheen(-0.45, DR)
    brillo = Circle(radius=0.035, fill_color=WHITE, fill_opacity=0.8,
                    stroke_width=0).move_to(cuerpo.get_center() + 0.05 * (UP + LEFT))
    return VGroup(cuerpo, brillo)


def texto(s, font_size=26, color=C_TEXTO, **kw):
    return Text(s, font_size=font_size, color=color, **kw)


def formula(*partes, font_size=36):
    """MathTex coloreado: partes = (tex, color) o tex suelto."""
    texs = [p[0] if isinstance(p, tuple) else p for p in partes]
    m = MathTex(*texs, font_size=font_size, color=C_TEXTO)
    for sub, p in zip(m, partes):
        if isinstance(p, tuple):
            sub.set_color(p[1])
    return m


def frase(*partes, font_size=26, color=C_TEXTO):
    """Texto y matemática en una línea: las partes entre $…$ van con MathTex.
    Todas las piezas comparten la línea de base."""
    piezas, bases = VGroup(), []
    for p in partes:
        s, c = p if isinstance(p, tuple) else (p, color)
        if s.startswith("$"):
            # una x de referencia marca la línea de base de la fórmula
            m = MathTex(s.strip("$"), r"\mathrm{x}", font_size=font_size * 1.3, color=c)
            base, m = m[1].get_bottom()[1], m[0]
        else:
            m = Text(s, font_size=font_size, color=c)
            base = np.median([g.get_bottom()[1] for g in m.submobjects])
        piezas.add(m)
        bases.append(base - m.get_center()[1])
    piezas.arrange(RIGHT, buff=0.13)
    y0 = piezas[0].get_center()[1] + bases[0]
    for m, b in zip(piezas, bases):
        m.shift((y0 - m.get_center()[1] - b) * UP)
    return piezas


# ---------------------------------------------------------------------------
# Escena
# ---------------------------------------------------------------------------
class PotencialRelieve(ThreeDScene):
    def __init__(self, **kw):
        super().__init__(camera_class=Camara, **kw)

    def construct(self):
        self.res = (48, 36) if config.pixel_height >= 720 else (32, 24)
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=ZOOM_TOP,
                                    focal_distance=FOCAL)
        self.s = ValueTracker(0.0)              # factor de crecimiento del relieve
        self.opac = ValueTracker(OPAC_SOLIDA)
        self.x1 = ValueTracker(0.0)             # la carga positiva arranca en el origen
        self.qs = [ValueTracker(1.0), ValueTracker(0.0), ValueTracker(0.0)]
        self._clave = None
        self.leyenda_actual = None

        self.acto_titulo()
        self.acto_cargas()
        self.acto_montania()
        self.acto_superposicion()
        self.acto_curvas_nivel()
        self.acto_gradiente()
        self.acto_lineas_campo()
        self.acto_bolita()
        self.acto_cierre()

    # ----- utilidades -----------------------------------------------------
    def cargas(self):
        return [(self.x1.get_value(), 0.0, self.qs[0].get_value()),
                (D, 0.0, self.qs[1].get_value()),
                (*CARGA_3, self.qs[2].get_value())]

    def z_sup(self, x, y):
        s = self.s.get_value()
        return Z_PISO + s * (z_mundo(potencial(x, y, self.cargas())) - Z_PISO)

    def fijo(self, *mobs):
        # solo en la cámara: ThreeDScene.add_fixed_in_frame_mobjects también los
        # agrega a la escena y aparecerían antes de su animación de entrada
        self.camera.add_fixed_in_frame_mobjects(*mobs)
        return mobs[0] if len(mobs) == 1 else mobs

    def cartel(self, *mobs):
        """Siempre de frente a la cámara, pero ubicado en 3D."""
        self.camera.add_fixed_orientation_mobjects(*mobs)
        return mobs[0] if len(mobs) == 1 else mobs

    def leyenda(self, *lineas, color=C_TEXTO):
        g = VGroup(*[l if isinstance(l, Mobject) else texto(l, color=color)
                     for l in lineas]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        g.to_corner(DL, buff=0.45)
        return self.fijo(g)

    def cambiar_leyenda(self, *lineas, run_time=0.8, extra=()):
        anims = [FadeOut(self.leyenda_actual)] if self.leyenda_actual else []
        nueva = self.leyenda(*lineas) if lineas else None
        if nueva:
            anims.append(FadeIn(nueva, shift=0.15 * UP))
        self.play(*anims, *extra, run_time=run_time)
        self.leyenda_actual = nueva

    def mover_camara(self, phi, theta=None, zoom=None, centro=None, run_time=2.5,
                     extra=()):
        self.stop_ambient_camera_rotation()
        if theta is not None:
            # el equivalente de theta más cercano al actual: sin vueltas de más
            actual = self.camera.get_theta()
            theta = actual + (theta - actual + PI) % TAU - PI
        self.move_camera(phi=phi, theta=theta, zoom=zoom, frame_center=centro,
                         run_time=run_time, added_anims=list(extra))

    def en_panel(self, mob, y):
        """Ubica un texto fijo en el panel derecho de las vistas cenitales."""
        return mob.move_to([X_PANEL, y, 0], aligned_edge=LEFT)

    def actualizar_relieve(self, m):
        s, a = self.s.get_value(), self.opac.get_value()
        cargas = self.cargas()
        clave = (s, a, *np.ravel(cargas))
        if clave == self._clave:
            return
        self._clave = clave
        H = altura(potencial(m.X, m.Y, cargas))
        Z = Z_PISO + s * (Z_CERO + H - Z_PISO)
        m.poner(Z, s * H / Z_MAX, a * min(1.0, s / 0.12))

    def seguir_carga(self, k, dz=0.13):
        def actualizar(m):
            x, y, _ = self.cargas()[k]
            m.move_to([x, y, self.z_sup(x, y) + dz])
        return actualizar

    # ----- 1. título ------------------------------------------------------
    def acto_titulo(self):
        titulo = texto("El potencial como relieve", font_size=52, color=WHITE)
        sub = texto("el campo eléctrico y el gradiente de Análisis Matemático II",
                    font_size=26, color=GREY_B).next_to(titulo, DOWN, buff=0.35)
        self.fijo(titulo, sub)
        self.play(FadeIn(titulo, shift=0.2 * UP), FadeIn(sub), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(titulo), FadeOut(sub), run_time=0.8)

    # ----- 2. las cargas en el plano -------------------------------------
    def acto_cargas(self):
        self.piso = piso()
        rotulo = texto("plano xy", font_size=24, color=GREY_B)
        rotulo.move_to([-XMAX * ZOOM_TOP, YMAX * ZOOM_TOP + 0.3, 0], aligned_edge=LEFT)
        self.fijo(rotulo)
        self.play(FadeIn(self.piso), FadeIn(rotulo), run_time=1.2)

        self.m_pos = self.cartel(marcador(+1))
        self.m_pos.add_updater(self.seguir_carga(0))
        self.seguir_carga(0)(self.m_pos)
        self.play(GrowFromCenter(self.m_pos), run_time=0.8)
        self.cambiar_leyenda("una carga positiva en el plano")
        self.wait(1.2)
        self.rotulo_piso = rotulo

    # ----- 3. una carga = una montaña ------------------------------------
    def acto_montania(self):
        self.relieve = Relieve(*self.res)
        self.relieve.add_updater(self.actualizar_relieve)
        self.add(self.relieve)

        self.f_V = formula(r"V", "=", r"\frac{k\,q}{r}", font_size=44)
        nota = formula(r"z = V(x,y)", r"\qquad k = 1", font_size=28)
        panel = VGroup(self.f_V, nota).arrange(DOWN, buff=0.25)
        panel.to_corner(UR, buff=0.5)
        self.fijo(self.f_V, nota)
        self.nota_z = nota

        self.mover_camara(PHI_OBL, THETA_OBL, ZOOM_OBL, run_time=2.5,
                          extra=[FadeOut(self.rotulo_piso), FadeOut(self.leyenda_actual)])
        self.leyenda_actual = None
        self.play(self.s.animate.set_value(1), Write(self.f_V), FadeIn(nota),
                  run_time=3.5, rate_func=smooth)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.cambiar_leyenda("una carga positiva es una montaña")
        self.wait(3.2)

    # ----- 4. superposición ----------------------------------------------
    def acto_superposicion(self):
        self.play(self.x1.animate.set_value(-D), run_time=1.5)
        self.m_neg = self.cartel(marcador(-1))
        self.m_neg.add_updater(self.seguir_carga(1))
        self.seguir_carga(1)(self.m_neg)
        self.play(GrowFromCenter(self.m_neg), run_time=0.6)
        self.cambiar_leyenda("una carga negativa es un pozo",
                            extra=[self.qs[1].animate.set_value(-1)], run_time=3)
        self.wait(0.6)

        f_suma = formula(r"V", "=", r"\sum_i \frac{k\,q_i}{r_i}", font_size=44)
        f_suma.move_to(self.f_V, aligned_edge=UP)
        self.fijo(f_suma)
        self.play(ReplacementTransform(self.f_V, f_suma),
                  self.nota_z.animate.next_to(f_suma, DOWN, buff=0.25), run_time=1.2)
        self.f_V = f_suma
        self.cambiar_leyenda(texto("los potenciales se suman", font_size=30, color=WHITE))
        self.wait(1.8)

        self.m_3 = self.cartel(marcador(+1))
        self.m_3.add_updater(self.seguir_carga(2))
        self.seguir_carga(2)(self.m_3)
        self.play(GrowFromCenter(self.m_3), run_time=0.5)
        self.cambiar_leyenda("una tercera carga: el relieve deja de ser simétrico",
                            extra=[self.qs[2].animate.set_value(1)], run_time=2.5)
        self.wait(1.8)
        self.cambiar_leyenda("volvemos al dipolo",
                            extra=[self.qs[2].animate.set_value(0), FadeOut(self.m_3)],
                            run_time=2)
        self.m_3.clear_updaters()
        self.wait(0.8)

    # ----- 5. curvas de nivel = equipotenciales --------------------------
    def curva_en_altura(self, nivel, color=C_CORTE, ancho=5):
        z = z_mundo(nivel) + 0.01
        return VGroup(*[trozos(np.c_[c, np.full(len(c), z)], color, ancho)
                        for c in contornos(self.V_fino, XS_F, YS_F, nivel)])

    def acto_curvas_nivel(self):
        self.V_fino = potencial(XF, YF, DIPOLO)
        self.mover_camara(PHI_OBL, -68 * DEGREES, ZOOM_OBL, run_time=2,
                          extra=[self.opac.animate.set_value(OPAC_VIDRIO)])
        self.equis = VGroup()                     # equipotenciales en el piso
        nx, ny = self.res
        plano = plano_de_corte(nx, ny)
        zc = ValueTracker(Z_CERO + Z_MAX + 0.25)

        def mover_plano(p):
            z = zc.get_value()
            p.shift((z - p.z) * OUT)
            p.z = z

        mover_plano(plano)
        plano.add_updater(mover_plano)
        # Un juego fijo de trozos que se rellenan en cada cuadro. Si se cambiaran los
        # submobjects, el play seguiría dibujando los que había al compilarse
        # (Scene.moving_mobjects) y quedaría colgada la curva del nivel anterior.
        corte = VGroup(*[Pieza(alzar=ALZAR_SUP).set_stroke(C_CORTE, 5) for _ in range(200)])

        def actualizar_corte(g):
            nivel = nivel_de_altura(zc.get_value() - Z_CERO)
            nuevos = self.curva_en_altura(nivel).family_members_with_points()
            for k, p in enumerate(g.submobjects):
                if k < len(nuevos):
                    p.points = nuevos[k].points
                else:
                    p.clear_points()

        corte.add_updater(actualizar_corte)
        self.add(corte)
        self.cambiar_leyenda(frase("un plano horizontal", "$z = c$", "corta el relieve"),
                            extra=[FadeIn(plano)], run_time=1)

        def cortar(nivel, primera):
            self.play(zc.animate.set_value(z_mundo(nivel)), run_time=2)
            if primera:
                self.cambiar_leyenda(frase("el corte es una curva de nivel:", "$V = c$"),
                                    "la proyectamos al plano xy", run_time=0.6)
            caida = self.curva_en_altura(nivel)
            self.add(caida)
            self.play(caida.animate.shift((Z_PISO + 0.005 - z_mundo(nivel) - 0.01) * OUT)
                      .set_stroke(C_EQUI, 3), run_time=1.6)
            self.remove(caida)
            for c in contornos(self.V_fino, XS_F, YS_F, nivel):
                self.equis.add(en_piso(c, C_EQUI, 3))
            self.add(self.equis)

        cortar(1.5, True)
        cortar(0.55, False)
        self.play(FadeOut(plano), run_time=0.5)
        # desde abajo, para los niveles negativos
        zc.set_value(Z_CERO - Z_MAX - 0.15)
        mover_plano(plano)                     # FadeIn suspende el updater
        self.play(FadeIn(plano), run_time=0.5)
        cortar(-0.9, False)
        self.play(FadeOut(plano), run_time=0.5)
        corte.clear_updaters()
        plano.clear_updaters()
        self.remove(corte, plano)

        resto = VGroup(*[en_piso(c, C_EQUI, 3) for nivel in NIVELES
                         if nivel not in (1.5, 0.55, -0.9)
                         for c in contornos(self.V_fino, XS_F, YS_F, nivel)])
        self.play(Create(resto, lag_ratio=0.15), run_time=2)
        self.equis.add(*resto)
        self.add(self.equis)
        clave = frase("equipotencial", "$=$", "curva de nivel", font_size=34, color=WHITE)
        caja = SurroundingRectangle(clave, color=C_EQUI, buff=0.2, corner_radius=0.08,
                                    stroke_width=2)
        self.titular = VGroup(clave, caja).to_edge(UP, buff=0.4).to_edge(LEFT, buff=0.5)
        self.fijo(self.titular)
        self.cambiar_leyenda(extra=[FadeIn(self.titular)], run_time=0.8)
        self.wait(2.5)

    # ----- 6. el gradiente -----------------------------------------------
    def acto_gradiente(self):
        self.play(FadeOut(self.titular), FadeOut(self.f_V), FadeOut(self.nota_z),
                  run_time=0.6)
        self.mover_camara(0, -90 * DEGREES, ZOOM_TOP, centro=FC_TOP, run_time=2.5,
                          extra=[self.opac.animate.set_value(OPAC_TENUE)])
        z = Z_PISO + 0.01
        L = 0.75
        # sobre las equipotenciales ±0.3: lejos de las cargas, así las flechas no las tocan
        puntos = [(0.3, DIPOLO[0], 120 * DEGREES), (0.3, DIPOLO[0], -95 * DEGREES),
                  (-0.3, DIPOLO[1], 65 * DEGREES), (-0.3, DIPOLO[1], -120 * DEGREES)]
        grupos = []
        for nivel, (xq, yq, _), ang in puntos:
            P2 = punto_en_contorno(nivel, (xq, yq), ang)
            gx, gy = gradiente(P2[0], P2[1], DIPOLO)
            g = np.array([gx, gy, 0]) / np.hypot(gx, gy)
            t = np.array([-g[1], g[0], 0])
            P = np.array([P2[0], P2[1], z])
            gr = Arrow(P, P + L * g, buff=0, color=C_GRAD, stroke_width=5,
                       max_tip_length_to_length_ratio=0.3)
            e = Arrow(P, P - L * g, buff=0, color=C_E, stroke_width=5,
                      max_tip_length_to_length_ratio=0.3)
            tg = DashedLine(P - 0.7 * t, P + 0.7 * t, color=WHITE, stroke_width=2.5,
                            dash_length=0.07)
            # RightAngle solo acepta líneas en z = 0: se arma ahí y se baja al piso
            P0 = np.array([P2[0], P2[1], 0])
            ang_r = RightAngle(Line(P0, P0 - g), Line(P0, P0 + t), length=0.2,
                               color=WHITE, stroke_width=2).shift(z * OUT)
            punto = Dot(P, radius=0.05, color=WHITE)
            grupos.append(dict(P=P, g=g, t=t, grad=gr, E=e, tg=tg, ang=ang_r, dot=punto))

        a = grupos[0]
        l_grad = MathTex(r"\nabla V", color=C_GRAD, font_size=36)
        l_grad.next_to(a["P"] + L * a["g"], a["t"], buff=0.1)
        l_E = MathTex(r"\vec E", color=C_E, font_size=36)
        l_E.next_to(a["P"] - L * a["g"], -a["g"], buff=0.06)
        u = Arrow(a["P"], a["P"] + 0.5 * a["t"], buff=0, color=WHITE, stroke_width=4,
                  max_tip_length_to_length_ratio=0.35)
        l_u = MathTex(r"\vec u", color=WHITE, font_size=32)
        l_u.next_to(a["P"] + 0.5 * a["t"], -a["g"], buff=0.08)

        p1 = self.en_panel(frase(("$\\nabla V$", C_GRAD), "apunta cuesta arriba",
                                 font_size=28), 2.9)
        p2 = self.en_panel(frase(("$\\vec E = -\\nabla V$", C_E), "apunta cuesta abajo",
                                 font_size=28), 2.2)
        self.fijo(p1, p2)

        self.play(FadeIn(a["dot"], scale=0.5), run_time=0.4)
        self.play(GrowArrow(a["grad"]), FadeIn(l_grad), FadeIn(p1), run_time=1)
        self.wait(0.6)
        self.play(GrowArrow(a["E"]), FadeIn(l_E), FadeIn(p2), run_time=1)
        self.wait(0.6)
        self.play(Create(a["tg"]), GrowArrow(u), FadeIn(l_u), run_time=0.9)
        self.play(Create(a["ang"]), run_time=0.6)
        self.wait(1)
        resto = grupos[1:]
        self.play(*[FadeIn(b["dot"]) for b in resto],
                  *[GrowArrow(b[k]) for b in resto for k in ("grad", "E")],
                  *[Create(b[k]) for b in resto for k in ("tg", "ang")], run_time=1.6)
        self.wait(0.8)

        f_der = formula(r"D_{\vec u}V", "=", r"\nabla V\cdot\vec u", "=", "0", font_size=46)
        f_der[2][0:2].set_color(C_GRAD)
        self.en_panel(f_der, 1.0)
        c1 = frase("si", "$\\vec u$", "es tangente a la curva de nivel", font_size=24)
        c2 = texto("en esa dirección V no cambia", font_size=24, color=GREY_B)
        VGroup(c1, c2).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(
            f_der, DOWN, buff=0.3, aligned_edge=LEFT)
        conc = frase("por eso", ("$\\nabla V \\perp$", C_GRAD), "curvas de nivel",
                     font_size=28, color=WHITE)
        conc.next_to(c2, DOWN, buff=0.6, aligned_edge=LEFT).shift(0.15 * RIGHT)
        caja = SurroundingRectangle(conc, color=C_GRAD, buff=0.2, corner_radius=0.08,
                                    stroke_width=2)
        self.fijo(f_der, c1, c2, conc, caja)
        self.play(Write(f_der), run_time=1.5)
        self.play(FadeIn(c1), FadeIn(c2), run_time=0.8)
        self.wait(1.6)
        self.play(FadeIn(conc), Create(caja), run_time=1)
        self.wait(2.2)
        flechas = VGroup(*[b[k] for b in grupos for k in ("grad", "E", "tg", "ang", "dot")],
                         l_grad, l_E, u, l_u)
        self.play(FadeOut(flechas), FadeOut(VGroup(p1, p2, f_der, c1, c2, conc, caja)),
                  run_time=1)

    # ----- 7. líneas de campo --------------------------------------------
    def acto_lineas_campo(self):
        caminos = lineas_del_dipolo()
        self.caminos = caminos
        self.sobre = VGroup(*[trozos(sobre_relieve(c, DIPOLO), C_LINEA, 3.5)
                              for c in caminos])
        self.proy = VGroup(*[en_piso(c, C_LINEA, 2.6, capa=CAPA_PISO_CURVAS + 100)
                             for c in caminos])
        self.mover_camara(PHI_OBL, -58 * DEGREES, ZOOM_OBL, centro=ORIGIN, run_time=2.5,
                          extra=[self.opac.animate.set_value(OPAC_VIDRIO)])
        f_lin = formula(r"\frac{d\vec r}{ds}", "=", r"-\frac{\nabla V}{\lVert\nabla V\rVert}",
                        font_size=40)
        f_lin.to_corner(UR, buff=0.5)
        self.fijo(f_lin)
        self.cambiar_leyenda("líneas de campo: el camino de máximo descenso",
                            extra=[Write(f_lin)], run_time=1.2)
        self.play(*[Create(g, lag_ratio=1) for g in self.sobre],
                  *[Create(p) for p in self.proy], run_time=3.5, rate_func=linear)
        self.wait(1.5)

        # puntas de flecha en el piso, a mitad de cada línea
        puntas = VGroup()
        for c in caminos:
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
            i = np.searchsorted(s, 0.55 * s[-1])
            d = normalize(np.r_[c[i + 1] - c[i - 1], 0])
            p = np.r_[c[i], Z_PISO + 0.01]
            n = np.array([-d[1], d[0], 0])
            puntas.add(Polygon(p + 0.11 * d, p - 0.08 * d + 0.08 * n, p - 0.08 * d - 0.08 * n,
                               stroke_width=0, fill_color=C_LINEA, fill_opacity=1))
        self.puntas = puntas
        self.cambiar_leyenda(extra=[FadeOut(f_lin)], run_time=0.6)
        self.mover_camara(0, -90 * DEGREES, ZOOM_TOP, centro=FC_TOP, run_time=2.5,
                          extra=[self.opac.animate.set_value(OPAC_TENUE),
                                 self.sobre.animate.set_stroke(opacity=0)])
        self.remove(self.sobre)
        titulo = self.en_panel(frase("líneas de campo", "$\\perp$", "equipotenciales",
                                     font_size=25, color=WHITE), 1.0)
        muestras = VGroup()
        for color, nombre in ((C_LINEA, "línea de campo: tangente a E"),
                              (C_EQUI, "equipotencial: V constante")):
            muestras.add(VGroup(Line(ORIGIN, 0.6 * RIGHT, color=color, stroke_width=4),
                                texto(nombre, font_size=24)).arrange(RIGHT, buff=0.2))
        muestras.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        muestras.next_to(titulo, DOWN, buff=0.5, aligned_edge=LEFT)
        self.fijo(titulo, muestras)
        self.panel_lineas = VGroup(titulo, muestras)
        self.play(FadeIn(puntas), FadeIn(titulo, shift=0.15 * UP), run_time=1)
        self.play(FadeIn(muestras), run_time=0.6)
        self.wait(3)

    # ----- 8. bolita de prueba -------------------------------------------
    def acto_bolita(self):
        camino = self.caminos[-2]                 # sale a −60° de la carga positiva
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(camino, axis=0), axis=1))])
        r = np.linalg.norm(camino - np.array(DIPOLO[0][:2]), axis=1)
        i0 = np.argmax(r > 0.35)
        P3 = sobre_relieve(camino, DIPOLO)
        # por delante de la línea verde que recorre el mismo camino
        resaltada = trozos(P3[i0:] + 0.01 * OUT, WHITE, 5.5, alzar=ALZAR_SUP + 0.2)
        self.sobre.set_stroke(opacity=1)
        self.play(FadeOut(self.panel_lineas), run_time=0.6)
        self.mover_camara(PHI_OBL, -55 * DEGREES, ZOOM_OBL, centro=ORIGIN, run_time=2.5,
                          extra=[self.opac.animate.set_value(0.88), FadeOut(self.puntas),
                                 FadeIn(self.sobre)])
        u = ValueTracker(s[i0])
        bola = self.cartel(bolita())

        def rodar(b):
            k = np.interp(u.get_value(), s, np.arange(len(s)))
            j = int(min(k, len(s) - 2))
            p = P3[j] + (k - j) * (P3[j + 1] - P3[j])
            b.move_to(p + 0.13 * OUT)

        rodar(bola)
        bola.add_updater(rodar)
        self.cambiar_leyenda(
            texto("una carga positiva va cuesta abajo", font_size=30, color=WHITE),
            texto("sigue la línea de campo, no rueda con inercia", font_size=20,
                  color=GREY_B),
            extra=[Create(resaltada, lag_ratio=1), FadeIn(bola, scale=0.5)], run_time=1.5)
        self.play(u.animate.set_value(s[-1]), run_time=5, rate_func=smooth)
        self.wait(1.5)
        self.bola, self.resaltada = bola, resaltada

    # ----- 9. cierre -------------------------------------------------------
    def acto_cierre(self):
        f = formula((r"\vec E", C_E), "=", (r"-\nabla V", C_GRAD), font_size=60)
        caja = SurroundingRectangle(f, color=WHITE, buff=0.25, corner_radius=0.1)
        sub = frase(("$\\vec E \\perp$", C_E), "equipotenciales", font_size=28, color=WHITE)
        g = VGroup(VGroup(f, caja), sub).arrange(DOWN, buff=0.3).to_corner(UR, buff=0.45)
        self.fijo(f, caja, sub)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.cambiar_leyenda(extra=[Write(f), Create(caja), FadeIn(sub),
                                    FadeOut(self.bola), FadeOut(self.resaltada)],
                             run_time=2)
        self.bola.clear_updaters()
        self.wait(6)


# ---------------------------------------------------------------------------
# Verificaciones: python potencial_relieve.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ∇V analítico contra diferencias centradas finas
    rng = np.random.default_rng(1)
    pts = rng.uniform([-XMAX, -YMAX], [XMAX, YMAX], (200, 2))
    h = 1e-5
    gx, gy = gradiente(pts[:, 0], pts[:, 1], DIPOLO)
    dx = (potencial(pts[:, 0] + h, pts[:, 1], DIPOLO) - potencial(pts[:, 0] - h, pts[:, 1], DIPOLO)) / (2 * h)
    dy = (potencial(pts[:, 0], pts[:, 1] + h, DIPOLO) - potencial(pts[:, 0], pts[:, 1] - h, DIPOLO)) / (2 * h)
    err = np.max(np.hypot(gx - dx, gy - dy) / np.hypot(gx, gy))
    print(f"∇V analítico vs numérico: error relativo máx = {err:.1e}")
    assert err < 1e-6

    # alturas: sin picos ni NaN
    for cargas in (DIPOLO, [(0, 0, 1)], DIPOLO + [(*CARGA_3, 1.0)]):
        Z = altura(potencial(XF, YF, cargas))
        assert np.all(np.isfinite(Z)) and np.abs(Z).max() < Z_MAX
    Z = z_mundo(potencial(XF, YF, DIPOLO))
    print(f"relieve en [{Z.min():.2f}, {Z.max():.2f}], piso en {Z_PISO:.2f}")
    assert Z.min() - Z_PISO > 0.3

    # contornos: V = c en todos sus puntos, cerrados y dentro del dominio
    VF = potencial(XF, YF, DIPOLO)
    for c in NIVELES:
        curvas = contornos(VF, XS_F, YS_F, c)
        P = np.concatenate(curvas)
        e = np.abs(potencial(P[:, 0], P[:, 1], DIPOLO) - c).max()
        cerradas = all(np.allclose(k[0], k[-1]) for k in curvas)
        dmin = min(np.hypot(P[:, 0] - xq, P[:, 1] - yq).min() for xq, yq, _ in DIPOLO)
        print(f"nivel {c:+.2f}: {len(curvas)} curva(s), |V−c| ≤ {e:.1e}, "
              f"cerradas={cerradas}, distancia mínima a una carga = {dmin:.2f}")
        assert e < 5e-3 and cerradas and dmin > 0.3

    # líneas de campo: bajan siempre, terminan en la negativa o en el borde
    lineas = lineas_del_dipolo()
    for a, c in zip(np.degrees(ANGULOS_LINEAS), lineas):
        V = potencial(c[:, 0], c[:, 1], DIPOLO)
        assert np.all(np.diff(V) < 0), a
        fin = np.hypot(c[-1, 0] - D, c[-1, 1])
        donde = "carga −" if fin < R_FIN + 0.02 else "borde"
        print(f"línea a {a:5.0f}°: {len(c):4d} pasos, termina en {donde}")
    assert sum(np.hypot(c[-1, 0] - D, c[-1, 1]) < R_FIN + 0.02 for c in lineas) >= 7

    # ortogonalidad: en cada cruce línea-equipotencial el ángulo es 90°
    peor = 0.0
    for c in lineas:
        V = potencial(c[:, 0], c[:, 1], DIPOLO)
        for nivel in NIVELES:
            for i in np.nonzero(np.diff(np.sign(V - nivel)))[0]:
                t = (nivel - V[i]) / (V[i + 1] - V[i])
                p = c[i] + t * (c[i + 1] - c[i])
                d_lin = normalize(c[i + 1] - c[i])
                P = np.concatenate(contornos(VF, XS_F, YS_F, nivel))
                k = np.argmin(np.hypot(*(P - p).T))
                d_eq = normalize(P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)])
                peor = max(peor, abs(d_lin @ d_eq))
    print(f"cruces línea de campo / equipotencial: |cos θ| máx = {peor:.3f} "
          f"(θ = {np.degrees(np.arccos(peor)):.1f}°)")
    assert peor < 0.05
    print("ok")
