"""
Taylor contra mínimos cuadrados, sobre sen x.

  Taylor:            polinomio de grado n en x = 0; usa las derivadas en un punto y no mira datos.
  Mínimos cuadrados: polinomio de grado n ajustado a puntos ruidosos de sen x en [-π, π].

Este módulo lo usan figuras.py (la nota) y streamlit_app.py (la app en el navegador).
"""
from math import factorial

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

A, B = -np.pi, np.pi


def datos(n=30, ruido=0.2, semilla=0):
    """n puntos (x, sen x + ruido gaussiano) con x uniforme en [-π, π]."""
    rng = np.random.default_rng(semilla)
    x = rng.uniform(A, B, n)
    return x, np.sin(x) + rng.normal(0, ruido, n)


def taylor(x, grado):
    """Polinomio de Taylor de sen x en 0: sólo entran las potencias impares ≤ grado."""
    return sum((-1) ** k * x ** (2 * k + 1) / factorial(2 * k + 1) for k in range((grado + 1) // 2))


def ajustar(x, y, grado):
    """Modelo de mínimos cuadrados de grado dado (estandariza para que x^12 no desbalancee el sistema)."""
    modelo = make_pipeline(PolynomialFeatures(grado, include_bias=False), StandardScaler(), LinearRegression())
    return modelo.fit(x.reshape(-1, 1), y)


def predecir(modelo, x):
    return modelo.predict(np.asarray(x).reshape(-1, 1))


def errores(x, y, max_grado=12, n_prueba=400):
    """DataFrame largo (grado, conjunto, rmse): error en los puntos de entrenamiento, en la curva limpia y de Taylor."""
    xp = np.linspace(A, B, n_prueba)
    filas = []
    for g in range(1, max_grado + 1):
        m = ajustar(x, y, g)
        filas.append((g, "Entrenamiento (con ruido)", mean_squared_error(y, predecir(m, x)) ** 0.5))
        filas.append((g, "Prueba (sen x exacto)", mean_squared_error(np.sin(xp), predecir(m, xp)) ** 0.5))
        filas.append((g, "Taylor en 0 (sen x exacto)", mean_squared_error(np.sin(xp), taylor(xp, g)) ** 0.5))
    return pd.DataFrame(filas, columns=["grado", "conjunto", "rmse"])
