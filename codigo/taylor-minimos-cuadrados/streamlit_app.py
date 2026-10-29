"""
App de Streamlit: Taylor contra mínimos cuadrados sobre sen x.

Corre en el navegador con stlite (index.html), o en local:  streamlit run streamlit_app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from ajuste import A, B, ajustar, datos, errores, predecir, taylor

st.set_page_config(page_title="Taylor contra mínimos cuadrados", layout="wide")
st.title("Taylor contra mínimos cuadrados")
st.caption("Aproximar sen x con un polinomio: derivadas en un punto (Taylor) o puntos ruidosos (mínimos cuadrados).")

col_a, col_b, col_c, col_d = st.columns(4)
grado = col_a.slider("Grado del polinomio", 1, 12, 5)
n = col_b.slider("Cantidad de puntos", 8, 80, 20)
ruido = col_c.slider("Ruido (desvío)", 0.0, 0.6, 0.25, 0.05)
semilla = col_d.number_input("Semilla aleatoria", 0, 999, 0)


@st.cache_data
def calcular(n, ruido, semilla):
    x, y = datos(n, ruido, semilla)
    return x, y, errores(x, y, max_grado=12)


x, y, df = calcular(n, ruido, int(semilla))
xs = np.linspace(A, B, 600)
modelo = ajustar(x, y, grado)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(xs, np.sin(xs), color="gray", lw=2.4, label="sen x")
ax1.scatter(x, y, color="gray", s=22, alpha=0.8, label="puntos con ruido", zorder=3)
ax1.plot(xs, np.clip(predecir(modelo, xs), -2.2, 2.2), color="#18bc9c", lw=2, label=f"mínimos cuadrados, grado {grado}")
ax1.plot(xs, taylor(xs, grado), color="#e0b100", lw=1.8, ls="--", label=f"Taylor, grado {grado}")
ax1.set_ylim(-2.2, 2.2)
ax1.legend(fontsize=8, loc="upper left")
ax1.set_xlabel("x")

sns.lineplot(data=df, x="grado", y="rmse", hue="conjunto", style="conjunto", markers=True, dashes=False, ax=ax2)
ax2.axvline(grado, color="gray", ls=":")
ax2.set_yscale("log")
ax2.set_xticks(range(1, 13))
ax2.set_ylabel("error RMSE (escala log)")
ax2.legend(title=None, fontsize=8)
st.pyplot(fig)

fila = df[df.grado == grado].set_index("conjunto").rmse
m1, m2, m3 = st.columns(3)
m1.metric("Entrenamiento (con ruido)", f"{fila['Entrenamiento (con ruido)']:.3f}")
m2.metric("Prueba (sen x exacto)", f"{fila['Prueba (sen x exacto)']:.3f}")
m3.metric("Taylor en 0", f"{fila['Taylor en 0 (sen x exacto)']:.3f}")

with st.expander("Tabla de errores (pandas)"):
    st.dataframe(df.pivot(index="conjunto", columns="grado", values="rmse").round(3))
