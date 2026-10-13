from __future__ import annotations

from itertools import product
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================
# Universo de discurso de cada variable
X_ASISTENCIA = np.linspace(0, 100, 501)
X_PROMEDIO = np.linspace(0, 20, 401)
X_ESTUDIO = np.linspace(0, 30, 301)
X_RIESGO = np.linspace(0, 100, 501)

# Parámetros de funciones trapezoidales/triangulares.
# Estos parámetros son decisiones de modelado que el grupo debe justificar.
PARAMETROS = {
    "asistencia": {
        "Baja": ("trap", (0, 0, 40, 60)),
        "Media": ("tri", (40, 65, 85)),
        "Alta": ("trap", (75, 90, 100, 100)),
    },
    "promedio": {
        "Bajo": ("trap", (0, 0, 6, 11)),
        "Regular": ("tri", (8, 12, 16)),
        "Alto": ("trap", (14, 17, 20, 20)),
    },
    "estudio": {
        "Pocas": ("trap", (0, 0, 4, 10)),
        "Moderadas": ("tri", (6, 15, 24)),
        "Muchas": ("trap", (18, 24, 30, 30)),
    },
    "riesgo": {
        "Bajo": ("trap", (0, 0, 20, 45)),
        "Medio": ("tri", (25, 50, 75)),
        "Alto": ("trap", (55, 80, 100, 100)),
    },
}

UNIVERSOS = {
    "asistencia": X_ASISTENCIA,
    "promedio": X_PROMEDIO,
    "estudio": X_ESTUDIO,
    "riesgo": X_RIESGO,
}


# ============================================================
# 2. FUNCIONES DE PERTENENCIA
# ============================================================
def triangular(x: float | np.ndarray, a: float, b: float, c: float):
    """Función de pertenencia triangular."""
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    if b > a:
        mask = (x_arr >= a) & (x_arr <= b)
        y[mask] = (x_arr[mask] - a) / (b - a)
    else:
        y[x_arr == b] = 1.0
    if c > b:
        mask = (x_arr >= b) & (x_arr <= c)
        y[mask] = np.maximum(y[mask], (c - x_arr[mask]) / (c - b))
    else:
        y[x_arr == b] = 1.0
    y[x_arr == b] = 1.0
    return float(y) if y.ndim == 0 else y


def trapezoidal(x: float | np.ndarray, a: float, b: float, c: float, d: float):
    """Función de pertenencia trapezoidal, incluidos hombros izquierdo/derecho."""
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)

    # Ascenso
    if b > a:
        mask = (x_arr >= a) & (x_arr < b)
        y[mask] = (x_arr[mask] - a) / (b - a)
    else:
        y[x_arr <= b] = 1.0

    # Meseta
    y[(x_arr >= b) & (x_arr <= c)] = 1.0

    # Descenso
    if d > c:
        mask = (x_arr > c) & (x_arr <= d)
        y[mask] = (d - x_arr[mask]) / (d - c)
    else:
        y[x_arr >= c] = 1.0

    return float(y) if y.ndim == 0 else y


def pertenencia(variable: str, etiqueta: str, x: float | np.ndarray):
    tipo, parametros = PARAMETROS[variable][etiqueta]
    if tipo == "tri":
        return triangular(x, *parametros)
    return trapezoidal(x, *parametros)


def grados_entrada(asistencia: float, promedio: float, estudio: float):
    """Calcula el grado de pertenencia de cada entrada a sus conjuntos."""
    valores = {
        "asistencia": asistencia,
        "promedio": promedio,
        "estudio": estudio,
    }
    return {
        variable: {
            etiqueta: float(pertenencia(variable, etiqueta, valor))
            for etiqueta in PARAMETROS[variable]
        }
        for variable, valor in valores.items()
    }


# ============================================================
# 3. BASE DE REGLAS (27 COMBINACIONES)
# ============================================================
ETIQUETAS_ENTRADA = {
    "asistencia": ["Baja", "Media", "Alta"],
    "promedio": ["Bajo", "Regular", "Alto"],
    "estudio": ["Pocas", "Moderadas", "Muchas"],
}

# Puntos de riesgo asignados a cada etiqueta:
# valores mayores representan condiciones asociadas a mayor riesgo.
PUNTAJE_RIESGO = {
    "asistencia": {"Baja": 2, "Media": 1, "Alta": 0},
    "promedio": {"Bajo": 2, "Regular": 1, "Alto": 0},
    "estudio": {"Pocas": 2, "Moderadas": 1, "Muchas": 0},
}


def construir_reglas(conservador: bool = False):
    """
    Genera las 27 reglas posibles. En modo conservador se necesita
    menos puntaje para activar una consecuencia de riesgo más alto.
    """
    reglas = []
    for asistencia, promedio, estudio in product(
        ETIQUETAS_ENTRADA["asistencia"],
        ETIQUETAS_ENTRADA["promedio"],
        ETIQUETAS_ENTRADA["estudio"],
    ):
        puntaje = (
            PUNTAJE_RIESGO["asistencia"][asistencia]
            + PUNTAJE_RIESGO["promedio"][promedio]
            + PUNTAJE_RIESGO["estudio"][estudio]
        )

        if conservador:
            consecuente = "Alto" if puntaje >= 3 else ("Medio" if puntaje >= 1 else "Bajo")
        else:
            consecuente = "Alto" if puntaje >= 4 else ("Medio" if puntaje >= 2 else "Bajo")

        reglas.append({
            "asistencia": asistencia,
            "promedio": promedio,
            "estudio": estudio,
            "riesgo": consecuente,
            "puntaje": puntaje,
        })
    return reglas


# ============================================================
# 4. INFERENCIA MAMDANI Y DESFUSIFICACIÓN POR CENTROIDE
# ============================================================
def evaluar(
    asistencia: float,
    promedio: float,
    estudio: float,
    conservador: bool = False,
):
    grados = grados_entrada(asistencia, promedio, estudio)
    reglas = construir_reglas(conservador)
    salida_agregada = np.zeros_like(X_RIESGO)
    reglas_activadas = []

    for numero, regla in enumerate(reglas, start=1):
        # Conjunción AND: mínimo de los grados de pertenencia.
        activacion = min(
            grados["asistencia"][regla["asistencia"]],
            grados["promedio"][regla["promedio"]],
            grados["estudio"][regla["estudio"]],
        )

        if activacion > 0:
            # Implicación Mamdani: recorte por mínimo.
            conjunto_salida = pertenencia("riesgo", regla["riesgo"], X_RIESGO)
            salida_recortada = np.minimum(activacion, conjunto_salida)

            # Agregación de reglas: máximo.
            salida_agregada = np.maximum(salida_agregada, salida_recortada)

            reglas_activadas.append({
                "Regla": numero,
                "SI asistencia": regla["asistencia"],
                "Y promedio": regla["promedio"],
                "Y estudio": regla["estudio"],
                "ENTONCES riesgo": regla["riesgo"],
                "Activación": round(float(activacion), 4),
            })

    # Centroide: integral(x * mu(x)) / integral(mu(x)).
    area = np.trapezoid(salida_agregada, X_RIESGO)
    if area == 0:
        valor_riesgo = 0.0
    else:
        momento = np.trapezoid(X_RIESGO * salida_agregada, X_RIESGO)
        valor_riesgo = float(momento / area)

    if valor_riesgo < 35:
        categoria = "Bajo"
    elif valor_riesgo < 65:
        categoria = "Medio"
    else:
        categoria = "Alto"

    return {
        "grados": grados,
        "reglas_activadas": reglas_activadas,
        "salida_agregada": salida_agregada,
        "valor": valor_riesgo,
        "categoria": categoria,
    }


# ============================================================
# 5. GRÁFICOS DE CONJUNTOS DIFUSOS
# ============================================================
def figura_conjuntos(variable: str, titulo: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    universo = UNIVERSOS[variable]
    for etiqueta in PARAMETROS[variable]:
        ax.plot(
            universo,
            pertenencia(variable, etiqueta, universo),
            label=etiqueta,
            linewidth=2,
        )
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Grado de pertenencia")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def figura_salida(resultado: dict):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for etiqueta in PARAMETROS["riesgo"]:
        ax.plot(
            X_RIESGO,
            pertenencia("riesgo", etiqueta, X_RIESGO),
            linestyle="--",
            linewidth=1.5,
            label=f"Conjunto {etiqueta.lower()}",
        )
    ax.fill_between(
        X_RIESGO,
        resultado["salida_agregada"],
        alpha=0.35,
        label="Salida agregada",
    )
    ax.axvline(
        resultado["valor"],
        linestyle=":",
        linewidth=2,
        label=f"Centroide = {resultado['valor']:.2f}",
    )
    ax.set_title("Inferencia Mamdani y desfusificación")
    ax.set_xlabel("Índice de riesgo académico (0–100)")
    ax.set_ylabel("Grado de pertenencia")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# ============================================================
# 6. INTERFAZ STREAMLIT
# ============================================================
def render():
    """Renderiza el panel de evaluación de riesgo académico."""
    st.title("🎓 Sistema inteligente de evaluación de riesgo académico")
    st.caption(
        "Modelo académico basado en lógica difusa Mamdani. "
        "El resultado es un índice de riesgo, no una probabilidad real de desaprobar."
    )
    pestana_eval, pestana_funciones, pestana_casos, pestana_metodo = st.tabs([
        "🧮 Evaluar estudiante",
        "📈 Funciones de pertenencia",
        "🧪 Casos de prueba",
        "📘 Método y análisis",
    ])

    with pestana_eval:
        st.subheader("Ingresar datos del estudiante")
        col1, col2, col3 = st.columns(3)

        with col1:
            asistencia = st.slider("Asistencia (%)", 0, 100, 80)
        with col2:
            promedio = st.slider("Promedio (0–20)", 0.0, 20.0, 13.0, 0.5)
        with col3:
            estudio = st.slider("Horas de estudio por semana", 0, 30, 10)

        conservador = st.checkbox(
            "Activar análisis de sensibilidad (reglas más conservadoras)",
            value=False,
            help="Cambia los umbrales de algunas consecuencias de las reglas para observar el efecto en el resultado.",
        )

        if st.button("Evaluar riesgo académico", type="primary"):
            st.session_state["resultado_actual"] = evaluar(
                asistencia, promedio, estudio, conservador
            )
            st.session_state["entradas_actuales"] = (asistencia, promedio, estudio)
            st.session_state["modo_actual"] = conservador

        if "resultado_actual" in st.session_state:
            resultado = st.session_state["resultado_actual"]
            entradas = st.session_state["entradas_actuales"]

            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Índice de riesgo académico", f"{resultado['valor']:.2f} / 100")
            m2.metric("Categoría final", resultado["categoria"])

            st.pyplot(figura_salida(resultado))

            st.markdown("#### Grados de pertenencia de las entradas")
            filas_grados = []
            for variable, valores in resultado["grados"].items():
                for etiqueta, grado in valores.items():
                    filas_grados.append({
                        "Variable": variable.capitalize(),
                        "Conjunto difuso": etiqueta,
                        "Grado de pertenencia": round(grado, 4),
                    })
            st.dataframe(pd.DataFrame(filas_grados), use_container_width=True, hide_index=True)

            st.markdown("#### Reglas activadas")
            if resultado["reglas_activadas"]:
                reglas_df = pd.DataFrame(resultado["reglas_activadas"])
                st.dataframe(reglas_df, use_container_width=True, hide_index=True)
                st.caption(f"Se activaron {len(reglas_df)} de las 27 reglas.")
            else:
                st.warning("Ninguna regla se activó. Revisar las funciones de pertenencia.")

            st.markdown("#### Interpretación automática")
            st.write(
                f"Para asistencia de {entradas[0]} %, promedio de {entradas[1]:.1f} "
                f"y {entradas[2]} horas de estudio semanales, el modelo produce un "
                f"índice de riesgo de {resultado['valor']:.2f}, clasificado como "
                f"**{resultado['categoria'].lower()}**. Este valor es una salida del "
                "modelo de reglas difusas, no una probabilidad de desaprobar."
            )

    with pestana_funciones:
        st.subheader("Conjuntos difusos de entrada y salida")
        st.info(
            "Las curvas se superponen intencionalmente: un mismo valor puede pertenecer "
            "parcialmente a dos conjuntos lingüísticos."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(figura_conjuntos("asistencia", "Asistencia", "Asistencia (%)"))
            st.pyplot(figura_conjuntos("estudio", "Estudio semanal", "Horas por semana"))
        with c2:
            st.pyplot(figura_conjuntos("promedio", "Promedio", "Promedio (0–20)"))
            st.pyplot(figura_conjuntos("riesgo", "Riesgo académico", "Índice de riesgo (0–100)"))

        st.markdown("#### Parámetros utilizados")
        filas_parametros = []
        nombres_variables = {
            "asistencia": "Asistencia (%)",
            "promedio": "Promedio",
            "estudio": "Horas de estudio",
            "riesgo": "Riesgo académico",
        }
        for variable, conjuntos in PARAMETROS.items():
            for etiqueta, (tipo, parametros) in conjuntos.items():
                filas_parametros.append({
                    "Variable": nombres_variables[variable],
                    "Etiqueta": etiqueta,
                    "Función": "Triangular" if tipo == "tri" else "Trapezoidal",
                    "Parámetros": str(parametros),
                })
        st.dataframe(pd.DataFrame(filas_parametros), use_container_width=True, hide_index=True)

    with pestana_casos:
        st.subheader("Cinco casos proporcionados y dos casos adicionales")
        casos = [
            {"Estudiante": "E01", "Asistencia": 95, "Promedio": 17, "Horas/semana": 20},
            {"Estudiante": "E02", "Asistencia": 72, "Promedio": 13, "Horas/semana": 10},
            {"Estudiante": "E03", "Asistencia": 48, "Promedio": 9, "Horas/semana": 4},
            {"Estudiante": "E04", "Asistencia": 85, "Promedio": 11, "Horas/semana": 8},
            {"Estudiante": "E05", "Asistencia": 60, "Promedio": 16, "Horas/semana": 15},
            {"Estudiante": "Adicional 1", "Asistencia": 98, "Promedio": 19, "Horas/semana": 28},
            {"Estudiante": "Adicional 2", "Asistencia": 35, "Promedio": 6, "Horas/semana": 2},
        ]

        resultados_casos = []
        for caso in casos:
            res = evaluar(
                caso["Asistencia"],
                caso["Promedio"],
                caso["Horas/semana"],
            )
            resultados_casos.append({
                **caso,
                "Índice de riesgo": round(res["valor"], 2),
                "Categoría": res["categoria"],
                "Reglas activadas": len(res["reglas_activadas"]),
            })

        df_casos = pd.DataFrame(resultados_casos)
        st.dataframe(df_casos, use_container_width=True, hide_index=True)

        st.markdown("#### Ver detalle de un caso")
        elegido = st.selectbox("Selecciona un estudiante", [c["Estudiante"] for c in casos])
        caso = next(c for c in casos if c["Estudiante"] == elegido)
        detalle = evaluar(caso["Asistencia"], caso["Promedio"], caso["Horas/semana"])

        st.write(
            f"**Entradas:** asistencia {caso['Asistencia']} %, promedio "
            f"{caso['Promedio']}, estudio {caso['Horas/semana']} h/semana."
        )
        st.write(
            f"**Salida:** {detalle['valor']:.2f}/100 — categoría **{detalle['categoria']}**."
        )
        st.markdown("**Reglas activadas y fuerza de activación:**")
        st.dataframe(
            pd.DataFrame(detalle["reglas_activadas"]),
            use_container_width=True,
            hide_index=True,
        )
        st.pyplot(figura_salida(detalle))

    with pestana_metodo:
        st.subheader("¿Cómo funciona el sistema?")
        st.markdown("""
        1. **Fusificación:** convierte cada entrada numérica en grados de pertenencia a conjuntos como Baja, Media o Alta.
        2. **Conjunción (AND):** se usa el mínimo de los grados de las tres condiciones de cada regla.
        3. **Implicación Mamdani:** se recorta el conjunto de salida de la regla usando su grado de activación (mínimo).
        4. **Agregación:** se combinan las salidas de todas las reglas usando el máximo.
        5. **Desfusificación:** se obtiene un número de 0 a 100 con el centroide del área agregada.
        """)

        st.markdown("#### Preguntas de análisis")
        st.markdown("""
        **¿Por qué no basta el promedio?**
        Porque el promedio no representa por sí solo la asistencia ni el tiempo dedicado al estudio. Dos estudiantes con el mismo promedio pueden tener hábitos y condiciones diferentes.

        **¿Qué diferencia existe entre pertenencia y probabilidad?**
        Un grado de pertenencia expresa cuánto encaja un valor en una categoría lingüística (por ejemplo, asistencia Media). No significa que exista esa probabilidad de desaprobar.

        **¿Cómo se interpreta pertenecer parcialmente a dos conjuntos?**
        Una asistencia de 60 % podría pertenecer en parte a Baja y en parte a Media. La lógica difusa conserva esa transición gradual en lugar de imponer una frontera rígida.

        **¿Qué efecto tiene modificar una regla?**
        Al activar el análisis de sensibilidad se cambian los umbrales usados para asignar consecuencias a las reglas. El índice puede subir para algunas combinaciones. Esto ilustra que los resultados dependen de las decisiones de diseño y deben justificarse.
        """)

        st.markdown("#### Base completa de reglas SI–ENTONCES")
        base_reglas = construir_reglas(conservador=False)
        tabla_reglas = []
        for i, regla in enumerate(base_reglas, start=1):
            tabla_reglas.append({
                "N.º": i,
                "Regla SI asistencia": regla["asistencia"],
                "Y promedio": regla["promedio"],
                "Y estudio": regla["estudio"],
                "ENTONCES riesgo": regla["riesgo"],
            })
        st.dataframe(pd.DataFrame(tabla_reglas), use_container_width=True, hide_index=True)


    st.divider()
