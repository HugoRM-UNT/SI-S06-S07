from __future__ import annotations

import streamlit as st

import app_optimizador_proyectos
import app_riesgo_academico


COLAB_URL = (
    "https://colab.research.google.com/drive/"
    "1fsLRZc6S92dFnLqLoaBK4LQlDuumvPg3?usp=sharing"
)


st.set_page_config(
    page_title="Sistemas Inteligentes · Sesiones 06 y 07",
    page_icon="🧠",
    layout="wide",
)


def main() -> None:
    st.title("🧠 Sistemas Inteligentes")
    st.caption(
        "Panel principal para explorar los trabajos de las sesiones 06 y 07: "
        "riesgo académico y optimización de proyectos."
    )

    with st.sidebar:
        st.header("Navegación")
        panel = st.radio(
            "Selecciona un panel",
            (
                "Inicio",
                "🎓 Riesgo académico · Sesión 06",
                "🧬 Optimizador de proyectos · Sesión 07",
            ),
        )
        st.divider()
        st.link_button("Abrir versión en Google Colab ↗", COLAB_URL, use_container_width=True)

    if panel == "Inicio":
        st.subheader("Bienvenido")
        st.write(
            "Selecciona uno de los paneles en la barra lateral para iniciar el análisis."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**Sesión 06 · Riesgo académico**\n\n"
                "Evalúa un índice de riesgo con inferencia difusa Mamdani."
            )
        with col2:
            st.info(
                "**Sesión 07 · Optimizador de proyectos**\n\n"
                "Busca la mejor combinación de proyectos con un algoritmo genético."
            )
        st.link_button("Ir a la versión colaborativa en Colab ↗", COLAB_URL)
    elif panel.startswith("🎓"):
        app_riesgo_academico.render()
    else:
        app_optimizador_proyectos.render()


if __name__ == "__main__":
    main()
