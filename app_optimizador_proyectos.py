from __future__ import annotations

import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# 1. DATOS DEL PROBLEMA
# ============================================================
PRESUPUESTO = 20_000

PROYECTOS = [
    {"id": "P1", "costo": 4000, "beneficio": 9},
    {"id": "P2", "costo": 6000, "beneficio": 14},
    {"id": "P3", "costo": 3000, "beneficio": 7},
    {"id": "P4", "costo": 5000, "beneficio": 11},
    {"id": "P5", "costo": 8000, "beneficio": 17},
    {"id": "P6", "costo": 2000, "beneficio": 5},
    {"id": "P7", "costo": 7000, "beneficio": 15},
    {"id": "P8", "costo": 4500, "beneficio": 10},
]

COSTOS = [p["costo"] for p in PROYECTOS]
BENEFICIOS = [p["beneficio"] for p in PROYECTOS]
IDS = [p["id"] for p in PROYECTOS]

CONFIGURACIONES = {
    "A": {"poblacion": 20, "cruce": 0.7, "mutacion": 0.01, "generaciones": 50},
    "B": {"poblacion": 40, "cruce": 0.8, "mutacion": 0.05, "generaciones": 100},
    "C": {"poblacion": 60, "cruce": 0.9, "mutacion": 0.1, "generaciones": 150},
}


# ============================================================
# 2. REPRESENTACIÓN Y EVALUACIÓN DE CROMOSOMAS
# ============================================================
def generar_cromosoma(rng: random.Random) -> list[int]:
    """Cromosoma binario de ocho genes: 1 selecciona, 0 no selecciona."""
    return [rng.randint(0, 1) for _ in PROYECTOS]


def calcular_costo(cromosoma: list[int]) -> int:
    return sum(gene * costo for gene, costo in zip(cromosoma, COSTOS))


def calcular_beneficio(cromosoma: list[int]) -> int:
    return sum(gene * beneficio for gene, beneficio in zip(cromosoma, BENEFICIOS))


def fitness(cromosoma: list[int]) -> int:
    """
    Estrategia explícita:
    - Si el costo supera el presupuesto, fitness = 0 (solución no factible).
    - Si es factible, fitness = beneficio total.
    Como los beneficios son positivos, toda solución factible no vacía
    obtiene fitness positivo y supera a una solución no factible.
    """
    if calcular_costo(cromosoma) > PRESUPUESTO:
        return 0
    return calcular_beneficio(cromosoma)


def proyectos_elegidos(cromosoma: list[int]) -> list[str]:
    return [IDS[i] for i, gene in enumerate(cromosoma) if gene == 1]


# ============================================================
# 3. OPERADORES GENÉTICOS
# ============================================================
def seleccion_torneo(
    poblacion: list[list[int]],
    rng: random.Random,
    tamano_torneo: int = 3,
) -> list[int]:
    """Elige al azar k individuos y devuelve una copia del de mayor fitness."""
    participantes = rng.sample(poblacion, k=min(tamano_torneo, len(poblacion)))
    ganador = max(participantes, key=fitness)
    return ganador.copy()


def cruce_un_punto(
    padre1: list[int],
    padre2: list[int],
    probabilidad_cruce: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Cruza dos cromosomas en un punto interno, si se activa la probabilidad."""
    if rng.random() >= probabilidad_cruce:
        return padre1.copy(), padre2.copy()

    punto = rng.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2


def mutacion_bits(
    cromosoma: list[int],
    probabilidad_mutacion: float,
    rng: random.Random,
) -> list[int]:
    """Cada gen tiene la probabilidad indicada de invertirse: 0↔1."""
    mutado = cromosoma.copy()
    for i in range(len(mutado)):
        if rng.random() < probabilidad_mutacion:
            mutado[i] = 1 - mutado[i]
    return mutado


# ============================================================
# 4. ALGORITMO GENÉTICO
# ============================================================
def ejecutar_algoritmo_genetico(
    poblacion_tamano: int,
    probabilidad_cruce: float,
    probabilidad_mutacion: float,
    generaciones: int,
    semilla: int,
) -> dict:
    rng = random.Random(semilla)
    inicio = time.perf_counter()

    poblacion = [generar_cromosoma(rng) for _ in range(poblacion_tamano)]

    # Si por azar la población no contiene ninguna solución factible,
    # se añade una solución factible sencilla: seleccionar solo P6.
    if max(map(fitness, poblacion), default=0) == 0:
        poblacion[0] = [0, 0, 0, 0, 0, 1, 0, 0]

    historial_mejor = []
    historial_promedio = []

    for _ in range(generaciones):
        valores_fitness = [fitness(individuo) for individuo in poblacion]
        historial_mejor.append(max(valores_fitness))
        historial_promedio.append(sum(valores_fitness) / len(valores_fitness))

        # Elitismo: conservar una copia del mejor individuo de la generación.
        elite = max(poblacion, key=fitness).copy()
        nueva_poblacion = [elite]

        while len(nueva_poblacion) < poblacion_tamano:
            padre1 = seleccion_torneo(poblacion, rng)
            padre2 = seleccion_torneo(poblacion, rng)

            hijo1, hijo2 = cruce_un_punto(
                padre1, padre2, probabilidad_cruce, rng
            )
            hijo1 = mutacion_bits(hijo1, probabilidad_mutacion, rng)
            hijo2 = mutacion_bits(hijo2, probabilidad_mutacion, rng)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < poblacion_tamano:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    # Evaluar también la población final.
    valores_finales = [fitness(individuo) for individuo in poblacion]
    historial_mejor.append(max(valores_finales))
    historial_promedio.append(sum(valores_finales) / len(valores_finales))

    mejor = max(poblacion, key=fitness).copy()
    costo = calcular_costo(mejor)
    beneficio = calcular_beneficio(mejor)

    return {
        "cromosoma": "".join(map(str, mejor)),
        "genes": mejor,
        "proyectos": proyectos_elegidos(mejor),
        "costo": costo,
        "beneficio": beneficio,
        "saldo": PRESUPUESTO - costo,
        "fitness": fitness(mejor),
        "historial_mejor": historial_mejor,
        "historial_promedio": historial_promedio,
        "semilla": semilla,
        "tiempo_segundos": time.perf_counter() - inicio,
    }


# ============================================================
# 5. BÚSQUEDA EXHAUSTIVA: 2^8 = 256 COMBINACIONES
# ============================================================
def busqueda_exhaustiva() -> dict:
    mejor_cromosoma = None
    mejor_beneficio = -1
    mejor_costo = 0

    for numero in range(2 ** len(PROYECTOS)):
        cromosoma = [
            (numero >> desplazamiento) & 1
            for desplazamiento in range(len(PROYECTOS))
        ]
        costo = calcular_costo(cromosoma)
        beneficio = calcular_beneficio(cromosoma)

        if costo <= PRESUPUESTO and beneficio > mejor_beneficio:
            mejor_cromosoma = cromosoma
            mejor_beneficio = beneficio
            mejor_costo = costo

    assert mejor_cromosoma is not None
    return {
        "cromosoma": "".join(map(str, mejor_cromosoma)),
        "genes": mejor_cromosoma,
        "proyectos": proyectos_elegidos(mejor_cromosoma),
        "costo": mejor_costo,
        "beneficio": mejor_beneficio,
        "saldo": PRESUPUESTO - mejor_costo,
        "combinaciones_evaluadas": 2 ** len(PROYECTOS),
    }


# ============================================================
# 6. EXPERIMENTOS: 3 CONFIGURACIONES × 5 SEMILLAS
# ============================================================
def ejecutar_experimentos() -> tuple[pd.DataFrame, pd.DataFrame]:
    filas_detalle = []
    filas_resumen = []
    semillas_base = [11, 23, 37, 49, 61]
    optimo = busqueda_exhaustiva()["beneficio"]

    for nombre, config in CONFIGURACIONES.items():
        resultados = []
        for semilla in semillas_base:
            resultado = ejecutar_algoritmo_genetico(
                poblacion_tamano=config["poblacion"],
                probabilidad_cruce=config["cruce"],
                probabilidad_mutacion=config["mutacion"],
                generaciones=config["generaciones"],
                semilla=semilla,
            )
            resultados.append(resultado)
            filas_detalle.append({
                "Configuración": nombre,
                "Semilla": semilla,
                "Cromosoma ganador": resultado["cromosoma"],
                "Proyectos": ", ".join(resultado["proyectos"]) or "Ninguno",
                "Costo (S/)": resultado["costo"],
                "Beneficio": resultado["beneficio"],
                "Saldo (S/)": resultado["saldo"],
                "Tiempo (s)": round(resultado["tiempo_segundos"], 5),
            })

        beneficios = [r["beneficio"] for r in resultados]
        filas_resumen.append({
            "Configuración": nombre,
            "Población": config["poblacion"],
            "P(cruce)": config["cruce"],
            "P(mutación)": config["mutacion"],
            "Generaciones": config["generaciones"],
            "Mejor beneficio": max(beneficios),
            "Beneficio promedio": sum(beneficios) / len(beneficios),
            "Desviación estándar": pd.Series(beneficios).std(ddof=0),
            "Ejecuciones óptimas": sum(b == optimo for b in beneficios),
            "% alcanza óptimo": 100 * sum(b == optimo for b in beneficios) / len(beneficios),
            "Tiempo total (s)": sum(r["tiempo_segundos"] for r in resultados),
        })

    return pd.DataFrame(filas_detalle), pd.DataFrame(filas_resumen)


# ============================================================
# ============================================================
# 7. INTERFAZ STREAMLIT
# ============================================================
def render():
    """Renderiza el panel del optimizador genético de proyectos."""
    st.title("🧬 Selección de proyectos con presupuesto limitado")
    st.caption(
        "Algoritmo genético con cromosomas binarios, selección por torneo, "
        "cruce de un punto, mutación, elitismo y validación exhaustiva."
    )
    st.metric("Presupuesto máximo", f"S/ {PRESUPUESTO:,.0f}")

    with st.expander("Ver datos de los proyectos", expanded=True):
        df_proyectos = pd.DataFrame(PROYECTOS)
        df_proyectos.columns = ["Proyecto", "Costo (S/)", "Beneficio"]
        st.dataframe(df_proyectos, use_container_width=True, hide_index=True)

    tab_opt, tab_exp, tab_exacto, tab_teoria = st.tabs([
        "🚀 Ejecutar algoritmo genético",
        "📊 Comparar configuraciones",
        "🔎 Óptimo exacto",
        "📘 Explicación y análisis",
    ])

    with tab_opt:
        st.subheader("Configurar una ejecución")
        col1, col2, col3 = st.columns(3)

        with col1:
            nombre_config = st.selectbox("Configuración predefinida", ["A", "B", "C"])
            valores = CONFIGURACIONES[nombre_config]
            poblacion_tamano = st.number_input(
                "Tamaño de población", min_value=20, max_value=300,
                value=valores["poblacion"], step=10
            )
        with col2:
            prob_cruce = st.slider(
                "Probabilidad de cruce", 0.0, 1.0,
                float(valores["cruce"]), 0.05
            )
            prob_mutacion = st.slider(
                "Probabilidad de mutación por gen", 0.0, 1.0,
                float(valores["mutacion"]), 0.01
            )
        with col3:
            generaciones = st.number_input(
                "Generaciones", min_value=50, max_value=1000,
                value=valores["generaciones"], step=10
            )
            semilla = st.number_input("Semilla aleatoria", min_value=0, max_value=999999, value=11)

        if st.button("Ejecutar optimización", type="primary"):
            st.session_state["resultado_ga"] = ejecutar_algoritmo_genetico(
                int(poblacion_tamano), float(prob_cruce), float(prob_mutacion),
                int(generaciones), int(semilla)
            )

        if "resultado_ga" in st.session_state:
            res = st.session_state["resultado_ga"]
            a, b, c = st.columns(3)
            a.metric("Beneficio", str(res["beneficio"]))
            b.metric("Costo total", f"S/ {res['costo']:,.0f}")
            c.metric("Saldo disponible", f"S/ {res['saldo']:,.0f}")

            st.write(f"**Cromosoma ganador:** `{res['cromosoma']}`")
            st.write("**Proyectos elegidos:** " + (", ".join(res["proyectos"]) or "Ninguno"))
            st.write(f"**Tiempo de ejecución:** {res['tiempo_segundos']:.5f} segundos")

            fig, ax = plt.subplots(figsize=(9, 4))
            generaciones_x = list(range(len(res["historial_mejor"])))
            ax.plot(generaciones_x, res["historial_mejor"], label="Mejor fitness")
            ax.plot(generaciones_x, res["historial_promedio"], label="Fitness promedio")
            ax.set_title("Evolución del fitness por generación")
            ax.set_xlabel("Generación")
            ax.set_ylabel("Fitness (beneficio factible)")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

            st.markdown("#### Detalle de los proyectos elegidos")
            df_seleccion = pd.DataFrame([
                {
                    "Proyecto": IDS[i],
                    "Seleccionado": "Sí" if gene else "No",
                    "Costo (S/)": COSTOS[i] if gene else 0,
                    "Beneficio": BENEFICIOS[i] if gene else 0,
                }
                for i, gene in enumerate(res["genes"])
            ])
            st.dataframe(df_seleccion, use_container_width=True, hide_index=True)

    with tab_exp:
        st.subheader("Tres configuraciones, cinco semillas cada una")
        st.write(
            "Se usan las semillas 11, 23, 37, 49 y 61 en cada configuración. "
            "Los tiempos dependen de la computadora y pueden variar."
        )
        if st.button("Ejecutar las 15 pruebas", type="primary"):
            with st.spinner("Ejecutando 15 pruebas del algoritmo genético..."):
                detalle, resumen = ejecutar_experimentos()
                st.session_state["detalle_exp"] = detalle
                st.session_state["resumen_exp"] = resumen

        if "resumen_exp" in st.session_state:
            st.markdown("#### Resumen comparativo")
            st.dataframe(
                st.session_state["resumen_exp"].round(3),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("#### Resultados por semilla")
            st.dataframe(
                st.session_state["detalle_exp"].round(3),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Descargar resultados detallados CSV",
                data=st.session_state["detalle_exp"].to_csv(index=False).encode("utf-8"),
                file_name="resultados_experimentos_geneticos.csv",
                mime="text/csv",
            )
            st.download_button(
                "Descargar resumen CSV",
                data=st.session_state["resumen_exp"].to_csv(index=False).encode("utf-8"),
                file_name="resumen_configuraciones_geneticas.csv",
                mime="text/csv",
            )

    with tab_exacto:
        st.subheader("Búsqueda exhaustiva de todas las soluciones")
        exacto = busqueda_exhaustiva()
        c1, c2, c3 = st.columns(3)
        c1.metric("Combinaciones evaluadas", exacto["combinaciones_evaluadas"])
        c2.metric("Beneficio óptimo", exacto["beneficio"])
        c3.metric("Costo óptimo", f"S/ {exacto['costo']:,.0f}")

        st.write(f"**Cromosoma óptimo:** `{exacto['cromosoma']}`")
        st.write("**Proyectos óptimos:** " + ", ".join(exacto["proyectos"]))
        st.write(f"**Saldo restante:** S/ {exacto['saldo']:,.0f}")
        st.info(
            "La búsqueda exhaustiva prueba las 2⁸ = 256 combinaciones y encuentra "
            "el óptimo global para estos datos. El algoritmo genético es heurístico: "
            "puede encontrarlo, pero no lo garantiza en general."
        )

    with tab_teoria:
        st.subheader("Decisiones de diseño")
        st.markdown("""
        **Representación:** un cromosoma tiene ocho bits, uno por proyecto. El bit 1 significa seleccionado y el bit 0 significa no seleccionado.

        **Fitness y presupuesto:** si el costo supera S/ 20 000, el fitness es 0. Si la solución es factible, su fitness es la suma de los beneficios. Se utiliza esta penalización porque todos los beneficios son positivos; así, una solución factible no vacía tiene mayor fitness que una solución que excede el presupuesto.

        **Selección por torneo:** se eligen al azar tres individuos y se conserva el de mayor fitness como progenitor.

        **Cruce de un punto:** se elige una posición entre los genes y se intercambian las partes posteriores de dos progenitores.

        **Mutación:** cada bit tiene una probabilidad independiente de invertirse (0 pasa a 1 y 1 pasa a 0).

        **Elitismo:** se copia el mejor individuo de la generación a la siguiente sin modificarlo, evitando perder la mejor solución encontrada hasta ese momento.

        **Criterio de parada:** el algoritmo se ejecuta durante el número de generaciones configurado. No se detiene anticipadamente al encontrar el óptimo.
        """)

        st.markdown("#### Preguntas de análisis")
        st.markdown("""
        **¿Cómo afecta la mutación a la convergencia?**
        Una mutación baja ayuda a mantener cambios pequeños; una mutación mayor explora más combinaciones y puede evitar que la población se estanque, pero también puede destruir combinaciones prometedoras.

        **¿Qué aporta el elitismo?**
        Conserva el mejor individuo conocido y evita que se pierda por cruce o mutación. Un elitismo excesivo, si se aplicara a muchos individuos, podría reducir la diversidad.

        **¿Garantiza el algoritmo genético el óptimo global?**
        No. Es un método heurístico y estocástico. En este problema pequeño podemos verificar el óptimo con búsqueda exhaustiva.

        **¿Cuándo sería preferible a la búsqueda exhaustiva?**
        Cuando el número de decisiones crece tanto que enumerar todas las combinaciones resulta costoso. Con ocho proyectos solo hay 256 combinaciones y la búsqueda exhaustiva es sencilla; el algoritmo genético se estudia para comprender su comportamiento y cómo escalar a problemas mayores.
        """)

    st.divider()
    st.caption(
        "Modelo académico. Verifica los resultados, explica las decisiones de diseño "
        "y cita las fuentes utilizadas en el informe."
    )
