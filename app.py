import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Previsão de Desempenho Acadêmico",
    layout="centered"
)

# Carrega o modelo treinado
modelo = joblib.load("modelo_arvore.joblib")

st.title("Previsão de Desempenho Acadêmico")

st.write(
    "Informe os dados do estudante para realizar a previsão da nota final."
)

st.subheader("Dados do estudante")

faltas = st.number_input(
    "Quantidade de faltas",
    min_value=0,
    step=1
)

g2 = st.number_input(
    "Nota do segundo período (G2)",
    min_value=0,
    max_value=20,
    step=1
)

tempo_estudo = st.selectbox(
    "Tempo de estudo semanal",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Menos de 2 horas",
        2: "De 2 a 5 horas",
        3: "De 5 a 10 horas",
        4: "Mais de 10 horas"
    }[x]
)

reprovacoes = st.number_input(
    "Quantidade de reprovações anteriores",
    min_value=0,
    step=1
)

if st.button("Prever nota final"):

    dados_aluno = pd.DataFrame(
        [[faltas, g2, tempo_estudo, reprovacoes]],
        columns=["absences", "G2", "studytime", "failures"]
    )

    previsao = modelo.predict(dados_aluno)[0]

    previsao = max(0, min(20, previsao))

    st.success(
        f"Nota final prevista (G3): {previsao:.1f}"
    )

st.divider()

st.write("Modelo utilizado: Árvore de Decisão.")

st.caption(
    "A previsão é uma estimativa baseada nos dados utilizados no treinamento do modelo."
)