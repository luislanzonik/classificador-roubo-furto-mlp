import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modelo_mlp_roubo_furto.keras"
PREPROCESSOR_FILE = BASE_DIR / "preprocessador.joblib"
METADATA_FILE = BASE_DIR / "metadados_modelo.json"

st.set_page_config(
    page_title="Classificador MLP | Furto e Roubo",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Estilos isolados por componente para evitar conflitos de contraste do tema Streamlit.
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"] {
        background: #f5f8fc !important;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Bloco de cabeçalho: todos os textos internos são explicitamente brancos. */
    div.hero-card {
        background: linear-gradient(120deg, #102a43 0%, #174f78 58%, #28739a 100%);
        border-radius: 20px;
        padding: 2.1rem 2.3rem;
        margin: 0 0 1.25rem 0;
        box-shadow: 0 12px 28px rgba(16, 42, 67, .15);
    }
    div.hero-card .hero-kicker {
        color: #c5e4f5 !important;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 0 0 .65rem 0;
    }
    div.hero-card h1 {
        color: #ffffff !important;
        font-size: clamp(1.9rem, 3vw, 2.55rem);
        font-weight: 850;
        line-height: 1.12;
        margin: 0 0 .7rem 0;
    }
    div.hero-card p {
        color: #f2f8fc !important;
        font-size: 1rem;
        line-height: 1.55;
        margin: 0;
        max-width: 820px;
    }

    /* Blocos azuis secundários com texto branco. */
    div.blue-panel {
        background: #e4eff8;
        border: 1px solid #c6dbea;
        border-left: 5px solid #28739a;
        border-radius: 13px;
        padding: 1rem 1.2rem;
        margin: .8rem 0 1.5rem 0;
    }
    div.blue-panel p, div.blue-panel strong {
        color: #18344d !important;
        margin: 0;
        line-height: 1.55;
    }
    div.section-band {
        background: #174f78;
        border-radius: 12px;
        padding: .8rem 1.1rem;
        margin: 1.3rem 0 .9rem 0;
    }
    div.section-band h2 {
        color: #ffffff !important;
        font-size: 1.18rem;
        font-weight: 800;
        margin: 0;
    }
    div.helper-text p, div.helper-text {
        color: #26384d !important;
    }

    /* Formulário e controles. */
    div[data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid #d8e3ee !important;
        border-radius: 17px;
        padding: 1.25rem 1.35rem .75rem 1.35rem;
        box-shadow: 0 7px 20px rgba(25, 48, 75, .05);
    }
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    label[data-testid="stWidgetLabel"] {
        color: #182b42 !important;
        font-weight: 750 !important;
    }
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #b9cbdc !important;
        border-radius: 9px !important;
        min-height: 2.8rem;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #172b40 !important;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: #123f63 !important;
        color: #ffffff !important;
        border: 1px solid #123f63 !important;
        border-radius: 10px !important;
        min-height: 3.05rem;
        font-weight: 800 !important;
        width: 100%;
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #0d304c !important;
        border-color: #0d304c !important;
    }

    /* Cartões de métricas/resultados. */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #d8e3ee !important;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 6px 18px rgba(25, 48, 75, .045);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #172b40 !important;
    }
    div.result-card {
        background: #ffffff;
        border: 1px solid #d8e3ee;
        border-radius: 15px;
        padding: 1.2rem 1.35rem;
        margin: .7rem 0 1rem 0;
    }
    div.result-card .result-caption {
        color: #52677d !important;
        font-size: .8rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    div.result-card .result-class {
        color: #123f63 !important;
        font-size: 1.8rem;
        font-weight: 850;
        margin-top: .25rem;
    }

    /* Caixas de status próprias: não dependem das cores automáticas de alertas. */
    div.status-box {
        border-radius: 11px;
        padding: .95rem 1.1rem;
        margin: .8rem 0;
        font-weight: 750;
        line-height: 1.5;
    }
    div.status-error {
        background: #fff1f0;
        border: 1px solid #efc2bd;
        color: #7b211b !important;
    }
    div.status-success {
        background: #eaf6ef;
        border: 1px solid #b9dfc7;
        color: #185b32 !important;
    }
    div.status-info {
        background: #e8f1f8;
        border: 1px solid #c8dbea;
        color: #193651 !important;
    }
    div.file-list {
        background: #ffffff;
        border: 1px solid #d8e3ee;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #172b40 !important;
        line-height: 1.8;
    }
    div.file-list code {
        color: #123f63 !important;
        background: #edf3f8 !important;
        padding: .12rem .35rem;
        border-radius: 5px;
    }
    div[data-testid="stCaptionContainer"] p,
    .stCaption, small {
        color: #52677d !important;
    }
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #d8e3ee;
        border-radius: 12px;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p {
        color: #172b40 !important;
        font-weight: 700;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        color: #172b40;
    }
    hr { border-color: #d8e3ee !important; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
      <div class="hero-kicker">Projeto acadêmico · Aprendizado profundo</div>
      <h1>Classificador de Ocorrências</h1>
      <p>Rede neural MLP para classificar registros de subtração de celulares em São Paulo (2025)
      como Furto ou Roubo, considerando as características informadas.</p>
    </div>
    <div class="blue-panel">
      <p><strong>Escopo do modelo:</strong> a ferramenta classifica uma ocorrência já registrada
      ou hipotética. Ela não prevê onde ou quando um crime acontecerá e não calcula risco
      criminal de bairros ou pessoas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Carregando a rede neural e o pré-processador...")
def load_artifacts():
    paths = (MODEL_FILE, PREPROCESSOR_FILE, METADATA_FILE)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Arquivos ausentes: " + ", ".join(missing))

    model = tf.keras.models.load_model(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, preprocessor, metadata


def read_categories(preprocessor, features):
    transformer = preprocessor.named_transformers_.get("categoricas")
    if transformer is None:
        raise ValueError("Transformador 'categoricas' não encontrado no pré-processador.")
    encoder = transformer.named_steps.get("onehot")
    if encoder is None or not hasattr(encoder, "categories_"):
        raise ValueError("OneHotEncoder ajustado não encontrado.")
    return {
        feature: [str(value) for value in values]
        for feature, values in zip(features, encoder.categories_)
    }


try:
    model, preprocessor, metadata = load_artifacts()
    features = metadata.get(
        "features",
        ["BAIRRO", "DESCR_PERIODO", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"],
    )
    required_features = [
        "BAIRRO", "DESCR_PERIODO", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"
    ]
    if features != required_features:
        raise ValueError("Os atributos dos metadados não correspondem aos campos do formulário.")
    categories = read_categories(preprocessor, features)
except Exception as error:
    st.markdown(
        '<div class="status-box status-error">Não foi possível carregar o modelo. '
        'Confira se os três arquivos exportados pelo Colab estão na mesma pasta de app.py.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="file-list">
          <strong>Arquivos necessários</strong><br>
          • <code>modelo_mlp_roubo_furto.keras</code><br>
          • <code>preprocessador.joblib</code><br>
          • <code>metadados_modelo.json</code><br><br>
          Depois de copiar os arquivos, atualize esta página.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Detalhes técnicos"):
        st.write(str(error))
    st.stop()

st.markdown(
    """
    <div class="section-band"><h2>1 · Características da ocorrência</h2></div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="helper-text">Selecione os mesmos quatro atributos utilizados no treinamento da MLP.</div>',
    unsafe_allow_html=True,
)

with st.form("form_classificacao", clear_on_submit=False):
    left, right = st.columns(2, gap="large")
    selected = {}
    with left:
        selected["BAIRRO"] = st.selectbox("Bairro", categories["BAIRRO"])
        selected["DESCR_PERIODO"] = st.selectbox(
            "Período", categories["DESCR_PERIODO"]
        )
    with right:
        selected["DESCR_TIPOLOCAL"] = st.selectbox(
            "Tipo de local", categories["DESCR_TIPOLOCAL"]
        )
        selected["DESCR_SUBTIPOLOCAL"] = st.selectbox(
            "Subtipo de local", categories["DESCR_SUBTIPOLOCAL"]
        )
    submitted = st.form_submit_button("Classificar ocorrência")

if submitted:
    try:
        input_df = pd.DataFrame(
            [{feature: selected[feature] for feature in features}],
            columns=features,
        )
        transformed = preprocessor.transform(input_df).astype("float32")
        prediction = np.asarray(model.predict(transformed, verbose=0)).reshape(-1)
        if prediction.size != 1 or not np.isfinite(prediction[0]):
            raise ValueError("A MLP não retornou uma probabilidade válida.")

        probability_robbery = float(np.clip(prediction[0], 0.0, 1.0))
        probability_theft = 1.0 - probability_robbery
        threshold = float(metadata.get("limiar_classificacao", 0.5))
        predicted_class = "Roubo" if probability_robbery >= threshold else "Furto"

        st.markdown(
            '<div class="section-band"><h2>2 · Resultado da classificação</h2></div>',
            unsafe_allow_html=True,
        )
        if predicted_class == "Roubo":
            st.markdown(
                '<div class="status-box status-error">Classe estimada pela MLP: ROUBO</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-box status-success">Classe estimada pela MLP: FURTO</div>',
                unsafe_allow_html=True,
            )

        metric_left, metric_right = st.columns(2, gap="large")
        metric_left.metric("Probabilidade estimada de Furto", f"{probability_theft:.2%}")
        metric_right.metric("Probabilidade estimada de Roubo", f"{probability_robbery:.2%}")
        st.progress(probability_robbery)
        st.caption(
            f"Limiar de decisão: {threshold:.2f}. A classe Roubo é atribuída quando "
            "a probabilidade estimada atinge ou supera esse limiar."
        )

        with st.expander("Ver características enviadas ao modelo"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="blue-panel">
              <p><strong>Interpretação:</strong> os percentuais representam a saída do
              classificador para os atributos selecionados. Não representam a chance de um
              crime acontecer naquele bairro, período ou tipo de local.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as error:
        st.markdown(
            '<div class="status-box status-error">Não foi possível concluir a classificação.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Detalhes técnicos"):
            st.exception(error)

with st.expander("Sobre o modelo e suas limitações"):
    st.write(
        f"**Arquitetura selecionada:** "
        f"{metadata.get('arquitetura_escolhida', 'Não informada')}"
    )
    st.write("**Classes:** 0 = Furto · 1 = Roubo")
    st.write("**Atributos:** " + ", ".join(features))
    st.write("**Base:** SSP-SP, registros de celulares subtraídos, ano de 2025.")
    st.write(
        "Os resultados são experimentais e dependem da qualidade, representatividade "
        "e recorte dos dados usados no treinamento."
    )
    st.write(
        "A ferramenta não deve ser usada isoladamente para decisões policiais, "
        "investigativas ou sobre indivíduos."
    )

st.markdown("---")
st.caption(
    "Protótipo acadêmico · MLP para classificação de registros · "
    "Não é ferramenta de previsão de crimes."
)
