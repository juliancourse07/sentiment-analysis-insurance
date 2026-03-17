"""
🎯 Sistema de Análisis de Sentimientos para Aseguradora
Aplicación web interactiva con Streamlit para analizar feedback de clientes.
Modelo: BETO (finiteautomata/beto-sentiment-analysis) - BERT en español fine-tuned
"""

import re
import io
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter

warnings.filterwarnings("ignore")

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Sentimientos | Aseguradora",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS personalizados ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main { background-color: #f5f7fa; }
        .block-container { padding-top: 1.5rem; }
        .metric-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            text-align: center;
        }
        .sentiment-positive { color: #00B050; font-weight: 700; }
        .sentiment-negative { color: #FF0000; font-weight: 700; }
        .sentiment-neutral  { color: #FFC000; font-weight: 700; }
        .sentiment-mixed    { color: #7030A0; font-weight: 700; }
        h1 { color: #1f4788; }
        h2 { color: #1f4788; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 6px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constantes ─────────────────────────────────────────────────────────────────
GOOGLE_SHEETS_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1OUzUl5UDrZEfBSaW4afk-Nzazs7gizes3VkNfXXuKmE/export?format=csv&gid=1726674730"
)

TARGET_ATTRIBUTES = [
    "Autos, ¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?",
    "Fianzas, ¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?  \n2",
    "Generales, ¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?  \n3",
    "Soat, ¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?  \n5",
    "Vida, ¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?  \n4",
]

ATTRIBUTE_LABELS = {
    TARGET_ATTRIBUTES[0]: "Autos",
    TARGET_ATTRIBUTES[1]: "Fianzas",
    TARGET_ATTRIBUTES[2]: "Generales",
    TARGET_ATTRIBUTES[3]: "Soat",
    TARGET_ATTRIBUTES[4]: "Vida",
}

SENTIMENT_COLORS = {
    "POSITIVO": "#00B050",
    "NEGATIVO": "#FF0000",
    "NEUTRAL": "#FFC000",
    "MIXTO": "#7030A0",
}

SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "un", "por", "con", "una", "su", "para", "es", "al", "lo", "como",
    "más", "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "si",
    "sobre", "entre", "cuando", "muy", "sin", "ser", "hay", "también",
    "me", "hasta", "desde", "nos", "durante", "uno", "ni", "contra",
    "ese", "esto", "mí", "antes", "bien", "gran", "poco", "pues", "era",
    "son", "donde", "todo", "porque", "aunque", "tan", "así", "has",
    "mi", "te", "tu", "este", "eso", "esta", "esto", "esa",
}


# ── Clase principal de análisis ────────────────────────────────────────────────
class SentimentAnalyzer:
    """Analizador híbrido: modelo BETO + keywords del sector asegurador."""

    KEYWORDS_POSITIVE = {
        "fácil", "rapido", "rápido", "claro", "eficiente", "amable",
        "excelente", "bueno", "satisfecho", "agil", "ágil", "simple",
        "intuitivo", "practico", "práctico", "completo", "preciso",
        "confiable", "util", "útil", "genial", "perfecto", "bien",
        "buena", "buenos", "buenas", "facil",
    }

    KEYWORDS_NEGATIVE = {
        "difícil", "dificil", "lento", "complicado", "confuso",
        "problema", "error", "demora", "malo", "deficiente",
        "insatisfecho", "engorroso", "complejo", "tedioso",
        "frustrante", "ineficiente", "pésimo", "pesimo", "mala",
        "malos", "malas", "problemas", "errores",
    }

    KEYWORDS_NEUTRAL = {
        "proceso", "tramite", "trámite", "documentacion", "documentación",
        "requisito", "procedimiento", "normal", "estandar", "estándar",
        "regular", "comun", "común",
    }

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    @st.cache_resource(show_spinner=False)
    def load_model(_self):  # noqa: N805 — underscore prefix required by st.cache_resource to skip hashing 'self'
        """Carga el modelo BETO con cache de Streamlit."""
        try:
            from transformers import pipeline
            classifier = pipeline(
                "text-classification",
                model="finiteautomata/beto-sentiment-analysis",
                tokenizer="finiteautomata/beto-sentiment-analysis",
                truncation=True,
                max_length=512,
            )
            return classifier
        except Exception as exc:
            st.warning(
                f"⚠️ No se pudo cargar el modelo BETO: {exc}. "
                "Se usará análisis basado en keywords."
            )
            return None

    def preprocess_text(self, text: str) -> str:
        """Limpia y normaliza el texto."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"\+?\d[\d\s\-]{7,}\d", " ", text)
        text = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ.,;:!?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _keyword_score(self, text: str):
        """Retorna (pos_count, neg_count, neu_count, dominant_label)."""
        words = set(text.split())
        pos = len(words & self.KEYWORDS_POSITIVE)
        neg = len(words & self.KEYWORDS_NEGATIVE)
        neu = len(words & self.KEYWORDS_NEUTRAL)
        return pos, neg, neu

    def _label_from_keywords(self, pos: int, neg: int):
        if pos > 0 and neg > 0:
            return "MIXTO"
        if pos > neg:
            return "POSITIVO"
        if neg > pos:
            return "NEGATIVO"
        return "NEUTRAL"

    def _beto_to_standard(self, beto_label: str) -> str:
        mapping = {"POS": "POSITIVO", "NEG": "NEGATIVO", "NEU": "NEUTRAL"}
        return mapping.get(beto_label.upper(), "NEUTRAL")

    def analyze_sentiment(self, text: str, classifier=None):
        """
        Análisis híbrido.

        Returns dict with keys:
            sentiment, score, confidence, keywords_pos, keywords_neg
        """
        clean = self.preprocess_text(text)
        if not clean:
            return {
                "sentiment": "NEUTRAL",
                "score": 0.0,
                "confidence": 0.0,
                "keywords_pos": 0,
                "keywords_neg": 0,
            }

        pos, neg, _ = self._keyword_score(clean)

        model_label = None
        model_confidence = 0.0

        if classifier is not None:
            try:
                result = classifier(clean[:512])[0]
                model_label = self._beto_to_standard(result["label"])
                model_confidence = float(result["score"])
            except Exception:
                pass

        # Hybrid decision
        if model_label is not None:
            if pos > 0 and neg > 0:
                final_label = "MIXTO"
                final_conf = model_confidence * 0.8
            elif pos > 0 and model_label == "POSITIVO":
                final_label = "POSITIVO"
                final_conf = min(model_confidence * 1.1, 1.0)
            elif neg > 0 and model_label == "NEGATIVO":
                final_label = "NEGATIVO"
                final_conf = min(model_confidence * 1.1, 1.0)
            else:
                final_label = model_label
                final_conf = model_confidence
        else:
            final_label = self._label_from_keywords(pos, neg)
            # Heuristic confidence based on keyword counts
            total_kw = pos + neg
            final_conf = min(0.5 + total_kw * 0.1, 0.95) if total_kw > 0 else 0.5

        score_map = {"POSITIVO": 1.0, "NEGATIVO": -1.0, "NEUTRAL": 0.0, "MIXTO": 0.5}
        score = score_map.get(final_label, 0.0)

        return {
            "sentiment": final_label,
            "score": score,
            "confidence": round(final_conf, 4),
            "keywords_pos": pos,
            "keywords_neg": neg,
        }


# ── Helpers de datos ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_from_google_sheets(url: str) -> pd.DataFrame:
    """Descarga datos desde Google Sheets exportado como CSV."""
    df = pd.read_csv(url)
    return df


def filter_open_responses(df: pd.DataFrame, attributes: list) -> pd.DataFrame:
    """
    Filtra el DataFrame pivoteado para quedarse sólo con los atributos
    de respuestas abiertas seleccionados.
    """
    col_attr = None
    col_val = None

    for col in df.columns:
        low = col.strip().lower()
        if low in ("atributo", "attribute"):
            col_attr = col
        if low in ("valor", "value"):
            col_val = col

    if col_attr is None or col_val is None:
        # Fallback: assume first two cols are Atributo / Valor
        col_attr, col_val = df.columns[0], df.columns[1]

    filtered = df[df[col_attr].isin(attributes)].copy()
    filtered = filtered.rename(columns={col_attr: "Atributo", col_val: "Valor"})
    filtered = filtered[filtered["Valor"].notna() & (filtered["Valor"].astype(str).str.strip() != "")]
    filtered["linea_negocio"] = filtered["Atributo"].map(ATTRIBUTE_LABELS)

    # Try to extract date if available
    date_cols = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()]
    if date_cols:
        filtered["fecha"] = pd.to_datetime(df.loc[filtered.index, date_cols[0]], errors="coerce")

    filtered = filtered.reset_index(drop=True)
    return filtered


def generate_sample_data(n: int = 50) -> pd.DataFrame:
    """Genera datos sintéticos para demostración."""
    rng = np.random.default_rng(42)
    lines = list(ATTRIBUTE_LABELS.values())
    comments_pos = [
        "El proceso fue muy fácil y rápido, excelente servicio",
        "Todo claro y eficiente, muy satisfecho con la atención",
        "La plataforma es intuitiva y práctica, perfecto",
        "Amable atención y proceso ágil, muy buena experiencia",
        "Simple y completo, confiable como siempre",
    ]
    comments_neg = [
        "El proceso fue difícil y muy lento, muchos problemas",
        "Complicado y confuso, demasiados errores en el sistema",
        "Frustrante experiencia, proceso engorroso e ineficiente",
        "Pésimo servicio, mucha demora y mala atención",
        "Tedioso trámite con muchas complicaciones",
    ]
    comments_neu = [
        "El proceso de documentación cumplió los requisitos estándar",
        "Trámite normal con los procedimientos regulares",
        "Proceso común dentro de los estándares esperados",
        "Documentación requerida dentro del procedimiento habitual",
    ]
    comments_mix = [
        "Fácil de iniciar pero luego complicado en los últimos pasos",
        "Buen servicio aunque con algunos problemas al final",
        "Rápido al principio pero lento en la resolución",
    ]
    all_comments = comments_pos * 10 + comments_neg * 8 + comments_neu * 5 + comments_mix * 4
    rows = []
    base_date = datetime(2024, 1, 1)
    for i in range(n):
        attr = TARGET_ATTRIBUTES[rng.integers(0, len(TARGET_ATTRIBUTES))]
        comment = all_comments[rng.integers(0, len(all_comments))]
        rows.append({
            "Atributo": attr,
            "Valor": comment,
            "linea_negocio": ATTRIBUTE_LABELS[attr],
            "fecha": base_date + timedelta(days=int(rng.integers(0, 180))),
        })
    return pd.DataFrame(rows)



# ── Renderizado de tabs ────────────────────────────────────────────────────────
def render_tab_dashboard(df: pd.DataFrame):
    st.subheader("📊 Dashboard General")

    total = len(df)
    pct_pos = (df["sentiment"] == "POSITIVO").mean() * 100
    pct_neg = (df["sentiment"] == "NEGATIVO").mean() * 100
    avg_score = df["score"].mean()
    avg_conf = df["confidence"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📋 Total respuestas", f"{total:,}")
    with c2:
        st.metric("😊 % Positivos", f"{pct_pos:.1f}%", delta=f"{pct_pos - 50:.1f}pp")
    with c3:
        st.metric(
            "😞 % Negativos",
            f"{pct_neg:.1f}%",
            delta=f"{-(pct_neg - 50):.1f}pp",
            delta_color="inverse",
        )
    with c4:
        st.metric("⭐ Score promedio", f"{avg_score:.2f}")
    with c5:
        st.metric("🎯 Confianza modelo", f"{avg_conf:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Dona
    with col_left:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentimiento", "Cantidad"]
        fig_pie = px.pie(
            counts,
            names="Sentimiento",
            values="Cantidad",
            hole=0.45,
            color="Sentimiento",
            color_discrete_map=SENTIMENT_COLORS,
            title="Distribución de Sentimientos",
        )
        fig_pie.update_layout(legend_title_text="Sentimiento")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Histograma confianza
    with col_right:
        avg_c = df["confidence"].mean()
        fig_hist = px.histogram(
            df,
            x="confidence",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            nbins=20,
            title="Distribución de Confianza del Modelo",
            labels={"confidence": "Confianza", "count": "Frecuencia"},
        )
        fig_hist.add_vline(
            x=avg_c,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Promedio: {avg_c:.2%}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Barras apiladas por línea de negocio
    if "linea_negocio" in df.columns:
        pivot = (
            df.groupby(["linea_negocio", "sentiment"])
            .size()
            .reset_index(name="n")
        )
        pivot["pct"] = pivot.groupby("linea_negocio")["n"].transform(
            lambda x: x / x.sum() * 100
        )
        fig_bar = px.bar(
            pivot,
            x="linea_negocio",
            y="pct",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            title="Sentimientos por Línea de Negocio (%)",
            labels={"linea_negocio": "Línea", "pct": "Porcentaje (%)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tendencia temporal
    if "fecha" in df.columns and df["fecha"].notna().any():
        df_time = df.dropna(subset=["fecha"]).copy()
        df_time["mes"] = df_time["fecha"].dt.to_period("M").astype(str)
        time_pivot = (
            df_time.groupby(["mes", "sentiment"])
            .size()
            .reset_index(name="n")
        )
        fig_line = px.line(
            time_pivot,
            x="mes",
            y="n",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            markers=True,
            title="Tendencia de Sentimientos por Mes",
            labels={"mes": "Mes", "n": "Cantidad"},
        )
        st.plotly_chart(fig_line, use_container_width=True)


def render_tab_detailed(df: pd.DataFrame):
    st.subheader("🔍 Análisis Detallado")

    # Mapa de calor
    if "linea_negocio" in df.columns:
        heat_data = (
            df.groupby(["linea_negocio", "sentiment"])
            .size()
            .unstack(fill_value=0)
        )
        fig_heat = px.imshow(
            heat_data,
            color_continuous_scale="Blues",
            title="Mapa de Calor: Sentimientos por Línea de Negocio",
            labels={"x": "Sentimiento", "y": "Línea de Negocio", "color": "Cantidad"},
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Resumen por Sentimiento**")
        summary_sent = (
            df.groupby("sentiment")
            .agg(
                cantidad=("sentiment", "count"),
                confianza_prom=("confidence", "mean"),
                score_prom=("score", "mean"),
            )
            .round(3)
            .reset_index()
        )
        summary_sent.columns = ["Sentimiento", "Cantidad", "Confianza Prom.", "Score Prom."]
        st.dataframe(summary_sent, use_container_width=True, hide_index=True)

    with col2:
        if "linea_negocio" in df.columns:
            st.markdown("**Resumen por Línea de Negocio**")
            summary_ln = (
                df.groupby("linea_negocio")
                .agg(
                    cantidad=("sentiment", "count"),
                    pct_pos=("sentiment", lambda x: (x == "POSITIVO").mean() * 100),
                    pct_neg=("sentiment", lambda x: (x == "NEGATIVO").mean() * 100),
                )
                .round(1)
                .reset_index()
            )
            summary_ln.columns = ["Línea", "Cantidad", "% Positivo", "% Negativo"]
            st.dataframe(summary_ln, use_container_width=True, hide_index=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        total_kw_pos = df["keywords_pos"].sum()
        total_kw_neg = df["keywords_neg"].sum()
        kw_fig = go.Figure(
            go.Bar(
                x=["Keywords Positivas", "Keywords Negativas"],
                y=[total_kw_pos, total_kw_neg],
                marker_color=[SENTIMENT_COLORS["POSITIVO"], SENTIMENT_COLORS["NEGATIVO"]],
            )
        )
        kw_fig.update_layout(title="Total Keywords Detectadas", showlegend=False)
        st.plotly_chart(kw_fig, use_container_width=True)

    with col4:
        st.markdown("**Resumen por Atributo**")
        summary_attr = (
            df.groupby("Atributo")
            .agg(cantidad=("sentiment", "count"))
            .reset_index()
        )
        summary_attr["Atributo"] = summary_attr["Atributo"].map(
            lambda a: ATTRIBUTE_LABELS.get(a, a)
        )
        summary_attr.columns = ["Atributo", "Cantidad"]
        st.dataframe(summary_attr, use_container_width=True, hide_index=True)


def render_tab_comments(df: pd.DataFrame):
    st.subheader("💬 Comentarios")

    col1, col2, col3 = st.columns(3)
    with col1:
        sents = st.multiselect(
            "Filtrar por sentimiento",
            options=sorted(df["sentiment"].unique()),
            default=sorted(df["sentiment"].unique()),
        )
    with col2:
        lines = sorted(df["linea_negocio"].unique()) if "linea_negocio" in df.columns else []
        selected_lines = st.multiselect(
            "Filtrar por línea de negocio",
            options=lines,
            default=lines,
        )
    with col3:
        min_conf = st.slider("Confianza mínima (%)", 0, 100, 0) / 100

    mask = df["sentiment"].isin(sents) & (df["confidence"] >= min_conf)
    if selected_lines:
        mask &= df["linea_negocio"].isin(selected_lines)

    filtered = df[mask].head(50)
    st.markdown(f"**Mostrando {len(filtered)} de {mask.sum()} comentarios filtrados**")

    for _, row in filtered.iterrows():
        text_preview = str(row["Valor"])[:80] + ("…" if len(str(row["Valor"])) > 80 else "")
        sent = row["sentiment"]
        color_class = f"sentiment-{sent.lower()}"
        with st.expander(f"[{ATTRIBUTE_LABELS.get(row['Atributo'], row['Atributo'])}] {text_preview}"):
            st.markdown(f"**Texto completo:** {row['Valor']}")
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"**Sentimiento:** <span class='{color_class}'>{sent}</span>",
                unsafe_allow_html=True,
            )
            c2.markdown(f"**Score:** `{row['score']:.2f}`")
            c3.markdown(f"**Confianza:** `{row['confidence']:.2%}`")
            if "linea_negocio" in df.columns:
                st.markdown(f"**Línea de negocio:** {row['linea_negocio']}")


def render_tab_keywords(df: pd.DataFrame):
    st.subheader("☁️ Palabras Clave")

    sent_filter = st.selectbox(
        "Filtrar nube por sentimiento",
        options=["Todos"] + sorted(df["sentiment"].unique()),
    )

    subset = df if sent_filter == "Todos" else df[df["sentiment"] == sent_filter]

    all_text = " ".join(subset["Valor"].dropna().astype(str).tolist()).lower()
    words = re.findall(r"\b[a-záéíóúüñ]{3,}\b", all_text)
    filtered_words = [w for w in words if w not in SPANISH_STOPWORDS]
    word_freq = Counter(filtered_words).most_common(50)

    if not word_freq:
        st.info("No hay suficientes palabras para generar visualización.")
        return

    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wc_color = (
            SENTIMENT_COLORS.get(sent_filter, "#1f4788")
            if sent_filter != "Todos"
            else "#1f4788"
        )
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="Blues",
            max_words=50,
            collocations=False,
        ).generate_from_frequencies(dict(word_freq))

        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
        plt.close(fig_wc)
    except ImportError:
        st.info("WordCloud no disponible. Mostrando frecuencias en gráfico.")

    # Top 20 bar chart
    top20 = word_freq[:20]
    words_list = [w[0] for w in top20]
    freqs = [w[1] for w in top20]

    fig_bar = go.Figure(
        go.Bar(
            y=words_list[::-1],
            x=freqs[::-1],
            orientation="h",
            marker_color=SENTIMENT_COLORS.get(sent_filter, "#1f4788")
            if sent_filter != "Todos"
            else "#1f4788",
        )
    )
    fig_bar.update_layout(
        title="Top 20 Palabras más Frecuentes",
        xaxis_title="Frecuencia",
        yaxis_title="Palabra",
        height=500,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def render_tab_export(df: pd.DataFrame):
    st.subheader("📥 Exportar Datos")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2 = st.columns(2)

    with col1:
        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ Descargar CSV (UTF-8 BOM)",
            data=csv_bytes,
            file_name=f"sentimientos_{timestamp}.csv",
            mime="text/csv",
        )

    with col2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Datos_Completos", index=False)

            summary_sent = (
                df.groupby("sentiment")
                .agg(
                    cantidad=("sentiment", "count"),
                    confianza_prom=("confidence", "mean"),
                    score_prom=("score", "mean"),
                )
                .round(3)
                .reset_index()
            )
            summary_sent.to_excel(writer, sheet_name="Resumen", index=False)

            if "linea_negocio" in df.columns:
                ln_summary = (
                    df.groupby("linea_negocio")
                    .agg(
                        cantidad=("sentiment", "count"),
                        pct_pos=("sentiment", lambda x: (x == "POSITIVO").mean() * 100),
                        pct_neg=("sentiment", lambda x: (x == "NEGATIVO").mean() * 100),
                    )
                    .round(1)
                    .reset_index()
                )
                ln_summary.to_excel(writer, sheet_name="Por_Linea_Negocio", index=False)

            attr_summary = (
                df.groupby("Atributo")
                .agg(cantidad=("sentiment", "count"))
                .reset_index()
            )
            attr_summary.to_excel(writer, sheet_name="Por_Atributo", index=False)

        excel_data = output.getvalue()
        st.download_button(
            label="⬇️ Descargar Excel (múltiples hojas)",
            data=excel_data,
            file_name=f"sentimientos_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")
    st.markdown("**Vista previa de datos**")
    st.dataframe(df, use_container_width=True)


# ── Pantalla de bienvenida ─────────────────────────────────────────────────────
def render_welcome():
    st.markdown(
        """
        <h1 style='color:#1f4788; text-align:center;'>
            🎯 Sistema de Análisis de Sentimientos
        </h1>
        <h3 style='color:#555; text-align:center;'>Aseguradora — Powered by BETO (BERT en Español)</h3>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            ### 🤖 Características del sistema
            - Modelo **BETO** fine-tuned en español
            - Análisis **híbrido**: modelo + keywords
            - Categorías: **POSITIVO, NEGATIVO, NEUTRAL, MIXTO**
            - Keywords especializadas del sector asegurador
            """
        )
    with col2:
        st.markdown(
            """
            ### 📊 Visualizaciones disponibles
            - Gráfico de dona de distribución
            - Mapa de calor por línea de negocio
            - Tendencia temporal
            - Nube de palabras interactiva
            - Top 20 palabras frecuentes
            """
        )
    with col3:
        st.markdown(
            """
            ### 📥 Opciones de exportación
            - **CSV** con encoding UTF-8 BOM (listo para Excel/Power BI)
            - **Excel** con múltiples hojas de análisis
            - Timestamp en nombre de archivo
            - Columnas calculadas incluidas
            """
        )

    st.markdown("---")
    st.markdown(
        """
        ### 🚀 Instrucciones de uso
        1. En el **panel lateral**, selecciona la fuente de datos.
        2. Elige entre Google Sheets, subir CSV o datos de ejemplo.
        3. Selecciona los atributos a analizar.
        4. Haz clic en **🔍 Analizar Sentimientos**.
        5. Explora los resultados en las 5 pestañas.
        6. Descarga los resultados desde la pestaña **Exportar**.
        """
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/analytics.png",
            width=80,
        )
        st.title("⚙️ Configuración")
        st.markdown("---")

        source = st.radio(
            "Fuente de datos",
            ["🔗 Google Sheets", "📂 Subir CSV", "🧪 Datos de ejemplo"],
            index=0,
        )

        df_raw = None
        if source == "🔗 Google Sheets":
            url_input = st.text_input(
                "URL de Google Sheets",
                value="https://docs.google.com/spreadsheets/d/1OUzUl5UDrZEfBSaW4afk-Nzazs7gizes3VkNfXXuKmE/edit?gid=1726674730",
            )
            if st.button("📥 Cargar desde Google Sheets"):
                # Convert edit URL to export URL
                export_url = re.sub(
                    r"/edit.*",
                    "/export?format=csv",
                    url_input,
                )
                gid_match = re.search(r"gid=(\d+)", url_input)
                if gid_match:
                    export_url += f"&gid={gid_match.group(1)}"
                try:
                    with st.spinner("Descargando datos…"):
                        df_raw = load_from_google_sheets(export_url)
                    st.success(f"✅ {len(df_raw):,} filas cargadas")
                    st.session_state["df_raw"] = df_raw
                except Exception as exc:
                    st.error(f"❌ Error al cargar: {exc}")

        elif source == "📂 Subir CSV":
            uploaded = st.file_uploader("Subir archivo CSV", type=["csv"])
            if uploaded is not None:
                try:
                    df_raw = pd.read_csv(uploaded, encoding="utf-8-sig")
                    st.success(f"✅ {len(df_raw):,} filas cargadas")
                    st.session_state["df_raw"] = df_raw
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")

        else:  # Datos de ejemplo
            if st.button("🧪 Generar datos de ejemplo"):
                df_raw = generate_sample_data(50)
                st.success("✅ 50 filas de ejemplo generadas")
                st.session_state["df_raw"] = df_raw

        st.markdown("---")
        # Attribute selection
        selected_attrs = st.multiselect(
            "Atributos a analizar",
            options=TARGET_ATTRIBUTES,
            default=TARGET_ATTRIBUTES,
            format_func=lambda a: ATTRIBUTE_LABELS.get(a, a),
        )

        st.markdown("---")
        analyze_btn = st.button("🔍 Analizar Sentimientos", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            "<small>Modelo: BETO (finiteautomata/beto-sentiment-analysis)</small>",
            unsafe_allow_html=True,
        )

    return selected_attrs, analyze_btn


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    selected_attrs, analyze_btn = render_sidebar()

    df_raw = st.session_state.get("df_raw", None)
    df_results = st.session_state.get("df_results", None)

    if df_raw is None and not analyze_btn:
        render_welcome()
        return

    if analyze_btn:
        if df_raw is None:
            st.warning("⚠️ Primero carga los datos desde el panel lateral.")
            return

        # Filter relevant attributes
        if "Atributo" in df_raw.columns or "atributo" in df_raw.columns.str.lower().tolist():
            df_filtered = filter_open_responses(df_raw, selected_attrs)
        else:
            # Assume it's already in plain format (CSV upload with Valor column)
            df_filtered = df_raw.copy()
            if "Valor" not in df_filtered.columns:
                df_filtered["Valor"] = df_filtered.iloc[:, 0]
            if "Atributo" not in df_filtered.columns:
                df_filtered["Atributo"] = "General"
            if "linea_negocio" not in df_filtered.columns:
                df_filtered["linea_negocio"] = "General"

        if df_filtered.empty:
            st.error("❌ No se encontraron respuestas con los atributos seleccionados.")
            return

        analyzer = SentimentAnalyzer()

        st.info("🤖 Cargando modelo BETO…")
        with st.spinner("Cargando modelo de lenguaje…"):
            classifier = analyzer.load_model()

        results = []
        progress = st.progress(0, text="Analizando sentimientos…")
        total = len(df_filtered)

        for idx, (_, row) in enumerate(df_filtered.iterrows()):
            res = analyzer.analyze_sentiment(str(row["Valor"]), classifier)
            results.append(res)
            if idx % 5 == 0 or idx == total - 1:
                progress.progress((idx + 1) / total, text=f"Analizando {idx + 1}/{total}…")

        progress.empty()

        df_results = df_filtered.copy()
        for key in ["sentiment", "score", "confidence", "keywords_pos", "keywords_neg"]:
            df_results[key] = [r[key] for r in results]

        st.session_state["df_results"] = df_results
        st.success(f"✅ Análisis completado: {total:,} respuestas procesadas")

    if df_results is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Dashboard General",
                "🔍 Análisis Detallado",
                "💬 Comentarios",
                "☁️ Palabras Clave",
                "📥 Exportar",
            ]
        )
        with tab1:
            render_tab_dashboard(df_results)
        with tab2:
            render_tab_detailed(df_results)
        with tab3:
            render_tab_comments(df_results)
        with tab4:
            render_tab_keywords(df_results)
        with tab5:
            render_tab_export(df_results)
    elif df_raw is not None:
        st.info("👈 Haz clic en **🔍 Analizar Sentimientos** para iniciar el análisis.")
    else:
        render_welcome()


if __name__ == "__main__":
    main()
