"""
🎯 Sistema de Análisis de Sentimientos para Aseguradora
Aplicación web interactiva con Streamlit para analizar feedback de clientes.
Modelo: BETO (finiteautomata/beto-sentiment-analysis) - BERT en español fine-tuned
IA Contextual: Groq (Llama 3.1 70B) especializada en el sector asegurador colombiano
"""

import os
import re
import io
import time
import warnings
from datetime import datetime, timedelta
from collections import Counter

import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de Sentimientos | Aseguradora",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS Premium ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        * { font-family: 'Poppins', sans-serif !important; }

        .main { background-color: #f0f4ff; }
        .block-container { padding-top: 1.5rem; }

        /* Metric cards */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
            border: 1px solid rgba(102, 126, 234, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="metric-container"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 28px rgba(102, 126, 234, 0.25);
        }

        /* Sentiment classes */
        .sentiment-positivo { color: #10b981; font-weight: 700; }
        .sentiment-negativo { color: #ef4444; font-weight: 700; }
        .sentiment-neutral  { color: #f59e0b; font-weight: 700; }
        .sentiment-mixto    { color: #8b5cf6; font-weight: 700; }

        /* Headings */
        h1 {
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            text-align: center;
        }
        h2, h3 { color: #1f4788; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(102, 126, 234, 0.05);
            padding: 8px;
            border-radius: 14px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
            padding: 8px 18px;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
        }
        [data-testid="stSidebar"] * { color: white !important; }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stRadio label { color: #bfdbfe !important; }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            transform: scale(1.03) !important;
            box-shadow: 0 6px 18px rgba(102, 126, 234, 0.5) !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background: rgba(102, 126, 234, 0.08);
            border-radius: 10px;
            font-weight: 600;
        }

        /* Plotly charts */
        .js-plotly-plot .plotly {
            border-radius: 16px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constantes ─────────────────────────────────────────────────────────────────
SHEET_ID = "1OUzUl5UDrZEfBSaW4afk-Nzazs7gizes3VkNfXXuKmE"
SHEET_GID = "1532105479"
GOOGLE_SHEETS_EXPORT_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={SHEET_GID}"
)

# Flexible regex patterns to detect open-response attributes
ATTRIBUTE_PATTERNS = [
    (r"Autos.*factores contribuyeron", "Autos"),
    (r"Fianzas.*factores contribuyeron", "Fianzas"),
    (r"Generales.*factores contribuyeron", "Generales"),
    (r"Soat.*factores contribuyeron", "Soat"),
    (r"Vida.*factores contribuyeron", "Vida"),
]

# Exact attribute strings (for backwards-compatible sample data / fallback)
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

# Exact texts in the "Atributo original" column (stripped for comparison)
TARGET_ATRIBUTO_ORIGINAL = [
    "¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?",
    "¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?",
    "¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?",
    "¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?",
    "¿Cuéntanos qué factores contribuyeron a que los aspectos anteriores los calificaras como Fácil o difícil?",
]

SENTIMENT_COLORS = {
    "POSITIVO": "#10b981",
    "NEGATIVO": "#ef4444",
    "NEUTRAL": "#f59e0b",
    "MIXTO": "#8b5cf6",
}

# Benchmark de satisfacción del sector asegurador colombiano (%)
SECTOR_BENCHMARK = 68

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
        """Retorna (pos_count, neg_count, neu_count)."""
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


# ── IA Contextual: Sector Asegurador Colombiano ────────────────────────────────
class ColombianInsuranceAI:
    """
    IA especializada en análisis de sentimientos para el sector asegurador colombiano.
    Usa Groq API (Llama 3.1 70B) si está disponible; análisis estadístico como fallback.
    """

    BENCHMARK = SECTOR_BENCHMARK

    def __init__(self):
        # Try Streamlit secrets first, then environment variable
        self.api_key = ""
        try:
            self.api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                import groq
                self.client = groq.Groq(api_key=self.api_key)
            except ImportError:
                pass

    def analyze_with_context(self, df_analyzed: pd.DataFrame, linea: str = None) -> str:
        """Genera insights contextualizados usando IA o estadísticas."""
        if self.client:
            return self._groq_analysis(df_analyzed, linea)
        return self._fallback_analysis(df_analyzed, linea)

    def _groq_analysis(self, df_analyzed: pd.DataFrame, linea: str = None) -> str:
        total = len(df_analyzed)
        if total == 0:
            return "No hay datos suficientes para el análisis."

        pct_pos = (df_analyzed["sentiment"] == "POSITIVO").sum() / total * 100
        pct_neg = (df_analyzed["sentiment"] == "NEGATIVO").sum() / total * 100

        top_neg = df_analyzed[df_analyzed["sentiment"] == "NEGATIVO"]["Valor"].astype(str).head(3).tolist()
        top_pos = df_analyzed[df_analyzed["sentiment"] == "POSITIVO"]["Valor"].astype(str).head(3).tolist()

        neg_items = "\n".join([f"- {c[:150]}" for c in top_neg[:3]])
        pos_items = "\n".join([f"- {c[:150]}" for c in top_pos[:3]])

        prompt = f"""
Eres un analista experto del sector asegurador colombiano, especializado en análisis de satisfacción del cliente y regulado por la Superfinanciera de Colombia.

DATOS DEL ANÁLISIS:
- Línea de negocio: {linea if linea else 'General'}
- Total respuestas: {total}
- Sentimiento positivo: {pct_pos:.1f}%
- Sentimiento negativo: {pct_neg:.1f}%

PRINCIPALES QUEJAS:
{neg_items}

PRINCIPALES ELOGIOS:
{pos_items}

CONTEXTO DEL SECTOR EN COLOMBIA:
- El mercado asegurador colombiano tiene desafíos de educación financiera
- Los clientes valoran: rapidez, claridad, transparencia, facilidad digital
- Puntos de dolor comunes: complejidad de trámites, tiempos de respuesta, documentación excesiva
- Benchmark del sector: satisfacción promedio entre 60-75%

INSTRUCCIONES:
1. Analiza los datos con perspectiva del mercado colombiano
2. Identifica 3 insights clave específicos para esta línea
3. Proporciona 2 recomendaciones accionables basadas en mejores prácticas locales
4. Compara con benchmarks del sector (si es relevante)
5. Sé conciso (máximo 300 palabras)

Formato de respuesta:
## 🎯 Insights Clave
[3 puntos específicos]

## 💡 Recomendaciones
[2 acciones concretas]

## 📊 Benchmark
[Comparativa si aplica]
"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "Eres un analista experto del sector asegurador colombiano."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception:
            return self._fallback_analysis(df_analyzed, linea)

    def _fallback_analysis(self, df_analyzed: pd.DataFrame, linea: str = None) -> str:
        total = len(df_analyzed)
        if total == 0:
            return "No hay datos suficientes para el análisis."

        pct_pos = (df_analyzed["sentiment"] == "POSITIVO").sum() / total * 100
        pct_neg = (df_analyzed["sentiment"] == "NEGATIVO").sum() / total * 100
        benchmark = self.BENCHMARK

        return f"""
## 🎯 Insights Clave

1. **Satisfacción General**: Con {pct_pos:.1f}% de sentimientos positivos, la línea {"supera" if pct_pos > benchmark else "requiere mejoras para alcanzar"} el benchmark del sector ({benchmark}%).

2. **Áreas de Atención**: {pct_neg:.1f}% de respuestas negativas indican oportunidades de mejora en experiencia del cliente.

3. **Confianza del Modelo**: Análisis robusto basado en {total} respuestas reales de clientes.

## 💡 Recomendaciones

1. **Priorizar**: Analizar comentarios negativos para identificar puntos de dolor recurrentes (tiempos, claridad, procesos).

2. **Amplificar**: Documentar y replicar las experiencias positivas como mejores prácticas internas.

## 📊 Benchmark Sector Asegurador Colombia
- Promedio industria: ~{benchmark}% satisfacción
- Tu resultado: {pct_pos:.1f}%
- {"✅ Por encima del promedio" if pct_pos > benchmark else "⚠️ Oportunidad de mejora"}

*Nota: Configura `GROQ_API_KEY` o `HF_API_TOKEN` para obtener insights más profundos con IA.*
"""


# ─── IA Contextual: Groq (Llama 3.2) ───────────────────────────────────────────

class GroqAnalyzer:
    """
    Análisis contextual usando Groq Inference API (Llama-3.1-8B-Instant).
    El token se lee desde st.secrets['GROQ_API_KEY'] o la variable de entorno GROQ_API_KEY.
    """

    def __init__(self):
        pass  # No almacenamos el token en __init__ para permitir recarga dinámica

    @property
    def api_token(self) -> str:
        """Lee el token dinámicamente cada vez que se accede."""
        try:
            token = st.secrets.get("GROQ_API_KEY", "")
            if token:
                return token
        except Exception:
            pass
        return os.getenv("GROQ_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self.api_token)

    def analyze_with_context(self, df_analyzed: pd.DataFrame, linea: str = None) -> str:
        """Genera insights usando Groq o estadísticas como fallback."""
        if self.api_token:
            return self._groq_analysis(df_analyzed, linea)
        return self._fallback_analysis(df_analyzed, linea)

    def _groq_analysis(self, df_analyzed: pd.DataFrame, linea: str = None) -> str:
            """Análisis usando Groq."""
            from groq import Groq
    
            st.write(f"🔍 DEBUG: Entró a _groq_analysis")
            st.write(f"🔍 DEBUG: api_token dentro = {self.api_token[:10] if self.api_token else 'VACIO'}")
    
            total = len(df_analyzed)
            if total == 0:
                return "No hay datos suficientes para el análisis."

    pct_pos = (df_analyzed["sentiment"] == "POSITIVO").sum() / total * 100
    pct_neg = (df_analyzed["sentiment"] == "NEGATIVO").sum() / total * 100

    prompt = (
        f"Eres analista del sector asegurador colombiano (Superfinanciera).\n\n"
        f"Datos: {total} respuestas, {pct_pos:.1f}% positivo, {pct_neg:.1f}% negativo\n"
        f"Línea: {linea or 'Todas'}\n\n"
        f"Da 3 insights clave y 2 recomendaciones para Colombia (max 250 palabras)."
    )

    try:
        st.write(f"🔍 DEBUG: A punto de llamar a Groq API")
        
        client = Groq(api_key=self.api_token)
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Eres un analista experto del sector asegurador colombiano."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        text = completion.choices[0].message.content
        
        st.write(f"🔍 DEBUG: Groq respondió exitosamente")
        st.write(f"🔍 DEBUG: Respuesta (primeros 100 chars) = {text[:100]}")
        
        return text.strip()
        
    except Exception as e:
        st.write(f"❌ DEBUG ERROR: {str(e)}")
        st.write(f"❌ DEBUG ERROR Type: {type(e).__name__}")
        import traceback
        st.write(f"❌ DEBUG TRACEBACK: {traceback.format_exc()}")
        return self._fallback_analysis(df_analyzed, linea)

    def _fallback_analysis(self, df_analyzed: pd.DataFrame, linea: str = None) -> str:
        total = len(df_analyzed)
        if total == 0:
            return "No hay datos suficientes para el análisis."
        pct_pos = (df_analyzed["sentiment"] == "POSITIVO").sum() / total * 100
        pct_neg = (df_analyzed["sentiment"] == "NEGATIVO").sum() / total * 100
        
        benchmark = SECTOR_BENCHMARK

        linea_texto = f"línea {linea}" if linea and linea != "Todas las líneas" else "todas las líneas"

        return f"""
## 🎯 Insights Clave

1. **Satisfacción General**: Con {pct_pos:.1f}% de sentimientos positivos, la línea {"supera" if pct_pos > benchmark else "requiere mejoras para alcanzar"} el benchmark del sector ({benchmark}%).

2. **Áreas de Atención**: {pct_neg:.1f}% de respuestas negativas indican oportunidades de mejora en experiencia del cliente.

3. **Confianza del Modelo**: Análisis robusto basado en {total} respuestas reales de clientes.

## 💡 Recomendaciones

1. **Priorizar**: Analizar comentarios negativos para identificar puntos de dolor recurrentes.

2. **Amplificar**: Documentar y replicar las experiencias positivas como mejores prácticas internas.

## 📊 Benchmark Sector Asegurador Colombia
- Promedio industria: ~{benchmark}% satisfacción
- Tu resultado: {pct_pos:.1f}%
- {"✅ Por encima del promedio" if pct_pos > benchmark else "⚠️ Oportunidad de mejora"}

*Nota: Configura `HF_API_TOKEN` en secrets.toml para obtener insights más profundos con IA.*
"""


# ── Helpers de token y seguridad ──────────────────────────────────────────────
def get_hf_token() -> str:
    """
    Lee el token de HuggingFace de forma segura:
    1) st.secrets['HF_API_TOKEN'] (Streamlit Cloud / secrets.toml)
    2) Variable de entorno HF_API_TOKEN
    Nunca incluir el token literal en el código ni en el repositorio.
    """
    try:
        token = st.secrets.get("HF_API_TOKEN", "")
        if token:
            return token
    except Exception:
        pass
    return os.getenv("HF_API_TOKEN", "")


# ── Helpers de sanitización para pyarrow / Streamlit ──────────────────────────
def _to_text_safe(x) -> str:
    """Convierte cualquier valor (bytes, int, list, None, str) a texto limpio."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)
    if isinstance(x, (list, tuple, dict)):
        return str(x)
    return str(x)


def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas dtype object con mezcla de bytes/ints/str a strings limpios
    para evitar pyarrow.lib.ArrowTypeError al serializar DataFrames en Streamlit.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_to_text_safe)
    return df


def _ensure_1d_str(series: pd.Series) -> pd.Series:
    """
    Convierte celdas listas/tuplas/dicts a str y normaliza a 1-D string Series.
    Útil para columnas clave antes de groupby o pivot.
    """
    return series.apply(_to_text_safe).astype(str).str.strip()


# ── Inferencia remota y orquestación ──────────────────────────────────────────
HF_SENTIMENT_MODEL = "PlanTL-GOB-ES/roberta-base-bne-sentiment"
_HF_MAX_RETRIES = 3
_HF_TIMEOUT = 30
_HF_MAX_INPUT_LENGTH = 512
_HF_RETRY_BACKOFF_BASE = 2
_HF_LABEL_MAP = {
    "POS": "POSITIVO", "POSITIVE": "POSITIVO",
    "NEG": "NEGATIVO", "NEGATIVE": "NEGATIVO",
    "NEU": "NEUTRAL", "NEUTRAL": "NEUTRAL",
}
_HF_TOKEN_ERROR_MSG = (
    "⚠️ HF token inválido o sin permisos (HTTP {status}). "
    "Revoca y genera uno nuevo en https://huggingface.co/settings/tokens"
)


def analyze_with_hf(
    texts: list[str],
    hf_token: str,
    model: str = HF_SENTIMENT_MODEL,
) -> list[dict]:
    """
    Llama a la HuggingFace Inference API para análisis de sentimiento en lote.
    Retorna lista de dicts con keys: label, score, confidence.
    Maneja errores HTTP 401/403/429 y hace retries con backoff.
    """
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    results = []

    for text in texts:
        payload = {"inputs": text[:_HF_MAX_INPUT_LENGTH]}
        for attempt in range(_HF_MAX_RETRIES):
            try:
                resp = requests.post(
                    api_url, headers=headers, json=payload, timeout=_HF_TIMEOUT
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # API returns [[{label, score},...]] for single input
                    if isinstance(data, list) and data:
                        inner = data[0] if isinstance(data[0], list) else data
                        best = max(inner, key=lambda x: x.get("score", 0))
                        label_raw = best.get("label", "NEU").upper()
                        label = _HF_LABEL_MAP.get(label_raw, "NEUTRAL")
                        score_val = {"POSITIVO": 1.0, "NEGATIVO": -1.0, "NEUTRAL": 0.0}.get(label, 0.0)
                        results.append({
                            "label": label,
                            "score": score_val,
                            "confidence": round(float(best.get("score", 0.5)), 4),
                            "source": "hf_remote",
                        })
                    else:
                        results.append(None)
                    break
                elif resp.status_code in (401, 403):
                    st.warning(_HF_TOKEN_ERROR_MSG.format(status=resp.status_code))
                    results.append(None)
                    break
                elif resp.status_code == 429:
                    time.sleep(_HF_RETRY_BACKOFF_BASE ** attempt)
                else:
                    results.append(None)
                    break
            except Exception:
                if attempt < _HF_MAX_RETRIES - 1:
                    time.sleep(1)
        else:
            results.append(None)

    return results


def analyze_local_beto(texts: list[str], classifier=None) -> list[dict]:
    """
    Analiza textos usando el modelo BETO local (o keywords como fallback).
    Retorna lista de dicts con keys: label, score, confidence, source.
    """
    analyzer = SentimentAnalyzer()
    if classifier is None:
        classifier = analyzer.load_model()
    results = []
    for text in texts:
        res = analyzer.analyze_sentiment(str(text), classifier)
        results.append({
            "label": res["sentiment"],
            "score": res["score"],
            "confidence": res["confidence"],
            "source": "local_beto",
        })
    return results


def analyze_texts(texts: list[str]) -> list[dict]:
    """
    Orquestador: intenta HuggingFace (si hay token) → fallback a BETO local.
    Retorna lista de dicts con keys: label, score, confidence, source.
    """
    hf_token = get_hf_token()
    if not hf_token:
        return analyze_local_beto(texts)

    hf_results = analyze_with_hf(texts, hf_token)

    # Fallback a BETO para los textos donde HF devolvió None
    failed_indices = [i for i, r in enumerate(hf_results) if r is None]
    if failed_indices:
        failed_texts = [texts[i] for i in failed_indices]
        local_results = analyze_local_beto(failed_texts)
        for idx, local_res in zip(failed_indices, local_results):
            hf_results[idx] = local_res

    return hf_results


# ── Helpers de datos ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_clientes_sheet() -> pd.DataFrame | None:
    """Carga automática desde Google Sheets pestaña CLIENTES (cache 5 min)."""
    try:
        df = pd.read_csv(GOOGLE_SHEETS_EXPORT_URL)
        return df
    except Exception as exc:
        st.sidebar.error(f"❌ Error al cargar automáticamente: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_from_google_sheets(url: str) -> pd.DataFrame:
    """Descarga datos desde Google Sheets exportado como CSV."""
    df = pd.read_csv(url)
    return df


def detect_columns(df: pd.DataFrame) -> dict:
    """Detecta automáticamente las columnas importantes del DataFrame."""
    attr_col = None
    val_col = None
    linea_col = None
    attr_original_col = None
    suc_col = None

    for col in df.columns:
        col_low = col.strip().lower()
        # Detect "Atributo original" column (priority over generic atributo)
        if attr_original_col is None and col_low in (
            "atributo_original", "atributo original"
        ):
            attr_original_col = col
        # Detect attribute column
        if attr_col is None:
            if col_low in ("atributo", "attribute"):
                attr_col = col
            elif df[col].astype(str).str.contains("¿", na=False).sum() > 5:
                attr_col = col
        # Detect value column
        if val_col is None and col_low in ("valor", "value", "respuesta", "response"):
            val_col = col
        # Detect line/ramo column
        if linea_col is None and ("linea" in col_low or "ramo" in col_low or "línea" in col_low):
            linea_col = col
        # Detect branch/sucursal column
        if suc_col is None and col_low in ("suc", "sucursal"):
            suc_col = col

    return {
        "atributo": attr_col,
        "atributo_original": attr_original_col,
        "valor": val_col,
        "linea": linea_col,
        "sucursal": suc_col,
    }


def _normalize_attribute_text(series: pd.Series) -> pd.Series:
    """
    Normalizes attribute text to enable flexible regex matching.
    Handles: newlines, multiple whitespace, and trailing numbers.
    """
    return (
        series.astype(str)
        .str.replace(r"\n", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\s*\d+\s*$", "", regex=True)
        .str.strip()
    )


def _coerce_columns_1d(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """
    Ensures the given columns in df are 1-D string Series.
    If a column is a DataFrame (due to duplicate column names after rename),
    takes the first sub-column and casts to str. Safe no-op if column absent.
    Uses _ensure_1d_str to handle lists/tuples/dicts/bytes cells safely.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                # Duplicate columns cause df[col] to return a DataFrame — take first
                df[col] = df[col].iloc[:, 0]
            df[col] = _ensure_1d_str(df[col])
    return df


def filter_open_responses(df: pd.DataFrame, selected_lineas: list | None = None) -> pd.DataFrame:
    """
    Filtra el DataFrame para quedarse sólo con los atributos de respuestas abiertas.
    Si existe la columna 'Atributo original', prioriza esa ruta usando TARGET_ATRIBUTO_ORIGINAL.
    De lo contrario, usa búsqueda flexible con regex (.str.contains) para manejar variaciones
    de texto, incluyendo saltos de línea, espacios múltiples y números al final.
    """
    cols = detect_columns(df)
    col_attr_original = cols["atributo_original"]
    col_attr = cols["atributo"]
    col_val = cols["valor"]
    col_linea = cols["linea"]
    col_suc = cols["sucursal"]

    # ── Priority path: "Atributo original" column ────────────────────────────
    if col_attr_original is not None:
        mask = df[col_attr_original].astype(str).str.strip().isin(TARGET_ATRIBUTO_ORIGINAL)

        # Determine response column: prefer literal 'Valor', then detected, then fallback
        if "Valor" in df.columns and "Valor" != col_attr_original:
            resp_col = "Valor"
        elif col_val is not None and col_val != col_attr_original:
            resp_col = col_val
        elif len(df.columns) > 1:
            resp_col = df.columns[1]
        else:
            resp_col = df.columns[0]

        filtered = df[mask].copy()
        # Rename only if source and target names differ to avoid duplicate columns
        rename_map = {}
        if col_attr_original != "Atributo":
            rename_map[col_attr_original] = "Atributo"
        if resp_col != "Valor":
            rename_map[resp_col] = "Valor"
        if rename_map:
            filtered = filtered.rename(columns=rename_map)
        # Remove any accidental duplicate columns (keep first occurrence)
        filtered = filtered.loc[:, ~filtered.columns.duplicated(keep="first")]
        filtered = filtered[
            filtered["Valor"].notna() & (filtered["Valor"].astype(str).str.strip() != "")
        ]

        # Ensure key columns are 1-D strings to prevent groupby errors
        filtered = _coerce_columns_1d(filtered, "Atributo", "Valor")

        # Map linea_negocio from detected linea column or fallback 'General'
        if col_linea is not None and col_linea in filtered.columns:
            filtered["linea_negocio"] = filtered[col_linea].fillna("General").astype(str).str.strip()
        else:
            filtered["linea_negocio"] = "General"

        # Map Sucursal from detected sucursal column
        if col_suc is not None and col_suc in filtered.columns:
            filtered["Sucursal"] = filtered[col_suc].fillna("").astype(str).str.strip()
        else:
            filtered["Sucursal"] = ""

        n_after_attr_filter = len(filtered)
        st.info(
            f"Filtrado por 'Atributo original' (columna: '{col_attr_original}'): "
            f"{n_after_attr_filter:,} filas retenidas."
        )

        # Extract date BEFORE resetting index (indices still align with original df)
        date_cols = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()]
        if date_cols:
            filtered["fecha"] = pd.to_datetime(
                df.loc[filtered.index, date_cols[0]], errors="coerce"
            )

        # Apply optional línea filter with normalized comparison
        if selected_lineas:
            sel_norm = [str(s).strip() for s in selected_lineas]
            mask_linea = filtered["linea_negocio"].isin(sel_norm)
            filtered_with_linea = filtered[mask_linea].reset_index(drop=True)
            if filtered_with_linea.empty:
                st.warning(
                    "⚠️ El filtro por Línea ha dejado 0 filas; se muestran resultados "
                    "sin filtro de línea. Desactiva el filtro si deseas una selección "
                    "más restrictiva."
                )
                # Auto-fallback: ignore the línea filter when it yields nothing
            else:
                filtered = filtered_with_linea

        filtered = filtered.reset_index(drop=True)
        return filtered

    # ── Fallback path: legacy Atributo column with regex patterns ────────────
    # Fallback: assume first two cols are Atributo / Valor
    if col_attr is None:
        col_attr = df.columns[0]
    if col_val is None:
        col_val = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Normalize attribute text to handle \n, multiple spaces and trailing numbers
    attr_clean = _normalize_attribute_text(df[col_attr])

    # Build combined regex mask from patterns (applied to normalized text)
    mask = pd.Series([False] * len(df), index=df.index)
    for pattern, _ in ATTRIBUTE_PATTERNS:
        mask |= attr_clean.str.contains(pattern, case=False, na=False, regex=True)

    # Also include exact target attributes (backward compatibility)
    mask |= df[col_attr].isin(TARGET_ATTRIBUTES)

    filtered = df[mask].copy()
    # Rename only if names differ to avoid duplicate columns
    rename_map = {}
    if col_attr != "Atributo":
        rename_map[col_attr] = "Atributo"
    if col_val != "Valor":
        rename_map[col_val] = "Valor"
    if rename_map:
        filtered = filtered.rename(columns=rename_map)
    # Remove any accidental duplicate columns
    filtered = filtered.loc[:, ~filtered.columns.duplicated(keep="first")]
    filtered = filtered[
        filtered["Valor"].notna() & (filtered["Valor"].astype(str).str.strip() != "")
    ]

    # Ensure key columns are 1-D strings to prevent groupby errors
    filtered = _coerce_columns_1d(filtered, "Atributo", "Valor")

    # Assign linea_negocio label from pattern matching
    def _get_linea(attr_text: str) -> str:
        for pattern, label in ATTRIBUTE_PATTERNS:
            if re.search(pattern, str(attr_text), re.IGNORECASE):
                return label
        return ATTRIBUTE_LABELS.get(attr_text, "General")

    filtered["linea_negocio"] = filtered["Atributo"].apply(_get_linea)

    # Map Sucursal from detected sucursal column
    if col_suc is not None:
        filtered["Sucursal"] = df.loc[filtered.index, col_suc].fillna("").astype(str).str.strip()
    else:
        filtered["Sucursal"] = ""

    # Extract date BEFORE resetting index (indices still align with original df)
    date_cols = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()]
    if date_cols:
        filtered["fecha"] = pd.to_datetime(
            df.loc[filtered.index, date_cols[0]], errors="coerce"
        )

    # Apply optional línea filter with normalized comparison
    if selected_lineas:
        sel_norm = [str(s).strip() for s in selected_lineas]
        mask_linea = filtered["linea_negocio"].isin(sel_norm)
        filtered_with_linea = filtered[mask_linea].reset_index(drop=True)
        if filtered_with_linea.empty:
            st.warning(
                "⚠️ El filtro por Línea ha dejado 0 filas; se muestran resultados "
                "sin filtro de línea. Desactiva el filtro si deseas una selección "
                "más restrictiva."
            )
            # Auto-fallback: ignore the línea filter when it yields nothing
        else:
            filtered = filtered_with_linea

    filtered = filtered.reset_index(drop=True)
    return filtered


def generate_sample_data(n: int = 50) -> pd.DataFrame:
    """Genera datos sintéticos para demostración."""
    rng = np.random.default_rng(42)
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


# ── Visualizaciones avanzadas ──────────────────────────────────────────────────
def create_gauge_chart(percentage: float, title: str, color: str) -> go.Figure:
    """Medidor circular tipo speedometer."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=percentage,
        title={"text": title, "font": {"size": 18}},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33], "color": "rgba(239,68,68,0.15)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.15)"},
                {"range": [66, 100], "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": percentage,
            },
        },
    ))
    fig.update_layout(height=240, margin=dict(t=60, b=10, l=10, r=10))
    return fig


def create_3d_scatter(df: pd.DataFrame) -> go.Figure:
    """Gráfico de dispersión 3D de sentimientos."""
    color_map = {
        "POSITIVO": "#10b981",
        "NEGATIVO": "#ef4444",
        "NEUTRAL": "#f59e0b",
        "MIXTO": "#8b5cf6",
    }
    fig = go.Figure(data=[go.Scatter3d(
        x=df["confidence"],
        y=df["score"],
        z=df["keywords_pos"] - df["keywords_neg"],
        mode="markers",
        marker=dict(
            size=7,
            color=[color_map.get(s, "#667eea") for s in df["sentiment"]],
            opacity=0.8,
            line=dict(width=0.5, color="white"),
        ),
        text=df["Valor"].astype(str).str[:60],
        hovertemplate="<b>%{text}</b><br>Confianza: %{x:.2f}<br>Score: %{y:.2f}<extra></extra>",
    )])
    fig.update_layout(
        title="Análisis 3D de Sentimientos",
        scene=dict(
            xaxis_title="Confianza",
            yaxis_title="Score",
            zaxis_title="Balance Keywords",
        ),
        height=600,
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig


def create_sankey_diagram(df: pd.DataFrame) -> go.Figure:
    """Diagrama de flujo Sankey: Línea → Sentimiento."""
    if "linea_negocio" not in df.columns or df.empty:
        return go.Figure()

    # Coerce linea_negocio to 1-D strings to prevent groupby errors
    df = _coerce_columns_1d(df, "linea_negocio")

    sentiments_by_linea = pd.crosstab(df["linea_negocio"], df["sentiment"])
    linea_nodes = list(sentiments_by_linea.index)
    sentiment_nodes = [s for s in ["POSITIVO", "NEGATIVO", "NEUTRAL", "MIXTO"] if s in sentiments_by_linea.columns]
    all_nodes = linea_nodes + sentiment_nodes

    source, target, value, link_colors = [], [], [], []
    node_colors = (
        ["#667eea"] * len(linea_nodes)
        + [SENTIMENT_COLORS.get(s, "#667eea") for s in sentiment_nodes]
    )

    for i, linea in enumerate(linea_nodes):
        for j, sent in enumerate(sentiment_nodes):
            val = int(sentiments_by_linea.loc[linea, sent]) if sent in sentiments_by_linea.columns else 0
            if val > 0:
                source.append(i)
                target.append(len(linea_nodes) + j)
                value.append(val)
                base = SENTIMENT_COLORS.get(sent, "#667eea").lstrip("#")
                r, g, b = int(base[0:2], 16), int(base[2:4], 16), int(base[4:6], 16)
                link_colors.append(f"rgba({r},{g},{b},0.4)")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color="rgba(0,0,0,0.2)", width=0.5),
            label=all_nodes,
            color=node_colors,
        ),
        link=dict(source=source, target=target, value=value, color=link_colors),
    ))
    fig.update_layout(
        title="Flujo de Sentimientos por Línea de Negocio",
        height=450,
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig


def create_radar_chart(df: pd.DataFrame) -> go.Figure:
    """Gráfico radar comparativo por línea de negocio."""
    if "linea_negocio" not in df.columns or df.empty:
        return go.Figure()

    # Coerce linea_negocio to 1-D strings to prevent groupby errors
    df = _coerce_columns_1d(df, "linea_negocio")

    lineas = df["linea_negocio"].value_counts().head(5).index.tolist()
    categories = ["% Positivo", "% Negativo", "% Neutral", "Confianza ×100", "Score +50"]

    fig = go.Figure()
    for linea in lineas:
        ld = df[df["linea_negocio"] == linea]
        n = len(ld)
        if n == 0:
            continue
        values = [
            (ld["sentiment"] == "POSITIVO").sum() / n * 100,
            (ld["sentiment"] == "NEGATIVO").sum() / n * 100,
            (ld["sentiment"] == "NEUTRAL").sum() / n * 100,
            ld["confidence"].mean() * 100,
            ld["score"].mean() * 50 + 50,  # scale [-1,1] → [0,100]
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=linea,
            opacity=0.65,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Comparativa Radar por Línea de Negocio",
        height=450,
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig


# ── Renderizado de tabs ────────────────────────────────────────────────────────
def render_tab_dashboard(df: pd.DataFrame):
    st.subheader("📊 Dashboard Premium")

    # Coerce key columns to 1-D strings to prevent groupby dimension errors
    df = _coerce_columns_1d(df, "Atributo", "linea_negocio", "Sucursal")

    total = len(df)
    pct_pos = (df["sentiment"] == "POSITIVO").mean() * 100
    pct_neg = (df["sentiment"] == "NEGATIVO").mean() * 100
    avg_conf = df["confidence"].mean() * 100
    nps = max(pct_pos - pct_neg, 0)

    # Gauge row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(create_gauge_chart(pct_pos, "😊 Positivos %", "#10b981"), use_container_width=True)
    with col2:
        st.plotly_chart(create_gauge_chart(pct_neg, "😞 Negativos %", "#ef4444"), use_container_width=True)
    with col3:
        st.plotly_chart(create_gauge_chart(avg_conf, "🎯 Confianza %", "#667eea"), use_container_width=True)
    with col4:
        st.plotly_chart(create_gauge_chart(nps, "📈 NPS Score", "#8b5cf6"), use_container_width=True)

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📋 Total respuestas", f"{total:,}")
    c2.metric("😊 % Positivos", f"{pct_pos:.1f}%", delta=f"{pct_pos - 50:.1f}pp")
    c3.metric("😞 % Negativos", f"{pct_neg:.1f}%", delta=f"{-(pct_neg - 50):.1f}pp", delta_color="inverse")
    c4.metric("⭐ Score promedio", f"{df['score'].mean():.2f}")
    c5.metric("🎯 Confianza modelo", f"{avg_conf:.1f}%")

    st.markdown("---")

    # Top-N metric cards for Ramo (línea de negocio)
    if "linea_negocio" in df.columns:
        top_lines = df["linea_negocio"].value_counts().head(4).index.tolist()
        if top_lines:
            st.markdown("**ℹ️ Desglose por Ramo (Top líneas)**")
            cols_line = st.columns(len(top_lines))
            for c, ln in zip(cols_line, top_lines):
                ln_df = df[df["linea_negocio"] == ln]
                n_ln = len(ln_df)
                pos_ln = (ln_df["sentiment"] == "POSITIVO").mean() * 100
                neg_ln = (ln_df["sentiment"] == "NEGATIVO").mean() * 100
                with c:
                    st.markdown(f"**{ln}**")
                    st.metric("😊 % Pos", f"{pos_ln:.1f}%")
                    st.metric("😞 % Neg", f"{neg_ln:.1f}%")
                    st.caption(f"{n_ln:,} respuestas")

    # Top-N metric cards for Sucursal
    if "Sucursal" in df.columns and df["Sucursal"].str.strip().ne("").any():
        top_sucs = (
            df.loc[df["Sucursal"].str.strip() != "", "Sucursal"]
            .value_counts()
            .head(4)
            .index.tolist()
        )
        if top_sucs:
            st.markdown("**ℹ️ Desglose por Sucursal (Top sucursales)**")
            cols_suc = st.columns(len(top_sucs))
            for c, s in zip(cols_suc, top_sucs):
                s_df = df[df["Sucursal"] == s]
                n_s = len(s_df)
                pos_s = (s_df["sentiment"] == "POSITIVO").mean() * 100
                neg_s = (s_df["sentiment"] == "NEGATIVO").mean() * 100
                with c:
                    st.markdown(f"**{s}**")
                    st.metric("😊 % Pos", f"{pos_s:.1f}%")
                    st.metric("😞 % Neg", f"{neg_s:.1f}%")
                    st.caption(f"{n_s:,} respuestas")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_sankey = create_sankey_diagram(df)
        st.plotly_chart(fig_sankey, use_container_width=True)

    with col_right:
        fig_radar = create_radar_chart(df)
        st.plotly_chart(fig_radar, use_container_width=True)

    # Stacked bar by line
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

    # Sentiment breakdown by Sucursal
    if "Sucursal" in df.columns and df["Sucursal"].str.strip().ne("").any():
        pivot_suc = (
            df[df["Sucursal"].str.strip() != ""]
            .groupby(["Sucursal", "sentiment"])
            .size()
            .reset_index(name="n")
        )
        pivot_suc["pct"] = pivot_suc.groupby("Sucursal")["n"].transform(
            lambda x: x / x.sum() * 100
        )
        fig_bar_suc = px.bar(
            pivot_suc,
            x="Sucursal",
            y="pct",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            title="Sentimientos por Sucursal (%)",
            labels={"Sucursal": "Sucursal", "pct": "Porcentaje (%)"},
        )
        st.plotly_chart(fig_bar_suc, use_container_width=True)

    # Temporal trend
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


def render_tab_3d(df: pd.DataFrame):
    st.subheader("🎯 Visualización 3D Interactiva")
    st.info("💡 Interactúa con el gráfico: rotar, zoom, hover para ver detalles de cada comentario.")

    # Coerce key columns to 1-D strings to prevent groupby dimension errors
    df = _coerce_columns_1d(df, "Atributo", "linea_negocio", "Sucursal")

    fig_3d = create_3d_scatter(df)
    st.plotly_chart(fig_3d, use_container_width=True)

    # Heatmap
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
        st.dataframe(sanitize_df_for_streamlit(summary_sent), use_container_width=True, hide_index=True)

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
            st.dataframe(sanitize_df_for_streamlit(summary_ln), use_container_width=True, hide_index=True)

    # Keywords bar
    col3, col4 = st.columns(2)
    with col3:
        total_kw_pos = df["keywords_pos"].sum()
        total_kw_neg = df["keywords_neg"].sum()
        kw_fig = go.Figure(go.Bar(
            x=["Keywords Positivas", "Keywords Negativas"],
            y=[total_kw_pos, total_kw_neg],
            marker_color=[SENTIMENT_COLORS["POSITIVO"], SENTIMENT_COLORS["NEGATIVO"]],
        ))
        kw_fig.update_layout(title="Total Keywords Detectadas", showlegend=False)
        st.plotly_chart(kw_fig, use_container_width=True)

    with col4:
        st.markdown("**Resumen por Atributo**")
        summary_attr = (
            df.groupby("Atributo")
            .agg(cantidad=("sentiment", "count"))
            .reset_index()
        )
        summary_attr["Atributo"] = summary_attr["Atributo"].apply(
            lambda a: ATTRIBUTE_LABELS.get(a, a)
        )
        summary_attr.columns = ["Atributo", "Cantidad"]
        st.dataframe(sanitize_df_for_streamlit(summary_attr), use_container_width=True, hide_index=True)


def render_tab_comments(df: pd.DataFrame):
    st.subheader("💬 Explorador de Comentarios")

    # Coerce key columns to 1-D strings to prevent groupby dimension errors
    df = _coerce_columns_1d(df, "Atributo", "linea_negocio", "Sucursal")

    has_sucursal = "Sucursal" in df.columns and df["Sucursal"].str.strip().ne("").any()

    # Pre-compute aggregate sentiment stats per línea and per sucursal
    line_stats: dict[str, str] = {}
    if "linea_negocio" in df.columns:
        gp_ln = df.groupby("linea_negocio")["sentiment"].value_counts().unstack(fill_value=0)
        for ln in gp_ln.index:
            total_ln = int(gp_ln.loc[ln].sum())
            pos_ln = gp_ln.loc[ln].get("POSITIVO", 0) / total_ln * 100 if total_ln else 0
            neg_ln = gp_ln.loc[ln].get("NEGATIVO", 0) / total_ln * 100 if total_ln else 0
            neu_ln = gp_ln.loc[ln].get("NEUTRAL", 0) / total_ln * 100 if total_ln else 0
            line_stats[ln] = f"{pos_ln:.0f}% Pos / {neg_ln:.0f}% Neg / {neu_ln:.0f}% Neu"

    suc_stats: dict[str, str] = {}
    if has_sucursal:
        gp_suc = (
            df[df["Sucursal"].str.strip() != ""]
            .groupby("Sucursal")["sentiment"]
            .value_counts()
            .unstack(fill_value=0)
        )
        for s in gp_suc.index:
            total_s = int(gp_suc.loc[s].sum())
            pos_s = gp_suc.loc[s].get("POSITIVO", 0) / total_s * 100 if total_s else 0
            neg_s = gp_suc.loc[s].get("NEGATIVO", 0) / total_s * 100 if total_s else 0
            neu_s = gp_suc.loc[s].get("NEUTRAL", 0) / total_s * 100 if total_s else 0
            suc_stats[s] = f"{pos_s:.0f}% Pos / {neg_s:.0f}% Neg / {neu_s:.0f}% Neu"

    col1, col2, col3, col4 = st.columns(4) if has_sucursal else st.columns(3)
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
    if has_sucursal:
        with col3:
            suc_options = sorted(df.loc[df["Sucursal"].str.strip() != "", "Sucursal"].unique())
            selected_suc = st.multiselect(
                "Filtrar por Sucursal",
                options=suc_options,
                default=suc_options,
            )
        with col4:
            min_conf = st.slider("Confianza mínima (%)", 0, 100, 0) / 100
    else:
        selected_suc = []
        with col3:
            min_conf = st.slider("Confianza mínima (%)", 0, 100, 0) / 100

    mask = df["sentiment"].isin(sents) & (df["confidence"] >= min_conf)
    if selected_lines and "linea_negocio" in df.columns:
        mask &= df["linea_negocio"].isin(selected_lines)
    if selected_suc and has_sucursal:
        mask &= df["Sucursal"].isin(selected_suc)

    filtered = df[mask].head(50)
    st.markdown(f"**Mostrando {len(filtered)} de {mask.sum()} comentarios filtrados**")

    for _, row in filtered.iterrows():
        text_preview = str(row["Valor"])[:80] + ("…" if len(str(row["Valor"])) > 80 else "")
        sent = row["sentiment"]
        color_class = f"sentiment-{sent.lower()}"
        linea_label = str(row.get("linea_negocio", ATTRIBUTE_LABELS.get(row["Atributo"], row["Atributo"])))
        suc_label = str(row.get("Sucursal", "")).strip() if has_sucursal else ""

        # Header includes Sucursal when present
        header_label = (
            f"[{linea_label} • {suc_label}] {text_preview}"
            if suc_label
            else f"[{linea_label}] {text_preview}"
        )

        with st.expander(header_label):
            st.markdown(f"**Texto completo:** {row['Valor']}")
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"**Sentimiento:** <span class='{color_class}'>{sent}</span>",
                unsafe_allow_html=True,
            )
            c2.markdown(f"**Score:** `{row['score']:.2f}`")
            c3.markdown(f"**Confianza:** `{row['confidence']:.2%}`")
            if "linea_negocio" in df.columns:
                st.markdown(f"**Línea de negocio:** {linea_label}")
                if line_stats.get(linea_label):
                    st.caption(f"Desglose Línea — {line_stats[linea_label]}")
            if suc_label:
                st.markdown(f"**Sucursal:** {suc_label}")
                if suc_stats.get(suc_label):
                    st.caption(f"Desglose Sucursal — {suc_stats[suc_label]}")


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

    top20 = word_freq[:20]
    words_list = [w[0] for w in top20]
    freqs = [w[1] for w in top20]

    fig_bar = go.Figure(go.Bar(
        y=words_list[::-1],
        x=freqs[::-1],
        orientation="h",
        marker_color=SENTIMENT_COLORS.get(sent_filter, "#667eea")
        if sent_filter != "Todos"
        else "#667eea",
    ))
    fig_bar.update_layout(
        title="Top 20 Palabras más Frecuentes",
        xaxis_title="Frecuencia",
        yaxis_title="Palabra",
        height=500,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def render_tab_ai(df: pd.DataFrame):
    st.subheader("🤖 Insights con IA")
    st.markdown("*Análisis contextualizado para el sector asegurador colombiano*")

    # Instantiate analyzers once per render, outside the column layout blocks
    groq_ai = ColombianInsuranceAI()
    hf_ai = GroqAnalyzer()
    
    st.write(f"🔍 DEBUG: Token detectado: {hf_ai.api_token[:10]}..." if hf_ai.api_token else "❌ No hay token")
    st.write(f"🔍 DEBUG: Available: {hf_ai.available}")

    proveedor_options = []
    if groq_ai.client:
        proveedor_options.append("🦙 Groq (Llama 3.1)")
    if hf_ai.available:
        proveedor_options.append("🤗 HuggingFace (Llama 3.2)")
    proveedor_options.append("📊 Estadístico")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        lineas_disponibles = ["📊 Todas las líneas"]
        if "linea_negocio" in df.columns:
            lineas_disponibles += [f"📋 {l}" for l in sorted(df["linea_negocio"].unique())]
        linea_ia = st.selectbox("Selecciona línea de negocio:", options=lineas_disponibles)

    with col2:
        proveedor = st.selectbox("Proveedor de IA:", options=proveedor_options)

    with col3:
        generar = st.button("🔮 Generar Insights", type="primary", use_container_width=True)

    if generar:
        st.session_state["generate_ia"] = True
        st.session_state["ia_proveedor"] = proveedor

    if st.session_state.get("generate_ia", False):
        selected_provider = st.session_state.get("ia_proveedor", proveedor)
        with st.spinner("🧠 Analizando con contexto del sector asegurador colombiano…"):
            if "HuggingFace" in selected_provider:
                analyzer_obj = hf_ai
            else:
                analyzer_obj = groq_ai

            if "Todas" in linea_ia:
                insights = analyzer_obj.analyze_with_context(df)
            else:
                linea_clean = linea_ia.replace("📋 ", "")
                linea_df = df[df["linea_negocio"] == linea_clean] if "linea_negocio" in df.columns else df
                insights = analyzer_obj.analyze_with_context(linea_df, linea_clean)

            st.markdown(insights)
            st.session_state["generate_ia"] = False

    groq_configured = groq_ai.client is not None
    hf_configured = hf_ai.available
    if not st.session_state.get("generate_ia", False) and not groq_configured and not hf_configured:
        st.info(
            "💡 Para obtener insights más profundos con IA, configura tu `GROQ_API_KEY` "
            "(https://console.groq.com) o tu `HF_API_TOKEN` (https://huggingface.co/settings/tokens) "
            "en `.streamlit/secrets.toml`."
        )


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
    st.dataframe(sanitize_df_for_streamlit(df), use_container_width=True)


# ── Pantalla de bienvenida ─────────────────────────────────────────────────────
def render_welcome():
    st.markdown(
        "<h1>🎯 Sistema de Análisis de Sentimientos</h1>"
        "<h3 style='text-align:center;color:#555;'>Aseguradora — Powered by BETO + IA Contextual</h3>",
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
            - **IA contextual** del sector asegurador colombiano
            """
        )
    with col2:
        st.markdown(
            """
            ### 📊 Visualizaciones disponibles
            - Gauges tipo speedometer
            - Gráfico **3D** interactivo
            - Diagrama **Sankey** por línea
            - Gráfico **radar** comparativo
            - Nube de palabras + Top 20
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
        1. Los datos de Google Sheets se cargan **automáticamente** al iniciar.
        2. También puedes elegir subir un CSV o usar datos de ejemplo.
        3. Usa los **filtros** de Línea/Ramo en el panel lateral.
        4. Haz clic en **🔍 Analizar Sentimientos**.
        5. Explora los resultados en las 6 pestañas.
        6. Genera **Insights con IA** en la pestaña correspondiente.
        7. Descarga los resultados desde la pestaña **Exportar**.
        """
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    """Renderiza la barra lateral y retorna (selected_lineas, analyze_btn)."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/analytics.png", width=80)
        st.title("⚙️ Configuración")
        st.markdown("---")

        # ── Automatic load ────────────────────────────────────────────────────
        if "df_raw" not in st.session_state:
            with st.spinner("⏳ Cargando datos de Google Sheets…"):
                auto_df = load_clientes_sheet()
            if auto_df is not None:
                st.session_state["df_raw"] = auto_df
                st.success(f"✅ {len(auto_df):,} registros cargados automáticamente")

        source = st.radio(
            "Fuente de datos",
            ["🔗 Google Sheets", "📂 Subir CSV", "🧪 Datos de ejemplo"],
            index=0,
        )

        if source == "🔗 Google Sheets":
            url_input = st.text_input(
                "URL de Google Sheets",
                value=(
                    "https://docs.google.com/spreadsheets/d/"
                    "1OUzUl5UDrZEfBSaW4afk-Nzazs7gizes3VkNfXXuKmE/edit?gid=1726674730"
                ),
            )
            if st.button("📥 Recargar desde Google Sheets"):
                export_url = re.sub(r"/edit.*", "/export?format=csv", url_input)
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

        # ── Line/Ramo filter ──────────────────────────────────────────────────
        selected_lineas = None
        df_current = st.session_state.get("df_raw")
        if df_current is not None:
            cols = detect_columns(df_current)
            linea_col = cols.get("linea")
            if linea_col:
                lineas_disponibles = sorted(df_current[linea_col].dropna().unique())
            else:
                lineas_disponibles = list(ATTRIBUTE_LABELS.values())

            selected_lineas = st.multiselect(
                "🎯 Filtrar por Línea/Ramo:",
                options=lineas_disponibles,
                default=lineas_disponibles,
                help="Selecciona las líneas de negocio a analizar",
            )
            if selected_lineas:
                st.info(f"📊 Filtrando por {len(selected_lineas)} línea(s)")

        # ── HF token banner ───────────────────────────────────────────────────
        hf_token_detected = bool(get_hf_token())
        if hf_token_detected:
            st.success("🤗 HuggingFace token detectado — IA remota habilitada")
        else:
            st.info("ℹ️ No hay HF token configurado — usando modelo local (BETO) como fallback")

        st.markdown("---")
        analyze_btn = st.button(
            "🔍 Analizar Sentimientos", type="primary", use_container_width=True
        )

        st.markdown("---")
        st.markdown(
            "<small>Modelo: BETO (finiteautomata/beto-sentiment-analysis)</small>",
            unsafe_allow_html=True,
        )

    return selected_lineas, analyze_btn


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    selected_lineas, analyze_btn = render_sidebar()

    df_raw = st.session_state.get("df_raw", None)
    df_results = st.session_state.get("df_results", None)

    if df_raw is None and not analyze_btn:
        render_welcome()
        return

    if analyze_btn:
        if df_raw is None:
            st.warning("⚠️ Primero carga los datos desde el panel lateral.")
            return

        # Check if data already has Atributo/Valor or needs detection
        cols = detect_columns(df_raw)
        col_attr = cols.get("atributo")
        col_val = cols.get("valor")

        if col_attr is not None and col_val is not None:
            df_filtered = filter_open_responses(df_raw, selected_lineas)
        elif "Atributo" in df_raw.columns and "Valor" in df_raw.columns:
            # Already pre-filtered (e.g. sample data)
            df_filtered = df_raw.copy()
            if "linea_negocio" not in df_filtered.columns:
                df_filtered["linea_negocio"] = df_filtered["Atributo"].apply(
                    lambda a: ATTRIBUTE_LABELS.get(a, "General")
                )
        else:
            # Fallback for plain CSV with just a text column
            df_filtered = df_raw.copy()
            if "Valor" not in df_filtered.columns:
                df_filtered["Valor"] = df_filtered.iloc[:, 0]
            if "Atributo" not in df_filtered.columns:
                df_filtered["Atributo"] = "General"
            if "linea_negocio" not in df_filtered.columns:
                df_filtered["linea_negocio"] = "General"

        if df_filtered.empty:
            st.error(
                "❌ No se encontraron respuestas con los atributos seleccionados. "
                "Verifica que el archivo tenga columnas Atributo/Valor con preguntas abiertas."
            )
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dashboard Premium",
            "🎯 Análisis 3D",
            "💬 Explorador de Comentarios",
            "☁️ Palabras Clave",
            "🤖 Insights con IA",
            "📥 Exportar",
        ])
        with tab1:
            render_tab_dashboard(df_results)
        with tab2:
            render_tab_3d(df_results)
        with tab3:
            render_tab_comments(df_results)
        with tab4:
            render_tab_keywords(df_results)
        with tab5:
            render_tab_ai(df_results)
        with tab6:
            render_tab_export(df_results)
    elif df_raw is not None:
        st.info("👈 Haz clic en **🔍 Analizar Sentimientos** para iniciar el análisis.")
    else:
        render_welcome()


if __name__ == "__main__":
    main()
