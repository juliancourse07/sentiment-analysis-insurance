# 🎯 Sistema de Análisis de Sentimientos para Aseguradora

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![BETO](https://img.shields.io/badge/Model-BETO%20(BERT%20ES)-orange)
![Groq](https://img.shields.io/badge/AI-Groq%20Llama%203.1-purple)
![License](https://img.shields.io/badge/License-MIT-green)

Aplicación web interactiva construida con **Streamlit** para analizar sentimientos en el feedback de clientes de una aseguradora. Utiliza **BETO** (BERT en español fine-tuned) combinado con un análisis híbrido de keywords especializadas del sector asegurador, más **IA contextual** con Groq para insights del mercado colombiano.

---

## ✨ Características principales

- 🤖 **Modelo BETO** (`finiteautomata/beto-sentiment-analysis`) — BERT fine-tuned en español
- 🧠 **Análisis híbrido** — combina predicción del modelo con keywords del sector asegurador
- 📊 **5 categorías** — POSITIVO, NEGATIVO, NEUTRAL, MIXTO
- 🔗 **Carga automática desde Google Sheets** — pestaña CLIENTES, sin necesidad de botón
- 🎯 **Filtros inteligentes por Línea/Ramo** — en el panel lateral
- 🔍 **Filtrado flexible con regex** — detecta atributos con variaciones de texto
- 📈 **Visualizaciones avanzadas**: Gauges, 3D scatter, Sankey, Radar, heatmap, tendencia
- 🤖 **IA Contextual** con Groq (Llama 3.1 70B) — insights del sector asegurador colombiano
- ☁️ **Nube de palabras** con filtro por sentimiento
- 📥 **Exportación** a CSV (UTF-8 BOM) y Excel multi-hoja listos para Power BI

---

## 📸 Capturas de pantalla

> *Las capturas se generan automáticamente al desplegar la aplicación.*

| Dashboard Premium | Análisis 3D |
|---|---|
| *(screenshot)* | *(screenshot)* |

| Insights con IA | Palabras Clave |
|---|---|
| *(screenshot)* | *(screenshot)* |

---

## 🚀 Instalación y ejecución local

### Requisitos
- Python 3.10+
- pip

### Pasos

```bash
# 1. Clona el repositorio
git clone https://github.com/juliancourse07/sentiment-analysis-insurance.git
cd sentiment-analysis-insurance

# 2. Crea un entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Instala dependencias
pip install -r requirements.txt

# 4. Inicia la aplicación
streamlit run app.py
```

O usa el script incluido:

```bash
chmod +x run.sh && ./run.sh
```

La aplicación estará disponible en `http://localhost:8501`.

---

## 🗂️ Estructura del proyecto

```
sentiment-analysis-insurance/
├── app.py                          # Aplicación principal Streamlit
├── requirements.txt                # Dependencias Python
├── Dockerfile                      # Imagen Docker para despliegue
├── run.sh                          # Script de instalación y arranque
├── .streamlit/
│   ├── config.toml                 # Tema y configuración de Streamlit
│   └── secrets.toml.example        # Ejemplo de configuración de secrets
├── README.md                       # Este archivo
├── DEPLOYMENT.md                   # Guía de despliegue
└── POWER_BI_GUIDE.md               # Guía de integración con Power BI
```

---

## 🤖 Configuración de IA Contextual (Opcional)

Para habilitar análisis profundo con IA especializada en el sector asegurador colombiano:

1. **Obtén una API key gratis** en: https://console.groq.com
2. **Crea el archivo** `.streamlit/secrets.toml` (copia desde el ejemplo):
   ```toml
   GROQ_API_KEY = "gsk_tu_api_key_aqui"
   ```
3. **O configura** la variable de entorno:
   ```bash
   export GROQ_API_KEY="gsk_tu_api_key_aqui"
   ```

Sin API key, la app funciona con análisis estadístico tradicional (modo fallback automático).

---

## 🤗 Configuración de HuggingFace API Token (Opcional)

Para habilitar análisis remoto con modelos de HuggingFace Inference API:

> ⚠️ **Seguridad**: Nunca incluyas el token directamente en el código ni en el repositorio.

### Localmente

```bash
export HF_API_TOKEN="hf_tu_token_aqui"
streamlit run app.py
```

### En Streamlit Cloud

1. Ve a tu app en [share.streamlit.io](https://share.streamlit.io).
2. Haz clic en **Manage app → Secrets**.
3. Agrega:
   ```toml
   HF_API_TOKEN = "hf_tu_token_aqui"
   ```
4. Guarda y espera ~1 minuto a que se propague.

### En `.streamlit/secrets.toml` (solo local, no subas este archivo al repo)

```toml
HF_API_TOKEN = "hf_tu_token_aqui"
```

> El archivo `.streamlit/secrets.toml` está en `.gitignore` — nunca lo incluyas en el repositorio.

Si no se configura ningún token, la app usará automáticamente el **modelo local BETO** como fallback.
Un banner en el sidebar mostrará el estado del proveedor activo.

---

## 🧰 Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| [Streamlit](https://streamlit.io) | Framework de aplicación web |
| [BETO / HuggingFace Transformers](https://huggingface.co/finiteautomata/beto-sentiment-analysis) | Modelo de NLP en español |
| [Groq API](https://console.groq.com) | IA contextual (Llama 3.1 70B) |
| [Plotly](https://plotly.com/python/) | Visualizaciones interactivas |
| [Pandas](https://pandas.pydata.org/) | Procesamiento de datos |
| [WordCloud](https://github.com/amueller/word_cloud) | Nube de palabras |
| [openpyxl](https://openpyxl.readthedocs.io/) | Exportación a Excel |

---

## 🔗 Configuración de Google Sheets

La URL de Google Sheets está **pre-configurada** y se carga **automáticamente** al iniciar:

```
https://docs.google.com/spreadsheets/d/1OUzUl5UDrZEfBSaW4afk-Nzazs7gizes3VkNfXXuKmE/edit?gid=1726674730
```
**Pestaña**: `CLIENTES` (GID: `1726674730`)

**Estructura esperada del Sheet:**

| Atributo | Valor |
|---|---|
| Autos, ¿Cuéntanos qué factores contribuyeron... | El proceso fue fácil... |
| Vida, ¿Cuéntanos qué factores contribuyeron... | Muy complicado el trámite... |

El sistema usa **búsqueda flexible con regex** para detectar los atributos, incluso con variaciones de texto.

---

## 🌐 Despliegue en Streamlit Cloud

1. Haz fork de este repositorio en tu cuenta de GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io).
3. Haz clic en **"New app"**.
4. Selecciona el repositorio, rama `main` y archivo `app.py`.
5. (Opcional) Configura `GROQ_API_KEY` en **Secrets** para IA contextual.
6. Haz clic en **"Deploy"**.

El modelo BETO se descargará automáticamente en el primer arranque (~400 MB).

Consulta [DEPLOYMENT.md](DEPLOYMENT.md) para más opciones de despliegue.

---

## 🛠️ Troubleshooting

| Problema | Solución |
|---|---|
| `ModuleNotFoundError: torch` | Ejecuta `pip install torch` |
| Google Sheets no carga | Verifica que el Sheet sea público (compartido con lectura) |
| Modelo lento en primera carga | Normal — BETO descarga ~400 MB la primera vez |
| `OSError` en WordCloud | Instala `pip install wordcloud` |
| Error de encoding en CSV | Usa la opción CSV con BOM o Excel |
| IA no disponible | Configura `GROQ_API_KEY` en secrets.toml o variable de entorno |

---

## 🤝 Contribuciones

1. Haz fork del proyecto.
2. Crea tu rama: `git checkout -b feature/mi-mejora`.
3. Haz commit: `git commit -m "Agrega mi mejora"`.
4. Push: `git push origin feature/mi-mejora`.
5. Abre un Pull Request.

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
