# 🎯 Sistema de Análisis de Sentimientos para Aseguradora

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![BETO](https://img.shields.io/badge/Model-BETO%20(BERT%20ES)-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Aplicación web interactiva construida con **Streamlit** para analizar sentimientos en el feedback de clientes de una aseguradora. Utiliza **BETO** (BERT en español fine-tuned) combinado con un análisis híbrido de keywords especializadas del sector asegurador.

---

## ✨ Características principales

- 🤖 **Modelo BETO** (`finiteautomata/beto-sentiment-analysis`) — BERT fine-tuned en español
- 🧠 **Análisis híbrido** — combina predicción del modelo con keywords del sector asegurador
- 📊 **5 categorías** — POSITIVO, NEGATIVO, NEUTRAL, MIXTO
- 🔗 **Carga desde Google Sheets** — URL pre-configurada lista para usar
- 📂 **Carga CSV** — sube tu propio archivo
- 🧪 **Datos de ejemplo** — 50 filas sintéticas para probar sin datos reales
- 📈 **Visualizaciones interactivas** con Plotly (dona, mapa de calor, tendencia temporal, histograma)
- ☁️ **Nube de palabras** con filtro por sentimiento
- 📥 **Exportación** a CSV (UTF-8 BOM) y Excel multi-hoja listos para Power BI

---

## 📸 Capturas de pantalla

> *Las capturas se generan automáticamente al desplegar la aplicación.*

| Dashboard General | Análisis Detallado |
|---|---|
| *(screenshot)* | *(screenshot)* |

| Comentarios | Nube de Palabras |
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
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt        # Dependencias Python
├── Dockerfile              # Imagen Docker para despliegue
├── run.sh                  # Script de instalación y arranque
├── .streamlit/
│   └── config.toml         # Tema y configuración de Streamlit
├── README.md               # Este archivo
├── DEPLOYMENT.md           # Guía de despliegue
└── POWER_BI_GUIDE.md       # Guía de integración con Power BI
```

---

## 🧰 Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| [Streamlit](https://streamlit.io) | Framework de aplicación web |
| [BETO / HuggingFace Transformers](https://huggingface.co/finiteautomata/beto-sentiment-analysis) | Modelo de NLP en español |
| [Plotly](https://plotly.com/python/) | Visualizaciones interactivas |
| [Pandas](https://pandas.pydata.org/) | Procesamiento de datos |
| [WordCloud](https://github.com/amueller/word_cloud) | Nube de palabras |
| [openpyxl](https://openpyxl.readthedocs.io/) | Exportación a Excel |

---

## 🔗 Configuración de Google Sheets

La URL de Google Sheets ya está **pre-configurada** en el código:

```
https://docs.google.com/spreadsheets/d/1OUzUl5UDrZEfBSaW4afk-Nzazs7gizes3VkNfXXuKmE/edit?gid=1726674730
```

El código convierte automáticamente la URL de edición al formato de exportación CSV.

**Estructura esperada del Sheet:**
| Atributo | Valor |
|---|---|
| Autos, ¿Cuéntanos... | El proceso fue fácil... |
| Vida, ¿Cuéntanos... | Muy complicado el trámite... |

---

## 🌐 Despliegue en Streamlit Cloud

1. Haz fork de este repositorio en tu cuenta de GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io).
3. Haz clic en **"New app"**.
4. Selecciona el repositorio, rama `main` y archivo `app.py`.
5. Haz clic en **"Deploy"**.

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
