# 🚀 Guía de Despliegue — Análisis de Sentimientos Aseguradora

## Opción 1: Local (pip + streamlit)

```bash
git clone https://github.com/juliancourse07/sentiment-analysis-insurance.git
cd sentiment-analysis-insurance
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`.

---

## Opción 2: Streamlit Cloud ⭐ (Recomendado)

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
2. Haz clic en **"New app"**.
3. Selecciona:
   - **Repository**: `juliancourse07/sentiment-analysis-insurance`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Haz clic en **"Deploy!"**.
5. Espera ~5-10 minutos mientras se instalan las dependencias y descarga el modelo.

### Notas Streamlit Cloud
- El modelo BETO (~400 MB) se descarga automáticamente en el primer arranque.
- Los datos en caché se borran al reiniciar la app.
- Plan gratuito soporta apps públicas con recursos limitados.

---

## Opción 3: Docker

```bash
# Construir imagen
docker build -t sentiment-insurance .

# Ejecutar contenedor
docker run -p 8501:8501 sentiment-insurance
```

Abre `http://localhost:8501`.

### Docker Compose (opcional)

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
```

```bash
docker-compose up
```

---

## Opción 4: Heroku

```bash
# Instala Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
heroku login
heroku create nombre-de-tu-app
heroku stack:set container

# Agrega Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

git add Procfile
git commit -m "Add Procfile"
git push heroku main
```

---

## Opción 5: Cloud Providers

### AWS (Elastic Beanstalk)

1. Instala [EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3.html).
2. Ejecuta `eb init` y selecciona Docker.
3. Ejecuta `eb create sentiment-env`.
4. Ejecuta `eb open`.

### Azure (Container Instances)

```bash
az acr create --resource-group myGroup --name myRegistry --sku Basic
az acr build --registry myRegistry --image sentiment-insurance .
az container create \
  --resource-group myGroup \
  --name sentiment-app \
  --image myRegistry.azurecr.io/sentiment-insurance \
  --ports 8501
```

### GCP (Cloud Run)

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/sentiment-insurance
gcloud run deploy --image gcr.io/PROJECT_ID/sentiment-insurance --platform managed --port 8501
```

---

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `STREAMLIT_SERVER_PORT` | Puerto del servidor | `8501` |
| `TRANSFORMERS_CACHE` | Directorio de caché del modelo | `~/.cache/huggingface` |

---

## Troubleshooting de despliegue

| Problema | Solución |
|---|---|
| `pip install` tarda mucho | Normal — PyTorch es pesado (~1 GB). Usa imagen base con torch pre-instalado. |
| `CUDA not available` | Ignorable — la app funciona en CPU. |
| Puerto 8501 bloqueado | Cambia con `--server.port=8080` |
| Memory error en Cloud | Aumenta la RAM asignada a ≥2 GB |
| Modelo no descarga | Verifica acceso a internet desde el servidor (huggingface.co) |
