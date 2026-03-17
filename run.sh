#!/bin/bash
set -e
echo "🚀 Instalando dependencias..."
pip install -r requirements.txt
echo "✅ Instalación completada"
echo "🌐 Iniciando aplicación Streamlit..."
streamlit run app.py
