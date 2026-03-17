# 📊 Guía de Integración con Power BI

## 1. Exportar datos desde la aplicación

1. Ejecuta el análisis de sentimientos en la app.
2. Ve a la pestaña **📥 Exportar**.
3. Descarga el archivo **Excel** (múltiples hojas) o **CSV**.

---

## 2. Importar a Power BI Desktop

### Desde Excel

1. Abre Power BI Desktop.
2. Ve a **Inicio → Obtener datos → Excel**.
3. Selecciona el archivo `sentimientos_YYYYMMDD_HHMMSS.xlsx`.
4. Importa las hojas deseadas:
   - `Datos_Completos` — registros con sentimiento calculado
   - `Resumen` — métricas por categoría de sentimiento
   - `Por_Linea_Negocio` — resumen por línea de negocio
   - `Por_Atributo` — resumen por atributo

### Desde CSV

1. **Inicio → Obtener datos → Texto/CSV**.
2. Selecciona `sentimientos_YYYYMMDD_HHMMSS.csv`.
3. Verifica que el encoding sea **UTF-8** (el archivo incluye BOM para compatibilidad).

---

## 3. Columnas del dataset exportado

| Columna | Tipo | Descripción |
|---|---|---|
| `Atributo` | Texto | Atributo original del survey |
| `Valor` | Texto | Texto del comentario |
| `linea_negocio` | Texto | Línea (Autos, Fianzas, etc.) |
| `sentiment` | Texto | POSITIVO / NEGATIVO / NEUTRAL / MIXTO |
| `score` | Decimal | Score numérico (-1 a +1) |
| `confidence` | Decimal | Confianza del modelo (0 a 1) |
| `keywords_pos` | Entero | Cantidad de keywords positivas detectadas |
| `keywords_neg` | Entero | Cantidad de keywords negativas detectadas |
| `fecha` | Fecha | Fecha de la respuesta (si disponible) |

---

## 4. Medidas DAX recomendadas

```dax
-- % Positivo
% Positivo =
DIVIDE(
    COUNTROWS(FILTER(Datos_Completos, Datos_Completos[sentiment] = "POSITIVO")),
    COUNTROWS(Datos_Completos),
    0
) * 100

-- % Negativo
% Negativo =
DIVIDE(
    COUNTROWS(FILTER(Datos_Completos, Datos_Completos[sentiment] = "NEGATIVO")),
    COUNTROWS(Datos_Completos),
    0
) * 100

-- Score Promedio
Score Promedio =
AVERAGE(Datos_Completos[score])

-- Confianza Promedio
Confianza Promedio =
AVERAGE(Datos_Completos[confidence])

-- NPS Aproximado (Positivos - Negativos)
NPS Aproximado =
[% Positivo] - [% Negativo]

-- Total Respuestas
Total Respuestas =
COUNTROWS(Datos_Completos)

-- Índice de Satisfacción (0 a 100)
Índice Satisfacción =
([Score Promedio] + 1) / 2 * 100
```

---

## 5. Visualizaciones sugeridas

### Tarjetas KPI (Card)
- Total Respuestas
- % Positivo
- % Negativo
- NPS Aproximado
- Índice de Satisfacción

### Gráfico de anillo (Donut Chart)
- **Leyenda**: `sentiment`
- **Valores**: `COUNTROWS()`
- **Colores**: Ver sección de formato condicional

### Gráfico de barras apiladas (Stacked Bar)
- **Eje Y**: `linea_negocio`
- **Valores**: Count de registros
- **Leyenda**: `sentiment`

### Gráfico de líneas temporal
- **Eje X**: `fecha` (jerarquía Mes/Año)
- **Valores**: Count por `sentiment`

### Tabla de comentarios
- **Columnas**: `linea_negocio`, `sentiment`, `score`, `confidence`, `Valor`
- **Filtros**: sentiment, linea_negocio, confidence

### Mapa de calor (Matrix)
- **Filas**: `linea_negocio`
- **Columnas**: `sentiment`
- **Valores**: Count
- **Formato condicional**: Escala de colores

---

## 6. Formato condicional por sentimiento

En Power BI, agrega una columna calculada para el color:

```dax
Color Sentimiento =
SWITCH(
    Datos_Completos[sentiment],
    "POSITIVO", "#00B050",
    "NEGATIVO", "#FF0000",
    "NEUTRAL",  "#FFC000",
    "MIXTO",    "#7030A0",
    "#888888"
)
```

Usa esta columna como **Color de datos** en tus visualizaciones.

---

## 7. Drill-through

1. Crea una página de detalle llamada **"Detalle Comentarios"**.
2. En esa página, agrega un filtro de drill-through en el campo `linea_negocio`.
3. Agrega una tabla con `Valor`, `sentiment`, `score`, `confidence`.
4. Desde cualquier gráfico con `linea_negocio`, haz clic derecho → **Drill through → Detalle Comentarios**.

---

## 8. Filtros recomendados en el panel

- `sentiment` — Segmentación de datos (Slicer) horizontal
- `linea_negocio` — Slicer vertical con selección múltiple
- `fecha` — Slicer de rango de fechas
- `confidence` — Slicer numérico (≥ 0.70 para alta confianza)

---

## 9. Actualización automática de datos

Para actualizar los datos automáticamente:

1. Publica el informe en **Power BI Service**.
2. Configura una **puerta de enlace de datos** o usa **Power BI Dataflows**.
3. Programa la actualización diaria/semanal del dataset.

Alternativamente, exporta un CSV nuevo desde la app y reemplaza el archivo de origen — Power BI detectará los cambios al actualizar.
