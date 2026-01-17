# Guía Rápida - Sistema RAG

## Instalación Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key
cp .env.example .env
# Editar .env y agregar tu OPENAI_API_KEY
```

## Uso Básico

### Opción 1: Modo Simple

```python
from rag_pipeline import RAGPipeline

# Inicializar y procesar
rag = RAGPipeline(pdf_path="mi_documento.pdf")
rag.initialize()

# Hacer preguntas
respuesta = rag.query("¿De qué trata el documento?")
print(respuesta['result'])
```

### Opción 2: Usando el Script de Ejemplo

```bash
# Modo básico
python ejemplo.py

# Modo paso a paso
python ejemplo.py paso-a-paso

# Modo interactivo
python ejemplo.py interactivo
```

### Opción 3: Ejecutar el Pipeline Principal

```bash
# Asegúrate de tener un archivo documento.pdf en el directorio
python rag_pipeline.py
```

## Parámetros de Configuración

### Fragmentación de Texto

```python
rag.split_text(
    chunk_size=1000,      # Tamaño de cada fragmento (caracteres)
    chunk_overlap=100     # Superposición entre fragmentos
)
```

### Modelo de OpenAI

```python
rag.setup_qa_chain(model_name="gpt-4o-mini")  # Por defecto
# O usar otros modelos:
# rag.setup_qa_chain(model_name="gpt-4")
# rag.setup_qa_chain(model_name="gpt-3.5-turbo")
```

### Directorio de Persistencia

```python
rag = RAGPipeline(
    pdf_path="documento.pdf",
    persist_directory="./mi_base_datos"  # Personalizar ubicación
)
```

## Estructura de Respuesta

```python
respuesta = rag.query("¿Pregunta?")

# respuesta contiene:
{
    'result': 'La respuesta generada por el modelo...',
    'source_documents': [
        # Lista de fragmentos del PDF usados para generar la respuesta
        Document(page_content='...', metadata={'page': 1, 'source': '...'}),
        Document(page_content='...', metadata={'page': 3, 'source': '...'}),
        # ...
    ]
}
```

## Casos de Uso Comunes

### 1. Análisis de Documento Único

```python
rag = RAGPipeline("informe.pdf")
rag.initialize()

preguntas = [
    "¿Cuál es el resumen ejecutivo?",
    "¿Qué conclusiones se presentan?",
    "¿Cuáles son las recomendaciones?"
]

for pregunta in preguntas:
    respuesta = rag.query(pregunta)
    print(f"P: {pregunta}")
    print(f"R: {respuesta['result']}\n")
```

### 2. Extracción de Información Específica

```python
rag = RAGPipeline("contrato.pdf")
rag.initialize()

# Buscar información específica
info = rag.query("¿Cuáles son las fechas mencionadas en el documento?")
print(info['result'])
```

### 3. Comparación de Contenido

```python
rag = RAGPipeline("manual_v2.pdf")
rag.initialize()

respuesta = rag.query(
    "¿Cuáles son las diferencias principales mencionadas "
    "respecto a versiones anteriores?"
)
print(respuesta['result'])
```

## Solución de Problemas

### Error: "OPENAI_API_KEY no encontrada"

```bash
# Verifica que existe el archivo .env
ls -la .env

# Verifica el contenido
cat .env
# Debe contener: OPENAI_API_KEY=sk-...
```

### Error: "No se encontró el archivo PDF"

```python
import os
print(os.path.exists("mi_documento.pdf"))  # Debe ser True

# O usa ruta absoluta
rag = RAGPipeline("/ruta/completa/a/documento.pdf")
```

### Error de Memoria con PDFs Grandes

```python
# Reduce el tamaño de los chunks
rag.split_text(chunk_size=500, chunk_overlap=50)

# O ajusta el número de documentos recuperados
from langchain.chains import RetrievalQA
rag.vectorstore.as_retriever(search_kwargs={"k": 2})  # En lugar de 3
```

## Tips y Mejores Prácticas

1. **PDFs con Imágenes**: Funciona mejor con PDFs que contienen texto extraíble
2. **Preguntas Específicas**: Formula preguntas claras y específicas
3. **Chunks Apropiados**: Para textos técnicos, considera aumentar chunk_size a 1500
4. **Temperatura**: El modelo usa temperature=0 para respuestas consistentes
5. **Persistencia**: La base de datos Chroma se guarda automáticamente

## Requisitos del Sistema

- Python 3.8+
- Conexión a Internet (para API de OpenAI)
- ~500MB de espacio en disco (dependencias)
- API Key válida de OpenAI

## Limitaciones

- Solo procesa PDFs con texto extraíble (no funciona bien con PDFs escaneados)
- Requiere conexión a Internet para generar embeddings y respuestas
- El costo depende del uso de la API de OpenAI
- La calidad depende de la legibilidad del PDF original

## Seguridad

- ⚠️ NUNCA compartas tu archivo `.env` o tu API key
- El archivo `.env` está en `.gitignore` por defecto
- Usa `.env.example` como plantilla solamente
