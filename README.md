# Sistema-RAG

Sistema de Recuperación y Generación Aumentada (RAG) modular usando LangChain y OpenAI para responder preguntas basadas en documentos PDF.

## 🚀 Características

- ✅ **Carga de PDFs**: Utiliza `PyPDFLoader` para extraer contenido de documentos PDF
- ✅ **Fragmentación inteligente**: `RecursiveCharacterTextSplitter` con chunks de 1000 caracteres y overlap de 100
- ✅ **Embeddings**: Generación de embeddings con `OpenAIEmbeddings`
- ✅ **Base de datos vectorial**: Almacenamiento persistente con `Chroma DB`
- ✅ **Q&A inteligente**: Sistema de preguntas y respuestas con `RetrievalQA` y `ChatOpenAI (gpt-4o-mini)`
- ✅ **Configuración segura**: Carga de API keys desde archivo `.env`

## 📋 Requisitos

- Python 3.8 o superior
- API Key de OpenAI

## 🔧 Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/raynarg/Sistema-RAG.git
cd Sistema-RAG
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Configurar variables de entorno**:
```bash
cp .env.example .env
```

Editar el archivo `.env` y agregar tu API key de OpenAI:
```
OPENAI_API_KEY=tu_api_key_aqui
```

## 📖 Uso

### Uso Básico

1. **Colocar tu documento PDF** en el directorio del proyecto con el nombre `documento.pdf` (o especificar la ruta en el código)

2. **Ejecutar el pipeline**:
```bash
python rag_pipeline.py
```

### Uso como Módulo

```python
from rag_pipeline import RAGPipeline

# Inicializar el pipeline con tu PDF
rag = RAGPipeline(pdf_path="mi_documento.pdf")

# Inicializar todo el sistema
rag.initialize(chunk_size=1000, chunk_overlap=100)

# Hacer una pregunta
response = rag.query("¿De qué trata el documento?")
print(response['result'])
```

### Uso Avanzado - Paso a Paso

```python
from rag_pipeline import RAGPipeline

# Crear instancia
rag = RAGPipeline(pdf_path="documento.pdf", persist_directory="./mi_db")

# Ejecutar cada paso manualmente
rag.load_pdf()                      # 1. Cargar PDF
rag.split_text(1000, 100)           # 2. Fragmentar texto
rag.create_vectorstore()            # 3. Crear embeddings y vectorstore
rag.setup_qa_chain("gpt-4o-mini")   # 4. Configurar cadena QA

# Realizar consultas
response = rag.query("¿Cuáles son los puntos principales?")
```

## 🏗️ Arquitectura

El pipeline RAG sigue estos pasos:

1. **Carga de Documento**: `PyPDFLoader` extrae el texto del PDF
2. **Fragmentación**: `RecursiveCharacterTextSplitter` divide el texto en chunks manejables
3. **Embeddings**: `OpenAIEmbeddings` genera representaciones vectoriales del texto
4. **Almacenamiento**: `Chroma DB` almacena los vectores para búsqueda eficiente
5. **Recuperación**: El sistema encuentra los fragmentos más relevantes para cada pregunta
6. **Generación**: `ChatOpenAI (gpt-4o-mini)` genera respuestas basadas en los fragmentos recuperados

```
PDF → PyPDFLoader → RecursiveTextSplitter → OpenAIEmbeddings → ChromaDB
                                                                    ↓
                                                                Retriever
                                                                    ↓
Usuario → Pregunta → RetrievalQA ← ChatOpenAI (gpt-4o-mini) ← Contexto
```

## 📦 Dependencias Principales

- **langchain**: Framework para aplicaciones con LLMs
- **langchain-openai**: Integraciones de OpenAI para LangChain
- **openai**: Cliente oficial de OpenAI
- **chromadb**: Base de datos vectorial
- **pypdf**: Lector de PDFs
- **python-dotenv**: Gestión de variables de entorno

## 🔒 Seguridad

- Las API keys se cargan desde archivos `.env` (no incluidos en el repositorio)
- El archivo `.env` está en `.gitignore` para prevenir commits accidentales
- Usa `.env.example` como plantilla para configurar tus credenciales

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👨‍💻 Autor

raynarg

## 🙏 Agradecimientos

- [LangChain](https://www.langchain.com/) por el excelente framework
- [OpenAI](https://openai.com/) por sus modelos de lenguaje
- La comunidad de código abierto