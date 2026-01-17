# Docu AI: Asistente RAG para Análisis de Documentación Técnica

Este proyecto consiste en la implementación de un sistema de **Generación Aumentada por Recuperación (RAG)** desarrollado como parte de la formación en **Desarrollo de Aplicaciones con Modelos de Lenguaje (LLMs)** dictada por la **UTN (Universidad Tecnológica Nacional)**.

El sistema permite cargar documentos PDF locales y realizar consultas en lenguaje natural, obteniendo respuestas precisas basadas exclusivamente en el contenido del archivo, eliminando alucinaciones del modelo.

## 🛠️ Stack Tecnológico
- **Lenguaje:** Python 3.10+
- **Orquestador:** LangChain
- **LLM:** OpenAI GPT-4o-mini (vía API)
- **Embeddings:** OpenAI Embeddings
- **Vector Store:** ChromaDB
- **Entorno:** Dotenv para gestión de variables de entorno

## 🧠 Arquitectura del Sistema
1. **Ingesta:** Carga de documentos mediante `PyPDFLoader`.
2. **Chunking:** Fragmentación semántica con `RecursiveCharacterTextSplitter` (1000 tokens/100 overlap).
3. **Vectorización:** Generación de embeddings vectoriales para representación numérica del texto.
4. **Recuperación:** Búsqueda por similitud de coseno en base de datos vectorial persistente.
5. **Generación:** Inyección de contexto relevante en el prompt del LLM para respuestas fundamentadas.



