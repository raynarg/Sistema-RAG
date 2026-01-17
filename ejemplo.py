"""
Ejemplo simple de uso del Sistema RAG
Este script muestra cómo usar el pipeline RAG de forma básica.
"""

from rag_pipeline import RAGPipeline


def ejemplo_basico():
    """Ejemplo básico de uso del sistema RAG."""
    
    print("="*60)
    print("Ejemplo de uso del Sistema RAG")
    print("="*60)
    
    # Ruta al PDF (ajusta según tu archivo)
    pdf_path = "documento.pdf"
    
    # 1. Crear instancia del pipeline
    print("\n1. Inicializando el pipeline RAG...")
    rag = RAGPipeline(pdf_path=pdf_path)
    
    # 2. Inicializar todo el sistema
    print("\n2. Procesando el documento...")
    rag.initialize(chunk_size=1000, chunk_overlap=100)
    
    # 3. Hacer preguntas
    print("\n3. Realizando consultas...")
    
    preguntas = [
        "¿De qué trata este documento?",
        "¿Cuáles son los conceptos principales?",
        "Resume el contenido en 3 puntos"
    ]
    
    for pregunta in preguntas:
        print("\n" + "-"*60)
        response = rag.query(pregunta)
        print(f"\nFuentes utilizadas: {len(response['source_documents'])} fragmentos")
    
    print("\n" + "="*60)
    print("Ejemplo completado")
    print("="*60)


def ejemplo_paso_a_paso():
    """Ejemplo mostrando cada paso del pipeline por separado."""
    
    print("="*60)
    print("Ejemplo paso a paso del Sistema RAG")
    print("="*60)
    
    pdf_path = "documento.pdf"
    
    # Crear instancia
    rag = RAGPipeline(pdf_path=pdf_path, persist_directory="./ejemplo_db")
    
    # Paso 1: Cargar PDF
    print("\nPaso 1: Cargando PDF...")
    documentos = rag.load_pdf()
    print(f"  → Cargadas {len(documentos)} páginas")
    
    # Paso 2: Fragmentar texto
    print("\nPaso 2: Fragmentando texto...")
    chunks = rag.split_text(chunk_size=1000, chunk_overlap=100)
    print(f"  → Creados {len(chunks)} fragmentos de texto")
    
    # Paso 3: Crear vectorstore
    print("\nPaso 3: Creando embeddings y vectorstore...")
    vectorstore = rag.create_vectorstore()
    print("  → Vectorstore creado exitosamente")
    
    # Paso 4: Configurar cadena QA
    print("\nPaso 4: Configurando cadena de preguntas y respuestas...")
    qa_chain = rag.setup_qa_chain(model_name="gpt-4o-mini")
    print("  → Cadena QA configurada")
    
    # Paso 5: Realizar consulta
    print("\nPaso 5: Realizando consulta...")
    response = rag.query("¿De qué trata el documento?")
    
    print("\n" + "="*60)
    print("Ejemplo completado")
    print("="*60)


def ejemplo_interactivo():
    """Ejemplo de sesión interactiva con el usuario."""
    
    print("="*60)
    print("Sistema RAG - Modo Interactivo")
    print("="*60)
    
    pdf_path = input("\nRuta al archivo PDF: ").strip() or "documento.pdf"
    
    # Inicializar sistema
    rag = RAGPipeline(pdf_path=pdf_path)
    rag.initialize()
    
    print("\n" + "="*60)
    print("Sistema listo. Escribe 'salir' para terminar.")
    print("="*60)
    
    while True:
        pregunta = input("\n📝 Tu pregunta: ").strip()
        
        if pregunta.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n¡Hasta luego! 👋")
            break
        
        if not pregunta:
            continue
        
        try:
            response = rag.query(pregunta)
            print(f"\n📚 Fuentes: {len(response['source_documents'])} fragmentos consultados")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        modo = sys.argv[1]
        if modo == "paso-a-paso":
            ejemplo_paso_a_paso()
        elif modo == "interactivo":
            ejemplo_interactivo()
        else:
            print("Modos disponibles: basico, paso-a-paso, interactivo")
            ejemplo_basico()
    else:
        ejemplo_basico()
