"""
Sistema RAG Modular - Pipeline de Recuperación y Generación Aumentada
Este módulo implementa un pipeline RAG completo usando LangChain para responder
preguntas basadas en documentos PDF.
"""

import os
from typing import Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


class RAGPipeline:
    """
    Pipeline modular para RAG (Retrieval-Augmented Generation).
    
    Características:
    - Carga de PDFs con PyPDFLoader
    - Fragmentación de texto con RecursiveCharacterTextSplitter
    - Embeddings con OpenAIEmbeddings
    - Almacenamiento vectorial con Chroma DB
    - Cadena de preguntas y respuestas con RetrievalQA
    """
    
    def __init__(self, pdf_path: str, persist_directory: str = "./chroma_db"):
        """
        Inicializa el pipeline RAG.
        
        Args:
            pdf_path: Ruta al archivo PDF a procesar
            persist_directory: Directorio para persistir la base de datos Chroma
        """
        # Cargar variables de entorno desde .env
        load_dotenv()
        
        # Verificar que existe la API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY no encontrada. "
                "Por favor, crea un archivo .env con tu API key de OpenAI."
            )
        
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.documents = None
        self.text_chunks = None
        self.vectorstore = None
        self.qa_chain = None
        
    def load_pdf(self):
        """Carga el documento PDF usando PyPDFLoader."""
        print(f"Cargando PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load()
        print(f"PDF cargado exitosamente. Total de páginas: {len(self.documents)}")
        return self.documents
    
    def split_text(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Fragmenta el texto en chunks usando RecursiveCharacterTextSplitter.
        
        Args:
            chunk_size: Tamaño de cada fragmento (default: 1000)
            chunk_overlap: Superposición entre fragmentos (default: 100)
        """
        if self.documents is None:
            raise ValueError("Primero debes cargar el PDF usando load_pdf()")
        
        print(f"Fragmentando texto (chunk_size={chunk_size}, overlap={chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Texto fragmentado en {len(self.text_chunks)} chunks")
        return self.text_chunks
    
    def create_vectorstore(self):
        """
        Crea embeddings con OpenAIEmbeddings y almacena en Chroma DB.
        """
        if self.text_chunks is None:
            raise ValueError("Primero debes fragmentar el texto usando split_text()")
        
        print("Creando embeddings y almacenando en Chroma DB...")
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.text_chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        print("Vectorstore creado exitosamente")
        return self.vectorstore
    
    def setup_qa_chain(self, model_name: str = "gpt-4o-mini"):
        """
        Configura la cadena RetrievalQA con ChatOpenAI.
        
        Args:
            model_name: Nombre del modelo de OpenAI (default: gpt-4o-mini)
        """
        if self.vectorstore is None:
            raise ValueError("Primero debes crear el vectorstore usando create_vectorstore()")
        
        print(f"Configurando cadena RetrievalQA con modelo {model_name}...")
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Recuperar los 3 fragmentos más relevantes
            ),
            return_source_documents=True
        )
        print("Cadena RetrievalQA configurada exitosamente")
        return self.qa_chain
    
    def initialize(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Inicializa todo el pipeline de una sola vez.
        
        Args:
            chunk_size: Tamaño de cada fragmento (default: 1000)
            chunk_overlap: Superposición entre fragmentos (default: 100)
        """
        self.load_pdf()
        self.split_text(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.create_vectorstore()
        self.setup_qa_chain()
        print("\n✓ Pipeline RAG inicializado completamente")
        
    def query(self, question: str) -> dict:
        """
        Realiza una consulta al sistema RAG.
        
        Args:
            question: Pregunta a responder basada en el PDF
            
        Returns:
            dict con 'result' (respuesta) y 'source_documents' (fuentes)
        """
        if self.qa_chain is None:
            raise ValueError(
                "Primero debes configurar la cadena QA. "
                "Ejecuta initialize() o setup_qa_chain()"
            )
        
        print(f"\nPregunta: {question}")
        response = self.qa_chain.invoke({"query": question})
        print(f"Respuesta: {response['result']}")
        return response


def main():
    """
    Función principal para demostrar el uso del pipeline RAG.
    """
    # Ejemplo de uso
    print("="*60)
    print("Sistema RAG Modular - LangChain + OpenAI")
    print("="*60)
    
    # Verificar si existe un PDF de ejemplo
    pdf_path = "documento.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠ No se encontró el archivo '{pdf_path}'")
        print("Por favor, coloca un archivo PDF en el directorio actual.")
        print("\nPara usar el sistema:")
        print("1. Coloca tu PDF en el directorio actual")
        print("2. Crea un archivo .env con tu OPENAI_API_KEY")
        print("3. Ejecuta este script nuevamente")
        return
    
    # Inicializar el pipeline
    rag = RAGPipeline(pdf_path=pdf_path)
    
    try:
        # Inicializar todo el pipeline
        rag.initialize(chunk_size=1000, chunk_overlap=100)
        
        # Hacer consultas de ejemplo
        print("\n" + "="*60)
        print("Sistema listo para responder preguntas")
        print("="*60)
        
        # Ejemplo de consulta
        example_questions = [
            "¿De qué trata el documento?",
            "¿Cuáles son los puntos principales?",
        ]
        
        for question in example_questions:
            response = rag.query(question)
            print(f"\nFuentes consultadas: {len(response['source_documents'])} fragmentos")
            print("-"*60)
        
        # Modo interactivo
        print("\n" + "="*60)
        print("Modo Interactivo - Escribe 'salir' para terminar")
        print("="*60)
        
        while True:
            user_question = input("\nTu pregunta: ").strip()
            if user_question.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            
            if user_question:
                rag.query(user_question)
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
