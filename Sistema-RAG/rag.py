from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
import os
import sys

class SimpleRAGPipeline:
    
    def __init__(self, pdf_path: str, db_path: str = "./chroma_db"):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
        
        self.llm = Ollama(model="llama3.2", temperature=0)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
    
    def query(self, question: str) -> dict:
        source_documents = self.retriever.invoke(question)
        
        context = "\n\n".join([doc.page_content for doc in source_documents])
        
        prompt = f"""Utiliza el siguiente contexto para responder a la pregunta. Si no sabes la respuesta basándote en el contexto, di que no lo sabes.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
        
        response = self.llm.invoke(prompt)
        
        return {
            "result": response,
            "source_documents": source_documents,
            "query": question
        }


def create_rag_pipeline(pdf_path: str, db_path: str = "./chroma_db"):
    return SimpleRAGPipeline(pdf_path, db_path)


def query_rag(rag_pipeline, question: str) -> dict:
    return rag_pipeline.query(question)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file = os.path.join(script_dir, "document.pdf")
    
    if not os.path.exists(pdf_file):
        print(f"Error: El archivo '{pdf_file}' no existe.")
        print(f"Por favor, coloca un archivo PDF en la ruta correcta o modifica la variable pdf_file.")
        sys.exit(1)
    
    rag = create_rag_pipeline(pdf_file)
    
    response = query_rag(rag, "¿Cuál es el tema principal del documento?")
    print(f"Answer: {response['result']}")
    print(f"\nSources: {[doc.metadata.get('source') for doc in response['source_documents']]}")