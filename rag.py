import os
import warnings
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Document conversion
def load_and_convert_document(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

# Splitting markdown content into chunks
def get_markdown_splits(markdown_content):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return markdown_splitter.split_text(markdown_content)

# Embedding and vector store setup
def setup_vector_store(chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    single_vector = embeddings.embed_query("this is some text data")
    index = faiss.IndexFlatL2(len(single_vector))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

# Formatting documents for RAG
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Setting up the RAG chain
def create_rag_chain(retriever):
    prompt = """
        tu és um assistente para tarefas de perguntas e respostas. Use as seguintes partes do contexto recuperado para responder à pergunta.  
        Se não souber a resposta, apenas diga que não sabe.  
        Responda em tópicos. Certifique-se de que sua resposta seja relevante para a pergunta e baseada apenas no contexto fornecido RESPONDA SEMPRE EM PORTUGÊS.  

        Pergunta: {question}  
        Contexto: {context}  
        Resposta:
    """
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(prompt)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

# Main execution logic
if __name__ == "__main__":
    # Load document
    source = "rag-dataset/FVG.pdf "
    markdown_content = load_and_convert_document(source)
    chunks = get_markdown_splits(markdown_content)
    
    # Create vector store
    vector_store = setup_vector_store(chunks)
    
    # Setup retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})

    # Create RAG chain
    rag_chain = create_rag_chain(retriever)
    
    # Questions for retrieval
    questions = [
        "Quem é o leonel messi?",
        #"Quais são as caracteristicas do indicador?",
        #"o que são fair value gaps?",
        #"Quando é que o preço se move em busca de liquidez"
    ]
    
    # Answer questions
    for question in questions:
        print(f"Question: {question}")
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 50 + "\n")
