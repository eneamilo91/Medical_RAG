import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle

# Initialize embeddings - using PubMed BERT for medical domain
embeddings = HuggingFaceEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings",
    model_kwargs={'device': 'cpu'}
)

print("**************************")
print("Embeddings initialized:", embeddings)
print("**************************")

# Load PDF documents from data directory
print("Loading PDF documents...")
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
print("**************************")
print(type(documents))
print("**************************")

# Split documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    length_function=len,
)
texts = text_splitter.split_documents(documents)

print(f"Created {len(texts)} text chunks")
print("**************************")
print(type(texts))
print("**************************")

# Create FAISS vector store
print("Creating FAISS vector store...")
try:
    # Create FAISS database from documents
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save the vector store locally
    vectorstore.save_local("faiss_index")
    
    print("FAISS Vector DB Successfully Created and Saved!")
    print("Vector store saved to 'faiss_index' directory")
    
    # Test the vector store
    print("\n**************************")
    print("Testing vector store with a sample query...")
    test_query = "What is cancer?"
    results = vectorstore.similarity_search(test_query, k=2)
    print(f"Found {len(results)} similar documents for test query: '{test_query}'")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.page_content[:100]}...")
    print("**************************")
    
except Exception as e:
    print(f"Error creating FAISS vector store: {e}")
    raise e