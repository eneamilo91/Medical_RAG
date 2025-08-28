# FAISS-based RAG application
# uvicorn rag:app
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import os
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# LLM Configuration
local_llm = "BioMistral-7B.Q4_K_M.gguf"

config = {
    'max_new_tokens': 2048,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.2,
    'top_k': 50,
    'top_p': 1,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

# Initialize LLM
print("Initializing LLM...")
llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",
    **config
)
print("LLM Initialized....")

# Prompt template for medical queries
prompt_template = """
Use the following pieces of medical information to provide accurate responses to the user's questions. Please refrain from speculation or providing false information.

Context: {context}
Question: {question}

Provide a concise and accurate answer based on the medical context.

Helpful answer:
"""

# Initialize embeddings (same as used in ingest.py)
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings",
    model_kwargs={'device': 'cpu'}
)

# Load FAISS vector store
print("Loading FAISS vector store...")
try:
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded successfully!")
except Exception as e:
    print(f"Error loading FAISS vector store: {e}")
    print("Please ensure you have run 'python ingest.py' first to create the vector store.")
    raise e

# Create prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    try:
        print(f"Processing query: {query}")
        
        # Create QA chain
        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True, 
            chain_type_kwargs=chain_type_kwargs, 
            verbose=True
        )
        
        # Get response
        response = qa(query)
        print("Response generated successfully")
        
        # Extract information
        answer = response['result']
        source_document = response['source_documents'][0].page_content
        
        # Handle metadata (FAISS metadata structure is different from Qdrant)
        metadata = response['source_documents'][0].metadata
        source_info = f"Source: {metadata.get('source', 'Unknown')}"
        if 'page' in metadata:
            source_info += f", Page: {metadata['page']}"
        source_info += " (FAISS Vector Store)"
        
        # Prepare response
        response_data = {
            "answer": answer, 
            "source_document": source_document, 
            "doc": source_info
        }
        
        response_json = jsonable_encoder(json.dumps(response_data))
        return Response(content=response_json, media_type="application/json")
        
    except Exception as e:
        print(f"Error processing query: {e}")
        error_response = {
            "answer": f"Error processing your query: {str(e)}", 
            "source_document": "N/A", 
            "doc": "Error"
        }
        error_json = jsonable_encoder(json.dumps(error_response))
        return Response(content=error_json, media_type="application/json")