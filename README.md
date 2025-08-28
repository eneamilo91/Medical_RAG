# Medical FAISS RAG System - Complete User Manual

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Project Setup](#project-setup)
5. [Running the Application](#running-the-application)
6. [Usage Guide](#usage-guide)
7. [Troubleshooting](#troubleshooting)
8. [File Structure](#file-structure)
9. [Technical Details](#technical-details)

## Overview

This Medical RAG (Retrieval-Augmented Generation) system uses FAISS vector database for efficient similarity search and the BioMistral-7B model for medical question answering. The system processes medical documents, creates embeddings, and provides accurate responses to medical queries.

**Key Features:**
- 🏠 **Local Processing**: Everything runs on your CPU
- 🧠 **Medical AI**: BioMistral-7B fine-tuned for medical domain
- 🔍 **FAISS Vector Search**: Fast similarity search
- 📚 **PubMed-BERT**: Medical domain embeddings
- 🌐 **Web Interface**: User-friendly web application

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: Multi-core processor (Intel/AMD)
- **Storage**: At least 10GB free space
- **OS**: Windows with WSL2

### Software Requirements
- Windows 10/11 with WSL2 enabled
- VS Code with Remote-WSL extension
- Python 3.8 or higher
- Git

## Installation Guide

### Step 1: Enable WSL2 on Windows

1. **Enable WSL feature:**
   ```powershell
   # Run PowerShell as Administrator
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

2. **Restart your computer**

3. **Set WSL2 as default:**
   ```powershell
   wsl --set-default-version 2
   ```

4. **Install Ubuntu:**
   - Open Microsoft Store
   - Search for "Ubuntu 22.04 LTS"
   - Install and launch
   - Set up username and password

### Step 2: Install VS Code and Extensions

1. **Download and install VS Code** from https://code.visualstudio.com/
2. **Install WSL extension:**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Remote - WSL"
   - Install the extension

### Step 3: Set Up WSL Environment

1. **Open WSL terminal in VS Code:**
   - Press `Ctrl+Shift+P`
   - Type "Remote-WSL: New WSL Window"
   - Select your Ubuntu distribution

2. **Update system packages:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

3. **Install Python and pip:**
   ```bash
   sudo apt install python3 python3-pip python3-venv git -y
   ```

## Project Setup

### Step 1: Clone the Repository

```bash
# Navigate to your desired directory
cd ~
mkdir projects && cd projects

# Clone the repository
git clone https://github.com/your-repo/medical-faiss-rag.git
cd medical-faiss-rag
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv medical_rag_env

# Activate virtual environment
source medical_rag_env/bin/activate

# Verify activation (you should see (medical_rag_env) in prompt)
which python
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Note**: Installation may take 15-20 minutes as it downloads large models and libraries.

### Step 4: Download the Language Model

1. **Create account on Hugging Face** (if you don't have one):
   - Go to https://huggingface.co/
   - Create free account

2. **Download BioMistral-7B model:**
   ```bash
   # Option 1: Direct download using wget
   wget -O BioMistral-7B.Q4_K_M.gguf https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf
   
   # Option 2: Using huggingface_hub (install if needed)
   pip install huggingface_hub
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='MaziyarPanahi/BioMistral-7B-GGUF', filename='BioMistral-7B.Q4_K_M.gguf', local_dir='.')"
   ```

### Step 5: Prepare Medical Documents

1. **Create data directory:**
   ```bash
   mkdir data
   ```

2. **Add your PDF documents:**
   - Place medical PDF documents in the `data/` folder
   - Supported format: PDF files only
   - Example documents you can use:
     - Medical textbooks
     - Research papers
     - Clinical guidelines
     - Patient information leaflets

**Note**: For testing, you can download sample medical PDFs from open sources like PubMed Central or medical organization websites.

### Step 6: Verify File Structure

Your project should look like this:
```
medical-faiss-rag/
├── data/                          # Your PDF documents go here
│   ├── medical_doc1.pdf
│   └── medical_doc2.pdf
├── templates/
│   └── index.html
├── medical_rag_env/               # Virtual environment
├── BioMistral-7B.Q4_K_M.gguf     # Language model file
├── ingest.py                      # Document processing script
├── rag.py                         # Main application
├── retriever.py                   # Testing script
├── requirements.txt               # Dependencies
└── README.md
```

## Running the Application

### Step 1: Process Documents (First Time Only)

```bash
# Activate virtual environment (if not already active)
source medical_rag_env/bin/activate

# Process documents and create FAISS database
python ingest.py
```

**Expected output:**
```
**************************
Embeddings initialized: ...
**************************
Loading PDF documents...
100%|████████████████████| 5/5 [00:02<00:00,  2.15it/s]
Loaded 25 documents
**************************
Splitting documents into chunks...
Created 150 text chunks
**************************
Creating FAISS vector store...
FAISS Vector DB Successfully Created and Saved!
Vector store saved to 'faiss_index' directory
**************************
Testing vector store with a sample query...
Found 2 similar documents for test query: 'What is cancer?'
Result 1: Cancer is a group of diseases involving abnormal cell growth...
Result 2: Malignant tumors are characterized by uncontrolled proliferation...
**************************
```

**Time**: This process typically takes 5-15 minutes depending on:
- Number and size of PDF documents
- Your system's CPU performance
- Internet speed (for downloading embeddings model first time)

### Step 2: Test the Retrieval System (Optional)

```bash
# Test if the vector database works correctly
python retriever.py
```

**Expected output:**
```
Initializing embeddings...
Embeddings initialized successfully
##############
Loading FAISS vector store...
FAISS vector store loaded successfully!
<langchain.vectorstores.faiss.FAISS object at 0x...>
######
Testing query: 'What is Metastatic disease?'

Top 2 retrieved chunks with metadata based on question: What is Metastatic disease?
================================================================================

--- Result 1 ---
Similarity Score: 0.8234
Content Preview: Metastatic disease refers to cancer that has spread from its original (primary) site to other parts of the body. When cancer cells break away from the primary tumor...
Metadata: {'source': 'data/oncology_textbook.pdf', 'page': 42}
--------------------------------------------------

--- Result 2 ---
Similarity Score: 0.7891
Content Preview: The process of metastasis involves several steps including local invasion, intravasation into blood or lymphatic vessels, circulation through the bloodstream...
Metadata: {'source': 'data/cancer_research.pdf', 'page': 15}
--------------------------------------------------
```

### Step 3: Start the Web Application

```bash
# Start the FastAPI server
uvicorn rag:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
Initializing LLM...
LLM Initialized....
Initializing embeddings...
FAISS vector store loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 4: Access the Web Interface

1. **Open your web browser** (Chrome, Firefox, Edge, etc.)
2. **Navigate to:** `http://localhost:8000`
3. **You should see:** The Medical FAISS RAG QA App interface

## Usage Guide

### Web Interface Features

#### Main Interface Elements:
- **Header**: "Medical FAISS RAG QA App" with gradient styling
- **About Section**: Expandable accordion with system information
- **Status Indicator**: Shows current system status (Ready/Processing/Error)
- **Text Input**: Large textarea for entering medical questions
- **Submit Button**: "🔍 Ask Question" button with loading animation
- **Response Area**: Displays answers with source information

#### How to Use:

1. **Enter Your Question:**
   - Type your medical question in the text area
   - Examples of good questions:
     - "What is diabetes mellitus?"
     - "Explain the symptoms of hypertension"
     - "What are the side effects of metformin?"
     - "How is pneumonia diagnosed?"

2. **Submit Your Query:**
   - Click the "🔍 Ask Question" button
   - OR press `Enter` (without Shift)
   - The button will show a loading spinner

3. **Wait for Response:**
   - Processing typically takes 25-45 seconds
   - Status indicator will show "⏳ Processing your query..."
   - The system runs entirely on CPU, so patience is required

4. **Review the Answer:**
   - **Answer Section**: AI-generated response based on your documents
   - **Source Context**: The exact text from documents used to generate the answer
   - **Document Reference**: Shows which PDF and page number was referenced

### Example Interaction:

**Question:** "What is metastatic disease?"

**Expected Response:**
```
📋 Answer:
Metastatic disease refers to cancer that has spread from its original primary site to distant organs or tissues in the body. This occurs when cancer cells break away from the primary tumor, travel through the bloodstream or lymphatic system, and establish secondary tumors in other locations. Metastasis is a hallmark of advanced cancer and significantly impacts prognosis and treatment options.

📖 Source Context:
Metastatic disease represents one of the most challenging aspects of cancer care. The process involves multiple steps including local invasion, intravasation into blood vessels, survival in circulation, extravasation at distant sites, and colonization of new tissues. Common sites of metastasis include the liver, lungs, bone, and brain, depending on the primary tumor type.

📄 Document Reference:
Source: data/oncology_textbook.pdf, Page: 42 (FAISS Vector Store)
```

### Command Line Usage

#### Processing New Documents:
```bash
# Add new PDFs to data/ folder, then:
python ingest.py
```

#### Testing Retrieval:
```bash
# Test specific queries
python retriever.py
```

#### Starting the Server:
```bash
# Development mode (auto-reload)
uvicorn rag:app --reload

# Production mode
uvicorn rag:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Installation Issues

**Problem**: `pip install` fails with dependency conflicts
**Solution**:
```bash
# Create fresh virtual environment
deactivate
rm -rf medical_rag_env
python3 -m venv medical_rag_env
source medical_rag_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Problem**: FAISS installation fails
**Solution**:
```bash
# Install FAISS specifically
pip install faiss-cpu --no-cache-dir
# Or for systems with specific requirements:
conda install -c conda-forge faiss-cpu
```

#### 2. Model Download Issues

**Problem**: BioMistral model download fails
**Solution**:
```bash
# Manual download using curl
curl -L -o BioMistral-7B.Q4_K_M.gguf "https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf"

# Verify file size (should be ~4-5 GB)
ls -lh BioMistral-7B.Q4_K_M.gguf
```

#### 3. Runtime Issues

**Problem**: "Error loading FAISS vector store"
**Solution**:
```bash
# Recreate the FAISS database
rm -rf faiss_index/
python ingest.py
```

**Problem**: "ModuleNotFoundError" when running scripts
**Solution**:
```bash
# Ensure virtual environment is activated
source medical_rag_env/bin/activate
# Verify Python path
which python
pip list | grep langchain
```

**Problem**: Server starts but web page doesn't load
**Solution**:
```bash
# Check if port 8000 is in use
netstat -tulpn | grep :8000
# Use different port
uvicorn rag:app --port 8080
```

**Problem**: Slow response times (>2 minutes)
**Solution**:
```bash
# Check system resources
htop
# Reduce chunk size in ingest.py
# Modify: chunk_size=500, chunk_overlap=50
```

#### 4. Document Processing Issues

**Problem**: "No documents found" error
**Solution**:
```bash
# Verify PDF files in data directory
ls -la data/
# Check PDF file permissions
chmod 644 data/*.pdf
```

**Problem**: PDF parsing errors
**Solution**:
```bash
# Install additional PDF processing libraries
pip install pymupdf
# Try with different PDF files
```

#### 5. Memory Issues

**Problem**: "Out of memory" during processing
**Solution**:
```bash
# Reduce batch size in ingest.py
# Process fewer documents at once
# Add swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Optimization

#### For Better Speed:
1. **Use SSD storage** for faster file I/O
2. **Increase RAM** to at least 16GB
3. **Process smaller documents** initially for testing
4. **Reduce context window** in model config:
   ```python
   config = {
       'max_new_tokens': 1024,  # Reduced from 2048
       'context_length': 1024,  # Reduced from 2048
       # ... other settings
   }
   ```

#### For Better Accuracy:
1. **Use more relevant medical documents**
2. **Increase chunk overlap**:
   ```python
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000, 
       chunk_overlap=200,  # Increased from 100
   )
   ```
3. **Increase retrieval results**:
   ```python
   retriever = db.as_retriever(search_kwargs={"k": 3})  # Instead of k=1
   ```

### Logs and Debugging

#### Enable Verbose Logging:
```bash
# Set environment variables
export LANGCHAIN_VERBOSE=true
export LANGCHAIN_DEBUG=true

# Run with debug output
python rag.py
```

#### Check Application Logs:
```bash
# Monitor real-time logs
tail -f ~/.uvicorn/logs/access.log

# Check error logs in terminal where uvicorn is running
```

## File Structure

### Complete Project Structure:
```
medical-faiss-rag/
├── data/                              # Medical PDF documents
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── templates/                         # Web interface templates
│   └── index.html                     # Main web page
├── faiss_index/                       # FAISS vector database (auto-created)
│   ├── index.faiss
│   └── index.pkl
├── medical_rag_env/                   # Virtual environment
│   ├── bin/
│   ├── lib/
│   └── ...
├── BioMistral-7B.Q4_K_M.gguf         # Language model file (4-5 GB)
├── ingest.py                          # Document processing script
├── rag.py                             # Main FastAPI application
├── retriever.py                       # Testing and debugging script
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
└── README.md                          # Project documentation
```

### Key Files Explained:

- **`ingest.py`**: Processes PDF documents, creates embeddings, and builds FAISS vector database
- **`rag.py`**: Main web application with FastAPI backend and LLM integration
- **`retriever.py`**: Testing script to verify vector database functionality
- **`requirements.txt`**: Python package dependencies
- **`templates/index.html`**: Web interface with modern styling and interactivity
- **`faiss_index/`**: Directory containing FAISS vector database files (created automatically)

## Technical Details

### Architecture Overview:

```
User Query → Web Interface → FastAPI Backend → RAG Pipeline
                                               ↓
PDF Documents → Chunking → Embeddings → FAISS Index
                                               ↓
Query Embedding → Similarity Search → Retrieved Context → LLM → Response
```

### Key Components:

#### 1. **Document Processing Pipeline**
- **PDF Loading**: PyPDFLoader extracts text from PDF documents
- **Text Chunking**: RecursiveCharacterTextSplitter creates overlapping chunks
- **Embeddings**: PubMed-BERT generates medical domain embeddings
- **Vector Storage**: FAISS stores embeddings for fast similarity search

#### 2. **Retrieval System**
- **Query Processing**: User questions are embedded using the same model
- **Similarity Search**: FAISS finds most relevant document chunks
- **Context Assembly**: Retrieved chunks are formatted for the language model

#### 3. **Language Model**
- **Model**: BioMistral-7B quantized to 4-bit for CPU inference
- **Configuration**: Optimized for medical domain responses
- **Generation**: Produces answers based on retrieved context

#### 4. **Web Interface**
- **Frontend**: HTML/CSS/JavaScript with Bootstrap styling
- **Backend**: FastAPI with Jinja2 templating
- **Communication**: JSON API for query/response exchange

### Model Specifications:

#### BioMistral-7B:
- **Parameters**: 7 billion
- **Quantization**: Q4_K_M (4-bit)
- **File Size**: ~4 GB
- **Domain**: Medical/Healthcare fine-tuned
- **Inference**: CPU-optimized

#### PubMed-BERT Embeddings:
- **Model**: NeuML/pubmedbert-base-embeddings
- **Dimension**: 768
- **Domain**: Medical literature trained
- **Purpose**: Dense vector representations

#### FAISS Configuration:
- **Index Type**: Flat (exact search)
- **Metric**: L2 distance
- **Storage**: Local filesystem
- **Features**: Batch processing, persistence

### Performance Characteristics:

#### Processing Times (Approximate):
- **Document Ingestion**: 2-5 minutes per 100 pages
- **Query Processing**: 25-45 seconds per question
- **Vector Search**: <1 second
- **LLM Generation**: 20-40 seconds (main bottleneck)

#### Memory Usage:
- **Base System**: ~2 GB
- **Model Loading**: ~4-6 GB
- **Document Processing**: ~1-2 GB
- **Total Recommended**: 8-16 GB RAM

#### Storage Requirements:
- **Model File**: ~4 GB
- **Dependencies**: ~2-3 GB
- **FAISS Index**: ~100-500 MB (depends on document count)
- **Documents**: Variable based on PDF collection

### Security Considerations:

#### Data Privacy:
- ✅ **Local Processing**: No data sent to external servers
- ✅ **Offline Capability**: Works without internet after setup
- ✅ **Private Documents**: Your medical documents stay on your system

#### File Safety:
- ⚠️ **PDF Validation**: Ensure PDF files are from trusted sources
- ⚠️ **Input Sanitization**: Be cautious with file uploads in production
- ⚠️ **Access Control**: Consider adding authentication for sensitive deployments

---

## Final Assessment: Is FAISS a Good Choice?

### ✅ **Advantages of FAISS:**

1. **Performance**: Extremely fast similarity search, even with millions of vectors
2. **CPU Optimized**: Works well on CPU-only systems without GPU requirements
3. **Memory Efficient**: Optimized memory usage with various index types
4. **No Dependencies**: No external services or databases required
5. **Persistence**: Easy save/load functionality for vector indexes
6. **Scalability**: Can handle large document collections efficiently
7. **Production Ready**: Battle-tested by Facebook/Meta in production environments

### ⚠️ **Considerations:**

1. **No Built-in Metadata**: Requires separate storage for document metadata (handled by LangChain)
2. **Static Index**: Need to rebuild index when adding new documents
3. **No Real-time Updates**: Not ideal for frequently changing document collections
4. **Single Node**: Doesn't support distributed deployments out of the box

### 🎯 **For Your Use Case (Medical RAG), FAISS is EXCELLENT because:**

1. **Medical Documents Are Relatively Static**: Medical textbooks and research papers don't change frequently
2. **CPU Performance**: Perfect for local deployment without GPU requirements
3. **Privacy**: Complete local processing aligns with healthcare data requirements
4. **Speed**: Fast enough for interactive Q&A sessions
5. **Reliability**: No network dependencies or external service failures
6. **Cost**: Free and open-source with no ongoing costs

### 📊 **FAISS vs. Alternatives:**

| Feature | FAISS | Qdrant | Chroma | Pinecone |
|---------|-------|---------|---------|----------|
| Local Deployment | ✅ | ✅ | ✅ | ❌ |
| CPU Performance | ✅ | ⚠️ | ⚠️ | ✅ |
| No Network Required | ✅ | ❌ | ✅ | ❌ |
| Memory Efficiency | ✅ | ⚠️ | ⚠️ | ✅ |
| Setup Complexity | ✅ | ❌ | ✅ | ✅ |
| Healthcare Compliance | ✅ | ⚠️ | ✅ | ❌ |

### **Recommendation: ✅ FAISS is the RIGHT choice for this project**

The switch from Qdrant to FAISS eliminates Docker complexity, provides better CPU performance, ensures complete local operation, and maintains excellent search quality for your medical document RAG system. It's particularly well-suited for healthcare applications where data privacy and offline operation are priorities.