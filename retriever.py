from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embeddings (same as used in ingest.py)
print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings",
    model_kwargs={'device': 'cpu'}
)

print("Embeddings initialized successfully")
print("##############")

# Load FAISS vector store
print("Loading FAISS vector store...")
try:
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded successfully!")
except Exception as e:
    print(f"Error loading FAISS vector store: {e}")
    print("Please ensure you have run 'python ingest.py' first to create the vector store.")
    exit(1)

print(f"Vector store type: {type(db)}")
print("######")

# Test query
query = "What is Metastatic disease?"
print(f"Testing query: '{query}'")

# Perform similarity search with scores
try:
    docs = db.similarity_search_with_score(query=query, k=2)
    print(f"\nTop {len(docs)} retrieved chunks with metadata based on question: {query}")
    print("=" * 80)
    
    for i, (doc, score) in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(f"Similarity Score: {score:.4f}")
        print(f"Content Preview: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
        
except Exception as e:
    print(f"Error during similarity search: {e}")

# Test regular similarity search (without scores)
print("\n" + "=" * 80)
print("Testing regular similarity search (without scores):")
try:
    docs_simple = db.similarity_search(query=query, k=2)
    print(f"Found {len(docs_simple)} documents")
    
    for i, doc in enumerate(docs_simple, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content: {doc.page_content[:300]}...")
        print(f"Metadata: {doc.metadata}")
        
except Exception as e:
    print(f"Error during simple similarity search: {e}")

print("\n" + "=" * 80)
print("Retriever test completed!")