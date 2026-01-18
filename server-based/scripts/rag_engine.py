# rag_engine.py 
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class SmolFusionRAG:
    """
    Hybrid RAG Engine for SmolFusion Architecture.
    Combines Dense Retrieval (Semantic Search) and Sparse Retrieval (Keyword Match)
    to compensate for the limited knowledge base of lightweight LLMs (1.7B).
    """
    def __init__(self, documents: list):
        # Local document store for retrieval
        self.documents = documents
        
        # 1. Dense Retrieval Setup (FAISS)
        # Using 'paraphrase-albert-small-v2' for its tiny footprint on Edge devices (Jetson).
        self.embedder = SentenceTransformer('paraphrase-albert-small-v2') 
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        
        # L2 distance index for similarity search
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Pre-compute and index document embeddings
        embeddings = self.embedder.encode(self.documents, show_progress_bar=False)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # 2. Sparse Retrieval Setup (BM25)
        # Provides robust keyword matching which semantic search might miss.
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 2):
        """
        Retrieves relevant context by fusing results from Dense and Sparse indices.
        """
        # Step 1: Dense Semantic Search
        query_vec = self.embedder.encode([query]).astype('float32')
        _, faiss_indices = self.index.search(query_vec, top_k)
        
        # Step 2: Sparse Keyword Search
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Step 3: Rank Fusion (Union of indices)
        # Merges results from both methods to maximize retrieval recall.
        final_indices = list(set(faiss_indices[0]) | set(bm25_indices))
        
        # Return the retrieved document snippets
        return [self.documents[i] for i in final_indices]
