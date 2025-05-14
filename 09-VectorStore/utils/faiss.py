from utils.base import DocumentManager
from utils.base import Iterable, Any, Optional, List, Dict
from langchain_core.documents import Document
import faiss
import numpy as np
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Iterable


class FaissDocumentManager(DocumentManager):
    def __init__(
        self, dim: int = 768, embedding: Optional[Any] = None, **kwargs
    ) -> None:
        """
        Initialize FAISS vector database manager
        
        Args:
            dim: Dimension of embedding vectors
            embedding: Optional embedding function
            **kwargs: Additional arguments
        """
        super().__init__()
        self.dim = dim
        self.embedding = embedding
        
        base_index = faiss.IndexFlatL2(dim)  
        self.index = faiss.IndexIDMap(base_index)  
        
        self.document_store = {}
        self.next_id = 0  
    
    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Embed texts and add them to FAISS index
        
        Args:
            texts: Documents or texts
            metadatas: Metadata
            ids: Unique IDs, auto-generated if None
            **kwargs: Additional parameters
        """
        texts_list = list(texts)
        
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts_list))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts_list))]
        
        if self.embedding:
            embeddings = self.embedding.embed_documents(texts_list)
        else:
            embeddings = [np.random.rand(self.dim).astype('float32') for _ in texts_list]
        
        vectors = np.array(embeddings).astype('float32')
        
        int_ids = np.array([i + self.next_id for i in range(len(texts_list))], dtype=np.int64)
        
        self.index.add_with_ids(vectors, int_ids)
        
        for i, (text, metadata, user_id) in enumerate(zip(texts_list, metadatas, ids)):
            idx = self.next_id + i
            
            self.document_store[user_id] = {
                'index': idx,  
                'text': text,
                'metadata': metadata
            }
        
        self.next_id += len(texts_list)

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Process texts in parallel, embed them and add to FAISS index
        
        Args:
            texts: Documents or texts
            metadatas: Metadata
            ids: Unique IDs, auto-generated if None
            batch_size: Size of each batch
            workers: Number of workers
            **kwargs: Additional parameters
        """
        texts_list = list(texts)
        total = len(texts_list)
        
        if ids is None:
            ids = [str(uuid4()) for _ in range(total)]
            
        if metadatas is None:
            metadatas = [{} for _ in range(total)]
        
        batches = [
            (
                texts_list[i : i + batch_size],
                metadatas[i : i + batch_size] if metadatas else None,
                ids[i : i + batch_size] if ids else None,
            )
            for i in range(0, total, batch_size)
        ]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda batch: self.upsert(*batch, **kwargs), batches))

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Dict]:
        """
        Search for documents most similar to the query
        
        Args:
            query: Search query
            k: Number of results to return
            **kwargs: Filtering options
        
        Returns:
            List of similar documents
        """
        if not self.document_store:  
            return []
            
        if self.embedding:
            query_embedding = self.embedding.embed_documents([query])[0]
        else:
            query_embedding = np.random.rand(self.dim).astype('float32')
        
        query_vector = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        
        index_to_id = {}
        for user_id, doc_info in self.document_store.items():
            index_to_id[doc_info['index']] = user_id
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  
                continue
            
            if idx not in index_to_id:
                continue
                
            user_id = index_to_id[idx]
            doc_info = self.document_store[user_id]
            
            score = 1.0 / (1.0 + distance)
            score = round(score, 3)
            
            result = {
                'text': doc_info['text'],
                'metadata': {
                    'id': user_id,
                    **doc_info['metadata']
                },
                'score': score
            }
            results.append(result)
        
        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete documents from the index
        
        Args:
            ids: List of document IDs to delete
            filters: Conditions to filter documents for deletion
            **kwargs: Additional parameters
            
        Returns:
            Boolean indicating success
        """
        if filters and not ids:
            ids_to_delete = []
            for user_id, doc_info in self.document_store.items():
                match = True
                for key, value in filters.items():
                    if key not in doc_info['metadata'] or doc_info['metadata'][key] != value:
                        match = False
                        break
                
                if match:
                    ids_to_delete.append(user_id)
            
            if ids_to_delete:
                return self.delete(ids=ids_to_delete)
            return True
        
        if ids is None and filters is None:
            base_index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIDMap(base_index)
            self.document_store = {}
            self.next_id = 0
            return True
        
        if ids:
            ids_to_delete = [id for id in ids if id in self.document_store]
            
            if not ids_to_delete:
                return True  
            
            faiss_ids = [self.document_store[user_id]['index'] for user_id in ids_to_delete]
            
            try:
                self.index.remove_ids(np.array(faiss_ids, dtype=np.int64))
                
                for user_id in ids_to_delete:
                    del self.document_store[user_id]
                
                return True
            except Exception as e:
                print(f"FAISS deletion error: {e}")
                return False