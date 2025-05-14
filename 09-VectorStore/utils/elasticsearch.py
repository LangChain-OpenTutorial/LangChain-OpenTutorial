# Python Library
from typing import Optional, Dict, List, Tuple, Generator, Iterable, Any
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import logging

# Elasticsearch
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError

# Langchain
from langchain_openai import OpenAIEmbeddings

# Interface
from utils.base import DocumentManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchIndexManager:
    def __init__(self, client):
        self.es = client

    def get_embedding_dims(self, embedding_model) -> int:
        """
        Get embedding dimension size from the embedding model.

        Parameters:
            embedding_model: An embedding model instance.

        Returns:
            int: Dimension of the embedding.
        """
        test_vector = embedding_model.embed_query("test")
        return len(test_vector)
    
    def create_index(
        self,
        embedding,
        index_name: str,
        metric: str = "cosine",
        settings: Optional[Dict] = None,
    ) -> str:
        """
        Create an Elasticsearch index using embedding dimension from the embedding model.

        Parameters:
            index_name (str): Name of the index to create.
            embedding_model: An embedding model instance to determine dimension.
            settings (Optional[Dict]): Additional settings definition for the index.

        Returns:
            str: Success or warning message.
        """
        dims = self.get_embedding_dims(embedding)
        
        mapping = {
            "properties": {
                "metadata": {"properties": {"doc_id": {"type": "keyword"}}},
                "text": {"type": "text"},  # Field for storing textual content
                "vector": {  # Field for storing vector embeddings
                    "type": "dense_vector",  # Specifies dense vector type
                    "dims": dims,  # Number of dimensions in the vector
                    "index": True,  # Enable indexing for vector search
                    "similarity": metric,  # Use cosine similarity for vector comparisons
                },
            }
        }
        try:
            if not self.es.indices.exists(index=index_name):
                body = {}
                if mapping:
                    body["mappings"] = mapping
                if settings:
                    body["settings"] = settings
                self.es.indices.create(index=index_name, body=body)
                print(f"✅ Index '{index_name}' created successfully.")
                return {
                    "status": "created",
                    "index_name": index_name,
                    "embedding_dims": dims,
                    "metric": metric,
                }
            else:
                print(f"⚠️ Index '{index_name}' already exists. Skipping creation.")
                return {
                    "status": "exists",
                    "index_name": index_name,
                    "embedding_dims": dims,
                    "metric": metric,
                }
        except Exception as e:
            logger.error(f"❌ Error creating index '{index_name}': {e}")
            raise

    def delete_index(self, index_name: str) -> str:
        """
        Delete an Elasticsearch index if it exists.

        Parameters:
            index_name (str): Name of the index to delete.

        Returns:
            str: Success or warning message.
        """
        try:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                return f"✅ Index '{index_name}' deleted successfully."
            else:
                return f"⚠️ Index '{index_name}' does not exist."
        except Exception as e:
            logger.error(f"❌ Error deleting index '{index_name}': {e}")
            raise


class ElasticsearchDocumentManager(DocumentManager):
    def __init__(self, client, index_name, embedding):
        self.es = client
        self.index_name = index_name
        self.embedding = embedding
    
    def _embed_doc(self, texts) -> List[float]:
        """
        Embed texts
        
        Args:
        - texts: List of text
        
        Return:
        List of floats.
        """
        embedded_documents = self.embedding.embed_documents(texts)
        return embedded_documents

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Upsert documents into Elasticsearch.

        Parameters:
            texts (Iterable[str]): List of text documents to upsert.
            metadatas (Optional[List[Dict]]): List of metadata dictionaries for each document.
            ids (Optional[List[str]]): List of document IDs.
            **kwargs (Any): Additional keyword arguments.
        """
        # 1. Generate embeddings
        embedded_documents = self._embed_doc(texts)
        # 2. Structure documents accordingly
        documents = []
        for i, (text, vector) in enumerate(zip(texts, embedded_documents)):
            doc_id = ids[i] if ids else str(uuid4())
            metadata = metadatas[i] if metadatas else {}
            doc = {
                "text": text,
                "vector": vector,
                "metadata": {
                    "doc_id": doc_id,
                    **metadata
                }
            }
            documents.append(doc)
        # 3. bulk upsert
        self._bulk_upsert(index_name=self.index_name, documents=documents)

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Perform parallel upsert of documents into Elasticsearch.

        Parameters:
            texts (Iterable[str]): List of text documents to upsert.
            metadatas (Optional[List[Dict]]): List of metadata dictionaries for each document.
            ids (Optional[List[str]]): List of document IDs.
            batch_size (int): Number of documents per batch.
            max_workers (int): Number of parallel threads.
            **kwargs (Any): Additional keyword arguments.
        """
        # 1. Generate embeddings
        embedded_documents = self._embed_doc(texts)
        # 2. Structure documents accordingly
        documents = []
        for i, (text, vector) in enumerate(zip(texts, embedded_documents)):
            doc_id = ids[i] if ids else str(uuid4())
            metadata = metadatas[i] if metadatas else {}
            doc = {
                "text": text,
                "vector": vector,
                "metadata": {
                    "doc_id": doc_id,
                    **metadata
                }
            }
            documents.append(doc)
        # 3. parallel bulk upsert
        self._parallel_bulk_upsert(
            index_name=self.index_name,
            documents=documents,
            batch_size=batch_size,
            max_workers=max_workers
        )

    def search_by_embedding(
        self,
        query: str,
        k: int = 10,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> list:
        """
        Search for documents using cosine similarity on dense vector embeddings, with optional metadata filters.

        Parameters:
            query (str): The search query.
            k (int): Number of top results to retrieve.
            filters (Optional[dict]): Metadata filters, e.g. {"title": "Chapter 4"}

        Returns:
            List[Document]: A list of LangChain Document objects sorted by similarity.
        """
        from langchain.schema import Document
        try:
            query_vector = self.embedding.embed_query(query)
            # Create filter query
            filter_clauses = []
            if filters:
                for key, value in filters.items():
                    filter_clauses.append({"match": {f"metadata.{key}": value}})
            base_query = {"bool": {"filter": filter_clauses}} if filter_clauses else {"match_all": {}}
            response = self.es.search(
                index=self.index_name,
                body={
                    "size": k,
                    "query": {
                        "script_score": {
                            "query": base_query,
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                }
            )
            hits = response["hits"]["hits"]
            documents = [
                Document(
                    page_content=hit["_source"]["text"],
                    metadata={**hit["_source"].get("metadata", {}), "score": hit["_score"] / 2}
                )
                for hit in hits
            ]
            return documents
        except Exception as e:
            logger.error(f"❌ Error in embedding similarity search: {e}")
            return []

    def search(
        self,
        query: str = None,
        k: int = 10,
        use_similarity: bool = True,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> list:
        """
        Search for documents using either BM25 match or cosine similarity on dense vectors, with optional metadata filters.

        Parameters:
            query (str): The search query.
            k (int): Number of top results to retrieve.
            use_similarity (bool): Whether to use cosine similarity search.
            filters (Optional[dict]): Metadata filters, e.g. {"title": "Chapter 4"}

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        if use_similarity:
            return self.search_by_embedding(query=query, k=k, filters=filters)
        else:
            from langchain.schema import Document
            try:
                must_clauses = [{"match": {"text": query}}] if query else []
                if filters:
                    for key, value in filters.items():
                        must_clauses.append({"match": {f"metadata.{key}": value}})
                es_query = {"query": {"bool": {"must": must_clauses}}}
                response = self.es.search(
                    index=self.index_name,
                    body=es_query,
                )["hits"]["hits"][:k]
                documents = [
                    Document(
                        page_content=hit["_source"]["text"],
                        metadata=hit["_source"].get("metadata", {})
                    )
                    for hit in response
                ]
                return documents
            except Exception as e:
                logger.error(f"❌ Error searching documents: {e}")
                return []

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete documents from Elasticsearch.

        Parameters:
            ids (Optional[List[str]]): List of metadata.doc_id values to delete.
            filters (Optional[Dict]): Metadata filters for deletion, e.g. {"title": "chapter 6"}
            **kwargs (Any): Additional keyword arguments.
        """
        if ids:
            # Treat ids as metadata.doc_id, search for ES _id and delete
            for doc_id in ids:
                # Search for documents with metadata.doc_id == doc_id
                query = {"query": {"term": {"metadata.doc_id": doc_id}}}
                response = self.es.search(index=self.index_name, body=query)
                hits = response.get("hits", {}).get("hits", [])
                for hit in hits:
                    es_id = hit["_id"]
                    self._delete_document(
                        index_name=self.index_name,
                        document_id=es_id,
                    )
        elif filters:
            # Convert filters dict to bool/must term queries on metadata fields
            must_clauses = []
            for key, value in filters.items():
                must_clauses.append({"term": {f"metadata.{key}": value}})
            es_query = {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}
            self._delete_by_query(index_name=self.index_name, query=es_query)
        else:
            # No ids or filters: delete all documents in the index
            self._delete_by_query(
                index_name=self.index_name,
                query={"match_all": {}},
            )

    def _delete_document(self, index_name: str, document_id: str) -> Dict:
        """
        Delete a single document by its ID.

        Parameters:
            index_name (str): The index to delete the document from.
            document_id (str): The ID of the document to delete.

        Returns:
            Dict: The response from Elasticsearch.
        """
        try:
            response = self.es.delete(index=index_name, id=document_id)
            if response.get("result") == "not_found":
                logger.info(f"Document {document_id} not found in index {index_name}.")
            return response
        except NotFoundError:
            logger.info(f"Document {document_id} not found in index {index_name}.")
            return {}
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
            return {}

    def _delete_by_query(self, index_name: str, query: Dict) -> Dict:
        """
        Delete documents based on a query.

        Parameters:
            index_name (str): The index to delete documents from.
            query (Dict): The query body for the delete operation.

        Returns:
            Dict: The response from Elasticsearch.
        """
        try:
            response = self.es.delete_by_query(
                index=index_name, body={"query": query}, conflicts="proceed"
            )
            return response
        except Exception as e:
            print(f"❌ Error deleting documents by query: {e}")
            return {}

    def _add_index_to_documents(self, documents: List[Dict], index_name: str) -> None:
        """
        Ensure each document includes an `_index` field.

        Parameters:
            documents (List[Dict]): List of documents to modify.
            index_name (str): The index name to add to each document.
        """
        for doc in documents:
            if "_index" not in doc:
                doc["_index"] = index_name

    def _bulk_upsert(
        self, index_name: str, documents: List[Dict], timeout: Optional[str] = None
    ) -> None:
        """
        Perform a bulk upsert operation.

        Parameters:
            index_name (str): Default index name for the documents.
            documents (List[Dict]): List of documents for bulk upsert.
            timeout (Optional[str]): Timeout duration (e.g., '60s', '2m'). If None, the default timeout is used.
        """
        try:
            self._add_index_to_documents(documents, index_name)
            helpers.bulk(self.es, documents, timeout=timeout)
            logger.info("✅ Bulk upsert completed successfully.")
        except Exception as e:
            logger.error(f"❌ Error in bulk upsert: {e}")

    def _parallel_bulk_upsert(
        self,
        index_name: str,
        documents: List[Dict],
        batch_size: int = 100,
        max_workers: int = 4,
        timeout: Optional[str] = None,
    ) -> None:
        """
        Perform a parallel bulk upsert operation.

        Parameters:
            index_name (str): Default index name for documents.
            documents (List[Dict]): List of documents for bulk upsert.
            batch_size (int): Number of documents per batch.
            max_workers (int): Number of parallel threads.
            timeout (Optional[str]): Timeout duration (e.g., '60s', '2m'). If None, the default timeout is used.
        """

        def chunk_data(
            data: List[Dict], chunk_size: int
        ) -> Generator[List[Dict], None, None]:
            """Split data into chunks."""
            for i in range(0, len(data), chunk_size):
                yield data[i : i + chunk_size]

        self._add_index_to_documents(documents, index_name)

        batches = list(chunk_data(documents, batch_size))

        def bulk_upsert_batch(batch: List[Dict]):
            helpers.bulk(self.es, batch, timeout=timeout)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in batches:
                executor.submit(bulk_upsert_batch, batch)

    def get_documents_ids(self, index_name: str, size: int = 1000) -> List[Dict]:
        """
        Retrieve all document IDs from a specified index.

        Parameters:
            index_name (str): The index from which to retrieve document IDs.
            size (int, optional): Maximum number of documents to retrieve. Defaults to 1000.

        Returns:
            List[Dict]: A list of document IDs.
        """
        response = self.es.search(
            index=index_name,
            body={"_source": False, "query": {"match_all": {}}},
            size=size,
        )
        return [doc["_id"] for doc in response["hits"]["hits"]]

    def get_documents_by_ids(self, index_name: str, ids: List[str]) -> List[Dict]:
        """
        Retrieve documents by their IDs from a specified index.

        Parameters:
            index_name (str): The index from which to retrieve documents.
            ids (List[str]): List of document IDs to retrieve.

        Returns:
            List[Dict]: A list of documents.
        """
        response = self.es.search(
            index=index_name, body={"query": {"ids": {"values": ids}}}
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
