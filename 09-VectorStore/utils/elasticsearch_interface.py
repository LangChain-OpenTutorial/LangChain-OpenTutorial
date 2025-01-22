from typing import Optional, Dict, List, Tuple, Generator
from elasticsearch import Elasticsearch, helpers
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from langchain_elasticsearch import ElasticsearchStore


class ElasticsearchDB:
    def __init__(
        self, es_url: str = "http://localhost:9200", api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the ElasticsearchManager with a connection to the Elasticsearch instance.

        Parameters:
            es_url (str): URL of the Elasticsearch host.
            api_key (Optional[str]): API key for authentication (optional).
        """
        # Initialize the Elasticsearch client
        if api_key:
            self.es = Elasticsearch(
                es_url, api_key=api_key, timeout=120, retry_on_timeout=True
            )
        else:
            self.es = Elasticsearch(es_url, timeout=120, retry_on_timeout=True)

        # Test connection
        if self.es.ping():
            print("✅ Successfully connected to Elasticsearch!")
        else:
            raise ConnectionError("❌ Failed to connect to Elasticsearch.")

    def create_index(
        self,
        index_name: str,
        mapping: Optional[Dict] = None,
        settings: Optional[Dict] = None,
    ) -> str:
        """
        Create an Elasticsearch index with optional mapping and settings.

        Parameters:
            index_name (str): Name of the index to create.
            mapping (Optional[Dict]): Mapping definition for the index.
            settings (Optional[Dict]): Settings definition for the index.

        Returns:
            str: Success or warning message.
        """
        try:
            if not self.es.indices.exists(index=index_name):
                body = {}
                if mapping:
                    body["mappings"] = mapping
                if settings:
                    body["settings"] = settings
                self.es.indices.create(index=index_name, body=body)
                return f"✅ Index '{index_name}' created successfully."
            else:
                return f"⚠️ Index '{index_name}' already exists. Skipping creation."
        except Exception as e:
            return f"❌ Error creating index '{index_name}': {e}"

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
            return f"❌ Error deleting index '{index_name}': {e}"

    def get_document(self, index_name: str, document_id: str) -> Optional[Dict]:
        """
        Retrieve a single document by its ID.

        Parameters:
            index_name (str): The index to retrieve the document from.
            document_id (str): The ID of the document to retrieve.

        Returns:
            Optional[Dict]: The document's content if found, None otherwise.
        """
        try:
            response = self.es.get(index=index_name, id=document_id)
            return response["_source"]
        except Exception as e:
            print(f"❌ Error retrieving document: {e}")
            return None

    def search_documents(self, index_name: str, query: Dict) -> List[Dict]:
        """
        Search for documents based on a query.

        Parameters:
            index_name (str): The index to search.
            query (Dict): The query body for the search.

        Returns:
            List[Dict]: List of documents that match the query.
        """
        try:
            response = self.es.search(index=index_name, body={"query": query})
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []

    def upsert_document(
        self, index_name: str, document_id: str, document: Dict
    ) -> Dict:
        """
        Perform an upsert operation on a single document.

        Parameters:
            index_name (str): The index to perform the upsert on.
            document_id (str): The ID of the document.
            document (Dict): The document content to upsert.

        Returns:
            Dict: The response from Elasticsearch.
        """
        try:
            response = self.es.update(
                index=index_name,
                id=document_id,
                body={"doc": document, "doc_as_upsert": True},
            )
            return response
        except Exception as e:
            print(f"❌ Error upserting document: {e}")
            return {}

    def bulk_upsert(
        self, index_name: str, documents: List[Dict], timeout: Optional[str] = None
    ) -> None:
        """
        Perform a bulk upsert operation.

        Parameters:
            index (str): Default index name for the documents.
            documents (List[Dict]): List of documents for bulk upsert.
            timeout (Optional[str]): Timeout duration (e.g., '60s', '2m'). If None, the default timeout is used.
        """
        try:
            # Ensure each document includes an `_index` field
            for doc in documents:
                if "_index" not in doc:
                    doc["_index"] = index_name

            # Perform the bulk operation
            helpers.bulk(self.es, documents, timeout=timeout)
            print("✅ Bulk upsert completed successfully.")
        except Exception as e:
            print(f"❌ Error in bulk upsert: {e}")

    def parallel_bulk_upsert(
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

        # Ensure each document has an `_index` field
        for doc in documents:
            if "_index" not in doc:
                doc["_index"] = index_name

        batches = list(chunk_data(documents, batch_size))

        def bulk_upsert_batch(batch: List[Dict]):
            helpers.bulk(self.es, batch, timeout=timeout)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in batches:
                executor.submit(bulk_upsert_batch, batch)

    def delete_document(self, index_name: str, document_id: str) -> Dict:
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
            return response
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return {}

    def delete_by_query(self, index_name: str, query: Dict) -> Dict:
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

    def prepare_documents_with_ids(
        self, docs: List[str], embedded_documents: List[List[float]]
    ) -> Tuple[List[Dict], List[str]]:
        """
        Prepare a list of documents with unique IDs and their corresponding embeddings.

        Parameters:
            docs (List[str]): List of document texts.
            embedded_documents (List[List[float]]): List of embedding vectors corresponding to the documents.

        Returns:
            Tuple[List[Dict], List[str]]: A tuple containing:
                - List of document dictionaries with `doc_id`, `text`, and `vector`.
                - List of unique document IDs (`doc_ids`).
        """
        # Generate unique IDs for each document
        doc_ids = [str(uuid4()) for _ in range(len(docs))]

        # Prepare the document list with IDs, texts, and embeddings
        documents = [
            {"doc_id": doc_id, "text": doc, "vector": embedding}
            for doc, doc_id, embedding in zip(docs, doc_ids, embedded_documents)
        ]
        return documents, doc_ids

    def initialize_vector_store(
        self,
        index_name: str,
        embedding_model,
        es_url: str = "http://localhost:9200",
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the ElasticsearchStore for vector operations.

        Parameters:
            index_name (str): Elasticsearch index name.
            embedding_model: Object responsible for generating text embeddings.
            es_url (str): Elasticsearch host URL.
            api_key (Optional[str]): API key for authentication (optional).
        """
        try:
            self.vector_store = ElasticsearchStore(
                index_name=index_name,
                embedding=embedding_model,
                es_url=es_url,
                es_api_key=api_key,
            )
            print(f"✅ Vector store initialized for index '{index_name}'.")
        except Exception as e:
            raise RuntimeError(f"❌ Error initializing vector store: {e}")

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents similar to the query using vector similarity.

        Parameters:
            query (str): Query text to search for similar documents.
            top_k (int): Number of top similar results to retrieve.

        Returns:
            List[Dict]: A list of similar documents.
        """
        if not self.vector_store:
            raise ValueError(
                "❌ Vector store is not initialized. Please initialize it first."
            )
        try:
            results = self.vector_store.similarity_search(query=query, k=top_k)
            print(f"✅ Found {len(results)} similar documents.")
            return results
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []

    def similarity_search_with_score(
        self, query: str, keyword: str, top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Perform a hybrid search combining semantic search and keyword filtering with scores.

        Parameters:
            query (str): Query text for semantic similarity.
            keyword (str): Keyword to filter documents.
            top_k (int): Number of top results to retrieve.

        Returns:
            List[Tuple[Dict, float]]: A list of documents with their similarity scores.
        """
        if not self.vector_store:
            raise ValueError(
                "❌ Vector store is not initialized. Please initialize it first."
            )
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=[{"term": {"text": keyword}}],
            )
            print(f"✅ Hybrid search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"❌ Error in hybrid search with score: {e}")
            return []
