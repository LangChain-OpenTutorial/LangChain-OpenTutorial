# Python Library
from typing import Optional, Dict, List, Tuple, Generator, Iterable, Any
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import logging

# Elasticsearch
from elasticsearch import Elasticsearch, helpers

# Langchain
from langchain_elasticsearch import ElasticsearchStore

# Interface
from utils.vectordbinterface import DocumentManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchConnectionManager:
    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        api_key: Optional[str] = None,
        embedding_model: Any = None,
        index_name: str = "langchain_tutorial_es",
    ) -> None:
        """
        Initialize the ElasticsearchConnectionManager with a connection to the Elasticsearch instance
        and initialize the ElasticsearchStore for vector operations.

        Parameters:
            es_url (str): URL of the Elasticsearch host.
            api_key (Optional[str]): API key for authentication (optional).
            embedding_model (Any): Object responsible for generating text embeddings.
            index_name (str): Elasticsearch index name.
        """
        self.es_url = es_url
        self.api_key = api_key
        self.embedding_model = embedding_model  # Store the embedding model
        self.es = Elasticsearch(
            es_url, api_key=api_key, timeout=120, retry_on_timeout=True
        )

        # Test connection
        if self.es.ping():
            logger.info("✅ Successfully connected to Elasticsearch!")
        else:
            raise ConnectionError("❌ Failed to connect to Elasticsearch.")

        # Initialize vector store
        try:
            self.vector_store = ElasticsearchStore(
                index_name=index_name,
                embedding=self.embedding_model,
                es_url=self.es_url,
                es_api_key=self.api_key,
            )
            logger.info(f"✅ Vector store initialized for index '{index_name}'.")
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            raise RuntimeError(f"Error initializing vector store: {e}")

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
    def __init__(self, connection_manager: ElasticsearchConnectionManager) -> None:
        """
        Initialize the ElasticsearchDocumentManager with a connection manager.

        Parameters:
            connection_manager (ElasticsearchConnectionManager): The connection manager for Elasticsearch.
        """
        self.connection_manager = connection_manager
        self.es = connection_manager.es
        self.embedding_model = (
            connection_manager.embedding_model
        )  # Access the embedding model

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

    def upsert(
        self,
        index_name: str,
        texts: Iterable[str],
        embedded_documents: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Upsert documents into Elasticsearch.

        Parameters:
            texts (Iterable[str]): List of text documents to upsert.
            embedded_documents (List[List[float]]): List of embedding vectors corresponding to the documents.
            metadatas (Optional[List[Dict]]): List of metadata dictionaries for each document.
            ids (Optional[List[str]]): List of document IDs.
            **kwargs (Any): Additional keyword arguments.
        """
        documents, doc_ids = self.prepare_documents_with_ids(texts, embedded_documents)
        self._bulk_upsert(index_name=index_name, documents=documents)
        self.doc_ids = doc_ids

    def upsert_parallel(
        self,
        index_name: str,
        texts: Iterable[str],
        embedded_documents: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Perform parallel upsert of documents into Elasticsearch.

        Parameters:
            texts (Iterable[str]): List of text documents to upsert.
            embedded_documents (List[List[float]]): List of embedding vectors corresponding to the documents.
            metadatas (Optional[List[Dict]]): List of metadata dictionaries for each document.
            ids (Optional[List[str]]): List of document IDs.
            **kwargs (Any): Additional keyword arguments.
        """
        documents, doc_ids = self.prepare_documents_with_ids(texts, embedded_documents)
        self._parallel_bulk_upsert(index_name=index_name, documents=documents)
        self.doc_ids = doc_ids

    def search(
        self,
        index_name: str = "langchain_tutorial_es",
        query: str = None,
        k: int = 10,
        use_similarity: bool = False,
        keyword: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Search for documents using different methods.

        Parameters:
            query (str): The search query.
            k (int): Number of top results to retrieve.
            use_similarity (bool): Whether to use similarity search.
            keyword (Optional[str]): Keyword for hybrid search.

        Returns:
            List[Dict]: A list of documents.
        """
        if not use_similarity:
            try:
                response = self.es.search(
                    index=index_name,
                    body={"query": {"match": {"text": query}}},
                )["hits"]["hits"][:k]
                documents = [hit["_source"]["text"] for hit in response]
                return documents
            except Exception as e:
                logger.error(f"❌ Error searching documents: {e}")
                return []
        else:
            if keyword:
                try:
                    results = self.connection_manager.vector_store.similarity_search_with_score(
                        query=query,
                        k=k,
                        filter=[{"term": {"text": keyword}}],
                    )
                    logger.info(
                        f"✅ Hybrid search completed. Found {len(results)} results."
                    )
                    return results
                except Exception as e:
                    logger.error(f"❌ Error in hybrid search with score: {e}")
                    return []
            else:
                try:
                    results = self.connection_manager.vector_store.similarity_search(
                        query=query, k=k
                    )
                    logger.info(f"✅ Found {len(results)} similar documents.")
                    documents = [result.page_content for result in results]
                    return documents
                except Exception as e:
                    logger.error(f"❌ Error in similarity search: {e}")
                    return []

    def delete(
        self,
        index_name: str,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete documents from Elasticsearch.

        Parameters:
            ids (Optional[List[str]]): List of document IDs to delete.
            filters (Optional[Dict]): Query to filter documents for deletion.
            **kwargs (Any): Additional keyword arguments.
        """
        if ids:
            for doc_id in ids:
                self._delete_document(
                    index_name=index_name,
                    document_id=doc_id,
                )
        elif filters:
            self._delete_by_query(index_name=index_name, query=filters)
        else:
            # Delete all documents
            self._delete_by_query(
                index_name=index_name,
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
            return response
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
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
