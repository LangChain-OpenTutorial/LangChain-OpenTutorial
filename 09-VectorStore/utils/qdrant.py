from typing import Any, Dict, Iterable, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    PointIdsList,
    Filter,
    VectorParams,
    Distance,
)
from qdrant_client import models
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.vectordbinterface import DocumentManager
from qdrant_client.http.models import Distance


class QdrantDocumentManager(DocumentManager):
    """Manages document operations with Qdrant, including upsert, search, and delete.

    This class interfaces with Qdrant to perform operations such as inserting,
    updating, searching, and deleting documents in a specified collection.
    """

    def __init__(
        self,
        collection_name: str,
        embedding,
        metric: Distance = Distance.COSINE,
        force_recreate: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the QdrantDocumentManager with a collection name and embedding model.

        Args:
            collection_name (str): The name of the collection in Qdrant.
            embedding: The embedding model used to convert texts into vectors.
            metric (Distance): The distance metric for vector comparisons.
            force_recreate (bool): Whether to forcefully recreate the collection if it exists.
            **kwargs (Any): Additional keyword arguments for QdrantClient configuration.
        """
        self.client = QdrantClient(**kwargs)
        self.collection_name = collection_name
        self.embedding = embedding
        self.metric = metric
        self._ensure_collection_exists(force_recreate=force_recreate)

    def create_collection(
        self,
        dense_vectors_config: Optional[VectorParams] = None,
        sparse_vector_config: Optional[dict] = None,
        force_recreate: bool = False,
    ) -> None:
        if force_recreate:
            self._delete_collection()

        collection_config = self._build_collection_config(
            dense_vectors_config, sparse_vector_config
        )

        self.client.create_collection(
            collection_name=self.collection_name, **collection_config
        )
        print(
            f"Collection '{self.collection_name}' created successfully with configuration: {collection_config}"
        )

    def _delete_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted for recreation.")
        except Exception as delete_exception:
            print(
                f"Failed to delete existing collection '{self.collection_name}': {delete_exception}"
            )
            raise

    def _build_collection_config(
        self,
        dense_vectors_config: Optional[VectorParams],
        sparse_vector_config: Optional[dict],
    ) -> dict:
        collection_config = {}
        if dense_vectors_config:
            collection_config["vectors_config"] = dense_vectors_config
        if sparse_vector_config:
            collection_config["sparse_vectors_config"] = sparse_vector_config
        if not collection_config:
            raise ValueError(
                "At least one of dense_vectors_config or sparse_vector_config must be provided."
            )
        return collection_config

    def _ensure_collection_exists(
        self, force_recreate: bool = False, sparse_embedding=None
    ) -> None:
        vector_size = len(self.embedding.embed_query("vector size check"))
        dense_vectors_config = VectorParams(size=vector_size, distance=self.metric)

        sparse_vector_config = None
        if sparse_embedding:
            sparse_vector_config = {
                "sparse-vector": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            }

        if not self._collection_exists() or force_recreate:
            print(
                f"Collection '{self.collection_name}' does not exist or force recreate is enabled. Creating new collection..."
            )
            self.create_collection(
                dense_vectors_config=dense_vectors_config,
                sparse_vector_config=sparse_vector_config,
                force_recreate=force_recreate,
            )

    def _collection_exists(self) -> bool:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info is not None
        except Exception:
            return False

    def _create_points(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]],
        ids: Optional[List[str]],
    ) -> List[PointStruct]:
        """Converts strings into Qdrant's point structure.

        Args:
            texts (Iterable[str]): The texts to be converted into points.
            metadatas (Optional[List[dict]]): Optional metadata for each text.
            ids (Optional[List[str]]): Optional list of ids for each text.

        Returns:
            List[PointStruct]: A list of PointStruct objects ready for insertion into Qdrant.
        """
        return [
            PointStruct(
                id=ids[i] if ids else str(i),
                vector=self.embedding.embed_query(texts[i]),  # Convert text to vector
                payload={
                    "page_content": texts[i],  # Store original text in 'content'
                    "metadata": metadatas[i],
                },
            )
            for i in range(len(texts))
        ]

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upserts documents into the collection and returns the upserted ids.

        Args:
            texts (Iterable[str]): The texts to be upserted.
            metadatas (Optional[List[dict]]): Optional metadata for each text.
            ids (Optional[List[str]]): Optional list of ids for each text.
            **kwargs (Any): Additional keyword arguments for the upsert operation.

        Returns:
            List[str]: The list of successfully upserted ids.
        """
        points = self._create_points(texts, metadatas, ids)
        self.client.upsert(collection_name=self.collection_name, points=points)

        # Return the ids used for the upsert operation
        return ids if ids else [str(i) for i in range(len(texts))]

    def batch_upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]],
        ids: Optional[List[str]],
        start: int,
        end: int,
    ) -> List[str]:
        """Performs batch upsert and returns the upserted ids.

        Args:
            texts (Iterable[str]): The texts to be upserted.
            metadatas (Optional[List[dict]]): Optional metadata for each text.
            ids (Optional[List[str]]): Optional list of ids for each text.
            start (int): The starting index of the batch.
            end (int): The ending index of the batch.

        Returns:
            List[str]: The list of upserted ids.
        """
        batch_points = self._create_points(
            texts[start:end],
            metadatas[start:end] if metadatas else None,
            ids[start:end] if ids else None,
        )
        self.client.upsert(collection_name=self.collection_name, points=batch_points)
        return ids[start:end] if ids else [str(i) for i in range(start, end)]

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> List[str]:
        """Performs parallel upsert of documents and returns the upserted ids.

        Args:
            texts (Iterable[str]): The texts to be upserted.
            metadatas (Optional[List[dict]]): Optional metadata for each text.
            ids (Optional[List[str]]): Optional list of ids for each text.
            batch_size (int): The size of each batch for upsert. Default is 32.
            workers (int): The number of worker threads to use. Default is 10.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[str]: The list of upserted ids.
        """
        all_ids = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.batch_upsert,
                    texts,
                    metadatas,
                    ids,
                    i,
                    min(i + batch_size, len(texts)),
                )
                for i in range(0, len(texts), batch_size)
            ]
            for future in as_completed(futures):
                all_ids.extend(future.result())

        return all_ids

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Dict[str, Any]]:
        """Performs a search query and returns a list of relevant documents.

        Args:
            query (str): The search query string to find similar documents.
            k (int): The number of top documents to return. Default is 10.
            **kwargs (Any): Additional keyword arguments for the search operation.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the payload, id, and score of each result.
        """
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.embedding.embed_query(query),
            limit=k,
            **kwargs,
        )
        return [
            {
                "payload": result.payload,
                "id": result.id,
                "score": result.score,
            }
            for result in search_results
        ]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Filter] = None,
        **kwargs: Any,
    ) -> None:
        """Deletes documents from the collection based on ids or filters.

        Args:
            ids (Optional[List[str]]): A list of document ids to delete. If None, no id-based deletion is performed.
            filters (Optional[Filter]): A Filter object to apply for deletion. If None, no filter-based deletion is performed.
            **kwargs (Any): Additional keyword arguments for the delete operation.

        Returns:
            None
        """
        if ids:
            points_selector = PointIdsList(points=ids)
            self.client.delete(
                collection_name=self.collection_name, points_selector=points_selector
            )
        elif filters:
            self.client.delete(collection_name=self.collection_name, filter=filters)

    def scroll(self, scroll_filter, with_vectors=False, k=None) -> List[Dict[str, Any]]:
        """
        Retrieve records from a Qdrant collection using the scroll method.

        Args:
            scroll_filter: The filter condition to apply for retrieving records.
            k (int, optional): The number of top records to return. If None, retrieve all records.

        Returns:
            List[Dict[str, Any]]: A list of records in the collection.
        """
        all_records = []
        next_page_offset = None
        total_retrieved = 0

        try:
            while True:
                limit = 100 if k is None else min(100, k - total_retrieved)
                response, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    scroll_filter=scroll_filter,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                all_records.extend(response)
                total_retrieved += len(response)

                if next_page_offset is None or (k is not None and total_retrieved >= k):
                    break

        except Exception as e:
            print(f"Error retrieving records: {e}")

        return all_records
