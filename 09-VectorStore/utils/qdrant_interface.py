from typing import Any, Dict, Iterable, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    PointIdsList,
    Filter,
    VectorParams,
    Distance,
)
from abc import ABC, abstractmethod
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed


class DocumentManagerInterface(ABC):
    """
    문서 insert/update (upsert, upsert_parallel)
    문서 search by query (search)
    문서 delete by id, delete by filter (delete)
    """

    @abstractmethod
    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """문서를 업서트합니다."""
        pass

    @abstractmethod
    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any
    ) -> None:
        """병렬로 문서를 업서트합니다."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """쿼리를 수행하고 관련 문서를 반환합니다.
        기본 기능: query (문자열) -> 비슷한 문서 k개 반환

        cosine_similarity 써치하는 것 의미 **문제될 경우 이슈제기

        -그외 기능 (추후 확장)
        metatdata search
        이미지 서치할 때 벡터 받는 것
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        """필터를 사용하여 문서를 삭제합니다.

        ids: List of ids to delete. If None, delete all. Default is None.
        filters: Dictionary of filters (querys) to apply. If None, no filters apply.

        """
        pass


class QdrantDocumentManager(DocumentManagerInterface):
    """Manages document operations with Qdrant, including upsert, search, and delete.

    This class interfaces with Qdrant to perform operations such as inserting,
    updating, searching, and deleting documents in a specified collection.
    """

    def __init__(self, collection_name: str, embedding, **kwargs: Any) -> None:
        """Initializes the QdrantDocumentManager with a collection name and embedding model.

        Args:
            collection_name (str): The name of the collection in Qdrant.
            embedding: The embedding model used to convert texts into vectors.
            **kwargs (Any): Additional keyword arguments for QdrantClient configuration.
        """
        self.client = QdrantClient(**kwargs)
        self.collection_name = collection_name
        self.embedding = embedding
        self.distance_metric = kwargs.get("distance_metric", Distance.COSINE)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensures that the specified collection exists in Qdrant.

        If the collection does not exist, it creates a new one with the specified
        vector size and distance metric.
        """
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            vector_size = len(self.embedding.embed_query("vetor size check"))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=self.distance_metric
                ),
            )

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
                    "content": texts[i],  # Store original text in 'content'
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
        **kwargs: Any
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
        **kwargs: Any
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
            **kwargs
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
        **kwargs: Any
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
