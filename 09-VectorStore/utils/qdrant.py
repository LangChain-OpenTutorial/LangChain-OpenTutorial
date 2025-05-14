from utils.base import DocumentManager
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    VectorStruct,
    UpdateResult,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    PointIdsList,
)
from typing import List, Dict, Any, Optional, Iterable, Callable
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document


class QdrantDocumentManager(DocumentManager):
    def __init__(self, client: QdrantClient, embedding: Callable, **kwargs) -> None:
        self.qdrant_client = client  # Qdrant Python SDK
        try:
            if "collection_name" not in kwargs:
                kwargs["collection_name"] = "qdrant_test"

            self.collection_name = kwargs["collection_name"]

            if "vectors_config" not in kwargs:
                # https://qdrant.tech/documentation/embeddings/openai/?utm_source=chatgpt.com
                kwargs["vectors_config"] = VectorParams(
                    size=1536,  # text-embedding-3-large support 1536
                    distance=Distance.COSINE,  # General Used Cosine Similarity
                )
            # Create Collection
            if not self.qdrant_client.collection_exists(
                "qdrant_test"
            ) and not self.qdrant_client.collection_exists(kwargs["collection_name"]):
                if self.qdrant_client.create_collection(**kwargs):
                    print(f"Success Create {kwargs.get('collection_name')} Collection")
                else:
                    raise Exception("Failed Create Collection")
        except Exception as e:
            print(e)
        # Embedding
        self.embedding = embedding

    def __ensure_indexed_fields(self, fields: List[str]):
        """
        Ensure that each specified payload field is indexed in the current Qdrant collection.

        This is required for enabling metadata filtering (e.g. via 'filters' argument in search).
        If an index already exists for a field, it will be skipped silently.

        Args:
            fields (List[str]): List of payload field names to index.

        Raises:
            Exception: If index creation fails for reasons other than 'already exists'.
        """

        for field in fields:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception as e:
                if "already exists" not in str(e):
                    raise

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Insert or update vectors in Qdrant using the provided embedding model.

        Each text is embedded into a vector and stored in the Qdrant collection
        along with its metadata (payload). If no ID is provided, UUIDs are generated.

        Args:
            texts (Iterable[str]): List or iterable of input texts to be embedded and stored.
            metadatas (Optional[List[Dict]]): Optional list of metadata dictionaries for each text.
            ids (Optional[List[str]]): Optional list of string IDs for the vectors.
            **kwargs: Optional arguments such as:
                - result_view (bool): If True, prints operation_id and status for each result.

        Returns:
            None
        """

        if ids is None:  # if the ids are None
            ids = [str(uuid4()) for _ in range(len(texts))]

        vectors: VectorStruct = self.embedding(texts)  # List[float] in VectorStruct

        metadatas = metadatas or [{}] * len(texts)

        payloads = [
            {"text": text} | metadata for text, metadata in zip(texts, metadatas)
        ]

        # Create Index
        index_fields = set()
        for payload in payloads:
            index_fields.update(k for k in payload.keys() if k != "text")

        self.__ensure_indexed_fields(list(index_fields) + ["text"])

        # make points
        points = [
            PointStruct(id=id, vector=vector, payload=payload)
            for id, vector, payload in zip(ids, vectors, payloads)
        ]

        results: UpdateResult = self.qdrant_client.upsert(
            collection_name=self.collection_name, points=points
        )

        if "result_view" in kwargs:
            print(f"Operation_id : {results.operation_id} | Status : {results.status}")

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
        Perform parallel upsert of vectors into Qdrant using ThreadPoolExecutor.

        This method embeds input texts into vectors and inserts them into the collection
        in parallel batches, which improves performance for large datasets.

        Args:
            texts (Iterable[str]): List or iterable of texts to be embedded and inserted.
            metadatas (Optional[List[Dict]]): Optional list of metadata dictionaries.
            ids (Optional[List[str]]): Optional list of string IDs. If None, UUIDs are auto-generated.
            batch_size (int): Number of points to upsert in each batch.
            workers (int): Number of threads to use for parallel execution.
            **kwargs: Optional arguments such as:
                - result_view (bool): If True, prints operation_id and status for each batch result.

        Returns:
            None
        """
        texts = list(texts)
        metadatas = metadatas or [{}] * len(texts)

        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts))]

        vectors: VectorStruct = self.embedding(texts)

        payloads = [
            {"text": text} | metadata for text, metadata in zip(texts, metadatas)
        ]

        # Create Index
        index_fields = set()
        for payload in payloads:
            index_fields.update(k for k in payload.keys() if k != "text")

        self.__ensure_indexed_fields(list(index_fields) + ["text"])

        # Prepare all points
        all_points = [
            PointStruct(id=id, vector=vector, payload=payload)
            for id, vector, payload in zip(ids, vectors, payloads)
        ]

        # Batching
        def batch_iterable(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]

        # upsert Batch
        def upsert_batch(batch: List[PointStruct]) -> UpdateResult:
            return self.qdrant_client.upsert(
                collection_name=self.collection_name, points=batch
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(upsert_batch, batch): batch
                for batch in batch_iterable(all_points, batch_size)
            }

            for future in as_completed(futures):
                result = future.result()
                if kwargs.get("result_view", False):
                    print(
                        f"Operation_id: {result.operation_id} | Status: {result.status}"
                    )

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """
        Perform a vector similarity search with optional metadata filtering.

        Args:
            query (str): The input query string to embed and search.
            k (int): The number of top results to return.
            **kwargs:
                filters (List[Dict[str, Any]]): Optional metadata filters.
                    Example: [{"category": "news"}, {"lang": "en"}]

        Returns:
            List[Document]: List of matching documents with metadata.
        """

        query_vector: VectorStruct = self.embedding(query)[0]

        # filtering
        query_filter = None
        if "filters" in kwargs:
            condition = []
            for f in kwargs["filters"]:
                for key, value in f.items():
                    condition.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            query_filter = Filter(must=condition)

        results = (
            self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                query_filter=query_filter,
            )
            .model_dump()
            .get("points", [])
        )
        return [
            Document(
                page_content=hit["payload"].get("text", ""),
                metadata={
                    **{k: v for k, v in hit["payload"].items() if k != "text"},
                    "score": hit["score"],
                    "id": hit["id"],
                },
            )
            for hit in results
        ]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete points from the Qdrant collection by ID, filter, or both.

        - If only `ids` are given: delete those points.
        - If only `filters` are given: delete all points matching the filter.
        - If both `ids` and `filters` are given: delete only points that match both conditions.
        - If neither is given: delete all points in the collection.

        Args:
            ids (Optional[List[str]]): List of point IDs to delete.
            filters (Optional[List[Dict[str, Any]]]): Metadata filter conditions.
            **kwargs: Reserved for future use.

        Returns:
            None
        """
        if ids and filters:
            # Delete by ids and filters
            conditions = []
            for f in filters:
                for key, value in f.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

            query_filter = Filter(must=conditions)

            # query points matching filter
            scroll_offset = None
            matched_ids = set()
            while True:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    with_payload=False,
                    limit=30,
                    offset=scroll_offset,
                )
                points_batch, scroll_offset = scroll_result

                if not points_batch:
                    print("Delete All Finished")
                    break

                ids_to_delete = [point.id for point in points_batch]
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=ids_to_delete),
                )

                print(f"{len(ids_to_delete)} data delete...")

        elif ids:
            # Delete by ids
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
            )
            print(f"{len(ids)} data delete...")

        elif filters:
            # Delete by filters
            conditions = []
            for f in filters:
                for key, value in f.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

            query_filter = Filter(must=conditions)

            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=query_filter),
            )
            print(f"Filters: {query_filter}")
            print("Delete All Finished")

        else:
            # all delete
            scroll_offset = None
            while True:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    with_payload=False,
                    limit=256,
                    offset=scroll_offset,
                )
                points_batch, scroll_offset = scroll_result

                if not points_batch:
                    print("Delete All Finished")
                    break

                ids_to_delete = [point.id for point in points_batch]

                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=ids_to_delete),
                )
                print(f"{len(ids_to_delete)} data delete...")
