from utils.vectordbinterface import DocumentManager
from utils.vectordbinterface import Iterable, Any, Optional, List, Dict
from chromadb.api import ClientAPI  # Client Type
from chromadb.utils import embedding_functions
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding: Any):
        self.embedding = embedding

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding.embed_documents(input)


class ChromaDocumentMangager(DocumentManager):
    def __init__(
        self, client: ClientAPI, embedding: Optional[Any] = None, **kwargs
    ) -> None:
        """
        ### kwargs
            - name[str] : collection name(unique)
            - configuration[CollectionConfiguration or None] : Configuration settings per collection
            - data_loader[DataLoadable or None] : Select data loader
            - embedding_function[Callable or None] : Embedding function, default -> `all-MiniLM-L6-v2`
            - metadata[Dict or None] :
                - hnsw:space[str or None] : l2(default,squared L2 norm),ip(Inner Product),cosine(Cosine Distance)
                - category[str or None] : Category by collection
                - created_by[str or None] : Creator by collection
                - description[str or None] : Descript about collection
                - version[int or None] : Version about collection
        """
        if "name" not in kwargs:
            raise Exception("Please enter the collection name 'name'")
        self.client = client  # Chroma Python SDK Client

        # fix hnsw:space cosine-distance `ChromaDocumentMangager`` v0.0.1
        if "metadata" in kwargs:
            if "hnsw:space" not in kwargs["metadata"]:
                kwargs["metadata"]["hnsw:space"] = "cosine"
        else:
            kwargs["metadata"] = dict({"hnsw:space": "cosine"})

        # Create Collection
        self.collection = client.get_or_create_collection(
            name=kwargs["name"],
            configuration=kwargs.get("configuration", None),
            data_loader=kwargs.get("data_loader", None),
            embedding_function=(
                embedding_functions.DefaultEmbeddingFunction()
                if embedding is None
                else CustomEmbeddingFunction(embedding)
            ),
            metadata=kwargs.get("metadata", None),
        )

        # embedding object
        self.embedding = embedding

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        texts: document or texts
        metadatas: metadata
        ids: unique ids, If the ids are None, automatically create and insert them.
        """

        if ids is None:  # if the ids are None
            ids = [str(uuid4()) for _ in range(len(texts))]

        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        )

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> None:
        # Create Batches
        batches = []
        total = len(texts)

        # I think it takes less time to just do `upsert` work several times than to create batches.....
        batches = [
            (
                texts[i : i + batch_size],
                metadatas[i : i + batch_size] if metadatas else None,
                ids[i : i + batch_size] if ids else None,
            )
            for i in range(0, total, batch_size)
        ]
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(lambda batch: self.upsert(*batch, **kwargs), batches)

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """
        Default Scoring : Cosine Similarity
        """
        # embedding query
        if self.embedding is None:
            query_embed = embedding_functions.DefaultEmbeddingFunction()([query])
        else:
            query_embed = self.embedding.embed_documents([query])

        # documents
        where_condition = kwargs["where"] if kwargs and "where" in kwargs else None

        where_document_condition = (
            kwargs["where_document"] if kwargs and "where_document" in kwargs else None
        )

        result = self.collection.query(
            query_embeddings=query_embed,
            n_results=k,
            where=where_condition,
            where_document=where_document_condition,
        )
        # Calculate Cosine Similarity
        # Cosine Similarity = 1 - Cosine Distance
        result["distances"] = [
            list(map(lambda x: round(1 - x, 2), result["distances"][0]))
        ]

        # Change Format Langchain Documents
        return [
            Document(
                page_content=document,
                metadata={"id": id, "score": distance, **metadata},
            )
            for document, id, distance, metadata in zip(
                result["documents"][0],
                result["ids"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            - ids: The ids of the embeddings to delete
            - filters
                - where: A Where type dict used to filter the delection by. E.g. {"$and": [{"color" : "red"}, {"price": {"$gte": 4.20}]}}. Optional.
                - where_document: A WhereDocument type dict used to filter the deletion by the document content. E.g. {$contains: {"text": "hello"}}. Optional.
        """

        if ids is None:  # all delete
            ids = self.collection.get(include=[])["ids"]

        where_condition = filters["where"] if filters and "where" in filters else None

        where_document_condition = (
            filters["where_document"]
            if filters and "where_document" in filters
            else None
        )

        self.collection.delete(
            ids=ids,
            where=where_condition,
            where_document=where_document_condition,
        )
