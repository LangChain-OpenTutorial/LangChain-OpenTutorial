from utils.vectorstoreInterface import VectorDBInterface
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, PointIdsList, VectorParams, Distance
from langchain_openai import OpenAIEmbeddings


class QdrantDB(VectorDBInterface):
    def __init__(self) -> None:
        """
        Initializes the QdrantDB instance.
        """
        self.client: Optional[QdrantClient] = None
        self.collection_name: Optional[str] = None
        self.embedding_model: Optional[OpenAIEmbeddings] = None

    def connect(
        self,
        collection_name: str,
        embeddings,
        **kwargs: Any,
    ) -> None:
        """
        Connects to the Qdrant database.

        Args:
            collection_name (str): The name of the collection.
            embeddings (OpenAIEmbeddings): An instance of the embedding model to be used.
            **kwargs: Additional connection parameters.
        """
        if not self.client:
            self.client = QdrantClient(**kwargs)
            self.collection_name = collection_name
            self.embedding_model = embeddings
            print(f"Connected to QdrantClient with parameters: {kwargs}")

    def create_collection(
        self, vector_size: Optional[int] = None, distance: Distance = Distance.COSINE
    ) -> None:
        """
        Creates a new collection in the Qdrant database.

        Args:
            vector_size (Optional[int]): The size of the vectors. Defaults to None.
            distance (Distance): The distance metric for the collection. Defaults to COSINE distance.
        """
        self._ensure_collection_set()
        vector_size = vector_size or len(self.generate_vector("test"))
        self.client.create_collection(
            self.collection_name, VectorParams(size=vector_size, distance=distance)
        )
        print(f"Collection '{self.collection_name}' created successfully.")

    def delete_collection(self) -> None:
        """
        Deletes the specified collection.
        """
        self._ensure_collection_set()
        self.client.delete_collection(self.collection_name)
        print(f"Collection '{self.collection_name}' deleted successfully.")

    def add_documents(self, split_docs: List[Any], parallel: bool = False) -> List[str]:
        """
        Adds documents to the collection.

        Args:
            split_docs (List[Any]): The list of document objects to be added.
            parallel (bool, optional): Whether to perform parallel processing. Defaults to False.

        Returns:
            List[str]: List of document IDs added.
        """
        self._ensure_collection_set()
        import uuid

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=self.generate_vector(doc.page_content),
                payload={"text": doc.page_content, "metadata": doc.metadata},
            )
            for doc in split_docs
        ]
        if parallel:
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(
                    executor, self.client.upsert, self.collection_name, points
                )
        else:
            self.client.upsert(self.collection_name, points)
        return [point.id for point in points]

    def delete_documents(self, parallel: bool = False, **kwargs: Any) -> bool:
        """
        Deletes documents from the collection.

        Args:
            parallel (bool, optional): Whether to perform parallel processing. Defaults to False.
            **kwargs: Additional filter parameters for deletion.

        Returns:
            bool: True if the deletion is successful.
        """
        self._ensure_collection_set()
        ids_to_delete = kwargs.get("ids", [])
        if ids_to_delete:
            if parallel:
                with ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(
                        executor,
                        self.client.delete,
                        self.collection_name,
                        PointIdsList(points=ids_to_delete),
                    )
            else:
                self.client.delete(
                    self.collection_name, PointIdsList(points=ids_to_delete)
                )
        return True

    def scroll(self, k: int = 5, **kwargs: Any) -> List[Any]:
        """
        Retrieves documents using scroll.

        Args:
            k (int, optional): Number of documents to retrieve. Defaults to 5.
            **kwargs: Additional scroll parameters.

        Returns:
            List[Any]: Retrieved documents.
        """
        self._ensure_collection_set()
        return self.client.scroll(self.collection_name, limit=k, **kwargs)

    def generate_vector(self, text: str) -> List[float]:
        """
        Converts text into an embedding vector.

        Args:
            text (str): The input text to be embedded.

        Returns:
            List[float]: The generated embedding vector.
        """
        return self.embedding_model.embed_query(text)

    def similarity_search(self, query: str, top_k: int, **kwargs: Any) -> List[Any]:
        """
        Performs similarity search.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
            **kwargs: Additional search parameters.

        Returns:
            List[Any]: Search results.
        """
        self._ensure_collection_set()
        query_vector = self.generate_vector(query)
        return self.client.search(
            self.collection_name, query_vector=query_vector, limit=top_k, **kwargs
        )

    def as_retriever(self, top_k: int = 10) -> Any:
        """
        Returns a retriever function for similarity search.

        Args:
            top_k (int, optional): Number of top results to return. Defaults to 10.

        Returns:
            Callable: A retriever function that performs search.
        """
        return lambda query: self.similarity_search(query, top_k)

    def _ensure_collection_set(self) -> None:
        """
        Ensures that a collection is set before performing operations.

        Raises:
            ValueError: If collection name is not set.
        """
        if not self.collection_name:
            raise ValueError(
                "Collection name is not set. Please connect to a collection first."
            )

    def getRetrival(self):

        return self.client
