from utils.base import DocumentManager
from typing import Optional, Any, Dict, Iterable, List, Callable
from pinecone import Pinecone, Vector, ServerlessSpec
from uuid import uuid4
from langchain_core.documents import Document
import traceback

# from concurrent.futures import ThreadPoolExecutor


class PineconeDocumentManager(DocumentManager):
    def __init__(
        self,
        client: Pinecone,
        embedding: Callable,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.pinecone_client = client
        # If No Index Create Index
        if index_name is None:
            index_name = "test-pinecone"
        if index_name not in [
            index["name"] for index in self.pinecone_client.list_indexes()
        ]:
            self.pinecone_client.create_index(
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                name=index_name,
                dimension=1536,  # text-embedding-3-large
                metric="cosine",  # This Tutorial used cosine similarity
            )
        self.index = self.pinecone_client.Index(index_name)
        self.namespace = namespace
        self.embedding = embedding

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Upserts documents into the Pinecone index.

        Args:
            texts (Iterable[str]): A list of text strings to be embedded and upserted.
            metadatas (Optional[List[Dict]]): A list of metadata dictionaries corresponding to each text.
            ids (Optional[List[str]]): A list of unique identifiers for the vectors.
            **kwargs (Any): Additional keyword arguments to be passed to the upsert operation.

        Raises:
            Exception: If any error occurs during the upsert process.
        """
        try:
            vectors = self.embedding(list(texts))

            if ids is None:
                ids = [uuid4() for _ in range(len(texts))]
            if metadatas is None:
                metadatas = [{"text": text} for text in range(len(texts))]
            else:
                metadatas = [
                    {**(meta if meta is not None else {}), "text": text}
                    for meta, text in zip(metadatas or [{}] * len(texts), texts)
                ]
            vector_datas = [
                Vector(id=id, values=vector, metadata=metadata)
                for id, vector, metadata in zip(ids, vectors, metadatas)
            ]

            result = self.index.upsert(
                vectors=vector_datas, namespace=self.namespace, show_progress=False
            )

            print(f"{len(vector_datas)} data upserted")

        except Exception as e:
            print(f"Error : Upsert Failed | Msg:")
            traceback.print_exc()

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
        Upserts documents in parallel into the Pinecone index.

        Args:
            texts (Iterable[str]): A list of text strings to be embedded and upserted.
            metadatas (Optional[List[Dict]]): A list of metadata dictionaries corresponding to each text.
            ids (Optional[List[str]]): A list of unique identifiers for the vectors.
            batch_size (int): Number of documents to process in each batch.
            workers (int): Number of threads to use for parallel processing.
            **kwargs (Any): Additional keyword arguments for the upsert operation.

        Raises:
            Exception: If any error occurs during the upsert process.
        """
        try:
            # Generate embeddings
            vectors = self.embedding(list(texts))

            # Generate unique IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in range(len(texts))]

            # Replace None with {} and add text to each metadata
            metadatas = [
                {**(meta if meta is not None else {}), "text": text}
                for meta, text in zip(metadatas or [{}] * len(texts), texts)
            ]

            vector_datas = [
                Vector(id=id, values=vector, metadata=metadata)
                for id, vector, metadata in zip(ids, vectors, metadatas)
            ]

            result = self.index.upsert(
                vectors=vector_datas,
                namespace=self.namespace,
                show_progress=False,
                batch_size=batch_size,
            )
            print(f"{len(vector_datas)} data upserted")
            # Batch processing function
        #     def batch_upsert(batch):
        #         try:
        #             vectors = [(ids[i], embeddings[i], metadatas[i]) for i in batch]
        #             self.index.upsert(
        #                 vectors=vectors, namespace=self.namespace, show_progress=False
        #             )
        #             print(f"Upserted batch of {len(vectors)} vectors.")
        #         except Exception as e:
        #             print(f"Error in batch upsert: {e}")

        #     # Parallel processing with ThreadPoolExecutor
        #     with ThreadPoolExecutor(max_workers=workers) as executor:
        #         batches = [
        #             range(i, min(i + batch_size, len(texts)))
        #             for i in range(0, len(texts), batch_size)
        #         ]
        #         executor.map(batch_upsert, batches)

        #     print(f"Parallel upsert completed for {len(texts)} vectors.")

        except Exception as e:
            print(f"Error: Parallel upsert failed | Msg:")
            traceback.print_exc()

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """
        Searches for the top-k most similar documents in the Pinecone index.

        Args:
            query (str): The input text query for similarity search.
            k (int): The number of top similar documents to retrieve. Default is 10.
            **kwargs (Any): Additional keyword arguments for the search operation.

        Returns:
            List[Document]: A list of Document objects containing the retrieved texts and metadata.

        Raises:
            Exception: If any error occurs during the search process.
        """

        try:
            query_vector = self.embedding([query])[0]

            response = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
                namespace=self.namespace,
                **kwargs,
            )

            documents = [
                Document(
                    page_content=match["metadata"].get("text", "No content"),
                    metadata=match["metadata"],
                )
                for match in response["matches"]
            ]
            return documents
        except Exception as e:
            print(f"Error: Search Failed | Msg :")
            traceback.print_exc()

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Deletes documents from the Pinecone index based on IDs or filters.

        Args:
            ids (Optional[List[str]]): A list of unique identifiers for the vectors to delete.
            filters (Optional[Dict]): A dictionary of conditions to filter documents for deletion.
            **kwargs (Any): Additional keyword arguments for the delete operation.

        Raises:
            Exception: If any error occurs during the delete process.
        """
        try:
            if not ids and not filters:
                self.index.delete(delete_all=True, namespace=self.namespace, **kwargs)
                print("All Data Deleted..")

            elif ids and not filters:
                self.index.delete(ids=ids, namespace=self.namespace, **kwargs)
                print(f"Delete by ids: {len(ids)} data deleted")
            elif not ids and filters:
                self.index.delete(filter=filters, namespace=self.namespace, **kwargs)
                print(f"Delete by Filter: {filters}")
            elif ids and filters:
                matched_ids = []
                for _id in ids:
                    if (
                        self.index.fetch(ids=[_id], namespace=self.namespace)
                        .vectors[_id]
                        .metadata["title"]
                        == filters["title"]
                    ):
                        matched_ids.append(_id)

                self.index.delete(ids=matched_ids, namespace=self.namespace, **kwargs)
                print(f"{len(matched_ids)} Data Deleted")
            return
        except Exception as e:
            print(f"Error: Delete Failed | Msg:")
            traceback.print_exc()
