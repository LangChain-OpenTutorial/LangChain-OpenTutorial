# Interface
from utils.chroma.base import VectorDBInterface

# Chroma Python SDK
import chromadb
from chromadb.utils import embedding_functions

# Langchain-Type and function
from langchain_core.documents import Document
from langchain_chroma.vectorstores import cosine_similarity
from langchain_core.vectorstores.base import VectorStoreRetriever

# Python Library
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    # return [
    #     # TODO: Chroma can do batch querying,
    #     # we shouldn't hard code to the 1st result
    #     (
    #         Document(page_content=result[0], metadata=result[1] or {}),
    #         result[2],
    #     )
    #     for result in zip(
    #         results["documents"][0],
    #         results["metadatas"][0],
    #         results["distances"][0],
    #     )
    # ]
    docs_and_scores = []
    for doc, metadata, score, doc_id in zip(
        results["documents"][0],  # documents
        results["metadatas"][0],  # metadata
        results["distances"][0],  # distances(similarity)
        results["ids"][0],  # document ID
    ):
        document = Document(page_content=doc, metadata=metadata)
        document.metadata["id"] = doc_id  # id insert in metadata
        docs_and_scores.append((document, score))
    return docs_and_scores


class ChromaDB(VectorDBInterface):
    def __init__(self, embeddings: Optional[Any] = None) -> None:
        self.chroma = None
        self.unique_ids = set()
        self._embeddings = embeddings if embeddings is not None else None
        self._embeddings_function = (
            embeddings.embed_documents
            if embeddings is not None
            else embedding_functions.DefaultEmbeddingFunction  # all-MiniLM-L6v2
        )

    def connect(self, **kwargs) -> None:
        """
        ChromaDB Connect
        """

        if kwargs["mode"] == "in-memory":  # In-Memory
            chroma_client = chromadb.Client()

        elif kwargs["mode"] == "persistent":  # Local
            chroma_client = chromadb.PersistentClient(path=kwargs["persistent_path"])

        elif kwargs["mode"] == "server":  # Server-Client
            chroma_client = chromadb.HttpClient(
                host=kwargs["host"], port=kwargs["port"]
            )
        else:
            raise Exception(
                "Invalid Input, Enter one of ['in-meory','persistent','server'] modes."
            )

        # The Chroma client allows you to get and delete existing collections by their name.
        # It also offers a get or create method to get a collection if it exists, or create it otherwise.

        # l2(default) : squared L2 norm
        # ip : Inner Product
        # cosine : Cosine Distance
        metadata = {
            "hnsw:space": (
                kwargs.get("hnsw:space") if kwargs.get("hnsw:space", None) else "l2"
            )
        }

        self.chroma = chroma_client.get_or_create_collection(
            name=kwargs["collection"], metadata=metadata
        )  # make collection

        # langchain_config["collection_name"] = kwargs["collection"]
        # langchain_config["collection_metadata"] = metadata

        existing_ids = self.chroma.get(include=[])["ids"]  # Get existing unique
        self.unique_ids.update(existing_ids)  # current unique ids update

    def from_text(self):
        pass

    def create_index(
        self, index_name: str, dimension: int, metric: str = "dotproduct", **kwargs
    ) -> Any:
        """
        Not used in Chroma
        """
        return None

    def get_index(self, index_name: str) -> Any:
        """
        Not used in Chroma
        """
        return None

    def delete_index(self, index_name: str) -> None:
        """
        Not used in Chroma
        """
        return None

    def list_indexs(self) -> List[str]:
        """
        Not used in Chroma
        """
        return None

    def add(self, pre_documents: List[Document], **kwargs) -> None:
        documents = []
        metadatas = []
        ids = []
        for doc in pre_documents:
            documents.append(doc.page_content)
            ids.append(doc.metadata["id"])
            metadatas.append(
                {key: value for key, value in doc.metadata.items() if key != "id"}
            )

        embeddings = self._embeddings_function(documents)  # embedding documents

        self.chroma.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        self.unique_ids.update(ids)

    def upsert_documents(
        self,
        documents: List[Dict],
        **kwargs,
    ) -> None:
        """
        Upsert documents to Chroma

        :param documents: List of documents
        :param embedding_function: Embedding function
        """
        # Embedding documents
        embeddings = self._embeddings_function([doc.page_content for doc in documents])
        # Generate unique ids
        unique_ids = [doc.metadata["id"] for doc in documents]
        # Upsert documents
        self.chroma.upsert(
            ids=unique_ids,
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in documents],
            documents=[doc.page_content for doc in documents],
        )

        # update unique_ids
        self.unique_ids.update(unique_ids)

    def upsert_documents_parallel(
        self,
        documents: List[Dict],
        batch_size: int = 32,
        max_workers: int = 10,
        **kwargs,
    ) -> None:
        """
        Parallel upsert documents to Chroma
        :param documents: List of documents
        :param batch_size: Batch size
        :param max_workers: Number of workers
        """
        # split documents into batches
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        all_unique_ids = set()  # Store all unique IDs from all batches
        failed_uids = []  # Store failed batches

        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.upsert_documents, batch, **kwargs)
                for batch in batches
            ]

        # Wait for all futures to complete
        for future, batch in zip(as_completed(futures), batches):
            try:
                future.result()  # Wait for the batch to complete
                # Extract unique IDs from the batch
                unique_ids = [doc.metadata["id"] for doc in batch]
                all_unique_ids.update(unique_ids)  # Add to the total set
            except Exception as e:
                print(f"An error occurred during upsert: {e}")
                failed_uids.append(unique_ids)  # Store failed batch for retry

        self.unique_ids.update(all_unique_ids)

    def _cosine_similarity_search_text(
        self, query: str, configs: Dict
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid Search : Text Search + Cosine Similarity
        """
        docs = self.similarity_search(**configs)

        embx = self._embeddings.embed_query(query)
        emb_d = self._embeddings.embed_documents([doc.page_content for doc in docs])

        scores = cosine_similarity([embx], emb_d)

        return sorted(
            [(doc, score) for score, doc in zip(scores[0], docs)],
            key=lambda x: x[1],
            reverse=True,
        )

    def _query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Query the chroma collection"""
        return self.chroma.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with chroma with distance(from langchain-chroma)"""

        if self._embeddings is None:
            results = self._query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        else:
            query_embedding = self._embeddings.embed_query(query)
            results = self._query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        return _results_to_docs_and_scores(results)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Chroma(from langchain-chroma)"""
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def query(
        self,
        query: str,
        top_k: int = 10,
        score: bool = False,
        filters: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        cs: bool = False,
        **kwargs,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        A Method that implements a search method using a LangChain-Chroma library.
        """

        configs = {
            "query": query,
            "k": top_k,
            "filter": filters,
            **kwargs,
        }
        # similarity_search
        if score:
            configs["where_document"] = where_document
            results = self.similarity_search_with_score(
                **configs
            )  # distance search score

        elif cs:  # cosine similarity search
            return self._cosine_similarity_search_text(query, configs)
        else:
            results = self.similarity_search(**configs)

        return results

    def delete_by_filter(
        self, unique_ids: List[str], filters: Optional[Dict] = None, **kwargs
    ) -> None:
        """
        Delete documents by filter
        :param unique_ids: List of unique ids
        :param filters: Filter conditions
        """
        try:
            self.chroma.delete(
                ids=unique_ids,
                where=filters,
            )
            pre_count = len(self.unique_ids)
            self.unique_ids = set(self.chroma.get(include=[])["ids"])

            print(f"Success Delete {pre_count-len(self.unique_ids)} Documents")

        except Exception as e:
            print(f"Error: {e}")

    def _get_retriever_tags(self) -> list[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        if self._embeddings:
            tags.append(self._embeddings.__class__.__name__)
        return tags

    def getRetriever(
        self,
        search_type: str = "mmr",
        search_kwargs: Optional[Dict] = None,
        tags: Optional[List] = None,
    ) -> VectorStoreRetriever:
        """
        ** Not implemented due to VectorStore inheritance problem **

        Get Retriever Method using a LangChain-Chroma library.

        Refer to the following document -> LangChain-Chroma Official Document.

        :param search_type: [similarity(default), mmr, similarity_score_threshold]
        :param search_kwargs: [k, fetch_k, lambda_mult, filter]

        """
        pass
        # return VectorStoreRetriever(
        #     vectorstore=self,
        #     tags=tags if tags is not None else [] + self._get_retriever_tags(),
        #     search_kwargs=search_kwargs,
        #     search_type=search_type,
        # )

    def preprocess_documents(
        self,
        documents: List[Document],
        source: Optional[str] = None,
        author: Optional[str] = None,
        chapter: bool = False,
        **kwargs,
    ) -> List[Dict]:
        """
        Change LangChain Document to Chroma

        Refer to the following document -> LangChain-Chroma Official Document.

        :param documents: List of LangChain documents
        :param source: Source of the document
        :param author: Author of the document
        :param chapter: Chapter of the document
        :return: List of Chroma documents
        """
        metadata = {}

        if source is not None:
            metadata["source"] = source
        if author is not None:
            metadata["author"] = author

        processed_docs = []
        current_chapter = None
        save_flag = False
        for doc in documents:
            content = doc.page_content

            content = content.replace("(picture)\n", "")

            # Chapter dectect
            if content.startswith("[ Chapter ") and "\n" in content:
                # Chapter Num (example: "[ Chapter 26 ]\n" -> 26)
                chapter_part, content_part = content.split("\n", 1)
                current_chapter = int(chapter_part.split()[2].strip("]"))
                content = content_part

            elif content.strip() == "[ END ]":
                break

            if current_chapter is not None:
                # add metadata
                if chapter:
                    metadata["chapter"] = current_chapter
                updated_metadata = {**doc.metadata, **metadata, "id": str(uuid4())}
                # Document append to processed_docs
                processed_docs.append(
                    Document(metadata=updated_metadata, page_content=content)
                )

        return processed_docs

    def get_api_key(self) -> str:
        """
        Not used in Chroma
        """
        return None
