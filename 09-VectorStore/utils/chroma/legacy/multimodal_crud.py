from utils.chroma.crud import ChromaDocumentMangager, CustomEmbeddingFunction
from chromadb.utils.embedding_functions import open_clip_embedding_function
from langchain_chroma.vectorstores import cosine_similarity
from langchain_core.documents import Document
from PIL import Image
from typing import Iterable, Any, Optional, List, Dict, Tuple, Union
import base64
import io
import tempfile
import uuid
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, ConfigDict


class ImageDocument(BaseModel):
    id: str
    image: Image.Image
    metadata: Optional[Dict] = None  # Metadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChromaMultimodalDocumentMangager(ChromaDocumentMangager):
    def __init__(self, client, embedding=None, **kwargs):
        if embedding is None:  # Not embedding
            embedding = open_clip_embedding_function.OpenCLIPEmbeddingFunction()
        """
        Chroma supports data loaders, for storing and querying with data stored outside Chroma itself, via URI.
        Chroma will not store this data, but will instead store the URI, and load the data from the URI when needed.
        """
        super().__init__(client, embedding, **kwargs)

    # base64 to Image
    def toPIL(self, image_str: str) -> Image.Image:
        image_data = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_data))
        return image

    # Get Image URI
    def getURIfromPILImage(self, image: Image) -> str:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        try:
            image.save(temp_file, format="JPEG")
            # Close the file to allow other processes to access it
            temp_file.close()
            return temp_file.name
        except Exception as e:
            temp_file.close()
            os.remove(temp_file.name)

    # Get base64 string from image URI.
    def _encode_image(self, uri: str) -> str:
        with open(uri, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # image upsert
    def image_upsert(
        self,
        images: Iterable[Image.Image],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Reference Langchain-Chroma `add_image` .
        """

        # Map from uris to b64 encoded strings
        b64_texts = []
        uris = []
        for image in images:
            # Get PIL.Image to URI
            uri = self.getURIfromPILImage(image)
            uris.append(uri)
            b64_texts.append(self._encode_image(uri))

        # Populate IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in uris]
        else:
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]

        embeddings = None

        # Setting Embedding
        if self.embedding is not None and hasattr(self.embedding, "embed_image"):
            embeddings = self.embedding.embed_image(uris=uris)

        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all images
            length_diff = len(uris) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)

            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                images_with_metadatas = [b64_texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self.collection.upsert(
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=images_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e

            if empty_ids:
                images_without_metadatas = [b64_texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self.collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=images_without_metadatas,
                    ids=ids_without_metadatas,
                )

        else:
            self.collection.upsert(embeddings=embeddings, documents=b64_texts, ids=ids)

    # image upsert parallel
    def image_upsert_parallel(
        self,
        images: Iterable[Image.Image],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs,
    ) -> None:
        # Create Batches
        batches = []
        total = len(images)

        batches = [
            (
                images[i : i + batch_size],
                metadatas[i : i + batch_size] if metadatas else None,
                ids[i : i + batch_size] if ids else None,
            )
            for i in range(0, total, batch_size)
        ]

        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(lambda batch: self.upsert(*batch, **kwargs), batches)

    # Search Tools

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
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
        return self.collection.query(
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

        if self.embedding is None:
            results = self._query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        else:
            query_embedding = self.embedding.embed_query(query)
            results = self._query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                **kwargs,
            )
        return self._results_to_docs_and_scores(results)

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

    def _results_to_docs(self, results: Any) -> List[Document]:
        return [doc for doc, _ in self._results_to_docs_and_scores(results)]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self._query_collection(
            query_embeddings=embedding,
            n_results=k,
            where=filter,
            where_document=where_document,
            **kwargs,
        )
        return self._results_to_docs(results)

    # image search by cosine similarity
    def search_image(
        self,
        image_or_text: Union[Image.Image, str],
        k: int = 1,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[float, List[ImageDocument]]]:

        if isinstance(image_or_text, Image.Image):
            # if image_or_text is PIL.Image to Image URI
            image_or_text = self.getURIfromPILImage(image_or_text)

            if self.embedding is None or not hasattr(self.embedding, "embed_image"):
                raise ValueError("The embedding function must support image embedding.")
            # Embedding Image Query
            emq = self.embedding.embed_image(uris=[image_or_text])

            docs = self.similarity_search_by_vector(
                embedding=emq, k=k, filter=filters, **kwargs
            )

            # Reshape
            emq = np.array(emq).reshape(-1).tolist()

        else:  # Text
            docs = self.similarity_search(
                query=image_or_text, k=k, filter=filters, **kwargs
            )
            # Embedding Text Query
            emq = self.embedding.embed_query(image_or_text)

        # cosine_similarity
        temp_uris = [
            self.getURIfromPILImage(self.toPIL(doc.page_content)) for doc in docs
        ]

        emdocs = self.embedding.embed_image(uris=temp_uris)

        scores = sorted(
            cosine_similarity(
                emq if isinstance(image_or_text, Image.Image) else [emq], emdocs
            )[0],
            reverse=True,
        )

        return [
            (
                score,
                ImageDocument(
                    **{
                        "id": doc.metadata["id"],
                        "image": self.toPIL(doc.page_content),
                        "metadata": {
                            "category": doc.metadata["category"],
                            "prompt": doc.metadata["prompt"],
                        },
                    }
                ),
            )
            for score, doc in zip(scores, docs)
        ]
