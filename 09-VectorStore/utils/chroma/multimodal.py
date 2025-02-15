from utils.chroma.basic import (
    ChromaDB,
    cosine_similarity,
    ThreadPoolExecutor,
    as_completed,
    Document,
    _results_to_docs_and_scores,
)
from typing import Optional, Any, List, Tuple, Dict
import base64
import io
import tempfile
import uuid
from pydantic import BaseModel
from PIL import Image


class ImageMetadata(BaseModel):
    category: str
    prompt: str


class ImageDocumentChroma(BaseModel):
    id: str  # unique id
    image_uri: str  # image uri path
    metadata: Optional[ImageMetadata] = None  # Metadata


class ChromaMulitmodalDB(ChromaDB):
    def __init__(self, embeddings: Optional[Any] = None):
        super().__init__(embeddings)

    def _encode_image(self, uri: str) -> str:
        """Get base64 string from image URI."""
        with open(uri, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def add_images(
        self, image_documents: List[ImageDocumentChroma], **kwargs: Any
    ) -> List[str]:
        """
        Langchain-Chroma 'add_images method' supports 'upsert' by default.
        """

        uris = []
        metadatas = []
        ids = []

        for docs in image_documents:
            ids.append(docs.id)
            uris.append(docs.image_uri)
            metadatas.append(docs.metadata.dict())

        # Map from uris to b64 encoded strings
        b64_texts = [self._encode_image(uri=uri) for uri in uris]

        # Populate IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in uris]
        else:
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]

        embeddings = None

        # Setting Embedding
        if self._embeddings is not None and hasattr(self._embeddings, "embed_image"):
            embeddings = self._embeddings.embed_image(uris=uris)

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
                    self.chroma.upsert(
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
                self.chroma.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=images_without_metadatas,
                    ids=ids_without_metadatas,
                )

        else:
            self.chroma.upsert(embeddings=embeddings, documents=b64_texts, ids=ids)

        # update unique ids
        self.unique_ids.update(ids)
        return ids

    def upsert_images_parallel(
        self,
        image_uri_documents: List[ImageDocumentChroma],
        batch_size: int = 32,
        max_workers: int = 10,
        **kwargs,
    ) -> List[str]:

        batches = [
            image_uri_documents[i : i + batch_size]
            for i in range(0, len(image_uri_documents), batch_size)
        ]

        all_unique_ids = set()  # Store all unique IDs from all batches

        failed_uids = []  # Store failed batches

        # Parallel processing

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = [
                executor.submit(self.add_images, batch, **kwargs) for batch in batches
            ]

        # Wait fo all futures to complete
        for future, batch in zip(as_completed(futures), batches):
            try:
                ids = future.result()  # Wait for the batch to complete
                # Extract unique IDs from the batch
                unique_ids = [i for i in ids]
                all_unique_ids.update(unique_ids)  # Add to the total set
            except Exception as e:
                print(f"An error occurred during upsert: {e}")
                failed_uids.append(unique_ids)  # Store failed batch for retry

        self.unique_ids.update(all_unique_ids)

        return all_unique_ids

    def preprocess_image_documents(
        self, uris: List[str], prompts: List[str], categories: List[str]
    ) -> List[ImageDocumentChroma]:

        documents = [
            ImageDocumentChroma(
                **{
                    "id": str(uuid.uuid4()),
                    "image_uri": uri,
                    "metadata": ImageMetadata(
                        **{"prompt": prompt, "category": category}
                    ),
                }
            )
            for uri, prompt, category in zip(uris, prompts, categories)
        ]

        return documents

    def _results_to_docs(self, results: Any) -> List[Document]:
        return [doc for doc, _ in _results_to_docs_and_scores(results)]

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

    def searching_text_query(
        self,
        text_query: str,
        k: int = 1,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[float, List[ImageDocumentChroma]]]:

        # docs = self.chroma_search.as_retriever(search_kwargs={"k": k}).invoke(
        #     text_query
        # )
        docs = self.similarity_search(query=text_query, k=k, filter=filter, **kwargs)

        # cosine_similarity
        emq = self._embeddings.embed_query(text_query)
        emdocs = self._embeddings.embed_image(
            [self.getURIfromPILImage(self.toPIL(doc.page_content)) for doc in docs]
        )
        scores = sorted(cosine_similarity([emq], emdocs)[0], reverse=True)

        return [
            (
                score,
                ImageDocumentChroma(
                    **{
                        "id": doc.metadata["id"],
                        "image_uri": doc.page_content,
                        "metadata": ImageMetadata(
                            **{
                                "category": doc.metadata["category"],
                                "prompt": doc.metadata["prompt"],
                            }
                        ),
                    }
                ),
            )
            for score, doc in zip(scores, docs)
        ]

    def search_image_query(
        self, image_uri: str, k: int = 1, filters: Optional[Dict] = None, **kwargs: Any
    ) -> List[Tuple[float, List[ImageDocumentChroma]]]:

        if self._embeddings is None or not hasattr(self._embeddings, "embed_image"):
            raise ValueError("The embedding function must support image embedding.")

        image_embedding = self._embeddings.embed_image(uris=[image_uri])

        docs = self.similarity_search_by_vector(
            embedding=image_embedding, k=k, filter=filters, **kwargs
        )

        # cosine_similarity
        emdocs = self._embeddings.embed_image(
            [self.getURIfromPILImage(self.toPIL(doc.page_content)) for doc in docs]
        )
        scores = sorted(cosine_similarity(image_embedding, emdocs)[0], reverse=True)

        return [
            (
                score,
                ImageDocumentChroma(
                    **{
                        "id": doc.metadata["id"],
                        "image_uri": doc.page_content,
                        "metadata": ImageMetadata(
                            **{
                                "category": doc.metadata["category"],
                                "prompt": doc.metadata["prompt"],
                            }
                        ),
                    }
                ),
            )
            for score, doc in zip(scores, docs)
        ]

    def toPIL(self, image_str: str) -> Image.Image:
        image_data = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_data))
        return image

    def getURIfromPILImage(self, image: Image) -> str:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file, format="JPEG")

        # Close the file to allow other processes to access it
        temp_file.close()
        return temp_file.name
