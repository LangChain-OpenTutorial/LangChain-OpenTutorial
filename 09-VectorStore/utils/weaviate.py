import weaviate
import logging
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from utils.base import DocumentManager
from weaviate.classes.config import Property, DataType
from weaviate.util import generate_uuid5
from weaviate.classes.query import MetadataQuery, Filter
from langchain.schema import Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WeaviateDocumentManager(DocumentManager):
    def __init__(self, client, collection_name, embeddings: Optional[Any] = None):
        self.client = client
        self.collection_name = collection_name
        self._embeddings = embeddings
        if not self._collection_exists():
            self._create_collection_with_vectorizer(text_key="text")

    def _collection_exists(self) -> bool:
        all_cols = self.client.collections.list_all().keys()
        exists = self.collection_name.lower() in {name.lower() for name in all_cols}

        print(f"[Weaviate] Collection '{self.collection_name}' {'exists' if exists else 'does not exist'}")
        return exists

    def _create_collection_with_vectorizer(self, text_key: str):
        props = [Property(name=text_key, data_type=DataType.TEXT)]
        vec_conf = [
            Configure.NamedVectors.text2vec_openai(
                name=f"{self.collection_name}_vector",
                model="text-embedding-3-large",
                source_properties=[text_key],
            )
        ]

        try:
            self.client.collections.create(
                name=self.collection_name,
                description=f"auto-created '{self.collection_name}'",
                properties=props,
                vectorizer_config=vec_conf,
            )
            print(f"[Weaviate] Created collection '{self.collection_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to auto-create schema: {e}")

    def create_collection(
        self,
        description: str,
        properties: List[Property],
        vectorizer_config: Configure.Vectorizer,
        generative_config: Optional[Configure.Generative] = None,
        **kwargs: Any,
    ) -> None:
        # Create the collection in Weaviate
        try:
            self.client.collections.create(
                name=self.collection_name,
                description=description,
                properties=properties,
                vectorizer_config=vectorizer_config
                or Configure.Vectorizer.text2vec_weaviate(),
                generative_config=generative_config or Configure.Generative.cohere(),
                **kwargs,
            )
            print(f"Collection '{self.collection_name}' created successfully.")

        except Exception as e:
            print(f"Failed to create collection '{self.collection_name}': {e}")

    def get_collection(self) -> Any:
        """Get Weaviate collection"""
        return self.client.collections.get(self.collection_name)

    def delete_collection(self):
        self.client.collections.delete(self.collection_name)
        print(f"Deleted index: {self.collection_name}")

    def delete_all_collections(self):
        self.client.collections.delete_all()
        print("Deleted all collections")

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        show_progress: bool = False,
        text_key: str = "text",
    ) -> None:
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [generate_uuid5({"text": txt}) for txt in texts]

        if self._embeddings:
            embeddings = self._embeddings.embed_documents(list(texts))
        else:
            embeddings = [None] * len(texts)

        collection = self.client.collections.get(self.collection_name)

        with collection.batch.dynamic() as batch:
            for idx, (text, md, uid, emb) in enumerate(
                zip(texts, metadatas, ids, embeddings), start=1
            ):
                props = {text_key: text, **md}
                batch.add_object(
                    properties=props,
                    vector=emb,
                    uuid=uid,
                )
                if show_progress:
                    print(f"Upserted {idx}/{len(list(texts))}")

        return ids

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        workers: int = 4,
        show_progress: bool = False,
        text_key: str = "text",
    ) -> None:
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [generate_uuid5({"text": txt}) for txt in texts]

        if self._embeddings:
            embeddings = self._embeddings.embed_documents(texts)
        else:
            embeddings = [None] * len(texts)

        collection = self.client.collections.get(self.collection_name)

        with collection.batch.fixed_size(
            batch_size=batch_size, concurrent_requests=workers
        ) as batch:
            for idx, (text, md, uid, emb) in enumerate(
                zip(texts, metadatas, ids, embeddings), start=1
            ):
                props = {text_key: text, **md}
                batch.add_object(
                    properties=props,
                    vector=emb,
                    uuid=uid,
                )
                if show_progress and idx % batch_size == 0:
                    print(f"Upserted {idx}/{len(texts)} documents")

        print(f"[Weaviate] {len(ids)} Documents Insert Complete")
        return ids

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Union[Filter, Dict[str, Any]]] = None,
    ) -> bool:

        col = self.client.collections.get(self.collection_name)

        def get_deleted_count(res):
            if isinstance(res, int):
                return res
            if hasattr(res, "successful"):
                succ = res.successful
                return succ if isinstance(succ, int) else len(succ)
            return 0

        if ids and not filters:
            if len(ids) == 1:
                success = col.data.delete_by_id(ids[0])
                deleted_count = 1 if success else 0
            else:
                id_filter = Filter.by_id().contains_any(ids)
                res = col.data.delete_many(where=id_filter)
                deleted_count = get_deleted_count(res)
            print(f"[Weaviate] {deleted_count} document(s) deleted by ID")
            return True

        if ids and filters:
            id_filter = Filter.by_id().contains_any(ids)

            if isinstance(filters, Filter):
                meta_filter = filters
            else:
                key, value = next(iter(filters.items()))
                meta_filter = Filter.by_property(key).equal(value)

            combined = id_filter & meta_filter
            res = col.data.delete_many(where=combined)
            deleted_count = get_deleted_count(res)
            print(f"[Weaviate] {deleted_count} document(s) deleted by ID+filter")
            return True

        self.client.collections.delete(self.collection_name)
        print(f"[Weaviate] Deleted collection: {self.collection_name}")
        return True

    def search(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
        text_key: str = "text",
    ) -> List[Document]:
        results = []
        collection = self.client.collections.get(self.collection_name)
        response = collection.query.near_text(
            query=query,
            filters=filters,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )

        for obj in response.objects:
            prop = obj.properties or {}
            text = prop.get(text_key, "")
            meta = {k: v for k, v in prop.items() if k != text_key}
            meta["uuid"] = str(obj.uuid)
            meta["distance"] = getattr(obj.metadata, "distance", None)
            results.append(Document(page_content=text, metadata=meta))

        return results

    def list_collections(self):
        """
        Lists all collections (indexes) in the Weaviate database, including their properties.
        """
        # Retrieve all collection configurations
        collections = self.client.collections.list_all()

        # Check if there are any collections
        if collections:
            print("Collections (indexes) in the Weaviate schema:")
            for name, config in collections.items():
                print(f"- Collection name: {name}")
                print(
                    f"  Description: {config.description if config.description else 'No description available'}"
                )
                print(f"  Properties:")
                for prop in config.properties:
                    print(f"    - Name: {prop.name}, Type: {prop.data_type}")
                print()
        else:
            print("No collections found in the schema.")


class VectorStoreRetriever:
    def __init__(
        self,
        vectorstore,
        search_type="similarity",
        search_kwargs=None,
        search_fn=None,
        **kwargs,
    ):
        self.vectorstore = vectorstore
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}
        self.kwargs = kwargs
        self.search_fn = search_fn

    def get_relevant_documents(self, query):
        if self._embeddings:
            query_vector = self._embeddings.embed_query(query)
            return self.vectorstore.search(
                query_vector=query_vector,
                search_fn=self.search_fn,
                **self.search_kwargs,
            )
        return []

    def __call__(self, query):
        return self.get_relevant_documents(query)

    @property
    def _embeddings(self):
        return self.vectorstore._embeddings

    def invoke(self, query):
        return self.get_relevant_documents(query)


class WeaviateDBManager(DocumentManager):
    def __init__(
        self, client: Any, api_key: str, url: str, embeddings: Optional[Any] = None
    ):
        self.api_key = api_key
        self.url = url
        self.client = None
        self._embeddings = embeddings

    def connect(self, **kwargs) -> None:
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
            **kwargs,
        )
        print("[Weaviate] Connection successful")

    def preprocess_documents(
        self,
        documents: List[Document],
        id_key: str = "id",
        text_key: str = "text",
        vector_key: str = "vector",
    ) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []

        for doc in documents:
            raw_id = doc.metadata.get(id_key)
            uid = raw_id or generate_uuid5({text_key: doc.page_content})
            text = doc.page_content
            vector: Optional[List[float]] = doc.metadata.get(vector_key)
            meta = {
                k: v for k, v in doc.metadata.items() if k not in {id_key, vector_key}
            }

            processed.append(
                {
                    "id": uid,
                    "text": text,
                    "metadata": meta,
                    "vector": vector,
                }
            )

        return processed

    def get_api_key(self) -> str:
        return self.api_key
