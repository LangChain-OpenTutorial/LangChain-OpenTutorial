import os
import certifi
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Any, Mapping, Union, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient
from pymongo.synchronous.collection import Collection
from pymongo.synchronous.cursor import Cursor
from pymongo.typings import _DocumentType, _Pipeline
from pymongo.operations import SearchIndexModel
from pymongo.results import (
    DeleteResult,
    InsertManyResult,
    InsertOneResult,
    UpdateResult,
)
from bson import encode
from bson.raw_bson import RawBSONDocument
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.base import TextSplitter
from utils.vectordbinterface import DocumentManager


class MongoDBAtlas:
    """Manages MongoDB collections and vector store.
    Provides methods to add, update, delete indexes and manage documents in the vector store.
    """

    def __init__(self, db_name: str, collection_name: str):
        """Initialize a MongoDB client and configures the database.

        Args:
            db_name (str): The name of the database to connect to.
            collection_name (str): The name of the collection to use.
        """
        MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI, tlsCAFile=certifi.where())
        self.database = client[db_name]
        self.collection_name = collection_name
        self.collection = None
        self.vector_store = None

    def connect(self) -> Collection[_DocumentType]:
        """Create a collection."""
        collection_names = self.database.list_collection_names()
        if self.collection_name not in collection_names:
            self.collection = self.database.create_collection(self.collection_name)
        else:
            self.collection = self.database[self.collection_name]
        return self.collection

    def _is_index_exists(self, index_name: str) -> bool:
        """Check whether the specified search index exists in the collection.

        Args:
            index_name (str): The name of the search index to check.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        search_indexes = self.collection.list_search_indexes()
        index_names = [search_index["name"] for search_index in search_indexes]
        return index_name in index_names

    def create_index(
        self, index_name: str, model: Union[Mapping[str, Any], SearchIndexModel]
    ):
        """Create a search index if it does not already exist.

        Args:
            index_name (str): The name of the search index to create.
            model (Union[Mapping[str, Any], SearchIndexModel]): The model for the new search index.
        """
        if not self._is_index_exists(index_name):
            self.collection.create_search_index(model)

    def update_index(self, index_name: str, definition: Mapping[str, Any]):
        """Update a search index by replacing the existing index definition.

        Args:
            index_name (str): The name of the search index to update.
            definition ([Mapping[str, Any]): The new search index definition.
        """
        if self._is_index_exists(index_name):
            self.collection.update_search_index(name=index_name, definition=definition)

    def delete_index(self, index_name: str):
        """Delete a search index.

        Args:
            index_name (str): The name of the search index to delete.
        """
        if self._is_index_exists(index_name):
            self.collection.drop_search_index(index_name)

    def create_vector_store(
        self, embedding: Embeddings, index_name: str, relevance_score_fn: str
    ):
        """Create a vector store.
        `MongoDBAtlasVectorSearch` is a vector store that integrates Atlas Vector Search and Langchain.

        Args:
            embedding (Embeddings): Text embedding model to use.
            index_name (str): The name of the search index to create.
            relevance_score_fn (str): The similarity score used for the index
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'
        """
        self.vector_index_name = index_name
        self.embedding = embedding
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=embedding,
            index_name=index_name,
            relevance_score_fn=relevance_score_fn,
        )

    def get_embedding(self, text: str) -> List[float]:
        return self.embedding.embed_query(text)

    def create_vector_search_index(
        self,
        dimensions: int,
        filters: Optional[List[str]] = None,
        update: bool = False,
    ) -> None:
        if not self._is_index_exists(self.vector_index_name):
            self.vector_store.create_vector_search_index(
                dimensions=dimensions, filters=filters, update=update
            )

    def update_vector_search_index(
        self, dimensions: int, filters: Optional[List[str]] = None
    ) -> None:
        self.vector_store.create_vector_search_index(
            dimensions=dimensions, filters=filters, update=True
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        return self.vector_store.add_documents(documents=documents)

    def delete_documents(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        return self.vector_store.delete(ids=ids, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return self.vector_store.similarity_search(
            query=query, k=k, pre_filter=pre_filter, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(
            query=query, k=k, pre_filter=pre_filter, **kwargs
        )


class MongoDBAtlasDocumentManager(DocumentManager):

    def __init__(self, atlas: MongoDBAtlas) -> None:
        self.collection = atlas.connect()
        self.embedding_function = atlas.get_embedding

    def get_documents(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ) -> list[Document]:
        loader = TextLoader(file_path, encoding, autodetect_encoding)
        return loader.load()

    def split_documents(
        self,
        documents: Iterable[Document],
        split_condition: Callable[[str], Iterable[str]],
        split_index_name: str,
    ) -> List[Document]:
        return [
            Document(page_content=text, metadata=metadata)
            for document in documents
            for text, metadata in self.split_texts(
                document.page_content, split_condition, split_index_name
            )
        ]

    def split_documents_by_splitter(
        self, splitter: TextSplitter, documents: Iterable[Document]
    ) -> List[Document]:
        return splitter.split_documents(documents)

    def split_texts(
        self,
        texts: str,
        split_condition: Callable[[str], Iterable[str]],
        split_index_name: str,
    ) -> List[Tuple[str, dict[str, Any]]]:
        return [
            (document, {split_index_name: index})
            for index, document in enumerate(split_condition(texts))
        ]

    def convert_document_to_raw_bson(
        self,
        document: Mapping[str, Any],
    ) -> RawBSONDocument:
        """Convert Document to RawBSONDocument.
        RawBSONDocument represent BSON document using the raw bytes.
        BSON, the binary representation of JSON, is primarily used internally by MongoDB.
        """
        return RawBSONDocument(encode(document))

    def convert_documents_to_raw_bson(
        self,
        documents: List[Mapping[str, Any]],
    ) -> Iterable[RawBSONDocument]:
        """Convert a list of Document objects to an iterable of RawBSONDocument.

        Each Document is individually converted to RawBSONDocument using
        convert_document_to_raw_bson.
        """
        for document in documents:
            yield self.convert_document_to_raw_bson(document)

    def _insert_one(self, document: Mapping[str, Any]) -> InsertOneResult:
        bson_document = self.convert_document_to_raw_bson(document)
        return self.collection.insert_one(bson_document)

    def _insert_many(self, documents: List[Mapping[str, Any]]) -> InsertManyResult:
        bson_documents = self.convert_documents_to_raw_bson(documents)
        return self.collection.insert_many(bson_documents)

    def find(self, *args: Any, **kwargs: Any) -> Cursor[_DocumentType]:
        """Query the database

        :param filter: find all documents that match the condition.
        """
        return self.collection.find(*args, **kwargs)

    def find_one_by_filter(
        self, filter: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Optional[_DocumentType]:
        return self.collection.find_one(filter=filter, *args, **kwargs)

    def find_all_by_filter(self, *args: Any, **kwargs: Any) -> List[Mapping[str, Any]]:
        cursor = self.collection.find(*args, **kwargs)
        documents = []
        for doc in cursor:
            documents.append(doc)
        return documents

    def update_one_by_filter(
        self,
        filter: Mapping[str, Any],
        update_operation: Union[Mapping[str, Any], _Pipeline],
        upsert: bool = False,
    ) -> UpdateResult:
        return self.collection.update_one(filter, update_operation, upsert)

    def update_many_by_filter(
        self,
        filter: Mapping[str, Any],
        update_operation: Union[Mapping[str, Any], _Pipeline],
        upsert: bool = False,
    ) -> UpdateResult:
        return self.collection.update_many(filter, update_operation, upsert)

    def upsert_one_by_filter(
        self,
        filter: Mapping[str, Any],
        update_operation: Union[Mapping[str, Any], _Pipeline],
    ) -> UpdateResult:
        return self.update_one_by_filter(filter, update_operation, True)

    def upsert_many_by_filter(
        self,
        filter: Mapping[str, Any],
        update_operation: Union[Mapping[str, Any], _Pipeline],
    ) -> UpdateResult:
        return self.update_many_by_filter(filter, update_operation, True)

    def delete_one_by_filter(
        self, filter: Mapping[str, Any], comment: Optional[Any] = None
    ) -> DeleteResult:
        return self.collection.delete_one(filter=filter, comment=comment)

    def delete_many_by_filter(
        self, filter: Mapping[str, Any], comment: Optional[Any] = None
    ) -> DeleteResult:
        return self.collection.delete_many(filter=filter, comment=comment)

    def get_metadata_and_content(
        self, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        results = []
        for doc in documents:
            results.append(
                {"page_content": doc["page_content"], "metadata": doc["metadata"]}
            )
        return results

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update documents that match the filter or insert new documents.
        """
        for i, text in enumerate(texts):
            embedding = self.embedding_function(text)
            doc = {
                "page_content": text,
                "embedding": embedding,
                "metadata": metadatas[i] if metadatas else {},
            }
            if ids:
                self.update_one_by_filter(
                    filter={"_id": ids[i]}, update_operation={"$set": doc}, upsert=True
                )
            else:
                self._insert_one(doc)

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> None:

        def upsert_batch(batch, batch_ids):
            requests = []
            for i, doc in enumerate(batch):
                if batch_ids and i < len(batch_ids):
                    requests.append(
                        self.update_one_by_filter(
                            filter={"_id": batch_ids[i]},
                            update_operation={"$set": doc},
                            upsert=True,
                        )
                    )
                else:
                    self._insert_one(doc)
            if requests:
                self.collection.bulk_write(requests)

        def get_embeddings_parallel(texts_batch: List[str]) -> List[Any]:
            embeddings = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.embedding_function, text)
                    for text in texts_batch
                ]
                for future in as_completed(futures):
                    embeddings.append(future.result())
            return embeddings

        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(0, len(texts), batch_size):
                texts_batch = texts[i : i + batch_size]
                metadatas_batch = metadatas[i : i + batch_size] if metadatas else []
                ids_batch = ids[i : i + batch_size] if ids else None

                embeddings = get_embeddings_parallel(texts_batch)
                batch_docs = [
                    {
                        "page_content": text,
                        "embedding": embeddings[j],
                        "metadata": metadatas_batch[j] if metadatas_batch else {},
                    }
                    for j, text in enumerate(texts_batch)
                ]

                future = executor.submit(
                    upsert_batch,
                    batch_docs,
                    ids_batch,
                )
                futures.append(future)

            for future in as_completed(futures):
                future.result()

    def search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        query_vector = self.embedding_function(query)
        vector_index = kwargs.get("vector_index")
        pipeline = [
            {
                "$vectorSearch": {
                    "index": vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": k * 5,
                    "limit": k,
                }
            }
        ]
        return list(self.collection.aggregate(pipeline))

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        if ids:
            self.delete_many_by_filter(filter={"_id": {"$in": ids}})
        elif filters:
            self.delete_many_by_filter(filter=filters)
        else:
            self.delete_many_by_filter(filter={})
