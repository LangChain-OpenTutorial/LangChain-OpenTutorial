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
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.base import TextSplitter
from utils.base import DocumentManager


class MongoDBAtlasCollectionManager:
    """Manages MongoDB collections and vector store.
    Provides methods to add, update, delete indexes and manage documents in the vector store.
    """

    def __init__(self, db_name: str, client=None):
        """Initialize a MongoDB client and configures the database.

        Args:
            db_name (str): The name of the database to connect to.
            collection_name (str): The name of the collection to use.
        """
        if client is None:
            MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
            self.client = MongoClient(
                MONGODB_ATLAS_CLUSTER_URI, tlsCAFile=certifi.where()
            )
        else:
            self.client = client
        self.database = self.client[db_name]
        self.collection = None

    def create_collection(self, collection_name: str) -> Collection[_DocumentType]:
        """Create a collection."""
        self.collection_name = collection_name
        collection_names = self.database.list_collection_names()
        if self.collection_name not in collection_names:
            self.collection = self.database.create_collection(self.collection_name)
        else:
            self.collection = self.database[self.collection_name]

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


class MongoDBAtlasDocumentManager(DocumentManager):
    """A document manager that handles document processing and CRUD operations in MongoDB Atlas."""

    def __init__(self, client, db_name, collection_name, embedding) -> None:
        """Initialize connection to the database and retrieve the embedding function.

        Args:
            atlas (MongoDBAtlas): A MongoDBAtlas instance to use MongoDB client function.
        """
        self.database = client[db_name]
        self.collection_name = collection_name
        self.collection = self.database[self.collection_name]
        self.embedding_function = embedding

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

        if metadatas is None:
            metadatas = [{}]*len(texts)
        for i, text in enumerate(texts):
            embeded = self.embedding_function.embed_documents([text])
            doc = {
                "page_content": text,
                "embedding": embeded,
                **metadatas[i]
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
        """Upsert on a collection by batching text data and updating or inserting documents.

        Args:
        texts (Iterable[str]): A collection of text data to be upserted.
        metadatas (Optional[list[dict]]): List of data corresponding each text.
        ids (Optional[List[str]]): List of unique document IDs.
            If provided, existing documents with matching IDs will be updated; otherwise, new documents are inserted.
        batch_size (int): The number of documents per batch.
        workers (int): The number of parallel threads.
        """

        def upsert_batch(batch, batch_ids):
            """Upsert documents in parallel."""
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
            """Uses multithreading to generate embeddings for the text data."""
            embeddings = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.embedding_function.embed_documents, [text])
                    for text in texts_batch
                ]
                for future in as_completed(futures):
                    embeddings.append(future.result())
            return embeddings

        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(0, len(texts), batch_size):
                texts_batch = texts[i : i + batch_size]
                metadatas_batch = metadatas[i : i + batch_size] if metadatas else [{}]*len(batch_size)
                ids_batch = ids[i : i + batch_size] if ids else None

                embeddings = get_embeddings_parallel(texts_batch)
                batch_docs = [
                    {
                        "page_content": text,
                        "embedding": embeddings[j],
                        **metadatas_batch[j],
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
                continue

    def search(self, query: str, k: int = 4, filters=None, **kwargs: Any) -> List[Document]:
        """Retrieve the top `k` most relevant documents.
        Converts the input query into an embedding using `embedding_function`.

        Args:
            query (str): The input query string to search for.
            k (int): The number of top results to retrieve.
            **kwargs (Any):
                - vector_index (str): The name of the vector index to use for the search.

        Returns:
            List[Document]: A list of documents that best match the query.
        """
        query_vector = self.embedding_function.embed_query(query)
        vector_index = kwargs.get("vector_index")
        vector_search_args = {
                "$vectorSearch": {
                    "index": vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": k * 5,
                    "limit": k,
                }
            }
        if filters is not None:
            vector_search_args['$vectorSearch']["filter"] = filters
        pipeline = [
            vector_search_args,
            {"$set": {"score": {"$meta": "vectorSearchScore"}}}
        ]
        return list(self.collection.aggregate(pipeline))

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Delete documents from the collection.
        If neither `ids` nor `filters` are provided, all documents in the collection will be deleted.

        Args:
            ids (Optional[list[str]]): A list of document IDs to delete. If provided,
                deletes all documents matching these IDs.
            filters (Optional[dict]): If provided and `ids` is None, deletes documents matching the filter.
        """
        if ids:
            self.delete_many_by_filter(filter={"_id": {"$in": ids}})
        elif filters:
            self.delete_many_by_filter(filter=filters)
        else:
            self.delete_many_by_filter(filter={})
