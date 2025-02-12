import os
import certifi
from pathlib import Path
from typing import List, Iterable, Tuple, Optional, Any, Mapping, Union, Dict, Callable
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
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.base import TextSplitter
from utils.vectordbinterface import DocumentManager


class MongoDBAtlas:

    def __init__(self, db_name: str, collection_name: str, embedding=None):
        MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI, tlsCAFile=certifi.where())
        self.database = client[db_name]
        self.collection_name = collection_name
        self.embedding = embedding
        self.collection = None
        self.vector_store = None

    def connect(self) -> Collection[_DocumentType]:
        collection_names = self.database.list_collection_names()
        if self.collection_name not in collection_names:
            self.collection = self.database.create_collection(self.collection_name)
        else:
            self.collection = self.database[self.collection_name]
        return self.collection

    # ==========================================
    # TODO: Index
    # ==========================================
    def _is_index_exists(self, index_name: str):
        search_indexes = self.collection.list_search_indexes()
        index_names = [search_index["name"] for search_index in search_indexes]
        return index_name in index_names

    def create_index(
        self, index_name: str, model: Union[Mapping[str, Any], SearchIndexModel]
    ):
        if not self._is_index_exists(index_name):
            self.collection.create_search_index(model)

    def update_index(self, index_name: str, definition: Mapping[str, Any]):
        if self._is_index_exists(index_name):
            self.collection.update_search_index(name=index_name, definition=definition)

    def delete_index(self, index_name: str):
        if self._is_index_exists(index_name):
            self.collection.drop_search_index(index_name)

    # ==========================================
    # TODO: langchain_mongodb
    # ==========================================

    def create_vector_store(self, index_name: str, relevance_score_fn: str):
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embedding,
            index_name=index_name,
            relevance_score_fn=relevance_score_fn,
        )

    def create_vector_search_index(
        self,
        index_name: str,
        dimensions: int,
        filters: Optional[List[str]] = None,
        update: bool = False,
    ) -> None:
        if not self._is_index_exists(index_name):
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
        split_documents = []
        for document in documents:
            split_documents.extend(
                self._split_texts(
                    document.page_content, split_condition, split_index_name
                )
            )
        return split_documents

    def split_documents_by_splitter(
        self, splitter: TextSplitter, documents: Iterable[Document]
    ) -> List[Document]:
        return splitter.split_documents(documents)

    def _split_texts(
        self,
        texts: str,
        split_condition: Callable[[str], Iterable[str]],
        split_index_name: str,
    ) -> Iterable[Document]:
        documents = split_condition(texts)
        for index, document in enumerate(documents):
            yield Document(page_content=document, metadata={split_index_name: index})

    def convert_document_to_raw_bson(
        self,
        document: Document,
    ) -> RawBSONDocument:
        document_dict = {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        return RawBSONDocument(encode(document_dict))

    def convert_documents_to_raw_bson(
        self,
        documents: List[Document],
    ) -> Iterable[RawBSONDocument]:
        for document in documents:
            yield self.convert_document_to_raw_bson(document)

    def insert_one(self, document: Document) -> InsertOneResult:
        bson_document = self.convert_document_to_raw_bson(document)
        return self.collection.insert_one(bson_document)

    def insert_many(self, documents: List[Document]) -> InsertManyResult:
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

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> None:
        pass

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        return super().search(query, k, **kwargs)

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        pass
