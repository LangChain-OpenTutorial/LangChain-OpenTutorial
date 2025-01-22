import os
import certifi
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from typing import List, Iterable, Optional, Any, Dict, Mapping
from bson import encode
from bson.raw_bson import RawBSONDocument
from langchain_core.documents import Document


class MongoDBAtlas:

    # ==========================================
    # TODO: setup
    # ==========================================

    def __init__(self, embedding=None):
        self.embedding = embedding
        self.collection = None
        self.vector_store = None

    def connect(self, db_name, collection_name):
        MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI, tlsCAFile=certifi.where())
        database = client[db_name]
        collection_names = database.list_collection_names()
        if collection_name not in collection_names:
            self.collection = database.create_collection(collection_name)
        else:
            self.collection = database[collection_name]

    # ==========================================
    # TODO: Index
    # ==========================================
    def is_index_exists(self, index_name):
        search_indexes = self.collection.list_search_indexes()
        index_names = [search_index["name"] for search_index in search_indexes]
        return index_name in index_names

    def create_index(self, index_name, model):
        if not self.is_index_exists(self.collection, index_name):
            self.collection.create_search_index(model)

    def update_index(self, index_name, definition):
        if self.is_index_exists(self.collection, index_name):
            self.collection.update_search_index(name=index_name, definition=definition)

    def delete_index(self, index_name):
        if self.is_index_exists(self.collection, index_name):
            self.collection.drop_search_index(index_name)

    # ==========================================
    # TODO: langchain_mongodb
    # ==========================================

    def create_vector_store(self, index_name, relevance_score_fn):
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embedding,
            index_name=index_name,
            relevance_score_fn=relevance_score_fn,
        )

    def create_vector_search_index(self, index_name, dimensions, filters, update=False):
        if not self.is_index_exists(self.collection, index_name):
            self.vector_store.create_vector_search_index(
                dimensions=dimensions, filters=filters, update=update
            )

    def update_vector_search_index(self, index_name, dimensions, filters):
        self.create_vector_search_index(self, index_name, dimensions, filters, True)

    def add_documents(self, documents) -> List[str]:
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
        self.vector_store.similarity_search(
            query=query, k=k, pre_filter=pre_filter, **kwargs
        )

    # ==========================================
    # TODO: pymongo
    # ==========================================

    def convert_document_to_raw_bson(
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

    def insert_one(self, document: Document):
        bson_document = self.convert_document_to_raw_bson(document)
        self.collection.insert_one(bson_document)

    def insert_many(self, documents: List[Document]):
        bson_documents = self.convert_documents_to_raw_bson(documents)
        self.collection.insert_many(bson_documents)

    def find_one_by_filter(self, filter) -> Mapping[str, Any]:
        return self.collection.find_one(filter=filter)

    def find_all_by_filter(self, filter) -> List[Mapping[str, Any]]:
        cursor = self.collection.find(filter=filter)
        documents = []
        for doc in cursor:
            documents.append(doc)
        return documents

    def update_one_by_filter(self, filter, update_operation, upsert=False):
        self.collection.update_one(filter, update_operation, upsert)

    def update_many_by_filter(self, filter, update_operation, upsert=False):
        self.collection.update_many(filter, update_operation, upsert)

    def upsert_one_by_filter(self, filter, update_operation):
        self.update_one_by_filter(filter, update_operation, True)

    def upsert_many_by_filter(self, filter, update_operation):
        self.update_many_by_filter(filter, update_operation, True)

    def delete_one_by_filter(self, filter, comment):
        self.collection.delete_one(filter, comment)

    def delete_many_by_filter(self, filter, comment):
        self.collection.delete_many(filter, comment)
