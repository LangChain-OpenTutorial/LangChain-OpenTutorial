from abc import ABC, abstractmethod
from typing import Any, List, Dict

class VectorDB(ABC):
    """Abstract class: Interface for interacting with a vector database."""

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """Connects to the DB."""
        pass

    @abstractmethod
    def create_index(self, **kwargs) -> None:
        """Creates a new index (schema), or return existing index""" # DB에 따라 작업시 해당 인덱스에 연결하여 정보를 불러올 필요가 있는 DB들이 있을 것 같습니다.
        pass

    @abstractmethod
    def delete_index(self, index_name: str, **kwargs) -> None:
        """Deletes a specific index."""
        pass

    @abstractmethod
    def list_indices(self, **kwargs) -> List[str]:
        """Lists all indices."""
        pass

    @abstractmethod
    def get_index(self, index_name: str, **kwargs) -> Dict:
        """Get index information"""
        pass

    @abstractmethod
    def insert_documents(self, docs: List[Dict[str, Any]], **kwargs) -> str:
        """Inserts data and a vector."""
        pass

    @abstractmethod
    def update_documents(self, docs: List[Dict[str, Any]], **kwargs) -> bool:
        """Updates existing data."""
        pass

    @abstractmethod
    def replace_documents(self, docs: List[Dict[str, Any]], **kwargs) -> bool:
        """Completely replaces existing data."""
        pass

    @abstractmethod
    def upsert_documents(self, docs: List[Dict[str, Any]], **kwargs) -> str:
        """Inserts or updates data."""
        pass
    
    @abstractmethod
    def upsert_documents_parallel(self, docs: List[Dict[str, Any]]) -> str:
        """Inserts or updates data in parallel."""
        pass

    @abstractmethod
    def delete_documents_by_filter(self, filter_query: Any) -> bool:
        """Deletes data by filter."""
        pass

    @abstractmethod
    def delete_documents_by_ids(self, ids: List[str]) -> bool:
        "Deletes data by id"
        pass

    @abstractmethod
    def delete_documents_by_query(self, query:str) -> bool:
        "Delete data by query"
        pass

    @abstractmethod
    def delete_documents(self, filter_query:Any, ids: List[str], query: str) -> bool:
        """Delete data by filter, id or query"""
        pass

    @abstractmethod
    def scroll(self, index_name: str, filter: Any, ids: List[str], query: str, **kwargs) -> List[Any]:
        """Get data by filter, id, or query"""
        pass

    @abstractmethod
    def scroll_by_id(self, index_name: str, ids: List[str], **kwargs) -> List[Any]:
        """Get data by id"""
        pass

    @abstractmethod
    def scroll_by_filter(self, index_name: str, **kwargs) -> List[Any]:
        """Get data by filter"""
        pass

    @abstractmethod
    def scroll_by_query(self, index_name: str, query: str, **kwargs) -> List[Any]:
        """Get data by query"""
        pass

    @abstractmethod
    def similarity_search(self, index_name: str, query: str, top_k: int, **kwargs) -> List[Any]:
        """Get data based on similarity score"""
        pass