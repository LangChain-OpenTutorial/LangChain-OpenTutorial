from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Iterable
from langchain_core.documents import Document


class DocumentManager(ABC):
    """
    Document insert/update (upsert, upsert_parallel)
    Document search by query (search)
    Document delete by id, delete by filter (delete)
    """

    @abstractmethod
    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Upsert Document"""
        pass

    @abstractmethod
    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any
    ) -> None:
        """Upsert Document Parallel"""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """Performs a query and returns relevant documents.
        Basic function: query (string) -> return k similar documents

        Meaning to search cosine_similarity **If a problem arises, raise an issue

        -Other features (future extensions)
        metatdata search
        Getting a vector when searching for an image
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        """Use a filter to delete the document.

        IDS: List of IDs to delete. Delete all if none. Default is none.
        Filter: filter (query) dictionary to be applied; no filter will be applied.

        """
        pass


"""
New Interface for VectorDB CRUD
"""

from typing import Optional, List, Iterable, Any, Dict


class DocumentManager(ABC):
    """
    Document insert/update (upsert, upsert_parallel)
    Document search by query (search)
    Document delete by id, delete by filter (delete)
    """

    @abstractmethod
    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Upsert Document"""
        pass

    @abstractmethod
    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any
    ) -> None:
        """Upsert Document Parallel"""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """Performs a query and returns relevant documents.
        Basic function: query (string) -> return k similar documents

        Meaning to search cosine_similarity **If a problem arises, raise an issue

        -Other features (future extensions)
        metatdata search
        Getting a vector when searching for an image
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        **kwargs: Any
    ) -> None:
        """Use a filter to delete the document.

        IDS: List of IDs to delete. Delete all if none. Default is none.
        Filter: filter (query) dictionary to be applied; no filter will be applied.

        """
        pass
