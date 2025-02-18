from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from langchain_core.documents import Document

"""
This file is legacy, It will be replaced in the near future.
"""


# ==========================================
# 1️⃣ Index manage Interface
# ==========================================
class IndexManagerInterface(ABC):
    """
    Index manage Interface
    """

    @abstractmethod
    def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str = "dotproduct",
        pod_spec=None,
        **kwargs
    ) -> Any:
        """Create Index and return"""
        pass

    @abstractmethod
    def list_indexs(self) -> Any:
        """Return Index list"""
        pass

    @abstractmethod
    def get_index(self, index_name: str) -> Any:
        """Get Index"""
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> None:
        """Delete Index"""
        pass


# ==========================================
# 2️⃣ Document Upsert
# ==========================================
class DocumentManagerInterface(ABC):
    """
    (upsert, upsert_parallel)
    """

    @abstractmethod
    def upsert_documents(
        self, index_name: str, documents: List[Dict], **kwargs
    ) -> None:
        """Upsert Document"""
        pass

    @abstractmethod
    def upsert_documents_parallel(
        self,
        index_name: str,
        documents: List[Dict],
        batch_size: int = 32,
        max_workers: int = 10,
        **kwargs
    ) -> None:
        """Upsert Document Parallel"""
        pass


# ==========================================
# 3️⃣ Document Search and Delete
# ==========================================
class QueryManagerInterface(ABC):
    """
    (query, delete_by_filter)
    """

    @abstractmethod
    def query(
        self, index_name: str, query_vector: List[float], top_k: int = 10, **kwargs
    ) -> List[Document]:
        """Document Query"""
        pass

    @abstractmethod
    def delete_by_filter(self, index_name: str, filters: Dict, **kwargs) -> None:
        """Delete Document by filters"""
        pass


# ==========================================
# 4️⃣ Integration Interface(VectorDBInterface)
# ==========================================
class VectorDBInterface(
    IndexManagerInterface, DocumentManagerInterface, QueryManagerInterface, ABC
):
    """
    Integration Interface for VectorDB
    """

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """DB Connect init"""
        pass

    @abstractmethod
    def preprocess_documents(self, documents: List[Document], **kwargs) -> List[Dict]:
        """LangChain Document to DB Style Object"""
        pass

    @abstractmethod
    def get_api_key(self) -> str:
        """Get API Key for DB Connect"""
        pass
