from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.documents import Document


# ==========================================
# 1️⃣ 인덱스 관리 인터페이스
# ==========================================
class IndexManagerInterface(ABC):
    """
    인덱스 관리 인터페이스
    """
    @abstractmethod
    def create_index(self, index_name: str, dimension: int, metric: str = "dotproduct", pod_spec=None, **kwargs) -> Any:
        """ 인덱스를 생성하고 반환합니다. 즉, index_name으로 인덱스를 생성하고 생성이 완료되면 index_name을 반환하고, 없다면 None을 반환합니다. """
        pass

    @abstractmethod
    def list_indexs(self) -> Any:
        """ 인덱스 리스트를 반환합니다"""
        pass
        
    @abstractmethod
    def get_index(self, index_name: str) -> Any:
        """ 인덱스를 조회합니다. 즉, index_name을 가진 인덱스가 있는지 조회하고, 있다면 index_name을 반환하고, 없다면 None을 반환합니다. """
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> None:
        """ 인덱스를 삭제합니다. 즉, index_name을 가진 인덱스를 삭제합니다. """
        pass


# ==========================================
# 2️⃣ 문서 업서트 인터페이스
# ==========================================
class DocumentManagerInterface(ABC):
    """
    문서 관리 인터페이스 (upsert, upsert_parallel)
    """

    @abstractmethod
    def upsert_documents(self, index_name: str, documents: List[Dict], **kwargs) -> None:
        """문서를 업서트합니다."""
        pass

    @abstractmethod
    def upsert_documents_parallel(self, index_name: str, documents: List[Dict], batch_size: int = 32, max_workers: int = 10, **kwargs) -> None:
        """병렬로 문서를 업서트합니다."""
        pass


# ==========================================
# 3️⃣ 문서 조회 및 삭제 인터페이스
# ==========================================
class QueryManagerInterface(ABC):
    """
    문서 검색 및 삭제 인터페이스 (query, delete_by_filter)
    """

    @abstractmethod
    def query(self, index_name: str, query_vector: List[float], top_k: int = 10, **kwargs) -> List[Document]:
        """쿼리를 수행하고 관련 문서를 반환합니다."""
        pass

    @abstractmethod
    def delete_by_filter(self, index_name: str, filters: Dict, **kwargs) -> None:
        """필터를 사용하여 문서를 삭제합니다."""
        pass


# ==========================================
# 4️⃣ 통합 인터페이스 (VectorDBInterface)
# ==========================================
class VectorDBInterface(IndexManagerInterface, DocumentManagerInterface, QueryManagerInterface, ABC):
    """
    벡터 데이터베이스의 통합 인터페이스
    - 인덱스 관리
    - 문서 업서트
    - 문서 검색 및 삭제
    """

    @abstractmethod
    def connect(self, **kwargs) -> None:
        """DB 연결을 초기화합니다."""
        pass

    @abstractmethod
    def preprocess_documents(self, documents: List[Document], **kwargs) -> List[Dict]:
        """LangChain Document 객체를 특정 DB에 맞는 형식으로 변환합니다."""
        pass

    @abstractmethod
    def get_api_key(self) -> str:
        """DB 연결을 위한 API 키 또는 인증 정보 반환"""
        pass



# ==========================================
# pinecone 구현 예시
# ==========================================
from pinecone import ServerlessSpec, Pinecone

class PineconeDB(VectorDBInterface):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pc = None

    def connect(self, **kwargs) -> None:
        """Pinecone API 연결"""
        self.pc = Pinecone(api_key=self.api_key)

    def create_index(self, index_name: str, dimension: int, metric: str = "dotproduct", **kwargs) -> Any:
        """Pinecone 인덱스 생성"""
        pod_spec = ServerlessSpec(cloud="aws", region="us-east-1")
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=pod_spec)

    def get_index(self, index_name: str) -> Any:
        """Pinecone 특정 인덱스 조회"""
        return self.pc.Index(index_name)

    def list_indexes(self) -> List[str]:
        """Pinecone에서 현재 존재하는 모든 인덱스를 조회"""
        return self.pc.list_indexes().names()

    def delete_index(self, index_name: str) -> None:
        """Pinecone 인덱스 삭제"""
        self.pc.delete_index(index_name)

    def upsert_documents(self, index_name: str, documents: List[Dict], **kwargs) -> None:
        """문서를 Pinecone에 업서트"""
        index = self.pc.Index(index_name)
        index.upsert(vectors=documents, namespace=kwargs.get("namespace"))
