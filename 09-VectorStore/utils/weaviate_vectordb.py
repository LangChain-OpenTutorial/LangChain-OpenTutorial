import datetime
from langchain_weaviate import WeaviateVectorStore
import weaviate
import logging
from tqdm import tqdm
from weaviate.classes.init import Auth
from weaviate.collections.classes.filters import Filter
from weaviate.classes.config import Configure, VectorDistances
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, Tuple, Iterable
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from utils.vectordbinterface import DocumentManager
from langchain_core.embeddings import Embeddings
from weaviate.classes.config import Property

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WeaviateDB(DocumentManager):
    def __init__(
        self,
        api_key: str,
        url: str,
        openai_api_key: str = None,
        embeddings: Embeddings = None,
    ):
        self._api_key = api_key
        self._url = url
        self._client = None
        self._openai_api_key = openai_api_key
        self._embeddings = embeddings

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embeddings

    def _create_filter_query(self, filters: Optional[dict] = None) -> Optional[dict]:
        """
        filters 파라미터가 존재할 경우, Weaviate where 조건에 맞게 변환하여 반환합니다.
        예시: {"source": "예시1", "category": "news"} 인 경우 And 조건으로 변환.

        Returns:
            dict: Weaviate의 where 조건 형식, 또는 None
        """
        if not filters:
            return None

        # 각 조건을 생성 (단일 필드에 대해 Equal 연산자를 사용)
        conditions = []
        for key, value in filters.items():
            condition = {
                "path": [key],
                "operator": "Equal",
                "valueString": value if isinstance(value, str) else str(value),
            }
            conditions.append(condition)

        # 조건이 한 개라면 단일 조건 반환, 여러 개라면 And 연산자 사용
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"operator": "And", "operands": conditions}

    def connect(
        self,
        **kwargs: Any,
    ) -> weaviate.Client:
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Could not import weaviate python package. "
                "Please install it with `pip install weaviate-client`"
            )

        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self._url,
            auth_credentials=Auth.api_key(self._api_key),
            headers={"X-OpenAI-Api-Key": self._openai_api_key},
            **kwargs,
        )
        return self._client

    def get_api_key(self):
        """API 키 반환"""
        return self._api_key

    def _json_serializable(self, value: Any) -> Any:
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        return value

    def create_collection(
        self,
        client: weaviate.Client,
        collection_name: str,
        description: str,
        properties: List[Property],
        vectorizer: Configure.Vectorizer,
        metric: str = "cosine",
    ) -> None:
        """
        Creates a new index (collection) in Weaviate with the specified properties.

        :param client: Weaviate client instance
        :param collection_name: Name of the index (collection) (e.g., "BookChunk")
        :param description: Description of the index (e.g., "A collection for storing book chunks")
        :param properties: List of properties, where each property is a dictionary with keys:
            - name (str): Name of the property
            - dataType (list[str]): Data types for the property (e.g., ["text"], ["int"])
            - description (str): Description of the property
        :param vectorizer: Vectorizer configuration created using Configure.Vectorizer
                          (e.g., Configure.Vectorizer.text2vec_openai())
        :return: None
        """
        distance_metric = getattr(VectorDistances, metric.upper(), None)

        # Set vector_index_config to hnsw
        vector_index_config = Configure.VectorIndex.hnsw(
            distance_metric=distance_metric
        )

        # Create the collection in Weaviate
        try:
            client.collections.create(
                name=collection_name,
                description=description,
                properties=properties,
                vectorizer_config=vectorizer,
                vector_index_config=vector_index_config,
            )
            print(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            print(f"Failed to create collection '{collection_name}': {e}")

    def delete_collection(self, client, collection_name):
        client.collections.delete(collection_name)
        print(f"Deleted index: {collection_name}")

    def delete_all_collections(self, client):
        client.collections.delete_all()
        print("Deleted all collections")

    def list_collections(self, client):
        """
        Lists all collections (indexes) in the Weaviate database, including their properties.
        """
        # Retrieve all collection configurations
        collections = client.collections.list_all()

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

    def lookup_collection(self, collection_name: str):
        return self._client.collections.get(collection_name)

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Upsert objects into Weaviate.

        Args:
            index_name: Collection name
            data_objects: Data objects to upsert
            unique_key: Unique key
            show_progress: Whether to show progress

        Returns:
            UUID list of successfully processed objects
        """
        metadatas = metadatas if metadatas is not None else [{} for _ in texts]
        ids = ids if ids is not None else [str(i) for i in range(len(texts))]

        successful_ids = []
        batch_size = kwargs.get("batch_size", 100)
        show_progress = kwargs.get("show_progress", False)
        collection_name = kwargs.get("collection_name", "default_collection")
        collection = self._client.collections.get(collection_name)
        text_key = kwargs.get("text_key", "text")

        embeddings: Optional[List[List[float]]] = None
        if self._embeddings:
            embeddings = self._embeddings.embed_documents(list(texts))

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size] if metadatas else None

                for j, text in enumerate(batch_texts):
                    data_properties = {text_key: text}
                    data_properties["order"] = j
                    if batch_metadatas:
                        data_properties.update(batch_metadatas[j])

                    try:
                        # 먼저 객체가 존재하는지 확인
                        exists = collection.data.exists(uuid=batch_ids[j])

                        if exists:
                            # 객체가 존재하면 업데이트
                            collection.data.replace(
                                uuid=batch_ids[j],
                                properties=data_properties,
                                vector=batch_embeddings[j],
                            )
                        else:
                            # 객체가 없으면 삽입
                            collection.data.insert(
                                uuid=batch_ids[j],
                                properties=data_properties,
                                vector=batch_embeddings[j],
                            )
                        successful_ids.append(batch_ids[j])

                    except Exception as e:
                        print(f"문서 처리 중 오류 발생 (ID: {batch_ids[j]}): {e}")
                        continue

                if show_progress:
                    print(
                        f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
                    )

        except Exception as e:
            print(f"Error during batch processing: {e}")

        return successful_ids

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        병렬로 문서를 업서트합니다.
        """
        metadatas = metadatas if metadatas is not None else [{} for _ in texts]
        ids = ids if ids is not None else [str(i) for i in range(len(texts))]

        collection_name = kwargs.get("collection_name", "default_collection")
        text_key = kwargs.get("text_key", "text")

        embeddings: Optional[List[List[float]]] = None
        if self._embeddings:
            embeddings = self._embeddings.embed_documents(list(texts))

        with self._client.batch.dynamic() as batch:
            for i, text in enumerate(texts):
                data_properties = {text_key: text}
                data_properties["order"] = i
                if metadatas is not None:
                    for key, val in metadatas[i].items():
                        data_properties[key] = self._json_serializable(val)

                batch.add_object(
                    collection=collection_name,
                    properties=data_properties,
                    uuid=ids[i],
                    vector=embeddings[i] if embeddings else None,
                )
        failed_objs = self._client.batch.failed_objects
        for obj in failed_objs:
            err_message = (
                f"Failed to add object: {obj.original_uuid}\nReason: {obj.message}"
            )

            logger.error(err_message)

        return ids

    def delete(
        self, ids: List[str] = None, filters: Optional[dict] = None, **kwargs: Any
    ) -> bool:
        """
        주어진 ids와 filters 조건을 만족하는 객체들을 삭제합니다.

        Args:
            ids (List[str], optional): 삭제할 객체의 ID 리스트
            filters (Optional[dict]): 추가 필터 조건. 예: {"source": "예시1"}
            **kwargs: 추가 옵션
                - collection_name (str): 컬렉션 이름
                - batch_size (int): 한 번에 삭제할 객체 수 (기본값: 10000)

        Returns:
            bool: 삭제 성공 여부
        """
        collection_name = kwargs.get("collection_name", "default_collection")
        collection = self._client.collections.get(collection_name)

        try:
            if ids and filters:
                # ID와 필터 조건을 모두 적용
                filter_builder = Filter.by_property

                # 필터 조건 변환
                weaviate_filter = None
                for key, value in filters.items():
                    if weaviate_filter is None:
                        weaviate_filter = filter_builder(key).equal(value)
                    else:
                        weaviate_filter = weaviate_filter.and_filter(
                            filter_builder(key).equal(value)
                        )
                # ID 조건 추가
                id_filter = Filter.by_id().in_list(ids)
                if weaviate_filter:
                    weaviate_filter = weaviate_filter.and_filter(id_filter)
                else:
                    weaviate_filter = id_filter

                # 조건을 모두 만족하는 객체 삭제
                collection.data.delete_many(
                    where=weaviate_filter,
                )

            elif ids:
                # ID만으로 삭제
                collection.data.delete_many(uuids=ids)

            elif filters:
                # 필터만으로 삭제
                filter_builder = Filter.by_property
                weaviate_filter = None
                for key, value in filters.items():
                    if weaviate_filter is None:
                        weaviate_filter = filter_builder(key).equal(value)
                    else:
                        weaviate_filter = weaviate_filter.and_filter(
                            filter_builder(key).equal(value)
                        )

                collection.data.delete_many(
                    where=weaviate_filter,
                )

            return True

        except Exception as e:
            print(f"삭제 중 오류 발생: {e}")
            return False

    def search(
        self,
        query: str,
        k: int = 4,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        의미 기반 유사도 검색을 수행합니다.

        Args:
            query (str): 검색할 텍스트 쿼리
            k (int): 반환할 결과 개수 (기본값: 4)
            filters (Optional[dict]): 검색 필터 조건 (예: {"category": "news"})
            **kwargs: 추가 매개변수
                - collection_name (str): 검색할 컬렉션 이름 (기본값: "default_collection")
                - properties (List[str]): 반환받을 속성 목록 (기본값: ["text"])

        Returns:
            List[Document]: 검색 결과 문서 리스트
        """
        collection_name = kwargs.get("collection_name", "default_collection")
        vector = kwargs.get("vector", None)
        collection = self._client.collections.get(collection_name)

        if vector is None:
            vector = self._embeddings.embed_query(query)

        weaviate_filter = None
        if filters:
            filter_builder = Filter.by_property
            for key, value in filters.items():
                if weaviate_filter is None:
                    weaviate_filter = filter_builder(key).equal(value)
                else:
                    weaviate_filter = weaviate_filter.and_filter(
                        filter_builder(key).equal(value)
                    )

        hybrid_kwargs = {"query": query, "vector": vector, "limit": k}

        if weaviate_filter:
            hybrid_kwargs["filters"] = weaviate_filter

        try:
            # near_text 쿼리 실행
            response = collection.query.hybrid(**hybrid_kwargs)

            # 결과를 Document 객체로 변환
            documents = []
            for obj in response.objects:
                # text를 제외한 나머지 속성들은 metadata로 저장
                metadata = {
                    key: value for key, value in obj.properties.items() if key != "text"
                }
                metadata["uuid"] = str(obj.uuid)

                doc = Document(
                    page_content=obj.properties.get("text", str(obj.properties)),
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return []

    def keyword_search(
        self,
        query: str,
        k: int = 4,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        BM25 키워드 기반 검색을 수행합니다.

        Args:
            query (str): 검색할 키워드
            k (int): 반환할 결과 개수 (기본값: 4)
            filters (Optional[dict]): 검색 필터 조건 (예: {"category": "news"})
            **kwargs: 추가 매개변수
                - collection_name (str): 검색할 컬렉션 이름
                - properties (List[str]): 검색할 특정 속성들

        Returns:
            List[Document]: 검색 결과 문서 리스트
        """
        collection_name = kwargs.pop("collection_name", "default_collection")
        collection = self._client.collections.get(collection_name)

        # BM25 검색을 위한 기본 설정
        bm25_kwargs = {"query": query, "limit": k}

        # 필터 변환 및 적용
        if filters:
            filter_builder = Filter.by_property
            weaviate_filter = None
            for key, value in filters.items():
                if weaviate_filter is None:
                    weaviate_filter = filter_builder(key).equal(value)
                else:
                    weaviate_filter = weaviate_filter.and_filter(
                        filter_builder(key).equal(value)
                    )
            bm25_kwargs["filters"] = weaviate_filter

        try:
            # BM25 검색 실행
            response = collection.query.bm25(**bm25_kwargs)

            # 결과를 Document 객체로 변환
            documents = []
            for obj in response.objects:
                metadata = {
                    key: value for key, value in obj.properties.items() if key != "text"
                }
                metadata["uuid"] = str(obj.uuid)

                doc = Document(
                    page_content=obj.properties.get("text", str(obj.properties)),
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return []


class WeaviateSearch:
    def __init__(self, vector_store: WeaviateVectorStore):
        self.vector_store = vector_store
        self.collection = vector_store._client.collections.get(vector_store._index_name)
        self.text_key = vector_store._text_key

    def _format_filter(self, filter_query: Filter) -> str:
        """
        Converts a Filter object to a readable string.

        Args:
            filter_query: Weaviate Filter object

        Returns:
            str: Filter description string
        """
        if not filter_query:
            return "No filter"

        try:
            # Converts the internal structure of the Filter object to a string
            if hasattr(filter_query, "filters"):  # Composite filter (AND/OR)
                operator = "AND" if filter_query.operator == "And" else "OR"
                filter_strs = []
                for f in filter_query.filters:
                    if hasattr(f, "value"):  # Single filter
                        filter_strs.append(
                            f"({f.target} {f.operator.lower()} {f.value})"
                        )
                return f" {operator} ".join(filter_strs)
            elif hasattr(filter_query, "value"):  # Single filter
                return f"{filter_query.target} {filter_query.operator.lower()} {filter_query.value}"
            else:
                return str(filter_query)
        except Exception:
            return "Complex filter"

    def similarity_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        k: int = 3,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform basic similarity search
        """
        documents = self.vector_store.similarity_search(
            query, k=k, filters=filter_query, **kwargs
        )
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        k: int = 3,
        **kwargs: Any,
    ):
        """
        Perform similarity search with score
        """
        documents_and_scores = self.vector_store.similarity_search_with_score(
            query, k=k, filters=filter_query, **kwargs
        )
        return documents_and_scores

    def mmr_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        k: int = 3,
        fetch_k: int = 10,
        **kwargs: Any,
    ):
        """
        Perform MMR algorithm-based diverse search
        """
        documents = self.vector_store.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, filters=filter_query, **kwargs
        )
        return documents

    def hybrid_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        alpha: float = 0.5,
        limit: int = 3,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Hybrid search (keyword + vector search)

        Args:
            query: Text to search
            filter_dict: Filter condition dictionary
            alpha: Weight for keyword and vector search (0: keyword only, 1: vector only)
            limit: Number of documents to return
            return_score: Whether to return similarity score

        Returns:
            List of Documents hybrid search results
        """
        embedding_vector = self.vector_store.embeddings.embed_query(query)
        results = self.collection.query.hybrid(
            query=query,
            vector=embedding_vector,
            alpha=alpha,
            limit=limit,
            filters=filter_query,
            **kwargs,
        )

        documents = []
        for obj in results.objects:
            metadata = {
                key: value
                for key, value in obj.properties.items()
                if key != self.text_key
            }
            metadata["uuid"] = str(obj.uuid)

            if hasattr(obj.metadata, "score"):
                metadata["score"] = obj.metadata.score

            doc = Document(
                page_content=obj.properties.get(self.text_key, str(obj.properties)),
                metadata=metadata,
            )

            documents.append(doc)

        return documents

    def semantic_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        limit: int = 3,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Semantic search (vector-based)
        """
        results = self.collection.query.near_text(
            query=query, limit=limit, filters=filter_query, **kwargs
        )

        documents = []
        for obj in results.objects:
            metadata = {
                key: value
                for key, value in obj.properties.items()
                if key != self.text_key
            }
            metadata["uuid"] = str(obj.uuid)
            documents.append(
                Document(
                    page_content=obj.properties.get(self.text_key, str(obj.properties)),
                    metadata=metadata,
                )
            )

        return documents

    def keyword_search(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        limit: int = 3,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Keyword-based search (BM25)
        """
        results = self.collection.query.bm25(
            query=query, limit=limit, filters=filter_query, **kwargs
        )

        documents = []
        for obj in results.objects:
            metadata = {
                key: value
                for key, value in obj.properties.items()
                if key != self.text_key
            }
            metadata["uuid"] = str(obj.uuid)
            documents.append(
                Document(
                    page_content=obj.properties.get(self.text_key, str(obj.properties)),
                    metadata=metadata,
                )
            )

        return documents

    def create_qa_chain(
        self,
        llm: BaseChatModel = None,
        chain_type: str = "stuff",
        retriever: BaseRetriever = None,
        **kwargs: Any,
    ):
        """
        Create search-QA chain
        """
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            **kwargs,
        )
        return qa_chain

    def print_results(
        self,
        results: Union[List[Document], List[Tuple[Document, float]]],
        search_type: str,
        filter_query: Optional[Filter] = None,
    ) -> None:
        """
        Print search results in a readable format

        Args:
            results: List of Document or (Document, score) tuples
            search_type: Search type (e.g., "Hybrid", "Semantic" etc.)
            filter_dict: Applied filter information
        """
        print(f"\n=== {search_type.upper()} SEARCH RESULTS ===")
        if filter_query:
            print(f"Filter: {self._format_filter(filter_query)}")

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")

            # Separate Document object and score
            if isinstance(result, tuple):
                doc, score = result
                print(f"Score: {score:.4f}")
            else:
                doc = result

            # Print content
            print(f"Content: {doc.page_content}")

            # Print metadata
            if doc.metadata:
                print("\nMetadata:")
                for key, value in doc.metadata.items():
                    if (
                        key != "score" and key != "uuid"
                    ):  # Exclude already printed information
                        print(f"  {key}: {value}")

            print("-" * 50)

    def print_search_comparison(
        self,
        query: str,
        filter_query: Optional[Filter] = None,
        limit: int = 5,
        alpha: float = 0.5,
        fetch_k: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Print comparison of all search methods' results

        Args:
            query: Search query
            filter_dict: Filter condition
            limit: Number of results
            alpha: Weight for hybrid search (0: keyword only, 1: vector only)
            fetch_k: Number of candidate documents for MMR search
            **kwargs: Additional search parameters
        """
        search_methods = [
            # 1. Basic similarity search
            {
                "name": "Similarity Search",
                "method": self.similarity_search,
                "params": {"k": limit},
            },
            # 2. Similarity search with score
            {
                "name": "Similarity Search with Score",
                "method": self.similarity_search_with_score,
                "params": {"k": limit},
            },
            # 3. MMR search
            {
                "name": "MMR Search",
                "method": self.mmr_search,
                "params": {"k": limit, "fetch_k": fetch_k},
            },
            # 4. Hybrid search
            {
                "name": "Hybrid Search",
                "method": self.hybrid_search,
                "params": {"limit": limit, "alpha": alpha},
            },
            # 5. Semantic search
            {
                "name": "Semantic Search",
                "method": self.semantic_search,
                "params": {"limit": limit},
            },
            # 6. Keyword search
            {
                "name": "Keyword Search",
                "method": self.keyword_search,
                "params": {"limit": limit},
            },
        ]

        print("\n=== SEARCH METHODS COMPARISON ===")
        print(f"Query: {query}")
        if filter_query:
            print(f"Filter: {self._format_filter(filter_query)}")
        print("=" * 50)

        for search_config in search_methods:
            try:
                method_params = {
                    **search_config["params"],
                    "query": query,
                    "filter_query": filter_query,
                    **kwargs,
                }

                results = search_config["method"](**method_params)

                print(f"\n>>> {search_config['name'].upper()} <<<")
                self.print_results(results, search_config["name"], filter_query)

            except Exception as e:
                print(f"\nError in {search_config['name']}: {str(e)}")

            print("\n" + "=" * 50)

    def delete_documents(self, filter_query: Any, ids: List[str], query: str) -> bool:
        """문서 삭제"""
        try:
            if ids:
                self.delete_documents_by_ids(ids)
            elif filter_query:
                self.delete_documents_by_filter(filter_query)
            elif query:
                self.delete_documents_by_query(query)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def delete_documents_by_ids(self, ids: List[str]) -> bool:
        """ID로 문서 삭제"""
        try:
            for doc_id in ids:
                self.collection.data.delete(doc_id)
            return True
        except Exception as e:
            print(f"Error deleting documents by IDs: {e}")
            return False

    def delete_documents_by_filter(self, filter_query: Any) -> bool:
        """필터로 문서 삭제"""
        try:
            self.collection.data.delete_many(filter_query)
            return True
        except Exception as e:
            print(f"Error deleting documents by filter: {e}")
            return False

    def delete_documents_by_query(self, query: str) -> bool:
        """쿼리로 문서 삭제"""
        try:
            results = self.semantic_search(query)
            if results:
                ids = [doc.metadata["uuid"] for doc in results]
                return self.delete_documents_by_ids(ids)
            return True
        except Exception as e:
            print(f"Error deleting documents by query: {e}")
            return False

    def insert_documents(self, documents: List[Dict]) -> bool:
        """문서 삽입"""
        try:
            self.upsert_documents(self._current_index, documents)
            return True
        except Exception as e:
            print(f"Error inserting documents: {e}")
            return False

    def update_documents(self, documents: List[Dict]) -> bool:
        """문서 업데이트"""
        try:
            self.upsert_documents(self._current_index, documents)
            return True
        except Exception as e:
            print(f"Error updating documents: {e}")
            return False

    def replace_documents(self, documents: List[Dict]) -> bool:
        """문서 교체"""
        try:
            self.upsert_documents(self._current_index, documents)
            return True
        except Exception as e:
            print(f"Error replacing documents: {e}")
            return False

    def scroll(
        self,
        index_name: str,
        filter_query: Any = None,
        ids: List[str] = None,
        query: str = None,
        **kwargs,
    ) -> List[Any]:
        """스크롤 검색"""
        if ids:
            return self.scroll_by_id(index_name, ids, **kwargs)
        elif filter_query:
            return self.scroll_by_filter(index_name, filter_query, **kwargs)
        elif query:
            return self.scroll_by_query(index_name, query, **kwargs)
        return []

    def scroll_by_id(self, index_name: str, ids: List[str], **kwargs) -> List[Any]:
        """ID로 스크롤 검색"""
        results = []
        for doc_id in ids:
            try:
                result = self.collection.data.get_by_id(doc_id)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error in scroll_by_id: {e}")
        return results

    def scroll_by_filter(
        self, index_name: str, filter_query: Any, **kwargs
    ) -> List[Any]:
        """필터로 스크롤 검색"""
        try:
            results = self.collection.data.get_many(filter_query)
            return list(results)
        except Exception as e:
            print(f"Error in scroll_by_filter: {e}")
            return []

    def scroll_by_query(self, index_name: str, query: str, **kwargs) -> List[Any]:
        """쿼리로 스크롤 검색"""
        try:
            results = self.semantic_search(query, **kwargs)
            return results
        except Exception as e:
            print(f"Error in scroll_by_query: {e}")
            return []
