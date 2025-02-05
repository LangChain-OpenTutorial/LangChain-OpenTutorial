import os
import weaviate
from tqdm import tqdm
from utils.base import VectorDB
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


class WeaviateDB(DocumentManager):
    def __init__(self, api_key: str, url: str, **kwargs):
        self._api_key = api_key
        self._url = url
        self._client = None

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
            **kwargs,
        )
        return self._client
    
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
        vector_index_config = Configure.VectorIndex.hnsw(distance_metric=distance_metric)

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

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
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
        embeddings = kwargs.get("embeddings", None)
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size] if metadatas else None

                for j, text in enumerate(batch_texts):
                    properties = {"text": text}
                    if batch_metadatas:
                        properties.update(batch_metadatas[j])

                    try:
                        # 먼저 객체가 존재하는지 확인
                        exists = collection.data.exists(uuid=batch_ids[j])

                        if exists:
                            # 객체가 존재하면 업데이트
                            collection.data.replace(
                                uuid=batch_ids[j],
                                properties=properties,
                                vector=batch_embeddings[j],
                            )
                        else:
                            # 객체가 없으면 삽입
                            collection.data.insert(
                                uuid=batch_ids[j],
                                properties=properties,
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

        successful_ids = []
        collection_name = kwargs.get("collection_name", "default_collection")
        batch_size = kwargs.get("batch_size", 100)
        max_workers = kwargs.get("max_workers", 4)
        show_progress = kwargs.get("show_progress", False)

        def create_batches(data: List, size: int) -> List[List]:
            return [data[i : i + size] for i in range(0, len(data), size)]

        # 데이터를 배치로 나누기
        text_batches = create_batches(list(texts), batch_size)
        metadata_batches = create_batches(metadatas, batch_size)
        id_batches = create_batches(ids, batch_size)

        def process_batch(batch_data: tuple) -> List[str]:
            batch_texts, batch_metadatas, batch_ids = batch_data
            try:
                return self.upsert(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    collection_name=collection_name,
                    batch_size=len(batch_texts),
                    show_progress=False,
                )
            except Exception as e:
                print(f"\nError occurred while processing batch: {e}")
                return []

        # 배치 데이터 준비
        batches = list(zip(text_batches, metadata_batches, id_batches))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            for future in tqdm(
                as_completed(futures),
                total=len(batches),
                desc="Processing batches",
                disable=not show_progress,
            ):
                try:
                    batch_ids = future.result()
                    successful_ids.extend(batch_ids)
                except Exception as e:
                    print(f"\nError occurred while processing batch: {e}")

        return successful_ids

    def delete(
        self, ids: List[str], filters: Optional[dict] = None, **kwargs: Any
    ) -> bool:
        """
        주어진 ids와 filters 조건을 만족하는 객체들을 삭제합니다.

        Args:
            ids (List[str]): 삭제할 객체의 ID 리스트
            filters (Optional[dict]): 추가 필터 조건. 예: {"source": "예시1"}
            **kwargs: 추가 옵션 (예: collection_name)
        """
        collection_name = kwargs.get("collection_name", "default_collection")
        collection = self._client.collections.get(collection_name)

        if not ids and not filters:
            raise ValueError(
                "삭제할 문서를 지정하기 위해서는 ids 또는 filters 중 하나 이상이 필요합니다."
            )

        try:
            # ID로 삭제
            if ids:
                for obj_id in ids:
                    collection.data.delete_by_id(
                        obj_id
                    )  # delete() 대신 delete_by_id() 사용

            # 필터 조건으로 삭제
            if filters:
                where_filter = self._create_filter_query(filters)
                collection.data.delete_many(where=where_filter)

            return True

        except Exception as e:
            print(f"삭제 중 오류 발생: {str(e)}")
            return False

    def search(
        self,
        query: Union[str, List[float]],
        filters: Optional[dict] = None,
        k: int = 10,
        **kwargs: Any,
    ) -> dict:
        """
        쿼리를 수행하고 관련 문서를 반환합니다.

        기본 기능:
          - query (문자열)를 입력받아 비슷한 문서 k개를 반환합니다.
          - underlying 검색은 cosine similarity 기반의 near_text 검색을 수행합니다.
            (단, query가 벡터(리스트 또는 튜플) 형태로 제공되면, near_vector 쿼리로 전환되어
             이미지 검색 등 벡터 검색에 활용할 수 있습니다.)

        추가 기능:
          - metadata search: filters 파라미터를 통해 메타데이터 기반의 필터링을 지원합니다.
          - 이미지 검색: 벡터 입력을 받아 cosine similarity로 이미지를 검색할 수 있습니다.

        Args:
            query (Union[str, List[float]]): 검색어 문자열 또는 벡터(리스트/튜플) 형태로 입력.
            filters (Optional[dict]): 검색 필터 조건을 딕셔너리 형태로 제공.
            k (int): 반환할 문서 수 (기본값: 10).
            **kwargs: 추가 옵션.
                - collection_name: 사용할 컬렉션(클래스) 이름 (기본값: "default_collection")
                - properties: 검색 시 반환할 프로퍼티 목록 (기본값: ["text", "metadata", "unique_key"])
                - show_progress: 진행 상황 표시 여부 (True/False)

        Returns:
            dict: Weaviate의 GraphQL 응답 형식에 따른 검색 결과.
        """
        from typing import Union, List  # 필요 시 임포트

        collection_name = kwargs.get("collection_name", "default_collection")
        collection = self._client.collections.get(collection_name)
        filter_query = self._create_filter_query(filters)

        properties = kwargs.get("properties", ["text", "metadata", "unique_key"])
        query_builder = self._client.query.get(collection_name, properties)

        # 입력 query의 타입에 따라 near_text 또는 near_vector 검색 실행
        if isinstance(query, str):
            query_builder = query_builder.with_near_text({"concepts": [query]})
        elif isinstance(query, (list, tuple)):
            query_builder = query_builder.with_near_vector({"vector": query})
        else:
            raise ValueError("query는 문자열 또는 벡터(리스트/튜플) 형태여야 합니다.")

        if filter_query is not None:
            query_builder = query_builder.with_where(filter_query)

        query_builder = query_builder.with_limit(k)

        if kwargs.get("show_progress", False):
            print("검색을 수행하는 중...")

        results = query_builder.do()
        return results


# class ConnectDB:
#     def __init__(self):
#         self.db_map = {
#             "weaviate": WeaviateDB,
#             # 다른 DB도 추가 가능
#         }

#     def connect(self, db_type, **kwargs):
#         client_creator = self.db_map.get(db_type)
#         if not client_creator:
#             raise ValueError(f"Unsupported database type: {db_type}")
#         return client_creator().connect(**kwargs)


# class WeaviateDB(VectorDB):
#     def __init__(self, api_key: str, url: str):
#         self._client = None
#         self._current_index = None
#         self._collection = None
#         self._text_key = "text"
#         self._api_key = api_key
#         self._url = url

#     def connect(
#         self,
#         **kwargs: Any,
#     ) -> weaviate.Client:
#         try:
#             import weaviate
#         except ImportError:
#             raise ImportError(
#                 "Could not import weaviate python package. "
#                 "Please install it with `pip install weaviate-client`"
#             )

#         self._client = weaviate.connect_to_weaviate_cloud(
#             cluster_url=self._url,
#             auth_credentials=Auth.api_key(self._api_key),
#             **kwargs
#         )
#         return self._client

#     def _format_filter(self, filter_query: Filter) -> str:
#         """
#         Converts a Filter object to a readable string.

#         Args:
#             filter_query: Weaviate Filter object

#         Returns:
#             str: Filter description string
#         """
#         if not filter_query:
#             return "No filter"

#         try:
#             # Converts the internal structure of the Filter object to a string
#             if hasattr(filter_query, "filters"):  # Composite filter (AND/OR)
#                 operator = "AND" if filter_query.operator == "And" else "OR"
#                 filter_strs = []
#                 for f in filter_query.filters:
#                     if hasattr(f, "value"):  # Single filter
#                         filter_strs.append(
#                             f"({f.target} {f.operator.lower()} {f.value})"
#                         )
#                 return f" {operator} ".join(filter_strs)
#             elif hasattr(filter_query, "value"):  # Single filter
#                 return f"{filter_query.target} {filter_query.operator.lower()} {filter_query.value}"
#             else:
#                 return str(filter_query)
#         except Exception:
#             return "Complex filter"

#     def get_api_key(self):
#         """API 키 반환"""
#         return self._api_key

#     def create_index(
#         self,
#         index_name: str,
#         description: Optional[str] = None,
#         metric: str = "cosine",
#         **kwargs,
#     ) -> None:
#         """Weaviate에 스키마(인덱스)를 생성합니다."""
#         distance_metric = getattr(VectorDistances, metric.upper(), None)

#         # Set vector_index_config to hnsw
#         vector_index_config = Configure.VectorIndex.hnsw(
#             distance_metric=distance_metric
#         )

#         # Create the collection in Weaviate
#         try:
#             self._client.collections.create(
#                 name=index_name,
#                 description=description,
#                 vector_index_config=vector_index_config,
#                 **kwargs
#             )
#             print(f"Collection '{index_name}' created successfully.")
#         except Exception as e:
#             print(f"Failed to create collection '{index_name}': {e}")

#     def delete_index(self, index_name: str) -> None:
#         """Weaviate에서 특정 클래스(인덱스)를 삭제합니다."""
#         self._client.collections.delete(name=index_name)
#         return print(f"Deleted index: {index_name}")

#     def delete_all_collections(self) -> None:
#         self._client.collections.delete_all()
#         return print("Deleted all collections")

#     def list_indices(self) -> str:
#         """현재 활성화된 클래스 목록(인덱스)의 상세 정보를 문자열로 반환합니다."""
#         collections = self._client.collections.list_all()
#         result = []

#         if collections:
#             result.append("Collections (indexes) in the Weaviate schema:")
#             for name, config in collections.items():
#                 result.append(f"- Collection name: {name}")
#                 result.append(
#                     f"  Description: {config.description if config.description else 'No description available'}"
#                 )
#                 result.append(f"  Properties:")
#                 for prop in config.properties:
#                     result.append(f"    - Name: {prop.name}, Type: {prop.data_type}")
#                 result.append("")
#         else:
#             result.append("No collections found in the schema.")

#         return "\n".join(result)

#     def get_index(self, index_name: str) -> Any:
#         """특정 인덱스를 조회합니다."""
#         return self._client.collections.get(index_name)

#     def preprocess_documents(
#         self,
#         split_docs: List[Document],
#         metadata: Dict[str, str] = None,
#     ) -> List[Dict[str, Dict[str, object]]]:
#         """
#         Processes a list of pre-split documents into a format suitable for storing in Weaviate.

#         :param split_docs: List of LangChain Document objects (each containing page_content and metadata).
#         :param metadata: Additional metadata to include in each chunk (e.g., title, source).
#         :return: A list of dictionaries, each representing a chunk in the format:
#                 {'properties': {'text': ..., 'order': ..., ...metadata}}
#         """
#         processed_chunks = []

#         # Iterate over Document objects
#         for idx, doc in enumerate(split_docs, start=1):
#             # Extract text from page_content and include metadata
#             chunk_data = {"text": doc.page_content, "order": idx}
#             # Combine with metadata from Document and additional metadata if provided
#             if metadata:
#                 chunk_data.update(metadata)
#             if doc.metadata:
#                 chunk_data.update(doc.metadata)

#             # Format for Weaviate
#             processed_chunks.append(chunk_data)

#         return processed_chunks

#     def upsert_documents(
#         self,
#         index_name: str,
#         data_objects: List[Dict],
#         unique_key: str = "order",
#         show_progress: bool = False,
#     ) -> List[str]:
#         """
#         Upsert objects into Weaviate.

#         Args:
#             index_name: Collection name
#             data_objects: Data objects to upsert
#             unique_key: Unique key
#             show_progress: Whether to show progress

#         Returns:
#             UUID list of successfully processed objects
#         """
#         collection = self._client.collections.get(index_name)
#         successful_ids = []

#         iterator = tqdm(
#             data_objects, desc="Document processing", disable=not show_progress
#         )
#         for data_object in iterator:
#             try:
#                 unique_value = str(data_object[unique_key])
#                 object_uuid = weaviate.util.generate_uuid5(index_name, unique_value)

#                 if collection.data.exists(object_uuid):
#                     collection.data.replace(
#                         uuid=object_uuid,
#                         properties=data_object,
#                     )
#                 else:
#                     collection.data.insert(
#                         uuid=object_uuid,
#                         properties=data_object,
#                     )
#                 successful_ids.append(object_uuid)
#             except Exception as e:
#                 print(f"\nError occurred while processing object: {e}")

#         return successful_ids

#     def upsert_documents_parallel(
#         self,
#         index_name: str,
#         data_objects: List[Dict],
#         unique_key: str = "order",
#         batch_size: int = 100,
#         max_workers: Optional[int] = 4,
#         show_progress: bool = False,
#     ) -> List[str]:
#         """
#         Utilizes ThreadPoolExecutor to upsert objects in parallel.

#         Args:
#             index_name: Collection name
#             data_objects: Data objects to upsert
#             unique_key: Unique key
#             batch_size: Batch size
#             max_workers: Maximum number of workers
#             show_progress: Whether to show progress

#         Returns:
#             List of UUIDs of successfully processed objects
#         """
#         def create_batches(data: List, size: int) -> List[List]:
#             return [data[i:i + size] for i in range(0, len(data), size)]

#         def process_batch(batch: List[Dict]) -> List[str]:
#             try:
#                 return self.upsert_documents(
#                     index_name=index_name,
#                     data_objects=batch,
#                     unique_key=unique_key,
#                     show_progress=False,
#                 )
#             except Exception as e:
#                 print(f"\nError occurred while processing batch: {e}")
#                 return []

#         all_successful_ids = []
#         batches = create_batches(data_objects, batch_size)

#         print(f"Total documents: {len(data_objects)}, Number of batches: {len(batches)}")
#         print(f"Batch size: {batch_size}, Number of threads: {max_workers}")

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [executor.submit(process_batch, batch) for batch in batches]

#             for future in tqdm(
#                 as_completed(futures),
#                 total=len(futures),
#                 desc="Processing batches",
#                 disable=not show_progress
#             ):
#                 try:
#                     batch_ids = future.result()
#                     all_successful_ids.extend(batch_ids)
#                 except Exception as e:
#                     print(f"\nError occurred while processing batch: {e}")

#         return all_successful_ids

#     def similarity_search(
#         self,
#         index_name: str,
#         query: str,
#         filter_query: Optional[Filter] = None,
#         top_k: int = 3,
#         **kwargs: Any,
#     ):
#         """
#         Perform basic similarity search
#         """
#         documents = self.vector_store.similarity_search(
#             query, k=top_k, filters=filter_query, **kwargs
#         )
#         return documents

#     def similarity_search_with_score(
#         self,
#         query: str,
#         filter_query: Optional[Filter] = None,
#         k: int = 3,
#         **kwargs: Any,
#     ):
#         """
#         Perform similarity search with score
#         """
#         documents_and_scores = self.vector_store.similarity_search_with_score(
#             query, k=k, filters=filter_query, **kwargs
#         )
#         return documents_and_scores

#     def mmr_search(
#         self,
#         query: str,
#         filter_query: Optional[Filter] = None,
#         k: int = 3,
#         fetch_k: int = 10,
#         **kwargs: Any,
#     ):
#         """
#         Perform MMR algorithm-based diverse search
#         """
#         documents = self.vector_store.max_marginal_relevance_search(
#             query=query, k=k, fetch_k=fetch_k, filters=filter_query, **kwargs
#         )
#         return documents

#     def hybrid_search(
#         self,
#         query: str,
#         filter_query: Optional[Filter] = None,
#         alpha: float = 0.5,
#         limit: int = 3,
#         **kwargs: Any,
#     ) -> List[Document]:
#         """
#         Hybrid search (keyword + vector search)

#         Args:
#             query: Text to search
#             filter_dict: Filter condition dictionary
#             alpha: Weight for keyword and vector search (0: keyword only, 1: vector only)
#             limit: Number of documents to return
#             return_score: Whether to return similarity score

#         Returns:
#             List of Documents hybrid search results
#         """
#         embedding_vector = self.vector_store.embeddings.embed_query(query)
#         results = self.collection.query.hybrid(
#             query=query,
#             vector=embedding_vector,
#             alpha=alpha,
#             limit=limit,
#             filters=filter_query,
#             **kwargs,
#         )

#         documents = []
#         for obj in results.objects:
#             metadata = {
#                 key: value
#                 for key, value in obj.properties.items()
#                 if key != self.text_key
#             }
#             metadata["uuid"] = str(obj.uuid)

#             if hasattr(obj.metadata, "score"):
#                 metadata["score"] = obj.metadata.score

#             doc = Document(
#                 page_content=obj.properties.get(self.text_key, str(obj.properties)),
#                 metadata=metadata,
#             )

#             documents.append(doc)

#         return documents

#     def semantic_search(
#         self,
#         query: str,
#         filter_query: Optional[Filter] = None,
#         limit: int = 3,
#         **kwargs: Any,
#     ) -> List[Dict]:
#         """
#         Semantic search (vector-based)
#         """
#         results = self.collection.query.near_text(
#             query=query, limit=limit, filters=filter_query, **kwargs
#         )

#         documents = []
#         for obj in results.objects:
#             metadata = {
#                 key: value
#                 for key, value in obj.properties.items()
#                 if key != self.text_key
#             }
#             metadata["uuid"] = str(obj.uuid)
#             documents.append(
#                 Document(
#                     page_content=obj.properties.get(self.text_key, str(obj.properties)),
#                     metadata=metadata,
#                 )
#             )

#         return documents

#     def keyword_search(
#         self,
#         query: str,
#         filter_query: Optional[Filter] = None,
#         limit: int = 3,
#         **kwargs: Any,
#     ) -> List[Dict]:
#         """
#         Keyword-based search (BM25)
#         """
#         results = self.collection.query.bm25(
#             query=query, limit=limit, filters=filter_query, **kwargs
#         )

#         documents = []
#         for obj in results.objects:
#             metadata = {
#                 key: value
#                 for key, value in obj.properties.items()
#                 if key != self.text_key
#             }
#             metadata["uuid"] = str(obj.uuid)
#             documents.append(
#                 Document(
#                     page_content=obj.properties.get(self.text_key, str(obj.properties)),
#                     metadata=metadata,
#                 )
#             )

#         return documents

#     def create_qa_chain(
#         self,
#         llm: BaseChatModel = None,
#         chain_type: str = "stuff",
#         retriever: BaseRetriever = None,
#         **kwargs: Any,
#     ):
#         """
#         Create search-QA chain
#         """
#         qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
#             llm=llm,
#             chain_type=chain_type,
#             retriever=retriever,
#             **kwargs,
#         )
#         return qa_chain

#     def print_results(
#         self,
#         results: Union[List[Document], List[Tuple[Document, float]]],
#         search_type: str,
#         filter_query: Optional[Filter] = None,
#     ) -> None:
#         """
#         Print search results in a readable format

#         Args:
#             results: List of Document or (Document, score) tuples
#             search_type: Search type (e.g., "Hybrid", "Semantic" etc.)
#             filter_dict: Applied filter information
#         """
#         print(f"\n=== {search_type.upper()} SEARCH RESULTS ===")
#         if filter_query:
#             print(f"Filter: {self._format_filter(filter_query)}")

#         for i, result in enumerate(results, 1):
#             print(f"\nResult {i}:")

#             # Separate Document object and score
#             if isinstance(result, tuple):
#                 doc, score = result
#                 print(f"Score: {score:.4f}")
#             else:
#                 doc = result

#             # Print content
#             print(f"Content: {doc.page_content}")

#             # Print metadata
#             if doc.metadata:
#                 print("\nMetadata:")
#                 for key, value in doc.metadata.items():
#                     if (
#                         key != "score" and key != "uuid"
#                     ):  # Exclude already printed information
#                         print(f"  {key}: {value}")

#             print("-" * 50)

#     def print_search_comparison(
#         self,
#         query: str,
#         filter_query: Optional[Filter] = None,
#         limit: int = 5,
#         alpha: float = 0.5,
#         fetch_k: int = 10,
#         **kwargs: Any,
#     ) -> None:
#         """
#         Print comparison of all search methods' results

#         Args:
#             query: Search query
#             filter_dict: Filter condition
#             limit: Number of results
#             alpha: Weight for hybrid search (0: keyword only, 1: vector only)
#             fetch_k: Number of candidate documents for MMR search
#             **kwargs: Additional search parameters
#         """
#         search_methods = [
#             # 1. Basic similarity search
#             {
#                 "name": "Similarity Search",
#                 "method": self.similarity_search,
#                 "params": {"k": limit},
#             },
#             # 2. Similarity search with score
#             {
#                 "name": "Similarity Search with Score",
#                 "method": self.similarity_search_with_score,
#                 "params": {"k": limit},
#             },
#             # 3. MMR search
#             {
#                 "name": "MMR Search",
#                 "method": self.mmr_search,
#                 "params": {"k": limit, "fetch_k": fetch_k},
#             },
#             # 4. Hybrid search
#             {
#                 "name": "Hybrid Search",
#                 "method": self.hybrid_search,
#                 "params": {"limit": limit, "alpha": alpha},
#             },
#             # 5. Semantic search
#             {
#                 "name": "Semantic Search",
#                 "method": self.semantic_search,
#                 "params": {"limit": limit},
#             },
#             # 6. Keyword search
#             {
#                 "name": "Keyword Search",
#                 "method": self.keyword_search,
#                 "params": {"limit": limit},
#             },
#         ]

#         print("\n=== SEARCH METHODS COMPARISON ===")
#         print(f"Query: {query}")
#         if filter_query:
#             print(f"Filter: {self._format_filter(filter_query)}")
#         print("=" * 50)

#         for search_config in search_methods:
#             try:
#                 method_params = {
#                     **search_config["params"],
#                     "query": query,
#                     "filter_query": filter_query,
#                     **kwargs,
#                 }

#                 results = search_config["method"](**method_params)

#                 print(f"\n>>> {search_config['name'].upper()} <<<")
#                 self.print_results(results, search_config["name"], filter_query)

#             except Exception as e:
#                 print(f"\nError in {search_config['name']}: {str(e)}")

#             print("\n" + "=" * 50)

#     def delete_documents(self, filter_query: Any, ids: List[str], query: str) -> bool:
#         """문서 삭제"""
#         try:
#             if ids:
#                 self.delete_documents_by_ids(ids)
#             elif filter_query:
#                 self.delete_documents_by_filter(filter_query)
#             elif query:
#                 self.delete_documents_by_query(query)
#             return True
#         except Exception as e:
#             print(f"Error deleting documents: {e}")
#             return False

#     def delete_documents_by_ids(self, ids: List[str]) -> bool:
#         """ID로 문서 삭제"""
#         try:
#             for doc_id in ids:
#                 self.collection.data.delete(doc_id)
#             return True
#         except Exception as e:
#             print(f"Error deleting documents by IDs: {e}")
#             return False

#     def delete_documents_by_filter(self, filter_query: Any) -> bool:
#         """필터로 문서 삭제"""
#         try:
#             self.collection.data.delete_many(filter_query)
#             return True
#         except Exception as e:
#             print(f"Error deleting documents by filter: {e}")
#             return False

#     def delete_documents_by_query(self, query: str) -> bool:
#         """쿼리로 문서 삭제"""
#         try:
#             results = self.semantic_search(query)
#             if results:
#                 ids = [doc.metadata["uuid"] for doc in results]
#                 return self.delete_documents_by_ids(ids)
#             return True
#         except Exception as e:
#             print(f"Error deleting documents by query: {e}")
#             return False

#     def insert_documents(self, documents: List[Dict]) -> bool:
#         """문서 삽입"""
#         try:
#             self.upsert_documents(self._current_index, documents)
#             return True
#         except Exception as e:
#             print(f"Error inserting documents: {e}")
#             return False

#     def update_documents(self, documents: List[Dict]) -> bool:
#         """문서 업데이트"""
#         try:
#             self.upsert_documents(self._current_index, documents)
#             return True
#         except Exception as e:
#             print(f"Error updating documents: {e}")
#             return False

#     def replace_documents(self, documents: List[Dict]) -> bool:
#         """문서 교체"""
#         try:
#             self.upsert_documents(self._current_index, documents)
#             return True
#         except Exception as e:
#             print(f"Error replacing documents: {e}")
#             return False

#     def scroll(self, index_name: str, filter_query: Any = None, ids: List[str] = None, query: str = None, **kwargs) -> List[Any]:
#         """스크롤 검색"""
#         if ids:
#             return self.scroll_by_id(index_name, ids, **kwargs)
#         elif filter_query:
#             return self.scroll_by_filter(index_name, filter_query, **kwargs)
#         elif query:
#             return self.scroll_by_query(index_name, query, **kwargs)
#         return []

#     def scroll_by_id(self, index_name: str, ids: List[str], **kwargs) -> List[Any]:
#         """ID로 스크롤 검색"""
#         results = []
#         for doc_id in ids:
#             try:
#                 result = self.collection.data.get_by_id(doc_id)
#                 if result:
#                     results.append(result)
#             except Exception as e:
#                 print(f"Error in scroll_by_id: {e}")
#         return results

#     def scroll_by_filter(self, index_name: str, filter_query: Any, **kwargs) -> List[Any]:
#         """필터로 스크롤 검색"""
#         try:
#             results = self.collection.data.get_many(filter_query)
#             return list(results)
#         except Exception as e:
#             print(f"Error in scroll_by_filter: {e}")
#             return []

#     def scroll_by_query(self, index_name: str, query: str, **kwargs) -> List[Any]:
#         """쿼리로 스크롤 검색"""
#         try:
#             results = self.semantic_search(query, **kwargs)
#             return results
#         except Exception as e:
#             print(f"Error in scroll_by_query: {e}")
#             return []
