import os
import weaviate
from tqdm import tqdm
from utils.base import VectorDB
from weaviate.classes.init import Auth
from weaviate.collections.classes.filters import Filter
from weaviate.classes.config import Configure, VectorDistances
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Tuple
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

if TYPE_CHECKING:
    import weaviate

class ConnectDB:
    def __init__(self):
        self.db_map = {
            "weaviate": WeaviateDB,
            # 다른 DB도 추가 가능
        }

    def connect(self, db_type, **kwargs):
        client_creator = self.db_map.get(db_type)
        if not client_creator:
            raise ValueError(f"Unsupported database type: {db_type}")
        return client_creator().connect(**kwargs)


class WeaviateDB(VectorDB):
    def __init__(self, api_key: str, url: str):
        self._client = None
        self._current_index = None
        self._collection = None
        self._text_key = "text"
        self._api_key = api_key
        self._url = url

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
            **kwargs
        )
        return self._client

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
        
    def get_api_key(self):
        """API 키 반환"""
        return self._api_key

    def create_index(
        self,
        index_name: str,
        description: Optional[str] = None,
        metric: str = "cosine",
        **kwargs,
    ) -> None:
        """Weaviate에 스키마(인덱스)를 생성합니다."""
        distance_metric = getattr(VectorDistances, metric.upper(), None)

        # Set vector_index_config to hnsw
        vector_index_config = Configure.VectorIndex.hnsw(
            distance_metric=distance_metric
        )

        # Create the collection in Weaviate
        try:
            self._client.collections.create(
                name=index_name,
                description=description,
                vector_index_config=vector_index_config,
                **kwargs
            )
            print(f"Collection '{index_name}' created successfully.")
        except Exception as e:
            print(f"Failed to create collection '{index_name}': {e}")

    def delete_index(self, index_name: str) -> None:
        """Weaviate에서 특정 클래스(인덱스)를 삭제합니다."""
        self._client.collections.delete(name=index_name)
        return print(f"Deleted index: {index_name}")

    def delete_all_collections(self) -> None:
        self._client.collections.delete_all()
        return print("Deleted all collections")

    def list_indices(self) -> str:
        """현재 활성화된 클래스 목록(인덱스)의 상세 정보를 문자열로 반환합니다."""
        collections = self._client.collections.list_all()
        result = []

        if collections:
            result.append("Collections (indexes) in the Weaviate schema:")
            for name, config in collections.items():
                result.append(f"- Collection name: {name}")
                result.append(
                    f"  Description: {config.description if config.description else 'No description available'}"
                )
                result.append(f"  Properties:")
                for prop in config.properties:
                    result.append(f"    - Name: {prop.name}, Type: {prop.data_type}")
                result.append("")
        else:
            result.append("No collections found in the schema.")

        return "\n".join(result)

    def get_index(self, index_name: str) -> Any:
        """특정 인덱스를 조회합니다."""
        return self._client.collections.get(index_name)

    def preprocess_documents(
        self,
        split_docs: List[Document],
        metadata: Dict[str, str] = None,
    ) -> List[Dict[str, Dict[str, object]]]:
        """
        Processes a list of pre-split documents into a format suitable for storing in Weaviate.

        :param split_docs: List of LangChain Document objects (each containing page_content and metadata).
        :param metadata: Additional metadata to include in each chunk (e.g., title, source).
        :return: A list of dictionaries, each representing a chunk in the format:
                {'properties': {'text': ..., 'order': ..., ...metadata}}
        """
        processed_chunks = []

        # Iterate over Document objects
        for idx, doc in enumerate(split_docs, start=1):
            # Extract text from page_content and include metadata
            chunk_data = {"text": doc.page_content, "order": idx}
            # Combine with metadata from Document and additional metadata if provided
            if metadata:
                chunk_data.update(metadata)
            if doc.metadata:
                chunk_data.update(doc.metadata)

            # Format for Weaviate
            processed_chunks.append(chunk_data)

        return processed_chunks

    def upsert_documents(
        self,
        index_name: str,
        data_objects: List[Dict],
        unique_key: str = "order",
        show_progress: bool = False,
    ) -> List[str]:
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
        collection = self._client.collections.get(index_name)
        successful_ids = []

        iterator = tqdm(
            data_objects, desc="Document processing", disable=not show_progress
        )
        for data_object in iterator:
            try:
                unique_value = str(data_object[unique_key])
                object_uuid = weaviate.util.generate_uuid5(index_name, unique_value)

                if collection.data.exists(object_uuid):
                    collection.data.replace(
                        uuid=object_uuid,
                        properties=data_object,
                    )
                else:
                    collection.data.insert(
                        uuid=object_uuid,
                        properties=data_object,
                    )
                successful_ids.append(object_uuid)
            except Exception as e:
                print(f"\nError occurred while processing object: {e}")

        return successful_ids
    
    def upsert_documents_parallel(
        self,
        index_name: str,
        data_objects: List[Dict],
        unique_key: str = "order",
        batch_size: int = 100,
        max_workers: Optional[int] = 4,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Utilizes ThreadPoolExecutor to upsert objects in parallel.

        Args:
            index_name: Collection name
            data_objects: Data objects to upsert
            unique_key: Unique key
            batch_size: Batch size
            max_workers: Maximum number of workers
            show_progress: Whether to show progress

        Returns:
            List of UUIDs of successfully processed objects
        """
        def create_batches(data: List, size: int) -> List[List]:
            return [data[i:i + size] for i in range(0, len(data), size)]

        def process_batch(batch: List[Dict]) -> List[str]:
            try:
                return self.upsert_documents(
                    index_name=index_name,
                    data_objects=batch,
                    unique_key=unique_key,
                    show_progress=False,
                )
            except Exception as e:
                print(f"\nError occurred while processing batch: {e}")
                return []

        all_successful_ids = []
        batches = create_batches(data_objects, batch_size)

        print(f"Total documents: {len(data_objects)}, Number of batches: {len(batches)}")
        print(f"Batch size: {batch_size}, Number of threads: {max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing batches",
                disable=not show_progress
            ):
                try:
                    batch_ids = future.result()
                    all_successful_ids.extend(batch_ids)
                except Exception as e:
                    print(f"\nError occurred while processing batch: {e}")

        return all_successful_ids
    
    def similarity_search(
        self,
        index_name: str,
        query: str,
        filter_query: Optional[Filter] = None,
        top_k: int = 3,
        **kwargs: Any,
    ):
        """
        Perform basic similarity search
        """
        documents = self.vector_store.similarity_search(
            query, k=top_k, filters=filter_query, **kwargs
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

    def scroll(self, index_name: str, filter_query: Any = None, ids: List[str] = None, query: str = None, **kwargs) -> List[Any]:
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

    def scroll_by_filter(self, index_name: str, filter_query: Any, **kwargs) -> List[Any]:
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