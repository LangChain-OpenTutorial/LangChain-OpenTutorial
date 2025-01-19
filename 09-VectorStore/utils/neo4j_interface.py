import neo4j
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time

METRIC = {
    "cosine": "COSINE",
    "euclidean": "EUCLIDEAN",
}


class Neo4jDB:
    def __init__(
        self,
        embedding=None,
        uri=None,
        username=None,
        password=None,
        index_name=None,
        node_label=None,
        _database="neo4j",
        metric=None,
        embedding_node_property=None,
        text_node_property=None,
        dimension=None,
    ):
        if uri is None:
            uri = os.environ.get("NEO4J_URI", None)
        if username is None:
            username = os.environ.get("NEO4J_USERNAME", None)
        if password is None:
            password = os.environ.get("NEO4J_PASSWORD", None)

        assert all(
            [uri, username, password]
        ), "You must set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD environmental variables or initialize Neo4jDB class by pass the variables directly"

        if embedding is not None:
            assert "embed_query" in dir(embedding) and "embed_documents" in dir(
                embedding
            ), "embedding must have have embed_query and embed_document methods.\nProvided embedding does not have both of those."

        self.uri = uri
        self.username = username
        self.password = password
        self.embedding = embedding
        self.index_name = index_name
        self.node_label = node_label
        self._database = _database
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self.metric = metric
        self.dimension = dimension

        try:
            self.client = neo4j.GraphDatabase.driver(
                uri=self.uri, auth=(self.username, self.password)
            )
        except Exception as e:
            print(e)
            raise e
        else:
            self.is_neo4j_above_523 = self.check_neo4j_version()
            if self.is_neo4j_above_523:
                version_str = "Neo4j version is above 5.23"
            else:
                version_str = "Neo4j version is below 5.24"
            if self.index_name is None:
                print("Connected to Neo4j database")
                print(f"Connection info\nURI={self.uri}\nusername={self.username}")
                print(version_str)

    def check_neo4j_version(self):
        db_data = self.client.execute_query("CALL dbms.components()")
        version = db_data[0][0]["versions"][0]

        if "aura" in version:
            version_tuple = tuple(map(int, version.split("-")[0].split("."))) + (0,)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

        if version_tuple >= (5, 23, 0):
            return True
        else:
            return False

    def connect(self) -> None:
        """Connect to neo4j graph database.
        If connection cannot be established, raise error
        If connection established succesfully, prints connection info and return None
        """
        return self.client

    def get_api_key(self):
        return {
            "NEO4J_URI": self.uri,
            "NEO4J_USERNAME": self.username,
            "NEO4J_PASSWORD": self.password,
        }

    def create_index(
        self,
        embedding,
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        index_name: str = "vector",
        metric: str = "cosine",
        node_label: str = "Chunk",
        _database: str = "neo4j",
        **kwargs,
    ):
        if index_name in self.list_indexes():
            print(f"index {index_name} exists")
            return self._return_exist_index(
                self.client,
                uri=self.uri,
                username=self.username,
                password=self.password,
                embedding=embedding,
                embedding_node_property=embedding_node_property,
                text_node_property=text_node_property,
                index_name=index_name,
                metric=metric,
                node_label=node_label,
                _database=_database,
            )

        return self._create_new_index(
            self.client,
            uri=self.uri,
            username=self.username,
            password=self.password,
            embedding=embedding,
            embedding_node_property=embedding_node_property,
            text_node_property=text_node_property,
            index_name=index_name,
            metric=metric,
            node_label=node_label,
            _database=_database,
        )

    @classmethod
    def _return_exist_index(
        cls,
        client,
        uri,
        username,
        password,
        embedding,
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        index_name: str = "vector",
        metric: str = "cosine",
        node_label: str = "Chunk",
        _database: str = "neo4j",
        **kwargs,
    ):
        query = f"SHOW INDEX YIELD * WHERE name='{index_name}' RETURN labelsOrTypes, properties"
        info = client.execute_query(query).records[0]
        node_label = info["labelsOrTypes"][0]
        embedding_node_property = info["properties"][0]
        return cls(
            uri=uri,
            username=username,
            password=password,
            embedding=embedding,
            index_name=index_name,
            node_label=node_label,
            _database=_database,
            metric=metric,
            embedding_node_property=embedding_node_property,
            text_node_property=text_node_property,
        )

    @classmethod
    def _create_new_index(
        cls,
        client,
        uri,
        username,
        password,
        embedding,
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        index_name: str = "vector",
        metric: str = "cosine",
        node_label: str = "Chunk",
        _database: str = "neo4j",
        **kwargs,
    ):
        """Create new vector index in Neo4j.

        Args:
            - index_name : Index name for new index. Default is `vector`
            - node_label : Node label for nodes in the index. Default is `Chunk`
            - embedding_node_property : Name for embedding. Default is `embedding`
            - metric : Distance used to calculate similarity. Default is `cosine`.
                Supports `cosine`, `euclidean`.

        Returns:
            - returns True if index is created successfully
        """

        assert (
            metric in METRIC.keys()
        ), f"Choose metric among {list(METRIC.keys())}. Your metric is {metric}"

        if embedding is None and kwargs.get("dimension", None) is None:
            raise ValueError(
                "You must provide either embedding function or dimension of resulting vector when you encode a document with your choice of embedding function."
            )

        if "dimension" in kwargs:
            dimension = kwargs["dimension"]
        else:
            dimension = len(embedding.embed_query("foo"))
        index_name = index_name
        node_label = node_label
        metric = METRIC[metric]

        index_query = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (m:`{node_label}`) ON m.`{embedding_node_property}` "
            "OPTIONS { indexConfig: { "
            "`vector.dimensions`: toInteger($embedding_dimension), "
            "`vector.similarity_function`: $similarity_metric }}"
        )

        parameters = {
            "embedding_dimension": dimension,
            "similarity_metric": metric,
        }

        try:
            client.execute_query(
                index_query, parameters_=parameters, database=_database
            )
        except Exception as e:
            print("Failed to create index")
            print(e)

        else:
            info_str = (
                f"Index name: {index_name}\n"
                f"Node label: {node_label}\n"
                f"Similarity metric: {metric}\n"
                f"Embedding dimension: {dimension}\n"
                f"Embedding node property: {embedding_node_property}\n"
                f"Text node property: {text_node_property}\n"
            )
            print("Created index information")
            print(info_str)
            return cls(
                uri=uri,
                username=username,
                password=password,
                embedding=embedding,
                index_name=index_name,
                node_label=node_label,
                _database=_database,
                metric=metric,
                embedding_node_property=embedding_node_property,
                text_node_property=text_node_property,
                dimension=dimension,
            )

    @classmethod
    def _connect_to_index(cls, client, embedding, index_name, node_label):
        return cls(index_name=index_name, embedding=embedding, node_label=node_label)

    def connect_to_index(self, index_name, embedding=None):
        """Connect to existing index
        Args:
            - index_name: Name of index to connect

        Return:
            - Neo4jDB instance
        """
        query = f"SHOW INDEX YIELD * WHERE name='{index_name}' RETURN labelsOrTypes"
        node_label = self.client.execute_query(query).records[0]["labelsOrTypes"][0]

        if embedding is not None:
            self.embedding = embedding

        return self._connect_to_index(
            self.client, self.embedding, index_name, node_label
        )

    def list_indexes(self):
        """Get list of index in current Neo4j database.
        Returns:
            - list of index names
        """

        query = """
        SHOW INDEXES
        """

        indexes = self.client.execute_query(query)

        result = [record["name"] for record in indexes.records]

        return result

    def get_index(self, index_name: str) -> Dict:
        """Get information for given index name

        Args:
            - index_name : index name to get information.

        Returns:
            Information about the index.
        """
        query = f"""
        SHOW INDEXES YIELD * WHERE name='{index_name}'
        """

        try:
            result = self.client.execute_query(query)
        except Exception as e:
            print("error occured while get index information")
            raise e
        else:
            if len(result.records) == 0:
                return None
            result = {k: result.records[0][k] for k in result.keys}
        return result

    def delete_index(self, index_name: str) -> Union[bool, None]:
        """Delete index

        Args:
            - index_name : index name to delete.

        Returns:
            True if index deleted successfully.
            If error occured, will raise error.
        """
        query = f"DROP INDEX {index_name}"
        if self.get_index(index_name) is None:
            return f"{index_name} does not exists"

        try:
            self.client.execute_query(query)
        except Exception as e:
            print(f"Drop index {index_name} failed")
            raise e
        else:
            return True

    # Query related functions
    def query(self, index_name, query_vector=None, top_k=10, **kwargs):
        pass

    def delete_node(
        self,
        index_name: str = None,
        filters: List[Dict] = None,
        ids: List = None,
        **kwargs,
    ) -> bool:
        """Delete nodes by filter
        One of filters or ids must be provided, but not both.

        Args:
            - index_name: index of nodes to delete
            - filters: Delete nodes matching these filters
            - ids: Delete nodes matching these ids

        Returns:
            - True if deletion was successful
            - raise error if deletion failed
        """
        if filters is None and ids is None:
            raise AssertionError("You must provide one of filters or ids")
        elif filters is not None and ids is not None:
            raise AssertionError("You must provide only one of filters or ids")

        if filters is not None:
            return self.delete_by_filter(index_name, filters)
        elif ids is not None:
            return self.delete_by_id(index_name, ids)

    def delete_by_id(self, index_name: str = None, ids: List = None, **kwargs) -> bool:
        """Delete nodes by filter
        One of filters or ids must be provided, but not both.

        Args:
            - index_name: index of nodes to delete
            - ids: Delete nodes matching these ids

        Returns:
            - True if deletion was successful
            - raise error if deletion failed
        """

        if index_name is not None:
            label = self.client.execute_query(
                f"SHOW INDEX YIELD * WHERE name='{index_name}' RETURN labelsOrTypes"
            ).records[0]["labelsOrTypes"][0]
            prefix_query = f"MATCH (n:{label})\n"
        else:
            prefix_query = "MATCH (n)\n"

        if ids is not None:
            if not isinstance(ids, list):
                ids = [ids]
            filter_query = f"n.id IN {ids}"

        query = prefix_query + " WHERE " + filter_query + "\nDETACH DELETE n"

        try:
            self.client.execute_query(query)
        except Exception as e:
            print("Delete by filter failed")
            raise e
        else:
            return True

    def delete_by_filter(
        self, index_name: str = None, filters: Dict = None, **kwargs
    ) -> bool:
        """Delete nodes by filter
        One of filters or ids must be provided, but not both.

        Args:
            - index_name: index of nodes to delete
            - filters: Delete nodes matching these filters

        Returns:
            - True if deletion was successful
            - raise error if deletion failed
        """

        if index_name is not None:
            label = self.client.execute_query(
                f"SHOW INDEX YIELD * WHERE name='{index_name}' RETURN labelsOrTypes"
            ).records[0]["labelsOrTypes"][0]
            prefix_query = f"MATCH (n:{label})\n"
        else:
            prefix_query = "MATCH (n)\n"

        if filters is not None:
            filter_queries = []
            for k, v in filters.items():
                if not isinstance(v, list):
                    v = [v]
                filter_queries.append(f"n.{k} IN {v}")
            filter_query = " AND ".join(filter_queries)

        query = prefix_query + " WHERE " + filter_query + "\nDETACH DELETE n"

        try:
            self.client.execute_query(query)
        except Exception as e:
            print("Delete by filter failed")
            raise e
        else:
            return True

    # Document upsert related functions

    def add_embedding(self, documents: list[Document], ids: list[str] = []) -> list:
        """Encode documents
        Args:
            - documents: List of documents to upsert into the vectorstore
            - ids: List of ids for each documents. If not provided, md5 hash function will created based on the text of each document.

        Returns: Returns (encoded_vectors, id, metadata) tuple list
        """

        texts = [doc.page_content for doc in documents]

        if not all(ids):
            ids = [md5(text.encode("utf-8")).hexdigest() for text in texts]

        metadatas = [doc.metadata for doc in documents]

        encoded = self.embedding.embed_documents(texts)

        return (texts, encoded, ids, metadatas)

    def _insert_documents(self, documents: list[Document], ids: list[str] = []):
        """util function for upsert_document.

        Args:
            - documents: List of Document to upsert to database
            - ids: List of ids paired with documents. If not provided will be created by md5 hash function.

        Return:
            - ids: List of ids upserted documents. If ids were provided this must be the same to the ids provided.
        """

        texts, encodes, ids, metadatas = self.add_embedding(documents, ids)

        parameters = {
            "data": [
                {"text": text, "metadata": metadata, "embedding": encode, "id": id}
                for text, metadata, encode, id in zip(texts, metadatas, encodes, ids)
            ]
        }

        if self.is_neo4j_above_523:
            call_prefix = "CALL (row) { "
        else:
            call_prefix = "CALL { WITH row "

        import_query = (
            "UNWIND $data AS row "
            f"{call_prefix}"
            f"MERGE (c:`{self.node_label}` {{id: row.id}}) "
            "WITH c, row "
            f"CALL db.create.setNodeVectorProperty(c, "
            f"'{self.embedding_node_property}', row.embedding) "
            f"SET c.`{self.text_node_property}` = row.text "
            "SET c += row.metadata "
            "} IN TRANSACTIONS OF 1000 ROWS "
        )
        try:
            self.client.execute_query(import_query, parameters_=parameters)
        except Exception as e:
            if "can only be executed in an implicit transaction" in str(e):
                self.client.session().run(neo4j.Query(text=import_query), parameters)
            elif "failed to obtain a connection from the pool" in str(e):
                time.sleep(10)
                self.client.session().run(neo4j.Query(text=import_query), parameters)

        return ids

    def upsert_documents(self, documents, batch_size=32, ids=None, **kwargs):
        """Upsert documents into the vectorstore

        Args:
            - documents: List of documents to upsert into the vectorstore
            - batch_size: Batch size of documents to add or update. Default is 32.
            - kwargs: Additional keyword arguments.
                    if kwargs contains ids and documents contain ids,
                    the ids in the kwargs will receive precedence.

        Returns:
            Returns list of ids of the documents upserted.
        """
        assert self.index_name is not None, "You MUST connect to index first."

        if self.node_label is None and self.index_name is not None:
            self.node_label = self.client.execute_query(
                f"SHOW INDEX YIELD * WHERE name='{self.index_name}' RETURN labelsOrTypes"
            ).records[0]["labelsOrTypes"][0]

        if ids is not None:
            assert len(ids) == len(
                documents
            ), "Size of documents and ids must be the same"

        else:
            ids = [False] * len(documents)

        id_batches = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

        if batch_size > len(documents):
            batch_size = len(documents)

        doc_batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        result_ids = []

        for doc_batch, id_batch in zip(doc_batches, id_batches):
            result_ids.extend(self._insert_documents(doc_batch, id_batch))

        return result_ids

    def upsert_documents_parallel(
        self, documents, batch_size=32, max_workers=10, ids=None, **kwargs
    ):
        """Add or update documents in the vectorstore parallel.

        Args:
            documents: Documents to add to the vectorstore.
            batch_size: Batch size of documents to add or update.
            Default is 32.
            max_workers: Number of threads to use.
            Default is 10.
            kwargs: Additional keyword arguments.
                if kwargs contains ids and documents contain ids,
                the ids in the kwargs will receive precedence.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of ids does not match the number of documents.
        """
        assert self.index_name is not None, "You MUST connect to index first."

        if ids is not None:
            assert len(ids) == len(
                documents
            ), "Size of documents and ids must be the same"

        if batch_size > len(documents):
            batch_size = len(documents)

        doc_bathces = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        id_batches = [
            ids[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        if self.node_label is None and self.index_name is not None:
            self.node_label = self.client.execute_query(
                f"SHOW INDEX YIELD * WHERE name='{self.index_name}' RETURN labelsOrTypes"
            ).records[0]["labelsOrTypes"][0]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.upsert_documents, batch, ids=ids)
                for batch, ids in zip(doc_bathces, id_batches)
            ]
            results = []
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Upserting documents..."
            ):
                result = future.result()
                if result:
                    results.extend(result)

        return results

    def delete_by_query(
        self, query: str, index_name: Optional[str | None] = None, **kwrags
    ) -> bool:
        """Delete nodes by query
        Args:
            - query: Cypher query
            - index_name: Optional. Default is None. If specified, will delete node only in the given index

        Returns:
            - True if deletion is successful else raise error
        """
        try:
            self.client.execute_query(query)
        except Exception as e:
            print(f"Error {e} occured during deletion")
            raise e
        else:
            return True

    def scroll_by_query(
        self, query: str, index_name: Optional[str | None] = None, **kwargs
    ) -> List:
        """Scroll nodes by query
        Args:
            - query: Cypher query
            - index_name: Optional. Default is None. If specified, will scroll node only in the given index

        Returns:
            - List of nodes if successful, else raise error
        """
        try:
            _result = self.client.execute_query(query)
        except Exception as e:
            print(f"Error {e} occured during scroll")
            raise e
        else:
            result = []
            for record in _result.records:
                result.append({k: record[k] for k in record.keys()})
            return result

    def scroll_by_filter(
        self,
        filters=None,
        ids=None,
        limit=10,
        include_embedding=False,
        include_meta=None,
        **kwargs,
    ) -> List[Dict]:
        """Query nodes by filter or id
        If none of filter or id provided, will return all nodes.
        If this method is called directly from client without index_name set, all nodes will be returned.

        Args:
            - filters: filter for query data
            - ids: id for query data
            - limit: number of nodes to return
            - include_embedding: Set True to include embedded vector to result. Default is False
            - include_meta: list of metadata keys to include. If set to None, all metadatas will be included. Default is None.

        Returns:
            - list of nodes
        """

        if self.index_name is not None:
            label = self.client.execute_query(
                f"SHOW INDEX YIELD * WHERE name='{self.index_name}' RETURN labelsOrTypes"
            ).records[0]["labelsOrTypes"][0]
            prefix_query = f"MATCH (n:{label})\n"
        else:
            prefix_query = "MATCH (n)\n"

        if filters is None and ids is None:
            filter_query = ""

        elif filters is not None:
            filter_queries = []
            for k, v in filters.items():
                if not isinstance(v, list):
                    v = [v]
                filter_queries.append(f"n.{k} IN {v}")
            filter_query = " AND ".join(filter_queries)

        elif ids is not None:
            filter_queries = []
            filter_query = f"n.id IN {ids}"

        limit_query = "\nRETURN n" if limit is None else f"\nRETURN n LIMIT {limit}"

        if filter_query != "":
            query = prefix_query + " WHERE " + filter_query + limit_query
        else:
            query = prefix_query + limit_query

        _results = self.client.execute_query(query)

        results = list()

        for _result in _results.records:
            node = _result["n"]

            result = {"id": node["id"]}
            if include_embedding:
                result.update({"embedding": node["embedding"]})
            if include_meta is None:
                include_meta = [k for k in node.keys() if k not in ["id", "embedding"]]
            result.update(
                {"metadata": {k: node[k] for k in node.keys() if k in include_meta}}
            )
            results.append(result)

        return results

    def scroll_by_ids(
        self,
        ids=None,
        limit=10,
        include_embedding=False,
        include_meta=None,
        **kwargs,
    ) -> List[Dict]:
        """Query nodes by filter or id
        If none of filter or id provided, will return all nodes.
        If this method is called directly from client without index_name set, all nodes will be returned.

        Args:
            - ids: id for query data
            - limit: number of nodes to return
            - include_embedding: Set True to include embedded vector to result. Default is False
            - include_meta: list of metadata keys to include. If set to None, all metadatas will be included. Default is None.

        Returns:
            - list of nodes
        """

        if self.index_name is not None:
            label = self.client.execute_query(
                f"SHOW INDEX YIELD * WHERE name='{self.index_name}' RETURN labelsOrTypes"
            ).records[0]["labelsOrTypes"][0]
            prefix_query = f"MATCH (n:{label})\n"
        else:
            prefix_query = "MATCH (n)\n"

        if ids is not None:
            if not isinstance(ids, list):
                ids = [ids]
            filter_query = f"n.id IN {ids}"

        limit_query = "\nRETURN n" if limit is None else f"\nRETURN n LIMIT {limit}"

        query = prefix_query + " WHERE " + filter_query + limit_query

        _results = self.client.execute_query(query)

        results = list()

        for _result in _results.records:
            node = _result["n"]

            result = {"id": node["id"]}
            if include_embedding:
                result.update({"embedding": node["embedding"]})
            if include_meta is None:
                include_meta = [k for k in node.keys() if k not in ["id", "embedding"]]
            result.update(
                {"metadata": {k: node[k] for k in node.keys() if k in include_meta}}
            )
            results.append(result)

        return results

    def scroll_nodes(
        self,
        filters=None,
        ids=None,
        query=None,
        limit=10,
        include_embedding=False,
        include_meta=None,
        **kwargs,
    ):
        if filters is not None:
            print("Scroll nodes by filter")
            return self.scroll_by_filter(
                filters=filters,
                include_embedding=include_embedding,
                include_meta=include_meta,
                limit=limit,
            )
        elif ids is not None:
            print("Scroll nodes by ids")
            return self.scroll_by_ids(
                ids=ids,
                include_embedding=include_embedding,
                include_meta=include_meta,
                limit=limit,
            )
        elif query is not None:
            print("Scroll nodes by query")
            return self.scroll_by_query(query=query)
        else:
            return self.scroll_by_filter(
                include_embedding=include_embedding,
                include_meta=include_meta,
                limit=limit,
            )

    @staticmethod
    def preprocess_documents(
        split_docs, metadata_keys, min_length, use_basename=False, **kwargs
    ):
        metadata = kwargs

        if use_basename:
            assert metadata.get("source", None) is not None, "source must be provided"
            metadata["source"] = metadata["source"].split("/")[-1]

        result_docs = []
        for idx, doc in enumerate(split_docs):
            if len(doc.page_content) < min_length:
                continue
            for k in metadata_keys:
                doc.metadata.update({k: metadata.get(k, "")})
            doc.metadata.update({"page": idx + 1})
            result_docs.append(doc)

        return result_docs

    def search(
        self,
        query=None,
        embeded_query=None,
        index_name=None,
        filters=[],
        with_score=False,
        top_k=3,
        **kwargs,
    ):
        assert self.index_name is not None, "You must provide index name"

        if query is None and embeded_query is None:
            raise ValueError("You must provide either query or embeded values of query")

        if query is not None and embeded_query is not None:
            print(
                "Both query and embeded value of query passed. Using embded value of query"
            )

        if embeded_query is None:
            embeded_query = self.embedding.embed_query(query)

        if kwargs.get("include_vector"):
            result_query = (
                f"MATCH (n:`{self.node_label}`) "
                f"WITH n, vector.similarity.cosine($embeded, n.embedding) AS score "
                f"ORDER BY score DESC "
                f"RETURN r, score LIMIT $k "
                f"n {{.*, `{self.text_node_property}`: Null, `{self.embedding_node_property}`: Null}} AS metadata LIMIT $k "
            )
        else:
            result_query = (
                f"MATCH (n:`{self.node_label}`) "
                f"WITH n, vector.similarity.cosine($embeded, n.embedding) AS score "
                f"ORDER BY score DESC "
                f"RETURN score, "
                f"n {{.*, `{self.embedding_node_property}`: Null}} AS metadata LIMIT $k "
            )

        parameters = {
            "k": top_k,
            "embeded": embeded_query,
        }

        try:
            _result = self.client.execute_query(result_query, parameters_=parameters)
        except:
            _result = self.client.session(database=self._database).run(
                neo4j.Query(text=result_query), parameters
            )

        result = []
        for _r in _result.records:
            result.append(
                {
                    "text": _r["metadata"].pop("text"),
                    "metadata": _r["metadata"],
                    "score": round(float(_r["score"]), 3),
                }
            )

        return result

    @staticmethod
    def remove_lucene_chars(text: str) -> str:
        """Remove Lucene special characters"""
        special_chars = [
            "+",
            "-",
            "&",
            "|",
            "!",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            "^",
            '"',
            "~",
            "*",
            "?",
            ":",
            "\\",
        ]
        for char in special_chars:
            if char in text:
                text = text.replace(char, " ")
        return text.strip()
