import neo4j
from .vectordbinterface import DocumentManager
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time


class Neo4jDocumentManager(DocumentManager):
    def __init__(self, client, index_name, embedding):
        self.index_name = index_name
        self.client = client
        self.embedding = embedding
        self.is_neo4j_above_523 = self.check_neo4j_version()
        self.get_index_info()

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

    def get_index_info(self):
        info_query = f"SHOW INDEX YIELD * WHERE name='{self.index_name}' RETURN labelsOrTypes, properties"
        info = self.client.execute_query(info_query).records[0]
        self.node_label = info["labelsOrTypes"][0]
        self.embedding_node_property = info["properties"][0]
        self.text_node_property = "text"

    def _embed_doc(self, texts) -> List[float]:
        """
        Embed texts

        Args:
        - texts: List of text

        Return:
        List of floats.
        """

        embedded = self.embedding.embed_documents(texts)
        return embedded

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Upsert documents into the Neo4j vectorstore

        Args:
        - texts: List of text. If ids is not provided, will create id based on the lowered text.
        - metadats: Optional. List of metadata. Default is None.
        - ids: Optional. List of id.

        Return:
        ids of upserted documents.
        """

        if ids is not None:
            assert len(ids) == len(
                texts
            ), "The length of ids and texts must be the same."

        elif ids is None:
            ids = [md5(text.lower().encode("utf-8")).hexdigest() for text in texts]

        embeds = self._embed_doc(texts)

        parameters = {
            "data": [
                {"text": text, "metadata": metadata, "embedding": embedded, "id": id}
                for text, metadata, embedded, id in zip(texts, metadatas, embeds, ids)
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

    def upsert_parallel(
        self, texts, metadatas=None, ids=None, batch_size=32, workers=10, **kwargs
    ):
        if ids is not None:
            assert len(ids) == len(texts), "Size of documents and ids must be the same"

        elif ids is None:
            ids = [md5(text.lower().encode("utf-8")).hexdigest() for text in texts]

        if batch_size > len(texts):
            batch_size = len(texts)

        text_batches = [
            texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]
        
        id_batches = [
            ids[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]
        
        meta_batches = [
            metadatas[i : i + batch_size] for i in range(0, len(texts), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [
                exe.submit(
                    self.upsert, texts=text_batch, metadatas=meta_batch, ids=id_batch
                )
                for text_batch, meta_batch, id_batch in zip(
                    text_batches, meta_batches, id_batches
                )
            ]
            results = []

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.extend(result)

        return results

    def search(self, query, k=10, **kwargs):
        embeded_query = self.embedding.embed_query(query)
        search_query = (
            f"MATCH (n:`{self.node_label}`) "
            f"WITH n, vector.similarity.cosine($embedded, n.embedding) AS score "
            f"ORDER BY score DESC "
            f"RETURN score, "
            f"n {{.*, `{self.embedding_node_property}`: Null}} AS metadata LIMIT $k "
        )

        parameters = {
            "k": k,
            "embedded": embeded_query,
        }

        try:
            _result = self.client.execute_query(search_query, parameters_=parameters)
        except Exception as e:
            _result = self.client.session(database="neo4j").run(
                neo4j.Query(text=search_query), parameters
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

    def delete(self, ids=None, filters=None, **kwargs):
        """
        Delete items by ids of filters.
        If ids and filters are both none, will delete all items in the index.
        If both are given, filters preceeds ids.

        Args:
        - ids: Delete items matching ids. Default is None
        - filters: Delete filters matching filters. Default is None

        Return:
        - True if deletion was successful
        - Raise error if deletion failed.
        """

        prefix_query = f"MATCH (n:{self.node_label})\n"

        delete_key = "all"

        if filters is not None:
            delete_key = "by filters"
            filter_queries = []
            for k, v in filters.items():
                if not isinstance(v, list):
                    v = [v]
                filter_queries.append(f"n.{k} IN {v}")
            filter_query = " AND ".join(filter_queries)

            delete_query = prefix_query + " WHERE " + filter_query + "\nDETACH DELETE n"

        elif filters is None and ids is not None:
            delete_key = "by ids"
            if not isinstance(ids, list):
                ids = [ids]
            filter_query = f"n.id IN {ids}"

            delete_query = prefix_query + " WHERE " + filter_query + "\nDETACH DELETE n"

        elif filters is None and ids is None:
            delete_query = prefix_query + "DETACH DELETE n"

        try:
            self.client.execute_query(delete_query)
        except Exception as e:
            print(f"Delete {delete_key} failed")
            raise e
        else:
            return True

    def scroll(
        self,
        ids: List = None,
        filters: Dict = None,
        k=10,
        meta_keys=None,
        include_embedding=False,
        **kwargs,
    ) -> List:
        """
        Scroll items from Neo4j Database based on given condition.
        If none of ids and filters are provided, will return k items in the index.
        if both ids and filters are provided, filters will precedent.

        Args:
        - ids: Scroll items that matches the given ids.
        - filters: Scroll items that matches the given filters.
        - k: Number of items to return
        - meta_keys: List of keys to include in metadata. If not provided, all metadatas will return except embedding. Default to None.
        - include_embedding: Boolean to determine include embedding or not. Default to False.

        Return
        - List of items.
        """

        base_query = f"MATCH (n:`{self.node_label}`)\n"

        if include_embedding:
            meta_keys.append(self.embedding_node_property)
            return_query = f"RETURN n LIMIT {k}"
        else:
            return_query = (
                f"RETURN n {{.*, `{self.embedding_node_property}`:Null}} LIMIT {k}"
            )

        condition_query = ""

        if filters is not None:
            filter_queries = []
            for k, v in filters.items():
                if not isinstance(v, list):
                    v = [v]
                filter_queries.append(f"n.{k} in {v}\n")
            condition_query = " AND ".join(filter_queries)
            condition_query = "WHERE\n" + condition_query

        elif ids is not None:
            condition_query = f"WHERE n.id IN {ids} "

        final_query = base_query + condition_query + return_query
        
        raw_results = self.client.execute_query(final_query)[0]

        items = []

        for raw_result in raw_results:
            value = raw_result.values()[0]
            tmp = dict()
            for k, v in value.items():
                if meta_keys is not None:
                    if k not in meta_keys:
                        continue
                if v is None:
                    continue
                tmp[k] = v
            items.append(tmp)

        return items


METRIC = {
    "cosine": "COSINE",
    "euclidean": "EUCLIDEAN",
}


class Neo4jIndexManager:
    def __init__(self, client):
        self.client = client

    def list_indexes(self):
        """
        Return names of indexes in Neo4j DB
        """
        list_query = "SHOW INDEXES"
        indexes = self.client.execute_query(list_query)

        return [record["name"] for record in indexes.records]

    def delete_index(self, index_name):
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

    def create_index(
        self,
        embedding,
        index_name: str = "vector",
        metric: str = "cosine",
        node_label: str = "Chunk",
        **kwargs,
    ):
        """Create new vector index in Neo4j.

        Args:
            - index_name : Index name for new index. Default is `vector`
            - node_label : Node label for nodes in the index. Default is `Chunk`
            - embedding_node_property : Name for embedding. Default is `embedding`
            - metric : Distance used to calculate similarity. Default is `cosine`.
                Supports `cosine`, `euclidean`, `maxinnerproduct`, `dotproduct`, `jaccard`

        Return:
        Returns created Neo4jDBManager object connected to created or already existed index.
        """
        assert metric in METRIC.keys(), f"Choose metric among {list(METRIC.keys())}"

        if index_name in self.list_indexes():
            print(
                f"Index with name {index_name} already exists.\nReturning Neo4jDBManager object."
            )

        self.embedding_node_property = kwargs.get(
            "embedding_node_property", "embedding"
        )

        self.metric = METRIC[metric.lower()]
        self.index_name = index_name
        self.node_label = node_label
        self.text_node_property = kwargs.get("text_node_property", "text")
        dimension = len(embedding.embed_query("foo"))

        index_query = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (m:`{node_label}`) ON m.`{self.embedding_node_property}` "
            "OPTIONS { indexConfig: { "
            "`vector.dimensions`: toInteger($embedding_dimension), "
            "`vector.similarity_function`: $similarity_metric }}"
        )

        parameters = {
            "embedding_dimension": dimension,
            "similarity_metric": self.metric,
        }

        try:
            self.client.execute_query(
                index_query, parameters_=parameters, database="neo4j"
            )
        except Exception as e:
            raise e

        else:
            info_str = (
                f"Index name: {index_name}",
                f"Node label: {node_label}",
                f"Similarity metric: {self.metric}",
                f"Embedding dimension: {dimension}",
                f"Embedding node property: {self.embedding_node_property}",
                f"Text node property: {self.text_node_property}",
            )
            print("Created index information")
            print(info_str)
            print("Index creation successful. Return Neo4jDBManager object.")
            return Neo4jDocumentManager(
                self.client, index_name=index_name, embedding=embedding
            )

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
