from vectordbinterface import DocumentManager
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time, uuid, json

import psycopg2
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

DISTANCE = {
    "cosine": "<=>",
    "l2": "<->",
    "l1": "<+>",
}

COMPARISION_OPERATORS = {
    "$eq": "==",
    "$ne": "!=",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

OPERATORS = {
    "$in" : "IN",
    "$nin": "NOT IN",
    "$between": "BETWEEN",
    "$exists": "EXISTS",
    "$like": "LIKE",
    "$ilike": "IN LIKE",
    "$and": "AND", 
    "$or": "OR", 
    "$not": "NOT"
}

class pgVectorIndexManager:
    def __init__(
        self,
        host=None,
        port=None,
        username=None,
        user=None,
        password=None,
        passwd=None,
        dbname=None,
        db=None,
    ):
        assert host is not None, "host is missing"
        assert port is not None, "port is missing"
        assert username is not None or user is not None, "username(or user) is missing"
        assert (
            password is not None or passwd is not None
        ), "password(or passwd) is missing"
        assert dbname is not None or db is not None, "dbname(or db) is missing"

        self.host = host
        self.port = port
        self.userName = username if username is not None else user
        self.passWord = password if password is not None else passwd
        self.dbName = dbname if dbname is not None else db

    def _connect(self):
        return psycopg2.connect(
            database=self.dbName,
            user=self.userName,
            password=self.passWord,
            port=self.port,
            host=self.host,
        )

    def list_indexes(self):
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE';
        """

        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(query)

        except Exception as e:
            msg = f"List collection failed due to {type(e)} {str(e)}"
        else:
            collections = cur.fetchall()
            msg = ""
        finally:
            conn.close()
            print(msg)
            return collections

    def delete_index(self, collection_name):
        query = f"DROP TABLE {collection_name};"
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(query)
                conn.commit()
        except Exception as e:
            flag = False
            msg = (
                f"Delete collection {collection_name} failed due to {type(e)} {str(e)}"
            )
        else:
            flag = True
            msg = f"Deleted collection {collection_name} successfully."
        finally:
            conn.close()
            print(msg)
            return flag

    def _check_extension(self):
        query = "SELECT * FROM pg_extension;"
        create_ext_query = "CREATE EXTENSION vector;"

        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(query)
                extensions = str(cur.fetchall())
        finally:
            conn.close()

        if "vector" not in extensions:
            try:
                with self._connect() as conn:
                    cur = conn.cursor()
                    cur.execute(create_ext_query)
                    conn.commit()
            finally:
                conn.close()

    def get_index(self, collection_name, embedding):
        connection_info = {
            "host": self.host,
            "port": self.port,
            "user": self.userName,
            "password": self.passWord,
            "dbname": self.dbName,
        }
        return pgVectorDocumentManager(
            connection_info=connection_info,
            collection_name=collection_name,
            embedding=embedding,
        )

    def create_index(self, collection_name, dimension=None, embedding=None):
        self._check_extension()

        assert (
            embedding is not None or dimension is not None
        ), "One of embedding or dimension must be provided"

        if dimension is None:
            dimension = len(embedding.embed_query("foo"))

        query = (
            f"CREATE TABLE {collection_name} "
            f"(id bigserial PRIMARY KEY, "
            f"embedding vector({dimension}) NOT NULL,"
            "metadata jsonb NULL, "
            "doc_id text NOT NULL UNIQUE)"
        )
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(query)
                conn.commit()
        except Exception as e:
            msg = (
                f"Create collection {collection_name} failed due to {type(e)} {str(e)}"
            )
            flag = False
        else:
            msg = f"Created collection {collection_name} successfully"
            flag = True
        finally:
            print(msg)
            conn.close()
            return flag


class pgVectorDocumentManager(DocumentManager):
    def __init__(self, connection_info, collection_name, embedding, distance="cosine"):
        self.collection_name = collection_name
        self.connection_info = connection_info
        self.embedding = embedding
        self.distance = distance.lower()

    def _connect(self):
        return psycopg2.connect(**self.connection_info)

    def _embed_doc(self, texts) -> List[float]:
        embedded = self.embedding.embed_documents(texts)
        return embedded

    def upsert(self, texts, metadatas=None, ids=None, **kwargs):
        if ids is not None:
            assert len(ids) == len(
                texts
            ), "The length of ids and texts must be the same."

        elif ids is None:
            ids = [md5(text.lower().encode("utf-8")).hexdigest() for text in texts]

        embeds = self._embed_doc(texts)

        params = [
            (
                json.dumps(embed),
                json.dumps(metadata) if metadata else "",
                doc_id,
                json.dumps(metadata) if metadata else "",
            )
            for embed, metadata, doc_id, metadata in zip(
                embeds, metadatas, ids, metadatas
            )
        ]

        query = (
            f"INSERT INTO {self.collection_name} "
            "(embedding, metadata, doc_id) "
            "VALUES (%s,%s,%s) "
            "ON CONFLICT(doc_id) "
            "DO UPDATE "
            "SET metadata=%s"
        )

        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.executemany(query, params)
        except Exception as e:
            msg = f"Error occured during upsert documents due to {type(e)} {str(e)}"
            conn.rollback()
        else:
            conn.commit()
            msg = f"Upsert successful"
        finally:
            print(msg)
            conn.close()
            return ids

    def upsert_parallel(
        self, texts, metadatas, ids, batch_size=32, workers=10, **kwargs
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

        id_batches = [ids[i : i + batch_size] for i in range(0, len(texts), batch_size)]

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

    def search(self, query, k=10, distance="cosine", **kwargs):
        self.distance = distance.lower()
        embeded_query = json.dumps(self.embedding.embed_query(query))
        search_query = (
            f"SELECT doc_id, metadata, 1-(embedding {DISTANCE[self.distance]} %s) FROM {self.collection_name} "
            f"ORDER BY embedding {DISTANCE[self.distance]} %s;"
        )
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(search_query, (embeded_query, embeded_query))
        except Exception as e:
            print(f"Search failed due to {type(e)} {str(e)}")
            conn.rollback()
        else:
            _result = cur.fetchall()
            result = [
                {"doc_id": _r[0], "metadata": _r[1], "score": _r[2]} for _r in _result
            ]
        finally:
            conn.close()
            return result

    def _type_cast(self, keys):
        tmps = []
        for key in keys:
            tmps.append( "jsonb_typeof(cmetadata -> %s)")

        query = "SELECT " + ','.join(tmps) + f" FROM {self.collection_name} LIMIT 1"
        print(query)
        params = keys
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(query, tuple(params))
        except Exception as e:
            print(f"Detect type failed due to {type(e)} {str(e)}")
            conn.rollback()
        else:
            types = [d for d in cur.fetchall()[0]]
        finally:
            conn.close()
            return {k:t for k, t in zip(keys, types)}
        

    def delete(self, ids=None, filters=None, **kwargs):
        """
        filters = {"meta_key": {"operator_type": "operator", "value": "values"}}
        """
        
        assert not(ids is not None and filters is not None), "Provide only one of ids or filters, not both"
        
        if ids is not None:
            format_str = ",".join(["%s"] * len(ids))
            query = f"DELETE FROM {self.collection_name} WHERE doc_id IN (%s)"
            params = ids

        elif filters is not None:
            query = f"DELETE FROM {self.collection_name} WHERE "
            filter_tmp = []
            format_len = 0
            types = self._type_cast(list(filters.keys()))
            params = []
            for k, v in filters.items():
                print(k, v)
                tmp_str = f"CAST(cmetadata->'{k}' AS {types[k]}) {OPERATORS[v['operator_type']]} (%s)"
                filter_tmp.append(tmp_str)
                format_len += len(v['value'])
                params.extend(v['value'])
            filter_query = ' AND '.join(filter_tmp)
            format_str = ",".join(["%s"]*format_len)
            query += filter_query

        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(query % format_str, params)
        except Exception as e:
            msg = f"Delete failed due to {type(e)} {str(e)}"
            conn.rollback()
        else:
            msg = "Delete by id successful"
            conn.commit()
        finally:
            print(msg)
            conn.close()

    def scroll(self):
        pass
