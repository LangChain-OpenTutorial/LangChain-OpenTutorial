from vectordbinterface import DocumentManager
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time, uuid, json
import contextlib

import sqlalchemy, psycopg2
from sqlalchemy import (
    SQLColumnExpression,
    cast,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB, JSONPATH, UUID, insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
    scoped_session,
    sessionmaker,
)
from typing import (
    cast as typing_cast,
)

from pgvector.sqlalchemy import Vector

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

TYPE_CAST = {
    "number": "numeric",
    "string": "text",
}

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
    "$in": "IN",
    "$nin": "NOT IN",
    "$between": "BETWEEN",
    "$exists": "EXISTS",
    "$like": "LIKE",
    "$ilike": "IN LIKE",
    "$and": "AND",
    "$or": "OR",
    "$not": "NOT",
}

Base = declarative_base()

def _get_embedding_collection_store(vector_dimension: Optional[int] = None) -> Any:

    class CollectionStore(Base):
        """Collection store."""

        __tablename__ = "langchain_pg_collection"

        uuid = sqlalchemy.Column(
            UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
        )
        name = sqlalchemy.Column(sqlalchemy.String, nullable=False, unique=True)
        cmetadata = sqlalchemy.Column(JSON)

        embeddings = relationship(
            "EmbeddingStore",
            back_populates="collection",
            passive_deletes=True,
        )

        @classmethod
        def get_by_name(
            cls, session: Session, name: str
        ) -> Optional["CollectionStore"]:
            return (
                session.query(cls)
                .filter(typing_cast(sqlalchemy.Column, cls.name) == name)
                .first()
            )

        @classmethod
        def get_or_create(
            cls,
            session: Session,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """Get or create a collection.
            Returns:
                 Where the bool is True if the collection was created.
            """  # noqa: E501
            created = False
            collection = cls.get_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            session.commit()
            created = True
            return collection, created

    class EmbeddingStore(Base):
        """Embedding store."""

        __tablename__ = "langchain_pg_embedding"

        id = sqlalchemy.Column(
            sqlalchemy.String, nullable=True, primary_key=True, index=True, unique=True
        )

        collection_id = sqlalchemy.Column(
            UUID(as_uuid=True),
            sqlalchemy.ForeignKey(
                f"{CollectionStore.__tablename__}.uuid",
                ondelete="CASCADE",
            ),
        )
        collection = relationship(CollectionStore, back_populates="embeddings")

        embedding: Vector = sqlalchemy.Column(Vector(vector_dimension))
        document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        cmetadata = sqlalchemy.Column(JSONB, nullable=True)

        __table_args__ = (
            sqlalchemy.Index(
                "ix_cmetadata_gin",
                "cmetadata",
                postgresql_using="gin",
                postgresql_ops={"cmetadata": "jsonb_path_ops"},
            ),
        )

    _classes = (EmbeddingStore, CollectionStore)

    return _classes

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
        connection = f"postgresql+psycopg://{self.userName}:{self.passWord}@{self.host}:{self.port}/{self.dbName}"
        self._engine = create_engine(url=connection, **({}))
        self.session_maker: scoped_session
        self.session_maker = scoped_session(sessionmaker(bind=self._engine))
        self.collection_metadata = None

    # def create_collection(self) -> None:
    #     with self._make_sync_session() as session:
    #         self.CollectionStore.get_or_create(
    #             session, self.collection_name, cmetadata=self.collection_metadata
    #         )
    #         session.commit()

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

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[Session, None, None]:
        """Make an async session."""
        with self.session_maker() as session:
            yield typing_cast(Session, session)

    def create_index(self, collection_name, embedding=None, dimension=None):
        assert (
            embedding is not None or dimension is not None
        ), "One of embedding or dimension must be provided"
        self.collection_name = collection_name
        if dimension is None:
            self.dimension = len(embedding.embed_query("foo"))

        self._check_extension()
        EmbeddingStore, CollectionStore = _get_embedding_collection_store(self.dimension)

        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        try:
            with self._make_sync_session() as session:
                self.CollectionStore.get_or_create(
                    session, self.collection_name, cmetadata=self.collection_metadata
                )
                session.commit()
        except Exception as e:
            print(f"Creating new collection {self.collection_name} failed due to {type(e)} {str(e)}")
        else:
            return self.CollectionStore
        
    def get_index(self, collection_name, embedding):
        self.embedding = embedding
        EmbeddingStore, CollectionStore = _get_embedding_collection_store()
        self.collection_name = collection_name
        self.CollectionStore = CollectionStore
        try:
            with self._make_sync_session() as session:
                self.CollectionStore.get_or_create(
                    session, self.collection_name, cmetadata=None
                )
                session.commit()
        except Exception as e:
            print(f"Creating new collection {self.collection_name} failed due to {type(e)} {str(e)}")
        else:
            return self.CollectionStore



class pgVectorDocumentManager(DocumentManager):
    def __init__(self, embedding, connection_info=None, collection_name=None, distance="cosine"):
        self.connection_info = connection_info
        self._engine = create_engine(url=self._make_conn_string(), **({}))
        self.session_maker: scoped_session
        self.session_maker = scoped_session(sessionmaker(bind=self._engine))
        self.collection_metadata = None
        self.collection_name = collection_name
        EmbeddingStore, CollectionStore = _get_embedding_collection_store()
        self.embedding = embedding
        self.distance = distance.lower()

    def _make_conn_string(self):
        self.userName = self.connection_info.get('user', 'langchain')
        self.passWord = self.connection_info.get('password', 'langchain')
        self.host = self.connection_info.get('host', 'localhost')
        self.port = self.connection_info.get('port', 6024)
        self.dbName = self.connection_info.get('dbname', 'langchain')
        connection_str = f"postgresql+psycopg://{self.userName}:{self.passWord}@{self.host}:{self.port}/{self.dbName}"
        return connection_str

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
            tmps.append("jsonb_typeof(cmetadata -> %s)")

        query = "SELECT " + ",".join(tmps) + f" FROM {self.collection_name} LIMIT 1"
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
            return {k: t for k, t in zip(keys, types)}

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[Session, None, None]:
        """Make an async session."""
        with self.session_maker() as session:
            yield typing_cast(Session, session)

    def delete(self, ids=None, filters=None, **kwargs):
        """
        filters = {"meta_key": {"operator_type": "operator", "value": "values"}}
        """

        assert not (
            ids is not None and filters is not None
        ), "Provide only one of ids or filters, not both"

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
                tmp_str = f"CAST(cmetadata->'{k}' AS {TYPE_CAST[types[k]]}) {OPERATORS[v['operator_type']]} (%s)"
                filter_tmp.append(tmp_str)
                format_len += len(v["value"])
                params.extend(v["value"])
            filter_query = " AND ".join(filter_tmp)
            format_str = ",".join(["%s"] * format_len)
            query += filter_query
            print(query)

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
