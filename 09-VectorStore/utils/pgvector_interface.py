from .vectordbinterface import DocumentManager
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time, uuid
import contextlib
import psycopg
import sqlalchemy
from sqlalchemy import SQLColumnExpression, cast, create_engine, delete, func, select
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
from typing import (
    cast as typing_cast,
)

Base = declarative_base()


def _create_vector_extension(conn: Connection) -> None:
    statement = sqlalchemy.text(
        "SELECT pg_advisory_xact_lock(1573678846307946496);"
        "CREATE EXTENSION IF NOT EXISTS vector;"
    )
    conn.execute(statement)
    conn.commit()


def _get_embedding_collection_store(vector_dimension: Optional[int] = None) -> Any:
    global _classes
    if _classes is not None:
        return _classes

    from pgvector.sqlalchemy import Vector  # type: ignore

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
        connection_str=None,
        host=None,
        port=None,
        username=None,
        user=None,
        password=None,
        passwd=None,
        dbname=None,
        db=None,
        collection_name=None,
        create_extension=True,
        dimension=None,
        pre_delete_collection=False,
    ):
        if connection_str is not None:
            self.connection_str = connection_str
        else:
            assert host is not None, "host is missing"
            assert port is not None, "port is missing"
            assert (
                username is not None or user is not None
            ), "username(or user) is missing"
            assert (
                password is not None or passwd is not None
            ), "password(or passwd) is missing"
            assert dbname is not None or db is not None, "dbname(or db) is missing"

            self.host = host
            self.port = port
            self.userName = username if username is not None else user
            self.passWord = password if password is not None else passwd
            self.dbName = dbname if dbname is not None else db

            self.connection_str = (
                "postgresql+psycopg://{userName}:{passWord}@{host}:{port}/{dbName}"
            )

        self._engine = create_engine(url=self.connection_str, **{})
        self.collection_name = collection_name
        self.use_jsonb = True
        self.create_extension = create_extension
        self.dimension = dimension
        self.pre_delete_collection = pre_delete_collection
        self.__post_init__()
        self.session_maker = scoped_session(sessionmaker(bind=self._engine))

    def __post_init__(self):
        if self.create_extension:
            self.create_vector_extension()

        EmbeddingStore, CollectionStore = _get_embedding_collection_store(
            self.dimension
        )

        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore

        self.create_tables_if_not_exists()
        self.create_collection()

    def create_tables_if_not_exists(self) -> None:
        with self._make_sync_session() as session:
            Base.metadata.create_all(session.get_bind())
            session.commit()

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with self._make_sync_session() as session:
            self.CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )
            session.commit()

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[Session, None, None]:
        """Make an async session."""
        if self.async_mode:
            raise ValueError(
                "Attempting to use a sync method in when async mode is turned on. "
                "Please use the corresponding async method instead."
            )
        with self.session_maker() as session:
            yield typing_cast(Session, session)

    def create_vector_extension(self) -> None:
        assert self._engine, "engine not found"
        try:
            with self._engine.connect() as conn:
                _create_vector_extension(conn)
        except Exception as e:
            raise Exception(f"Failed to create vector extension: {e}") from e

    def list_indexes(self):
        pass

    def delete_index(self, index_name):
        pass

    def create_index(self):
        pass

    def get_index(self, index_name: str) -> Dict:
        pass


class pgVectorDocumentManager(DocumentManager):
    def __init__(self, client):
        pass

    def get_index_info(self):
        pass

    def _embed_doc(self, texts) -> List[float]:
        pass

    def upsert(self):
        pass

    def upsert_parallel(self):
        pass

    def search(self, query, k=10, **kwargs):
        pass

    def delete(self, ids=None, filters=None, **kwargs):
        pass

    def scroll(self):
        pass
