from .vectordbinterface import DocumentManager
from langchain_core.documents import Document
from typing import List, Union, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from hashlib import md5
import os, time, uuid, json, enum
import contextlib
from langchain_core.retrievers import BaseRetriever, LangSmithRetrieverParams
from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict, Field, model_validator
import sqlalchemy
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
    ClassVar,
)

from collections.abc import Collection

COMPARISONS_TO_NATIVE = {
    "$eq": "==",
    "$ne": "!=",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

SPECIAL_CASED_OPERATORS = {
    "$in",
    "$nin",
    "$between",
    "$exists",
}

TEXT_OPERATORS = {
    "$like",
    "$ilike",
}

LOGICAL_OPERATORS = {"$and", "$or", "$not"}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(TEXT_OPERATORS)
    .union(LOGICAL_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
)

Base = declarative_base()
_classes: Any = None


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE


def _get_embedding_collection_store(vector_dimension: Optional[int] = None) -> Any:
    global _classes
    if _classes is not None:
        return _classes

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
        connection=None,
        host=None,
        port=None,
        username=None,
        user=None,
        password=None,
        passwd=None,
        dbname=None,
        db=None,
    ):
        if connection is not None:
            self.connection_str = connection

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
            self.connection_str = f"postgresql+psycopg://{self.userName}:{self.passWord}@{self.host}:{self.port}/{self.dbName}"

        self._engine = create_engine(url=self.connection_str, **({}))
        self.session_maker: scoped_session
        self.session_maker = scoped_session(sessionmaker(bind=self._engine))
        self.collection_metadata = None
        self._check_extension()
        EmbeddingStore, CollectionStore = _get_embedding_collection_store()
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        with self._make_sync_session() as session:
            Base.metadata.create_all(session.get_bind())
            session.commit()

    def _connect(self):
        return self._engine.connect()

    def list_indexes(self):
        query = """
        SELECT name FROM langchain_pg_collection;
        """

        try:
            with self._connect() as conn:
                exe = conn.execute(sqlalchemy.text(query))

        except Exception as e:
            msg = f"List collection failed due to {type(e)} {str(e)}"
        else:
            collections = exe.fetchall()
            msg = ""
        finally:
            conn.close()
            if msg:
                print(msg)
            return [col[0] for col in collections]

    def delete_index(self, collection_name):
        try:
            with self._make_sync_session() as session:
                collection = self.CollectionStore.get_by_name(session, collection_name)

                if collection is None:
                    print(f"Collection {collection_name} does not exist")
                    return True

                stmt = delete(self.EmbeddingStore)
                filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
                stmt = stmt.filter(*filter_by)

                session.execute(stmt)
                session.commit()

        except Exception as e:
            print(
                f"Deleting data from langchain_pg_embedding failed due to {type(e)} {str(e)}"
            )
            return False
        else:
            try:
                with self._connect() as conn:
                    query = sqlalchemy.text(
                        f"DELETE FROM langchain_pg_collection WHERE name = '{collection_name}'"
                    )
                    conn.execute(query)
                    conn.commit()
            except Exception as e:
                print(
                    f"Delete collection information row failed due to {type(e)} {str(e)}"
                )
            else:
                return True

    def _check_extension(self):
        query = "SELECT * FROM pg_extension;"
        create_ext_query = "CREATE EXTENSION vector;"

        with self._engine.connect() as conn:
            stmt = sqlalchemy.text(query)
            extensions = str(conn.execute(stmt).fetchall())

            if "vector" not in extensions:
                conn.execute(sqlalchemy.text(create_ext_query))
                conn.commit()

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

        EmbeddingStore, CollectionStore = _get_embedding_collection_store(
            self.dimension
        )

        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        try:
            with self._make_sync_session() as session:
                self.CollectionStore.get_or_create(
                    session, self.collection_name, cmetadata=self.collection_metadata
                )
                session.commit()
        except Exception as e:
            print(
                f"Creating new collection {self.collection_name} failed due to {type(e)} {str(e)}"
            )
            return False
        else:
            return pgVectorDocumentManager(
                embedding=embedding,
                connection_info=self.connection_str,
                collection_name=collection_name,
            )

    def get_index(self, embedding, collection_name):
        return pgVectorDocumentManager(
            embedding=embedding,
            connection_info=self.connection_str,
            collection_name=collection_name,
        )


class pgVectorDocumentManager(DocumentManager):
    def __init__(
        self, embedding, connection_info=None, collection_name=None, distance="cosine"
    ):
        if isinstance(connection_info, str):
            self.connection_info = connection_info
        elif isinstance(connection_info, dict):
            self.connection_info = self._make_conn_string(connection_info)
        self._engine = create_engine(url=self.connection_info, **({}))
        self.session_maker: scoped_session
        self.session_maker = scoped_session(sessionmaker(bind=self._engine))
        self.collection_metadata = None
        self.collection_name = collection_name
        self.EmbeddingStore, self.CollectionStore = _get_embedding_collection_store()
        with self._make_sync_session() as session:
            self.CollectionStore.get_by_name(session, self.collection_name)
        self.embeddings = embedding
        self.distance = distance.lower()
        self._distance_strategy = DEFAULT_DISTANCE_STRATEGY

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self.EmbeddingStore.embedding.cosine_distance
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self.EmbeddingStore.embedding.max_inner_product
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def _make_conn_string(self, connection_info):
        self.userName = connection_info.get("user", "langchain")
        self.passWord = connection_info.get("password", "langchain")
        self.host = connection_info.get("host", "localhost")
        self.port = connection_info.get("port", 6024)
        self.dbName = connection_info.get("dbname", "langchain")
        connection_str = f"postgresql+psycopg://{self.userName}:{self.passWord}@{self.host}:{self.port}/{self.dbName}"
        return connection_str

    def _embed_doc(self, texts) -> List[float]:
        embedded = self.embeddings.embed_documents(texts)
        return embedded

    def upsert(self, texts, metadatas=None, ids=None, **kwargs):
        if ids is not None:
            assert len(ids) == len(
                texts
            ), "The length of ids and texts must be the same."

        elif ids is None:
            ids = [md5(text.lower().encode("utf-8")).hexdigest() for text in texts]

        embeds = self._embed_doc(texts)

        with self._make_sync_session() as session:
            collection = self.CollectionStore.get_by_name(session, self.collection_name)
            collection_id = collection.uuid
            try:
                data = [
                    {
                        "id": doc_id,
                        "collection_id": collection_id,
                        "embedding": embed,
                        "document": text,
                        "cmetadata": metadata,
                    }
                    for embed, text, metadata, doc_id in zip(
                        embeds, texts, metadatas, ids
                    )
                ]

                stmt = insert(self.EmbeddingStore).values(data)
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "embedding": stmt.excluded.embedding,
                        "document": stmt.excluded.document,
                        "cmetadata": stmt.excluded.cmetadata,
                    },
                )
                session.execute(on_conflict_stmt)
                session.commit()
            except Exception as e:
                print(f"Upsert failed due to {type(e)} {str(e)}")
            finally:
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

    def search(self, query, k=10, distance="cosine", filter=None, **kwargs):
        self.distance = distance.lower()
        embeded_query = self.embeddings.embed_query(query)

        results = self.__query_collection(embeded_query, k, filter)

        if distance == "cosine":
            self._distance_strategy = DistanceStrategy.COSINE
        elif distance == "inner":
            self._distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT
        elif distance == "l2":
            self._distance_strategy = DistanceStrategy.EUCLIDEAN

        docs = [
            {
                "content": result.EmbeddingStore.document,
                "metadata": result.EmbeddingStore.cmetadata,
                "embedding": (
                    result.EmbeddingStore.embedding
                    if kwargs.get("include_embedding", False) == True
                    else None
                ),
                "score": result.distance if distance == "l2" else 1 - result.distance,
            }
            for result in results
        ]

        return docs

    def _create_filter_clause(self, filters: Any) -> Any:
        """Convert LangChain IR filter representation to matching SQLAlchemy clauses.

        At the top level, we still don't know if we're working with a field
        or an operator for the keys. After we've determined that we can
        call the appropriate logic to handle filter creation.

        Args:
            filters: Dictionary of filters to apply to the query.

        Returns:
            SQLAlchemy clause to apply to the query.
        """
        if isinstance(filters, dict):
            if len(filters) == 1:
                # The only operators allowed at the top level are $AND, $OR, and $NOT
                # First check if an operator or a field
                key, value = list(filters.items())[0]
                if key.startswith("$"):
                    # Then it's an operator
                    if key.lower() not in ["$and", "$or", "$not"]:
                        raise ValueError(
                            f"Invalid filter condition. Expected $and, $or or $not "
                            f"but got: {key}"
                        )
                else:
                    # Then it's a field
                    return self._handle_field_filter(key, filters[key])

                if key.lower() == "$and":
                    if not isinstance(value, list):
                        raise ValueError(
                            f"Expected a list, but got {type(value)} for value: {value}"
                        )
                    and_ = [self._create_filter_clause(el) for el in value]
                    if len(and_) > 1:
                        return sqlalchemy.and_(*and_)
                    elif len(and_) == 1:
                        return and_[0]
                    else:
                        raise ValueError(
                            "Invalid filter condition. Expected a dictionary "
                            "but got an empty dictionary"
                        )
                elif key.lower() == "$or":
                    if not isinstance(value, list):
                        raise ValueError(
                            f"Expected a list, but got {type(value)} for value: {value}"
                        )
                    or_ = [self._create_filter_clause(el) for el in value]
                    if len(or_) > 1:
                        return sqlalchemy.or_(*or_)
                    elif len(or_) == 1:
                        return or_[0]
                    else:
                        raise ValueError(
                            "Invalid filter condition. Expected a dictionary "
                            "but got an empty dictionary"
                        )
                elif key.lower() == "$not":
                    if isinstance(value, list):
                        not_conditions = [
                            self._create_filter_clause(item) for item in value
                        ]
                        not_ = sqlalchemy.and_(
                            *[
                                sqlalchemy.not_(condition)
                                for condition in not_conditions
                            ]
                        )
                        return not_
                    elif isinstance(value, dict):
                        not_ = self._create_filter_clause(value)
                        return sqlalchemy.not_(not_)
                    else:
                        raise ValueError(
                            f"Invalid filter condition. Expected a dictionary "
                            f"or a list but got: {type(value)}"
                        )
                else:
                    raise ValueError(
                        f"Invalid filter condition. Expected $and, $or or $not "
                        f"but got: {key}"
                    )
            elif len(filters) > 1:
                # Then all keys have to be fields (they cannot be operators)
                for key in filters.keys():
                    if key.startswith("$"):
                        raise ValueError(
                            f"Invalid filter condition. Expected a field but got: {key}"
                        )
                # These should all be fields and combined using an $and operator
                and_ = [self._handle_field_filter(k, v) for k, v in filters.items()]
                if len(and_) > 1:
                    return sqlalchemy.and_(*and_)
                elif len(and_) == 1:
                    return and_[0]
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            else:
                raise ValueError("Got an empty dictionary for filters.")
        else:
            raise ValueError(
                f"Invalid type: Expected a dictionary but got type: {type(filters)}"
            )

    def _handle_field_filter(
        self,
        field: str,
        value: Any,
    ) -> SQLColumnExpression:
        """Create a filter for a specific field.

        Args:
            field: name of field
            value: value to filter
                If provided as is then this will be an equality filter
                If provided as a dictionary then this will be a filter, the key
                will be the operator and the value will be the value to filter by

        Returns:
            sqlalchemy expression
        """
        if not isinstance(field, str):
            raise ValueError(
                f"field should be a string but got: {type(field)} with value: {field}"
            )

        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: "
                f"{field}"
            )

        # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
        if not field.isidentifier():
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

        if isinstance(value, dict):
            # This is a filter specification
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which "
                    "is a dictionary with a single key that corresponds to an operator "
                    f"but got a dictionary with {len(value)} keys. The first few "
                    f"keys are: {list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            # Verify that that operator is an operator
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. "
                    f"Expected one of {SUPPORTED_OPERATORS}"
                )
        else:  # Then we assume an equality operator
            operator = "$eq"
            filter_value = value

        if operator in COMPARISONS_TO_NATIVE:
            # Then we implement an equality filter
            # native is trusted input
            native = COMPARISONS_TO_NATIVE[operator]
            return func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                cast(f"$.{field} {native} $value", JSONPATH),
                cast({"value": filter_value}, JSONB),
            )
        elif operator == "$between":
            # Use AND with two comparisons
            low, high = filter_value

            lower_bound = func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                cast(f"$.{field} >= $value", JSONPATH),
                cast({"value": low}, JSONB),
            )
            upper_bound = func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                cast(f"$.{field} <= $value", JSONPATH),
                cast({"value": high}, JSONB),
            )
            return sqlalchemy.and_(lower_bound, upper_bound)
        elif operator in {"$in", "$nin", "$like", "$ilike"}:
            # We'll do force coercion to text
            if operator in {"$in", "$nin"}:
                for val in filter_value:
                    if not isinstance(val, (str, int, float)):
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

                    if isinstance(val, bool):  # b/c bool is an instance of int
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

            queried_field = self.EmbeddingStore.cmetadata[field].astext

            if operator in {"$in"}:
                return queried_field.in_([str(val) for val in filter_value])
            elif operator in {"$nin"}:
                return ~queried_field.in_([str(val) for val in filter_value])
            elif operator in {"$like"}:
                return queried_field.like(filter_value)
            elif operator in {"$ilike"}:
                return queried_field.ilike(filter_value)
            else:
                raise NotImplementedError()
        elif operator == "$exists":
            if not isinstance(filter_value, bool):
                raise ValueError(
                    "Expected a boolean value for $exists "
                    f"operator, but got: {filter_value}"
                )
            condition = func.jsonb_exists(
                self.EmbeddingStore.cmetadata,
                field,
            )
            return condition if filter_value else ~condition
        else:
            raise NotImplementedError()

    def __query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> Sequence[Any]:
        """Query the collection."""
        with self._make_sync_session() as session:  # type: ignore[arg-type]
            collection = self.CollectionStore.get_by_name(
                session, name=self.collection_name
            )
            if not collection:
                raise ValueError("Collection not found")

            filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
            if filter:
                filter_clauses = self._create_filter_clause(filter)
                if filter_clauses is not None:
                    filter_by.append(filter_clauses)

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),
                )
                .filter(*filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )

        return results

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[Session, None, None]:
        """Make an async session."""
        with self.session_maker() as session:
            yield typing_cast(Session, session)

    def delete(self, ids=None, filter=None, **kwargs):
        try:
            with self._make_sync_session() as session:  # type: ignore[arg-type]
                collection = self.CollectionStore.get_by_name(
                    session, name=self.collection_name
                )
                if not collection:
                    raise ValueError("Collection not found")
                stmt = delete(self.EmbeddingStore)
                if ids is not None:
                    stmt = stmt.where(self.EmbeddingStore.id.in_(ids))
                    session.execute(stmt)

                elif filter:
                    filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
                    filter_clauses = self._create_filter_clause(filter)
                    if filter_clauses is not None:
                        filter_by.append(filter_clauses)
                    stmt = stmt.where(filter_clauses)
                    session.execute(stmt)
                session.commit()
        except Exception as e:
            msg = f"Delete failed due to {type(e)} {str(e)}"
            return False
        else:
            msg = "Delete done successfully"
            return True
        finally:
            print(msg)

    def _get_retriever_tags(self) -> list[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        if self.embeddings:
            tags.append(self.embeddings.__class__.__name__)
        return tags

    def as_retriever(self, **kwargs):
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return pgVectorRetriever(vectorstore=self, tags=tags, **kwargs)

    def scroll(self, ids=None, filter=None, k=10, **kwargs):
        with self._make_sync_session() as session:  # type: ignore[arg-type]
            collection = self.CollectionStore.get_by_name(
                session, name=self.collection_name
            )
            if not collection:
                raise ValueError("Collection not found")

            filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
            if ids:
                filter_by.append(self.EmbeddingStore.id.in_(ids))

            elif filter:
                filter_clauses = self._create_filter_clause(filter)
                if filter_clauses is not None:
                    filter_by.append(filter_clauses)

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                )
                .filter(*filter_by)
                .limit(k)
                .all()
            )

        docs = [
            {
                "content": result.document,
                "metadata": result.cmetadata,
                "embedding": (
                    result.embedding if kwargs.get("include_embedding", False) else None
                ),
            }
            for result in results
        ]

        return docs


class pgVectorRetriever(BaseRetriever):
    vectorstore: pgVectorDocumentManager
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: dict) -> Any:
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            msg = (
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
            raise ValueError(msg)
        if search_type == "similarity_score_threshold":
            score_threshold = values.get("search_kwargs", {}).get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                msg = (
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
                raise ValueError(msg)
        return values

    def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
        """Get standard params for tracing."""

        _kwargs = self.search_kwargs | kwargs

        ls_params = super()._get_ls_params(**_kwargs)
        ls_params["ls_vector_store_provider"] = self.vectorstore.__class__.__name__

        if self.vectorstore.embeddings:
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embeddings.__class__.__name__
            )
        elif hasattr(self.vectorstore, "embedding") and isinstance(
            self.vectorstore.embedding, Embeddings
        ):
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embedding.__class__.__name__
            )

        return ls_params

    def _get_relevant_documents(
        self, query: str, *, run_manager, **kwargs: Any
    ) -> list[Document]:
        _kwargs = self.search_kwargs | kwargs
        print(f"_kwargs: {_kwargs}")
        if self.search_type == "similarity":
            docs = self.vectorstore.search(query, **_kwargs)
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            List of IDs of the added texts.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.vectorstore.upsert(texts, metadatas, **kwargs)
