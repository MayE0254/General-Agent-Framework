"""Milvus vector-store adapter for the knowledge pipeline.

Wraps ``pymilvus.MilvusClient`` and exposes the three operations the
ingestion/retrieval layers need: ``ensure_collection`` (create-if-missing),
``upsert_chunks`` (write embedding rows keyed by a stable chunk id) and
``search`` (cosine top-k).  All connection details come from
``[knowledge.milvus]`` and point at the separately deployed Milvus 2.6 host.

Row identity: the primary key is derived from the stable ``chunk_id`` (see
``_stable_pk``) rather than from a running counter, so re-ingesting a file
overwrites its rows in place instead of appending duplicates.

Embedding dimension must match the configured local embedding model
(bge-base-zh-v1.5 => 768).  See ``knowledge.embedding_model``.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.settings import MilvusSettings

from app.knowledge.base import KnowledgeHit

# Default dimension for BAAI/bge-base-zh-v1.5; overridable via DIMENSION const.
EMBEDDING_DIM = 768

# Field names shared with the Milvus collection schema.
_PK_FIELD = "pk"
_CHUNK_FIELD = "chunk_id"
_VEC_FIELD = "embedding"
_TEXT_FIELD = "text"
_TITLE_FIELD = "title"
_SOURCE_FIELD = "source"
_FILE_NO_FIELD = "file_no"
_DOC_TYPE_FIELD = "doc_type"
_SECTION_FIELD = "section"
_PARENT_ID_FIELD = "parent_id"
_PARENT_TEXT_FIELD = "parent_text"

# Every scalar field the retrieval layer reads back.  Kept as one list so the
# schema check and the search output_fields can never drift apart.
_SCALAR_FIELDS = (
    _TITLE_FIELD,
    _SOURCE_FIELD,
    _FILE_NO_FIELD,
    _DOC_TYPE_FIELD,
    _SECTION_FIELD,
    _PARENT_ID_FIELD,
    _PARENT_TEXT_FIELD,
)
_EXPECTED_FIELDS = {_PK_FIELD, _CHUNK_FIELD, _VEC_FIELD, _TEXT_FIELD, *_SCALAR_FIELDS}


def _stable_pk(chunk_id: str) -> int:
    """Deterministic 60-bit primary key derived from the stable chunk id.

    Milvus resolves ``upsert`` by primary key, so idempotence hinges on the pk
    being a pure function of the chunk.  A running counter (the previous
    scheme) handed the same chunk a *new* pk on every re-run, so losing the
    progress marker duplicated the whole corpus instead of overwriting it.
    60 bits keeps the value inside a signed int64 and the collision odds for a
    ~2.5e4-chunk corpus negligible.
    """
    return int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


class MilvusStore:
    """Thin client over pymilvus for the knowledge_chunks collection."""

    def __init__(self, settings: MilvusSettings) -> None:
        self.settings = settings
        self._client = None
        self._loaded = False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._loaded = False

    def _client_impl(self):
        if self._client is None:
            from pymilvus import MilvusClient

            kwargs: dict[str, Any] = {"uri": f"http://{self.settings.host}:{self.settings.port}"}
            if self.settings.user:
                kwargs["user"] = self.settings.user
                kwargs["password"] = self.settings.password
            elif self.settings.password:
                kwargs["token"] = f"root:{self.settings.password}"
            self._client = MilvusClient(**kwargs)
        return self._client

    def _ensure_index(self) -> None:
        """Create an HNSW index on the vector field if none exists.

        Milvus requires an index before a collection can be loaded/searched
        (else load fails with ``index not found``).  Idempotent — safe to call
        on an already-indexed collection."""
        client = self._client_impl()
        try:
            indexes = client.list_indexes(collection_name=self.collection)
            if indexes and _VEC_FIELD in {i.get("field_name") for i in (indexes or [])}:
                return
        except Exception:
            return
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=_VEC_FIELD,
            index_type="HNSW",
            metric_type="COSINE",
            index_name="knowledge_vec_idx",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_index(collection_name=self.collection, index_params=index_params)

    def _ensure_loaded(self, *, timeout: float = 60.0) -> None:
        """Load the collection into query nodes and wait until it is queryable
        (Milvus load is async; searching too early raises
        ``collection not loaded``).  Also guarantees the schema is current and a
        vector index exists.  Only runs once per store instance."""
        if self._loaded:
            return
        import time

        client = self._client_impl()
        names = client.list_collections()
        if isinstance(names, list) and self.collection not in names:
            self._loaded = True
            return
        # Read paths reach the store here and nowhere else, so this is the one
        # place a stale collection can be caught before it serves a query.
        self.verify_schema()
        self._ensure_index()
        started = time.monotonic()
        # Idempotent — safe to call even if another worker loaded it.
        client.load_collection(collection_name=self.collection)
        # LoadState proto: 0=NotLoad, 1=Loading, 2=Loaded. get_load_state
        # returns a dict {"state": <int>}; >= 2 means queryable.
        while time.monotonic() - started < timeout:
            try:
                info = client.get_load_state(collection_name=self.collection)
                if int(info.get("state", 0)) >= 2:
                    break
            except Exception:
                pass
            time.sleep(2)
        self._loaded = True

    @property
    def collection(self) -> str:
        return self.settings.collection

    def ensure_collection(self, *, dim: int = EMBEDDING_DIM) -> bool:
        """Create the collection if missing; return True when it was missing.

        Built with an explicit schema so the scalar provenance fields
        (title/source/file_no/doc_type/section/parent_id/parent_text) exist for
        output_fields in search queries.  A shortcut ``create_collection`` would
        omit those fields and make provenance retrieval fail.
        """
        client = self._client_impl()
        existing = client.list_collections()
        if self.collection in existing:
            self.verify_schema()
            return False
        from pymilvus import CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema(name=_PK_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name=_CHUNK_FIELD, dtype=DataType.VARCHAR, max_length=40),
            FieldSchema(name=_VEC_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name=_TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=_TITLE_FIELD, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=_SOURCE_FIELD, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=_FILE_NO_FIELD, dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name=_DOC_TYPE_FIELD, dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name=_SECTION_FIELD, dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name=_PARENT_ID_FIELD, dtype=DataType.VARCHAR, max_length=40),
            FieldSchema(name=_PARENT_TEXT_FIELD, dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields)
        client.create_collection(
            collection_name=self.collection,
            schema=schema,
            consistency_level="Bounded",
        )
        self._ensure_index()
        return True

    def verify_schema(self) -> None:
        """Fail loudly when the live collection predates a schema change.

        Milvus cannot add fields to an existing collection, so after this module
        gains a field an old collection still exists but rejects the new rows.
        Without this check the failure surfaces much later as an opaque
        ``upsert``/``search`` error -- or, worse, as a query that silently
        returns nothing because the store is asked for a field the collection
        does not have.  The corpus in ``data/knowledge/raw`` is the source of
        truth, so the correct remedy is a rebuild and the message says so.

        Called by ``ensure_collection`` (write path) and ``_ensure_loaded``
        (read path), and directly by callers that must not degrade silently --
        an evaluation run, for instance, would otherwise measure a chain whose
        vector leg is dead without saying so.
        """
        client = self._client_impl()
        info = client.describe_collection(collection_name=self.collection)
        live = {f.get("name") for f in (info.get("fields") or [])}
        missing = _EXPECTED_FIELDS - live
        if missing:
            raise RuntimeError(
                f"Milvus collection '{self.collection}' is missing field(s) "
                f"{sorted(missing)}. Its schema predates the current code and "
                f"Milvus cannot add fields in place -- rebuild with: "
                f"python scripts/ingest_knowledge.py --reset"
            )

    def reset(self) -> None:
        """Drop the collection (schema + rows) for a clean rebuild.

        Destructive by design: the corpus in ``data/knowledge/raw`` is the
        source of truth, so dropping the vector store only costs a re-embed.
        Used by the ingest CLI's ``--reset``; the ingestion path itself never
        calls it.
        """
        client = self._client_impl()
        if self.collection in client.list_collections():
            client.drop_collection(collection_name=self.collection)
        self._loaded = False

    def upsert_chunks(
        self,
        rows: list[dict],
    ) -> int:
        """Insert/overwrite chunk rows. ``rows`` is a list of dicts with keys
        ``chunk_id, embedding(list[float]), text, title, source, file_no,
        doc_type, section, parent_id, parent_text``.  Returns the number
        written."""
        client = self._client_impl()
        if not rows:
            return 0
        data = []
        for r in rows:
            chunk_id = r["chunk_id"][:40]
            data.append(
                {
                    _PK_FIELD: _stable_pk(chunk_id),
                    _CHUNK_FIELD: chunk_id,
                    _VEC_FIELD: r["embedding"],
                    _TEXT_FIELD: r["text"][:65000],
                    _TITLE_FIELD: r["title"][:512],
                    _SOURCE_FIELD: r["source"][:512],
                    _FILE_NO_FIELD: r.get("file_no", "")[:16],
                    _DOC_TYPE_FIELD: r.get("doc_type", "")[:32],
                    _SECTION_FIELD: r.get("section", "")[:128],
                    _PARENT_ID_FIELD: r.get("parent_id", "")[:40],
                    _PARENT_TEXT_FIELD: r.get("parent_text", "")[:65000],
                }
            )
        client.upsert(collection_name=self.collection, data=data)
        return len(data)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        output_fields: list[str] | None = None,
    ) -> list[KnowledgeHit]:
        """Cosine search; returns KnowledgeHit list with vector score attached."""
        client = self._client_impl()
        self._ensure_loaded()
        fields = output_fields or [_CHUNK_FIELD, _TEXT_FIELD, *_SCALAR_FIELDS]
        res = client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            output_fields=fields,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )
        hits: list[KnowledgeHit] = []
        if not res:
            return hits
        for item in res[0]:
            entity = item.get("entity", {})
            hits.append(self._row_to_hit(entity, float(item.get("distance", 0.0))))
        return hits

    def _row_to_hit(self, entity: dict, score: float) -> KnowledgeHit:
        """Build one KnowledgeHit from a Milvus scalar row (shared by the
        vector-search and by-id lookup paths so both expose the same fields)."""
        return KnowledgeHit(
            id=str(entity.get(_CHUNK_FIELD, "")),
            title=str(entity.get(_TITLE_FIELD, "")),
            content=str(entity.get(_TEXT_FIELD, "")),
            score=round(score, 4),
            source=str(entity.get(_SOURCE_FIELD, "milvus")),
            parent_id=str(entity.get(_PARENT_ID_FIELD, "") or ""),
            parent_text=str(entity.get(_PARENT_TEXT_FIELD, "") or ""),
            metadata_payload={
                "vector_score": round(score, 4),
                "file_no": entity.get(_FILE_NO_FIELD, ""),
                "doc_type": entity.get(_DOC_TYPE_FIELD, ""),
                "section": entity.get(_SECTION_FIELD, ""),
                "parent_id": entity.get(_PARENT_ID_FIELD, ""),
            },
        )

    def count(self) -> int:
        """Accurate row count.

        ``get_collection_stats`` reports a ``row_count`` that Milvus refreshes
        lazily (only on flush/compaction), so right after an upsert it returns
        a stale value.  The ingestion pipeline seeds its primary-key counter
        from this number, so a stale value would make it re-use pks and
        silently overwrite already-indexed chunks.  A ``count(*)`` aggregation
        reflects the committed state; the cheap stats call stays as fallback.
        """
        client = self._client_impl()
        try:
            self._ensure_loaded()
            res = client.query(
                collection_name=self.collection,
                filter=f"{_PK_FIELD} >= 0",
                output_fields=["count(*)"],
            )
            if res:
                return int(res[0].get("count(*)", 0))
        except Exception:
            pass
        stats = client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0) or 0)

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Read rows for the given stable chunk ids (for BM25/rerank overlap)."""
        client = self._client_impl()
        out: list[dict] = []
        for cid in chunk_ids:
            res = client.query(
                collection_name=self.collection,
                filter=f'{_CHUNK_FIELD} == "{cid}"',
                output_fields=[_CHUNK_FIELD, _TEXT_FIELD, _TITLE_FIELD, _SOURCE_FIELD,
                               _FILE_NO_FIELD, _DOC_TYPE_FIELD, _SECTION_FIELD],
            )
            for r in res:
                out.append(r)
        return out
