import importlib
import sys
import types
from pathlib import Path

import numpy as np


def _load_memory_modules_without_optional_package_imports():
    """Load only the memory modules under test, avoiding unrelated package exports."""
    package_root = Path(__file__).resolve().parents[1] / "hello_agents"
    packages = {
        "_hello_agents_test": package_root,
        "_hello_agents_test.memory": package_root / "memory",
        "_hello_agents_test.memory.rag": package_root / "memory" / "rag",
        "_hello_agents_test.memory.storage": package_root / "memory" / "storage",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    embedding = types.ModuleType("_hello_agents_test.memory.embedding")
    embedding.get_text_embedder = lambda: None
    embedding.get_dimension = lambda default: default
    sys.modules[embedding.__name__] = embedding

    pipeline = importlib.import_module("_hello_agents_test.memory.rag.pipeline")
    qdrant_store = importlib.import_module("_hello_agents_test.memory.storage.qdrant_store")
    return pipeline, qdrant_store.QdrantVectorStore


pipeline, QdrantVectorStore = _load_memory_modules_without_optional_package_imports()


class FakeEmbedder:
    def __init__(self, max_batch_size=8):
        self.max_batch_size = max_batch_size
        self.batch_sizes = []

    def encode(self, texts):
        self.batch_sizes.append(len(texts))
        if len(texts) > self.max_batch_size:
            raise ValueError("batch exceeds provider limit")
        return [np.array([index + 1, index + 2, index + 3]) for index in range(len(texts))]


class FakeStore:
    def add_vectors(self, vectors, metadata, ids):
        self.vectors = vectors
        self.metadata = metadata
        self.ids = ids
        return True


def test_index_chunks_normalizes_numpy_vectors_after_batch_retry(monkeypatch):
    embedder = FakeEmbedder()
    store = FakeStore()
    monkeypatch.setattr(pipeline, "get_text_embedder", lambda: embedder)
    monkeypatch.setattr(pipeline, "get_dimension", lambda default: 3)
    monkeypatch.setattr("time.sleep", lambda _: None)

    chunks = [
        {"id": f"chunk-{index}", "content": f"content {index}"}
        for index in range(20)
    ]
    pipeline.index_chunks(store=store, chunks=chunks, batch_size=16)

    assert len(store.vectors) == len(chunks)
    assert len(store.metadata) == len(chunks)
    assert store.ids == [chunk["id"] for chunk in chunks]
    assert all(vector != [0.0, 0.0, 0.0] for vector in store.vectors)
    assert embedder.batch_sizes == [16, 8, 8, 4]


def test_index_chunks_default_batch_size_respects_dashscope_limit(monkeypatch):
    embedder = FakeEmbedder(max_batch_size=10)
    store = FakeStore()
    monkeypatch.setattr(pipeline, "get_text_embedder", lambda: embedder)
    monkeypatch.setattr(pipeline, "get_dimension", lambda default: 3)

    chunks = [
        {"id": f"chunk-{index}", "content": f"content {index}"}
        for index in range(22)
    ]
    pipeline.index_chunks(store=store, chunks=chunks)

    assert embedder.batch_sizes == [10, 10, 2]
    assert len(store.vectors) == len(chunks)
    assert store.ids == [chunk["id"] for chunk in chunks]


def test_qdrant_store_rejects_mismatched_vector_metadata_counts():
    store = object.__new__(QdrantVectorStore)
    store.vector_size = 3
    store.collection_name = "test"
    store.client = None

    assert store.add_vectors(vectors=[[1.0, 2.0, 3.0]], metadata=[], ids=["chunk-1"]) is False
