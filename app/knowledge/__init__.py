"""Knowledge retrieval package: base protocol + baseline backends."""
from app.knowledge.base import BaseKnowledgeRetriever, KnowledgeHit
from app.knowledge.inmemory import InMemoryKnowledgeRetriever

__all__ = [
    "BaseKnowledgeRetriever",
    "InMemoryKnowledgeRetriever",
    "KnowledgeHit",
]
