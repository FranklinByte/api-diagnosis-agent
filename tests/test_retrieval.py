from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from api_diagnosis.retrieval import (
    KeywordRetriever,
    load_changelog_documents,
    load_knowledge_documents,
    load_openapi_documents,
)


ROOT = Path(__file__).resolve().parents[1]
OPENAPI_PATH = ROOT / "specs" / "mock-api.json"
CHANGELOG_PATH = ROOT / "data" / "changelog.json"


def test_loaders_create_citable_documents_with_flat_metadata():
    openapi_documents = load_openapi_documents(OPENAPI_PATH)
    changelog_documents = load_changelog_documents(CHANGELOG_PATH)

    assert openapi_documents
    assert changelog_documents
    operation = openapi_documents[0]
    change = changelog_documents[0]
    assert isinstance(operation, Document)
    assert operation.metadata["document_type"] == "openapi"
    assert operation.metadata["method"] == "POST"
    assert operation.metadata["path"] == "/weather/forecast"
    assert operation.metadata["version"] == "v2"
    assert change.metadata["document_type"] == "changelog"
    assert change.metadata["changelog_id"] == "weather-v2-001"
    assert change.metadata["breaking"] is True


def test_keyword_search_ranks_endpoint_and_supports_metadata_filters():
    retriever = KeywordRetriever.from_paths(OPENAPI_PATH, CHANGELOG_PATH, default_k=5)

    results = retriever.search(
        "POST /weather/forecast X-API-Version",
        metadata_filters={"document_type": "openapi", "method": "post"},
    )

    assert len(results) == 1
    assert results[0].metadata["endpoint"] == "POST /weather/forecast"
    assert "X-API-Version" in results[0].page_content


def test_keyword_search_can_retrieve_changelog_evidence():
    retriever = KeywordRetriever.from_paths(OPENAPI_PATH, CHANGELOG_PATH)

    results = retriever.search(
        "breaking migration v1 v2",
        metadata_filters={"document_type": "changelog", "breaking": True},
    )

    assert [document.metadata["changelog_id"] for document in results] == ["weather-v2-001"]
    assert "Send X-API-Version: v2" in results[0].page_content


def test_langchain_invoke_uses_default_k_and_default_filters():
    retriever = KeywordRetriever.from_paths(
        OPENAPI_PATH,
        CHANGELOG_PATH,
        default_k=1,
        metadata_filters={"document_type": "changelog"},
    )

    results = retriever.invoke("API version")

    assert len(results) == 1
    assert results[0].metadata["document_type"] == "changelog"


def test_blank_or_unknown_queries_do_not_return_arbitrary_documents():
    retriever = KeywordRetriever.from_paths(OPENAPI_PATH, CHANGELOG_PATH)

    assert retriever.search("   ") == []
    assert retriever.search("term-that-does-not-exist") == []


def test_search_validates_query_and_limit():
    retriever = KeywordRetriever.from_paths(OPENAPI_PATH, CHANGELOG_PATH)

    with pytest.raises(TypeError):
        retriever.search(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        retriever.search("weather", k=0)
