"""Small deterministic retrieval layer for the V1 knowledge base.

The first RAG increment deliberately has no embedding model or vector store.
OpenAPI operations and changelog entries are converted into LangChain
``Document`` objects with stable metadata.  Retrieval then applies exact
metadata filters followed by a deterministic keyword score.  This keeps the
evidence source inspectable while leaving a narrow interface for a later
semantic retriever.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from .paths import project_root


PROJECT_ROOT = project_root()
DEFAULT_OPENAPI_PATH = PROJECT_ROOT / "specs" / "mock-api.json"
DEFAULT_CHANGELOG_PATH = PROJECT_ROOT / "data" / "changelog.json"

_OPERATION_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"}
)
_TOKEN_PATTERN = re.compile(r"[\w]+(?:[./:_-][\w]+)*", re.UNICODE)


class RetrievalDataError(ValueError):
    """Raised when a knowledge-base JSON file cannot be loaded safely."""


def _read_json(path: Path | str) -> Any:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Knowledge source not found: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetrievalDataError(f"Invalid JSON knowledge source: {source}") from exc


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _slug(value: Any) -> str:
    """Create one stable, human-readable API identifier from a title."""

    normalized = re.sub(r"[^a-z0-9]+", "-", _text(value).lower()).strip("-")
    if normalized.endswith("-api"):
        normalized = normalized[:-4]
    return normalized or "unknown-api"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _operation_content(
    *,
    title: str,
    version: str,
    path: str,
    method: str,
    operation: Mapping[str, Any],
) -> str:
    """Render an operation as searchable text while retaining raw evidence."""

    lines = [
        f"API: {title}",
        f"Version: {version}",
        f"Endpoint: {method} {path}",
    ]
    for key in ("operationId", "summary", "description"):
        value = _text(operation.get(key))
        if value:
            lines.append(f"{key}: {value}")
    parameters = operation.get("parameters")
    if parameters:
        lines.append(f"Parameters: {_json_text(parameters)}")
    request_body = operation.get("requestBody")
    if request_body:
        lines.append(f"Request body: {_json_text(request_body)}")
    responses = operation.get("responses")
    if responses:
        lines.append(f"Responses: {_json_text(responses)}")
    lines.append(f"OpenAPI operation: {_json_text(operation)}")
    return "\n".join(lines)


def load_openapi_documents(path: Path | str = DEFAULT_OPENAPI_PATH) -> list[Document]:
    """Load one :class:`Document` for every OpenAPI operation.

    Metadata is intentionally flat and JSON-compatible so it can be used for
    exact filters and later persisted by a vector store without transformation.
    """

    source = Path(path)
    document = _read_json(source)
    if not isinstance(document, Mapping):
        raise RetrievalDataError(f"OpenAPI document must be an object: {source}")

    info = document.get("info")
    if not isinstance(info, Mapping):
        raise RetrievalDataError(f"OpenAPI document is missing an info object: {source}")
    title = _text(info.get("title"), "Untitled API")
    version = _text(info.get("version"), "unknown")
    api_name = _slug(info.get("x-api-name") or title)
    paths = document.get("paths")
    if not isinstance(paths, Mapping):
        raise RetrievalDataError(f"OpenAPI document is missing a paths object: {source}")

    loaded: list[Document] = []
    resolved_source = str(source.resolve())
    for path_name, path_item in paths.items():
        if not isinstance(path_item, Mapping):
            continue
        path_text = _text(path_name)
        for method_name, operation in path_item.items():
            method = _text(method_name).upper()
            if method not in _OPERATION_METHODS or not isinstance(operation, Mapping):
                continue
            responses = operation.get("responses")
            status_codes = ""
            if isinstance(responses, Mapping):
                status_codes = ",".join(str(code) for code in responses)
            operation_id = _text(operation.get("operationId"))
            pointer_path = path_text.replace("~", "~0").replace("/", "~1")
            pointer = f"/paths/{pointer_path}/{method.lower()}"
            metadata: dict[str, Any] = {
                "source": resolved_source,
                "source_type": "openapi",
                "document_type": "openapi",
                "section": "operation",
                "api_name": api_name,
                "title": title,
                "version": version,
                "method": method,
                "path": path_text,
                "endpoint": f"{method} {path_text}",
                "operation_id": operation_id,
                "status_codes": status_codes,
                "pointer": pointer,
            }
            loaded.append(
                Document(
                    page_content=_operation_content(
                        title=title,
                        version=version,
                        path=path_text,
                        method=method,
                        operation=operation,
                    ),
                    metadata=metadata,
                )
            )
    return loaded


def _change_content(
    *,
    api_name: str,
    current_version: str,
    entry: Mapping[str, Any],
) -> str:
    identifier = _text(entry.get("id"), "unidentified-change")
    from_version = _text(entry.get("from_version"), "unknown")
    to_version = _text(entry.get("to_version"), current_version or "unknown")
    endpoint = _text(entry.get("endpoint"))
    method = _text(entry.get("method")).upper()
    lines = [
        f"Changelog: {identifier}",
        f"API: {api_name}",
        f"Version transition: {from_version} -> {to_version}",
    ]
    if endpoint:
        lines.append(f"Endpoint: {method + ' ' if method else ''}{endpoint}")
    for key in ("published_at", "breaking", "summary", "migration"):
        if key in entry:
            lines.append(f"{key}: {_text(entry[key])}")
    changes = entry.get("changes")
    if changes:
        lines.append(f"Changes: {_json_text(changes)}")
    lines.append(f"Changelog entry: {_json_text(entry)}")
    return "\n".join(lines)


def load_changelog_documents(path: Path | str = DEFAULT_CHANGELOG_PATH) -> list[Document]:
    """Load one searchable document for each static changelog entry."""

    source = Path(path)
    document = _read_json(source)
    if not isinstance(document, Mapping):
        raise RetrievalDataError(f"Changelog must be an object: {source}")
    raw_entries = document.get("entries", [])
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise RetrievalDataError(f"Changelog entries must be an array: {source}")

    api_name = _slug(document.get("api_name") or document.get("api") or "unknown-api")
    current_version = _text(document.get("current_version"), "unknown")
    resolved_source = str(source.resolve())
    loaded: list[Document] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise RetrievalDataError(f"Changelog entry {index} must be an object: {source}")
        identifier = _text(raw_entry.get("id"), f"entry-{index}")
        from_version = _text(raw_entry.get("from_version"), "")
        to_version = _text(raw_entry.get("to_version"), current_version)
        path_text = _text(raw_entry.get("path") or raw_entry.get("endpoint"))
        method = _text(raw_entry.get("method")).upper()
        endpoint = f"{method} {path_text}".strip() if path_text else ""
        metadata: dict[str, Any] = {
            "source": resolved_source,
            "source_type": "changelog",
            "document_type": "changelog",
            "section": "changelog",
            "api_name": api_name,
            "version": to_version,
            "current_version": current_version,
            "from_version": from_version,
            "to_version": to_version,
            "method": method,
            "path": path_text,
            "endpoint": endpoint,
            "changelog_id": identifier,
            "breaking": bool(raw_entry.get("breaking", False)),
            "updated_at": _text(raw_entry.get("published_at")),
            "pointer": f"/entries/{index}",
        }
        loaded.append(
            Document(
                page_content=_change_content(
                    api_name=api_name,
                    current_version=current_version,
                    entry=raw_entry,
                ),
                metadata=metadata,
            )
        )
    return loaded


def load_knowledge_documents(
    openapi_path: Path | str = DEFAULT_OPENAPI_PATH,
    changelog_path: Path | str = DEFAULT_CHANGELOG_PATH,
) -> list[Document]:
    """Load the V1 OpenAPI and changelog sources in a stable order."""

    return load_openapi_documents(openapi_path) + load_changelog_documents(changelog_path)


def _tokens(value: Any) -> list[str]:
    return _TOKEN_PATTERN.findall(_text(value).casefold())


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (set, frozenset, list, tuple)):
        return any(_value_matches(actual, candidate) for candidate in expected)
    if isinstance(actual, (set, frozenset, list, tuple)):
        return any(_value_matches(candidate, expected) for candidate in actual)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.casefold() == expected.casefold()
    return actual == expected


def _metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    return all(
        key in metadata and _value_matches(metadata[key], expected)
        for key, expected in filters.items()
    )


class KeywordRetriever(BaseRetriever):
    """LangChain-compatible deterministic keyword retriever.

    ``search`` exposes metadata filters explicitly. ``invoke``/``ainvoke`` use
    the retriever's default filters and ``default_k`` through BaseRetriever.
    """

    documents: tuple[Document, ...] = Field(default_factory=tuple)
    default_k: int = Field(default=5, ge=1, le=100)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_paths(
        cls,
        openapi_path: Path | str = DEFAULT_OPENAPI_PATH,
        changelog_path: Path | str = DEFAULT_CHANGELOG_PATH,
        *,
        default_k: int = 5,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> "KeywordRetriever":
        """Build a retriever directly from the two V1 static sources."""

        return cls(
            documents=tuple(load_knowledge_documents(openapi_path, changelog_path)),
            default_k=default_k,
            metadata_filters=dict(metadata_filters or {}),
        )

    def search(
        self,
        query: str,
        *,
        metadata_filters: Mapping[str, Any] | None = None,
        k: int | None = None,
    ) -> list[Document]:
        """Apply metadata filters and return top keyword matches.

        Matching is OR across query terms and AND across metadata filters. A
        blank query intentionally returns no documents instead of returning an
        arbitrary first page of the knowledge base.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        limit = self.default_k if k is None else k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("k must be a positive integer")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        filters = self.metadata_filters if metadata_filters is None else dict(metadata_filters)
        ranked: list[tuple[int, int, Document]] = []
        query_term_set = set(query_tokens)
        for index, document in enumerate(self.documents):
            if not _metadata_matches(document.metadata, filters):
                continue
            content_tokens = _tokens(document.page_content)
            metadata_tokens = _tokens(" ".join(str(value) for value in document.metadata.values()))
            content_counts = {token: content_tokens.count(token) for token in query_term_set}
            score = sum(
                count + (2 if token in metadata_tokens else 0)
                for token, count in content_counts.items()
                if count
            )
            if score:
                ranked.append((score, index, document))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [document for _, _, document in ranked[:limit]]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        return self.search(query)


DeterministicKeywordRetriever = KeywordRetriever


def build_retriever(
    openapi_path: Path | str = DEFAULT_OPENAPI_PATH,
    changelog_path: Path | str = DEFAULT_CHANGELOG_PATH,
    *,
    default_k: int = 5,
    metadata_filters: Mapping[str, Any] | None = None,
) -> KeywordRetriever:
    """Convenience constructor for the default static V1 knowledge base."""

    return KeywordRetriever.from_paths(
        openapi_path,
        changelog_path,
        default_k=default_k,
        metadata_filters=metadata_filters,
    )


__all__ = [
    "DEFAULT_CHANGELOG_PATH",
    "DEFAULT_OPENAPI_PATH",
    "DeterministicKeywordRetriever",
    "KeywordRetriever",
    "RetrievalDataError",
    "build_retriever",
    "load_changelog_documents",
    "load_knowledge_documents",
    "load_openapi_documents",
]
