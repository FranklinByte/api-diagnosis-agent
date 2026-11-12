"""Store for OpenAPI documents supplied by the user.

The original project could only diagnose one bundled Mock contract.  This
module lets a caller register any OpenAPI 3.x document (JSON or YAML text) and
refer to it later by a content-derived ``spec_id``.  Documents are parsed and
validated before they are written; nothing is fetched from the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .paths import project_root

MAX_SPEC_BYTES = 2_000_000
BUILTIN_SPEC_ID = "weather-mock"
_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")
_SPEC_ID = re.compile(r"^[a-f0-9]{16}$")


class SpecError(ValueError):
    """The supplied text is not a usable OpenAPI 3.x document."""


def _store_dir() -> Path:
    configured = os.getenv("API_DIAGNOSIS_SPEC_DIR")
    return Path(configured) if configured else project_root() / "data" / "specs"


def _examples_dir() -> Path:
    return project_root() / "specs" / "examples"


def _builtin_path() -> Path:
    return project_root() / "specs" / "mock-api.json"


def _jsonable(value: Any) -> Any:
    """YAML allows integer keys (``200:``) and dates; JSON Schema tools do not."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_document(text: str) -> dict[str, Any]:
    """Parse JSON or YAML text into a plain JSON-compatible mapping."""

    if not isinstance(text, str) or not text.strip():
        raise SpecError("接口文档内容为空。")
    if len(text.encode("utf-8")) > MAX_SPEC_BYTES:
        raise SpecError("接口文档超过 2 MB 上限。")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        try:
            # PyYAML ships no type stubs and the locked CI environment does not add them.
            import yaml  # type: ignore[import-untyped,unused-ignore]

            document = yaml.safe_load(text)
        except Exception as exc:  # yaml.YAMLError or a missing optional import
            raise SpecError("无法解析：内容既不是合法的 JSON，也不是合法的 YAML。") from exc
    if not isinstance(document, Mapping):
        raise SpecError("接口文档的最外层必须是一个对象。")
    return _jsonable(document)


def validate_document(document: Mapping[str, Any]) -> None:
    """Reject documents that the diagnosis tools cannot work with."""

    version = str(document.get("openapi") or "")
    if document.get("swagger"):
        raise SpecError("这是 Swagger 2.0 文档，请先转换为 OpenAPI 3.x 再上传。")
    if not version.startswith("3."):
        raise SpecError("缺少 openapi 版本字段，或版本不是 3.x。")
    paths = document.get("paths")
    if not isinstance(paths, Mapping) or not paths:
        raise SpecError("文档里没有任何接口路径（paths 为空）。")
    try:
        from openapi_spec_validator import validate

        validate(dict(document))
    except SpecError:
        raise
    except Exception as exc:
        message = str(getattr(exc, "message", exc)).splitlines()[0][:300]
        raise SpecError(f"文档不符合 OpenAPI 规范：{message}") from exc


def list_operations(document: Mapping[str, Any]) -> list[dict[str, str]]:
    operations: list[dict[str, str]] = []
    for path, item in (document.get("paths") or {}).items():
        if not isinstance(item, Mapping):
            continue
        for method in _HTTP_METHODS:
            operation = item.get(method)
            if isinstance(operation, Mapping):
                operations.append(
                    {
                        "method": method.upper(),
                        "path": str(path),
                        "summary": str(operation.get("summary") or operation.get("operationId") or "")[:200],
                    }
                )
    return operations


def _info(spec_id: str, document: Mapping[str, Any], *, kind: str, name: Optional[str] = None, uploaded_at: Optional[str] = None) -> dict[str, Any]:
    raw_meta = document.get("info")
    meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
    servers = document.get("servers") or []
    return {
        "spec_id": spec_id,
        "kind": kind,
        "name": name or str(meta.get("title") or spec_id),
        "title": str(meta.get("title") or ""),
        "version": str(meta.get("version") or ""),
        "server": str(servers[0].get("url")) if servers and isinstance(servers[0], Mapping) else "",
        "operation_count": len(list_operations(document)),
        "uploaded_at": uploaded_at,
    }


def save_spec(text: str, name: Optional[str] = None) -> dict[str, Any]:
    """Validate and persist one document; return its public description."""

    document = parse_document(text)
    validate_document(document)
    canonical = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    spec_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clean_name = (name or "").strip()[:80] or None
    record = {
        "spec_id": spec_id,
        "name": clean_name,
        "uploaded_at": uploaded_at,
        "document": document,
    }
    directory = _store_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{spec_id}.json"
    staging = directory / f"{spec_id}.tmp"
    staging.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    staging.replace(target)
    return _info(spec_id, document, kind="uploaded", name=clean_name, uploaded_at=uploaded_at)


def _example_documents() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    directory = _examples_dir()
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        try:
            found[f"example-{path.stem}"] = parse_document(path.read_text(encoding="utf-8"))
        except SpecError:
            continue
    return found


def list_specs() -> list[dict[str, Any]]:
    """Built-in Mock contract, bundled examples, then uploaded documents."""

    items = [_info(BUILTIN_SPEC_ID, json.loads(_builtin_path().read_text(encoding="utf-8")), kind="builtin")]
    for spec_id, document in _example_documents().items():
        items.append(_info(spec_id, document, kind="example"))
    directory = _store_dir()
    if directory.is_dir():
        uploaded = []
        for path in directory.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                uploaded.append(
                    _info(record["spec_id"], record["document"], kind="uploaded", name=record.get("name"), uploaded_at=record.get("uploaded_at"))
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
        items.extend(sorted(uploaded, key=lambda item: item.get("uploaded_at") or "", reverse=True))
    return items


def load_spec(spec_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(info, document)`` or raise ``KeyError`` for an unknown id."""

    if spec_id == BUILTIN_SPEC_ID:
        document = json.loads(_builtin_path().read_text(encoding="utf-8"))
        return _info(spec_id, document, kind="builtin"), document
    if spec_id.startswith("example-"):
        examples = _example_documents()
        if spec_id in examples:
            return _info(spec_id, examples[spec_id], kind="example"), examples[spec_id]
        raise KeyError(spec_id)
    # The id is used to build a file name, so only the generated form is accepted.
    if not _SPEC_ID.match(spec_id):
        raise KeyError(spec_id)
    path = _store_dir() / f"{spec_id}.json"
    if not path.is_file():
        raise KeyError(spec_id)
    record = json.loads(path.read_text(encoding="utf-8"))
    document = record["document"]
    return _info(spec_id, document, kind="uploaded", name=record.get("name"), uploaded_at=record.get("uploaded_at")), document


__all__ = [
    "BUILTIN_SPEC_ID",
    "MAX_SPEC_BYTES",
    "SpecError",
    "list_operations",
    "list_specs",
    "load_spec",
    "parse_document",
    "save_spec",
    "validate_document",
]
