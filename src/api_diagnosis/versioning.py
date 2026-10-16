"""Deterministic API version checks backed by ``data/changelog.json``.

The checker intentionally does not infer compatibility from an LLM or from a
free-form document search.  It only returns ``compatible`` when the changelog
contains an explicit compatible path (or the requested and current versions
are identical); otherwise it returns ``unknown`` so callers can abstain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .models import Evidence, EvidenceType, VersionChange, VersionCheckResult
from .paths import project_root


DEFAULT_CHANGELOG_PATH = project_root() / "data" / "changelog.json"


class ChangelogFormatError(ValueError):
    """Raised when the changelog JSON cannot be interpreted safely."""


@dataclass(frozen=True)
class _Entry:
    api_name: Optional[str]
    from_version: Optional[str]
    to_version: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    breaking: bool
    summary: Optional[str]
    affected_fields: tuple[str, ...]
    details: Mapping[str, Any]
    pointer: str

    def as_change(self) -> VersionChange:
        return VersionChange(
            from_version=self.from_version,
            to_version=self.to_version,
            endpoint=self.endpoint,
            method=self.method,
            breaking=self.breaking,
            summary=self.summary,
            affected_fields=list(self.affected_fields),
            details=dict(self.details),
        )


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_text(item: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = _text(item.get(key))
        if value:
            return value
    return None


def _version_key(value: Optional[str]) -> tuple[Any, ...]:
    """Comparable key for common ``v1``, ``1.2.0`` and date-like versions."""

    if value is None:
        return ()
    normalized = value.strip().lower().lstrip("v")
    parts: list[Any] = []
    for piece in re.split(r"[.\-_]+", normalized):
        if piece.isdigit():
            parts.append((0, int(piece)))
        else:
            parts.append((1, piece))
    return tuple(parts)


def _is_breaking(item: Mapping[str, Any], changes: Any) -> bool:
    value = item.get("breaking", item.get("breaking_change", item.get("is_breaking")))
    if isinstance(value, bool):
        return value
    if value is not None:
        return str(value).strip().lower() in {"1", "true", "yes", "breaking"}
    if isinstance(changes, list):
        for change in changes:
            if isinstance(change, Mapping):
                kind = str(change.get("type", change.get("change_type", ""))).lower()
                if kind in {"remove", "rename", "type_change", "breaking"}:
                    return True
    return False


def _affected_fields(item: Mapping[str, Any], changes: Any) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("affected_fields", "changed_fields", "fields"):
        raw = item.get(key)
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(str(value) for value in raw)
    if isinstance(changes, list):
        for change in changes:
            if isinstance(change, Mapping):
                value = _first_text(change, "field", "path", "pointer", "name")
                if value:
                    values.append(value)
            elif isinstance(change, str) and change.startswith(("/", "$.")):
                values.append(change)
    return tuple(dict.fromkeys(value for value in values if value))


def _normalize_entry(
    item: Mapping[str, Any],
    *,
    pointer: str,
    api_name: Optional[str] = None,
    version_hint: Optional[str] = None,
) -> Optional[_Entry]:
    from_version = _first_text(
        item,
        "from_version",
        "from",
        "previous_version",
        "old_version",
    )
    to_version = _first_text(
        item,
        "to_version",
        "to",
        "new_version",
        "target_version",
    )
    explicit_version = _first_text(item, "version", "release_version")
    if explicit_version and not to_version:
        to_version = explicit_version
    if version_hint and not to_version:
        to_version = version_hint
    endpoint = _first_text(item, "endpoint", "path", "route")
    method = _first_text(item, "method", "http_method")
    changes = item.get("changes", item.get("change", []))
    summary = _first_text(item, "summary", "description", "message", "title")
    nested_api = _first_text(item, "api_name", "api", "service", "name") or api_name
    if not any((from_version, to_version, endpoint, summary, changes)):
        return None
    return _Entry(
        api_name=nested_api,
        from_version=from_version,
        to_version=to_version,
        endpoint=endpoint,
        method=method.upper() if method else None,
        breaking=_is_breaking(item, changes),
        summary=summary,
        affected_fields=_affected_fields(item, changes),
        details=dict(item),
        pointer=pointer,
    )


def _collect_entries(
    value: Any,
    *,
    pointer: str = "",
    api_name: Optional[str] = None,
    version_hint: Optional[str] = None,
) -> list[_Entry]:
    """Collect entries from a few deliberately small changelog shapes.

    The canonical V0 shape is a top-level list or ``{"entries": [...]}``,
    while the ``apis`` and ``versions`` forms make the loader tolerant of a
    hand-authored fixture without allowing arbitrary executable content.
    """

    entries: list[_Entry] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_pointer = f"{pointer}/{index}"
            if isinstance(item, Mapping):
                normalized = _normalize_entry(
                    item,
                    pointer=child_pointer,
                    api_name=api_name,
                    version_hint=version_hint,
                )
                if normalized:
                    entries.append(normalized)
                # A release item can contain an array of endpoint changes.
                for key in ("entries", "changes", "endpoints"):
                    nested = item.get(key)
                    if isinstance(nested, (list, Mapping)):
                        entries.extend(
                            _collect_entries(
                                nested,
                                pointer=f"{child_pointer}/{key}",
                                api_name=normalized.api_name if normalized else api_name,
                                version_hint=normalized.to_version if normalized else version_hint,
                            )
                        )
        return entries
    if not isinstance(value, Mapping):
        return entries

    own_api = _first_text(value, "api_name", "api", "service") or api_name
    normalized = _normalize_entry(
        value,
        pointer=pointer or "/",
        api_name=own_api,
        version_hint=version_hint,
    )
    # Do not emit a container such as {"entries": [...]} as an entry.
    if normalized and not any(key in value for key in ("entries", "apis", "versions", "releases")):
        entries.append(normalized)

    for key in ("entries", "changes", "changelog", "releases"):
        nested = value.get(key)
        if isinstance(nested, (list, Mapping)):
            entries.extend(
                _collect_entries(
                    nested,
                    pointer=f"{pointer}/{key}",
                    api_name=own_api,
                    version_hint=version_hint or _first_text(value, "current_version", "latest_version"),
                )
            )
    apis = value.get("apis")
    if isinstance(apis, Mapping):
        for name, nested in apis.items():
            entries.extend(
                _collect_entries(
                    nested,
                    pointer=f"{pointer}/apis/{name}",
                    api_name=str(name),
                    version_hint=_first_text(nested, "current_version", "latest_version")
                    if isinstance(nested, Mapping)
                    else None,
                )
            )
    versions = value.get("versions")
    if isinstance(versions, Mapping):
        for version, nested in versions.items():
            if isinstance(nested, Mapping):
                entries.extend(
                    _collect_entries(
                        nested,
                        pointer=f"{pointer}/versions/{version}",
                        api_name=own_api,
                        version_hint=str(version),
                    )
                )
            elif isinstance(nested, list):
                entries.extend(
                    _collect_entries(
                        nested,
                        pointer=f"{pointer}/versions/{version}",
                        api_name=own_api,
                        version_hint=str(version),
                    )
                )
    return entries


def load_changelog(path: Path | str = DEFAULT_CHANGELOG_PATH) -> list[_Entry]:
    """Load and normalize the static JSON changelog."""

    changelog_path = Path(path)
    if not changelog_path.exists():
        raise FileNotFoundError(f"Changelog not found: {changelog_path}")
    try:
        with changelog_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ChangelogFormatError(f"Invalid changelog JSON: {changelog_path}") from exc
    entries = _collect_entries(document)
    if not entries:
        raise ChangelogFormatError(
            f"No version entries found in changelog: {changelog_path}"
        )
    return entries


def _matches(
    entry: _Entry,
    *,
    api_name: Optional[str],
    endpoint: Optional[str],
    method: Optional[str],
) -> bool:
    if api_name and entry.api_name and entry.api_name != api_name:
        return False
    if endpoint and entry.endpoint and entry.endpoint != endpoint:
        return False
    if method and entry.method and entry.method != method.upper():
        return False
    return True


class VersionChecker:
    """Version checker that reads a bounded, static changelog once."""

    def __init__(self, changelog_path: Path | str = DEFAULT_CHANGELOG_PATH):
        self.changelog_path = Path(changelog_path)
        self._entries: Optional[list[_Entry]] = None

    @property
    def entries(self) -> list[_Entry]:
        if self._entries is None:
            self._entries = load_changelog(self.changelog_path)
        return self._entries

    def reload(self) -> None:
        self._entries = None

    def check(
        self,
        requested_version: Optional[str] = None,
        current_version: Optional[str] = None,
        *,
        api_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        user_version: Optional[str] = None,
    ) -> VersionCheckResult:
        requested = requested_version or user_version
        if not requested:
            raise ValueError("requested_version (or user_version) is required")
        requested = str(requested).strip()
        method = method.upper() if method else None
        candidates = [
            entry
            for entry in self.entries
            if _matches(
                entry,
                api_name=api_name,
                endpoint=endpoint,
                method=method,
            )
        ]
        if current_version is None:
            current_version = self._infer_current(candidates)
        matching = self._relevant_entries(candidates, requested, current_version)
        changes = [entry.as_change() for entry in matching]
        affected_fields = list(
            dict.fromkeys(field for entry in matching for field in entry.affected_fields)
        )
        evidence = [
            Evidence(
                source=str(self.changelog_path),
                source_type=EvidenceType.CHANGELOG,
                api_name=api_name,
                endpoint=endpoint,
                method=method,
                version=current_version,
                pointer=entry.pointer,
                excerpt=entry.summary,
                details={"breaking": entry.breaking},
            )
            for entry in matching
        ]

        if current_version and _version_key(requested) == _version_key(current_version):
            return VersionCheckResult(
                status="compatible",
                compatible=True,
                requested_version=requested,
                current_version=current_version,
                api_name=api_name,
                endpoint=endpoint,
                method=method,
                changes=changes,
                affected_fields=affected_fields,
                reason="Requested version matches the current contract.",
                evidence=evidence,
            )
        if not matching:
            return VersionCheckResult(
                status="unknown",
                compatible=None,
                requested_version=requested,
                current_version=current_version,
                api_name=api_name,
                endpoint=endpoint,
                method=method,
                reason="No matching changelog evidence proves compatibility.",
                evidence=evidence,
            )
        breaking = any(entry.breaking for entry in matching)
        return VersionCheckResult(
            status="incompatible" if breaking else "compatible",
            compatible=not breaking,
            requested_version=requested,
            current_version=current_version,
            api_name=api_name,
            endpoint=endpoint,
            method=method,
            changes=changes,
            affected_fields=affected_fields,
            reason=(
                "A breaking change affects the requested endpoint/version."
                if breaking
                else "The changelog records no breaking change for this version path."
            ),
            evidence=evidence,
        )

    @staticmethod
    def _infer_current(entries: Sequence[_Entry]) -> Optional[str]:
        versions = [entry.to_version for entry in entries if entry.to_version]
        if not versions:
            versions = [entry.from_version for entry in entries if entry.from_version]
        return max(versions, key=_version_key) if versions else None

    @staticmethod
    def _relevant_entries(
        entries: Sequence[_Entry], requested: str, current: Optional[str]
    ) -> list[_Entry]:
        if not current:
            return [
                entry
                for entry in entries
                if entry.from_version
                and _version_key(entry.from_version) == _version_key(requested)
            ]
        requested_key = _version_key(requested)
        current_key = _version_key(current)
        selected: list[_Entry] = []
        for entry in entries:
            from_key = _version_key(entry.from_version)
            to_key = _version_key(entry.to_version)
            if from_key == requested_key:
                selected.append(entry)
                continue
            # Aggregate intermediate releases when checking an older version
            # against a newer current contract.
            if from_key and to_key and requested_key <= from_key < current_key:
                selected.append(entry)
        return selected


def check_version(
    requested_version: Optional[str] = None,
    current_version: Optional[str] = None,
    *,
    user_version: Optional[str] = None,
    api_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    changelog_path: Path | str = DEFAULT_CHANGELOG_PATH,
) -> VersionCheckResult:
    """Convenience function for a one-shot static changelog comparison."""

    return VersionChecker(changelog_path).check(
        requested_version=requested_version,
        current_version=current_version,
        user_version=user_version,
        api_name=api_name,
        endpoint=endpoint,
        method=method,
    )


__all__ = [
    "ChangelogFormatError",
    "DEFAULT_CHANGELOG_PATH",
    "VersionChecker",
    "check_version",
    "load_changelog",
]
