from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
PERSONA_ARCHETYPES_BASENAME = "persona_archetypes.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _whitespace_re.sub(" ", (text or "").strip())


def normalize_url(url: str) -> str:
    """Light URL normalization for dedupe: trim, drop fragment and trailing slash."""
    u = (url or "").strip()
    if not u:
        return ""
    u = u.split("#", 1)[0].strip()
    while u.endswith("/"):
        u = u[:-1]
    return u


URL_ERROR = "Links must be full web addresses starting with http:// or https://"


def is_http_url(url: str) -> bool:
    """True for an absolute http(s) URL with a host.

    This is the only kind of link accepted from users: it rules out
    `javascript:`, `data:` and other schemes that would run when the stored
    link is rendered and clicked.
    """
    u = (url or "").strip()
    # Spaces or control characters inside a link are never legitimate and are
    # a common way to smuggle a scheme past naive checks.
    if not u or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in u):
        return False
    try:
        parts = urlsplit(u)
        host = parts.hostname
    except ValueError:
        return False
    return parts.scheme.lower() in ("http", "https") and bool(host)


# Review lifecycle for claims and answers. A record with no `status` is
# published: everything written before the field existed was public, and the
# normal write paths (POST /api/arguments, /api/counter-arguments) still publish
# straight away. `pending` is waiting for an admin or trusted-member votes;
# `rejected` stays in the file for the record but is never shown.
STATUS_PUBLISHED = "published"
STATUS_PENDING = "pending"
STATUS_REJECTED = "rejected"
CONTENT_STATUSES = (STATUS_PUBLISHED, STATUS_PENDING, STATUS_REJECTED)


def status_of(item: Optional[Dict[str, Any]]) -> str:
    status = (item or {}).get("status") if isinstance(item, dict) else None
    return status if status in CONTENT_STATUSES else STATUS_PUBLISHED


def is_published(item: Optional[Dict[str, Any]]) -> bool:
    return isinstance(item, dict) and status_of(item) == STATUS_PUBLISHED


def published(items: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Only the records the public may see."""
    return [it for it in (items or []) if is_published(it)]


def load_json_file(filepath: str, default: Any) -> Any:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json_file(filepath: str, data: Any) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def default_store() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "arguments": [],
        "persona_archetypes": [],
        "counter_arguments": [],
        "ratings": [],
        "resources": [],
        "importance_ratings": [],
        "source_checks": [],
    }


def _infer_persona_path(argumentation_path: str) -> str:
    return os.path.join(os.path.dirname(argumentation_path), PERSONA_ARCHETYPES_BASENAME)


def default_persona_store() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "persona_archetypes": [],
    }


def load_persona_archetypes(path: str) -> List[Dict[str, Any]]:
    data = load_json_file(path, None)
    if not isinstance(data, dict):
        return []
    personas = data.get("persona_archetypes")
    if not isinstance(personas, list):
        return []
    return [p for p in personas if isinstance(p, dict)]


def save_persona_archetypes(path: str, personas: List[Dict[str, Any]]) -> None:
    payload = default_persona_store()
    payload["updated_at"] = _now_iso()
    payload["persona_archetypes"] = [p for p in (personas or []) if isinstance(p, dict)]
    save_json_file(path, payload)


def load_store(path: str, persona_path: Optional[str] = None) -> Dict[str, Any]:
    store = load_json_file(path, default_store())
    if not isinstance(store, dict):
        return default_store()

    personas_from_main = store.get("persona_archetypes") if isinstance(store.get("persona_archetypes"), list) else None

    persona_path_to_use = persona_path or _infer_persona_path(path)
    personas_from_sidecar: Optional[List[Dict[str, Any]]] = None
    if os.path.exists(persona_path_to_use):
        personas_from_sidecar = load_persona_archetypes(persona_path_to_use)

    # Forward-compatible defaults
    store.setdefault("schema_version", SCHEMA_VERSION)
    store.setdefault("updated_at", _now_iso())
    store.setdefault("arguments", [])
    store["persona_archetypes"] = (
        personas_from_sidecar
        if personas_from_sidecar is not None
        else (personas_from_main if isinstance(personas_from_main, list) else [])
    )
    store.setdefault("counter_arguments", [])
    store.setdefault("ratings", [])
    store.setdefault("resources", [])
    store.setdefault("importance_ratings", [])
    store.setdefault("source_checks", [])
    return store


def save_store(path: str, store: Dict[str, Any], persona_path: Optional[str] = None) -> None:
    store["updated_at"] = _now_iso()

    persona_path_to_use = persona_path or _infer_persona_path(path)
    save_persona_archetypes(persona_path_to_use, store.get("persona_archetypes", []) or [])

    store_to_save = dict(store)
    store_to_save.pop("persona_archetypes", None)
    save_json_file(path, store_to_save)


def _next_int_id(items: List[Dict[str, Any]]) -> int:
    max_id = 0
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            max_id = max(max_id, int(item["id"]))
    return max_id + 1


def _index_by_norm_text(items: List[Dict[str, Any]], field: str = "text") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        norm = normalize_text(str(it.get(field) or ""))
        if norm:
            out[norm.lower()] = it
    return out


def upsert_argument(
    store: Dict[str, Any],
    text: str,
    *,
    created_by_user_id: Optional[int] = None,
    tags: Optional[List[str]] = None,
    imported_from: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a claim, or return the existing one with the same text.

    `status` only applies to a newly created claim; an existing claim keeps its
    own status (a pending proposal never un-publishes a public claim).
    """
    text_norm = normalize_text(text)
    if not text_norm:
        raise ValueError("Argument text is required")
    if status is not None and status not in CONTENT_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(CONTENT_STATUSES)}")

    idx = _index_by_norm_text(store["arguments"], field="text")
    existing = idx.get(text_norm.lower())
    if existing:
        return existing

    arg = {
        "id": _next_int_id(store["arguments"]),
        "text": text_norm,
        "tags": tags or [],
        "created_at": _now_iso(),
        "created_by_user_id": created_by_user_id,
        "imported_from": imported_from,
    }
    if status and status != STATUS_PUBLISHED:
        arg["status"] = status
    store["arguments"].append(arg)
    return arg


def upsert_persona_archetype(
    store: Dict[str, Any],
    name: str,
    description: Optional[str],
    *,
    imported_from: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    name_norm = normalize_text(name)
    if not name_norm:
        raise ValueError("Persona name is required")

    description_norm = normalize_text(description or "")
    key = (name_norm.lower(), description_norm.lower())

    for p in store["persona_archetypes"]:
        if not isinstance(p, dict):
            continue
        pn = normalize_text(str(p.get("name") or "")).lower()
        pd = normalize_text(str(p.get("description") or "")).lower()
        if (pn, pd) == key:
            return p

    persona = {
        "id": _next_int_id(store["persona_archetypes"]),
        "name": name_norm,
        "description": description_norm or None,
        "created_at": _now_iso(),
        "imported_from": imported_from,
    }
    store["persona_archetypes"].append(persona)
    return persona


def _merge_unique(existing: Optional[List[Any]], incoming: Optional[List[Any]]) -> List[Any]:
    """Union two lists, preserving order and dropping duplicates."""
    out: List[Any] = []
    seen = set()
    for item in list(existing or []) + list(incoming or []):
        key = item.lower() if isinstance(item, str) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def upsert_resource(
    store: Dict[str, Any],
    url: str,
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    argument_ids: Optional[List[int]] = None,
    created_by_user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Create or return a first-class Resource, deduped by normalized URL.

    A Resource is shared: the same source can back many debunks. Re-submitting a
    known URL enriches it (fills a missing title/description, unions tags and the
    claims it speaks to) rather than creating a duplicate.
    """
    store.setdefault("resources", [])
    norm = normalize_url(url)
    if not norm:
        raise ValueError("Resource URL is required")

    tags_norm = [normalize_text(t) for t in (tags or []) if normalize_text(str(t or ""))]
    arg_ids = [int(a) for a in (argument_ids or []) if isinstance(a, int)]

    key = norm.lower()
    for r in store["resources"]:
        if not isinstance(r, dict):
            continue
        if normalize_url(str(r.get("url") or "")).lower() == key:
            # Opportunistically fill in missing metadata.
            if title and not r.get("title"):
                r["title"] = normalize_text(title)
            if description and not r.get("description"):
                r["description"] = normalize_text(description)
            r["tags"] = _merge_unique(r.get("tags"), tags_norm)
            r["argument_ids"] = _merge_unique(r.get("argument_ids"), arg_ids)
            return r

    resource = {
        "id": _next_int_id(store["resources"]),
        "url": norm,
        "title": normalize_text(title) if title else None,
        "description": normalize_text(description) if description else None,
        "tags": tags_norm,
        "argument_ids": arg_ids,
        "created_at": _now_iso(),
        "created_by_user_id": created_by_user_id,
    }
    store["resources"].append(resource)
    return resource


def resources_by_id(store: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        r["id"]: r
        for r in store.get("resources", []) or []
        if isinstance(r, dict) and isinstance(r.get("id"), int)
    }


def create_counter_argument(
    store: Dict[str, Any],
    *,
    argument_id: int,
    text: str,
    persona_archetype_id: Optional[int] = None,
    source_url: Optional[str] = None,
    resource_ids: Optional[List[int]] = None,
    created_by_user_id: Optional[int] = None,
    imported_from: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    text_norm = normalize_text(text)
    if not text_norm:
        raise ValueError("Counter-argument text is required")
    if status is not None and status not in CONTENT_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(CONTENT_STATUSES)}")

    # Light dedupe: same argument + persona + identical text
    for ca in store["counter_arguments"]:
        if not isinstance(ca, dict):
            continue
        if ca.get("argument_id") != argument_id:
            continue
        if ca.get("persona_archetype_id") != persona_archetype_id:
            continue
        if normalize_text(str(ca.get("text") or "")).lower() == text_norm.lower():
            return ca

    counter = {
        "id": _next_int_id(store["counter_arguments"]),
        "argument_id": int(argument_id),
        "text": text_norm,
        "persona_archetype_id": int(persona_archetype_id) if persona_archetype_id is not None else None,
        "source_url": (source_url or "").strip() or None,
        "resource_ids": [int(r) for r in (resource_ids or []) if isinstance(r, int)],
        "created_at": _now_iso(),
        "created_by_user_id": created_by_user_id,
        "imported_from": imported_from,
    }
    if status and status != STATUS_PUBLISHED:
        counter["status"] = status
    store["counter_arguments"].append(counter)
    return counter


def set_review_status(
    store: Dict[str, Any],
    counter: Dict[str, Any],
    status: str,
    *,
    reviewed_by: Any = None,
) -> None:
    """Publish or reject a proposed answer, and its claim if that is new too.

    A claim that arrived with the answer shares its fate: publishing the answer
    publishes the claim; rejecting it rejects the claim unless another public
    answer already hangs off it.
    """
    if status not in CONTENT_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(CONTENT_STATUSES)}")
    now = _now_iso()
    counter["status"] = status
    counter["reviewed_at"] = now
    counter["reviewed_by"] = reviewed_by

    args_by_id, _ = build_indexes(store)
    arg = args_by_id.get(counter.get("argument_id"))

    # Answers in review keep their link on `source_url` only, so an unreviewed
    # link never enters the shared Resources. It becomes one on publication.
    if status == STATUS_PUBLISHED and not counter.get("resource_ids") and is_http_url(counter.get("source_url") or ""):
        res = upsert_resource(
            store,
            counter["source_url"],
            argument_ids=[arg["id"]] if arg else None,
            created_by_user_id=counter.get("created_by_user_id"),
        )
        counter["resource_ids"] = [res["id"]]

    if not arg or is_published(arg):
        return
    if status == STATUS_PUBLISHED:
        arg["status"] = STATUS_PUBLISHED
        arg["reviewed_at"] = now
        arg["reviewed_by"] = reviewed_by
    elif status == STATUS_REJECTED:
        others = [
            ca for ca in store.get("counter_arguments", []) or []
            if isinstance(ca, dict) and ca is not counter
            and ca.get("argument_id") == arg.get("id") and status_of(ca) != STATUS_REJECTED
        ]
        if not others:
            arg["status"] = STATUS_REJECTED
            arg["reviewed_at"] = now
            arg["reviewed_by"] = reviewed_by


def resource_urls_for(store: Dict[str, Any], counter: Dict[str, Any]) -> List[str]:
    """A debunk's source links: its Resources, else the legacy source_url."""
    res_by_id = resources_by_id(store)
    urls = [res_by_id[rid].get("url") for rid in counter.get("resource_ids") or [] if rid in res_by_id]
    urls = [u for u in urls if u]
    if not urls and counter.get("source_url"):
        urls = [counter["source_url"]]
    return urls


def legacy_contribution_view(
    store: Dict[str, Any],
    counter: Dict[str, Any],
    *,
    args_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
    personas_by_id: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """One answer shaped like an old contributions.json entry.

    contributions.json was retired in favour of this store; the endpoints that
    used to serve it (/api/contributions, profile lists) build this view so old
    callers keep working. `id` is the counter-argument id.
    """
    if args_by_id is None or personas_by_id is None:
        args_by_id, personas_by_id = build_indexes(store)
    arg = args_by_id.get(counter.get("argument_id"))
    if not arg:
        return None
    persona = personas_by_id.get(counter.get("persona_archetype_id")) or {}
    narrative = str(arg.get("text") or "")
    title = (f"{persona['name']}: " if persona.get("name") else "") + narrative
    if len(title) > 80:
        title = title[:79].rstrip() + "…"
    urls = resource_urls_for(store, counter)
    return {
        "id": counter.get("id"),
        "counter_argument_id": counter.get("id"),
        "argument_id": arg.get("id"),
        "title": title,
        "narrative": narrative,
        "counter_argument": counter.get("text"),
        "tags": list(arg.get("tags") or []),
        "source": urls[0] if urls else None,
        "sources": urls,
        "persona_archetype_id": counter.get("persona_archetype_id"),
        "date": counter.get("created_at"),
        "status": status_of(counter),
        "user_id": counter.get("created_by_user_id"),
    }


def create_rating(
    store: Dict[str, Any],
    *,
    counter_argument_id: int,
    persona_archetype_id: Optional[int],
    score: int,
    user_id: Optional[int],
) -> Dict[str, Any]:
    if not isinstance(score, int) or score < 1 or score > 5:
        raise ValueError("Score must be an integer between 1 and 5")

    rating = {
        "id": _next_int_id(store["ratings"]),
        "counter_argument_id": int(counter_argument_id),
        "persona_archetype_id": int(persona_archetype_id) if persona_archetype_id is not None else None,
        "score": int(score),
        "user_id": int(user_id) if user_id is not None else None,
        "created_at": _now_iso(),
    }
    store["ratings"].append(rating)
    return rating


def set_importance_rating(
    store: Dict[str, Any],
    *,
    argument_id: int,
    score: int,
    user_id: Optional[int],
) -> Dict[str, Any]:
    """Record how urgent a claim is to debunk (1..5).

    Unlike effectiveness ratings — which accumulate, one per person per debunk —
    importance is a standing opinion on a claim, so a second vote from the same
    person replaces the first instead of stacking.
    """
    if not isinstance(score, int) or score < 1 or score > 5:
        raise ValueError("Score must be an integer between 1 and 5")

    store.setdefault("importance_ratings", [])

    if user_id is not None:
        for ir in store["importance_ratings"]:
            if not isinstance(ir, dict):
                continue
            if ir.get("argument_id") == int(argument_id) and ir.get("user_id") == int(user_id):
                ir["score"] = int(score)
                ir["updated_at"] = _now_iso()
                return ir

    rating = {
        "id": _next_int_id(store["importance_ratings"]),
        "argument_id": int(argument_id),
        "score": int(score),
        "user_id": int(user_id) if user_id is not None else None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    store["importance_ratings"].append(rating)
    return rating


SOURCE_CHECK_VERDICTS = ("supports", "partial", "does_not_support", "broken")


def set_source_check(
    store: Dict[str, Any],
    *,
    counter_argument_id: int,
    verdict: str,
    user_id: int,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Record whether a debunk's sources actually back it up.

    Like importance, a check is a standing opinion: re-checking replaces the
    previous verdict instead of stacking a second vote.
    """
    if verdict not in SOURCE_CHECK_VERDICTS:
        raise ValueError(f"verdict must be one of: {', '.join(SOURCE_CHECK_VERDICTS)}")

    store.setdefault("source_checks", [])
    note_norm = normalize_text(note or "") or None

    for sc in store["source_checks"]:
        if not isinstance(sc, dict):
            continue
        if sc.get("counter_argument_id") == int(counter_argument_id) and sc.get("user_id") == int(user_id):
            sc["verdict"] = verdict
            sc["note"] = note_norm
            sc["updated_at"] = _now_iso()
            return sc

    check = {
        "id": _next_int_id(store["source_checks"]),
        "counter_argument_id": int(counter_argument_id),
        "user_id": int(user_id),
        "verdict": verdict,
        "note": note_norm,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    store["source_checks"].append(check)
    return check


def source_checks_by_counter(store: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Group source checks by the debunk they were cast on."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for sc in store.get("source_checks", []) or []:
        if not isinstance(sc, dict):
            continue
        cid = sc.get("counter_argument_id")
        if isinstance(cid, int) and sc.get("verdict") in SOURCE_CHECK_VERDICTS:
            out.setdefault(cid, []).append(sc)
    return out


def importance_by_argument(store: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Group importance ratings by the claim they were cast on."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for ir in store.get("importance_ratings", []) or []:
        if not isinstance(ir, dict):
            continue
        aid = ir.get("argument_id")
        if isinstance(aid, int) and isinstance(ir.get("score"), int):
            out.setdefault(aid, []).append(ir)
    return out


def build_indexes(store: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    args_by_id = {a["id"]: a for a in store.get("arguments", []) if isinstance(a, dict) and isinstance(a.get("id"), int)}
    personas_by_id = {
        p["id"]: p
        for p in store.get("persona_archetypes", [])
        if isinstance(p, dict) and isinstance(p.get("id"), int)
    }
    return args_by_id, personas_by_id


def avg_rating_for_counter_argument(store: Dict[str, Any], counter_argument_id: int) -> Optional[float]:
    scores = [
        int(r.get("score"))
        for r in store.get("ratings", [])
        if isinstance(r, dict) and r.get("counter_argument_id") == counter_argument_id and isinstance(r.get("score"), int)
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)
