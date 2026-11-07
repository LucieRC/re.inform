"""Unified library of external resources.

Resources reach the project through several doors — a curated list, the
argumentation store (both hand-submitted and auto-created while saving a
debunk), the gallery seeds, and user submissions. This module folds all of them
into one catalogue keyed by normalized URL, so the same source referenced from
three places is one entry rather than three.

Two rules shape what users see:

- **Only reviewed resources are public.** An entry is public once an admin has
  reviewed it, or once it is *vetted* — curated lists, repo-committed gallery
  seeds, and importer output are trusted by virtue of where they came from.
- **Review decisions live in an overlay**, keyed by normalized URL, so they
  survive re-imports and never require rewriting the source files.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from argumentation_store import load_json_file, normalize_url, save_json_file


SCHEMA_VERSION = 1

# Review states an entry can be in.
STATUS_APPROVED = "approved"
STATUS_PENDING = "pending"
STATUS_REJECTED = "rejected"

# Resource types, ordered as they should appear in the UI's filter bar.
TYPES = [
    "article",
    "study",
    "report",
    "infographic",
    "video",
    "podcast",
    "dataset",
    "website",
]

TYPE_LABELS = {
    "article": "Article",
    "study": "Study / paper",
    "report": "Report / PDF",
    "infographic": "Infographic",
    "video": "Video",
    "podcast": "Podcast / audio",
    "dataset": "Data",
    "website": "Website",
}

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
_AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".m4a")
_VIDEO_EXTS = (".mp4", ".webm", ".mov")

_VIDEO_HOSTS = {"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "peertube.fr", "twitch.tv"}
_PODCAST_HOSTS = {"podcasts.apple.com", "deezer.com", "podcloud.fr", "ausha.co", "acast.com"}
_STUDY_HOSTS = {
    "doi.org", "arxiv.org", "nature.com", "science.org", "sciencedirect.com", "springer.com",
    "wiley.com", "tandfonline.com", "plos.org", "pubmed.ncbi.nlm.nih.gov", "cairn.info",
    "hal.science", "theconversation.com", "ipcc.ch", "biorxiv.org", "researchgate.net",
}
_DATA_HOSTS = {
    "data.gouv.fr", "ourworldindata.org", "insee.fr", "statista.com", "kaggle.com",
    "ec.europa.eu", "worldbank.org", "globalcarbonatlas.org",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def entry_key(url: str) -> str:
    """Stable identity for a resource: its normalized URL, lowercased."""
    return normalize_url(url).lower()


def entry_id(url: str) -> str:
    """Short, URL-safe handle for an entry — stable across restarts."""
    return hashlib.sha1(entry_key(url).encode("utf-8")).hexdigest()[:12]


def _host_of(url: str) -> Optional[str]:
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)", (url or "").strip())
    if not match:
        return None
    host = match.group(1).lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def publisher_of(url: str) -> str:
    """Who published this — the facet users actually recognise (bonpote.com…)."""
    return _host_of(url) or "Uploaded to re.inform"


def _path_of(url: str) -> str:
    stripped = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/]+", "", (url or "").strip())
    return stripped.split("?", 1)[0].split("#", 1)[0].lower()


def infer_type(url: str, *, title: Optional[str] = None) -> str:
    """Best-effort type for a URL. Admins can correct it during review."""
    path = _path_of(url)
    host = _host_of(url)

    # File extensions are the strongest signal — they beat host heuristics.
    if path.endswith(_IMAGE_EXTS):
        return "infographic"
    if path.endswith(".pdf"):
        return "report"
    if path.endswith(_VIDEO_EXTS):
        return "video"
    if path.endswith(_AUDIO_EXTS):
        return "podcast"

    if host:
        if host in _VIDEO_HOSTS:
            return "video"
        if host in _PODCAST_HOSTS:
            return "podcast"
        if host.startswith("open.spotify.com") or "/episode/" in path:
            return "podcast"
        if host in _STUDY_HOSTS:
            return "study"
        if host in _DATA_HOSTS:
            return "dataset"

    # Deliberately no keyword sniffing on the title or slug: an article *about*
    # infographics is not an infographic, and typing it as one puts an imageless
    # card at the top of the visual view. Extensions and hosts are reliable;
    # anything else is the admin's call at review time.

    # A URL pointing at a specific page is an article; a bare domain is a site.
    return "article" if path.strip("/") else "website"


# ========================================
# Review overlay
# ========================================


def default_reviews() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": _now_iso(), "reviews": {}}


def load_reviews(path: str) -> Dict[str, Any]:
    data = load_json_file(path, default_reviews())
    if not isinstance(data, dict) or not isinstance(data.get("reviews"), dict):
        return default_reviews()
    data.setdefault("schema_version", SCHEMA_VERSION)
    return data


def save_reviews(path: str, data: Dict[str, Any]) -> None:
    data["updated_at"] = _now_iso()
    save_json_file(path, data)


def record_review(
    reviews: Dict[str, Any],
    url: str,
    *,
    status: str,
    reviewer: str,
    resource_type: Optional[str] = None,
    topics: Optional[List[str]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Record an admin decision, appending to the entry's reviewer roll.

    Re-reviewing an entry updates its status and adds the reviewer if new — the
    count of distinct reviewers is what "reviewed by at least one admin" means.
    """
    if status not in (STATUS_APPROVED, STATUS_REJECTED, STATUS_PENDING):
        raise ValueError("status must be approved, rejected or pending")
    if resource_type is not None and resource_type not in TYPES:
        raise ValueError(f"type must be one of: {', '.join(TYPES)}")

    key = entry_key(url)
    if not key:
        raise ValueError("A resource URL is required")

    entry = reviews.setdefault("reviews", {}).setdefault(key, {
        "url": normalize_url(url),
        "reviewers": [],
        "created_at": _now_iso(),
    })

    entry["status"] = status
    entry["updated_at"] = _now_iso()
    if resource_type is not None:
        entry["type"] = resource_type
    if topics is not None:
        entry["topics"] = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
    if title:
        entry["title"] = title.strip()
    if description:
        entry["description"] = description.strip()
    if note:
        entry["note"] = note.strip()

    reviewers = entry.setdefault("reviewers", [])
    if not any(r.get("name") == reviewer for r in reviewers if isinstance(r, dict)):
        reviewers.append({"name": reviewer, "at": _now_iso()})

    return entry


# ========================================
# Aggregation
# ========================================


_GENERIC_SLUGS = {"watch", "index", "embed", "view", "post", "page", "article", "video", "home", "default"}


def title_from_url(url: str) -> str:
    """Readable fallback title from a URL slug.

    Imported resources arrive as bare links, and the debunk that cites them is
    titled after the *claim*, not the source — so the slug is the most honest
    label available until an admin writes a better one.
    """
    path = _path_of(url).rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    slug = re.sub(r"\.\w{1,5}$", "", slug)
    slug = re.sub(r"[-_+]+", " ", slug).strip()
    # Routing slugs ("/watch", "/index") name the site's plumbing, not the
    # resource — the publisher is more informative until an admin retitles it.
    if not slug or slug in _GENERIC_SLUGS or len(slug) < 4:
        return publisher_of(url)
    return slug[:1].upper() + slug[1:]


# Tags that record where data came from rather than what it's about. They make
# useless facets ("bonpote" on two thirds of the library), so they never surface.
_PROVENANCE_TAGS = {
    "bonpote", "bonpotejs", "science_feedback", "sciencefeedback", "sf",
    "observatoire_medias_ecologie", "import", "imported", "seed",
}

# The corpus is bilingual; without folding these, every facet appears twice.
_TOPIC_ALIASES = {
    "climat": "climate",
    "economie": "economy", "économie": "economy",
    "energie": "energy", "énergie": "energy",
    "societe": "society", "société": "society",
    "politique": "policy",
    "technologie": "technology",
    "biodiversite": "biodiversity", "biodiversité": "biodiversity",
    "medias": "media", "médias": "media",
    "sante": "health", "santé": "health",
    "environnement": "environment",
    "alimentation": "food",
    "logement": "housing",
    "agriculture": "agriculture",
    "transport": "transport",
    "adaptation": "adaptation",
    # Narrower or untranslated tags from the Bon Pote / gallery imports.
    "giec": "climate", "ipcc": "climate",
    "ecologie": "environment", "écologie": "environment", "ecology": "environment",
    "numerique": "technology", "numérique": "technology",
    "fiscalite": "economy", "fiscalité": "economy", "finance": "economy",
    "finances-publiques": "economy", "richesse": "economy",
    "inegalites": "society", "inégalités": "society",
    "journalisme": "media", "desinformation": "media", "désinformation": "media",
    "collectivites": "policy", "collectivités": "policy", "maires": "policy",
    "institutions": "policy", "politics": "policy",
    "dechets": "pollution", "déchets": "pollution",
    "outils": "tools",
}


def normalize_topic(tag: Any) -> Optional[str]:
    """Fold a raw tag into a topic facet, or drop it if it isn't one."""
    if not isinstance(tag, str):
        return None
    cleaned = re.sub(r"\s+", " ", tag.strip().lower())
    # "sf_narratif_09", "sf_qualifier_inexact"… are Science Feedback import keys.
    if not cleaned or cleaned in _PROVENANCE_TAGS or cleaned.startswith("sf_"):
        return None
    return _TOPIC_ALIASES.get(cleaned, cleaned)


def _clean(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _tags(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    topics = []
    for tag in raw:
        topic = normalize_topic(tag)
        if topic and topic not in topics:
            topics.append(topic)
    return topics


class _Entry(dict):
    """A library entry under construction.

    Fields are filled first-wins in source-precedence order, so the curated list
    wins over an importer's bare URL.
    """

    def fill(self, field: str, value: Any) -> None:
        if value and not self.get(field):
            self[field] = value

    def add_topics(self, topics: Iterable[str]) -> None:
        seen = {t.lower() for t in self["topics"]}
        for topic in topics:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                self["topics"].append(topic)


def _new_entry(url: str) -> _Entry:
    normalized = normalize_url(url)
    return _Entry({
        "id": entry_id(normalized),
        "url": normalized,
        "title": None,
        "description": None,
        "type": None,
        "topics": [],
        "publisher": publisher_of(normalized),
        "origins": [],
        "context_url": None,
        "preview_image": None,
        "submitted_by": None,
        "submitted_at": None,
        "used_in_responses": 0,
        "linked_argument_ids": [],
        "vetted": False,
    })


def build_library(
    *,
    curated: Optional[List[Dict[str, Any]]] = None,
    store: Optional[Dict[str, Any]] = None,
    gallery_seeds: Optional[List[Dict[str, Any]]] = None,
    submissions: Optional[List[Dict[str, Any]]] = None,
    contributions: Optional[List[Dict[str, Any]]] = None,
    reviews: Optional[Dict[str, Any]] = None,
    resolve_user: Optional[Callable[[int], Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Fold every known resource source into one reviewed catalogue.

    Sources are processed in precedence order, richest first, so better metadata
    wins. Every entry comes back — filtering by review status is the caller's
    job, because admins need to see what users must not.
    """
    review_map = (reviews or {}).get("reviews", {}) if reviews else {}
    entries: Dict[str, _Entry] = {}

    def touch(url: Any, origin: str, *, vetted: bool = False) -> Optional[_Entry]:
        key = entry_key(url if isinstance(url, str) else "")
        if not key:
            return None
        entry = entries.get(key)
        if entry is None:
            entry = _new_entry(url)
            entries[key] = entry
        if origin not in entry["origins"]:
            entry["origins"].append(origin)
        if vetted:
            entry["vetted"] = True
        return entry

    def user_brief(user_id: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(user_id, int) or resolve_user is None:
            return None
        user = resolve_user(user_id)
        if not user:
            return None
        return {"id": user["id"], "username": user["username"]}

    # 1. Curated list — hand-picked, so it wins on metadata and is pre-vetted.
    for item in (curated or []):
        if not isinstance(item, dict):
            continue
        entry = touch(item.get("source"), "curated", vetted=True)
        if entry is None:
            continue
        entry.fill("title", _clean(item.get("title")))
        entry.fill("description", _clean(item.get("summary")) or _clean(item.get("narrative")))
        entry.fill("submitted_at", _clean(item.get("date")))
        entry.add_topics(_tags(item.get("tags")))
        if entry["submitted_by"] is None:
            entry["submitted_by"] = user_brief(item.get("user_id"))

    # 2. Gallery seeds — committed to the repo, so equally pre-vetted. The image
    #    itself is the resource; any article it came from is context.
    for item in (gallery_seeds or []):
        if not isinstance(item, dict):
            continue
        entry = touch(item.get("source"), "gallery", vetted=True)
        if entry is None:
            continue
        entry.fill("title", _clean(item.get("title")))
        entry.fill("description", _clean(item.get("description")))
        entry.fill("context_url", _clean(item.get("article_url")) or _clean(item.get("source_page")))
        entry.fill("submitted_at", _clean(item.get("first_submitted_at")))
        entry.add_topics(_tags(item.get("tags")))
        source = _clean(item.get("source")) or ""
        if source.lower().split("?", 1)[0].endswith(_IMAGE_EXTS):
            entry.fill("preview_image", source)
        first_by = item.get("first_submitted_by")
        if entry["submitted_by"] is None and isinstance(first_by, dict) and first_by.get("username"):
            entry["submitted_by"] = {"id": first_by.get("id"), "username": first_by["username"]}

    # 3. Approved contributions — an admin already accepted the submission that
    #    carried this source, which is exactly the review bar we enforce here.
    for item in (contributions or []):
        if not isinstance(item, dict):
            continue
        entry = touch(item.get("source"), "contribution", vetted=True)
        if entry is None:
            continue
        # Deliberately not the contribution's title or counter-argument text:
        # those describe the debunk, not the source being cited.
        entry.fill("submitted_at", _clean(item.get("date")))
        entry.add_topics(_tags(item.get("tags")))
        if entry["submitted_by"] is None:
            entry["submitted_by"] = user_brief(item.get("user_id"))

    # 4. Raw submissions — pending ones stay pending; rejected stay out.
    for item in (submissions or []):
        if not isinstance(item, dict):
            continue
        status = (item.get("status") or "pending").lower()
        entry = touch(item.get("source"), f"submission:{status}", vetted=(status == "approved"))
        if entry is None:
            continue
        entry.fill("submitted_at", _clean(item.get("submitted_at")))
        entry.add_topics(_tags(item.get("tags")))
        if entry["submitted_by"] is None:
            entry["submitted_by"] = user_brief(item.get("user_id"))

    # 5. Argumentation store. Importer output (no author) is vetted; anything a
    #    user submitted themselves has to be reviewed.
    store = store or {}
    store_resources = [r for r in (store.get("resources") or []) if isinstance(r, dict)]

    # How often each resource actually backs a response — the "most used" sort.
    uses_by_id: Dict[int, int] = {}
    uses_by_key: Dict[str, int] = {}
    args_by_resource: Dict[int, List[int]] = {}
    for ca in (store.get("counter_arguments") or []):
        if not isinstance(ca, dict):
            continue
        argument_id = ca.get("argument_id")
        for rid in (ca.get("resource_ids") or []):
            if isinstance(rid, int):
                uses_by_id[rid] = uses_by_id.get(rid, 0) + 1
                if isinstance(argument_id, int):
                    args_by_resource.setdefault(rid, []).append(argument_id)
        legacy = entry_key(ca.get("source_url") or "")
        if legacy and not (ca.get("resource_ids") or []):
            uses_by_key[legacy] = uses_by_key.get(legacy, 0) + 1

    # Topics for store resources come from the claims they speak to — the only
    # topic signal importer-created resources have.
    tags_by_argument = {
        a["id"]: _tags(a.get("tags"))
        for a in (store.get("arguments") or [])
        if isinstance(a, dict) and isinstance(a.get("id"), int)
    }

    for resource in store_resources:
        url = resource.get("url")
        imported = resource.get("created_by_user_id") is None
        origin = "import" if imported else "submitted"
        entry = touch(url, origin, vetted=imported)
        if entry is None:
            continue

        entry.fill("title", _clean(resource.get("title")))
        entry.fill("description", _clean(resource.get("description")))
        entry.fill("submitted_at", _clean(resource.get("created_at")))
        entry.add_topics(_tags(resource.get("tags")))
        if entry["submitted_by"] is None:
            entry["submitted_by"] = user_brief(resource.get("created_by_user_id"))

        rid = resource.get("id")
        linked = list(resource.get("argument_ids") or [])
        if isinstance(rid, int):
            entry["used_in_responses"] += uses_by_id.get(rid, 0)
            linked.extend(args_by_resource.get(rid, []))

        for argument_id in linked:
            if isinstance(argument_id, int) and argument_id not in entry["linked_argument_ids"]:
                entry["linked_argument_ids"].append(argument_id)
                entry.add_topics(tags_by_argument.get(argument_id, []))

    for key, count in uses_by_key.items():
        if key in entries:
            entries[key]["used_in_responses"] += count

    # Finalise: apply admin overrides, then resolve the review state.
    library: List[Dict[str, Any]] = []
    for key, entry in entries.items():
        review = review_map.get(key) if isinstance(review_map.get(key), dict) else None

        if review:
            if review.get("title"):
                entry["title"] = review["title"]
            if review.get("description"):
                entry["description"] = review["description"]
            if review.get("topics"):
                entry["topics"] = list(review["topics"])

        entry["type"] = (
            review.get("type") if review and review.get("type") in TYPES
            else infer_type(entry["url"], title=entry["title"])
        )
        entry["type_label"] = TYPE_LABELS.get(entry["type"], entry["type"].title())
        if not entry["title"]:
            entry["title"] = title_from_url(entry["url"])

        reviewers = [r for r in (review.get("reviewers") or []) if isinstance(r, dict)] if review else []
        if review and review.get("status") in (STATUS_APPROVED, STATUS_REJECTED, STATUS_PENDING):
            status = review["status"]
        elif entry["vetted"]:
            # Trusted by provenance rather than by a person clicking approve.
            status = STATUS_APPROVED
        else:
            status = STATUS_PENDING

        entry["review"] = {
            "status": status,
            "reviewers": reviewers,
            "reviewer_count": len(reviewers),
            "by_provenance": status == STATUS_APPROVED and not reviewers,
            "note": review.get("note") if review else None,
        }
        entry["topics"] = sorted(entry["topics"], key=str.lower)
        library.append(dict(entry))

    library.sort(key=lambda e: (-e["used_in_responses"], str(e["title"] or "").lower()))
    return library


def facet_counts(library: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Counts per filter value, so the UI can show and prune its facets."""
    def tally(values_of: Callable[[Dict[str, Any]], Iterable[str]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in library:
            for value in values_of(entry):
                counts[value] = counts.get(value, 0) + 1
        return counts

    types = tally(lambda e: [e["type"]])
    topics = tally(lambda e: e["topics"])
    publishers = tally(lambda e: [e["publisher"]])

    return {
        "types": [
            {"value": t, "label": TYPE_LABELS.get(t, t.title()), "count": types[t]}
            for t in TYPES if t in types
        ],
        "topics": [
            {"value": t, "label": t, "count": c}
            for t, c in sorted(topics.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        ],
        "publishers": [
            {"value": p, "label": p, "count": c}
            for p, c in sorted(publishers.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        ],
    }
