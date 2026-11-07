import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Iterable

BONPOTE_INFOGRAPHIES_URL = "https://bonpote.com/infographies/"


BONPOTE_CATEGORY_TAGS: dict[str, list[str]] = {
    "Climat": ["climat"],
    "Mobilité": ["transport"],
    "Outils": ["outils", "communication"],
    "Politique": ["politique"],
    "Agriculture": ["agriculture"],
    "Biodiversité": ["biodiversite"],
    "Banque & Finance": ["economie", "finance"],
    "Matières premières": ["matieres-premieres"],
    "Solutions": ["solutions"],
    "GIEC": ["giec", "climat"],
}


@dataclass(frozen=True)
class BonPoteInfographie:
    title: str
    image_url: str
    pdf_url: str | None
    article_url: str | None
    categories: list[str]
    published_month: str | None  # ISO-like YYYY-MM


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (re.inform; +https://example.invalid)"
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "ignore")


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _extract_upload_month(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"/wp-content/uploads/(\d{4})/(\d{2})/", url)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def _normalize_tag(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = unescape(raw)

    # Basic ascii-ish normalization for tags
    raw = (
        raw.replace("œ", "oe")
        .replace("æ", "ae")
        .replace("’", "'")
        .replace("&", "and")
    )
    # Remove accents cheaply
    raw = (
        raw.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ä", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ö", "o")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ü", "u")
        .replace("ç", "c")
    )
    raw = re.sub(r"\s+", " ", raw)

    # slugify
    raw = re.sub(r"[^a-z0-9\s-]", "", raw)
    raw = raw.strip().replace(" ", "-")
    raw = re.sub(r"-+", "-", raw)
    return raw


def _infer_tags_from_title(title: str) -> list[str]:
    t = (title or "").lower()
    t = unescape(t)

    tags: list[str] = []

    def add(*vals: str) -> None:
        for v in vals:
            if v and v not in tags:
                tags.append(v)

    # Climate / environment
    if any(k in t for k in ["climat", "co2", "ges", "emissions", "accord de paris", "cop", "rechauff", "empreinte carbone"]):
        add("climat")
    if any(k in t for k in ["biodivers", "limites planetaires", "ocean", "oceans", "acidification"]):
        add("biodiversite")
    if any(k in t for k in ["plastique", "dechets"]):
        add("dechets", "pollution")

    # Transport / mobility
    if any(k in t for k in ["avion", "train", "voiture", "autoroute", "110", "zfe", "ski", "mobilite"]):
        add("transport")

    # Economy / finance
    if any(k in t for k in ["economie", "cout", "coute", "impot", "impots", "fisc", "fraude", "banque", "investissement", "milliard", "decile", "patrimoine"]):
        add("economie")
    if any(k in t for k in ["banque", "finance", "investissement"]):
        add("finance")
    if any(k in t for k in ["impot", "impots", "fisc", "fraude"]):
        add("fiscalite")

    # Digital
    if any(k in t for k in ["numerique", "intelligence artificielle", "ia", "5g", "satellite"]):
        add("numerique")

    # Politics
    if any(k in t for k in ["legisl", "municipales", "elections", "politique", "zfe"]):
        add("politique")

    # GIEC
    if "giec" in t:
        add("giec", "climat")

    return tags


def parse_bonpote_infographies_from_html(html: str) -> list[BonPoteInfographie]:
    # The listing page HTML can be messy; segment cards using the recurring image-lightbox anchors.
    anchor_re = re.compile(
        r'<a\s+[^>]*href="(?P<img>https://bonpote\.com/wp-content/uploads/[^"]+\.(?:png|jpe?g|webp))"'
        r'[^>]*data-elementor-open-lightbox="yes"[^>]*>',
        flags=re.IGNORECASE,
    )

    anchors = list(anchor_re.finditer(html))
    if not anchors:
        return []

    items: list[BonPoteInfographie] = []

    for i, m in enumerate(anchors):
        start = m.start()
        end = anchors[i + 1].start() if i + 1 < len(anchors) else min(len(html), m.start() + 12000)
        chunk = html[start:end]

        image_url = m.group('img')

        # Title: closest semantic heading inside this chunk.
        title = None
        title_match = re.search(
            r"<h2\s+class=\"elementor-heading-title[^\"]*\"[^>]*>(.*?)</h2>",
            chunk,
            flags=re.S,
        )
        if title_match:
            title = unescape(_strip_tags(title_match.group(1)))

        # Fallback: Elementor lightbox title.
        if not title:
            lightbox_title_match = re.search(r"data-elementor-lightbox-title=\"([^\"]+)\"", chunk)
            if lightbox_title_match:
                title = unescape(lightbox_title_match.group(1)).strip()

        if not title:
            continue

        hrefs = re.findall(r'href="(https?://[^\"]+)"', chunk)

        def is_pdf_href(h: str) -> bool:
            h_lower = h.lower()
            return "/wp-content/uploads/" in h_lower and re.search(r"\.pdf(\?|$)", h_lower) is not None

        pdf_url = next((h for h in hrefs if is_pdf_href(h)), None)

        # Article URL: bonpote.com, not wp-content, not listing page, not the infographie post itself.
        article_url = None
        for h in hrefs:
            if not h.startswith("https://bonpote.com/"):
                continue
            if "/wp-content/uploads/" in h:
                continue
            if h.rstrip("/") == BONPOTE_INFOGRAPHIES_URL.rstrip("/"):
                continue
            if "/infographie/" in h:
                continue
            article_url = h
            break

        categories = []
        for tmatch in re.finditer(r"elementor-post-info__terms-list-item\">([^<]+)<", chunk):
            cat = unescape(tmatch.group(1)).strip()
            if cat and cat not in categories:
                categories.append(cat)

        published_month = _extract_upload_month(pdf_url) or _extract_upload_month(image_url)

        items.append(
            BonPoteInfographie(
                title=title,
                image_url=image_url,
                pdf_url=pdf_url,
                article_url=article_url,
                categories=categories,
                published_month=published_month,
            )
        )

    # De-duplicate by image URL (stable unique key)
    by_image: dict[str, BonPoteInfographie] = {}
    for it in items:
        by_image[it.image_url] = it

    # Stable ordering: newest month first, then title
    def sort_key(it: BonPoteInfographie):
        # reverse month: use string comparison on YYYY-MM works
        month = it.published_month or "0000-00"
        return (month, it.title.lower())

    return sorted(by_image.values(), key=sort_key, reverse=True)


def fetch_bonpote_infographies() -> list[BonPoteInfographie]:
    html = _fetch_html(BONPOTE_INFOGRAPHIES_URL)
    return parse_bonpote_infographies_from_html(html)


def to_gallery_seed_items(items: Iterable[BonPoteInfographie]) -> list[dict]:
    out: list[dict] = []
    now_iso = datetime.now().isoformat()

    for idx, it in enumerate(items, start=1):
        # Topic tags (French). Keep the source tag for provenance.
        tags: list[str] = ["bonpote"]

        for cat in it.categories:
            mapped = BONPOTE_CATEGORY_TAGS.get(cat)
            if mapped:
                for m in mapped:
                    if m and m not in tags:
                        tags.append(m)
            else:
                normalized = _normalize_tag(cat)
                if normalized and normalized not in tags:
                    tags.append(normalized)

        for inferred in _infer_tags_from_title(it.title):
            if inferred and inferred not in tags:
                tags.append(inferred)

        # Ensure at least one topic tag besides provenance.
        if tags == ["bonpote"]:
            tags.append("ecologie")

        # Add an approximate timestamp based on WP upload month so sorting is meaningful.
        first_submitted_at = None
        if it.published_month:
            first_submitted_at = f"{it.published_month}-01T00:00:00"

        out.append(
            {
                "id": f"bonpote-{idx}",
                "title": it.title,
                "description": None,
                "source": it.image_url,
                "url": it.pdf_url or "",
                "article_url": it.article_url or "",
                "source_page": BONPOTE_INFOGRAPHIES_URL,
                "attribution": "Bon Pote",
                "tags": sorted(set(tags)),
                "first_submitted_at": first_submitted_at,
                "first_submitted_by": {
                    "id": None,
                    "username": "Bon Pote",
                    "role": "external",
                },
                "synced_at": now_iso,
            }
        )

    return out


def write_bonpote_gallery_json(output_path: str) -> list[dict]:
    infographies = fetch_bonpote_infographies()
    data = to_gallery_seed_items(infographies)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data
