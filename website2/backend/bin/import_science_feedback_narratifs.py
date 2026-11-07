#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# Ensure imports work when run as a script
SCRIPT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, BACKEND_DIR)

import argumentation_store as argstore  # noqa: E402


DEFAULT_SOURCE_URL = (
    "https://science.feedback.org/fr/informations-trompeuses-climat-television-radio-france-analyse-ampleur-recits-frequents/"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    s = argstore.normalize_text(text).lower()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _shorten(text: str, max_len: int) -> str:
    text = argstore.normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _get_str(d: Dict[str, Any], key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def load_narratifs(path: str) -> List[Dict[str, Any]]:
    data = argstore.load_json_file(path, [])
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def build_tags(number: Optional[int], qualifier: Optional[str]) -> List[str]:
    tags = ["science_feedback", "observatoire_medias_ecologie"]
    if isinstance(number, int):
        tags.append(f"sf_narratif_{number:02d}")
    if qualifier:
        tags.append(f"sf_qualifier_{_slug(qualifier)}")
    return tags


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import the 19 Science Feedback 'narratifs' JSON into the argumentation store "
            "(arguments + counter-arguments)."
        )
    )

    data_dir = os.path.join(BACKEND_DIR, "data")

    parser.add_argument(
        "--in",
        dest="in_path",
        default=os.path.join(data_dir, "science_feedback_19_narratifs.json"),
        help="Input JSON (scraped narratifs).",
    )
    parser.add_argument(
        "--argumentation",
        default=os.path.join(data_dir, "argumentation.json"),
        help="Argumentation store path.",
    )
    parser.add_argument(
        "--personas",
        default=os.path.join(data_dir, "persona_archetypes.json"),
        help="Persona sidecar path.",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="Source URL to store on counter-arguments.",
    )
    parser.add_argument(
        "--use-full-debunk",
        action="store_true",
        help="Use the full debunk text as counter-argument (default: use 'resume' when available).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute what would be added, but don't write files.",
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.in_path):
        print(f"Input not found: {args.in_path}")
        return 2

    narratifs = load_narratifs(args.in_path)
    if not narratifs:
        print(f"No narratifs found in {args.in_path} (expected a JSON array of objects)")
        return 2

    store = argstore.load_store(args.argumentation, args.personas)

    imported_args = 0
    imported_counters = 0
    skipped = 0

    for row in narratifs:
        number = row.get("number")
        number_int = int(number) if isinstance(number, int) else None

        qualifier = _get_str(row, "qualifier")
        claim = _get_str(row, "claim")
        resume = _get_str(row, "resume")
        debunk = _get_str(row, "debunk")

        if not claim:
            skipped += 1
            continue

        counter_text = None
        if args.use_full_debunk:
            counter_text = debunk or resume
        else:
            counter_text = resume or debunk

        counter_text = (counter_text or "").strip()
        if not counter_text:
            skipped += 1
            continue

        tags = build_tags(number_int, qualifier)

        imported_from = {
            "project": "ScienceFeedback",
            "source_file": os.path.basename(args.in_path),
            "narratif_number": number_int,
            "qualifier": qualifier,
            # Preserve richer text without forcing the UI to display it.
            "debunk_full": debunk,
            "title": _get_str(row, "title"),
        }

        arg = argstore.upsert_argument(
            store,
            text=claim,
            tags=tags,
            imported_from=imported_from,
        )

        # If this argument is new, upsert_argument will have appended it.
        # We can't easily detect it without re-indexing, but we can keep a light heuristic.
        # The end result is deterministic and idempotent due to text-based de-dupe.
        imported_args += 1

        ca = argstore.create_counter_argument(
            store,
            argument_id=arg["id"],
            text=counter_text,
            persona_archetype_id=None,
            source_url=args.source_url,
            imported_from=imported_from,
        )
        if ca:
            imported_counters += 1

    if args.dry_run:
        print("Dry run: no files written")
        print(f"Would process rows: {len(narratifs)}")
        print(f"Would import arguments (upsert calls): {imported_args}")
        print(f"Would import counter-arguments (create calls): {imported_counters}")
        print(f"Would skip rows (missing claim/debunk): {skipped}")
        return 0

    argstore.save_store(args.argumentation, store, args.personas)

    print(f"Processed rows: {len(narratifs)}")
    print(f"Upserted arguments: {imported_args}")
    print(f"Created counter-arguments: {imported_counters}")
    print(f"Skipped rows (missing claim/debunk): {skipped}")
    print(f"Wrote store: {args.argumentation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
