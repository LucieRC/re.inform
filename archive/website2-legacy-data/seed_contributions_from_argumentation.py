#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, BACKEND_DIR)

import argumentation_store as argstore  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shorten(text: str, max_len: int) -> str:
    text = argstore.normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _contrib_dedupe_key(c: Dict[str, Any]) -> Tuple[str, str, str]:
    narrative = argstore.normalize_text(str(c.get("narrative") or "")).lower()
    counter = argstore.normalize_text(str(c.get("counter_argument") or "")).lower()
    source = str(c.get("source") or "").strip()
    return (narrative, counter, source)


def main() -> int:
    data_dir = os.path.join(BACKEND_DIR, "data")
    argumentation_path = os.path.join(data_dir, "argumentation.json")
    contributions_path = os.path.join(data_dir, "contributions.json")

    store = argstore.load_store(argumentation_path)
    args_by_id, personas_by_id = argstore.build_indexes(store)

    contributions: List[Dict[str, Any]] = argstore.load_json_file(contributions_path, [])
    if not isinstance(contributions, list):
        contributions = []

    existing_keys = set()
    for c in contributions:
        if isinstance(c, dict):
            existing_keys.add(_contrib_dedupe_key(c))

    next_id = 1
    for c in contributions:
        if isinstance(c, dict) and isinstance(c.get("id"), int):
            next_id = max(next_id, int(c["id"]) + 1)

    added = 0
    for ca in store.get("counter_arguments", []) or []:
        if not isinstance(ca, dict):
            continue

        argument_id = ca.get("argument_id")
        if not isinstance(argument_id, int):
            continue
        arg = args_by_id.get(argument_id)
        if not arg:
            continue

        narrative = str(arg.get("text") or "").strip()
        counter_text = str(ca.get("text") or "").strip()
        if not narrative or not counter_text:
            continue

        source_url = (ca.get("source_url") or "").strip() or None
        created_at = ca.get("created_at") or store.get("updated_at") or _now_iso()

        persona_name: Optional[str] = None
        persona_id = ca.get("persona_archetype_id")
        if isinstance(persona_id, int):
            persona = personas_by_id.get(persona_id)
            if isinstance(persona, dict):
                persona_name = str(persona.get("name") or "").strip() or None

        title_prefix = f"{persona_name}: " if persona_name else ""
        title = _shorten(title_prefix + narrative, 80)

        contribution = {
            "id": next_id,
            "title": title or f"Contribution #{next_id}",
            "narrative": narrative,
            "counter_argument": counter_text,
            "tags": ["bonpote"],
            "source": source_url,
            "date": created_at,
            # Keep track of provenance without affecting the UI.
            "imported_from": ca.get("imported_from") or arg.get("imported_from"),
        }

        key = _contrib_dedupe_key(contribution)
        if key in existing_keys:
            continue

        contributions.append(contribution)
        existing_keys.add(key)
        next_id += 1
        added += 1

    argstore.save_json_file(contributions_path, contributions)
    print(f"Wrote {len(contributions)} contributions to {contributions_path}")
    print(f"Added {added} new contributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
