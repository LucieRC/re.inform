#!/usr/bin/env python3

from __future__ import annotations

import ast
import os
import sys
from typing import Any, Dict, List, Optional


# Ensure imports work when run as a script
SCRIPT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, BACKEND_DIR)

from argumentation_store import (  # noqa: E402
    create_counter_argument,
    default_store,
    load_store,
    save_store,
    upsert_argument,
    upsert_persona_archetype,
)


def _parse_bonpote_file(path: str) -> List[Dict[str, Any]]:
    """Parse BonPoteJS_JSON.json.

    This file is NOT strict JSON: it's a comma-separated sequence of JSON-looking
    objects (missing the surrounding array) and uses python-style escapes like
    \\xNN and \\' inside strings.

    We parse it as a Python literal (list[dict]) after a small normalization.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    def _split_objects(text: str) -> List[str]:
        chunks: List[str] = []

        in_string = False
        escape = False
        depth = 0
        start = None

        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        chunks.append(text[start : i + 1])
                        start = None

        return chunks

    out: List[Dict[str, Any]] = []
    for chunk in _split_objects(raw):
        # Convert JSON null/true/false to Python for literal_eval
        py_src = chunk.replace(":null", ":None").replace(":true", ":True").replace(":false", ":False")
        try:
            item = ast.literal_eval(py_src)
        except Exception:
            # Skip malformed rows rather than failing the whole import.
            continue
        if isinstance(item, dict):
            out.append(item)

    return out


def _get_str(d: Dict[str, Any], key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def main() -> int:
    data_dir = os.path.join(BACKEND_DIR, "data")
    in_path = os.path.join(data_dir, "BonPoteJS_JSON.json")
    out_path = os.path.join(data_dir, "argumentation.json")

    if not os.path.exists(in_path):
        print(f"Input not found: {in_path}")
        return 2

    store = load_store(out_path)
    if not store:
        store = default_store()

    rows = _parse_bonpote_file(in_path)

    imported = 0
    skipped = 0
    for row in rows:
        name = _get_str(row, "Nom")
        description = _get_str(row, "Description")
        argument_text = _get_str(row, "Arguments")
        response_text = _get_str(row, "Réponse")
        source_url = _get_str(row, "En savoir plus")

        # Skip separators / empty rows
        if not argument_text or not response_text:
            skipped += 1
            continue

        imported_from = {
            "project": "BonPoteJS",
            "source_file": "BonPoteJS_JSON.json",
        }

        persona_id = None
        if name:
            persona = upsert_persona_archetype(
                store,
                name=name,
                description=description,
                imported_from=imported_from,
            )
            persona_id = persona["id"]

        arg = upsert_argument(
            store,
            text=argument_text,
            imported_from=imported_from,
        )

        create_counter_argument(
            store,
            argument_id=arg["id"],
            text=response_text,
            persona_archetype_id=persona_id,
            source_url=source_url,
            imported_from=imported_from,
        )

        imported += 1

    save_store(out_path, store)

    print(f"Imported rows: {imported}")
    print(f"Skipped rows (missing argument/response): {skipped}")
    print(f"Wrote store: {out_path}")
    print(
        f"Counts: arguments={len(store.get('arguments', []))}, personas={len(store.get('persona_archetypes', []))}, "
        f"counter_arguments={len(store.get('counter_arguments', []))}, ratings={len(store.get('ratings', []))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
