#!/usr/bin/env python3

import os
import sys

# Ensure imports work when run as a script
SCRIPT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, BACKEND_DIR)

from bonpote_scraper import write_bonpote_gallery_json  # noqa: E402


def main() -> int:
    data_dir = os.path.join(BACKEND_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "bonpote_gallery.json")
    items = write_bonpote_gallery_json(out_path)

    print(f"Wrote {len(items)} items to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
