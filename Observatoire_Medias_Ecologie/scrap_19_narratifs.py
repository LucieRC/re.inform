"""Scrape Science Feedback page for the 19 'narratifs' and their debunk.

Target page (French):
https://science.feedback.org/fr/informations-trompeuses-climat-television-radio-france-analyse-ampleur-recits-frequents/

What it extracts for each narratif:
- number (1..19)
- qualifier (e.g. TROMPEUR, INFONDÉ, ...)
- claim (blockquote under the narratif heading)
- debunk (the full 'À RETENIR' section until the next narratif)
- resume (best-effort extraction from debunk)
- links (best-effort extraction from 'En savoir plus')

This script is dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Iterable, List, Optional


url_to_scrap = (
	"https://science.feedback.org/fr/informations-trompeuses-climat-television-radio-france-analyse-ampleur-recits-frequents/"
)


@dataclass(frozen=True)
class Narratif:
	number: int
	qualifier: Optional[str]
	title: str
	claim: str
	debunk: str
	resume: Optional[str]
	en_savoir_plus: List[str]


class _HTMLToText(HTMLParser):
	"""Very small HTML -> text converter.

	We preserve structure with newlines for block-ish tags.
	"""

	_BLOCK_TAGS = {
		"p",
		"div",
		"section",
		"article",
		"header",
		"footer",
		"aside",
		"br",
		"hr",
		"li",
		"ul",
		"ol",
		"blockquote",
		"h1",
		"h2",
		"h3",
		"h4",
		"h5",
		"h6",
		"table",
		"tr",
		"td",
		"th",
	}

	def __init__(self) -> None:
		super().__init__(convert_charrefs=False)
		self._chunks: List[str] = []

	def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
		tag = tag.lower()
		if tag in self._BLOCK_TAGS:
			self._chunks.append("\n")

	def handle_endtag(self, tag: str) -> None:
		tag = tag.lower()
		if tag in self._BLOCK_TAGS:
			self._chunks.append("\n")

	def handle_data(self, data: str) -> None:
		if data:
			self._chunks.append(data)

	def handle_entityref(self, name: str) -> None:
		self._chunks.append(unescape(f"&{name};"))

	def handle_charref(self, name: str) -> None:
		self._chunks.append(unescape(f"&#{name};"))

	def get_text(self) -> str:
		text = unescape("".join(self._chunks))
		text = text.replace("\u00a0", " ")
		text = re.sub(r"[ \t\f\v]+", " ", text)
		text = re.sub(r"\n{3,}", "\n\n", text)
		return text.strip()


def html_to_text(html_fragment: str) -> str:
	parser = _HTMLToText()
	parser.feed(html_fragment)
	parser.close()
	return parser.get_text()


def fetch_html(url: str, timeout: int = 30) -> str:
	req = urllib.request.Request(
		url,
		headers={
			"User-Agent": (
				"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
				"AppleWebKit/537.36 (KHTML, like Gecko) "
				"Chrome/120.0.0.0 Safari/537.36"
			),
			"Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
		},
		method="GET",
	)
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		charset = "utf-8"
		content_type = resp.headers.get("Content-Type")
		if content_type:
			m = re.search(r"charset=([\w\-]+)", content_type, flags=re.IGNORECASE)
			if m:
				charset = m.group(1)
		raw = resp.read()
	return raw.decode(charset, errors="replace")


_NARRATIF_H3_RE = re.compile(
	r"<h3\b[^>]*>\s*(NARRATIF\s*\d{2}[^<]*)</h3>",
	flags=re.IGNORECASE,
)


def _iter_narratif_sections(page_html: str) -> Iterable[tuple[str, str]]:
	"""Yield (heading_text, section_html) for each narratif."""
	matches = list(_NARRATIF_H3_RE.finditer(page_html))
	if not matches:
		raise RuntimeError(
			"No narratif headings found. The page structure may have changed."
		)

	for idx, m in enumerate(matches):
		start = m.start()
		end = matches[idx + 1].start() if idx + 1 < len(matches) else len(page_html)
		heading_text = html_to_text(m.group(0))
		section_html = page_html[start:end]
		yield heading_text, section_html


def _parse_heading(heading_text: str) -> tuple[int, Optional[str], str]:
	# Example: "NARRATIF 01 (TROMPEUR) :"
	number_match = re.search(r"NARRATIF\s*(\d{1,2})", heading_text, flags=re.IGNORECASE)
	if not number_match:
		raise RuntimeError(f"Could not parse narratif number from heading: {heading_text!r}")
	number = int(number_match.group(1))

	qualifier_match = re.search(r"\(([^)]+)\)", heading_text)
	qualifier = qualifier_match.group(1).strip() if qualifier_match else None

	title = heading_text.strip().rstrip(":").strip()
	return number, qualifier, title


def _extract_first_blockquote_text(section_html: str) -> str:
	m = re.search(
		r"<blockquote\b[^>]*>(.*?)</blockquote>",
		section_html,
		flags=re.IGNORECASE | re.DOTALL,
	)
	if not m:
		return ""
	return html_to_text(m.group(1))


def _extract_a_retenir_html(section_html: str) -> str:
	# Find the 'À RETENIR' heading inside the narratif block.
	# It can be written with accented A in the HTML text, so match loosely.
	m = re.search(
		r"<h2\b[^>]*>\s*(?:À|A)\s*RETENIR\s*:?.*?</h2>",
		section_html,
		flags=re.IGNORECASE | re.DOTALL,
	)
	if not m:
		return ""

	after = section_html[m.end() :]
	return after


def _best_effort_resume(debunk_text: str) -> Optional[str]:
	# Debunk text contains headings like:
	# "Résumé :" then text until "En savoir plus" or "References".
	m = re.search(
		r"\bRésumé\s*:\s*(.+?)(?=\n\s*(?:En savoir plus\s*:|References\s*:|Références\s*:|NARRATIF\s*\d{1,2}\b)|\Z)",
		debunk_text,
		flags=re.IGNORECASE | re.DOTALL,
	)
	if not m:
		return None

	resume = m.group(1).strip()
	resume = re.sub(r"\n{3,}", "\n\n", resume)
	return resume or None


def _best_effort_en_savoir_plus(debunk_text: str) -> List[str]:
	# The page uses bullet-like "•" in the text dump.
	# Extract between "En savoir plus:" and the next section header.
	m = re.search(
		r"\bEn savoir plus\s*:\s*(.+?)(?=\n\s*(?:References\s*:|Références\s*:|NARRATIF\s*\d{1,2}\b)|\Z)",
		debunk_text,
		flags=re.IGNORECASE | re.DOTALL,
	)
	if not m:
		return []

	block = m.group(1)
	lines = [ln.strip() for ln in block.splitlines()]
	items: List[str] = []
	for ln in lines:
		if not ln:
			continue
		ln = ln.lstrip("•").strip()
		# Keep only reasonable lines (avoid accidental paragraph carry-over)
		if len(ln) < 3:
			continue
		items.append(ln)
	# De-duplicate while preserving order
	seen = set()
	deduped: List[str] = []
	for item in items:
		key = item.lower()
		if key in seen:
			continue
		seen.add(key)
		deduped.append(item)
	return deduped


def extract_narratifs(page_html: str) -> List[Narratif]:
	narratifs: List[Narratif] = []
	for heading_text, section_html in _iter_narratif_sections(page_html):
		number, qualifier, title = _parse_heading(heading_text)
		claim = _extract_first_blockquote_text(section_html)
		a_retenir_html = _extract_a_retenir_html(section_html)
		debunk = html_to_text(a_retenir_html) if a_retenir_html else ""

		resume = _best_effort_resume(debunk) if debunk else None
		en_savoir_plus = _best_effort_en_savoir_plus(debunk) if debunk else []

		narratifs.append(
			Narratif(
				number=number,
				qualifier=qualifier,
				title=title,
				claim=claim,
				debunk=debunk,
				resume=resume,
				en_savoir_plus=en_savoir_plus,
			)
		)

	narratifs.sort(key=lambda n: n.number)
	return narratifs


def write_json(path: str, narratifs: List[Narratif]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump([asdict(n) for n in narratifs], f, ensure_ascii=False, indent=2)


def write_csv(path: str, narratifs: List[Narratif]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	fieldnames = [
		"number",
		"qualifier",
		"title",
		"claim",
		"resume",
		"debunk",
		"en_savoir_plus",
	]
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for n in narratifs:
			row = asdict(n)
			row["en_savoir_plus"] = " | ".join(n.en_savoir_plus)
			writer.writerow(row)


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Scrape the 19 climate misinformation narratifs and debunks from Science Feedback."
	)
	parser.add_argument(
		"--url",
		default=url_to_scrap,
		help="Page URL to scrape.",
	)
	default_out_dir = os.path.join(os.path.dirname(__file__), "output")
	parser.add_argument(
		"--out-json",
		default=os.path.join(default_out_dir, "science_feedback_19_narratifs.json"),
		help="Output JSON path.",
	)
	parser.add_argument(
		"--out-csv",
		default=os.path.join(default_out_dir, "science_feedback_19_narratifs.csv"),
		help="Output CSV path.",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=30,
		help="HTTP timeout (seconds).",
	)
	return parser


def main(argv: Optional[List[str]] = None) -> int:
	args = build_argparser().parse_args(argv)
	html = fetch_html(args.url, timeout=args.timeout)
	narratifs = extract_narratifs(html)

	if len(narratifs) != 19:
		print(
			f"Warning: extracted {len(narratifs)} narratifs (expected 19). "
			"The page structure may have changed.",
			file=sys.stderr,
		)

	write_json(args.out_json, narratifs)
	write_csv(args.out_csv, narratifs)

	print(f"Extracted {len(narratifs)} narratifs")
	print(f"- JSON: {args.out_json}")
	print(f"- CSV:  {args.out_csv}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())