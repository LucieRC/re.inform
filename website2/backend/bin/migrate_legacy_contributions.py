#!/usr/bin/env python3
"""
Move the old contribution files into the argumentation store.

argumentation.json is the single source of truth for claims and answers. Before
it, the site kept two other lists: submissions.json (the review queue of the
old free-text form) and contributions.json (approved entries). This script folds
both into the store, without creating duplicates:

  contributions.json  -> published claims and answers
  submissions.json    -> pending:  answer (and claim, if new) waiting for review
                         approved: same as a contribution (unless already one)
                         rejected: skipped

Test content is skipped, using the bin/test_data.py rule: a record flagged
"is_test": true, or written by an account flagged as test in users.db. Pass
--include-test to migrate it anyway (it keeps "is_test": true, so
`test_data.py export` still leaves it out).

Matching is by text. A claim matches when its text is the same once spacing,
case, quote marks and footnote markers like "[1]" are ignored; an answer
matches when it hangs off that claim with the same text under the same rule.
A match is not copied again; it only fills in what the store lacks (a source
link, topic tags). Running the script twice changes nothing the second time.

    python bin/migrate_legacy_contributions.py              # dry run on backend/data
    python bin/migrate_legacy_contributions.py --sandbox    # dry run on the test sandbox
    python bin/migrate_legacy_contributions.py --sandbox --apply
    python bin/migrate_legacy_contributions.py --data DIR [--apply]

--apply first writes timestamped backups next to each file it may touch
(e.g. argumentation.json.bak-20261107-142233), then saves the store. The two
legacy files are never modified; the app no longer reads them. They are read
from the data folder if present there, else from archive/website2-legacy-data/
(or --legacy-dir). Applying to the real backend/data also needs
--confirm-real-data.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

import argumentation_store as argstore  # noqa: E402

REAL_DATA = os.path.join(BACKEND, 'data')
SANDBOX_DATA = os.path.join(tempfile.gettempdir(), 'reinform-sandbox', 'data')  # same as bin/test_sandbox.py
# Where the two retired files were moved once migrated.
ARCHIVE_LEGACY = os.path.abspath(os.path.join(BACKEND, '..', '..', 'archive', 'website2-legacy-data'))

CONTRIBUTIONS = 'contributions.json'
SUBMISSIONS = 'submissions.json'
ARGUMENTATION = 'argumentation.json'
PERSONAS = argstore.PERSONA_ARCHETYPES_BASENAME

# Tags the importers used to say where an entry came from, not what it is
# about. The store already keeps that in imported_from / import_tags.
PROVENANCE_TAGS = {'bonpote', 'science_feedback', 'observatoire_medias_ecologie'}

_footnote_re = re.compile(r'\s*\[\d+(?:\s*[,;–-]\s*\d+)*\]')
_quotes = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"', '«': '"', '»': '"', ' ': ' '})


def loose(text: Any) -> str:
    """Comparison key: ignores case, spacing, quote style and [n] footnote marks."""
    t = str(text or '').translate(_quotes)
    t = _footnote_re.sub('', t)
    return argstore.normalize_text(t).lower()


def topic_tags(tags: Any, already: Optional[List[str]] = None) -> List[str]:
    skip = {str(t).lower() for t in (already or [])}
    out = []
    for t in tags if isinstance(tags, list) else []:
        t = argstore.normalize_text(str(t or ''))
        low = t.lower()
        if not t or low in PROVENANCE_TAGS or low.startswith('sf_') or low in skip:
            continue
        out.append(t)
    return out


def short(text: Any, n: int = 70) -> str:
    t = argstore.normalize_text(str(text or ''))
    return t if len(t) <= n else t[: n - 1] + '…'


class Plan:
    """What the run did (or would do), for the printed summary."""

    def __init__(self) -> None:
        self.lines: Dict[str, List[str]] = {}
        self.counts: Dict[str, int] = {}

    def add(self, bucket: str, line: str) -> None:
        self.lines.setdefault(bucket, []).append(line)
        self.counts[bucket] = self.counts.get(bucket, 0) + 1


class Matcher:
    """Finds existing claims and answers in the store by loose text."""

    def __init__(self, store: Dict[str, Any]) -> None:
        self.store = store
        self.args: Dict[str, Dict[str, Any]] = {}
        self.answers: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for a in store.get('arguments') or []:
            if isinstance(a, dict) and isinstance(a.get('id'), int):
                self.args.setdefault(loose(a.get('text')), a)
        for ca in store.get('counter_arguments') or []:
            if isinstance(ca, dict) and isinstance(ca.get('argument_id'), int):
                self.answers.setdefault((ca['argument_id'], loose(ca.get('text'))), ca)
        self.migrated = set()
        for item in (store.get('arguments') or []) + (store.get('counter_arguments') or []):
            imp = item.get('imported_from') if isinstance(item, dict) else None
            if isinstance(imp, dict) and imp.get('legacy_file'):
                self.migrated.add((imp['legacy_file'], imp.get('legacy_id')))

    def claim(self, text: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        key = loose(text)
        a = self.args.get(key)
        if not a:
            return None, ''
        return a, ('exact' if argstore.normalize_text(a.get('text') or '').lower()
                   == argstore.normalize_text(str(text or '')).lower() else 'near')

    def answer(self, argument_id: int, text: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        ca = self.answers.get((argument_id, loose(text)))
        if not ca:
            return None, ''
        return ca, ('exact' if argstore.normalize_text(ca.get('text') or '').lower()
                    == argstore.normalize_text(str(text or '')).lower() else 'near (footnote marks / quotes only)')

    def remember(self, arg: Dict[str, Any], ca: Optional[Dict[str, Any]] = None) -> None:
        self.args.setdefault(loose(arg.get('text')), arg)
        if ca is not None:
            self.answers.setdefault((ca['argument_id'], loose(ca.get('text'))), ca)


def test_account_ids(data_dir: str) -> set:
    """Accounts flagged as test in users.db (read only), like bin/test_data.py."""
    path = os.path.join(data_dir, 'users.db')
    if not os.path.exists(path):
        return set()
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        cols = {r[1] for r in conn.execute('PRAGMA table_info(users)')}
        ids = {r[0] for r in conn.execute('SELECT id FROM users WHERE is_test = 1')} if 'is_test' in cols else set()
        conn.close()
        return ids
    except sqlite3.Error:
        return set()


def test_reason(item: Dict[str, Any], test_ids: set) -> Optional[str]:
    if item.get('is_test') is True:
        return 'flagged is_test'
    if item.get('user_id') in test_ids and item.get('user_id') is not None:
        who = item.get('username') or f"#{item.get('user_id')}"
        return f'by test account {who}'
    return None


def migrate(store: Dict[str, Any], contributions: List[Any], submissions: List[Any], *,
            test_ids: Optional[set] = None, include_test: bool = False) -> Plan:
    plan = Plan()
    m = Matcher(store)
    test_ids = test_ids or set()

    def get_or_create_claim(text, *, tags, user_id, created_at, legacy, status=None):
        arg, how = m.claim(text)
        if arg:
            return arg, how, False
        arg = argstore.upsert_argument(store, text, created_by_user_id=user_id,
                                       tags=topic_tags(tags), imported_from=legacy, status=status)
        if created_at:
            arg['created_at'] = created_at
        m.remember(arg)
        return arg, 'new', True

    def merge_into(arg, ca, item, label):
        """Fill what the store lacks from a duplicate legacy entry."""
        added = []
        extra = topic_tags(item.get('tags'), already=(arg.get('tags') or []) + (arg.get('import_tags') or []))
        if extra:
            arg['tags'] = argstore._merge_unique(arg.get('tags'), extra)
            added.append(f"tags {extra}")
        src = str(item.get('source') or '').strip()
        if src and argstore.is_http_url(src) and not argstore.resource_urls_for(store, ca):
            if argstore.is_published(ca):
                res = argstore.upsert_resource(store, src, argument_ids=[arg['id']],
                                               created_by_user_id=ca.get('created_by_user_id'))
                ca['resource_ids'] = [res['id']]
            ca['source_url'] = src
            added.append('source link')
        if added:
            plan.add('merged', f"{label} -> claim #{arg['id']} / answer #{ca['id']}: added {', '.join(added)}")
        return bool(added)

    # ---- contributions.json: approved entries, published as they were.
    approved_submission_ids = set()
    for c in contributions:
        if not isinstance(c, dict):
            plan.add('skipped', 'contribution (not a record)')
            continue
        cid = c.get('id')
        label = f"contribution #{cid}"
        if isinstance(c.get('submission_id'), int):
            approved_submission_ids.add(c['submission_id'])
        why = test_reason(c, test_ids)
        if why and not include_test:
            plan.add('skipped', f"{label}: test content ({why})")
            continue
        if (CONTRIBUTIONS, cid) in m.migrated:
            plan.add('skipped', f"{label}: already migrated")
            continue
        narrative, answer = c.get('narrative'), c.get('counter_argument')
        if not argstore.normalize_text(str(narrative or '')) or not argstore.normalize_text(str(answer or '')):
            plan.add('skipped', f"{label}: missing claim or answer text")
            continue
        _migrate_one(plan, m, store, c, label, CONTRIBUTIONS, cid, narrative, answer,
                     status=None, created_at=c.get('date'), get_or_create_claim=get_or_create_claim,
                     merge_into=merge_into)

    # ---- submissions.json: the old review queue.
    for s in submissions:
        if not isinstance(s, dict):
            plan.add('skipped', 'submission (not a record)')
            continue
        sid = s.get('id')
        label = f"submission #{sid}"
        status = str(s.get('status') or 'pending').lower()
        why = test_reason(s, test_ids)
        if why and not include_test:
            plan.add('skipped', f"{label}: test content ({why}): {short(s.get('narrative'), 50)!r}")
            continue
        if status == 'rejected':
            plan.add('skipped', f"{label}: rejected by a reviewer")
            continue
        if status == 'approved' and sid in approved_submission_ids:
            plan.add('skipped', f"{label}: approved, already handled as a contribution")
            continue
        if (SUBMISSIONS, sid) in m.migrated:
            plan.add('skipped', f"{label}: already migrated")
            continue
        narrative, answer = s.get('narrative'), s.get('counter_argument')
        if not argstore.normalize_text(str(narrative or '')) or not argstore.normalize_text(str(answer or '')):
            plan.add('skipped', f"{label}: missing claim or answer text")
            continue
        _migrate_one(plan, m, store, s, label, SUBMISSIONS, sid, narrative, answer,
                     status=(None if status == 'approved' else argstore.STATUS_PENDING),
                     created_at=s.get('submitted_at'), get_or_create_claim=get_or_create_claim,
                     merge_into=merge_into)

    _report_store_duplicates(plan, store)
    return plan


def _migrate_one(plan, m, store, item, label, legacy_file, legacy_id, narrative, answer, *,
                 status, created_at, get_or_create_claim, merge_into):
    user_id = item.get('user_id') if isinstance(item.get('user_id'), int) else None
    legacy = {'legacy_file': legacy_file, 'legacy_id': legacy_id}
    if item.get('submission_id') is not None:
        legacy['submission_id'] = item.get('submission_id')

    arg, claim_how, claim_new = get_or_create_claim(
        narrative, tags=item.get('tags'), user_id=user_id, created_at=created_at, legacy=legacy, status=status)
    if claim_new and item.get('is_test') is True:
        arg['is_test'] = True

    existing, answer_how = (None, '') if claim_new else m.answer(arg['id'], answer)
    if existing:
        plan.add('duplicates', f"{label} = claim #{arg['id']} ({claim_how}) / answer #{existing['id']} ({answer_how}): "
                               f"{short(narrative, 50)!r}")
        merge_into(arg, existing, item, label)
        return

    src = str(item.get('source') or '').strip()
    note = ''
    if src and not argstore.is_http_url(src):
        note = ' (unsafe link dropped)'
        src = ''
    resource_ids: List[int] = []
    if src and status is None:
        res = argstore.upsert_resource(store, src, argument_ids=[arg['id']], created_by_user_id=user_id)
        resource_ids = [res['id']]
    ca = argstore.create_counter_argument(
        store, argument_id=arg['id'], text=answer, persona_archetype_id=None,
        source_url=src or None, resource_ids=resource_ids, created_by_user_id=user_id,
        imported_from=legacy, status=status)
    if created_at:
        ca['created_at'] = created_at
    if item.get('is_test') is True:
        ca['is_test'] = True
    m.remember(arg, ca)
    who = f" by user #{user_id}" if user_id is not None else ' (no author)'
    state = 'pending review' if status == argstore.STATUS_PENDING else 'published'
    claim_part = f"new claim #{arg['id']}" if claim_new else f"existing claim #{arg['id']} ({claim_how})"
    plan.add('created', f"{label}{who} -> {claim_part}, new answer #{ca['id']} [{state}]{note}: "
                        f"{short(narrative, 50)!r}")


def _report_store_duplicates(plan: Plan, store: Dict[str, Any]) -> None:
    """Duplicates already inside the store, for information (nothing is changed)."""
    seen: Dict[str, int] = {}
    for a in store.get('arguments') or []:
        if not isinstance(a, dict):
            continue
        k = loose(a.get('text'))
        if k in seen:
            plan.add('store duplicates', f"claim #{a['id']} looks like claim #{seen[k]}: {short(a.get('text'), 50)!r}")
        else:
            seen[k] = a['id']
    seen_ca: Dict[Tuple[Any, str], int] = {}
    for ca in store.get('counter_arguments') or []:
        if not isinstance(ca, dict):
            continue
        k = (ca.get('argument_id'), loose(ca.get('text')))
        if k in seen_ca:
            plan.add('store duplicates', f"answer #{ca['id']} looks like answer #{seen_ca[k]} on claim #{k[0]}")
        else:
            seen_ca[k] = ca['id']


def backup(paths: List[str]) -> List[str]:
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    made = []
    for p in paths:
        if os.path.exists(p):
            dest = f"{p}.bak-{stamp}"
            shutil.copy2(p, dest)
            made.append(dest)
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group()
    where.add_argument('--data', help=f'data folder (default: {REAL_DATA})')
    where.add_argument('--sandbox', action='store_true', help=f'use the test sandbox data ({SANDBOX_DATA})')
    parser.add_argument('--apply', action='store_true', help='write the changes (after backups); default is a dry run')
    parser.add_argument('--confirm-real-data', action='store_true',
                        help='required with --apply on the real backend/data folder')
    parser.add_argument('--legacy-dir', help='folder holding contributions.json / submissions.json '
                        f'(default: the data folder if they are there, else {ARCHIVE_LEGACY})')
    parser.add_argument('--include-test', action='store_true',
                        help='also migrate test content (it keeps is_test so exports leave it out)')
    parser.add_argument('--quiet', action='store_true', help='print counts only')
    args = parser.parse_args()

    data = os.path.abspath(SANDBOX_DATA if args.sandbox else (args.data or REAL_DATA))
    if not os.path.isdir(data):
        sys.exit(f'No data folder at {data}' + (' (start bin/test_sandbox.py once to create it)' if args.sandbox else ''))
    is_real = os.path.realpath(data) == os.path.realpath(REAL_DATA)
    if args.apply and is_real and not args.confirm_real_data:
        sys.exit('Refusing to write the real data folder without --confirm-real-data.')

    legacy_dir = args.legacy_dir or (
        data if any(os.path.exists(os.path.join(data, n)) for n in (CONTRIBUTIONS, SUBMISSIONS)) else ARCHIVE_LEGACY)
    paths = {name: os.path.join(data, name) for name in (ARGUMENTATION, PERSONAS)}
    paths.update({name: os.path.join(legacy_dir, name) for name in (CONTRIBUTIONS, SUBMISSIONS)})
    store = argstore.load_store(paths[ARGUMENTATION], paths[PERSONAS])
    contributions = argstore.load_json_file(paths[CONTRIBUTIONS], [])
    submissions = argstore.load_json_file(paths[SUBMISSIONS], [])
    contributions = contributions if isinstance(contributions, list) else []
    submissions = submissions if isinstance(submissions, list) else []

    before = {k: len(store.get(k) or []) for k in ('arguments', 'counter_arguments', 'resources')}
    plan = migrate(store, contributions, submissions,
                   test_ids=test_account_ids(data), include_test=args.include_test)
    after = {k: len(store.get(k) or []) for k in before}

    print(f"{'APPLY' if args.apply else 'DRY RUN (nothing written)'} on {data} (legacy files from {legacy_dir})")
    print(f"Read {len(contributions)} contributions and {len(submissions)} submissions; "
          f"store has {before['arguments']} claims, {before['counter_arguments']} answers, {before['resources']} resources.")
    order = ('created', 'duplicates', 'merged', 'skipped', 'store duplicates')
    for bucket in order:
        lines = plan.lines.get(bucket, [])
        print(f"\n{bucket.upper()}: {len(lines)}")
        if not args.quiet:
            for line in lines:
                print(f"  {line}")
    pending = sum(1 for ca in store.get('counter_arguments') or []
                  if isinstance(ca, dict) and argstore.status_of(ca) == argstore.STATUS_PENDING)
    print(f"\nStore after: {after['arguments']} claims (+{after['arguments'] - before['arguments']}), "
          f"{after['counter_arguments']} answers (+{after['counter_arguments'] - before['counter_arguments']}), "
          f"{after['resources']} resources (+{after['resources'] - before['resources']}); "
          f"{pending} answer(s) waiting for review.")

    if not args.apply:
        print('\nDry run only. Re-run with --apply to write (backups are made first).')
        return 0

    changed = plan.counts.get('created', 0) + plan.counts.get('merged', 0)
    if not changed:
        print('\nNothing to write.')
        return 0
    made = backup([paths[ARGUMENTATION], paths[PERSONAS]])
    print('\nBackups:')
    for p in made:
        print(f"  {p}")
    argstore.save_store(paths[ARGUMENTATION], store, paths[PERSONAS])
    print(f"Saved {paths[ARGUMENTATION]}. {CONTRIBUTIONS} and {SUBMISSIONS} were not modified.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
