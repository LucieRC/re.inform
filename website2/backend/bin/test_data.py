"""
Flag test data, and leave it out when publishing.

Nothing is deleted from your working data. Test content is *flagged*, and
`export` writes a clean copy of data/ for the published site without it.

What counts as test data:
  - accounts flagged as test (users.is_test = 1), and everything they created:
    claims, responses, resources, ratings, source checks, follows, roles,
    avatars. Content a test account posts later
    is caught automatically, with no extra step.
  - any record with "is_test": true, for test content that has no author
    (e.g. a curated resource with no author).

    python bin/test_data.py report                      # what is flagged, what looks like a test
    python bin/test_data.py mark-user alice bob         # flag accounts  (--unmark to undo)
    python bin/test_data.py mark response 12            # flag one record (--unmark to undo)
    python bin/test_data.py export --out ../publish-data

Record kinds for `mark`: claim, response, resource, curated, gallery.

Deploy the exported folder as backend/data/ on the published site.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sqlite3
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(BACKEND, 'data')

# kind → (file, key of the list inside it, or None when the file is the list)
KINDS = {
    'claim': ('argumentation.json', 'arguments'),
    'response': ('argumentation.json', 'counter_arguments'),
    'resource': ('argumentation.json', 'resources'),
    'curated': ('resources.json', None),
    'gallery': ('gallery.json', None),
}

# Plain list files whose records may carry a user_id or an is_test flag.
LIST_FILES = ('resources.json', 'gallery.json')

# Tables that point at users, and the columns that do it.
USER_REFS = {
    'endorsements': ('endorser_id', 'endorsed_id'),
    'votes': ('user_id',),
    'moderator_nominations': ('nominator_id', 'nominee_id'),
    'rate_limits': ('user_id',),
    'user_followed_tags': ('user_id',),
    'user_follows': ('follower_id', 'followed_id'),
    'user_roles': ('user_id',),
}

# Only a hint for `report`: accounts are never flagged on a name alone.
_SUSPECT = re.compile(r'test|example\.(com|org|net)|sandbox|@test\.|\.test$|^(alice|bob|mod_\w+)$', re.I)


# ---------------------------------------------------------------- helpers

def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def records(data, key):
    items = data.get(key) if key else data
    return items if isinstance(items, list) else []


def connect(data_dir):
    conn = sqlite3.connect(os.path.join(data_dir, 'users.db'))
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute('PRAGMA table_info(users)')}
    if 'is_test' not in cols:
        conn.execute('ALTER TABLE users ADD COLUMN is_test INTEGER NOT NULL DEFAULT 0')
        conn.commit()
    return conn


def test_user_ids(conn):
    return {r['id'] for r in conn.execute('SELECT id FROM users WHERE is_test = 1')}


def is_test(record, test_ids):
    if not isinstance(record, dict):
        return False
    if record.get('is_test') is True:
        return True
    return any(record.get(k) in test_ids for k in ('created_by_user_id', 'user_id') if record.get(k) is not None)


def _urls(record):
    for key in ('url', 'source', 'source_url'):
        value = record.get(key) if isinstance(record, dict) else None
        if isinstance(value, str) and value.strip():
            yield value


# ---------------------------------------------------------------- commands

def cmd_report(args):
    conn = connect(args.data)
    test_ids = test_user_ids(conn)
    users = conn.execute('SELECT id, username, email, is_test FROM users ORDER BY id').fetchall()

    print('Test accounts:')
    for u in users:
        if u['is_test']:
            print(f"  #{u['id']} {u['username']}")
    if not test_ids:
        print('  (none)')

    suspects = [u for u in users if not u['is_test'] and (_SUSPECT.search(u['username']) or _SUSPECT.search(u['email'] or ''))]
    if suspects:
        print('\nLook like tests but are not flagged (check, then `mark-user`):')
        for u in suspects:
            print(f"  #{u['id']} {u['username']}  <{u['email']}>")

    print('\nTest content (left out of an export):')
    found = 0
    for kind, (fname, key) in KINDS.items():
        path = os.path.join(args.data, fname)
        if not os.path.exists(path):
            continue
        for r in records(load(path), key):
            if is_test(r, test_ids):
                found += 1
                why = 'flagged' if r.get('is_test') is True else 'by a test account'
                label = r.get('text') or r.get('title') or r.get('narrative') or next(_urls(r), '')
                print(f"  {kind} {r.get('id')}: {str(label)[:70]!r}  ({why})")
    if not found:
        print('  (none)')


def cmd_mark_user(args):
    conn = connect(args.data)
    value = 0 if args.unmark else 1
    for name in args.usernames:
        cur = conn.execute('UPDATE users SET is_test = ? WHERE username = ?', (value, name))
        print(f"{'unflagged' if args.unmark else 'flagged'}: {name}" if cur.rowcount else f'no such user: {name}')
    conn.commit()


def cmd_mark(args):
    fname, key = KINDS[args.kind]
    path = os.path.join(args.data, fname)
    data = load(path)
    wanted = set(args.ids)
    hit = set()
    for r in records(data, key):
        if isinstance(r, dict) and str(r.get('id')) in wanted:
            if args.unmark:
                r.pop('is_test', None)
            else:
                r['is_test'] = True
            hit.add(str(r.get('id')))
    save(path, data)
    for i in args.ids:
        print(f"{'unflagged' if args.unmark else 'flagged'}: {args.kind} {i}" if i in hit else f'not found: {args.kind} {i}')


def cmd_export(args):
    src = os.path.abspath(args.data)
    out = os.path.abspath(args.out)
    if out == src or out.startswith(src + os.sep):
        sys.exit('--out must be outside the data folder')
    if os.path.exists(out):
        if not args.force:
            sys.exit(f'{out} already exists (use --force to replace it)')
        shutil.rmtree(out)

    # users.db is copied with SQLite's backup API, so a running server can't hand
    # us a half-written file; everything else is copied as is, except the
    # timestamped backups migration scripts leave next to the files (*.bak-*):
    # they still hold any test content and must not be published.
    shutil.copytree(src, out, ignore=shutil.ignore_patterns('users.db', 'users.db-*', '*.bak-*'))
    with sqlite3.connect(os.path.join(src, 'users.db')) as live, sqlite3.connect(os.path.join(out, 'users.db')) as copy:
        live.backup(copy)

    conn = connect(out)
    test_ids = test_user_ids(conn)
    removed_urls, counts = set(), {}

    def drop(kind, items, pred):
        kept = []
        for r in items:
            if pred(r):
                counts[kind] = counts.get(kind, 0) + 1
                removed_urls.update(_urls(r))
            else:
                kept.append(r)
        return kept

    # Argumentation store, with cascades: a test claim takes its responses with
    # it, a removed response takes its ratings and source checks.
    path = os.path.join(out, 'argumentation.json')
    if os.path.exists(path):
        s = load(path)
        s['arguments'] = drop('claims', s.get('arguments', []), lambda r: is_test(r, test_ids))
        arg_ids = {a.get('id') for a in s['arguments']}
        s['counter_arguments'] = drop('responses', s.get('counter_arguments', []),
                                      lambda r: is_test(r, test_ids) or r.get('argument_id') not in arg_ids)
        ca_ids = {c.get('id') for c in s['counter_arguments']}
        s['resources'] = drop('resources', s.get('resources', []), lambda r: is_test(r, test_ids))
        res_ids = {r.get('id') for r in s['resources']}
        for name in ('ratings', 'source_checks'):
            s[name] = drop(name, s.get(name, []),
                           lambda r: is_test(r, test_ids) or r.get('counter_argument_id') not in ca_ids)
        s['importance_ratings'] = drop('importance ratings', s.get('importance_ratings', []),
                                       lambda r: is_test(r, test_ids) or r.get('argument_id') not in arg_ids)
        for c in s['counter_arguments']:
            if isinstance(c.get('resource_ids'), list):
                c['resource_ids'] = [i for i in c['resource_ids'] if i in res_ids]
        for r in s['resources']:
            if isinstance(r.get('argument_ids'), list):
                r['argument_ids'] = [i for i in r['argument_ids'] if i in arg_ids]
        save(path, s)

    kept_urls = set()
    for fname in LIST_FILES:
        path = os.path.join(out, fname)
        if not os.path.exists(path):
            continue
        items = drop(fname.replace('.json', ''), load(path), lambda r: is_test(r, test_ids))
        for r in items:
            kept_urls.update(_urls(r))
        save(path, items)
    if os.path.exists(os.path.join(out, 'argumentation.json')):
        s = load(os.path.join(out, 'argumentation.json'))
        for name in ('resources', 'counter_arguments'):
            for r in s.get(name, []):
                kept_urls.update(_urls(r))

    # Review decisions about a URL that only test content cited.
    path = os.path.join(out, 'resource_reviews.json')
    if os.path.exists(path):
        sys.path.insert(0, BACKEND)
        from resource_library import entry_key
        gone = {entry_key(u) for u in removed_urls} - {entry_key(u) for u in kept_urls}
        reviews = load(path)
        before = len(reviews.get('reviews', {}))
        reviews['reviews'] = {k: v for k, v in reviews.get('reviews', {}).items() if k not in gone}
        if before - len(reviews['reviews']):
            counts['resource reviews'] = before - len(reviews['reviews'])
        save(path, reviews)

    # Accounts, their avatars and every row pointing at them.
    if test_ids:
        marks = ','.join('?' * len(test_ids))
        ids = tuple(test_ids)
        for row in conn.execute(f'SELECT avatar_file FROM users WHERE id IN ({marks})', ids):
            if row['avatar_file']:
                for f in glob.glob(os.path.join(out, 'avatars', os.path.basename(row['avatar_file']))):
                    os.remove(f)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, cols in USER_REFS.items():
            if table in tables:
                where = ' OR '.join(f'{c} IN ({marks})' for c in cols)
                conn.execute(f'DELETE FROM {table} WHERE {where}', ids * len(cols))
        conn.execute(f'DELETE FROM users WHERE id IN ({marks})', ids)
        counts['accounts'] = len(test_ids)
    conn.commit()
    conn.execute('VACUUM')  # so deleted rows can't be read back from free pages
    conn.close()

    print(f'Clean copy written to {out}')
    for kind, n in counts.items():
        print(f'  left out {n} {kind}')
    if not counts:
        print('  nothing flagged as test')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA, help='data folder (default: backend/data)')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('report', help='list test accounts and test content')

    p = sub.add_parser('mark-user', help='flag accounts as test accounts')
    p.add_argument('usernames', nargs='+')
    p.add_argument('--unmark', action='store_true')

    p = sub.add_parser('mark', help='flag one or more records as test')
    p.add_argument('kind', choices=sorted(KINDS))
    p.add_argument('ids', nargs='+')
    p.add_argument('--unmark', action='store_true')

    p = sub.add_parser('export', help='write a copy of data/ without test data')
    p.add_argument('--out', required=True)
    p.add_argument('--force', action='store_true', help='replace --out if it exists')

    args = parser.parse_args()
    {'report': cmd_report, 'mark-user': cmd_mark_user, 'mark': cmd_mark, 'export': cmd_export}[args.command](args)


if __name__ == '__main__':
    main()
