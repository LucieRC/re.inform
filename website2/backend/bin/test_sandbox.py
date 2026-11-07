"""
Run the site on a throwaway copy of data/ for manual testing.

Nothing you do in the browser touches the real users.db or JSON files: the
sandbox lives in your temp directory and survives restarts until --reset.

    python bin/test_sandbox.py           # start (or resume) the sandbox
    python bin/test_sandbox.py --reset   # start over from a fresh copy

Test accounts are created on first start; their logins are printed below and
saved to <sandbox>/accounts.txt.
"""
import argparse
import os
import secrets
import shutil
import sys
import tempfile

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

SANDBOX = os.path.join(tempfile.gettempdir(), 'reinform-sandbox')
DATA = os.path.join(SANDBOX, 'data')
ACCOUNTS_FILE = os.path.join(SANDBOX, 'accounts.txt')
PORT = int(os.environ.get('PORT', 5055))

# username → users.role. Each one plays a part in the test scripts.
TEST_ACCOUNTS = {
    'test_author': 'registered',    # writes debunks
    'test_sourcer_a': 'registered', # apprentice Sourcer
    'test_sourcer_b': 'registered', # second apprentice, to settle a verdict together
    'test_moderator': 'moderator',  # settles a verdict alone
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reset', action='store_true', help='discard the sandbox and copy data/ again')
    args = parser.parse_args()

    if args.reset and os.path.exists(SANDBOX):
        shutil.rmtree(SANDBOX)
    if not os.path.exists(DATA):
        shutil.copytree(os.path.join(BACKEND, 'data'), DATA)

    import database as db
    db.DATABASE_PATH = os.path.join(DATA, 'users.db')
    import app as site
    for name in ('RESOURCES_FILE', 'GALLERY_FILE',
                 'BONPOTE_GALLERY_FILE', 'ARGUMENTATION_FILE', 'PERSONA_ARCHETYPES_FILE',
                 'RESOURCE_REVIEWS_FILE'):
        setattr(site, name, os.path.join(DATA, os.path.basename(getattr(site, name))))
    site.ARTICLES_DIR = os.path.join(DATA, 'articles')
    site.AVATARS_DIR = os.path.join(DATA, 'avatars')
    db.init_db()

    if not os.path.exists(ACCOUNTS_FILE):
        # One shared random password: these accounts exist only in the sandbox.
        password = secrets.token_urlsafe(9)
        for username, role in TEST_ACCOUNTS.items():
            if not db.get_user_by_username(username):
                uid = db.create_user(username, f'{username}@sandbox.test', password)
                db.update_user_role(uid, role)
        with open(ACCOUNTS_FILE, 'w', encoding='utf-8') as f:
            f.write(f'password (all accounts): {password}\n')
            for username, role in TEST_ACCOUNTS.items():
                f.write(f'{username}  ({role})\n')

    print('\n' + '=' * 60)
    print(' re.inform TEST SANDBOX: changes here never reach real data')
    print(f' Open http://localhost:{PORT}')
    print(f' Data copy: {DATA}')
    print(' Test accounts:')
    with open(ACCOUNTS_FILE, encoding='utf-8') as f:
        for line in f:
            print('   ' + line.rstrip())
    print('=' * 60 + '\n')

    site.app.run(host='127.0.0.1', port=PORT, debug=False)


if __name__ == '__main__':
    main()
