"""
Database module for re.inform user authentication and reputation system
"""
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'users.db')

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_columns(cursor, table, columns):
    """Add missing columns to an existing table (lightweight migration).

    CREATE TABLE IF NOT EXISTS never updates an already-created table, so
    columns added after the first launch need this ALTER TABLE pass.
    """
    cursor.execute(f'PRAGMA table_info({table})')
    existing = {row[1] for row in cursor.fetchall()}
    for name, decl in columns:
        if name not in existing:
            cursor.execute(f'ALTER TABLE {table} ADD COLUMN {name} {decl}')

def init_db():
    """Initialize database with schema."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'registered',
            reputation_score INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')

    # Avatar columns: filename in data/avatars/ and a timestamp that doubles as
    # the cache-buster query param in avatar URLs.
    _ensure_columns(cursor, 'users', [
        ('avatar_file', 'TEXT'),
        ('avatar_updated_at', 'TEXT'),
        # Test accounts: they and everything they create are left out of a
        # publish export (see bin/test_data.py).
        ('is_test', 'INTEGER NOT NULL DEFAULT 0'),
    ])
    
    # Endorsements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS endorsements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endorser_id INTEGER NOT NULL,
            endorsed_id INTEGER NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (endorser_id) REFERENCES users(id),
            FOREIGN KEY (endorsed_id) REFERENCES users(id),
            UNIQUE(endorser_id, endorsed_id)
        )
    ''')
    
    # Votes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            submission_id INTEGER NOT NULL,
            vote_type TEXT NOT NULL,
            weight INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, submission_id)
        )
    ''')
    
    # Moderator nominations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS moderator_nominations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nominator_id INTEGER NOT NULL,
            nominee_id INTEGER NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            reviewed_by INTEGER,
            FOREIGN KEY (nominator_id) REFERENCES users(id),
            FOREIGN KEY (nominee_id) REFERENCES users(id),
            FOREIGN KEY (reviewed_by) REFERENCES users(id)
        )
    ''')
    
    # Rate limiting table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            ip_address TEXT,
            action_type TEXT NOT NULL,
            action_count INTEGER DEFAULT 1,
            window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, action_type, window_start)
        )
    ''')
    
    # User followed tags table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_followed_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, tag)
        )
    ''')
    
    # User follows (following other users) table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_follows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            follower_id INTEGER NOT NULL,
            followed_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (follower_id) REFERENCES users(id),
            FOREIGN KEY (followed_id) REFERENCES users(id),
            UNIQUE(follower_id, followed_id)
        )
    ''')

    # Contributor roles (function), distinct from users.role (rights).
    # scope_kind/scope_id are '' / 0 for site-wide roles rather than NULL,
    # because SQLite treats NULLs as distinct and UNIQUE would never fire.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT 'apprentice',
            origin TEXT NOT NULL DEFAULT 'declared',
            scope_kind TEXT NOT NULL DEFAULT '',
            scope_id INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, role, scope_kind, scope_id)
        )
    ''')

    # Votes cast before signing up, keyed by an anonymous session id. They
    # don't count anywhere until claimed: on register/login they become the
    # user's own ratings (see _claim_pending_votes in app.py), then these rows
    # are deleted.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anon_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            persona_id INTEGER,
            score INTEGER NOT NULL,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_votes_anon ON pending_votes(anon_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_votes_ip ON pending_votes(ip_address, created_at)')

    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_PATH}")

# ========================================
# User Management Functions
# ========================================

def create_user(username, email, password):
    """Create a new user."""
    conn = get_db()
    cursor = conn.cursor()
    
    password_hash = generate_password_hash(password)
    
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError as e:
        conn.close()
        raise ValueError("Username or email already exists")

def get_user_by_id(user_id):
    """Get user by ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_username(username):
    """Get user by username."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_email(email):
    """Get user by email."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def verify_password(user, password):
    """Verify user password."""
    return check_password_hash(user['password_hash'], password)

def update_user_reputation(user_id, points):
    """Update user reputation score."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users 
        SET reputation_score = reputation_score + ?
        WHERE id = ?
    ''', (points, user_id))
    conn.commit()
    
    # Check if user should be promoted to trusted
    cursor.execute('SELECT reputation_score FROM users WHERE id = ?', (user_id,))
    reputation = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM endorsements WHERE endorsed_id = ?', (user_id,))
    endorsement_count = cursor.fetchone()[0]
    
    if reputation >= 50 and endorsement_count >= 3:
        cursor.execute('''
            UPDATE users 
            SET role = 'trusted'
            WHERE id = ? AND role = 'registered'
        ''', (user_id,))
        conn.commit()
    
    conn.close()

def update_user_role(user_id, new_role):
    """Update user role."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
    conn.commit()
    conn.close()

def set_user_avatar(user_id, file_name):
    """Record a user's profile-picture filename and bump the cache-buster."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users
        SET avatar_file = ?, avatar_updated_at = ?
        WHERE id = ?
    ''', (file_name, datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()

def clear_user_avatar(user_id):
    """Remove a user's profile picture."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users
        SET avatar_file = NULL, avatar_updated_at = ?
        WHERE id = ?
    ''', (datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()

# ========================================
# Endorsement Functions
# ========================================

def create_endorsement(endorser_id, endorsed_id, reason):
    """Create an endorsement."""
    if endorser_id == endorsed_id:
        raise ValueError("Cannot endorse yourself")
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO endorsements (endorser_id, endorsed_id, reason)
            VALUES (?, ?, ?)
        ''', (endorser_id, endorsed_id, reason))
        conn.commit()
        conn.close()
        
        # Update reputation
        update_user_reputation(endorsed_id, 5)
        return True
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Already endorsed this user")

def get_user_endorsements(user_id):
    """Get endorsements received by a user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT e.*, u.username as endorser_username
        FROM endorsements e
        JOIN users u ON e.endorser_id = u.id
        WHERE e.endorsed_id = ?
        ORDER BY e.created_at DESC
    ''', (user_id,))
    endorsements = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return endorsements

# ========================================
# Voting Functions
# ========================================

def create_vote(user_id, submission_id, vote_type, weight=1):
    """Create or update a vote."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO votes (user_id, submission_id, vote_type, weight)
        VALUES (?, ?, ?, ?)
    ''', (user_id, submission_id, vote_type, weight))
    conn.commit()
    conn.close()

def get_submission_votes(submission_id):
    """Get all votes for a submission."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT v.*, u.username, u.role
        FROM votes v
        JOIN users u ON v.user_id = u.id
        WHERE v.submission_id = ?
        ORDER BY v.created_at DESC
    ''', (submission_id,))
    votes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return votes

def get_vote_summary(submission_id):
    """Get vote summary with weighted scores."""
    votes = get_submission_votes(submission_id)
    
    approve_score = sum(v['weight'] for v in votes if v['vote_type'] == 'approve')
    reject_score = sum(v['weight'] for v in votes if v['vote_type'] == 'reject')
    needs_work_count = sum(1 for v in votes if v['vote_type'] == 'needs_work')
    
    return {
        'approve_score': approve_score,
        'reject_score': reject_score,
        'needs_work_count': needs_work_count,
        'total_votes': len(votes),
        'votes': votes
    }

# ========================================
# Moderator Nomination Functions
# ========================================

def nominate_moderator(nominator_id, nominee_id, reason):
    """Nominate a user for moderator role."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO moderator_nominations (nominator_id, nominee_id, reason)
            VALUES (?, ?, ?)
        ''', (nominator_id, nominee_id, reason))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def get_pending_nominations():
    """Get all pending moderator nominations."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT n.*, 
               u1.username as nominator_username,
               u2.username as nominee_username,
               u2.reputation_score as nominee_reputation
        FROM moderator_nominations n
        JOIN users u1 ON n.nominator_id = u1.id
        JOIN users u2 ON n.nominee_id = u2.id
        WHERE n.status = 'pending'
        ORDER BY n.created_at DESC
    ''')
    nominations = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return nominations

def review_nomination(nomination_id, reviewer_id, approved):
    """Review a moderator nomination."""
    conn = get_db()
    cursor = conn.cursor()
    
    status = 'approved' if approved else 'rejected'
    cursor.execute('''
        UPDATE moderator_nominations
        SET status = ?, reviewed_at = CURRENT_TIMESTAMP, reviewed_by = ?
        WHERE id = ?
    ''', (status, reviewer_id, nomination_id))
    
    if approved:
        cursor.execute('SELECT nominee_id FROM moderator_nominations WHERE id = ?', (nomination_id,))
        nominee_id = cursor.fetchone()[0]
        cursor.execute('UPDATE users SET role = "moderator" WHERE id = ?', (nominee_id,))
    
    conn.commit()
    conn.close()

# ========================================
# Rate Limiting Functions
# ========================================

def check_rate_limit(user_id, ip_address, action_type, limit):
    """Check if action is within rate limit."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Clean old entries (older than 24 hours)
    cursor.execute('''
        DELETE FROM rate_limits
        WHERE datetime(window_start) < datetime('now', '-1 day')
    ''')
    
    # Check current count. Each call lands in its own row (window_start is the
    # call's timestamp), so the count over the last day is the SUM — reading a
    # single row meant no limit ever triggered. Signed-in callers are counted
    # per account; anonymous ones per IP.
    if user_id:
        cursor.execute('''
            SELECT COALESCE(SUM(action_count), 0) FROM rate_limits
            WHERE user_id = ? AND action_type = ?
            AND datetime(window_start) >= datetime('now', '-1 day')
        ''', (user_id, action_type))
    else:
        cursor.execute('''
            SELECT COALESCE(SUM(action_count), 0) FROM rate_limits
            WHERE user_id IS NULL AND ip_address = ? AND action_type = ?
            AND datetime(window_start) >= datetime('now', '-1 day')
        ''', (ip_address, action_type))

    result = cursor.fetchone()

    if result and result[0] >= limit:
        conn.close()
        return False
    
    # Increment counter
    cursor.execute('''
        INSERT INTO rate_limits (user_id, ip_address, action_type, action_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(user_id, action_type, window_start) 
        DO UPDATE SET action_count = action_count + 1
    ''', (user_id, ip_address, action_type))
    
    conn.commit()
    conn.close()
    return True

# ========================================
# Tag Following Functions
# ========================================

# Followable topics: the tags claims are actually filed under, so following one
# can surface something. Keep in sync with the tags in argumentation.json.
PREDEFINED_TAGS = [
    'climate', 'energy', 'transport', 'agriculture', 'biodiversity',
    'adaptation', 'economy', 'policy', 'society', 'technology', 'media',
]

def get_all_tags():
    """Get all available tags (predefined list)."""
    return PREDEFINED_TAGS

def get_followed_tags(user_id):
    """Get all tags a user is following."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT tag, created_at FROM user_followed_tags
        WHERE user_id = ?
        ORDER BY tag
    ''', (user_id,))
    tags = [{'tag': row['tag'], 'followed_at': row['created_at']} for row in cursor.fetchall()]
    conn.close()
    return tags

def follow_tag(user_id, tag):
    """Follow a tag."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO user_followed_tags (user_id, tag)
            VALUES (?, ?)
        ''', (user_id, tag))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Already following

def unfollow_tag(user_id, tag):
    """Unfollow a tag."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM user_followed_tags
        WHERE user_id = ? AND tag = ?
    ''', (user_id, tag))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def is_following_tag(user_id, tag):
    """Check if user is following a tag."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 1 FROM user_followed_tags
        WHERE user_id = ? AND tag = ?
    ''', (user_id, tag))
    result = cursor.fetchone() is not None
    conn.close()
    return result

# ========================================
# User Following Functions
# ========================================

def get_following(user_id):
    """Get all users that a user is following."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.id, u.username, u.role, u.reputation_score, uf.created_at as followed_at
        FROM user_follows uf
        JOIN users u ON uf.followed_id = u.id
        WHERE uf.follower_id = ?
        ORDER BY uf.created_at DESC
    ''', (user_id,))
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

def get_followers(user_id):
    """Get all users following a user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.id, u.username, u.role, u.reputation_score, uf.created_at as followed_at
        FROM user_follows uf
        JOIN users u ON uf.follower_id = u.id
        WHERE uf.followed_id = ?
        ORDER BY uf.created_at DESC
    ''', (user_id,))
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

def follow_user(follower_id, followed_id):
    """Follow another user."""
    if follower_id == followed_id:
        return False  # Can't follow yourself
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO user_follows (follower_id, followed_id)
            VALUES (?, ?)
        ''', (follower_id, followed_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Already following

def unfollow_user(follower_id, followed_id):
    """Unfollow a user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM user_follows
        WHERE follower_id = ? AND followed_id = ?
    ''', (follower_id, followed_id))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def is_following_user(follower_id, followed_id):
    """Check if a user is following another user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 1 FROM user_follows
        WHERE follower_id = ? AND followed_id = ?
    ''', (follower_id, followed_id))
    result = cursor.fetchone() is not None
    conn.close()
    return result

def get_following_count(user_id):
    """Get count of users a user is following."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM user_follows WHERE follower_id = ?', (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_followers_count(user_id):
    """Get count of followers for a user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM user_follows WHERE followed_id = ?', (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

# ========================================
# Contributor Role Functions
# ========================================

def get_user_roles(user_id):
    """All contributor roles a user holds, oldest first."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT role, tier, origin, scope_kind, scope_id, created_at, updated_at
        FROM user_roles
        WHERE user_id = ?
        ORDER BY created_at, id
    ''', (user_id,))
    roles = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return roles

def add_user_role(user_id, role, origin='declared'):
    """Give a user a site-wide role as an apprentice. Returns False if already held."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO user_roles (user_id, role, origin)
            VALUES (?, ?, ?)
        ''', (user_id, role, origin))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def remove_user_role(user_id, role):
    """Drop a site-wide role. Evidence is kept, so re-declaring restores the tier."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM user_roles
        WHERE user_id = ? AND role = ? AND scope_kind = '' AND scope_id = 0
    ''', (user_id, role))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def set_user_role_tier(user_id, role, tier):
    """Set the tier of a site-wide role the user already holds."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE user_roles
        SET tier = ?, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ? AND role = ? AND scope_kind = '' AND scope_id = 0
    ''', (tier, user_id, role))
    conn.commit()
    conn.close()

def get_role_tiers(role):
    """Map user_id → tier for everyone holding a site-wide role."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_id, tier FROM user_roles
        WHERE role = ? AND scope_kind = '' AND scope_id = 0
    ''', (role,))
    tiers = {row['user_id']: row['tier'] for row in cursor.fetchall()}
    conn.close()
    return tiers

def get_user_ids_by_site_role(*site_roles):
    """IDs of users whose rights role (users.role) is one of site_roles."""
    if not site_roles:
        return set()
    conn = get_db()
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in site_roles)
    cursor.execute(f'SELECT id FROM users WHERE role IN ({placeholders})', site_roles)
    ids = {row['id'] for row in cursor.fetchall()}
    conn.close()
    return ids

if __name__ == '__main__':
    init_db()


# ========================================
# Pending votes (cast before signing up)
# ========================================

PENDING_VOTE_KINDS = ('importance', 'effectiveness')
PENDING_VOTE_TTL_DAYS = 30


def add_pending_vote(anon_id, kind, target_id, score, persona_id=None, ip_address=None):
    """Store a vote from a visitor without an account.

    A visitor re-voting on the same thing replaces their previous vote, so
    pending votes can't stack. Votes older than the TTL are dropped.
    """
    if kind not in PENDING_VOTE_KINDS:
        raise ValueError(f'Unknown vote kind: {kind}')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM pending_votes WHERE datetime(created_at) < datetime('now', ?)",
        (f'-{PENDING_VOTE_TTL_DAYS} days',),
    )
    cursor.execute(
        'DELETE FROM pending_votes WHERE anon_id = ? AND kind = ? AND target_id = ? AND persona_id IS ?',
        (anon_id, kind, target_id, persona_id),
    )
    cursor.execute(
        'INSERT INTO pending_votes (anon_id, kind, target_id, persona_id, score, ip_address) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (anon_id, kind, target_id, persona_id, score, ip_address),
    )
    conn.commit()
    conn.close()


def get_pending_votes(anon_id):
    """All votes waiting for this visitor to sign up, oldest first."""
    if not anon_id:
        return []
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM pending_votes WHERE anon_id = ? AND datetime(created_at) >= datetime('now', ?) ORDER BY id",
        (anon_id, f'-{PENDING_VOTE_TTL_DAYS} days'),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def count_recent_pending_votes_by_ip(ip_address, hours=24):
    """Anonymous votes from one IP in the last `hours` (abuse guard)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM pending_votes WHERE ip_address = ? AND datetime(created_at) >= datetime('now', ?)",
        (ip_address, f'-{int(hours)} hours'),
    )
    n = cursor.fetchone()[0]
    conn.close()
    return n


def delete_pending_votes(anon_id):
    """Drop a visitor's pending votes (after they've been claimed)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM pending_votes WHERE anon_id = ?', (anon_id,))
    conn.commit()
    conn.close()
