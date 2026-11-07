from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import json
import os
import secrets
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
import database as db
import argumentation_store as argstore
import resource_library as library
import roles as roleslib

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app, supports_credentials=True)


@app.after_request
def _revalidate_frontend(response):
    """HTML/CSS/JS must always revalidate so edits show up immediately.

    While the frontend is in active development (files changing under a running
    server), browsers holding a stale stylesheet/JS cached from earlier visits
    rendered pages as broken mosaics — e.g. the filter bar stacked vertically and
    avatars appeared as squares. 'no-cache' keeps ETag 304s (cheap) but forces a
    freshness check on every load. Uploaded images keep proper caching because
    their URLs are version-stamped (?v=…).
    """
    ctype = response.headers.get('Content-Type', '')
    if any(t in ctype for t in ('text/html', 'text/css', 'javascript')):
        response.headers['Cache-Control'] = 'no-cache'
    return response

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Paths to data files
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESOURCES_FILE = os.path.join(DATA_DIR, 'resources.json')
CONTRIBUTIONS_FILE = os.path.join(DATA_DIR, 'contributions.json')
SUBMISSIONS_FILE = os.path.join(DATA_DIR, 'submissions.json')
GALLERY_FILE = os.path.join(DATA_DIR, 'gallery.json')
BONPOTE_GALLERY_FILE = os.path.join(DATA_DIR, 'bonpote_gallery.json')
ARTICLES_DIR = os.path.join(DATA_DIR, 'articles')
ARGUMENTATION_FILE = os.path.join(DATA_DIR, 'argumentation.json')
PERSONA_ARCHETYPES_FILE = os.path.join(DATA_DIR, 'persona_archetypes.json')
RESOURCE_REVIEWS_FILE = os.path.join(DATA_DIR, 'resource_reviews.json')

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTICLES_DIR, exist_ok=True)

# Uploaded profile pictures live here as avatar_<user_id>.<ext> (one per user,
# deterministic name so re-uploads replace the previous file).
AVATARS_DIR = os.path.join(DATA_DIR, 'avatars')
os.makedirs(AVATARS_DIR, exist_ok=True)

# Keep uploads modest: plenty for an avatar, small enough to not hurt anything.
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# Cap avatars at 3 MB via the request-length guard.
AVATAR_MAX_BYTES = 3 * 1024 * 1024

# Admin password (in production, use environment variables)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'change_me_in_production')

# Active admin session tokens. Each successful /api/admin/login mints a fresh,
# unguessable token (replacing the old hardcoded 'admin_authenticated' constant,
# which anyone reading the repo could forge). Tokens live only in memory, so a
# server restart forces admins to log in again — acceptable for this stage.
_active_admin_tokens = set()


def _is_valid_admin_token(req) -> bool:
    """Return True if the request carries a valid 'Bearer <token>' admin token."""
    auth = req.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return False
    token = auth[len('Bearer '):].strip()
    return bool(token) and token in _active_admin_tokens

#========================================
# User Class for Flask-Login
# ========================================

class User(UserMixin):
    def __init__(self, user_dict):
        self.id = user_dict['id']
        self.username = user_dict['username']
        self.email = user_dict['email']
        self.role = user_dict['role']
        self.reputation_score = user_dict['reputation_score']
        self._is_active = bool(user_dict['is_active'])
    
    @property
    def is_active(self):
        return self._is_active
    
    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    user_dict = db.get_user_by_id(int(user_id))
    if user_dict:
        return User(user_dict)
    return None

# ========================================
# Permission Decorators
# ========================================

def require_role(*roles):
    """Decorator to require specific roles."""
    def decorator(f):
        @wraps(f)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                return jsonify({'error': 'Insufficient permissions'}), 403
            return f(*args, **kwargs)
        return wrapped
    return decorator

def get_vote_weight(role):
    """Get vote weight based on role."""
    weights = {
        'trusted': 1,
        'moderator': 2,
        'admin': 3
    }
    return weights.get(role, 0)

def load_json_file(filepath, default=None):
    """Load JSON file or return default if it doesn't exist."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default if default is not None else []

def save_json_file(filepath, data):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def send_email_notification(submission):
    """Send email notification for new submission (configure with your email settings)."""
    # This is a placeholder - configure with your actual email settings
    email_enabled = os.environ.get('EMAIL_ENABLED', 'false').lower() == 'true'
    
    if not email_enabled:
        print(f"Email notification disabled. New submission: {submission['id']}")
        return
    
    try:
        sender_email = os.environ.get('SMTP_SENDER_EMAIL')
        receiver_email = os.environ.get('ADMIN_EMAIL')
        smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        smtp_password = os.environ.get('SMTP_PASSWORD')
        
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = 'New Counter-Narrative Submission - re.inform'
        
        body = f"""
        New submission received:
        
        ID: {submission['id']}
        Narrative: {submission['narrative']}
        Counter-Argument: {submission['counter_argument']}
        Source: {submission.get('source', 'N/A')}
        Date: {submission['submitted_at']}
        
        Review at: {os.environ.get('SITE_URL', 'http://localhost:5000')}/admin
        """
        
        message.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, smtp_password)
            server.send_message(message)
            
        print(f"Email notification sent for submission: {submission['id']}")
    except Exception as e:
        print(f"Failed to send email notification: {e}")

@app.route('/')
def serve_index():
    """Serve the main index.html."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/admin')
def serve_admin():
    """Serve the admin panel at the friendly /admin path (not just /admin.html)."""
    return send_from_directory(app.static_folder, 'admin.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory(app.static_folder, path)

# ========================================
# Authentication Endpoints
# ========================================

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validation
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Create user
        try:
            user_id = db.create_user(username, email, password)
            user_dict = db.get_user_by_id(user_id)
            user = User(user_dict)
            login_user(user, remember=True)
            
            return jsonify({
                'success': True,
                'message': 'Registration successful',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'reputation_score': user.reputation_score,
                    'avatar_url': _user_avatar_url(user_dict)
                }
            }), 201
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        import traceback
        print(f"Registration error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login a user."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        remember = data.get('remember', False)
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Get user
        user_dict = db.get_user_by_username(username)
        if not user_dict:
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Verify password
        if not db.verify_password(user_dict, password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Login user
        user = User(user_dict)
        login_user(user, remember=remember)
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'reputation_score': user.reputation_score,
                'avatar_url': _user_avatar_url(user_dict)
            }
        })
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    """Logout current user."""
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/current-user', methods=['GET'])
def get_current_user():
    """Get current logged-in user."""
    if current_user.is_authenticated:
        avatar_url = _user_avatar_url(db.get_user_by_id(current_user.id))
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'role': current_user.role,
                'reputation_score': current_user.reputation_score,
                'avatar_url': avatar_url
            }
        })
    return jsonify({'authenticated': False})


def _user_avatar_url(user_dict):
    """Public URL for a user's profile picture, or None.

    The ?v= timestamp busts browser cache after re-upload.
    """
    if not isinstance(user_dict, dict) or not user_dict.get('avatar_file'):
        return None
    updated = (user_dict.get('avatar_updated_at') or '').replace(':', '').replace('-', '').replace('.', '')
    return f"/api/user/{user_dict['id']}/avatar?v={updated}"


# File-format sniffing by magic bytes, deliberately stricter than trusting the
# client's Content-Type or extension: only raster formats browsers can display
# as an <img> get accepted, so nothing script-like can be stashed then served.
def _sniff_image_ext(stream):
    """Return 'jpg'|'png'|'gif'|'webp' if the stream's header matches, else None."""
    head = stream.read(16)
    stream.seek(0)
    if head[:3] == b'\xff\xd8\xff':
        return 'jpg'
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if head[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'webp'
    return None


@app.route('/api/user/<int:user_id>/avatar', methods=['GET'])
def serve_user_avatar(user_id):
    """Serve a user's profile picture (public — avatars are public info)."""
    user_dict = db.get_user_by_id(user_id)
    file_name = (user_dict or {}).get('avatar_file')
    if not user_dict or not file_name:
        return jsonify({'error': 'Avatar not found'}), 404
    # The filename is generated by us (never user-controlled), so directory
    # traversal isn't possible — but stay explicit anyway.
    if os.path.basename(file_name) != file_name:
        return jsonify({'error': 'Avatar not found'}), 404
    return send_from_directory(AVATARS_DIR, file_name, max_age=3600)


@app.route('/api/me/avatar', methods=['POST'])
@login_required
def upload_avatar():
    """Upload or replace the signed-in user's profile picture."""
    file = request.files.get('avatar')
    if not file or not file.filename:
        return jsonify({'error': 'No file provided'}), 400

    ext = _sniff_image_ext(file.stream)
    if not ext:
        return jsonify({'error': 'Unsupported image format (use JPG, PNG, GIF or WebP)'}), 400

    file.seek(0, os.SEEK_END)
    if file.tell() > AVATAR_MAX_BYTES:
        return jsonify({'error': 'Image is too large (max 3 MB)'}), 413
    file.seek(0)

    # Deterministic name: a re-upload replaces the previous picture.
    file_name = f"avatar_{current_user.id}.{ext}"
    old_file = (db.get_user_by_id(current_user.id) or {}).get('avatar_file')
    file.save(os.path.join(AVATARS_DIR, file_name))
    if old_file and old_file != file_name and os.path.basename(old_file) == old_file:
        try:
            os.remove(os.path.join(AVATARS_DIR, old_file))
        except OSError:
            pass

    db.set_user_avatar(current_user.id, file_name)
    user_dict = db.get_user_by_id(current_user.id)
    return jsonify({
        'success': True,
        'avatar_url': _user_avatar_url(user_dict)
    })


@app.route('/api/me/avatar', methods=['DELETE'])
@login_required
def delete_avatar():
    """Remove the signed-in user's profile picture."""
    user_dict = db.get_user_by_id(current_user.id)
    file_name = (user_dict or {}).get('avatar_file')

    db.clear_user_avatar(current_user.id)
    if file_name and os.path.basename(file_name) == file_name:
        try:
            os.remove(os.path.join(AVATARS_DIR, file_name))
        except OSError:
            pass

    return jsonify({'success': True})

@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get user profile."""
    user_dict = db.get_user_by_id(user_id)
    if not user_dict:
        return jsonify({'error': 'User not found'}), 404
    
    endorsements = db.get_user_endorsements(user_id)
    
    # Get user's contributions (from submissions with user_id)
    submissions = load_json_file(SUBMISSIONS_FILE, [])
    user_submissions = [s for s in submissions if s.get('user_id') == user_id]
    
    contributions = load_json_file(CONTRIBUTIONS_FILE, [])
    user_contributions = [c for c in contributions if c.get('user_id') == user_id]
    
    # Get following stats
    following_count = db.get_following_count(user_id)
    followers_count = db.get_followers_count(user_id)
    
    # Check if current user is following this user
    is_following = False
    is_own_profile = False
    if current_user.is_authenticated:
        is_own_profile = current_user.id == user_id
        if not is_own_profile:
            is_following = db.is_following_user(current_user.id, user_id)
    
    return jsonify({
        'user': {
            'id': user_dict['id'],
            'username': user_dict['username'],
            'role': user_dict['role'],
            'reputation_score': user_dict['reputation_score'],
            'created_at': user_dict['created_at'],
            'avatar_url': _user_avatar_url(user_dict)
        },
        'endorsements': endorsements,
        # Public roles only: Field Agent stays private (see roles.ROLE_CATALOG).
        'roles': roleslib.describe_roles(
            argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE), user_id, public_only=True),
        'contribution_count': len(user_contributions),
        'submission_count': len(user_submissions),
        'following_count': following_count,
        'followers_count': followers_count,
        'is_following': is_following,
        'is_own_profile': is_own_profile,
        'contributions': user_contributions
    })

@app.route('/api/me/contributions', methods=['GET'])
@login_required
def my_contributions():
    """The signed-in user's own submissions with their review status.

    Powers the profile 'status tracker': counter-narrative submissions (with
    pending/approved/rejected status) and resources they submitted to the library
    (with the library's review status). Own-profile only — never exposed for
    another user, since pending/rejected work is private.
    """
    uid = current_user.id
    items = []

    # Counter-narrative submissions.
    for s in load_json_file(SUBMISSIONS_FILE, []):
        if not isinstance(s, dict) or s.get('user_id') != uid:
            continue
        narrative = (s.get('narrative') or s.get('counter_argument') or '').strip()
        items.append({
            'kind': 'response',
            'title': narrative[:140] or 'Untitled response',
            'status': (s.get('status') or 'pending').lower(),
            'submitted_at': s.get('submitted_at'),
            'source': s.get('source'),
        })

    # Resources the user submitted to the library, with their review status.
    for entry in _build_resource_library():
        submitter = entry.get('submitted_by') or {}
        origins = entry.get('origins') or []
        if submitter.get('id') != uid or 'submitted' not in origins:
            continue
        items.append({
            'kind': 'source',
            'title': entry.get('title') or entry.get('url'),
            'status': (entry.get('review') or {}).get('status', 'pending'),
            'submitted_at': entry.get('submitted_at'),
            'url': entry.get('url'),
        })

    items.sort(key=lambda x: str(x.get('submitted_at') or ''), reverse=True)

    counts = {'pending': 0, 'approved': 0, 'rejected': 0}
    for it in items:
        counts[it['status']] = counts.get(it['status'], 0) + 1

    return jsonify({'items': items, 'counts': counts})


@app.route('/api/me/following-activity', methods=['GET'])
@login_required
def following_activity():
    """Recent actions by the people the signed-in user follows.

    There's no event log, so this synthesizes a feed from timestamped items the
    followed users authored: approved contributions, counter-arguments in the
    argumentation store, and resources they submitted. Content-first (names the
    thing, not points); bare ratings are intentionally left out as too noisy.
    """
    followed = {u['id']: u for u in db.get_following(current_user.id)}
    if not followed:
        return jsonify({'items': []})

    def _snippet(text, n=120):
        text = (text or '').strip()
        return (text[:n].rstrip() + '…') if len(text) > n else text

    events = []

    # Approved contributions (community feed).
    for c in load_json_file(CONTRIBUTIONS_FILE, []):
        uid = c.get('user_id')
        if uid in followed and c.get('date'):
            events.append({
                'user': {'id': uid, 'username': followed[uid]['username']},
                'kind': 'response',
                'verb': 'answered a claim',
                'title': _snippet(c.get('title') or c.get('narrative')),
                'at': c.get('date'),
                'href': None,
            })

    # Counter-arguments and submitted resources from the argumentation store.
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    for ca in (store.get('counter_arguments') or []):
        uid = ca.get('created_by_user_id')
        if uid in followed and ca.get('created_at'):
            events.append({
                'user': {'id': uid, 'username': followed[uid]['username']},
                'kind': 'response',
                'verb': 'wrote a response',
                'title': _snippet(ca.get('text')),
                'at': ca.get('created_at'),
                'href': f"argument.html?id={ca.get('argument_id')}" if ca.get('argument_id') is not None else None,
            })
    for r in (store.get('resources') or []):
        uid = r.get('created_by_user_id')
        if uid in followed and r.get('submitted_directly') and r.get('created_at'):
            events.append({
                'user': {'id': uid, 'username': followed[uid]['username']},
                'kind': 'source',
                'verb': 'shared a source',
                'title': _snippet(r.get('title') or r.get('url')),
                'at': r.get('created_at'),
                'href': r.get('url'),
            })

    events.sort(key=lambda e: str(e.get('at') or ''), reverse=True)
    return jsonify({'items': events[:20]})


# ========================================
# Tags API
# ========================================

@app.route('/api/tags', methods=['GET'])
def get_all_tags():
    """Get all available predefined tags."""
    return jsonify({
        'tags': db.get_all_tags()
    })

# ========================================
# Tag Following API
# ========================================

@app.route('/api/user/<int:user_id>/followed-tags', methods=['GET'])
def get_user_followed_tags(user_id):
    """Get tags a user is following."""
    tags = db.get_followed_tags(user_id)
    return jsonify({
        'user_id': user_id,
        'followed_tags': tags
    })

@app.route('/api/user/<int:user_id>/followed-tags', methods=['POST'])
@login_required
def follow_tag(user_id):
    """Follow a tag."""
    if current_user.id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    tag = data.get('tag', '').strip().lower()
    
    if not tag:
        return jsonify({'error': 'Tag is required'}), 400
    
    success = db.follow_tag(user_id, tag)
    if success:
        return jsonify({'success': True, 'message': f'Now following #{tag}'})
    else:
        return jsonify({'error': 'Already following this tag'}), 400

@app.route('/api/user/<int:user_id>/followed-tags/<tag>', methods=['DELETE'])
@login_required
def unfollow_tag(user_id, tag):
    """Unfollow a tag."""
    if current_user.id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    success = db.unfollow_tag(user_id, tag)
    if success:
        return jsonify({'success': True, 'message': f'Unfollowed #{tag}'})
    else:
        return jsonify({'error': 'Not following this tag'}), 400

# ========================================
# User Following API
# ========================================

@app.route('/api/user/<int:user_id>/following', methods=['GET'])
def get_user_following(user_id):
    """Get users that this user is following."""
    following = db.get_following(user_id)
    return jsonify({
        'user_id': user_id,
        'following': following,
        'count': len(following)
    })

@app.route('/api/user/<int:user_id>/followers', methods=['GET'])
def get_user_followers(user_id):
    """Get users following this user."""
    followers = db.get_followers(user_id)
    return jsonify({
        'user_id': user_id,
        'followers': followers,
        'count': len(followers)
    })

@app.route('/api/user/<int:user_id>/follow', methods=['POST'])
@login_required
def follow_user_endpoint(user_id):
    """Follow a user."""
    if current_user.id == user_id:
        return jsonify({'error': 'Cannot follow yourself'}), 400
    
    # Check if target user exists
    target_user = db.get_user_by_id(user_id)
    if not target_user:
        return jsonify({'error': 'User not found'}), 404
    
    success = db.follow_user(current_user.id, user_id)
    if success:
        return jsonify({'success': True, 'message': f'Now following {target_user["username"]}'})
    else:
        return jsonify({'error': 'Already following this user'}), 400

@app.route('/api/user/<int:user_id>/follow', methods=['DELETE'])
@login_required
def unfollow_user_endpoint(user_id):
    """Unfollow a user."""
    success = db.unfollow_user(current_user.id, user_id)
    if success:
        return jsonify({'success': True, 'message': 'Unfollowed user'})
    else:
        return jsonify({'error': 'Not following this user'}), 400

@app.route('/api/user/<int:user_id>/is-following', methods=['GET'])
def check_is_following(user_id):
    """Check if current user is following target user."""
    if not current_user.is_authenticated:
        return jsonify({'is_following': False})
    
    is_following = db.is_following_user(current_user.id, user_id)
    return jsonify({'is_following': is_following})

# ========================================
# API Routes
# ========================================

@app.route('/api/contributions', methods=['GET'])
def get_contributions():
    """Get all approved contributions from the community."""
    contributions = load_json_file(CONTRIBUTIONS_FILE, [])
    
    # Optional filtering and sorting
    sort_by = request.args.get('sort', 'date')  # date, title
    tag_filter = request.args.get('tag', None)
    
    # Filter by tag if specified
    if tag_filter:
        contributions = [c for c in contributions if tag_filter in c.get('tags', [])]
    
    # Sort
    if sort_by == 'title':
        contributions.sort(key=lambda x: x.get('title', ''))
    else:  # date
        contributions.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    return jsonify(contributions)

@app.route('/api/resources', methods=['GET'])
def get_resources():
    """Get all curated resources (aggregated best contributions)."""
    resources = load_json_file(RESOURCES_FILE, [])

    # Enrich author information using the user database.
    # resources.json may store:
    # - author: "Anonymous" (string)
    # - author: <user_id> (number)
    # Also supports legacy: user_id
    enriched_resources = []
    for r in (resources or []):
        if not isinstance(r, dict):
            continue

        author_user_id = None
        author_field = r.get('author')
        if isinstance(author_field, int):
            author_user_id = author_field
        elif isinstance(author_field, str) and author_field.strip().isdigit():
            author_user_id = int(author_field.strip())
        elif isinstance(r.get('user_id'), int):
            author_user_id = int(r.get('user_id'))

        if author_user_id is not None:
            user_dict = db.get_user_by_id(author_user_id)
            if user_dict:
                r['user_id'] = user_dict['id']
                r['username'] = user_dict['username']
                r['author'] = user_dict['username']
            else:
                # Unknown user id; fall back to Anonymous display.
                r['author'] = 'Anonymous'
        else:
            if not isinstance(r.get('author'), str) or not r.get('author').strip():
                r['author'] = 'Anonymous'

        enriched_resources.append(r)

    resources = enriched_resources

    # Community-submitted resources live in the argumentation store. Only those
    # submitted deliberately to the library belong here — the store also holds
    # URLs auto-created as a side effect of writing a debunk, which are bare
    # links with no title and would drown the curated list.
    curated_urls = {
        argstore.normalize_url(str(r.get('source') or '')).lower()
        for r in resources
        if isinstance(r, dict) and r.get('source')
    }
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    for r in (store.get('resources', []) or []):
        if not isinstance(r, dict) or not r.get('url'):
            continue
        if not r.get('submitted_directly'):
            continue
        if argstore.normalize_url(str(r['url'])).lower() in curated_urls:
            continue

        author_id = r.get('created_by_user_id')
        user_dict = db.get_user_by_id(author_id) if isinstance(author_id, int) else None
        resources.append({
            'id': f"community-{r.get('id')}",
            'title': r.get('title') or r['url'],
            'summary': r.get('description') or '',
            'source': r['url'],
            'tags': r.get('tags', []) or [],
            'date': r.get('created_at', ''),
            'effectiveness_rating': 0,
            'community_submitted': True,
            'user_id': user_dict['id'] if user_dict else None,
            'username': user_dict['username'] if user_dict else None,
            'author': user_dict['username'] if user_dict else 'Anonymous',
        })

    # Optional filtering and sorting
    sort_by = request.args.get('sort', 'rating')  # date, rating, title
    tag_filter = request.args.get('tag', None)
    
    # Filter by tag if specified
    if tag_filter:
        resources = [r for r in resources if tag_filter in r.get('tags', [])]
    
    # Sort
    if sort_by == 'rating':
        resources.sort(key=lambda x: x.get('effectiveness_rating', 0), reverse=True)
    elif sort_by == 'title':
        resources.sort(key=lambda x: x.get('title', ''))
    else:  # date
        resources.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    return jsonify(resources)


@app.route('/api/articles/<path:filename>', methods=['GET'])
def get_article_markdown(filename: str):
    """Serve Markdown articles stored in backend/data/articles.

    This keeps the UX simple: resources can link to internal articles via
    `/api/articles/<filename>.md`.
    """
    filename = (filename or '').strip()

    # Minimal validation: only serve markdown files from the articles dir.
    if not filename or not filename.lower().endswith('.md'):
        return jsonify({'error': 'Not found'}), 404

    return send_from_directory(ARTICLES_DIR, filename, mimetype='text/markdown; charset=utf-8')


@app.route('/api/submit-contribution', methods=['POST'])
def submit_contribution():
    """Submit a new counter-narrative contribution."""
    try:
        data = request.get_json()
        
        # Check rate limiting
        user_id = current_user.id if current_user.is_authenticated else None
        ip_address = request.remote_addr
        limit = 20 if current_user.is_authenticated else 5
        
        if not db.check_rate_limit(user_id, ip_address, 'submission', limit):
            return jsonify({'error': f'Rate limit exceeded. Maximum {limit} submissions per day.'}), 429
        
        # Validate required fields
        if not data.get('source'):
            return jsonify({'error': 'Source is required'}), 400
        
        # Load existing submissions
        submissions = load_json_file(SUBMISSIONS_FILE, [])
        
        # Create new submission
        submission = {
            'id': len(submissions) + 1,
            'narrative': data.get('narrative', '').strip(),
            'counter_argument': data.get('counter_argument', '').strip(),
            'source': data['source'].strip(),
            'tags': data.get('tags', []),
            'submitted_at': datetime.now().isoformat(),
            'status': 'pending',
            'user_id': user_id,
            'username': current_user.username if current_user.is_authenticated else None
        }
        
        # Add to submissions
        submissions.append(submission)
        save_json_file(SUBMISSIONS_FILE, submissions)
        
        # Send email notification
        send_email_notification(submission)
        
        return jsonify({
            'success': True,
            'message': 'Contribution submitted successfully! It will soon be reviewed by our team.',
            'submission_id': submission['id']
        }), 201
        
    except Exception as e:
        print(f"Error submitting contribution: {e}")
        return jsonify({'error': 'Failed to submit contribution'}), 500

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Simple admin login."""
    data = request.get_json()
    password = data.get('password', '')
    
    # Constant-time comparison to avoid leaking the password via timing.
    if password and secrets.compare_digest(password, ADMIN_PASSWORD):
        token = secrets.token_urlsafe(32)
        _active_admin_tokens.add(token)
        return jsonify({'success': True, 'token': token})
    else:
        return jsonify({'error': 'Invalid password'}), 401


@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Invalidate the current admin token."""
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        _active_admin_tokens.discard(auth[len('Bearer '):].strip())
    return jsonify({'success': True})

@app.route('/api/admin/submissions', methods=['GET'])
def get_submissions():
    """Get all pending submissions (requires authentication)."""
    if not _is_valid_admin_token(request):
        return jsonify({'error': 'Unauthorized'}), 401

    submissions = load_json_file(SUBMISSIONS_FILE, [])
    return jsonify(submissions)

@app.route('/api/admin/approve-submission/<int:submission_id>', methods=['POST'])
def approve_submission(submission_id):
    """Approve a submission and add it to counter-arguments."""
    if not _is_valid_admin_token(request):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        
        # Load submissions
        submissions = load_json_file(SUBMISSIONS_FILE, [])
        submission = next((s for s in submissions if s['id'] == submission_id), None)
        
        if not submission:
            return jsonify({'error': 'Submission not found'}), 404
        
        # Load contributions
        contributions = load_json_file(CONTRIBUTIONS_FILE, [])
        
        # Create contribution entry from submission
        contribution = {
            'id': len(contributions) + 1,
            'title': data.get('title', f"Contribution #{len(contributions) + 1}"),
            'narrative': submission['narrative'],
            'counter_argument': submission['counter_argument'],
            'tags': submission.get('tags', []),
            'source': submission.get('source', ''),
            'date': datetime.now().isoformat(),
            'submission_id': submission_id
        }
        
        # Add to contributions
        contributions.append(contribution)
        save_json_file(CONTRIBUTIONS_FILE, contributions)
        
        # Update submission status
        submission['status'] = 'approved'
        save_json_file(SUBMISSIONS_FILE, submissions)
        
        return jsonify({'success': True, 'contribution': contribution})
        
    except Exception as e:
        print(f"Error approving submission: {e}")
        return jsonify({'error': 'Failed to approve submission'}), 500

@app.route('/api/admin/reject-submission/<int:submission_id>', methods=['POST'])
def reject_submission(submission_id):
    """Reject a submission."""
    if not _is_valid_admin_token(request):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Load submissions
        submissions = load_json_file(SUBMISSIONS_FILE, [])
        submission = next((s for s in submissions if s['id'] == submission_id), None)
        
        if not submission:
            return jsonify({'error': 'Submission not found'}), 404
        
        # Update submission status
        submission['status'] = 'rejected'
        save_json_file(SUBMISSIONS_FILE, submissions)
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error rejecting submission: {e}")
        return jsonify({'error': 'Failed to reject submission'}), 500

# ========================================
# Voting Endpoints
# ========================================

@app.route('/api/vote/<int:submission_id>', methods=['POST'])
@require_role('trusted', 'moderator', 'admin')
def vote_submission(submission_id):
    """Vote on a submission (trusted+ only)."""
    try:
        data = request.get_json()
        vote_type = data.get('vote_type')  # approve, reject, needs_work
        
        if vote_type not in ['approve', 'reject', 'needs_work']:
            return jsonify({'error': 'Invalid vote type'}), 400
        
        # Check rate limit
        if not db.check_rate_limit(current_user.id, request.remote_addr, 'vote', 50):
            return jsonify({'error': 'Rate limit exceeded. Maximum 50 votes per day.'}), 429
        
        # Get vote weight based on role
        weight = get_vote_weight(current_user.role)
        
        # Create vote
        db.create_vote(current_user.id, submission_id, vote_type, weight)
        
        # Get vote summary
        vote_summary = db.get_vote_summary(submission_id)
        
        # Auto-approve if score >= 3
        if vote_summary['approve_score'] >= 3:
            # Load submissions and approve
            submissions = load_json_file(SUBMISSIONS_FILE, [])
            submission = next((s for s in submissions if s['id'] == submission_id), None)
            
            if submission and submission['status'] == 'pending':
                # Create contribution
                contributions = load_json_file(CONTRIBUTIONS_FILE, [])
                contribution = {
                    'id': len(contributions) + 1,
                    'title': f"Contribution #{len(contributions) + 1}",
                    'narrative': submission.get('narrative', ''),
                    'counter_argument': submission.get('counter_argument', ''),
                    'tags': submission.get('tags', []),
                    'source': submission.get('source', ''),
                    'date': datetime.now().isoformat(),
                    'submission_id': submission_id,
                    'user_id': submission.get('user_id'),
                    'username': submission.get('username')
                }
                contributions.append(contribution)
                save_json_file(CONTRIBUTIONS_FILE, contributions)
                
                # Update submission status
                submission['status'] = 'approved'
                save_json_file(SUBMISSIONS_FILE, submissions)
                
                # Award reputation to contributor
                if submission.get('user_id'):
                    db.update_user_reputation(submission['user_id'], 10)
        
        return jsonify({
            'success': True,
            'vote_summary': vote_summary
        })
        
    except Exception as e:
        print(f"Error voting: {e}")
        return jsonify({'error': 'Failed to record vote'}), 500

@app.route('/api/submission/<int:submission_id>/votes', methods=['GET'])
@require_role('trusted', 'moderator', 'admin')
def get_submission_votes_endpoint(submission_id):
    """Get vote summary for a submission."""
    vote_summary = db.get_vote_summary(submission_id)
    return jsonify(vote_summary)

# ========================================
# Endorsement Endpoints
# ========================================

@app.route('/api/endorse/<int:user_id>', methods=['POST'])
@require_role('trusted', 'moderator', 'admin')
def endorse_user(user_id):
    """Endorse a user (trusted+ only)."""
    try:
        data = request.get_json()
        reason = data.get('reason', '').strip()
        
        if not reason:
            return jsonify({'error': 'Reason is required'}), 400
        
        # Check rate limit
        if not db.check_rate_limit(current_user.id, request.remote_addr, 'endorsement', 10):
            return jsonify({'error': 'Rate limit exceeded. Maximum 10 endorsements per day.'}), 429
        
        try:
            db.create_endorsement(current_user.id, user_id, reason)
            return jsonify({'success': True, 'message': 'User endorsed successfully'})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        print(f"Error endorsing user: {e}")
        return jsonify({'error': 'Failed to endorse user'}), 500

# ========================================
# Moderator Nomination Endpoints
# ========================================

@app.route('/api/nominate-moderator', methods=['POST'])
@require_role('trusted', 'moderator', 'admin')
def nominate_moderator_endpoint():
    """Nominate a user for moderator role."""
    try:
        data = request.get_json()
        nominee_username = data.get('nominee_username', '').strip()
        reason = data.get('reason', '').strip()
        
        if not nominee_username or not reason:
            return jsonify({'error': 'Nominee username and reason are required'}), 400
        
        nominee = db.get_user_by_username(nominee_username)
        if not nominee:
            return jsonify({'error': 'User not found'}), 404
        
        if db.nominate_moderator(current_user.id, nominee['id'], reason):
            return jsonify({'success': True, 'message': 'Nomination submitted for admin review'})
        else:
            return jsonify({'error': 'Failed to submit nomination'}), 500
            
    except Exception as e:
        print(f"Error nominating moderator: {e}")
        return jsonify({'error': 'Failed to nominate moderator'}), 500

@app.route('/api/admin/nominations', methods=['GET'])
@require_role('admin')
def get_nominations():
    """Get pending moderator nominations (admin only)."""
    nominations = db.get_pending_nominations()
    return jsonify(nominations)

@app.route('/api/admin/nominations/<int:nomination_id>/review', methods=['POST'])
@require_role('admin')
def review_nomination_endpoint(nomination_id):
    """Review a moderator nomination (admin only)."""
    try:
        data = request.get_json()
        approved = data.get('approved', False)
        
        db.review_nomination(nomination_id, current_user.id, approved)
        
        return jsonify({
            'success': True,
            'message': 'Nomination approved' if approved else 'Nomination rejected'
        })
        
    except Exception as e:
        print(f"Error reviewing nomination: {e}")
        return jsonify({'error': 'Failed to review nomination'}), 500


# ========================================
# Argumentation (v1) Endpoints
# ========================================


@app.route('/api/arguments', methods=['GET'])
def list_arguments_v1():
    """List debunkable arguments.

    This is the core entity for the community workflow.
    """
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    _args_by_id, personas_by_id = argstore.build_indexes(store)

    q = (request.args.get('q') or '').strip().lower()

    # Average rating per counter-argument (for the "top response" preview).
    scores_by_counter = {}
    for r in store.get('ratings', []) or []:
        if not isinstance(r, dict):
            continue
        cid = r.get('counter_argument_id')
        if isinstance(cid, int) and isinstance(r.get('score'), int):
            scores_by_counter.setdefault(cid, []).append(int(r['score']))

    def _avg(cid):
        s = scores_by_counter.get(cid)
        return (sum(s) / len(s)) if s else None

    def _persona_brief(pid):
        p = personas_by_id.get(pid) if isinstance(pid, int) else None
        if not p:
            return None
        return {
            'id': p.get('id'),
            'name': p.get('name'),
            'emoji': p.get('icon_emoji'),
        }

    # Group counter-arguments by argument.
    counters_by_arg = {}
    for ca in store.get('counter_arguments', []) or []:
        if not isinstance(ca, dict):
            continue
        aid = ca.get('argument_id')
        if isinstance(aid, int):
            counters_by_arg.setdefault(aid, []).append(ca)

    # Importance ratings are cast on the claim itself: how urgent is this to debunk?
    importance_by_arg = argstore.importance_by_argument(store)
    viewer_id = current_user.id if current_user.is_authenticated else None

    arguments = []
    for a in store.get('arguments', []) or []:
        if not isinstance(a, dict) or not isinstance(a.get('id'), int):
            continue
        text = a.get('text') or ''
        if q and q not in str(text).lower():
            continue

        cas = counters_by_arg.get(a['id'], [])

        # Distinct personas covered, in stable order.
        personas = []
        seen = set()
        for ca in cas:
            pid = ca.get('persona_archetype_id')
            if isinstance(pid, int) and pid not in seen:
                brief = _persona_brief(pid)
                if brief:
                    personas.append(brief)
                    seen.add(pid)

        # Top response = highest average rating, then most-rated, then first.
        top_response = None
        if cas:
            def _rank(ca):
                cid = ca.get('id')
                avg = _avg(cid)
                return (avg if isinstance(avg, (int, float)) else -1, len(scores_by_counter.get(cid, [])))
            best = max(cas, key=_rank)
            snippet = str(best.get('text') or '').strip()
            if len(snippet) > 160:
                snippet = snippet[:160].rstrip() + '…'
            top_response = {
                'text': snippet,
                'avg_score': _avg(best.get('id')),
                'rating_count': len(scores_by_counter.get(best.get('id'), [])),
                'persona': _persona_brief(best.get('persona_archetype_id')),
            }

        rated = any(scores_by_counter.get(ca.get('id')) for ca in cas)

        # Viewer relationship to this claim — used to exclude "own content" from
        # recommendations (never suggest a claim you wrote or already answered).
        viewer_is_author = viewer_id is not None and a.get('created_by_user_id') == viewer_id
        viewer_has_responded = viewer_id is not None and any(
            ca.get('created_by_user_id') == viewer_id for ca in cas
        )

        # Total contributions on this claim, across every interaction type:
        #  - each debunk submitted
        #  - each rating cast on a debunk
        #  - each resource attached to a debunk
        #  - each importance rating cast on the claim itself
        debunks_n = len(cas)
        ratings_n = sum(len(scores_by_counter.get(ca.get('id'), [])) for ca in cas)
        resources_n = sum(len(ca.get('resource_ids') or []) for ca in cas)
        importance = importance_by_arg.get(a['id'], [])
        importance_n = len(importance)
        contribution_count = debunks_n + ratings_n + resources_n + importance_n

        importance_scores = [int(ir['score']) for ir in importance]
        my_importance = next(
            (int(ir['score']) for ir in importance if viewer_id is not None and ir.get('user_id') == viewer_id),
            None,
        )

        arguments.append({
            'id': a['id'],
            'text': text,
            'tags': a.get('tags', []) or [],
            'counter_argument_count': debunks_n,
            'contribution_count': contribution_count,
            'contribution_breakdown': {
                'debunks': debunks_n,
                'ratings': ratings_n,
                'resources': resources_n,
                'importance': importance_n,
            },
            'personas': personas,
            'top_response': top_response,
            'rated': rated,
            'viewer_is_author': viewer_is_author,
            'viewer_has_responded': viewer_has_responded,
            'importance': {
                'avg_score': (sum(importance_scores) / len(importance_scores)) if importance_scores else None,
                'count': importance_n,
                'my_score': my_importance,
            },
        })

    arguments.sort(key=lambda x: (x.get('contribution_count', 0), str(x.get('text') or '').lower()), reverse=True)
    return jsonify(arguments)


@app.route('/api/arguments/<int:argument_id>', methods=['GET'])
def get_argument_v1(argument_id: int):
    """Get one argument with its counter-arguments and aggregated ratings."""
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, personas_by_id = argstore.build_indexes(store)
    res_by_id = argstore.resources_by_id(store)

    arg = args_by_id.get(argument_id)
    if not arg:
        return jsonify({'error': 'Argument not found'}), 404

    def _resources_for(ca):
        """Resolve a debunk's resource_ids; fall back to legacy source_url."""
        out = []
        for rid in ca.get('resource_ids') or []:
            r = res_by_id.get(rid)
            if r:
                out.append({'id': r['id'], 'url': r.get('url'), 'title': r.get('title')})
        if not out and ca.get('source_url'):
            out.append({'id': None, 'url': ca.get('source_url'), 'title': None})
        return out

    # Pre-group ratings by counter_argument_id
    rating_by_counter = {}
    for r in store.get('ratings', []) or []:
        if not isinstance(r, dict):
            continue
        cid = r.get('counter_argument_id')
        if not isinstance(cid, int) or not isinstance(r.get('score'), int):
            continue
        rating_by_counter.setdefault(cid, []).append(int(r['score']))

    # Source-verified stamp, plus whether the viewer can add a check.
    checks_by_counter = argstore.source_checks_by_counter(store)
    check_weights = roleslib.check_weights()
    sourcer_ids = set(db.get_role_tiers('sourcer'))
    viewer_id = current_user.id if current_user.is_authenticated else None
    viewer_role = current_user.role if current_user.is_authenticated else None

    def _source_check_for(ca):
        if not roleslib.has_sources(ca):
            return {'status': 'unsourced', 'count': 0, 'can_check': False, 'my_verdict': None}
        checks = checks_by_counter.get(ca.get('id'), [])
        can_check, _reason = roleslib.can_check_source(viewer_id, viewer_role, ca, sourcer_ids)
        summary = roleslib.source_status(checks, check_weights)
        return {
            'status': summary['status'],
            'count': summary['count'],
            'can_check': can_check,
            'my_verdict': next((sc['verdict'] for sc in checks if sc.get('user_id') == viewer_id), None),
        }

    counter_arguments = []
    for ca in store.get('counter_arguments', []) or []:
        if not isinstance(ca, dict) or ca.get('argument_id') != argument_id:
            continue

        persona = None
        pid = ca.get('persona_archetype_id')
        if isinstance(pid, int):
            persona = personas_by_id.get(pid)

        scores = rating_by_counter.get(ca.get('id'), [])
        avg_score = (sum(scores) / len(scores)) if scores else None

        counter_arguments.append({
            'id': ca.get('id'),
            'text': ca.get('text'),
            'source_url': ca.get('source_url'),
            'resources': _resources_for(ca),
            'persona_archetype': persona,
            'rating_count': len(scores),
            'avg_score': avg_score,
            'source_check': _source_check_for(ca),
        })

    # Highest-rated first, then most-rated, then stable
    def _sort_key(x):
        avg = x.get('avg_score')
        avg_sort = avg if isinstance(avg, (int, float)) else -1
        return (avg_sort, x.get('rating_count', 0))

    counter_arguments.sort(key=_sort_key, reverse=True)

    return jsonify({
        'argument': {
            'id': arg.get('id'),
            'text': arg.get('text'),
            'tags': arg.get('tags', []) or [],
        },
        'counter_arguments': counter_arguments,
    })


@app.route('/api/arguments', methods=['POST'])
@login_required
def create_argument_v1():
    data = request.get_json() or {}
    text = (data.get('text') or '').strip()
    tags = data.get('tags')
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        return jsonify({'error': 'tags must be a list'}), 400

    try:
        store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
        arg = argstore.upsert_argument(store, text=text, created_by_user_id=current_user.id, tags=tags)
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
        return jsonify({'success': True, 'argument': arg}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error creating argument: {e}")
        return jsonify({'error': 'Failed to create argument'}), 500


@app.route('/api/persona-archetypes', methods=['GET'])
def list_persona_archetypes_v1():
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    personas = [p for p in (store.get('persona_archetypes', []) or []) if isinstance(p, dict) and isinstance(p.get('id'), int)]
    personas.sort(key=lambda p: (str(p.get('name') or '').lower(), p.get('id')))
    return jsonify(personas)


@app.route('/api/counter-arguments', methods=['POST'])
@login_required
def create_counter_argument_v1():
    data = request.get_json() or {}
    argument_id = data.get('argument_id')
    text = (data.get('text') or '').strip()
    persona_id = data.get('persona_archetype_id')

    if not isinstance(argument_id, int):
        return jsonify({'error': 'argument_id must be an integer'}), 400
    # A debunk targets exactly one persona.
    if not isinstance(persona_id, int):
        return jsonify({'error': 'A target persona is required'}), 400

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, personas_by_id = argstore.build_indexes(store)
    if argument_id not in args_by_id:
        return jsonify({'error': 'Argument not found'}), 404
    if persona_id not in personas_by_id:
        return jsonify({'error': 'Persona archetype not found'}), 404
    # Numbered EVS clusters are analysis placeholders, not targets for a response.
    if personas_by_id[persona_id].get('layer') == 'evs_cluster':
        return jsonify({'error': 'Choose a named persona'}), 400

    # Collect source URLs from `sources` (list) and/or legacy `source_url`.
    raw_sources = []
    sources = data.get('sources')
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, str) and s.strip():
                raw_sources.append(s.strip())
            elif isinstance(s, dict) and str(s.get('url') or '').strip():
                raw_sources.append(str(s['url']).strip())
    legacy = (data.get('source_url') or '').strip()
    if legacy:
        raw_sources.append(legacy)

    # De-duplicate (case-insensitive), then resolve to first-class Resources.
    seen = set()
    unique_urls = []
    for u in raw_sources:
        k = u.lower()
        if k not in seen:
            seen.add(k)
            unique_urls.append(u)

    resource_ids = []
    seen_rids = set()
    for u in unique_urls:
        try:
            res = argstore.upsert_resource(store, u, created_by_user_id=current_user.id)
        except ValueError:
            continue
        # Two raw URLs can normalize to the same Resource — keep ids unique.
        rid = res['id']
        if rid not in seen_rids:
            seen_rids.add(rid)
            resource_ids.append(rid)

    try:
        counter = argstore.create_counter_argument(
            store,
            argument_id=argument_id,
            text=text,
            persona_archetype_id=persona_id,
            source_url=(unique_urls[0] if unique_urls else None),
            resource_ids=resource_ids,
            created_by_user_id=current_user.id,
        )
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
        return jsonify({'success': True, 'counter_argument': counter}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error creating counter-argument: {e}")
        return jsonify({'error': 'Failed to create counter-argument'}), 500


@app.route('/api/effectiveness-ratings', methods=['POST'])
@login_required
def create_effectiveness_rating_v1():
    data = request.get_json() or {}
    counter_argument_id = data.get('counter_argument_id')
    persona_id = data.get('persona_archetype_id')
    score = data.get('score')

    if not isinstance(counter_argument_id, int):
        return jsonify({'error': 'counter_argument_id must be an integer'}), 400
    if persona_id is not None and not isinstance(persona_id, int):
        return jsonify({'error': 'persona_archetype_id must be an integer or null'}), 400
    if not isinstance(score, int):
        return jsonify({'error': 'score must be an integer (1..5)'}), 400

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    counter_ids = {c.get('id') for c in (store.get('counter_arguments', []) or []) if isinstance(c, dict) and isinstance(c.get('id'), int)}
    persona_ids = {p.get('id') for p in (store.get('persona_archetypes', []) or []) if isinstance(p, dict) and isinstance(p.get('id'), int)}

    if counter_argument_id not in counter_ids:
        return jsonify({'error': 'Counter-argument not found'}), 404
    if persona_id is not None and persona_id not in persona_ids:
        return jsonify({'error': 'Persona archetype not found'}), 404

    try:
        rating = argstore.create_rating(
            store,
            counter_argument_id=counter_argument_id,
            persona_archetype_id=persona_id,
            score=score,
            user_id=current_user.id,
        )
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
        return jsonify({'success': True, 'rating': rating}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error creating rating: {e}")
        return jsonify({'error': 'Failed to create rating'}), 500

# ========================================
# Resource library
# ========================================


def _build_resource_library():
    """Fold every resource source into one catalogue (all review states)."""
    return library.build_library(
        curated=load_json_file(RESOURCES_FILE, []),
        store=argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE),
        gallery_seeds=(
            (load_json_file(GALLERY_FILE, []) or []) + (load_json_file(BONPOTE_GALLERY_FILE, []) or [])
        ),
        submissions=load_json_file(SUBMISSIONS_FILE, []),
        contributions=load_json_file(CONTRIBUTIONS_FILE, []),
        reviews=library.load_reviews(RESOURCE_REVIEWS_FILE),
        resolve_user=db.get_user_by_id,
    )


@app.route('/api/library', methods=['GET'])
def get_resource_library():
    """Browse admin-reviewed external resources.

    Query params: `type`, `topic`, `publisher`, `q`, `sort` (used|recent|title).
    Facet counts are computed before filtering by the facets themselves, so the
    counts stay stable as the user narrows down.
    """
    entries = [e for e in _build_resource_library() if e['review']['status'] == library.STATUS_APPROVED]

    q = (request.args.get('q') or '').strip().lower()
    if q:
        def _matches(entry):
            haystack = ' '.join([
                str(entry.get('title') or ''),
                str(entry.get('description') or ''),
                str(entry.get('publisher') or ''),
                ' '.join(entry.get('topics') or []),
            ]).lower()
            return q in haystack
        entries = [e for e in entries if _matches(e)]

    type_filter = (request.args.get('type') or '').strip().lower()
    topic_filter = (request.args.get('topic') or '').strip().lower()
    publisher_filter = (request.args.get('publisher') or '').strip().lower()

    def by_type(items):
        return [e for e in items if e['type'] == type_filter] if type_filter else items

    def by_topic(items):
        if not topic_filter:
            return items
        return [e for e in items if any(t.lower() == topic_filter for t in e['topics'])]

    def by_publisher(items):
        return [e for e in items if e['publisher'].lower() == publisher_filter] if publisher_filter else items

    # Each facet is counted against the *other* filters but not its own, so the
    # options a facet offers always lead somewhere. Without this, landing on a
    # type filter would advertise topics that yield nothing within that type.
    facets = {
        'types': library.facet_counts(by_publisher(by_topic(entries)))['types'],
        'topics': library.facet_counts(by_publisher(by_type(entries)))['topics'],
        'publishers': library.facet_counts(by_topic(by_type(entries)))['publishers'],
    }

    entries = by_publisher(by_topic(by_type(entries)))

    sort_by = (request.args.get('sort') or 'used').strip().lower()
    if sort_by == 'recent':
        entries.sort(key=lambda e: str(e.get('submitted_at') or ''), reverse=True)
    elif sort_by == 'title':
        entries.sort(key=lambda e: str(e.get('title') or '').lower())
    else:
        entries.sort(key=lambda e: (-e['used_in_responses'], str(e.get('title') or '').lower()))

    return jsonify({'resources': entries, 'facets': facets, 'total': len(entries)})


@app.route('/api/admin/library', methods=['GET'])
def get_admin_resource_library():
    """Every resource with its review state — the admin's review queue."""
    if not _is_valid_admin_token(request):
        return jsonify({'error': 'Unauthorized'}), 401

    entries = _build_resource_library()
    status_filter = (request.args.get('status') or '').strip().lower()
    if status_filter:
        entries = [e for e in entries if e['review']['status'] == status_filter]

    counts = {}
    for entry in _build_resource_library():
        status = entry['review']['status']
        counts[status] = counts.get(status, 0) + 1

    return jsonify({
        'resources': entries,
        'counts': counts,
        'types': [{'value': t, 'label': library.TYPE_LABELS[t]} for t in library.TYPES],
    })


@app.route('/api/admin/library/review', methods=['POST'])
def review_resource():
    """Approve or reject a resource, optionally correcting its categorisation."""
    if not _is_valid_admin_token(request):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    status = (data.get('status') or '').strip().lower()
    topics = data.get('topics')

    if not url:
        return jsonify({'error': 'A resource URL is required'}), 400
    if topics is not None and not isinstance(topics, list):
        return jsonify({'error': 'topics must be a list'}), 400

    try:
        reviews = library.load_reviews(RESOURCE_REVIEWS_FILE)
        library.record_review(
            reviews,
            url,
            status=status,
            # Admin sessions are password-based and share one identity, so the
            # roll records the role rather than a per-person account.
            reviewer=(data.get('reviewer') or 'admin').strip() or 'admin',
            resource_type=(data.get('type') or None),
            topics=topics,
            title=(data.get('title') or None),
            description=(data.get('description') or None),
            note=(data.get('note') or None),
        )
        library.save_reviews(RESOURCE_REVIEWS_FILE, reviews)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error reviewing resource: {e}")
        return jsonify({'error': 'Failed to save review'}), 500

    key = library.entry_key(url)
    updated = next((e for e in _build_resource_library() if e['id'] == library.entry_id(key)), None)
    return jsonify({'success': True, 'resource': updated})


@app.route('/api/importance-ratings', methods=['POST'])
@login_required
def create_importance_rating_v1():
    """Rate how urgent a claim is to debunk (1..5).

    One standing vote per person per claim — re-rating replaces the old score.
    """
    data = request.get_json() or {}
    argument_id = data.get('argument_id')
    score = data.get('score')

    if not isinstance(argument_id, int):
        return jsonify({'error': 'argument_id must be an integer'}), 400
    if not isinstance(score, int):
        return jsonify({'error': 'score must be an integer (1..5)'}), 400

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, _personas_by_id = argstore.build_indexes(store)
    if argument_id not in args_by_id:
        return jsonify({'error': 'Argument not found'}), 404

    try:
        rating = argstore.set_importance_rating(
            store,
            argument_id=argument_id,
            score=score,
            user_id=current_user.id,
        )
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)

        scores = [
            int(ir['score'])
            for ir in argstore.importance_by_argument(store).get(argument_id, [])
        ]
        return jsonify({
            'success': True,
            'rating': rating,
            'importance': {
                'avg_score': (sum(scores) / len(scores)) if scores else None,
                'count': len(scores),
                'my_score': rating['score'],
            },
        }), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error creating importance rating: {e}")
        return jsonify({'error': 'Failed to save importance rating'}), 500


@app.route('/api/resources', methods=['POST'])
@login_required
def create_resource_v1():
    """Submit a standalone resource (study, article, chart) to the shared library.

    Resources are deduped by normalized URL, so re-submitting a known source
    enriches it instead of creating a second entry.
    """
    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    title = (data.get('title') or '').strip()
    description = (data.get('description') or '').strip()
    tags = data.get('tags')
    argument_id = data.get('argument_id')

    if not url:
        return jsonify({'error': 'A resource URL is required'}), 400
    if not url.lower().startswith(('http://', 'https://')):
        return jsonify({'error': 'The URL must start with http:// or https://'}), 400
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        return jsonify({'error': 'tags must be a list'}), 400
    if argument_id is not None and not isinstance(argument_id, int):
        return jsonify({'error': 'argument_id must be an integer or null'}), 400

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, _personas_by_id = argstore.build_indexes(store)
    if argument_id is not None and argument_id not in args_by_id:
        return jsonify({'error': 'Argument not found'}), 404

    try:
        resource = argstore.upsert_resource(
            store,
            url,
            title=title or None,
            description=description or None,
            tags=[str(t) for t in tags],
            argument_ids=[argument_id] if argument_id is not None else [],
            created_by_user_id=current_user.id,
        )
        # Marks this as a deliberate submission to the library, as opposed to a
        # URL the store created on its own while saving someone's debunk.
        resource['submitted_directly'] = True
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
        return jsonify({'success': True, 'resource': resource}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error creating resource: {e}")
        return jsonify({'error': 'Failed to save resource'}), 500


# ========================================
# Contributor roles & source-verified stamp
# ========================================


@app.route('/api/roles', methods=['GET'])
def list_roles():
    """The contributor roles a member can take on, with duties and promotion rules."""
    return jsonify(roleslib.catalog())


@app.route('/api/me/roles', methods=['GET'])
@login_required
def my_roles():
    """The signed-in user's roles with progress toward confirmed (promotes if earned)."""
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    return jsonify({'roles': roleslib.describe_roles(store, current_user.id)})


@app.route('/api/me/roles', methods=['POST'])
@login_required
def take_role():
    """Declare a role. You start as an apprentice; your work confirms you."""
    role = ((request.get_json() or {}).get('role') or '').strip()
    if role not in roleslib.ROLE_CATALOG:
        return jsonify({'error': 'Unknown role'}), 400
    if not db.add_user_role(current_user.id, role):
        return jsonify({'error': 'You already hold this role'}), 409

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    return jsonify({'success': True, 'roles': roleslib.describe_roles(store, current_user.id)}), 201


@app.route('/api/me/roles/<role>', methods=['DELETE'])
@login_required
def drop_role(role):
    """Step down from a role. Your past work still counts if you take it again."""
    if not db.remove_user_role(current_user.id, role):
        return jsonify({'error': 'You don’t hold this role'}), 404
    return jsonify({'success': True})


@app.route('/api/me/source-check-queue', methods=['GET'])
@login_required
def my_source_check_queue():
    """Debunks waiting for a source check this user is allowed to give."""
    sourcer_ids = set(db.get_role_tiers('sourcer'))
    if current_user.id not in sourcer_ids and current_user.role not in ('moderator', 'admin'):
        return jsonify({'items': []})

    try:
        limit = max(1, min(int(request.args.get('limit', 3)), 20))
    except ValueError:
        limit = 3
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    items = roleslib.source_check_queue(store, current_user.id, roleslib.check_weights(), limit)
    return jsonify({'items': items})


@app.route('/api/counter-arguments/<int:counter_argument_id>/source-checks', methods=['POST'])
@login_required
def check_counter_argument_sources(counter_argument_id: int):
    """A Sourcer's verdict on whether a debunk's sources back it up.

    Body: {verdict: supports|partial|does_not_support|broken, note?}.
    Resolving the stamp pays the agreeing checkers and, if verified, the author.
    """
    data = request.get_json() or {}
    verdict = (data.get('verdict') or '').strip()
    note = (data.get('note') or '').strip()[:500]

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    ca = next((c for c in store.get('counter_arguments', []) or []
               if isinstance(c, dict) and c.get('id') == counter_argument_id), None)
    if not ca:
        return jsonify({'error': 'Counter-argument not found'}), 404

    allowed, reason = roleslib.can_check_source(
        current_user.id, current_user.role, ca, set(db.get_role_tiers('sourcer')))
    if not allowed:
        return jsonify({'error': reason}), 403

    if not db.check_rate_limit(current_user.id, request.remote_addr, 'source_check', 50):
        return jsonify({'error': 'Rate limit exceeded. Maximum 50 source checks per day.'}), 429

    try:
        argstore.set_source_check(store, counter_argument_id=counter_argument_id,
                                  verdict=verdict, user_id=current_user.id, note=note)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    weights = roleslib.check_weights()
    payouts = roleslib.settle_source_rewards(store, ca, weights)
    argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)

    for uid, points in payouts:
        db.update_user_reputation(uid, points)
    # A resolved stamp may complete someone's path to confirmed Sourcer.
    for uid in {uid for uid, _ in payouts}:
        roleslib.describe_roles(store, uid, weights=weights)

    checks = argstore.source_checks_by_counter(store).get(counter_argument_id, [])
    return jsonify({
        'success': True,
        'source_check': roleslib.source_status(checks, weights),
        'points_earned': sum(p for uid, p in payouts if uid == current_user.id),
    }), 201


if __name__ == '__main__':
    # Initialize database
    db.init_db()
    
    # Initialize data files if they don't exist
    if not os.path.exists(RESOURCES_FILE):
        save_json_file(RESOURCES_FILE, [])
    if not os.path.exists(CONTRIBUTIONS_FILE):
        save_json_file(CONTRIBUTIONS_FILE, [])
    if not os.path.exists(SUBMISSIONS_FILE):
        save_json_file(SUBMISSIONS_FILE, [])
    if not os.path.exists(ARGUMENTATION_FILE):
        argstore.save_store(ARGUMENTATION_FILE, argstore.default_store(), PERSONA_ARCHETYPES_FILE)
    
    # Production-safety warnings: never ship with the committed defaults.
    if app.secret_key == 'dev-secret-key-change-in-production':
        print('WARNING: using the default SECRET_KEY. Set SECRET_KEY in the environment before deploying.')
    if ADMIN_PASSWORD == 'change_me_in_production':
        print('WARNING: using the default ADMIN_PASSWORD. Set ADMIN_PASSWORD in the environment before deploying.')

    # Debug mode is opt-in via env, so a production launch can't accidentally
    # expose the interactive debugger / code execution surface.
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')

    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=debug)
