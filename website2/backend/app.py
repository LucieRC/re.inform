from flask import Flask, request, jsonify, send_from_directory, session, has_request_context
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import json
import os
import secrets
import time
import unicodedata
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

# `ENV=production` turns the committed dev defaults from warnings into refusals.
ENV = (os.environ.get('ENV') or 'development').strip().lower()

# CORS: the frontend is served by this same app, so the browser needs no CORS
# at all. Other origins (a separate dev server, a future domain) must be listed
# in CORS_ORIGINS, comma-separated, e.g.
#   CORS_ORIGINS=http://localhost:3000,https://reinform.example
# Credentials (the login cookie) are only ever shared with those origins;
# before, any site could make logged-in requests on a visitor's behalf.
CORS_ORIGINS = [o.strip().rstrip('/') for o in (os.environ.get('CORS_ORIGINS') or '').split(',') if o.strip()]
if CORS_ORIGINS:
    CORS(app, resources={r'/api/*': {'origins': CORS_ORIGINS}}, supports_credentials=True)
# The login cookie is not sent on cross-site POSTs (basic CSRF protection).
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True


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
# Claims and answers live only in argumentation.json. The old contributions.json
# and submissions.json were folded into it (bin/migrate_legacy_contributions.py)
# and moved to archive/website2-legacy-data/.
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
# Tokens expire after ADMIN_TOKEN_TTL seconds. token → expiry (epoch seconds).
ADMIN_TOKEN_TTL = 12 * 3600
ADMIN_LOGIN_ATTEMPTS_PER_DAY = 30
_active_admin_tokens = {}


def _is_valid_admin_token(req) -> bool:
    """Return True if the request carries a valid, unexpired 'Bearer <token>' admin token."""
    auth = req.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return False
    token = auth[len('Bearer '):].strip()
    expiry = _active_admin_tokens.get(token) if token else None
    if expiry is None:
        return False
    if expiry < time.time():
        _active_admin_tokens.pop(token, None)
        return False
    return True


def require_admin(f):
    """Admin endpoints: a signed-in account with users.role 'admin', or the
    admin-page token (password login, kept for the current admin page)."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        if current_user.is_authenticated and current_user.role == 'admin':
            return f(*args, **kwargs)
        if _is_valid_admin_token(request):
            return f(*args, **kwargs)
        return api_error('admin_auth_required', 'Unauthorized', 401)
    return wrapped

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
                return api_error('forbidden_role', 'Insufficient permissions', 403)
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

# ========================================
# Error responses
# ========================================
# Every error carries a stable machine `code` (snake_case, never translated)
# next to the English `error` text, which stays for older callers. The frontend
# maps `code` to its own wording. The full list is in backend/README.md.

def api_error(code, message, status=400, **extra):
    payload = {'error': message, 'code': code}
    payload.update(extra)
    return jsonify(payload), status


# ValueErrors raised by the stores carry English text only; map them to codes.
_VALUE_ERROR_CODES = (
    ('Username or email already exists', 'account_exists'),
    ('Cannot endorse yourself', 'endorse_self'),
    ('Already endorsed this user', 'already_endorsed'),
    ('Argument text is required', 'claim_text_required'),
    ('Counter-argument text is required', 'answer_text_required'),
    ('Resource URL is required', 'url_required'),
    ('A resource URL is required', 'url_required'),
    ('Score must be', 'invalid_score'),
    ('verdict must be one of', 'invalid_verdict'),
    ('status must be', 'invalid_status'),
    ('type must be one of', 'invalid_type'),
    ('Links must be full web addresses', 'invalid_url'),
)


def _value_error_code(exc):
    text = str(exc)
    for prefix, code in _VALUE_ERROR_CODES:
        if text.startswith(prefix):
            return code
    return 'invalid_input'


@login_manager.unauthorized_handler
def _unauthorized():
    # API callers get JSON, not a redirect to a POST-only login route.
    return api_error('auth_required', 'Log in to do this.', 401)


_HTTP_CODES = {400: 'bad_request', 401: 'auth_required', 403: 'forbidden', 404: 'not_found',
               405: 'method_not_allowed', 413: 'payload_too_large', 415: 'unsupported_media_type',
               429: 'rate_limited',
               500: 'server_error'}


def _avatar_too_large():
    return api_error('image_too_large', 'Image is too large (max 3 MB)', 413)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(413)
@app.errorhandler(415)
@app.errorhandler(500)
def _http_error(exc):
    status = getattr(exc, 'code', 500) or 500
    if status == 413 and request.endpoint == 'upload_avatar':
        # Past MAX_CONTENT_LENGTH the avatar view never gets to measure the file.
        return _avatar_too_large()
    if not request.path.startswith('/api/'):
        if status == 404 and os.path.isfile(os.path.join(app.static_folder, '404.html')):
            # Unknown pages get the site's own French 404 page (API 404s stay JSON).
            response = send_from_directory(app.static_folder, '404.html')
            response.status_code = 404
            return response
        return exc if status != 500 else ('Internal Server Error', 500)
    return api_error(_HTTP_CODES.get(status, f'http_{status}'),
                     getattr(exc, 'description', None) or 'Request failed', status)


@app.after_request
def _ensure_error_code(response):
    """Safety net: any JSON error that slipped through without a code gets one."""
    if response.status_code >= 400 and response.is_json:
        data = response.get_json(silent=True)
        if isinstance(data, dict) and 'error' in data and 'code' not in data:
            data['code'] = _HTTP_CODES.get(response.status_code, f'http_{response.status_code}')
            response.set_data(json.dumps(data))
    return response


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

def site_url():
    """Public base URL of the site, for links in emails and other out-of-page text.

    SITE_URL wins when set (e.g. behind a proxy or on a real domain). Otherwise
    the host the current request came in on, and outside a request the port
    this process listens on (PORT, default 5002).
    """
    configured = (os.environ.get('SITE_URL') or '').strip()
    if configured:
        return configured.rstrip('/')
    if has_request_context():
        return request.host_url.rstrip('/')
    return f"http://localhost:{int(os.environ.get('PORT', 5002))}"


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
        
        Review at: {site_url()}/admin
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

USERNAME_MIN, USERNAME_MAX = 3, 30
_USERNAME_EXTRA_CHARS = set('._-')


def _username_chars_ok(username):
    """Letters (accents welcome), digits, and . _ - only; no spaces or markup."""
    return all(
        ch in _USERNAME_EXTRA_CHARS
        or unicodedata.category(ch)[0] == 'L'
        or unicodedata.category(ch) == 'Nd'
        for ch in username
    )


@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.get_json(silent=True) or {}
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        # Validation
        if not username or not email or not password:
            return api_error('fields_required', 'All fields are required', 400)
        
        username = unicodedata.normalize('NFC', username)
        if len(username) < USERNAME_MIN:
            return api_error('username_too_short', 'Le nom d’utilisateur doit faire au moins 3 caractères.', 400)
        if len(username) > USERNAME_MAX:
            return api_error('username_too_long', 'Le nom d’utilisateur doit faire au plus 30 caractères.', 400)
        if not _username_chars_ok(username):
            return api_error('username_invalid_chars',
                             'Le nom d’utilisateur ne peut contenir que des lettres, des chiffres, '
                             'des points, des tirets et des tirets bas.', 400)
        
        if len(password) < 6:
            return api_error('password_too_short', 'Password must be at least 6 characters', 400)
        
        # Create user
        try:
            user_id = db.create_user(username, email, password)
            user_dict = db.get_user_by_id(user_id)
            user = User(user_dict)
            login_user(user, remember=True)
            claimed_votes = _claim_pending_votes(user.id)

            return jsonify({
                'success': True,
                'message': 'Registration successful',
                'code': 'registered',
                'claimed_votes': claimed_votes,
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
            return api_error(_value_error_code(e), str(e), 400)
            
    except Exception as e:
        import traceback
        print(f"Registration error: {e}")
        traceback.print_exc()
        return api_error('server_error', 'Registration failed', 500)

@app.route('/api/login', methods=['POST'])
def login():
    """Login a user."""
    try:
        data = request.get_json(silent=True) or {}
        username = data.get('username', '').strip()
        password = data.get('password', '')
        remember = data.get('remember', False)
        
        if not username or not password:
            return api_error('credentials_required', 'Username and password required', 400)
        
        # Get user (sign-up stores names NFC-normalised; older names may not be).
        user_dict = (db.get_user_by_username(username)
                     or db.get_user_by_username(unicodedata.normalize('NFC', username)))
        if not user_dict:
            return api_error('invalid_credentials', 'Invalid username or password', 401)
        
        # Verify password
        if not db.verify_password(user_dict, password):
            return api_error('invalid_credentials', 'Invalid username or password', 401)
        
        # Login user
        user = User(user_dict)
        login_user(user, remember=remember)
        claimed_votes = _claim_pending_votes(user.id)

        return jsonify({
            'success': True,
            'message': 'Login successful',
            'code': 'logged_in',
            'claimed_votes': claimed_votes,
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
        return api_error('server_error', 'Login failed', 500)

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    """Logout current user."""
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully', 'code': 'logged_out'})

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
    return jsonify({'authenticated': False, 'pending_votes': len(db.get_pending_votes(_anon_id()))})


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
        return api_error('avatar_not_found', 'Avatar not found', 404)
    # The filename is generated by us (never user-controlled), so directory
    # traversal isn't possible — but stay explicit anyway.
    if os.path.basename(file_name) != file_name:
        return api_error('avatar_not_found', 'Avatar not found', 404)
    return send_from_directory(AVATARS_DIR, file_name, max_age=3600)


@app.route('/api/me/avatar', methods=['POST'])
@login_required
def upload_avatar():
    """Upload or replace the signed-in user's profile picture."""
    file = request.files.get('avatar')
    if not file or not file.filename:
        return api_error('file_required', 'No file provided', 400)

    ext = _sniff_image_ext(file.stream)
    if not ext:
        return api_error('unsupported_image', 'Unsupported image format (use JPG, PNG, GIF or WebP)', 400)

    file.seek(0, os.SEEK_END)
    if file.tell() > AVATAR_MAX_BYTES:
        return _avatar_too_large()
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

def _claim_contribution_view(arg):
    """One claim the user added, for profile contribution lists.

    Shaped close to legacy_contribution_view (narrative/title/tags/date) so the
    existing contribution cards render it, plus `kind: 'claim'`, `text` and `href`.
    """
    text = str(arg.get('text') or '').strip()
    title = text if len(text) <= 80 else text[:79].rstrip() + '…'
    return {
        'kind': 'claim',
        'id': arg.get('id'),
        'argument_id': arg.get('id'),
        'href': f"argument.html?id={arg.get('id')}",
        'title': title,
        'text': text,
        'narrative': text,
        'tags': list(arg.get('tags') or []),
        'date': arg.get('created_at'),
        'status': argstore.status_of(arg),
        'user_id': arg.get('created_by_user_id'),
    }


@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get user profile."""
    user_dict = db.get_user_by_id(user_id)
    if not user_dict:
        return api_error('user_not_found', 'User not found', 404)
    
    endorsements = db.get_user_endorsements(user_id)

    # The user's answers, from the argumentation store (the single source of
    # truth since contributions.json/submissions.json were retired). Public
    # list = published answers; submission_count also counts ones in review.
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, personas_by_id = argstore.build_indexes(store)
    own_answers = [
        ca for ca in (store.get('counter_arguments') or [])
        if isinstance(ca, dict) and ca.get('created_by_user_id') == user_id
        and argstore.status_of(ca) != argstore.STATUS_REJECTED
    ]
    user_contributions = []
    for ca in own_answers:
        if not argstore.is_published(ca) or not argstore.is_published(args_by_id.get(ca.get('argument_id'))):
            continue
        view = argstore.legacy_contribution_view(store, ca, args_by_id=args_by_id, personas_by_id=personas_by_id)
        if view:
            view['kind'] = 'response'
            view['username'] = user_dict['username']
            user_contributions.append(view)
    # Claims the user added: published ones are public; pending ones only
    # count towards submission_count (like answers in review).
    own_claims = [
        a for a in (store.get('arguments') or [])
        if isinstance(a, dict) and a.get('created_by_user_id') == user_id
        and argstore.status_of(a) != argstore.STATUS_REJECTED
    ]
    for arg in own_claims:
        if argstore.is_published(arg):
            view = _claim_contribution_view(arg)
            view['username'] = user_dict['username']
            user_contributions.append(view)
    user_contributions.sort(key=lambda c: str(c.get('date') or ''), reverse=True)

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
        'roles': roleslib.describe_roles(store, user_id, public_only=True),
        'contribution_count': len(user_contributions),
        'submission_count': len(own_answers) + len(own_claims),
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

    Powers the profile 'status tracker': the user's answers in the argumentation
    store (published answers show as 'approved', plus any pending/rejected), the
    claims they added (`kind: 'claim'`, same statuses) and resources they
    submitted to the library (with the library's review status).
    Own-profile only — never exposed for another user, since pending/rejected
    work is private.
    """
    uid = current_user.id
    items = []

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, _personas = argstore.build_indexes(store)
    tracker_status = {
        argstore.STATUS_PUBLISHED: 'approved',
        argstore.STATUS_PENDING: 'pending',
        argstore.STATUS_REJECTED: 'rejected',
    }
    for ca in store.get('counter_arguments') or []:
        if not isinstance(ca, dict) or ca.get('created_by_user_id') != uid:
            continue
        arg = args_by_id.get(ca.get('argument_id')) or {}
        title = str(arg.get('text') or ca.get('text') or '').strip()
        urls = argstore.resource_urls_for(store, ca)
        status = argstore.status_of(ca)
        items.append({
            'kind': 'response',
            'title': title[:140] or 'Untitled response',
            'status': tracker_status[status],
            'submitted_at': ca.get('created_at'),
            'source': urls[0] if urls else None,
            'argument_id': ca.get('argument_id'),
            'counter_argument_id': ca.get('id'),
            'href': (f"argument.html?id={ca.get('argument_id')}"
                     if status == argstore.STATUS_PUBLISHED and argstore.is_published(arg) else None),
        })

    # Claims the user added (own-profile, so pending/rejected show too).
    for arg in store.get('arguments') or []:
        if not isinstance(arg, dict) or arg.get('created_by_user_id') != uid:
            continue
        status = argstore.status_of(arg)
        text = str(arg.get('text') or '').strip()
        items.append({
            'kind': 'claim',
            'title': text[:140] or 'Untitled claim',
            'text': text,
            'status': tracker_status[status],
            'submitted_at': arg.get('created_at'),
            'date': arg.get('created_at'),
            'argument_id': arg.get('id'),
            'href': (f"argument.html?id={arg.get('id')}"
                     if status == argstore.STATUS_PUBLISHED else None),
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
    followed users authored: published counter-arguments in the argumentation
    store, and resources they submitted. Content-first (names the thing, not
    points); bare ratings are intentionally left out as too noisy.
    """
    followed = {u['id']: u for u in db.get_following(current_user.id)}
    if not followed:
        return jsonify({'items': []})

    def _snippet(text, n=120):
        text = (text or '').strip()
        return (text[:n].rstrip() + '…') if len(text) > n else text

    events = []

    # Counter-arguments and submitted resources from the argumentation store.
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    public_arg_ids = {a.get('id') for a in argstore.published(store.get('arguments'))}
    for ca in argstore.published(store.get('counter_arguments')):
        uid = ca.get('created_by_user_id')
        if ca.get('argument_id') not in public_arg_ids:
            continue
        if uid in followed and ca.get('created_at'):
            events.append({
                'user': {'id': uid, 'username': followed[uid]['username']},
                'kind': 'response',
                'verb': 'wrote a response',
                'verb_code': 'wrote_response',
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
                'verb_code': 'shared_source',
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
        return api_error('not_your_account', 'Unauthorized', 403)
    
    data = request.get_json()
    tag = data.get('tag', '').strip().lower()
    
    if not tag:
        return api_error('tag_required', 'Tag is required', 400)
    
    success = db.follow_tag(user_id, tag)
    if success:
        return jsonify({'success': True, 'message': f'Now following #{tag}', 'code': 'tag_followed', 'tag': tag})
    else:
        return api_error('already_following_tag', 'Already following this tag', 400)

@app.route('/api/user/<int:user_id>/followed-tags/<tag>', methods=['DELETE'])
@login_required
def unfollow_tag(user_id, tag):
    """Unfollow a tag."""
    if current_user.id != user_id:
        return api_error('not_your_account', 'Unauthorized', 403)
    
    success = db.unfollow_tag(user_id, tag)
    if success:
        return jsonify({'success': True, 'message': f'Unfollowed #{tag}', 'code': 'tag_unfollowed', 'tag': tag})
    else:
        return api_error('not_following_tag', 'Not following this tag', 400)

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
        return api_error('follow_self', 'Cannot follow yourself', 400)
    
    # Check if target user exists
    target_user = db.get_user_by_id(user_id)
    if not target_user:
        return api_error('user_not_found', 'User not found', 404)
    
    success = db.follow_user(current_user.id, user_id)
    if success:
        return jsonify({'success': True, 'message': f'Now following {target_user["username"]}', 'code': 'user_followed'})
    else:
        return api_error('already_following_user', 'Already following this user', 400)

@app.route('/api/user/<int:user_id>/follow', methods=['DELETE'])
@login_required
def unfollow_user_endpoint(user_id):
    """Unfollow a user."""
    success = db.unfollow_user(current_user.id, user_id)
    if success:
        return jsonify({'success': True, 'message': 'Unfollowed user', 'code': 'user_unfollowed'})
    else:
        return api_error('not_following_user', 'Not following this user', 400)

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

def _published_contribution_views(store):
    """Every public answer, shaped like the old contributions.json entries."""
    args_by_id, personas_by_id = argstore.build_indexes(store)
    usernames = {}
    out = []
    for ca in argstore.published(store.get('counter_arguments')):
        if not argstore.is_published(args_by_id.get(ca.get('argument_id'))):
            continue
        view = argstore.legacy_contribution_view(store, ca, args_by_id=args_by_id, personas_by_id=personas_by_id)
        if not view:
            continue
        uid = view.get('user_id')
        if isinstance(uid, int):
            if uid not in usernames:
                u = db.get_user_by_id(uid)
                usernames[uid] = u['username'] if u else None
            view['username'] = usernames[uid]
        out.append(view)
    return out


@app.route('/api/contributions', methods=['GET'])
def get_contributions():
    """All public answers, in the old contributions.json shape.

    Kept for old callers: contributions.json is retired and this is now a view
    over the argumentation store (`id` is the counter-argument id). New code
    should read /api/arguments and /api/arguments/<id>.
    """
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    contributions = _published_contribution_views(store)

    # Optional filtering and sorting
    sort_by = request.args.get('sort', 'date')  # date, title
    tag_filter = request.args.get('tag', None)
    
    # Filter by tag if specified
    if tag_filter:
        contributions = [c for c in contributions if tag_filter in c.get('tags', [])]
    
    # Sort
    if sort_by == 'title':
        contributions.sort(key=lambda x: str(x.get('title') or ''))
    else:  # date
        contributions.sort(key=lambda x: str(x.get('date') or ''), reverse=True)
    
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
        if library.is_placeholder_curated(r):
            continue  # draft entry with no content yet (e.g. "..." summary): hidden until filled

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
        return api_error('not_found', 'Not found', 404)

    return send_from_directory(ARTICLES_DIR, filename, mimetype='text/markdown; charset=utf-8')


@app.route('/api/submit-contribution', methods=['POST'])
def submit_contribution():
    """Retired: the free-text claim+answer form.

    Contributions now go straight into the argumentation store as structured
    entities: create (or find) the claim with POST /api/arguments, then post
    the answer with POST /api/counter-arguments (one target persona, optional
    sources). Nothing is written here any more.
    """
    return api_error(
        'endpoint_moved',
        'This form has moved. Add the claim with POST /api/arguments, '
        'then the answer with POST /api/counter-arguments.',
        410,
        moved_to=['/api/arguments', '/api/counter-arguments'],
    )

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Password login for the admin page; returns a short-lived bearer token."""
    data = request.get_json(silent=True) or {}
    password = str(data.get('password') or '')

    if ENV == 'production' and ADMIN_PASSWORD == 'change_me_in_production':
        return api_error('admin_password_not_set',
                         'Admin login is disabled until ADMIN_PASSWORD is set.', 503)
    # Every attempt counts, so the password can't be guessed by brute force.
    if not db.check_rate_limit(None, request.remote_addr, 'admin_login', ADMIN_LOGIN_ATTEMPTS_PER_DAY):
        return api_error('rate_limited', 'Too many admin login attempts today.', 429)

    # Constant-time comparison to avoid leaking the password via timing.
    if password and secrets.compare_digest(password, ADMIN_PASSWORD):
        token = secrets.token_urlsafe(32)
        now = time.time()
        for t, exp in list(_active_admin_tokens.items()):
            if exp < now:
                _active_admin_tokens.pop(t, None)
        _active_admin_tokens[token] = now + ADMIN_TOKEN_TTL
        return jsonify({'success': True, 'token': token, 'expires_in': ADMIN_TOKEN_TTL})
    else:
        return api_error('invalid_admin_password', 'Invalid password', 401)


@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Invalidate the current admin token."""
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        _active_admin_tokens.pop(auth[len('Bearer '):].strip(), None)
    return jsonify({'success': True})

# ========================================
# Review queue (answers awaiting approval)
# ========================================
# The old submissions.json queue now lives in the argumentation store: an answer
# with status 'pending' (and its claim, if that is new too) waits here until an
# admin approves or rejects it, or trusted members vote it through. The URLs and
# response shape are the old ones so the admin page keeps working; `id` is now
# the counter-argument id.

def _review_items(store):
    """Answers that went (or are going) through review, as admin cards."""
    args_by_id, _personas = argstore.build_indexes(store)
    shown = {'published': 'approved', 'pending': 'pending', 'rejected': 'rejected'}
    out = []
    for ca in store.get('counter_arguments') or []:
        if not isinstance(ca, dict):
            continue
        status = argstore.status_of(ca)
        if status == argstore.STATUS_PUBLISHED and not ca.get('reviewed_at'):
            continue  # published directly, never queued
        arg = args_by_id.get(ca.get('argument_id')) or {}
        urls = argstore.resource_urls_for(store, ca)
        uid = ca.get('created_by_user_id')
        user = db.get_user_by_id(uid) if isinstance(uid, int) else None
        out.append({
            'id': ca.get('id'),
            'counter_argument_id': ca.get('id'),
            'argument_id': ca.get('argument_id'),
            'argument_is_new': argstore.status_of(arg) != argstore.STATUS_PUBLISHED or bool(arg.get('reviewed_at')),
            'narrative': arg.get('text'),
            'counter_argument': ca.get('text'),
            'source': urls[0] if urls else None,
            'sources': urls,
            'tags': list(arg.get('tags') or []),
            'persona_archetype_id': ca.get('persona_archetype_id'),
            'submitted_at': ca.get('created_at'),
            'status': shown[status],
            'reviewed_at': ca.get('reviewed_at'),
            'user_id': uid,
            'username': user['username'] if user else None,
        })
    return out


def _find_counter(store, counter_argument_id):
    return next((c for c in store.get('counter_arguments') or []
                 if isinstance(c, dict) and c.get('id') == counter_argument_id), None)


@app.route('/api/admin/submissions', methods=['GET'])
@require_admin
def get_submissions():
    """Answers in the review queue, plus ones already reviewed (admin token)."""
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    return jsonify(_review_items(store))

@app.route('/api/admin/approve-submission/<int:submission_id>', methods=['POST'])
@require_admin
def approve_submission(submission_id):
    """Publish a pending answer (and its claim if new). `submission_id` is the counter-argument id.

    Optional body: {tags: [...]} adds topic tags to the claim. A `title` is
    accepted for old callers but ignored: answers have no title of their own.
    """
    try:
        data = request.get_json(silent=True) or {}
        store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
        ca = _find_counter(store, submission_id)
        if not ca or argstore.status_of(ca) != argstore.STATUS_PENDING:
            return api_error('submission_not_pending', 'Submission not found or not pending', 404)

        # Anything saved before links were checked could hold an unsafe one;
        # don't publish it.
        for url in argstore.resource_urls_for(store, ca):
            if not argstore.is_http_url(url):
                return api_error('invalid_url', f'This submission\'s source is not a valid web link. {argstore.URL_ERROR}', 400)

        argstore.set_review_status(store, ca, argstore.STATUS_PUBLISHED, reviewed_by='admin')
        tags = data.get('tags')
        if isinstance(tags, list):
            args_by_id, _ = argstore.build_indexes(store)
            arg = args_by_id.get(ca.get('argument_id'))
            if arg is not None:
                clean = [argstore.normalize_text(str(t)) for t in tags if argstore.normalize_text(str(t or ''))]
                arg['tags'] = argstore._merge_unique(arg.get('tags'), clean)
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)

        view = next((v for v in _review_items(store) if v['id'] == submission_id), None)
        return jsonify({'success': True, 'contribution': view})

    except Exception as e:
        print(f"Error approving submission: {e}")
        return api_error('server_error', 'Failed to approve submission', 500)

@app.route('/api/admin/reject-submission/<int:submission_id>', methods=['POST'])
@require_admin
def reject_submission(submission_id):
    """Reject a pending answer. `submission_id` is the counter-argument id."""
    try:
        store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
        ca = _find_counter(store, submission_id)
        if not ca or argstore.status_of(ca) != argstore.STATUS_PENDING:
            return api_error('submission_not_pending', 'Submission not found or not pending', 404)

        argstore.set_review_status(store, ca, argstore.STATUS_REJECTED, reviewed_by='admin')
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
        return jsonify({'success': True})

    except Exception as e:
        print(f"Error rejecting submission: {e}")
        return api_error('server_error', 'Failed to reject submission', 500)

# ========================================
# Voting Endpoints
# ========================================

@app.route('/api/vote/<int:submission_id>', methods=['POST'])
@require_role('trusted', 'moderator', 'admin')
def vote_submission(submission_id):
    """Vote on a pending answer (trusted+ only). `submission_id` is the counter-argument id."""
    try:
        data = request.get_json(silent=True) or {}
        vote_type = data.get('vote_type')  # approve, reject, needs_work

        if vote_type not in ['approve', 'reject', 'needs_work']:
            return api_error('invalid_vote_type', 'Invalid vote type', 400)

        store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
        ca = _find_counter(store, submission_id)
        if not ca or argstore.status_of(ca) != argstore.STATUS_PENDING:
            return api_error('submission_not_pending', 'Submission not found or not pending', 404)

        # Check rate limit
        if not db.check_rate_limit(current_user.id, request.remote_addr, 'vote', 50):
            return api_error('rate_limited', 'Rate limit exceeded. Maximum 50 votes per day.', 429)

        # Get vote weight based on role
        weight = get_vote_weight(current_user.role)

        # Create vote
        db.create_vote(current_user.id, submission_id, vote_type, weight)

        # Get vote summary
        vote_summary = db.get_vote_summary(submission_id)

        # Auto-approve if score >= 3
        if vote_summary['approve_score'] >= 3:
            argstore.set_review_status(store, ca, argstore.STATUS_PUBLISHED,
                                       reviewed_by={'votes': vote_summary['approve_score']})
            argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)

            # Award reputation to contributor
            if isinstance(ca.get('created_by_user_id'), int):
                db.update_user_reputation(ca['created_by_user_id'], 10)

        return jsonify({
            'success': True,
            'vote_summary': vote_summary
        })

    except Exception as e:
        print(f"Error voting: {e}")
        return api_error('server_error', 'Failed to record vote', 500)

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
            return api_error('reason_required', 'Reason is required', 400)
        
        # Check rate limit
        if not db.check_rate_limit(current_user.id, request.remote_addr, 'endorsement', 10):
            return api_error('rate_limited', 'Rate limit exceeded. Maximum 10 endorsements per day.', 429)
        
        try:
            db.create_endorsement(current_user.id, user_id, reason)
            return jsonify({'success': True, 'message': 'User endorsed successfully', 'code': 'user_endorsed'})
        except ValueError as e:
            return api_error(_value_error_code(e), str(e), 400)
            
    except Exception as e:
        print(f"Error endorsing user: {e}")
        return api_error('server_error', 'Failed to endorse user', 500)

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
            return api_error('nomination_fields_required', 'Nominee username and reason are required', 400)
        
        nominee = db.get_user_by_username(nominee_username)
        if not nominee:
            return api_error('user_not_found', 'User not found', 404)
        
        if db.nominate_moderator(current_user.id, nominee['id'], reason):
            return jsonify({'success': True, 'message': 'Nomination submitted for admin review', 'code': 'nomination_submitted'})
        else:
            return api_error('server_error', 'Failed to submit nomination', 500)
            
    except Exception as e:
        print(f"Error nominating moderator: {e}")
        return api_error('server_error', 'Failed to nominate moderator', 500)

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
            'message': 'Nomination approved' if approved else 'Nomination rejected',
            'code': 'nomination_approved' if approved else 'nomination_rejected',
        })
        
    except Exception as e:
        print(f"Error reviewing nomination: {e}")
        return api_error('server_error', 'Failed to review nomination', 500)


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
    for r in argstore.counted_ratings(store):
        scores_by_counter.setdefault(r['counter_argument_id'], []).append(int(r['score']))

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

    # Group counter-arguments by argument (public ones only: pending and
    # rejected answers wait in the review queue).
    counters_by_arg = {}
    for ca in argstore.published(store.get('counter_arguments')):
        aid = ca.get('argument_id')
        if isinstance(aid, int):
            counters_by_arg.setdefault(aid, []).append(ca)

    # Importance ratings are cast on the claim itself: how urgent is this to debunk?
    importance_by_arg = argstore.importance_by_argument(store)
    viewer_id = current_user.id if current_user.is_authenticated else None
    pending_importance = {} if viewer_id is not None else {
        v['target_id']: v['score'] for v in db.get_pending_votes(_anon_id()) if v['kind'] == 'importance'
    }

    arguments = []
    for a in argstore.published(store.get('arguments')):
        if not isinstance(a.get('id'), int):
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
            pending_importance.get(a['id']),
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
    if not arg or not argstore.is_published(arg):
        return api_error('claim_not_found', 'Argument not found', 404)

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

    viewer_id = current_user.id if current_user.is_authenticated else None
    viewer_role = current_user.role if current_user.is_authenticated else None

    # Pre-group ratings by counter_argument_id (one per person x answer x profile),
    # and collect the viewer's own ratings so the page can show them.
    rating_by_counter = {}
    my_ratings_by_counter = {}
    for r in argstore.counted_ratings(store):
        cid = r['counter_argument_id']
        rating_by_counter.setdefault(cid, []).append(int(r['score']))
        if viewer_id is not None and r.get('user_id') == viewer_id:
            my_ratings_by_counter.setdefault(cid, []).append(
                {'persona_archetype_id': r.get('persona_archetype_id'), 'score': int(r['score']),
                 'pending': False})
    if viewer_id is None:
        # A visitor's votes wait in pending until they sign up; still show them back.
        for v in db.get_pending_votes(_anon_id()):
            if v['kind'] == 'effectiveness':
                my_ratings_by_counter.setdefault(v['target_id'], []).append(
                    {'persona_archetype_id': v['persona_id'], 'score': int(v['score']), 'pending': True})

    def _my_score(ca, mine):
        """The viewer's score for the answer's own profile, else for 'everyone'."""
        by_persona = {m['persona_archetype_id']: m['score'] for m in mine}
        own = ca.get('persona_archetype_id')
        if own in by_persona:
            return by_persona[own]
        return by_persona.get(None)

    # Source-verified stamp, plus whether the viewer can add a check.
    checks_by_counter = argstore.source_checks_by_counter(store)
    check_weights = roleslib.check_weights()
    sourcer_ids = set(db.get_role_tiers('sourcer'))

    def _source_check_for(ca):
        if not roleslib.has_sources(ca):
            return {'status': 'unsourced', 'count': 0, 'can_check': False, 'my_verdict': None}
        checks = checks_by_counter.get(ca.get('id'), [])
        can_check, refusal = roleslib.can_check_source(viewer_id, viewer_role, ca, sourcer_ids)
        summary = roleslib.source_status(checks, check_weights)
        return {
            'status': summary['status'],
            'count': summary['count'],
            'can_check': can_check,
            'refusal_code': refusal,
            'my_verdict': next((sc['verdict'] for sc in checks if sc.get('user_id') == viewer_id), None),
        }

    counter_arguments = []
    for ca in argstore.published(store.get('counter_arguments')):
        if ca.get('argument_id') != argument_id:
            continue

        persona = None
        pid = ca.get('persona_archetype_id')
        if isinstance(pid, int):
            persona = personas_by_id.get(pid)

        scores = rating_by_counter.get(ca.get('id'), [])
        avg_score = (sum(scores) / len(scores)) if scores else None
        mine = my_ratings_by_counter.get(ca.get('id'), [])

        counter_arguments.append({
            'id': ca.get('id'),
            'text': ca.get('text'),
            'source_url': ca.get('source_url'),
            'resources': _resources_for(ca),
            'persona_archetype': persona,
            'rating_count': len(scores),
            'avg_score': avg_score,
            'my_score': _my_score(ca, mine),
            'my_ratings': mine,
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
        return api_error('tags_not_list', 'tags must be a list', 400)

    try:
        store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
        arg = argstore.upsert_argument(store, text=text, created_by_user_id=current_user.id, tags=tags)
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
        return jsonify({'success': True, 'argument': arg}), 201
    except ValueError as e:
        return api_error(_value_error_code(e), str(e), 400)
    except Exception as e:
        print(f"Error creating argument: {e}")
        return api_error('server_error', 'Failed to create argument', 500)


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
        return api_error('invalid_argument_id', 'argument_id must be an integer', 400)
    # A debunk targets exactly one persona.
    if not isinstance(persona_id, int):
        return api_error('persona_required', 'A target persona is required', 400)

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, personas_by_id = argstore.build_indexes(store)
    if not argstore.is_published(args_by_id.get(argument_id)):
        return api_error('claim_not_found', 'Argument not found', 404)
    if persona_id not in personas_by_id:
        return api_error('persona_not_found', 'Persona archetype not found', 404)
    # Numbered EVS clusters are analysis placeholders, not targets for a response.
    if personas_by_id[persona_id].get('layer') == 'evs_cluster':
        return api_error('persona_not_named', 'Choose a named persona', 400)

    # Collect source URLs from `sources` (list) and/or legacy `source_url`.
    raw_sources = []
    sources = data.get('sources')
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, str) and s.strip():
                raw_sources.append(s.strip())
            elif isinstance(s, dict) and str(s.get('url') or '').strip():
                raw_sources.append(str(s['url']).strip())
    legacy = str(data.get('source_url') or '').strip()
    if legacy:
        raw_sources.append(legacy)
    # Sources are optional, but every one given must be a real web link.
    if any(not argstore.is_http_url(u) for u in raw_sources):
        return api_error('invalid_url', f'Source: {argstore.URL_ERROR}', 400)

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
        _clear_published_draft(current_user.id, argument_id, persona_id, text)
        return jsonify({'success': True, 'counter_argument': counter}), 201
    except ValueError as e:
        return api_error(_value_error_code(e), str(e), 400)
    except Exception as e:
        print(f"Error creating counter-argument: {e}")
        return api_error('server_error', 'Failed to create counter-argument', 500)


# ========================================
# Answer drafts
# ========================================
# An answer left unpublished stays on the author's profile. Only ever visible
# to its author, and never published (bin/test_data.py export drops the table).
# Saved repeatedly while typing, so it doesn't use the daily posting limits.
# Sources are kept as typed: the frontend must pass them through safeHref.

# Answers themselves have no length limit; this only keeps a draft a sane size.
DRAFT_TEXT_MAX = 20000
DRAFTS_PER_USER = 50
DRAFT_SOURCES_MAX = 50
DRAFT_SOURCE_LENGTH_MAX = 2048
DRAFT_EVERYONE_NAME = 'Tout le monde'


def _is_row_id(value):
    # bool is an int in Python; past 2^63 SQLite raises instead of not matching.
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value < 2 ** 63


def _draft_json(draft, claim_text, persona):
    pid = draft['persona_archetype_id']
    try:
        sources = json.loads(draft['sources'])
    except ValueError:
        sources = []
    return {
        'id': draft['id'],
        'argument_id': draft['argument_id'],
        'persona_archetype_id': pid,
        'text': draft['text'],
        'sources': sources,
        'updated_at': draft['updated_at'],
        'claim_text': claim_text,
        'persona_name': DRAFT_EVERYONE_NAME if pid is None else (persona or {}).get('name'),
        # A profile retired since the draft was saved opens the composer unset.
        'href': f"argument.html?id={draft['argument_id']}&write={pid if persona else 'any'}",
    }


def _clear_published_draft(user_id, argument_id, persona_id, text):
    """Remove the draft an answer just came from. Never fails the publish.

    Publishing needs a persona, so a draft started under « Tout le monde » can't
    match on persona alone: it goes too when it holds the text just published.
    """
    try:
        published = argstore.normalize_text(text)
        for d in db.get_answer_drafts(user_id):
            if d['argument_id'] != argument_id:
                continue
            same_persona = d['persona_archetype_id'] == persona_id
            same_general = d['persona_archetype_id'] is None and argstore.normalize_text(d['text']) == published
            if same_persona or same_general:
                db.delete_answer_draft(user_id, d['id'])
    except Exception as e:
        print(f"Error clearing draft after publishing: {e}")


@app.route('/api/me/drafts', methods=['GET'])
@login_required
def list_my_drafts():
    uid = current_user.id
    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, personas_by_id = argstore.build_indexes(store)

    out = []
    for d in db.get_answer_drafts(uid):
        arg = args_by_id.get(d['argument_id'])
        if arg is None:
            # An empty store means a missing file, not that every claim is gone.
            if args_by_id:
                db.delete_answer_draft(uid, d['id'])
            continue
        # Set aside by moderation: its page is hidden, but it can come back.
        if not argstore.is_published(arg):
            continue
        out.append(_draft_json(d, arg.get('text'), personas_by_id.get(d['persona_archetype_id'])))
    return jsonify({'drafts': out})


@app.route('/api/me/drafts', methods=['PUT'])
@login_required
def save_my_draft():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    argument_id = data.get('argument_id')
    persona_id = data.get('persona_archetype_id')
    text = data.get('text') if data.get('text') is not None else ''
    raw_sources = data.get('sources') if data.get('sources') is not None else []

    if not _is_row_id(argument_id):
        return api_error('invalid_argument_id', 'argument_id must be an integer', 400)
    if persona_id is not None and not _is_row_id(persona_id):
        return api_error('invalid_persona_id', 'persona_archetype_id must be an integer or null', 400)
    if not isinstance(text, str):
        return api_error('invalid_input', 'text must be a string', 400)
    if not isinstance(raw_sources, list):
        return api_error('invalid_sources', 'sources must be a list', 400)

    sources = []
    for s in raw_sources:
        url = s.get('url') if isinstance(s, dict) else s
        if isinstance(url, str) and url.strip():
            sources.append(url.strip())

    # Emptied composer: no draft left. Checked before the claim lookup so a
    # draft can still be cleared after its claim has gone.
    if not text.strip() and not sources:
        db.clear_answer_draft(current_user.id, argument_id, persona_id)
        return jsonify({'success': True, 'code': 'draft_deleted', 'draft': None})

    if len(text) > DRAFT_TEXT_MAX:
        return api_error('draft_text_too_long', f'A draft can be at most {DRAFT_TEXT_MAX} characters', 400,
                         max_length=DRAFT_TEXT_MAX)
    if len(sources) > DRAFT_SOURCES_MAX or any(len(u) > DRAFT_SOURCE_LENGTH_MAX for u in sources):
        return api_error('draft_sources_too_long', 'Too many sources, or one is too long', 400)

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, personas_by_id = argstore.build_indexes(store)
    arg = args_by_id.get(argument_id)
    if not argstore.is_published(arg):
        return api_error('claim_not_found', 'Argument not found', 404)
    persona = personas_by_id.get(persona_id) if persona_id is not None else None
    if persona_id is not None:
        if persona is None:
            return api_error('persona_not_found', 'Persona archetype not found', 404)
        if persona.get('layer') == 'evs_cluster':
            return api_error('persona_not_named', 'Choose a named persona', 400)

    saved = db.save_answer_draft(current_user.id, argument_id, persona_id, text,
                                 json.dumps(sources, ensure_ascii=False), DRAFTS_PER_USER)
    if saved is None:
        return api_error('drafts_limit', f'At most {DRAFTS_PER_USER} drafts: delete one to save a new one', 400,
                         limit=DRAFTS_PER_USER)
    return jsonify({'success': True, 'code': 'draft_saved', 'draft': _draft_json(saved, arg.get('text'), persona)})


@app.route('/api/me/drafts/<int:draft_id>', methods=['DELETE'])
@login_required
def delete_my_draft(draft_id):
    # Someone else's draft is reported exactly like one that doesn't exist.
    if not _is_row_id(draft_id) or not db.delete_answer_draft(current_user.id, draft_id):
        return api_error('draft_not_found', 'Draft not found', 404)
    return jsonify({'success': True, 'code': 'draft_deleted'})


# ========================================
# Votes before signing up
# ========================================
# Rating takes one tap and no account. A visitor's votes are kept as pending,
# keyed by a random id in their session cookie, and don't count anywhere until
# they register or log in — then they become that user's own ratings. So
# clearing cookies to vote again gains nothing.

ANON_VOTES_PER_VISITOR = 100
ANON_VOTES_PER_IP_PER_DAY = 300


def _anon_id(create=False):
    """This visitor's anonymous id, minted on their first vote."""
    anon_id = session.get('anon_id')
    if not anon_id and create:
        anon_id = secrets.token_urlsafe(16)
        session['anon_id'] = anon_id
        session.permanent = True  # survive closing the browser
    return anon_id


def _record_pending_vote(kind, target_id, score, persona_id=None):
    """Save an anonymous vote. Returns (pending_count, error_response)."""
    anon_id = _anon_id(create=True)
    pending = db.get_pending_votes(anon_id)
    already = any(
        v['kind'] == kind and v['target_id'] == target_id and v['persona_id'] == persona_id
        for v in pending
    )
    if not already:
        if len(pending) >= ANON_VOTES_PER_VISITOR:
            return None, api_error('anon_vote_limit', 'Create a free account to keep voting.', 403, needs_account=True)
        if db.count_recent_pending_votes_by_ip(request.remote_addr) >= ANON_VOTES_PER_IP_PER_DAY:
            return None, api_error('anon_votes_ip_limit', 'Too many votes from this network today. Create a free account to keep voting.', 429, needs_account=True)
    db.add_pending_vote(anon_id, kind, target_id, score, persona_id=persona_id, ip_address=request.remote_addr)
    return len(pending) + (0 if already else 1), None


def _claim_pending_votes(user_id):
    """Turn this session's pending votes into the user's own. Returns how many counted."""
    anon_id = session.pop('anon_id', None)
    votes = db.get_pending_votes(anon_id)
    if not votes:
        return 0

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, personas_by_id = argstore.build_indexes(store)
    args_by_id = {k: v for k, v in args_by_id.items() if argstore.is_published(v)}
    counter_ids = {c.get('id') for c in argstore.published(store.get('counter_arguments'))}
    # An account that already rated an answer keeps that rating (no stacking).
    already_rated = {
        (r.get('counter_argument_id'), r.get('persona_archetype_id'))
        for r in (store.get('ratings') or [])
        if isinstance(r, dict) and r.get('user_id') == user_id
    }

    claimed = 0
    for v in votes:
        try:
            if v['kind'] == 'importance' and v['target_id'] in args_by_id:
                argstore.set_importance_rating(store, argument_id=v['target_id'], score=v['score'], user_id=user_id)
                claimed += 1
            elif v['kind'] == 'effectiveness' and v['target_id'] in counter_ids:
                persona_id = v['persona_id'] if v['persona_id'] in personas_by_id else None
                if (v['target_id'], persona_id) in already_rated:
                    continue
                argstore.create_rating(
                    store,
                    counter_argument_id=v['target_id'],
                    persona_archetype_id=persona_id,
                    score=v['score'],
                    user_id=user_id,
                )
                already_rated.add((v['target_id'], persona_id))
                claimed += 1
        except ValueError:
            continue  # a vote that's no longer valid is simply dropped

    if claimed:
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
    db.delete_pending_votes(anon_id)
    return claimed


@app.route('/api/effectiveness-ratings', methods=['POST'])
def create_effectiveness_rating_v1():
    data = request.get_json() or {}
    counter_argument_id = data.get('counter_argument_id')
    persona_id = data.get('persona_archetype_id')
    score = data.get('score')

    if not isinstance(counter_argument_id, int):
        return api_error('invalid_counter_argument_id', 'counter_argument_id must be an integer', 400)
    if persona_id is not None and not isinstance(persona_id, int):
        return api_error('invalid_persona_id', 'persona_archetype_id must be an integer or null', 400)
    if not isinstance(score, int):
        return api_error('invalid_score', 'score must be an integer (1..5)', 400)

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    counter_ids = {c.get('id') for c in argstore.published(store.get('counter_arguments')) if isinstance(c.get('id'), int)}
    persona_ids = {p.get('id') for p in (store.get('persona_archetypes', []) or []) if isinstance(p, dict) and isinstance(p.get('id'), int)}

    if counter_argument_id not in counter_ids:
        return api_error('answer_not_found', 'Counter-argument not found', 404)
    if persona_id is not None and persona_id not in persona_ids:
        return api_error('persona_not_found', 'Persona archetype not found', 404)
    if not 1 <= score <= 5:
        return api_error('invalid_score', 'Score must be an integer between 1 and 5', 400)

    if not current_user.is_authenticated:
        pending_count, error = _record_pending_vote('effectiveness', counter_argument_id, score, persona_id)
        if error:
            return error
        return jsonify({'success': True, 'pending': True, 'pending_count': pending_count}), 202

    try:
        # One rating per person x answer x profile: rating again replaces the score.
        replaced = argstore.find_user_rating(
            store, counter_argument_id=counter_argument_id,
            persona_archetype_id=persona_id, user_id=current_user.id,
        ) is not None
        rating = argstore.create_rating(
            store,
            counter_argument_id=counter_argument_id,
            persona_archetype_id=persona_id,
            score=score,
            user_id=current_user.id,
        )
        argstore.save_store(ARGUMENTATION_FILE, store, PERSONA_ARCHETYPES_FILE)
        scores = [int(r['score']) for r in argstore.counted_ratings(store)
                  if r['counter_argument_id'] == counter_argument_id]
        return jsonify({
            'success': True,
            'replaced': replaced,
            'rating': rating,
            'my_score': rating['score'],
            'rating_count': len(scores),
            'avg_score': (sum(scores) / len(scores)) if scores else None,
        }), (200 if replaced else 201)
    except ValueError as e:
        return api_error(_value_error_code(e), str(e), 400)
    except Exception as e:
        print(f"Error creating rating: {e}")
        return api_error('server_error', 'Failed to create rating', 500)

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
@require_admin
def get_admin_resource_library():
    """Every resource with its review state — the admin's review queue."""
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
@require_admin
def review_resource():
    """Approve or reject a resource, optionally correcting its categorisation."""
    data = request.get_json() or {}
    url = (data.get('url') or '').strip()
    status = (data.get('status') or '').strip().lower()
    topics = data.get('topics')

    if not url:
        return api_error('url_required', 'A resource URL is required', 400)
    if topics is not None and not isinstance(topics, list):
        return api_error('topics_not_list', 'topics must be a list', 400)

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
        return api_error(_value_error_code(e), str(e), 400)
    except Exception as e:
        print(f"Error reviewing resource: {e}")
        return api_error('server_error', 'Failed to save review', 500)

    key = library.entry_key(url)
    updated = next((e for e in _build_resource_library() if e['id'] == library.entry_id(key)), None)
    return jsonify({'success': True, 'resource': updated})


@app.route('/api/importance-ratings', methods=['POST'])
def create_importance_rating_v1():
    """Rate how urgent a claim is to debunk (1..5).

    One standing vote per person per claim — re-rating replaces the old score.
    """
    data = request.get_json() or {}
    argument_id = data.get('argument_id')
    score = data.get('score')

    if not isinstance(argument_id, int):
        return api_error('invalid_argument_id', 'argument_id must be an integer', 400)
    if not isinstance(score, int):
        return api_error('invalid_score', 'score must be an integer (1..5)', 400)

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, _personas_by_id = argstore.build_indexes(store)
    if not argstore.is_published(args_by_id.get(argument_id)):
        return api_error('claim_not_found', 'Argument not found', 404)
    if not 1 <= score <= 5:
        return api_error('invalid_score', 'Score must be an integer between 1 and 5', 400)

    if not current_user.is_authenticated:
        pending_count, error = _record_pending_vote('importance', argument_id, score)
        if error:
            return error
        # Public tally is unchanged: pending votes don't count until claimed.
        scores = [int(ir['score']) for ir in argstore.importance_by_argument(store).get(argument_id, [])]
        return jsonify({
            'success': True,
            'pending': True,
            'pending_count': pending_count,
            'importance': {
                'avg_score': (sum(scores) / len(scores)) if scores else None,
                'count': len(scores),
                'my_score': score,
            },
        }), 202

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
        return api_error(_value_error_code(e), str(e), 400)
    except Exception as e:
        print(f"Error creating importance rating: {e}")
        return api_error('server_error', 'Failed to save importance rating', 500)


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
        return api_error('url_required', 'A resource URL is required', 400)
    if not argstore.is_http_url(url):
        return api_error('invalid_url', argstore.URL_ERROR, 400)
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        return api_error('tags_not_list', 'tags must be a list', 400)
    if argument_id is not None and not isinstance(argument_id, int):
        return api_error('invalid_argument_id', 'argument_id must be an integer or null', 400)

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    args_by_id, _personas_by_id = argstore.build_indexes(store)
    if argument_id is not None and argument_id not in args_by_id:
        return api_error('claim_not_found', 'Argument not found', 404)

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
        return api_error(_value_error_code(e), str(e), 400)
    except Exception as e:
        print(f"Error creating resource: {e}")
        return api_error('server_error', 'Failed to save resource', 500)


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
        return api_error('unknown_role', 'Unknown role', 400)
    if not db.add_user_role(current_user.id, role):
        return api_error('role_already_held', 'You already hold this role', 409)

    store = argstore.load_store(ARGUMENTATION_FILE, PERSONA_ARCHETYPES_FILE)
    return jsonify({'success': True, 'roles': roleslib.describe_roles(store, current_user.id)}), 201


@app.route('/api/me/roles/<role>', methods=['DELETE'])
@login_required
def drop_role(role):
    """Step down from a role. Your past work still counts if you take it again."""
    if not db.remove_user_role(current_user.id, role):
        return api_error('role_not_held', 'You don’t hold this role', 404)
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
    ca = next((c for c in argstore.published(store.get('counter_arguments'))
               if c.get('id') == counter_argument_id), None)
    if not ca:
        return api_error('answer_not_found', 'Counter-argument not found', 404)

    allowed, refusal = roleslib.can_check_source(
        current_user.id, current_user.role, ca, set(db.get_role_tiers('sourcer')))
    if not allowed:
        return api_error(refusal, roleslib.CHECK_REFUSALS.get(refusal, 'Not allowed'), 403)

    if not db.check_rate_limit(current_user.id, request.remote_addr, 'source_check', 50):
        return api_error('rate_limited', 'Rate limit exceeded. Maximum 50 source checks per day.', 429)

    try:
        argstore.set_source_check(store, counter_argument_id=counter_argument_id,
                                  verdict=verdict, user_id=current_user.id, note=note)
    except ValueError as e:
        return api_error(_value_error_code(e), str(e), 400)

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
    if not os.path.exists(ARGUMENTATION_FILE):
        argstore.save_store(ARGUMENTATION_FILE, argstore.default_store(), PERSONA_ARCHETYPES_FILE)
    
    # Production-safety warnings: never ship with the committed defaults.
    if app.secret_key == 'dev-secret-key-change-in-production':
        if ENV == 'production':
            raise SystemExit('Refusing to start with ENV=production and the default SECRET_KEY.')
        print('WARNING: using the default SECRET_KEY. Set SECRET_KEY in the environment before deploying.')
    if ADMIN_PASSWORD == 'change_me_in_production':
        print('WARNING: using the default ADMIN_PASSWORD. Set ADMIN_PASSWORD in the environment before deploying.')

    # Debug mode is opt-in via env, so a production launch can't accidentally
    # expose the interactive debugger / code execution surface.
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')

    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=debug)
