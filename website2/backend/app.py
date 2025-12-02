from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Paths to data files
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESOURCES_FILE = os.path.join(DATA_DIR, 'resources.json')
CONTRIBUTIONS_FILE = os.path.join(DATA_DIR, 'contributions.json')
SUBMISSIONS_FILE = os.path.join(DATA_DIR, 'submissions.json')

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Admin password (in production, use environment variables)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'change_me_in_production')

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

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory(app.static_folder, path)

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

@app.route('/api/submit-contribution', methods=['POST'])
def submit_contribution():
    """Submit a new counter-narrative contribution."""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('narrative') or not data.get('counter_argument'):
            return jsonify({'error': 'Narrative and counter-argument are required'}), 400
        
        # Load existing submissions
        submissions = load_json_file(SUBMISSIONS_FILE, [])
        
        # Create new submission
        submission = {
            'id': len(submissions) + 1,
            'narrative': data['narrative'].strip(),
            'counter_argument': data['counter_argument'].strip(),
            'source': data.get('source', '').strip(),
            'tags': data.get('tags', []),
            'submitted_at': datetime.now().isoformat(),
            'status': 'pending'
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
    
    if password == ADMIN_PASSWORD:
        return jsonify({'success': True, 'token': 'admin_authenticated'})
    else:
        return jsonify({'error': 'Invalid password'}), 401

@app.route('/api/admin/submissions', methods=['GET'])
def get_submissions():
    """Get all pending submissions (requires authentication)."""
    auth = request.headers.get('Authorization', '')
    
    if auth != 'Bearer admin_authenticated':
        return jsonify({'error': 'Unauthorized'}), 401
    
    submissions = load_json_file(SUBMISSIONS_FILE, [])
    return jsonify(submissions)

@app.route('/api/admin/approve-submission/<int:submission_id>', methods=['POST'])
def approve_submission(submission_id):
    """Approve a submission and add it to counter-arguments."""
    auth = request.headers.get('Authorization', '')
    
    if auth != 'Bearer admin_authenticated':
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
    auth = request.headers.get('Authorization', '')
    
    if auth != 'Bearer admin_authenticated':
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

if __name__ == '__main__':
    # Initialize data files if they don't exist
    if not os.path.exists(RESOURCES_FILE):
        save_json_file(RESOURCES_FILE, [])
    if not os.path.exists(CONTRIBUTIONS_FILE):
        save_json_file(CONTRIBUTIONS_FILE, [])
    if not os.path.exists(SUBMISSIONS_FILE):
        save_json_file(SUBMISSIONS_FILE, [])
    
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
