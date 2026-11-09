# re.inform - Simplified Website (v2)

A clean, focused platform for building evidence-based counter-narratives to extremist rhetoric. This simplified version prioritizes contribution and exploration features with minimal complexity.

## Features

- **Simple, Clean Design**: Desktop-first responsive design using green (#29683e) and gold (#d6b969) color scheme
- **Contribution-Focused**: Easy-to-use form for submitting counter-narratives
- **Explore Counter-Arguments**: Browse, filter, and sort evidence-based counter-narratives
- **Admin Panel**: Review and approve/reject submissions
- **Backend API**: Flask server with JSON data storage
- **Email Notifications**: Optional email alerts for new submissions

## Project Structure

```
website2/
├── backend/
│   ├── app.py                  # Flask server
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example           # Environment configuration template
│   ├── README.md              # Backend documentation
│   ├── bin/test_data.py       # Flag test data; export a clean data/ for publishing
│   └── data/
│       ├── argumentation.json      # Claims, responses, resources, ratings
│       └── submissions.json        # Pending submissions
└── frontend/
    ├── index.html              # Homepage
    ├── contribute.html         # Get involved: pick a contributor profile → matching tasks
    ├── explore.html           # Browse counter-arguments
    ├── admin.html             # Admin panel
    ├── css/
    │   └── styles.css         # All styles in one file
    └── js/
        └── main.js            # All JavaScript in one file
```

## Quick Start

### 1. Backend Setup

```bash
cd website2/backend

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env with your settings

# Run the server
python app.py
```

The backend will start on `http://localhost:5000`

### 2. Frontend Setup

The frontend is served by the Flask backend. Once the backend is running, open your browser to:

- Homepage: `http://localhost:5000`
- Contribute: `http://localhost:5000/contribute.html`
- Explore: `http://localhost:5000/explore.html`
- Admin: `http://localhost:5000/admin.html`

## Data

Content data (`backend/data/`, gallery images) is not in this repository: part of it is third-party content. The site needs a local copy of it to run.

## Configuration

### Admin Password

Set your admin password in the `.env` file:

```
ADMIN_PASSWORD=your_secure_password_here
```

Default password is `change_me_in_production` (change this immediately!)

### Email Notifications

To enable email notifications when new contributions are submitted:

1. Set `EMAIL_ENABLED=true` in `.env`
2. Configure your SMTP settings:

```
EMAIL_ENABLED=true
SMTP_SENDER_EMAIL=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ADMIN_EMAIL=admin@example.com
```

**Gmail Setup**: If using Gmail, you'll need to create an [App Password](https://support.google.com/accounts/answer/185833) instead of using your regular password.

## Usage

### For Contributors

1. Visit the **Contribute** page
2. Fill in the form:
   - Describe the extremist narrative you're addressing
   - Provide your counter-argument or strategy
   - Add sources (optional but recommended)
3. Submit and await review

### For Admins

1. Visit the **Admin** page at `/admin.html`
2. Log in with your admin password
3. Review pending submissions
4. For each submission:
   - Click "Approve" to add it to the public database
   - Fill in title, effectiveness rating, and tags
   - Click "Reject" if the submission doesn't meet standards
5. Approved submissions appear immediately on the Explore page

### For Visitors

1. Visit the **Explore** page
2. Browse counter-arguments
3. Filter by topic or sort by effectiveness/date
4. Click source links to see supporting evidence

## API Endpoints

### Public Endpoints

- `GET /api/counter-arguments` - Get all approved counter-arguments
  - Query params: `sort` (date|rating|title), `tag` (filter by tag)
- `POST /api/submit-contribution` - Submit a new contribution
  - Body: `{ narrative, counter_argument, source? }`

### Admin Endpoints (Require Authentication)

- `POST /api/admin/login` - Admin login
- `GET /api/admin/submissions` - Get all submissions
- `POST /api/admin/approve-submission/<id>` - Approve submission
- `POST /api/admin/reject-submission/<id>` - Reject submission

## Data Storage

Data is stored in JSON files in `backend/data/`:

- `argumentation.json` - Claims, responses, resources and ratings
- `submissions.json` - Pending and reviewed submissions

**Note**: For production use with high traffic, consider migrating to a proper database (PostgreSQL, MongoDB, etc.)

## Deployment

### Development

```bash
cd website2/backend
python app.py
```

### Production

For production deployment:

1. **Set strong admin password** in environment variables
2. **Use environment variables** for all sensitive data (never commit `.env`)
3. **Set up HTTPS** (use a reverse proxy like Nginx)
4. **Configure CORS** properly in `app.py` for your domain
5. **Use a production WSGI server** (e.g., Gunicorn):

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

6. **Consider using a database** instead of JSON files for better performance and reliability
7. **Set up monitoring and logging**
8. **Configure email notifications** for submission alerts

### Deploy to Cloud

#### Heroku

```bash
# Add a Procfile
echo "web: gunicorn app:app" > backend/Procfile

# Add runtime
echo "python-3.11.0" > backend/runtime.txt

# Deploy
heroku create your-app-name
git subtree push --prefix website2/backend heroku main
```

#### DigitalOcean, AWS, etc.

1. Set up a VPS
2. Install Python and dependencies
3. Configure environment variables
4. Set up Nginx as reverse proxy
5. Use systemd or supervisor to manage the process

## Customization

### Colors

Edit CSS variables in `frontend/css/styles.css`:

```css
:root {
    --primary-color: #29683e;  /* Dark green */
    --secondary-color: #d6b969; /* Gold */
    /* ... other colors ... */
}
```

### Discord Link

Update the Discord link in `frontend/index.html`:

```html
<a href="https://discord.gg/your-discord-link" class="btn btn-discord">
```

### Test data

For manual testing, prefer `python bin/test_sandbox.py`: it runs the site on a
throwaway copy of `data/`, so nothing you do reaches the real files.

Test content that did reach the real data is *flagged*, not deleted, and left out
when publishing (run from `backend/`):

```bash
python bin/test_data.py report                  # what is flagged, what looks like a test
python bin/test_data.py mark-user alice         # flag a test account (and all it creates)
python bin/test_data.py mark submission 1       # flag one record with no author
python bin/test_data.py export --out ../publish-data
```

Deploy `publish-data/` as `backend/data/` on the published site. Anything a flagged
account posts later is left out automatically.

## Differences from Original Website

This simplified version removes:

- ❌ Translation system (EN/FR switching)
- ❌ Multiple resource pages and articles
- ❌ About page with timeline and FAQ
- ❌ Feedback page
- ❌ Scroll animations
- ❌ Multiple CTAs and redundant Discord buttons
- ❌ Modal systems
- ❌ Complex navigation

And focuses on:

- ✅ **Contribution form** - Easy submissions
- ✅ **Explore page** - Browse counter-arguments
- ✅ **Admin panel** - Review and approve
- ✅ **Backend API** - Functional data management
- ✅ **Clean design** - Minimal, focused UX
- ✅ **Email notifications** - Stay informed

## Future Enhancements

Potential additions as the platform grows:

- User accounts and authentication
- Comments and discussions on counter-arguments
- Upvoting/downvoting system
- Advanced search and filtering
- Export counter-arguments (PDF, CSV)
- API for third-party integrations
- Analytics dashboard
- Translation system (when needed)
- More sophisticated admin tools

## Support

For issues or questions:
- Check the backend README at `backend/README.md`
- Review the API documentation above
- Ensure environment variables are set correctly
- Check browser console for frontend errors
- Check terminal/logs for backend errors

## License

[Add your license here]

## Contributing

Contributions welcome! This is designed to be a simple starting point that can grow organically based on user needs.
