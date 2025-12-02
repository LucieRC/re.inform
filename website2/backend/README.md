# re.inform Backend

Simple Flask backend for the re.inform website.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## Configuration

### Admin Password
Set `ADMIN_PASSWORD` in your `.env` file. Default is `change_me_in_production`.

### Email Notifications (Optional)
To enable email notifications when new contributions are submitted:
1. Set `EMAIL_ENABLED=true` in `.env`
2. Configure SMTP settings (example for Gmail):
   - `SMTP_SENDER_EMAIL`: Your Gmail address
   - `SMTP_PASSWORD`: Your Gmail app password (not your regular password)
   - `SMTP_SERVER`: smtp.gmail.com
   - `SMTP_PORT`: 587
   - `ADMIN_EMAIL`: Email where notifications should be sent

## API Endpoints

### Public Endpoints

- `GET /api/counter-arguments` - Get all approved counter-arguments
  - Query params: `sort` (date|rating|title), `tag` (filter by tag)
- `POST /api/submit-contribution` - Submit a new contribution
  - Body: `{ narrative, counter_argument, source? }`

### Admin Endpoints

- `POST /api/admin/login` - Admin login
  - Body: `{ password }`
- `GET /api/admin/submissions` - Get all submissions (requires auth)
- `POST /api/admin/approve-submission/<id>` - Approve submission
  - Body: `{ title, effectiveness_rating, tags }`
- `POST /api/admin/reject-submission/<id>` - Reject submission

## Data Storage

Data is stored in JSON files in the `data/` directory:
- `counter-arguments.json` - Approved counter-arguments
- `submissions.json` - Pending/reviewed submissions

## Deployment

For production deployment:
1. Set strong `ADMIN_PASSWORD`
2. Use environment variables for all sensitive data
3. Consider using a proper database instead of JSON files for scalability
4. Set up HTTPS
5. Configure proper CORS settings in `app.py`
