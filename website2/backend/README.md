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
- `GET /api/resources` - Curated resources, plus resources submitted to the library
  - Community entries are flagged `community_submitted: true`
- `GET /api/library` - The Resources panel: every external resource an admin has reviewed
  - Query params: `type`, `topic`, `publisher`, `q`, `sort` (used|recent|title)
  - Returns `{ resources, facets: { types, topics, publishers }, total }`
  - Facet counts are computed after the text search but before the facet filters,
    so counts stay stable while the user narrows down

### Authenticated Endpoints

- `POST /api/importance-ratings` - Rate how urgent a claim is to debunk
  - Body: `{ argument_id, score }` (score 1..5)
  - One standing vote per user per claim: re-rating replaces the previous score
  - Aggregates are exposed on `GET /api/arguments` as `importance: { avg_score, count, my_score }`
- `POST /api/resources` - Submit a resource to the shared library
  - Body: `{ url, title?, description?, tags?, argument_id? }`
  - Deduped by normalized URL: re-submitting a known source enriches it (fills a
    missing title/description, unions tags and linked claims) instead of duplicating

### Admin Endpoints

- `POST /api/admin/login` - Admin login
  - Body: `{ password }`
- `GET /api/admin/submissions` - Get all submissions (requires auth)
- `POST /api/admin/approve-submission/<id>` - Approve submission
  - Body: `{ title, effectiveness_rating, tags }`
- `POST /api/admin/reject-submission/<id>` - Reject submission
- `GET /api/admin/library` - Resource review queue (all states)
  - Query param: `status` (pending|approved|rejected)
- `POST /api/admin/library/review` - Approve or reject a resource
  - Body: `{ url, status, type?, topics?, title?, description?, note? }`
  - Admins can correct the inferred type and topics while reviewing

## Resource library

`GET /api/library` folds every source of external resources into one catalogue,
keyed by normalized URL, so the same link referenced from several places is one
entry (see `resource_library.py`):

- `data/resources.json` (curated), `data/gallery.json` + `data/bonpote_gallery.json`
  (gallery seeds), `data/contributions.json` (approved submissions),
  `data/submissions.json` (pending/rejected), and the argumentation store's resources.
- **Only reviewed resources are public.** An entry qualifies once an admin reviews
  it, or by *provenance*: curated lists, repo-committed seeds, importer output, and
  sources carried by an already-approved submission are trusted by where they came
  from. That is what keeps the panel useful without a 125-item review backlog.
- The old Gallery page and its `/api/gallery` endpoint were removed: its images are
  `type=infographic` entries here, and the Resources panel opens with that type
  selected so it still serves as the visual entry point. `data/gallery.json` and
  `data/bonpote_gallery.json` remain, as seeds for those entries.
- Type (article/study/report/infographic/video/podcast/dataset/website) is inferred
  from the URL — file extension first, then host — and correctable by an admin.
  Inference deliberately ignores title/slug keywords: an article *about*
  infographics is not an infographic. Topics come from tags plus the claims a
  resource is linked to, with provenance tags dropped and FR/EN synonyms folded.

## Data Storage

Data is stored in JSON files in the `data/` directory:
- `submissions.json` - Pending/reviewed submissions

Additional structured data used by the newer argumentation workflow:
- `argumentation.json` - Arguments + counter-arguments + ratings (via `argumentation_store.py`)
- `persona_archetypes.json` - Persona archetypes sidecar file
- `resource_reviews.json` - Admin review decisions for the resource library, keyed by
  normalized URL. It is an *overlay*: source files are never rewritten, so re-running
  an importer cannot undo a review.

## Import: Science Feedback (19 narratifs)

If you have scraped the Science Feedback page into `data/science_feedback_19_narratifs.json`, you can import it into:
- `data/argumentation.json` (arguments + counter-arguments)
- `data/contributions.json` (legacy community feed)

Run (from `website2/backend`):

```bash
python bin/import_science_feedback_narratifs.py --dry-run
python bin/import_science_feedback_narratifs.py
```

By default, it uses the `resume` field as the counter-argument text when available. To store the full debunk text instead:

```bash
python bin/import_science_feedback_narratifs.py --use-full-debunk
```

## Deployment

For production deployment:
1. Set strong `ADMIN_PASSWORD`
2. Use environment variables for all sensitive data
3. Consider using a proper database instead of JSON files for scalability
4. Set up HTTPS
5. Configure proper CORS settings in `app.py`
