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

### Security and URLs
- `ENV=production` turns the dev defaults into refusals: the app won't start with the
  default `SECRET_KEY`, and admin password login is disabled while `ADMIN_PASSWORD` is the default.
- `CORS_ORIGINS` (comma-separated, e.g. `http://localhost:3000,https://reinform.example`):
  extra origins allowed to call `/api/*` with the login cookie. Empty by default: the
  frontend is served by this app, so same-origin needs no CORS. Session cookie is `SameSite=Lax`.
- `SITE_URL`: public base URL used in emails (admin review link). If unset: the request's
  host, else `http://localhost:$PORT` (default 5002).

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

- `GET /api/contributions` - All published answers in the old contributions shape (view over the store)
  - Query params: `sort` (date|title), `tag` (filter by tag)
- `POST /api/submit-contribution` - Retired: returns 410 `endpoint_moved`; use
  `POST /api/arguments` then `POST /api/counter-arguments`
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

Allowed for a signed-in account whose `users.role` is `admin`, or with the admin-page token
(`Authorization: Bearer <token>`, from the password login; expires after 12 h; 30 login
attempts per IP per day).

- `POST /api/admin/login` - Admin login
  - Body: `{ password }` → `{ token, expires_in }`
- `GET /api/admin/submissions` - Review queue: pending answers in the store, plus reviewed ones
- `POST /api/admin/approve-submission/<id>` - Publish a pending answer (`id` = counter-argument id)
  - Body: `{ tags? }` (added to the claim)
- `POST /api/admin/reject-submission/<id>` - Reject a pending answer
- `GET /api/admin/library` - Resource review queue (all states)
  - Query param: `status` (pending|approved|rejected)
- `POST /api/admin/library/review` - Approve or reject a resource
  - Body: `{ url, status, type?, topics?, title?, description?, note? }`
  - Admins can correct the inferred type and topics while reviewing

### Error and message codes

Every error response is `{ error, code, ...extra }`: `error` is English text kept for old
callers, `code` is stable and never translated — the frontend maps it to its own wording.
Success messages that used to be English-only also carry a `code`.

- Auth / accounts: `auth_required`, `forbidden_role`, `not_your_account`, `fields_required`,
  `credentials_required`, `invalid_credentials`, `username_too_short`, `password_too_short`,
  `account_exists`, `user_not_found`
- Admin: `admin_auth_required`, `invalid_admin_password`, `admin_password_not_set`,
  `submission_not_pending`, `invalid_vote_type`
- Claims / answers: `claim_not_found`, `answer_not_found`, `claim_text_required`,
  `answer_text_required`, `invalid_argument_id`, `invalid_counter_argument_id`,
  `persona_required`, `persona_not_found`, `persona_not_named`, `invalid_persona_id`,
  `tags_not_list`, `endpoint_moved`
- Votes: `invalid_score`, `anon_vote_limit`, `anon_votes_ip_limit` (both with `needs_account: true`)
- Sources / resources: `invalid_url`, `url_required`, `topics_not_list`, `invalid_status`,
  `invalid_type`, `invalid_verdict`, and the source-check refusals `check_login_required`,
  `check_sourcer_role_required`, `check_own_debunk`, `check_no_sources` (also in
  `source_check.refusal_code` on `GET /api/arguments/<id>`)
- Profile / follows / roles: `avatar_not_found`, `file_required`, `unsupported_image`,
  `image_too_large`, `tag_required`, `already_following_tag`, `not_following_tag`,
  `follow_self`, `already_following_user`, `not_following_user`, `endorse_self`,
  `already_endorsed`, `reason_required`, `nomination_fields_required`, `unknown_role`,
  `role_already_held`, `role_not_held`
- Generic: `rate_limited`, `invalid_input`, `not_found`, `method_not_allowed`,
  `payload_too_large`, `bad_request`, `server_error` (any uncoded JSON error gets `http_<status>`)
- Success codes: `registered`, `logged_in`, `logged_out`, `tag_followed`, `tag_unfollowed`,
  `user_followed`, `user_unfollowed`, `user_endorsed`, `nomination_submitted`,
  `nomination_approved`, `nomination_rejected`

Label ids already in payloads: roles `role` (`wordsmith|sourcer|field_agent`) and
`progress.unit_code` (`checks_matching_verdict|debunks_well_rated`); activity feed
`kind` + `verb_code` (`wrote_response|shared_source`); library `type` and facet `value`
(`article|study|report|infographic|video|podcast|dataset|website`), `review.status`.

## Resource library

`GET /api/library` folds every source of external resources into one catalogue,
keyed by normalized URL, so the same link referenced from several places is one
entry (see `resource_library.py`):

- `data/resources.json` (curated), `data/gallery.json` + `data/bonpote_gallery.json`
  (gallery seeds), and the argumentation store's resources.
- **Only reviewed resources are public.** An entry qualifies once an admin reviews
  it, or by *provenance*: curated lists, repo-committed seeds, importer output, and
  sources carried by already-published answers are trusted by where they came
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
- `argumentation.json` - the single source of truth for claims and answers: arguments,
  counter-arguments, ratings, resources (via `argumentation_store.py`). Claims and answers
  may carry `status` (`pending` / `rejected`; missing = published); only published ones are
  public, pending ones form the admin review queue (`/api/admin/submissions`).
  The old `contributions.json` / `submissions.json` were folded in by
  `bin/migrate_legacy_contributions.py` and moved to `archive/website2-legacy-data/`.
- `persona_archetypes.json` - Persona archetypes sidecar file
- `resource_reviews.json` - Admin review decisions for the resource library, keyed by
  normalized URL. It is an *overlay*: source files are never rewritten, so re-running
  an importer cannot undo a review.

## Import: Science Feedback (19 narratifs)

If you have scraped the Science Feedback page into `data/science_feedback_19_narratifs.json`, you can import it into:
- `data/argumentation.json` (arguments + counter-arguments)

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
