# Plan: Rebuild Simplified re.inform Website

Build a minimal 3-page website focused on contributions and counter-argument exploration, using the existing green (#29683e) and gold (#d6b969) color scheme. Strip away translation system, animations, complex forms, and unnecessary features to create a clean foundation that can grow organically.

## Steps

1. **Create new simplified homepage** (`index.html`) with brief project intro, single Discord CTA, and navigation to two core sections—eliminate hero animations, multiple CTAs, featured resources grid, and methodology visualizations.

2. **Build contribution-focused page** (`contribute.html`) as the centerpiece with streamlined counter-narrative submission form directly on page (no modal), basic fields (narrative description, counter-argument, optional source), and simple success/error messaging—remove contact forms, project needs sections, and multiple contribution types.

3. **Create counter-arguments exploration page** (`explore.html`) with simple filterable/sortable list of top-ranked counter-arguments displayed as cards (title, summary, effectiveness rating, tags)—use static data structure in JavaScript for now, no complex filtering or backend integration.

4. **Simplify CSS and remove JavaScript complexity** by consolidating all styles into single `styles.css`, removing translation system, scroll animations, modal logic, and FAQ accordions—keep only mobile navigation toggle and basic form validation.

5. **Update navigation and footer** to reflect new 3-page structure (Home, Contribute, Explore), remove language selector, floating feedback button, and redundant Discord links—keep single Discord CTA in footer.

## Technical Decisions

1. **Explore page data structure**: JSON file for counter-arguments data, enabling easy growth and future database migration.

2. **Form submission handling**: Custom backend implementation with Python/Flask or Node.js/Express for form submissions, storing contributions in JSON/database, with email notifications to admins.

3. **Design approach**: Desktop-first responsive design that scales down gracefully for mobile devices.

## Backend Implementation

6. **Set up simple backend server** (Python Flask or Node.js Express) with endpoints for:
   - POST `/api/submit-contribution` - receives and stores counter-narrative submissions
   - GET `/api/counter-arguments` - returns list of approved counter-arguments
   - Email notification system for new submissions

7. **Create data storage structure** with JSON files for:
   - `counter-arguments.json` - approved counter-arguments with metadata (title, summary, effectiveness rating, tags, date)
   - `submissions.json` - pending contributions awaiting review
   - Simple file-based storage to start, easy to migrate to database later

8. **Add basic admin functionality** - simple password-protected page (`/admin`) for reviewing submissions and publishing to counter-arguments list (can be very basic HTML form).
