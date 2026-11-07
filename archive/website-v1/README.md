# re.inform Website

This repository contains the source code for the re.inform website, a platform aimed at countering extremism through personality-based counter-narratives.

## Project Structure

```
website/
├── index.html              # Homepage
├── manifesto.html          # Project manifesto and values
├── resources.html          # Resource library
├── contribute.html         # Ways to contribute
├── about.html              # Project information and FAQ
├── feedback.html           # User feedback form
├── css/
│   └── styles.css          # Main stylesheet
├── js/
│   └── main.js             # JavaScript functionality
└── img/                    # Image assets (to be added)
```

## Local Development

To run this website locally:

1. Clone the repository:
```
git clone [repository-url]
cd re.inform/website
```

2. Open the website in your browser:
   - Option 1: Simply open the index.html file directly in your browser
   - Option 2: Use a local server (recommended for proper functionality)

Using Python's built-in HTTP server:
```
# Python 3
python -m http.server

# Python 2
python -m SimpleHTTPServer
```

Then open `http://localhost:8000` in your browser.

## Deployment

For the prototype phase, the website can be deployed using any basic static site hosting service:

- GitHub Pages
- Netlify
- Vercel
- Amazon S3
- Any standard web hosting service

## Customization

### Adding Resources

To add new resources to the resources page:

1. Open `resources.html`
2. Find the `<div class="resource-list">` section
3. Add a new resource using the following template:

```html
<div class="resource-item [category]">
    <h3>[Resource Title]</h3>
    <p>[Resource Description]</p>
    <div class="resource-tags">
        <span class="resource-tag">[Tag 1]</span>
        <span class="resource-tag">[Tag 2]</span>
    </div>
    <a href="[resource-url]" class="read-more">View Resource</a>
</div>
```

### Changing Discord Link

To update the Discord community link:

1. Search for all instances of `<a href="https://discord.gg/fu5aveEJ" class="button discord">Join our Discord</a>` in the HTML files
2. Replace the `#` with your Discord invite URL

## Browser Compatibility

The website is designed to be compatible with:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers

## Future Development

Planned features for future phases:
- Interactive personality type explorer tool
- User accounts for personalized resources
- Integration with chatbot for real-time guidance
- Community forum beyond Discord
