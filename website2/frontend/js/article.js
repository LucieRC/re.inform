// ========================================
// re.inform - Article Viewer
// ========================================

(function () {
    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function normalizeArticleSrc(src) {
        const s = String(src || '').trim();
        if (!s) return '';

        // Only allow backend-served internal articles.
        if (s.startsWith('/api/articles/') && s.toLowerCase().endsWith('.md')) {
            return s;
        }
        return '';
    }

    function getFirstHeading(markdown) {
        const text = String(markdown || '');
        const lines = text.split(/\r?\n/);
        for (const line of lines) {
            const m = /^#\s+(.+?)\s*$/.exec(line);
            if (m) return m[1];
        }
        return '';
    }

    function stripFirstH1(markdown) {
        const text = String(markdown || '');
        const lines = text.split(/\r?\n/);
        for (let i = 0; i < lines.length; i++) {
            if (/^#\s+/.test(lines[i])) {
                lines.splice(i, 1);
                // Also drop immediate blank lines following the title.
                while (lines[i] !== undefined && lines[i].trim() === '') {
                    lines.splice(i, 1);
                }
                break;
            }
        }
        return lines.join('\n');
    }

    function inlineMarkdownToHtml(text) {
        // Input is already HTML-escaped.
        let html = text;

        // Links: [text](url)
        html = html.replace(/\[([^\]]+?)\]\(([^)\s]+?)\)/g, (match, label, url) => {
            const safeLabel = label;
            const safeUrl = url.replace(/"/g, '%22');
            const isHttp = /^https?:\/\//i.test(safeUrl);
            const target = isHttp ? ' target="_blank" rel="noopener noreferrer"' : '';
            return `<a href="${safeUrl}"${target}>${safeLabel}</a>`;
        });

        // Bold: **text**
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic: *text*
        html = html.replace(/(^|[^*])\*(?!\s)(.+?)(?!\s)\*(?!\*)/g, '$1<em>$2</em>');

        // Inline code: `code`
        html = html.replace(/`([^`]+?)`/g, '<code>$1</code>');

        return html;
    }

    function markdownToHtml(markdown) {
        const raw = String(markdown || '');
        const escaped = escapeHtml(raw);
        const lines = escaped.split(/\r?\n/);

        let html = '';
        let inUl = false;
        let inOl = false;

        function closeLists() {
            if (inUl) {
                html += '</ul>';
                inUl = false;
            }
            if (inOl) {
                html += '</ol>';
                inOl = false;
            }
        }

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const trimmed = line.trim();

            // Horizontal rule
            if (/^---+$/.test(trimmed)) {
                closeLists();
                html += '<hr />';
                continue;
            }

            // Headings
            const h = /^(#{1,6})\s+(.+)$/.exec(trimmed);
            if (h) {
                closeLists();
                const level = h[1].length;
                html += `<h${level}>${inlineMarkdownToHtml(h[2])}</h${level}>`;
                continue;
            }

            // Unordered list
            const ul = /^[-*]\s+(.+)$/.exec(trimmed);
            if (ul) {
                if (inOl) {
                    html += '</ol>';
                    inOl = false;
                }
                if (!inUl) {
                    html += '<ul>';
                    inUl = true;
                }
                html += `<li>${inlineMarkdownToHtml(ul[1])}</li>`;
                continue;
            }

            // Ordered list
            const ol = /^\d+\.\s+(.+)$/.exec(trimmed);
            if (ol) {
                if (inUl) {
                    html += '</ul>';
                    inUl = false;
                }
                if (!inOl) {
                    html += '<ol>';
                    inOl = true;
                }
                html += `<li>${inlineMarkdownToHtml(ol[1])}</li>`;
                continue;
            }

            // Blank line
            if (!trimmed) {
                closeLists();
                continue;
            }

            // Paragraph
            closeLists();
            html += `<p>${inlineMarkdownToHtml(trimmed)}</p>`;
        }

        closeLists();
        return html;
    }

    async function initArticlePage() {
        const titleEl = document.getElementById('articleTitle');
        const metaEl = document.getElementById('articleMeta');
        const loadingEl = document.getElementById('articleLoading');
        const errorEl = document.getElementById('articleError');
        const bodyEl = document.getElementById('articleBody');

        const params = new URLSearchParams(window.location.search);
        const requestedSrc = normalizeArticleSrc(params.get('src'));

        if (!requestedSrc) {
            loadingEl.style.display = 'none';
            errorEl.textContent = 'Missing or invalid article source.';
            errorEl.style.display = 'block';
            return;
        }

        try {
            const response = await fetch(requestedSrc, { headers: { 'Accept': 'text/markdown' } });
            if (!response.ok) {
                throw new Error(`Failed to load article (${response.status})`);
            }

            const markdown = await response.text();
            const firstHeading = getFirstHeading(markdown);

            if (firstHeading) {
                titleEl.textContent = firstHeading;
                document.title = `${firstHeading} - re.inform`;
            }

            metaEl.innerHTML = `Source: <a href="${requestedSrc}" target="_blank" rel="noopener noreferrer">raw markdown</a>`;
            metaEl.style.display = 'block';

            const markdownForBody = firstHeading ? stripFirstH1(markdown) : markdown;
            bodyEl.innerHTML = markdownToHtml(markdownForBody);

            loadingEl.style.display = 'none';
            bodyEl.style.display = 'block';
        } catch (e) {
            loadingEl.style.display = 'none';
            errorEl.textContent = 'Failed to load article. Please try again later.';
            errorEl.style.display = 'block';
            console.error(e);
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        initArticlePage();
    });
})();
