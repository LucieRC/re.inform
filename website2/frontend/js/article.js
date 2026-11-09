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
            // Only web and on-site links (no javascript: and the like).
            const safeUrl = safeHref(url.replace(/"/g, '%22'));
            if (!safeUrl) return safeLabel;
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

    // Table separator line, e.g. "| --- | :---: |".
    const TABLE_SEPARATOR = /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$/;

    function tableCells(line) {
        return line.trim().replace(/^\|/, '').replace(/\|$/, '')
            .split(/(?<!\\)\|/)
            .map((cell) => cell.trim().replace(/\\\|/g, '|'));
    }

    function tableToHtml(headerLine, separatorLine, rowLines) {
        const aligns = tableCells(separatorLine).map((cell) => {
            const left = cell.startsWith(':');
            const right = cell.endsWith(':');
            if (left && right) return ' style="text-align:center"';
            if (right) return ' style="text-align:right"';
            return '';
        });
        const headers = tableCells(headerLine);
        const cellHtml = (tag, text, col) => `<${tag}${aligns[col] || ''}>${inlineMarkdownToHtml(text)}</${tag}>`;
        const head = headers.map((text, col) => cellHtml('th', text, col)).join('');
        const body = rowLines.map((line) => {
            const cells = tableCells(line);
            return `<tr>${headers.map((_, col) => cellHtml('td', cells[col] || '', col)).join('')}</tr>`;
        }).join('');
        return `<div class="table-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function markdownToHtml(markdown) {
        const raw = String(markdown || '');
        const escaped = escapeHtml(raw);
        const lines = escaped.split(/\r?\n/);

        let html = '';
        let inUl = false;
        let inOl = false;
        // Quotes: consecutive "> …" lines make one <blockquote>; a bare ">" splits paragraphs.
        let quote = null;

        function closeQuote() {
            if (!quote) return;
            const paras = quote.join('\n').split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
            html += `<blockquote>${paras.map((p) => `<p>${inlineMarkdownToHtml(p.replace(/\n/g, ' '))}</p>`).join('')}</blockquote>`;
            quote = null;
        }

        function closeLists() {
            closeQuote();
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

            // Blockquote (the text is already escaped, so ">" is "&gt;")
            const q = /^&gt;\s?(.*)$/.exec(trimmed);
            if (q) {
                if (!quote) {
                    closeLists();
                    quote = [];
                }
                quote.push(q[1]);
                continue;
            }
            closeQuote();

            // Table: a line with "|" followed by a separator line
            const nextLine = (lines[i + 1] || '').trim();
            if (trimmed.includes('|') && nextLine.includes('|') && TABLE_SEPARATOR.test(nextLine)) {
                closeLists();
                const rows = [];
                let j = i + 2;
                while (j < lines.length && lines[j].trim() && lines[j].includes('|')) {
                    rows.push(lines[j]);
                    j++;
                }
                html += tableToHtml(trimmed, nextLine, rows);
                i = j - 1;
                continue;
            }

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
            errorEl.textContent = t('article.invalid');
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
                document.title = t('site.pageTitle', { page: firstHeading });
            }

            // The methodology is one of our own pages: no raw-text link, and back goes to Profils.
            if (requestedSrc === METHODOLOGY_SRC) {
                const back = document.querySelector('.article-breadcrumb a');
                back.href = `personalities.html${METHODOLOGY_HASH}`;
                back.removeAttribute('data-i18n');
                back.textContent = t('article.backMethodology');
            } else {
                metaEl.innerHTML = t('article.sourceHtml', { href: escapeHtml(requestedSrc) });
                metaEl.style.display = 'block';
            }

            const markdownForBody = firstHeading ? stripFirstH1(markdown) : markdown;
            bodyEl.innerHTML = markdownToHtml(markdownForBody);

            loadingEl.style.display = 'none';
            bodyEl.style.display = 'block';
        } catch (e) {
            loadingEl.style.display = 'none';
            errorEl.textContent = t('article.loadError');
            errorEl.style.display = 'block';
            console.error(e);
        }
    }

    // Also used by the methodology section of the Profils page.
    window.markdownToHtml = markdownToHtml;

    document.addEventListener('DOMContentLoaded', () => {
        if (document.getElementById('articleBody')) initArticlePage();
    });
})();
