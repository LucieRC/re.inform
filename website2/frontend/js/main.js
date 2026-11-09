// ========================================
// re.inform - Main JavaScript
// ========================================

// Configuration
// The Flask backend serves both the frontend and the API, so the API always
// lives on the same origin as the page — whatever host/port that is. Deriving
// it from window.location keeps admin/login and every other call working on any
// port, instead of a hardcoded localhost:5002 that broke off-port setups.
const API_BASE_URL = /^https?:$/.test(window.location.protocol)
    ? `${window.location.origin}/api`
    : 'http://localhost:5002/api'; // fallback for pages opened via file://

// Current user state
let currentUser = null;
// Votes this visitor cast before having an account (they count once they sign up).
let pendingVotes = 0;

// ========================================
// Icons: monochrome line icons (inline SVG, currentColor) used instead of emoji
// so they follow the text color in both themes. icon('name') → decorative <svg>.
// ========================================

const ICON_PATHS = {
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    message: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z"/>',
    flask: '<path d="M9 3h6M10 3v6l-5.6 9.6A1.6 1.6 0 0 0 5.8 21h12.4a1.6 1.6 0 0 0 1.4-2.4L14 9V3M7 15h10"/>',
    pen: '<path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    palette: '<path d="M12 3a9 9 0 0 0 0 18c1.1 0 2-.9 2-2 0-.5-.2-1-.5-1.3-.3-.4-.5-.8-.5-1.3 0-1.1.9-2 2-2h2.4A4.6 4.6 0 0 0 21 9.8C21 6 17 3 12 3Z"/><circle cx="7.5" cy="11" r="1"/><circle cx="10" cy="7" r="1"/><circle cx="15" cy="7.5" r="1"/>',
    code: '<path d="m8 7-5 5 5 5M16 7l5 5-5 5"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    star: '<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1 6.2-5.5-2.9-5.5 2.9 1-6.2L3 9.6l6.2-.9Z"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0M16 4.5a3.5 3.5 0 0 1 0 7M18 14a6.5 6.5 0 0 1 3.5 6"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    book: '<path d="M4 19.5v-15A1.5 1.5 0 0 1 5.5 3H20v15H5.5A1.5 1.5 0 0 0 4 19.5Zm0 0A1.5 1.5 0 0 0 5.5 21H20"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
    chart: '<path d="M4 4v16h16M8 16v-4M12 16V8M16 16v-6"/>',
    trend: '<path d="m3 17 6-6 4 4 8-8M15 7h6v6"/>',
    check: '<path d="m5 12 5 5 9-10"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    ban: '<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',
    hourglass: '<path d="M6 3h12M6 21h12M7 3c0 5 10 5 10 9s-10 4-10 9M17 3c0 5-10 5-10 9s10 4 10 9"/>',
    zap: '<path d="M13 2 4 14h7l-1 8 9-12h-7Z"/>',
    layers: '<path d="m12 3 9 5-9 5-9-5Z"/><path d="m3 13 9 5 9-5"/>',
    bulb: '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3Z"/>',
    tag: '<path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9Z"/><circle cx="7.5" cy="7.5" r="1.2"/>',
    clip: '<path d="m21 11-8.5 8.5a5 5 0 0 1-7-7L14 4a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.5a1.7 1.7 0 0 1-2.4-2.4L15.5 7"/>',
    copy: '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>',
    file: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
    video: '<rect x="2" y="6" width="14" height="12" rx="2"/><path d="m16 10 6-3v10l-6-3"/>',
    headphones: '<path d="M3 18v-6a9 9 0 0 1 18 0v6"/><rect x="3" y="14" width="4" height="7" rx="1.5"/><rect x="17" y="14" width="4" height="7" rx="1.5"/>',
    globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    quote: '<path d="M10 7H6a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h3v1a3 3 0 0 1-3 3M20 7h-4a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h3v1a3 3 0 0 1-3 3M10 7v8M20 7v8"/>',
    building: '<rect x="5" y="3" width="14" height="18" rx="1"/><path d="M9 7h2M13 7h2M9 11h2M13 11h2M9 15h2M13 15h2M10 21v-3h4v3"/>',
};

function icon(name) {
    const paths = ICON_PATHS[name];
    return paths ? `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${paths}</svg>` : '';
}

function getSelectedTagsFromFilter(selectEl) {
    if (!selectEl) return [];
    return Array.from(selectEl.selectedOptions)
        .map((opt) => opt.value)
        .filter((value) => value);
}

function itemMatchesAnySelectedTag(item, selectedTags) {
    if (!selectedTags.length) return true;
    if (!item || !Array.isArray(item.tags)) return false;
    return selectedTags.some((tag) => item.tags.includes(tag));
}

function ensureMultiTagDropdown(selectEl, { placeholder = t('filter.allTopics') } = {}) {
    if (!selectEl || !(selectEl instanceof HTMLSelectElement)) return;

    // Keep the original select as the source of truth, but allow multiple selections.
    selectEl.multiple = true;

    const existingWrapper = selectEl.closest('.multi-tag-select');
    let wrapper = existingWrapper;
    let trigger;
    let menu;

    if (!wrapper) {
        wrapper = document.createElement('div');
        wrapper.className = 'multi-tag-select';

        trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'multi-tag-trigger';
        trigger.setAttribute('aria-haspopup', 'listbox');
        trigger.setAttribute('aria-expanded', 'false');

        menu = document.createElement('div');
        menu.className = 'multi-tag-menu';
        menu.setAttribute('role', 'listbox');
        menu.setAttribute('aria-multiselectable', 'true');

        // Insert wrapper in place of the select, then move the select inside.
        selectEl.parentNode.insertBefore(wrapper, selectEl);
        wrapper.appendChild(trigger);
        wrapper.appendChild(menu);
        wrapper.appendChild(selectEl);

        const closeIfClickedOutside = (e) => {
            if (!menu.classList.contains('is-open')) return;
            if (wrapper.contains(e.target)) return;
            closeMultiTagMenu(trigger, menu);
        };

        trigger.addEventListener('click', () => {
            if (menu.classList.contains('is-open')) {
                closeMultiTagMenu(trigger, menu);
            } else {
                openMultiTagMenu(trigger, menu);
            }
        });

        trigger.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeMultiTagMenu(trigger, menu);
            }
        });

        document.addEventListener('click', closeIfClickedOutside);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeMultiTagMenu(trigger, menu);
            }
        });

        // When selection changes via other code, keep the UI text in sync.
        selectEl.addEventListener('change', () => {
            updateMultiTagTriggerText(selectEl, trigger, placeholder);
            syncMultiTagMenuChecks(selectEl, menu);
        });
    } else {
        trigger = wrapper.querySelector('.multi-tag-trigger');
        menu = wrapper.querySelector('.multi-tag-menu');
    }

    rebuildMultiTagMenu(selectEl, menu);
    updateMultiTagTriggerText(selectEl, trigger, placeholder);
    syncMultiTagMenuChecks(selectEl, menu);
}

function openMultiTagMenu(trigger, menu) {
    menu.classList.add('is-open');
    trigger.setAttribute('aria-expanded', 'true');
}

function closeMultiTagMenu(trigger, menu) {
    menu.classList.remove('is-open');
    trigger.setAttribute('aria-expanded', 'false');
}

function rebuildMultiTagMenu(selectEl, menuEl) {
    if (!selectEl || !menuEl) return;
    menuEl.innerHTML = '';

    Array.from(selectEl.options)
        .filter((opt) => opt.value) // exclude the "All topics" empty option
        .forEach((opt) => {
            const label = document.createElement('label');
            label.className = 'multi-tag-option';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = opt.selected;

            const text = document.createElement('span');
            text.textContent = opt.textContent || opt.value;

            checkbox.addEventListener('change', () => {
                opt.selected = checkbox.checked;

                // Ensure the empty option isn't selected when any tag is selected.
                const emptyOpt = Array.from(selectEl.options).find((o) => !o.value);
                if (emptyOpt) emptyOpt.selected = false;

                selectEl.dispatchEvent(new Event('change', { bubbles: true }));
            });

            label.appendChild(checkbox);
            label.appendChild(text);
            menuEl.appendChild(label);
        });
}

function syncMultiTagMenuChecks(selectEl, menuEl) {
    if (!selectEl || !menuEl) return;
    const selected = new Set(getSelectedTagsFromFilter(selectEl));
    Array.from(menuEl.querySelectorAll('label.multi-tag-option')).forEach((label) => {
        const checkbox = label.querySelector('input[type="checkbox"]');
        const labelText = label.querySelector('span')?.textContent;
        if (!checkbox) return;

        // Find the option that matches by displayed text or value.
        const match = Array.from(selectEl.options).find((o) => o.value && ((o.textContent || o.value) === labelText || o.value === labelText));
        if (!match) return;
        checkbox.checked = selected.has(match.value);
    });
}

function updateMultiTagTriggerText(selectEl, triggerEl, placeholder) {
    if (!selectEl || !triggerEl) return;
    const selected = getSelectedTagsFromFilter(selectEl);
    if (!selected.length) {
        triggerEl.textContent = placeholder;
        return;
    }
    if (selected.length === 1) {
        const opt = Array.from(selectEl.options).find((o) => o.value === selected[0]);
        triggerEl.textContent = opt?.textContent || selected[0];
        return;
    }
    triggerEl.textContent = t('filter.topicsCount', { n: selected.length });
}

// ========================================
// Navigation panel (off-canvas below 1100px — see styles.css)
// ========================================

function setupNavPanel() {
    const toggle = document.querySelector('.site-menu-btn, .mobile-menu-toggle');
    const navLinks = document.querySelector('.site-links, .nav-links');
    if (!toggle || !navLinks) return;

    // One backdrop per page, created here so every page gets it without
    // repeating markup in a dozen HTML files. (The new header's panel drops
    // down under the bar and needs no backdrop styling, but closing on an
    // outside click is still handled through it.)
    let backdrop = document.querySelector('.nav-backdrop');
    if (!backdrop) {
        backdrop = document.createElement('div');
        backdrop.className = 'nav-backdrop';
        document.body.appendChild(backdrop);
    }

    toggle.setAttribute('aria-expanded', 'false');
    if (!navLinks.id) navLinks.id = 'navLinks';
    toggle.setAttribute('aria-controls', navLinks.id);

    const isOpen = () => navLinks.classList.contains('active');

    const setOpen = (open) => {
        navLinks.classList.toggle('active', open);
        backdrop.classList.toggle('active', open);
        document.body.classList.toggle('nav-panel-open', open);
        toggle.setAttribute('aria-expanded', String(open));
    };

    toggle.addEventListener('click', () => {
        const opening = !isOpen();
        setOpen(opening);
        // Move focus into the panel so keyboard users land where they opened.
        if (opening) navLinks.querySelector('a')?.focus();
    });

    backdrop.addEventListener('click', () => setOpen(false));

    // Navigating away closes the panel; so does Escape, returning focus.
    navLinks.addEventListener('click', (e) => {
        if (e.target.closest('a')) setOpen(false);
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isOpen()) {
            setOpen(false);
            toggle.focus();
        }
    });

    // Widening past the breakpoint must not leave the page scroll-locked.
    const wide = window.matchMedia(navLinks.classList.contains('site-links') ? '(min-width: 861px)' : '(min-width: 1101px)');
    const syncToWidth = () => {
        if (wide.matches && isOpen()) setOpen(false);
    };
    wide.addEventListener('change', syncToWidth);
}

// Expose the sticky header's height as --header-height so sticky bars below it
// (the filter bar) stop right under the header instead of sliding beneath it.
// The header search is narrow on phones: use the short placeholder there.
function syncSearchPlaceholder() {
    const input = document.getElementById('siteSearch');
    if (!input || !window.matchMedia) return;
    const narrow = window.matchMedia('(max-width: 600px)');
    const apply = () => { input.placeholder = t(narrow.matches ? 'nav.searchPlaceholderShort' : 'nav.searchPlaceholder'); };
    apply();
    narrow.addEventListener('change', apply);
}

function syncHeaderHeight() {
    const header = document.querySelector('body > header');
    if (!header) return;
    const apply = () => document.documentElement.style.setProperty('--header-height', `${header.offsetHeight}px`);
    apply();
    if ('ResizeObserver' in window) new ResizeObserver(apply).observe(header);
    else window.addEventListener('resize', apply);
}

document.addEventListener('DOMContentLoaded', async () => {
    setupNavPanel();
    syncHeaderHeight();
    syncSearchPlaceholder();

    // Check authentication state first
    await checkAuthState();
    
    // The « page introuvable » page has nothing to load (and its address can be anything).
    if (document.body.dataset.page === '404') return;

    // Initialize page-specific functionality
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    
    if (currentPage.includes('argument')) {
        initArgumentDetailPage();
    } else if (currentPage.includes('respond')) {
        initRespondPage();
    } else if (currentPage.includes('contribute') && !currentPage.includes('contributions')) {
        initContributePage();
    } else if (currentPage.includes('library')) {
        initLibraryPage();
    } else if (currentPage.includes('resources')) {
        initBlogArticlesPage();
    } else if (currentPage.includes('personalities')) {
        initPersonalitiesPage();
    } else if (currentPage.includes('personality')) {
        initPersonalityDetailPage();
    } else if (currentPage.includes('admin')) {
        initAdminPage();
    } else if (currentPage.includes('login')) {
        initLoginPage();
    } else if (currentPage.includes('register')) {
        initRegisterPage();
    } else if (currentPage.includes('profile')) {
        initProfilePage();
    } else if (currentPage === '' || currentPage.includes('index') || currentPage.includes('home')) {
        // Home is the claims feed — the site's centerpiece.
        setupFeedComposer();
        initRespondPage();
    }
});

// ========================================
// Authentication Functions
// ========================================

async function checkAuthState() {
    try {
        const response = await fetch(`${API_BASE_URL}/current-user`, {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (data.authenticated) {
            currentUser = data.user;
            updateNavForLoggedInUser();
        } else {
            currentUser = null;
            pendingVotes = data.pending_votes || 0;
            updateNavForLoggedOutUser();
        }
    } catch (error) {
        console.error('Error checking auth state:', error);
        currentUser = null;
        updateNavForLoggedOutUser();
    }
}

function setAuthLinks(html) {
    document.querySelectorAll('[data-auth-links], #authLinks').forEach((el) => { el.innerHTML = html; });
}

function updateNavForLoggedInUser() {
    const name = currentUser?.username || t('nav.profileFallback');
    const usernameEscaped = escapeHtml(name);
    const initial = escapeHtml(name.trim().charAt(0).toUpperCase() || '?');
    // Real profile picture when one is uploaded; otherwise the first letter.
    const avatar = safeHref(currentUser?.avatar_url)
        ? `<img src="${escapeHtml(safeHref(currentUser.avatar_url))}" alt="" onerror="this.remove()">`
        : '';

    setAuthLinks(`
        <a href="#" onclick="logout(); return false;">${escapeHtml(t('nav.logout'))}</a>
        <a href="profile.html" class="site-avatar" aria-label="${escapeHtml(t('nav.profile', { name }))}" title="${usernameEscaped}">${initial}${avatar}</a>
    `);
}

function updateNavForLoggedOutUser() {
    setAuthLinks(`<a href="${authHref()}">${escapeHtml(t('nav.login'))}</a>`);
}

async function logout() {
    try {
        const response = await fetch(`${API_BASE_URL}/logout`, {
            method: 'POST',
            credentials: 'include'
        });
        
        if (response.ok) {
            currentUser = null;
            window.location.href = 'index.html';
        }
    } catch (error) {
        console.error('Logout error:', error);
    }
}

// ========================================
// Login Page
// ========================================

// Votes cast without an account are kept until the visitor signs up.
function pendingVotesHtml(count) {
    const n = Math.max(1, count || 1);
    return t('votes.pendingHtml', { n, href: authHref('register.html') });
}

function claimedVotesText(count) {
    return count ? t('votes.claimed', { n: count }) : '';
}

// After login/register, return to the page that asked for an account.
// Only same-site relative pages are accepted, so `next` can't redirect off-site.
function postAuthDestination() {
    const next = new URLSearchParams(window.location.search).get('next') || '';
    return /^[a-z0-9_-]+\.html([?#][^\s]*)?$/i.test(next) ? next : 'profile.html';
}

// login.html / register.html link that brings the user back to this page.
function authHref(page = 'login.html') {
    const here = window.location.pathname.split('/').pop() || 'index.html';
    if (/^(login|register)\.html$/.test(here)) return page;
    return `${page}?next=${encodeURIComponent(here + window.location.search + window.location.hash)}`;
}

// Same, but the return address gets some query params changed, e.g. { write: 3 }
// so the answer form opens again after logging in.
function authHrefWith(page, params = {}) {
    const url = new URL(window.location.href);
    Object.entries(params).forEach(([k, v]) => {
        if (v == null || v === '') url.searchParams.delete(k); else url.searchParams.set(k, String(v));
    });
    const here = url.pathname.split('/').pop() || 'index.html';
    return `${page}?next=${encodeURIComponent(here + url.search + url.hash)}`;
}

// « Se connecter » and « Créer un compte » toast links, both returning here.
function authToastActions(params = {}) {
    return [
        { label: t('common.login'), href: authHrefWith('login.html', params) },
        { label: t('common.createAccount'), href: authHrefWith('register.html', params) },
    ];
}

// The session ended (API 401 auth_required): go to login, then come back here.
function goToLogin() {
    window.location.href = authHref('login.html');
}

// An API error as a toast, with a way out when the cause is "log in" or "create an account".
function apiErrorToast(data, fallbackKey, res = null) {
    let action = {};
    if (isAuthRequired(res, data) || data?.code === 'auth_required') {
        action = { actionLabel: t('common.login'), actionHref: authHref('login.html'), duration: 8000 };
    } else if (data?.needs_account) {
        action = { actionLabel: t('common.createAccount'), actionHref: authHref('register.html'), duration: 8000 };
    }
    showToast(apiError(data, fallbackKey), action);
}

function initLoginPage() {
    const form = document.getElementById('loginForm');
    if (!form) return;

    // Keep the return destination when switching between login and register.
    document.querySelectorAll('a[href="register.html"]').forEach((a) => { a.href = `register.html${window.location.search}`; });

    if (pendingVotes) {
        const note = document.getElementById('pendingVotesNote') || document.getElementById('formMessage');
        if (note) {
            note.hidden = false;
            note.textContent = t('votes.waitingLogin', { n: pendingVotes });
            note.className = 'form-message success';
        }
    }
    
    // Redirect if already logged in
    if (currentUser) {
        window.location.href = postAuthDestination();
        return;
    }
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value;
        const remember = document.getElementById('remember').checked;
        const formMessage = document.getElementById('formMessage');
        const loginBtn = document.getElementById('loginBtn');

        // Each attempt starts clean (no message left over from the last one).
        formMessage.textContent = '';
        formMessage.className = 'form-message';
        if (!username || !password) {
            formMessage.textContent = t('error.credentials_required');
            formMessage.className = 'form-message error';
            return;
        }

        loginBtn.disabled = true;
        loginBtn.textContent = t('common.loggingIn');
        
        try {
            const response = await fetch(`${API_BASE_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, password, remember })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                formMessage.textContent = t('login.success', { claimed: claimedVotesText(data.claimed_votes) });
                formMessage.className = 'form-message success';
                setTimeout(() => {
                    window.location.href = postAuthDestination();
                }, data.claimed_votes ? 1800 : 1000);
            } else {
                formMessage.textContent = apiError(data, 'login.failed');
                formMessage.className = 'form-message error';
            }
        } catch (error) {
            console.error('Login error:', error);
            formMessage.textContent = t('common.networkError');
            formMessage.className = 'form-message error';
        } finally {
            loginBtn.disabled = false;
            loginBtn.textContent = t('common.login');
        }
    });
}

// ========================================
// Register Page
// ========================================

function initRegisterPage() {
    const form = document.getElementById('registerForm');
    if (!form) return;

    // Keep the return destination when switching between login and register.
    document.querySelectorAll('a[href="login.html"]').forEach((a) => { a.href = `login.html${window.location.search}`; });

    if (pendingVotes) {
        const note = document.getElementById('pendingVotesNote') || document.getElementById('formMessage');
        if (note) {
            note.hidden = false;
            note.textContent = t('votes.waitingRegister', { n: pendingVotes });
            note.className = 'form-message success';
        }
    }
    
    // Redirect if already logged in
    if (currentUser) {
        window.location.href = postAuthDestination();
        return;
    }
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('username').value.trim();
        const email = document.getElementById('email').value.trim();
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const formMessage = document.getElementById('formMessage');
        const registerBtn = document.getElementById('registerBtn');

        // Each attempt starts clean; then the same checks as the server, in French.
        formMessage.textContent = '';
        formMessage.className = 'form-message';
        const problem = !username || !email || !password ? 'error.fields_required'
            : username.length < 3 ? 'error.username_too_short'
            : username.length > 30 ? 'error.username_too_long'
            : !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) ? 'register.invalidEmail'
            : password.length < 6 ? 'error.password_too_short'
            : password !== confirmPassword ? 'register.mismatch'
            : null;
        if (problem) {
            formMessage.textContent = t(problem);
            formMessage.className = 'form-message error';
            return;
        }
        
        registerBtn.disabled = true;
        registerBtn.textContent = t('register.submitting');
        
        try {
            const response = await fetch(`${API_BASE_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, email, password })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                formMessage.textContent = t('register.success', { claimed: claimedVotesText(data.claimed_votes) });
                formMessage.className = 'form-message success';
                setTimeout(() => {
                    window.location.href = postAuthDestination();
                }, data.claimed_votes ? 1800 : 1000);
            } else {
                formMessage.textContent = apiError(data, 'register.failed');
                formMessage.className = 'form-message error';
            }
        } catch (error) {
            console.error('Registration error:', error);
            formMessage.textContent = t('common.networkError');
            formMessage.className = 'form-message error';
        } finally {
            registerBtn.disabled = false;
            registerBtn.textContent = t('register.submit');
        }
    });
}

// ========================================
// Profile Page
// ========================================

// Profile page state
let profileUserId = null;
let isOwnProfile = false;
let isFollowingProfile = false;

async function initProfilePage() {
    // Get profile ID from URL or use current user
    const urlParams = new URLSearchParams(window.location.search);
    const viewUserId = urlParams.get('id');
    
    if (viewUserId) {
        // Viewing someone else's profile (or own via direct link)
        profileUserId = parseInt(viewUserId);
        isOwnProfile = currentUser && currentUser.id === profileUserId;
    } else {
        // No ID in URL - must be logged in to view own profile
        if (!currentUser) {
            goToLogin();
            return;
        }
        profileUserId = currentUser.id;
        isOwnProfile = true;
    }
    
    setupProfileControls();

    // Load profile data (an unknown member stops here: only a "not found" message).
    if (!(await loadProfileData())) return;

    // If own profile, load following sections
    if (isOwnProfile) {
        document.getElementById('levelBar').style.display = 'block';
        document.getElementById('recommendationsSection').style.display = 'block';
        document.getElementById('followingActivitySection').style.display = 'block';
        document.getElementById('statusTrackerSection').style.display = 'block';
        document.getElementById('tagsFollowingSection').style.display = 'block';
        document.getElementById('followingSection').style.display = 'block';
        document.getElementById('contributionsTitle').textContent = t('profile.contributionsMine');
        document.getElementById('rolesSection').style.display = 'block';
        const helpSection = document.getElementById('contributeSection');
        if (helpSection) {
            helpSection.style.display = 'block';
            initContributePage();
        }
        await Promise.all([
            loadMyRoles(),
            loadRecommendations(),
            loadFollowingActivity(),
            loadFollowedTags(),
            loadFollowingProfiles(),
            loadContributionStatus(),
            loadDrafts(),
        ]);
    } else {
        document.getElementById('contributionsTitle').textContent = t('profile.contributionsTitle');
        // Show follow button if logged in and viewing someone else
        if (currentUser) {
            document.getElementById('followButtonContainer').style.display = 'block';
        }
    }
}

// Wire up the profile page's buttons (no inline handlers, so keyboard and focus
// behave and the markup stays clean).
function setupProfileControls() {
    document.getElementById('followButton')?.addEventListener('click', toggleFollow);
    setupAvatarControls();
}

// ---- Level bar (reputation → progression, per the gamification design) ----
const LEVELS = [
    { key: 'level.observer', min: 0 },
    { key: 'level.engager', min: 51 },
    { key: 'level.strategist', min: 201 },
    { key: 'level.researcher', min: 501 },
    { key: 'level.leader', min: 1001 },
].map((l) => ({ ...l, name: t(l.key) }));

function renderLevelBar(points) {
    const bar = document.getElementById('levelBar');
    if (!bar) return;

    const p = Number(points) || 0;
    let idx = 0;
    for (let i = 0; i < LEVELS.length; i++) if (p >= LEVELS[i].min) idx = i;
    const current = LEVELS[idx];
    const next = LEVELS[idx + 1] || null;

    document.getElementById('levelName').textContent = current.name;
    const nextEl = document.getElementById('levelNext');
    const fill = document.getElementById('levelFill');
    const track = document.getElementById('levelProgress');

    let pct;
    if (next) {
        pct = Math.round(((p - current.min) / (next.min - current.min)) * 100);
        nextEl.textContent = t('level.next', { n: next.min - p, level: next.name });
    } else {
        pct = 100;
        nextEl.textContent = t('level.top');
    }
    pct = Math.max(0, Math.min(100, pct));
    fill.style.width = `${pct}%`;
    track.setAttribute('aria-valuenow', String(pct));
    track.setAttribute('aria-valuetext', next ? t('level.valueText', { level: current.name, pct, next: next.name }) : current.name);
}

// ---- Contributor roles (function, not rights: see docs/design/GAMIFICATION_DESIGN.md) ----

// Role names and texts come from the API in English; the catalog has them by role id.
function roleText(roleId, field, fallback) {
    return tOr(`role.${roleId}.${field}`, fallback);
}

// users.role (rights): member, trusted, moderator, admin.
function userRoleLabel(role) {
    return tOr(`userRole.${role}`, role);
}

function renderRoleChips(roles) {
    const el = document.getElementById('profileRoles');
    if (!el) return;
    el.innerHTML = roles.map((r) => {
        const confirmed = r.tier === 'confirmed';
        const name = roleText(r.role, 'name', r.name);
        const title = t('role.chipTitle', { name, tier: t(confirmed ? 'role.tier.confirmed' : 'role.tier.apprentice') });
        return `<span class="role-chip${confirmed ? ' is-confirmed' : ''}" title="${escapeHtml(title)}">
            ${icon(r.icon)}${escapeHtml(name)}${confirmed ? ` ${icon('check')}` : ''}</span>`;
    }).join('');
}

async function loadMyRoles() {
    const container = document.getElementById('rolesContent');
    if (!container) return;
    try {
        const [catalogRes, mineRes] = await Promise.all([
            fetch(`${API_BASE_URL}/roles`),
            fetch(`${API_BASE_URL}/me/roles`, { credentials: 'include' }),
        ]);
        if (mineRes.status === 401) return goToLogin();
        if (!catalogRes.ok || !mineRes.ok) throw new Error('Failed to load roles');
        const catalog = await catalogRes.json();
        const mine = (await mineRes.json()).roles || [];
        renderMyRoles(catalog, mine);

        const isSourcer = mine.some((r) => r.role === 'sourcer');
        const canModerate = ['moderator', 'admin'].includes(currentUser?.role);
        document.getElementById('sourceQueueSection').style.display = (isSourcer || canModerate) ? 'block' : 'none';
        if (isSourcer || canModerate) loadSourceQueue();
    } catch (error) {
        console.error('Roles error:', error);
        container.innerHTML = `<p class="empty-message">${escapeHtml(t('role.loadError'))}</p>`;
    }
}

function renderMyRoles(catalog, mine) {
    const container = document.getElementById('rolesContent');
    const held = new Map(mine.map((r) => [r.role, r]));

    container.innerHTML = catalog.map((c) => {
        const r = held.get(c.role);
        const name = roleText(c.role, 'name', c.name);
        const duty = roleText(c.role, 'duty', c.duty);
        const power = roleText(c.role, 'power', c.power);
        let body;
        if (!r) {
            body = `<p>${escapeHtml(duty)}</p>`;
        } else if (r.tier === 'confirmed') {
            body = `<p>${escapeHtml(power)}</p>`;
        } else if (r.progress) {
            const pct = Math.min(100, Math.round((r.progress.done / r.progress.needed) * 100));
            body = `<p>${escapeHtml(duty)}</p>
                <div class="role-progress">
                    <div class="level-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"
                         aria-label="${escapeHtml(t('role.progressLabel', { name }))}">
                        <span class="level-fill" style="width: ${pct}%"></span>
                    </div>
                    <span class="role-progress-label">${r.progress.done} / ${r.progress.needed} ${escapeHtml(roleText(c.role, 'unit', r.progress.unit))}</span>
                </div>`;
        } else {
            body = `<p>${escapeHtml(power)}</p>`;
        }
        const privacy = r && !c.public ? `<p>${escapeHtml(t('role.private'))}</p>` : '';
        const tier = r ? `<span class="role-tier${r.tier === 'confirmed' ? ' tier-confirmed' : ''}">${escapeHtml(t(r.tier === 'confirmed' ? 'role.tier.confirmed' : 'role.tier.apprentice'))}</span>` : '';
        const action = r
            ? `<button type="button" class="link-btn" data-role-drop="${c.role}">${escapeHtml(t('role.stepDown'))}</button>`
            : `<button type="button" class="btn btn-secondary btn-small" data-role-take="${c.role}">${escapeHtml(t('role.take'))}</button>`;
        return `
            <div class="role-card${r ? ' is-held' : ''}">
                <div class="role-card-head">
                    ${icon(c.icon)}
                    <span class="role-card-name">${escapeHtml(name)}</span>
                    ${tier}
                </div>
                ${body}
                ${privacy}
                <div class="role-card-actions">${action}</div>
            </div>`;
    }).join('');

    container.querySelectorAll('[data-role-take]').forEach((btn) => {
        btn.addEventListener('click', () => changeRole(btn.dataset.roleTake, 'POST'));
    });
    container.querySelectorAll('[data-role-drop]').forEach((btn) => {
        btn.addEventListener('click', () => changeRole(btn.dataset.roleDrop, 'DELETE'));
    });
}

async function changeRole(role, method) {
    try {
        const res = await fetch(
            method === 'POST' ? `${API_BASE_URL}/me/roles` : `${API_BASE_URL}/me/roles/${encodeURIComponent(role)}`,
            {
                method,
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: method === 'POST' ? JSON.stringify({ role }) : undefined,
            }
        );
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            apiErrorToast(data, 'role.error', res);
            return;
        }
        showToast(t(method === 'POST' ? 'role.taken' : 'role.dropped'));
        await Promise.all([loadMyRoles(), loadProfileData()]);
    } catch (error) {
        console.error('Role change error:', error);
        showToast(t('role.networkError'));
    }
}

async function loadSourceQueue() {
    const container = document.getElementById('sourceQueueContent');
    if (!container) return;
    try {
        const res = await fetch(`${API_BASE_URL}/me/source-check-queue?limit=3`, { credentials: 'include' });
        if (res.status === 401) return goToLogin();
        if (!res.ok) throw new Error('Failed to load queue');
        const items = (await res.json()).items || [];
        if (!items.length) {
            container.innerHTML = `<p class="empty-message">${escapeHtml(t('queue.empty'))}</p>`;
            return;
        }
        container.innerHTML = items.map((it) => {
            const hosts = (it.resources || []).map((r) => hostnameOf(r.url)).join(', ');
            const checks = escapeHtml(it.check_count
                ? t('queue.checks', { n: it.check_count })
                : t('queue.noCheck'));
            return `
                <a class="queue-item" href="argument.html?id=${encodeURIComponent(it.argument_id)}&check=${encodeURIComponent(it.counter_argument_id)}">
                    <span class="queue-claim">“${escapeHtml(it.argument_text || '')}”</span>
                    <span class="queue-text">${escapeHtml(it.text || '')}</span>
                    <span class="queue-meta">${it.persona ? escapeHtml(t('queue.forPersona', { name: it.persona })) : ''}${icon('clip')} ${escapeHtml(hosts)} · ${checks}</span>
                </a>`;
        }).join('');
    } catch (error) {
        console.error('Source queue error:', error);
        container.innerHTML = `<p class="empty-message">${escapeHtml(t('queue.loadError'))}</p>`;
    }
}

// ---- Contribution status tracker ----
const STATUS_META = {
    pending:  { icon: icon('hourglass'), label: t('status.pending'),  cls: 'status-pending' },
    approved: { icon: icon('check'),     label: t('status.approved'), cls: 'status-approved' },
    rejected: { icon: icon('x'),         label: t('status.rejected'), cls: 'status-rejected' },
};

// « Brouillons »: answers started but not published. Hidden when there are none.
async function loadDrafts() {
    try {
        const res = await fetch(`${API_BASE_URL}/me/drafts`, { credentials: 'include' });
        if (!res.ok) return;
        const data = await res.json();
        renderDrafts(Array.isArray(data.drafts) ? data.drafts : []);
    } catch (error) {
        console.error('Error loading drafts:', error);
    }
}

function renderDrafts(drafts) {
    const section = document.getElementById('draftsSection');
    const list = document.getElementById('draftsList');
    if (!section || !list) return;
    section.style.display = drafts.length ? 'block' : 'none';
    list.innerHTML = '';
    drafts.forEach((d) => {
        const personaId = d.persona_archetype_id ?? null;
        const who = personaId == null ? d.persona_name : t('profile.draftFor', { name: d.persona_name });
        const row = document.createElement('div');
        row.className = 'draft-row';
        row.innerHTML = `
            <div class="draft-main">
                <p class="draft-claim">${d.claim_text ? quotedClaimHtml(d.claim_text) : escapeHtml(t('common.untitled'))}</p>
                <p class="draft-meta">${escapeHtml(who || '')} · ${escapeHtml(formatTimeAgo(d.updated_at))}</p>
            </div>
            <a class="btn btn-secondary btn-small" href="argument.html?id=${encodeURIComponent(d.argument_id)}&amp;write=${encodeURIComponent(personaId ?? 'any')}">${escapeHtml(t('profile.draftResume'))}</a>
            <button type="button" class="draft-delete" aria-label="${escapeHtml(t('profile.draftDelete'))}" title="${escapeHtml(t('profile.draftDelete'))}">${icon('x')}</button>`;
        row.querySelector('.draft-delete').addEventListener('click', () => removeDraft(d, drafts));
        list.appendChild(row);
    });
}

async function removeDraft(draft, drafts) {
    renderDrafts(drafts.filter((d) => d !== draft));
    try {
        const res = await deleteDraft(draft.id);
        if (!res.ok && res.status !== 404) throw new Error(`Delete failed: ${res.status}`);
    } catch (error) {
        console.error('Error deleting draft:', error);
        showToast(t('profile.draftDeleteError'));
        return loadDrafts();
    }
    showToast(t('profile.draftDeleted'), {
        duration: 6000,
        actions: [{ label: t('common.cancel'), onClick: () => undoDraftDelete(draft) }],
    });
}

async function undoDraftDelete(draft) {
    try {
        const res = await fetch(`${API_BASE_URL}/me/drafts`, {
            method: 'PUT',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                argument_id: draft.argument_id,
                persona_archetype_id: draft.persona_archetype_id ?? null,
                text: draft.text || '',
                sources: draftSourceUrls(draft),
            }),
        });
        if (!res.ok) showToast(apiError(await res.json().catch(() => ({})), 'composer.draftError'));
    } catch (_) {
        showToast(t('composer.draftError'));
    }
    loadDrafts();
}

async function loadContributionStatus() {
    const container = document.getElementById('statusTrackerContent');
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE_URL}/me/contributions`, { credentials: 'include' });
        if (res.status === 401) return goToLogin();
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        const data = await res.json();
        renderContributionStatus(data.items || [], data.counts || {});
    } catch (error) {
        console.error('Status tracker error:', error);
        container.innerHTML = `<p class="empty-message">${escapeHtml(t('tracker.loadError'))}</p>`;
    }
}

// Tracker "Type" column: icon + label per kind of thing sent.
const TRACKER_KIND = {
    source: ['book', 'tracker.source'],
    claim: ['quote', 'tracker.claim'],
    response: ['message', 'tracker.response'],
};

function renderContributionStatus(items, counts) {
    const container = document.getElementById('statusTrackerContent');
    if (!container) return;

    if (!items.length) {
        container.innerHTML = `<p class="empty-message">${t('tracker.emptyHtml')}</p>`;
        return;
    }

    const summary = ['pending', 'approved', 'rejected']
        .filter((s) => counts[s])
        .map((s) => `<span class="status-chip ${STATUS_META[s].cls}">${STATUS_META[s].icon} ${counts[s]} ${escapeHtml(STATUS_META[s].label.toLowerCase())}</span>`)
        .join('');

    const rows = items.map((it) => {
        const meta = STATUS_META[it.status] || STATUS_META.pending;
        const [kindIcon, kindKey] = TRACKER_KIND[it.kind] || TRACKER_KIND.response;
        const kind = `<span class="tracker-kind">${icon(kindIcon)} ${escapeHtml(t(kindKey))}</span>`;
        const title = it.title || it.text || t('common.untitled');
        const when = it.submitted_at
            ? `<time datetime="${escapeHtml(it.submitted_at)}">${escapeHtml(formatDate(it.submitted_at))}</time>`
            : '—';
        // Answers link to their claim page (`href`, once published); sources to the source.
        const inner = safeHref(it.href)
            || (it.kind !== 'source' && it.argument_id != null && it.status === 'approved' ? `argument.html?id=${encodeURIComponent(it.argument_id)}` : null);
        const link = inner
            ? `<a href="${escapeHtml(inner)}">${escapeHtml(title)}</a>`
            : safeHref(it.url)
                ? `<a href="${escapeHtml(safeHref(it.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(title)}</a>`
                : escapeHtml(title);
        return `
            <tr>
                <td data-label="${escapeHtml(t('tracker.colType'))}">${kind}</td>
                <td data-label="${escapeHtml(t('tracker.colWhat'))}">${link}</td>
                <td data-label="${escapeHtml(t('tracker.colSent'))}">${when}</td>
                <td data-label="${escapeHtml(t('tracker.colStatus'))}"><span class="status-chip ${meta.cls}">${meta.icon} ${escapeHtml(meta.label)}</span></td>
            </tr>`;
    }).join('');

    container.innerHTML = `
        ${summary ? `<div class="status-summary">${summary}</div>` : ''}
        <div class="table-scroll">
            <table class="status-table">
                <thead>
                    <tr>${['tracker.colType', 'tracker.colWhat', 'tracker.colSent', 'tracker.colStatus'].map((k) => `<th scope="col">${escapeHtml(t(k))}</th>`).join('')}</tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

// The claims, once per page: they give the topic order and the recommendations.
// Credentialed so the API can flag claims we authored or already answered.
let profileClaims = null;
function fetchProfileClaims() {
    profileClaims ||= fetch(`${API_BASE_URL}/arguments`, { credentials: 'include' })
        .then((res) => (res.ok ? res.json() : []))
        .then((data) => (Array.isArray(data) ? data : []));
    return profileClaims;
}

// ---- Recommended for you (claims matched to followed tags) ----
async function loadRecommendations() {
    const container = document.getElementById('recommendationsContent');
    if (!container) return;

    try {
        const [tagsRes, args] = await Promise.all([
            fetch(`${API_BASE_URL}/user/${currentUser.id}/followed-tags`, { credentials: 'include' }),
            fetchProfileClaims(),
        ]);
        if (tagsRes.status === 401) return goToLogin();
        const followed = tagsRes.ok ? (await tagsRes.json()).followed_tags.map((t) => t.tag) : [];
        renderRecommendations(args, followed);
    } catch (error) {
        console.error('Recommendations error:', error);
        container.innerHTML = `<p class="empty-message">${escapeHtml(t('reco.loadError'))}</p>`;
    }
}

function renderRecommendations(args, followedTags) {
    const container = document.getElementById('recommendationsContent');
    if (!container) return;

    // No declared interests yet → invite the user to pick some (progressive
    // profiling: this single action powers the whole section).
    if (!followedTags.length) {
        container.innerHTML = `
            <p class="empty-message">${escapeHtml(t('reco.pickTopics'))}
            <button type="button" class="btn btn-secondary btn-small" id="recAddTags">${escapeHtml(t('reco.pickTopicsBtn'))}</button></p>`;
        document.getElementById('recAddTags')?.addEventListener('click', goToTopicPills);
        return;
    }

    const followedSet = new Set(followedTags.map((t) => t.toLowerCase()));
    const urgency = (a) => {
        const n = a.counter_argument_count || 0;
        return n === 0 ? 3 : (n < 3 ? 1 : 0);
    };

    const scored = args
        // Never recommend a claim you wrote or already answered (§3.3 own_content).
        .filter((a) => !a.viewer_is_author && !a.viewer_has_responded)
        .map((a) => {
            const matched = (a.tags || []).filter((t) => followedSet.has(String(t).toLowerCase()));
            return { a, matched, score: matched.length * 2 + urgency(a) };
        })
        .filter((x) => x.matched.length > 0) // must relate to a followed topic
        .sort((x, y) => y.score - x.score
            || (y.a.importance?.avg_score || 0) - (x.a.importance?.avg_score || 0)
            || (x.a.contribution_count || 0) - (y.a.contribution_count || 0))
        .slice(0, 3);

    if (!scored.length) {
        container.innerHTML = `<p class="empty-message">${t('reco.noneHtml')}</p>`;
        return;
    }

    container.innerHTML = '';
    scored.forEach(({ a, matched }) => {
        const count = a.counter_argument_count || 0;
        const why = escapeHtml(t('reco.because', { tags: matched.slice(0, 2).map(topicLabel).join(', ') }));
        const urgencyNote = count === 0
            ? `<span class="needs-badge needs-badge-hot">${icon('zap')} ${escapeHtml(t('reco.noResponse'))}</span>`
            : `<span class="needs-badge">${escapeHtml(t('reco.responses', { n: count }))}</span>`;

        const card = document.createElement('a');
        card.className = 'needs-card';
        card.href = `argument.html?id=${encodeURIComponent(a.id)}`;
        card.innerHTML = `
            <p class="rec-why">${icon('target')} ${why}</p>
            <p class="needs-quote">${quotedClaimHtml(a.text)}</p>
            <div class="needs-meta">${urgencyNote}</div>
            <span class="needs-cta">${count === 0 ? `${icon('pen')} ${escapeHtml(t('reco.beFirst'))}` : `${icon('message')} ${escapeHtml(t('reco.addYours'))}`}</span>`;
        container.appendChild(card);
    });
}

// ---- Activity from people you follow ----
function timeAgo(iso) {
    return formatTimeAgo(iso);
}

async function loadFollowingActivity() {
    const container = document.getElementById('followingActivityContent');
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE_URL}/me/following-activity`, { credentials: 'include' });
        if (res.status === 401) return goToLogin();
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        const data = await res.json();
        renderFollowingActivity(data.items || []);
    } catch (error) {
        console.error('Following activity error:', error);
        container.innerHTML = `<p class="empty-message">${escapeHtml(t('activity.loadError'))}</p>`;
    }
}

const ACTIVITY_ICON = { response: icon('message'), source: icon('book'), claim: icon('quote') };
// The API sends an English verb; show ours, keyed by that verb (or the kind).
const ACTIVITY_VERB_KEYS = {
    'answered a claim': 'activity.answeredClaim',
    'wrote a response': 'activity.wroteResponse',
    'shared a source': 'activity.sharedSource',
};
const ACTIVITY_VERB_CODES = {
    wrote_response: 'activity.wroteResponse',
    shared_source: 'activity.sharedSource',
};
function activityVerb(it) {
    const key = ACTIVITY_VERB_CODES[it.verb_code] || ACTIVITY_VERB_KEYS[it.verb]
        || (it.kind === 'source' ? 'activity.sharedSource' : 'activity.wroteResponse');
    return t(key);
}

function renderFollowingActivity(items) {
    const container = document.getElementById('followingActivityContent');
    if (!container) return;

    if (!items.length) {
        container.innerHTML = `<p class="empty-message">${t('activity.emptyHtml')}</p>`;
        return;
    }

    container.innerHTML = items.map((it) => {
        const kindIcon = ACTIVITY_ICON[it.kind] || '•';
        const who = `<a href="profile.html?id=${encodeURIComponent(it.user.id)}" class="activity-user">${escapeHtml(it.user.username)}</a>`;
        const what = safeHref(it.href)
            ? `<a href="${escapeHtml(safeHref(it.href))}"${it.kind === 'source' ? ' target="_blank" rel="noopener noreferrer"' : ''}>“${escapeHtml(it.title)}”</a>`
            : `“${escapeHtml(it.title)}”`;
        const when = it.at ? `<time datetime="${escapeHtml(it.at)}" class="activity-time">${escapeHtml(timeAgo(it.at))}</time>` : '';
        return `
            <div class="activity-item">
                <span class="activity-icon" aria-hidden="true">${kindIcon}</span>
                <p class="activity-text">${who} ${escapeHtml(activityVerb(it))} ${what}</p>
                ${when}
            </div>`;
    }).join('');
}

async function loadProfileData() {
    try {
        const response = await fetch(`${API_BASE_URL}/user/${profileUserId}`, {
            credentials: 'include'
        });
        
        if (response.status === 404) {
            showProfileNotFound();
            return false;
        }
        if (!response.ok) {
            throw new Error('Failed to load profile');
        }
        
        const data = await response.json();
        
        // Update profile header
        document.getElementById('profileUsername').textContent = data.user.username;
        document.getElementById('profileRole').textContent = userRoleLabel(data.user.role);
        document.getElementById('profileRole').className = `role-badge ${data.user.role}`;
        document.getElementById('profileMemberSince').innerHTML = t('profile.memberSinceHtml', {
            date: `<time datetime="${escapeHtml(data.user.created_at)}">${escapeHtml(formatDate(data.user.created_at))}</time>`,
        });

        // Profile picture (works for own and other profiles)
        renderProfileAvatar(data.user);
        
        // Update stats
        // Stats, with singular/plural labels (« 1 contribution », « 2 abonnés »).
        document.getElementById('reputationScore').textContent = formatNumber(data.user.reputation_score);
        [['contributionCount', data.contribution_count, 'profile.statContributions'],
         ['followingCount', data.following_count, 'profile.statFollowing'],
         ['followersCount', data.followers_count, 'profile.statFollowers']].forEach(([id, n, key]) => {
            const dd = document.getElementById(id);
            if (!dd) return;
            dd.textContent = formatNumber(n);
            const dt = dd.parentElement?.querySelector('dt');
            if (dt) dt.textContent = t(key, { n: Number(n) || 0 });
        });

        // Level bar (own profile only — the container is hidden otherwise).
        if (isOwnProfile) renderLevelBar(data.user.reputation_score);
        renderRoleChips(data.roles || []);
        
        // Update follow button state
        isFollowingProfile = data.is_following;
        updateFollowButton();
        
        // Display endorsements
        displayEndorsements(data.endorsements);
        
        // Display contributions
        displayProfileContributions(data.contributions || []);
        
        // Update page title
        document.title = t('site.pageTitle', { page: data.user.username });
        return true;
    } catch (error) {
        console.error('Error loading profile:', error);
        showProfileNotFound();
        return false;
    }
}

// No such member: replace the whole profile layout with one short message.
function showProfileNotFound() {
    const layout = document.querySelector('.profile-layout');
    if (!layout) return;
    document.title = t('site.pageTitle', { page: t('profile.notFound') });
    layout.outerHTML = `
        <div class="not-found">
            <h2>${escapeHtml(t('profile.notFound'))}</h2>
            <p>${t('profile.notFoundHtml')}</p>
        </div>`;
}

function updateFollowButton() {
    const btn = document.getElementById('followButton');
    if (!btn) return;
    
    if (isFollowingProfile) {
        btn.textContent = t('profile.unfollow');
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
    } else {
        btn.textContent = t('profile.follow');
        btn.classList.remove('btn-secondary');
        btn.classList.add('btn-primary');
    }
}

// ========================================
// Profile Picture (upload / remove / display)
// ========================================

function renderProfileAvatar(user) {
    const wrap = document.getElementById('profileAvatar');
    if (!wrap) return;

    const initial = ((user.username || '?').trim().charAt(0) || '?').toUpperCase();
    const initials = wrap.querySelector('.avatar-initials');
    if (initials) initials.textContent = initial;

    let img = document.getElementById('profileAvatarImg');
    if (safeHref(user.avatar_url)) {
        if (!img) {
            img = document.createElement('img');
            img.id = 'profileAvatarImg';
            img.alt = '';
            img.onerror = () => img.remove();
            wrap.appendChild(img);
        }
        img.src = safeHref(user.avatar_url);
    } else if (img) {
        img.remove();
    }

    // Changing the picture only makes sense on your own profile (needs login anyway).
    const changeBtn = document.getElementById('changeAvatarBtn');
    if (changeBtn) changeBtn.hidden = !(isOwnProfile && currentUser);
}

function setupAvatarControls() {
    const changeBtn = document.getElementById('changeAvatarBtn');
    const fileInput = document.getElementById('avatarInput');
    const errorEl = document.getElementById('avatarError');
    if (!changeBtn || !fileInput) return;

    changeBtn.addEventListener('click', () => {
        if (!currentUser) {
            goToLogin();
            return;
        }
        errorEl.style.display = 'none';
        fileInput.click();
    });

    fileInput.addEventListener('change', async () => {
        const file = fileInput.files && fileInput.files[0];
        fileInput.value = ''; // allow re-selecting the same file later
        if (!file) return;

        errorEl.style.display = 'none';

        const form = new FormData();
        form.append('avatar', file);

        changeBtn.disabled = true;
        changeBtn.setAttribute('aria-label', t('profile.uploading'));

        try {
            const res = await fetch(`${API_BASE_URL}/me/avatar`, {
                method: 'POST',
                credentials: 'include',
                body: form
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                errorEl.textContent = apiError(data, 'profile.uploadFailed');
                errorEl.style.display = 'block';
                return;
            }

            if (currentUser) {
                currentUser.avatar_url = data.avatar_url;
                updateNavForLoggedInUser();
            }
            renderProfileAvatar({
                username: currentUser?.username || '',
                avatar_url: data.avatar_url
            });
        } catch (error) {
            console.error('Avatar upload error:', error);
            errorEl.textContent = t('profile.uploadRetry');
            errorEl.style.display = 'block';
        } finally {
            changeBtn.disabled = false;
            changeBtn.setAttribute('aria-label', t('profile.changePhoto'));
        }
    });
}

async function toggleFollow() {
    if (!currentUser) {
        goToLogin();
        return;
    }
    
    const btn = document.getElementById('followButton');
    btn.disabled = true;
    
    try {
        const method = isFollowingProfile ? 'DELETE' : 'POST';
        const response = await fetch(`${API_BASE_URL}/user/${profileUserId}/follow`, {
            method: method,
            credentials: 'include'
        });
        
        if (response.status === 401) return goToLogin();
        
        if (response.ok) {
            isFollowingProfile = !isFollowingProfile;
            updateFollowButton();
            
            // Update followers count
            const countEl = document.getElementById('followersCount');
            const currentCount = parseInt(countEl.textContent);
            countEl.textContent = isFollowingProfile ? currentCount + 1 : currentCount - 1;
        }
    } catch (error) {
        console.error('Error toggling follow:', error);
    }
    
    btn.disabled = false;
}

function displayEndorsements(endorsements) {
    const container = document.getElementById('endorsementsList');
    
    if (!endorsements || endorsements.length === 0) {
        container.innerHTML = `<p class="empty-message">${escapeHtml(t('profile.endorsementsEmpty'))}</p>`;
        return;
    }
    
    container.innerHTML = endorsements.map(e => `
        <div class="endorsement-card">
            <div class="endorser">
                <a href="profile.html?id=${encodeURIComponent(e.endorser_id)}" class="contributor-link">${escapeHtml(e.endorser_username)}</a>
            </div>
            <div class="reason">“${escapeHtml(e.reason)}”</div>
            <div class="date">${escapeHtml(formatDate(e.created_at))}</div>
        </div>
    `).join('');
}

function displayProfileContributions(contributions) {
    const container = document.getElementById('contributionsList');
    
    if (contributions.length === 0) {
        const message = isOwnProfile
            ? t('profile.contributionsEmptyMineHtml')
            : escapeHtml(t('profile.contributionsEmpty'));
        container.innerHTML = `<p class="empty-message">${message}</p>`;
        return;
    }
    
    container.innerHTML = '';
    contributions.forEach(contribution => {
        const card = createContributionCard(contribution);
        container.appendChild(card);
    });
}

// ========================================
// Tag Following Functions
// ========================================

let followedTopics = new Set();
const savingTopics = new Set();

async function loadFollowedTags() {
    const container = document.getElementById('followedTagsContent');

    try {
        const [response, claims] = await Promise.all([
            fetch(`${API_BASE_URL}/user/${currentUser.id}/followed-tags`, { credentials: 'include' }),
            fetchProfileClaims(),
        ]);
        if (!response.ok) throw new Error('Failed to load followed tags');

        const data = await response.json();
        followedTopics = new Set(data.followed_tags.map(({ tag }) => tag.toLowerCase()));

        // Homepage order; a followed theme no claim uses any more stays listed so it can be unfollowed.
        const topics = [...new Set([...topicsByUse(claims).map((tag) => tag.toLowerCase()), ...followedTopics])];

        container.innerHTML = topics.map((tag) =>
            `<button type="button" class="topic" data-value="${escapeHtml(tag)}" aria-pressed="${followedTopics.has(tag)}">${escapeHtml(topicLabel(tag))}</button>`
        ).join('');
        container.onclick = (e) => {
            const pill = e.target.closest('.topic');
            if (pill) toggleFollowedTopic(pill);
        };
    } catch (error) {
        console.error('Error loading followed tags:', error);
        container.innerHTML = `<p class="error-message">${escapeHtml(t('profile.tagsLoadError'))}</p>`;
    }
}

// One click saves right away; the pill flips first and flips back if the save fails.
async function toggleFollowedTopic(pill) {
    const tag = pill.dataset.value;
    if (savingTopics.has(tag)) return;
    savingTopics.add(tag);

    const wasFollowing = pill.getAttribute('aria-pressed') === 'true';
    pill.setAttribute('aria-pressed', String(!wasFollowing));
    const url = `${API_BASE_URL}/user/${currentUser.id}/followed-tags`;

    try {
        const response = await fetch(wasFollowing ? `${url}/${encodeURIComponent(tag)}` : url, wasFollowing
            ? { method: 'DELETE', credentials: 'include' }
            : { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify({ tag }) });

        const data = response.ok ? {} : await response.json().catch(() => ({}));
        // Already in the wanted state (changed in another tab) counts as saved.
        if (!response.ok && !['already_following_tag', 'not_following_tag'].includes(data.code)) {
            pill.setAttribute('aria-pressed', String(wasFollowing));
            apiErrorToast(data, 'profile.tagSaveError', response);
            return;
        }

        if (wasFollowing) followedTopics.delete(tag); else followedTopics.add(tag);
        renderRecommendations(await fetchProfileClaims(), [...followedTopics]);
    } catch (error) {
        console.error('Error saving followed tag:', error);
        pill.setAttribute('aria-pressed', String(wasFollowing));
        showToast(t('common.serverUnreachable'));
    } finally {
        savingTopics.delete(tag);
    }
}

// From the recommendations' empty state to the pills that fill it.
function goToTopicPills() {
    document.getElementById('tagsFollowingSection')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    document.querySelector('#followedTagsContent .topic')?.focus({ preventScroll: true });
}

// ========================================
// Profile Following Functions
// ========================================

async function loadFollowingProfiles() {
    const container = document.getElementById('followingList');
    
    try {
        const response = await fetch(`${API_BASE_URL}/user/${currentUser.id}/following`, {
            credentials: 'include'
        });
        
        if (response.status === 401) return goToLogin();
        if (!response.ok) throw new Error('Failed to load following');
        
        const data = await response.json();
        
        if (data.following.length === 0) {
            container.innerHTML = `<p class="empty-message">${escapeHtml(t('profile.followingEmptyLong'))}</p>`;
            return;
        }
        
        container.innerHTML = data.following.map(user => `
            <div class="user-card">
                <a href="profile.html?id=${encodeURIComponent(user.id)}" class="user-card-link">
                    <div class="user-card-name">${escapeHtml(user.username)}</div>
                    <span class="role-badge ${escapeHtml(user.role)}">${escapeHtml(userRoleLabel(user.role))}</span>
                </a>
                <button type="button" class="btn btn-secondary btn-small" data-unfollow="${encodeURIComponent(user.id)}">${escapeHtml(t('profile.unfollow'))}</button>
            </div>
        `).join('');

        // Delegated unfollow (replaces inline onclick).
        if (!container.dataset.bound) {
            container.dataset.bound = 'true';
            container.addEventListener('click', (e) => {
                const btn = e.target.closest('[data-unfollow]');
                if (btn) unfollowUser(Number(btn.dataset.unfollow));
            });
        }

    } catch (error) {
        console.error('Error loading following:', error);
        container.innerHTML = `<p class="error-message">${escapeHtml(t('profile.followingLoadError'))}</p>`;
    }
}

async function unfollowUser(userId) {
    try {
        const response = await fetch(`${API_BASE_URL}/user/${userId}/follow`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
        if (response.status === 401) return goToLogin();
        
        if (response.ok) {
            await loadFollowingProfiles();
            // Update following count
            const countEl = document.getElementById('followingCount');
            countEl.textContent = parseInt(countEl.textContent) - 1;
        }
    } catch (error) {
        console.error('Error unfollowing user:', error);
    }
}

// ========================================
// Contribute: profiles → tasks
// ========================================

// Who the contributor is → what they can do right now. Profiles and tasks are
// plain data so adding a way to help means adding an entry, not a page.
const REPO_URL = 'https://github.com/LucieRC/re.inform';

const CONTRIBUTOR_PROFILES = [
    { id: 'quick', icon: 'clock', tasks: ['prioritize', 'rate', 'claim'] },
    { id: 'talk', icon: 'message', tasks: ['claim', 'answer', 'personas'] },
    { id: 'science', icon: 'flask', tasks: ['review', 'resource', 'answer'] },
    { id: 'write', icon: 'pen', tasks: ['answer', 'improve', 'personas'] },
    { id: 'visual', icon: 'palette', tasks: ['visual', 'personas'] },
    { id: 'tech', icon: 'code', tasks: ['code', 'data'] },
].map((p) => ({ ...p, title: t(`who.${p.id}.title`), who: t(`who.${p.id}.who`) }));

// `open` reveals a workspace on this page; `href` (or `hrefFrom`, computed once
// claims are loaded) leads to where the work happens.
const CONTRIBUTOR_TASKS = Object.fromEntries(Object.entries({
    // Voting, rating and answering happen in the home page's « Contribuer » mode.
    prioritize: { icon: 'target', account: false, href: 'index.html?mode=help&task=vote' },
    rate: { icon: 'star', account: false, href: 'index.html?mode=help&task=rate' },
    claim: { icon: 'plus', account: true, href: 'index.html?add=1' },
    answer: { icon: 'message', account: true, href: 'index.html?mode=help&task=write' },
    personas: { icon: 'users', account: false, href: 'personalities.html' },
    review: { icon: 'search', account: true, hrefFrom: 'review' },
    resource: { icon: 'book', account: true, open: 'ws-resource' },
    improve: { icon: 'pen', account: true, hrefFrom: 'review' },
    visual: { icon: 'image', account: true, open: 'ws-resource' },
    code: { icon: 'code', account: false, href: `${REPO_URL}/issues`, external: true },
    data: { icon: 'chart', account: false, href: REPO_URL, external: true },
}).map(([id, task]) => [id, {
    ...task,
    title: t(`task.${id}.title`),
    desc: t(`task.${id}.desc`),
    time: t(`task.${id}.time`),
}]));

const PROFILE_STORAGE_KEY = 'reinform.contributorProfile';
const contributeState = { args: [], archetypes: [], profileId: null, taskId: null };

function initContributePage() {
    const picker = document.getElementById('profilePicker');
    if (!picker) return;

    const notice = document.getElementById('accountNotice');
    if (notice) notice.hidden = !!currentUser;

    picker.innerHTML = CONTRIBUTOR_PROFILES.map((p) => `
        <button type="button" class="onb-profile" data-profile="${p.id}" aria-pressed="false">
            <span class="onb-profile-icon">${icon(p.icon)}</span>
            <span class="onb-profile-title">${escapeHtml(p.title)}</span>
            <span class="onb-profile-who">${escapeHtml(p.who)}</span>
        </button>
    `).join('');
    picker.querySelectorAll('.onb-profile').forEach((btn) => {
        btn.addEventListener('click', () => selectContributorProfile(btn.dataset.profile, { scroll: true }));
    });

    setupClaimForm();
    setupResourceForm();
    loadContributeData();

    // ?as=<profile>&task=<task> makes every profile and task linkable (home page, outreach).
    const params = new URLSearchParams(window.location.search);
    let profileId = params.get('as');
    if (!profileId) {
        try { profileId = localStorage.getItem(PROFILE_STORAGE_KEY); } catch (_) { /* storage unavailable */ }
    }
    if (CONTRIBUTOR_PROFILES.some((p) => p.id === profileId)) {
        selectContributorProfile(profileId, { scroll: params.has('as') });
    }
    const taskId = params.get('task');
    if (CONTRIBUTOR_TASKS[taskId]?.open) openContributorTask(taskId);
}

function selectContributorProfile(profileId, { scroll = false } = {}) {
    const profile = CONTRIBUTOR_PROFILES.find((p) => p.id === profileId);
    if (!profile) return;
    contributeState.profileId = profileId;
    try { localStorage.setItem(PROFILE_STORAGE_KEY, profileId); } catch (_) { /* storage unavailable */ }

    document.querySelectorAll('#profilePicker .onb-profile').forEach((btn) => {
        btn.setAttribute('aria-pressed', String(btn.dataset.profile === profileId));
    });

    const section = document.getElementById('taskSection');
    const list = document.getElementById('taskList');
    list.innerHTML = '';
    profile.tasks.forEach((taskId, i) => list.appendChild(createTaskCard(taskId, i === 0)));
    section.hidden = false;

    // A different profile means a different set of tasks: close any open workspace.
    if (!profile.tasks.includes(contributeState.taskId)) closeContributorWorkspaces();
    syncContributeUrl();
    if (scroll) section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function createTaskCard(taskId, isFirst) {
    const task = CONTRIBUTOR_TASKS[taskId];
    const el = document.createElement(task.open ? 'button' : 'a');
    el.className = `help-path${isFirst ? ' help-path-primary' : ''}`;
    el.dataset.task = taskId;
    if (task.open) {
        el.type = 'button';
        el.setAttribute('aria-controls', task.open);
        el.setAttribute('aria-expanded', 'false');
        el.addEventListener('click', () => openContributorTask(taskId));
    } else {
        el.href = taskHref(task);
        if (task.external) { el.target = '_blank'; el.rel = 'noopener noreferrer'; }
    }
    const accountTag = task.account && !currentUser ? t('contribute.freeAccount') : '';
    el.innerHTML = `
        <span class="help-path-icon">${icon(task.icon)}</span>
        <span class="help-path-body">
            <span class="help-path-title">${escapeHtml(task.title)}</span>
            <span class="help-path-desc">${escapeHtml(task.desc)}</span>
        </span>
        <span class="help-path-effort">${escapeHtml(task.time + accountTag)}</span>
    `;
    return el;
}

// Point "go there" tasks at a real claim that needs the work, once claims are loaded.
function taskHref(task) {
    if (task.href) return task.href;
    const args = contributeState.args;
    if (!args.length) return 'index.html';
    const unrated = args.filter((a) => a.top_response && !a.top_response.rating_count);
    const pool = unrated.length ? unrated : args;
    const pick = task.hrefFrom === 'unrated'
        ? pool[Math.floor(Math.random() * pool.length)]
        : args.slice().sort((a, b) => (b.importance?.avg_score || 0) - (a.importance?.avg_score || 0))[0];
    return `argument.html?id=${encodeURIComponent(pick.id)}`;
}

function openContributorTask(taskId) {
    const task = CONTRIBUTOR_TASKS[taskId];
    const ws = task && document.getElementById(task.open);
    if (!ws) return;

    closeContributorWorkspaces();
    contributeState.taskId = taskId;
    ws.hidden = false;
    document.querySelectorAll(`#taskList [data-task="${taskId}"]`).forEach((b) => b.setAttribute('aria-expanded', 'true'));

    // Logged out: say so up front, with a way back to exactly this task.
    const hint = ws.querySelector('.onb-login-hint');
    if (hint) {
        const links = { registerHref: authHref('register.html'), loginHref: authHref('login.html') };
        hint.innerHTML = currentUser ? '' : `
            <p class="onb-account onb-account-inline">${t(task.account ? 'contribute.needAccountHtml' : 'contribute.voteNoAccountHtml', links)}</p>`;
    }

    syncContributeUrl();
    ws.scrollIntoView({ behavior: 'smooth', block: 'start' });
    ws.focus({ preventScroll: true });
}

function closeContributorWorkspaces() {
    document.querySelectorAll('.onb-workspace').forEach((w) => { w.hidden = true; });
    document.querySelectorAll('#taskList [aria-expanded]').forEach((b) => b.setAttribute('aria-expanded', 'false'));
    if (!document.querySelector('.onb-workspace:not([hidden])')) contributeState.taskId = null;
}

function syncContributeUrl() {
    // Keep the page's other parameters (e.g. a profile ?id=).
    const url = new URL(window.location.href);
    const set = (key, value) => (value ? url.searchParams.set(key, value) : url.searchParams.delete(key));
    set('as', contributeState.profileId);
    set('task', contributeState.taskId);
    history.replaceState(null, '', url);
    // Login links on this page should bring the user back to the same profile and task.
    document.querySelectorAll('[data-auth]').forEach((a) => { a.href = authHref(a.dataset.auth); });
}

async function loadContributeData() {
    try {
        const [argsRes, personaRes] = await Promise.all([
            // Credentials so the API can tell us which claims this user already rated.
            fetch(`${API_BASE_URL}/arguments`, { credentials: 'include' }),
            fetch(`${API_BASE_URL}/persona-archetypes`),
        ]);
        const args = argsRes.ok ? await argsRes.json() : [];
        const personas = personaRes.ok ? await personaRes.json() : [];
        contributeState.args = Array.isArray(args) ? args : [];
        contributeState.archetypes = (Array.isArray(personas) ? personas : []).filter((p) => isActivePersona(p) && p.layer === 'archetype');
    } catch (error) {
        console.error('Contribute data error:', error);
    }

    renderPriorityList(contributeState.args);
    renderAnswerGaps();
    populateResourceClaimPicker(contributeState.args);
    // Refresh link targets that depend on the claims.
    document.querySelectorAll('#taskList a[data-task]').forEach((a) => { a.href = taskHref(CONTRIBUTOR_TASKS[a.dataset.task]); });
}

// Claims that still lack an answer for one of the named personas — the most
// urgent first — each with one-tap "write for <persona>" links.
function renderAnswerGaps() {
    const el = document.getElementById('needsHelp');
    if (!el) return;

    const gaps = contributeState.args
        .map((a) => {
            const covered = new Set((a.personas || []).map((p) => p.id));
            return { a, missing: contributeState.archetypes.filter((p) => !covered.has(p.id)) };
        })
        .filter((g) => g.missing.length)
        .sort((x, y) => ((y.a.importance?.avg_score || 0) - (x.a.importance?.avg_score || 0))
            || ((x.a.counter_argument_count || 0) - (y.a.counter_argument_count || 0)))
        .slice(0, 6);

    if (!gaps.length) {
        el.innerHTML = `<p class="empty-message">${t('contribute.gapsEmptyHtml')}</p>`;
        return;
    }

    el.innerHTML = gaps.map(({ a, missing }) => `
        <div class="needs-card">
            <p class="needs-quote">${quotedClaimHtml(a.text)}</p>
            <div class="onb-gap-links">
                ${missing.slice(0, 3).map((p) => `
                    <a class="gap-chip" href="argument.html?id=${encodeURIComponent(a.id)}&amp;write=${encodeURIComponent(p.id)}">${icon('pen')} ${escapeHtml(t('contribute.gapFor', { name: p.name }))}</a>
                `).join('')}
            </div>
        </div>
    `).join('') + `<p class="onb-more"><a href="index.html">${escapeHtml(t('contribute.seeAll'))}</a></p>`;
}

// Home feed: the "Heard a claim…" prompt opens the claim form in place.
// A « + Ajouter… » button that unfolds its form in place.
function setupFoldingForm({ toggleId, formId, cancelId, messageId, focusId, authHrefFor }) {
    const toggle = document.getElementById(toggleId);
    const form = document.getElementById(formId);
    if (!toggle || !form) return null;
    const messageEl = document.getElementById(messageId);
    const isBlank = () => [...form.querySelectorAll('input, textarea')].every((f) => !f.value.trim());

    const open = () => {
        toggle.hidden = true;
        toggle.setAttribute('aria-expanded', 'true');
        form.hidden = false;
        // Logged out: say so up front rather than after they've typed.
        if (!currentUser && !messageEl.textContent) {
            messageEl.innerHTML = t('claimForm.needAccountHtml', { registerHref: authHrefFor('register.html'), loginHref: authHrefFor('login.html') });
            messageEl.className = 'form-message';
        }
        form.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        document.getElementById(focusId).focus({ preventScroll: true });
    };
    const close = ({ refocus = true } = {}) => {
        form.hidden = true;
        toggle.hidden = false;
        toggle.setAttribute('aria-expanded', 'false');
        // A "posted" confirmation shouldn't greet them the next time they open it.
        if (messageEl.classList.contains('success')) {
            messageEl.textContent = '';
            messageEl.className = 'form-message';
        }
        if (refocus) toggle.focus();
    };

    // Leaving the form (click or Tab elsewhere): keep it open if they've started
    // writing so the draft stays visible; otherwise fold back to the button.
    const collapseIfEmpty = (target) => {
        if (form.hidden || form.contains(target) || toggle.contains(target)) return;
        if (!isBlank()) return;
        close({ refocus: false });
    };
    document.addEventListener('pointerdown', (e) => collapseIfEmpty(e.target));
    form.addEventListener('focusout', (e) => {
        if (e.relatedTarget) collapseIfEmpty(e.relatedTarget);
    });

    toggle.addEventListener('click', open);
    document.getElementById(cancelId).addEventListener('click', () => close());
    return { open };
}

function setupFeedComposer() {
    const params = new URLSearchParams(window.location.search);
    const claim = setupFoldingForm({
        toggleId: 'feedComposerToggle', formId: 'feedComposer', cancelId: 'feedComposerCancel',
        messageId: 'claimFormMessage', focusId: 'claimText', authHrefFor: authHref,
    });
    if (claim) {
        document.querySelectorAll('[data-open-composer]').forEach((b) => b.addEventListener('click', claim.open));
        setupClaimForm('feedComposer');
        // ?add=1 (e.g. from the profile's "Add a claim" task) opens it straight away.
        if (params.get('add')) claim.open();
    }

    const resource = setupFoldingForm({
        toggleId: 'feedResourceToggle', formId: 'feedResourceForm', cancelId: 'feedResourceCancel',
        messageId: 'quickResourceMessage', focusId: 'quickResourceUrl',
        authHrefFor: (page) => authHrefWith(page, { addResource: 1 }),
    });
    if (resource) {
        setupQuickResourceForm();
        if (params.get('addResource')) {
            resource.open();
            restoreQuickResourceDraft();
        }
    }
}

// What was typed in « Ajouter une ressource » survives the detour through login, for this tab.
const RESOURCE_DRAFT_KEY = 'reinform.resourceDraft';

function keepQuickResourceDraft() {
    const url = document.getElementById('quickResourceUrl').value;
    const note = document.getElementById('quickResourceNote').value;
    try {
        if (url.trim() || note.trim()) sessionStorage.setItem(RESOURCE_DRAFT_KEY, JSON.stringify({ url, note }));
        else sessionStorage.removeItem(RESOURCE_DRAFT_KEY);
    } catch (_) { /* private mode */ }
}

function clearQuickResourceDraft() {
    try { sessionStorage.removeItem(RESOURCE_DRAFT_KEY); } catch (_) { /* private mode */ }
}

function restoreQuickResourceDraft() {
    let draft = null;
    try { draft = JSON.parse(sessionStorage.getItem(RESOURCE_DRAFT_KEY) || 'null'); } catch (_) { return; }
    if (!draft) return;
    const urlEl = document.getElementById('quickResourceUrl');
    const noteEl = document.getElementById('quickResourceNote');
    if (!urlEl.value) urlEl.value = draft.url || '';
    if (!noteEl.value) noteEl.value = draft.note || '';
}

// « Ajouter une ressource »: a link and an optional title or note. Goes to moderator review.
function setupQuickResourceForm() {
    const form = document.getElementById('feedResourceForm');
    const messageEl = document.getElementById('quickResourceMessage');
    const submitBtn = document.getElementById('quickResourceSubmit');
    const showMessage = (html, type) => {
        messageEl.innerHTML = html;
        messageEl.className = `form-message ${type}`;
    };
    const loginFirst = () => showMessage(t('claimForm.loginFirstHtml', {
        registerHref: authHrefWith('register.html', { addResource: 1 }),
        loginHref: authHrefWith('login.html', { addResource: 1 }),
    }), 'error');

    form.addEventListener('input', keepQuickResourceDraft);
    document.getElementById('feedResourceCancel')?.addEventListener('click', clearQuickResourceDraft);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentUser) return loginFirst();
        const url = document.getElementById('quickResourceUrl').value.trim();
        const title = document.getElementById('quickResourceNote').value.trim();
        if (!url) return showMessage(escapeHtml(t('resourceForm.needLink')), 'error');
        if (!isWebLink(url)) return showMessage(escapeHtml(t('composer.sourceInvalid')), 'error');

        submitBtn.disabled = true;
        try {
            const response = await fetch(`${API_BASE_URL}/resources`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ url, title }),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                if (data?.code === 'auth_required') return loginFirst();
                return showMessage(escapeHtml(apiError(data, 'resourceForm.error')), 'error');
            }
            form.reset();
            clearQuickResourceDraft();
            showMessage(escapeHtml(t('resourceForm.success')), 'success');
        } catch (error) {
            console.error('Resource submission error:', error);
            showMessage(escapeHtml(t('common.serverUnreachable')), 'error');
        } finally {
            submitBtn.disabled = false;
        }
    });
}

function setupClaimForm(formId = 'claimForm') {
    const form = document.getElementById(formId);
    if (!form) return;
    const messageEl = document.getElementById('claimFormMessage');
    const submitBtn = document.getElementById('claimSubmitBtn');
    const showMessage = (html, type) => {
        messageEl.innerHTML = html;
        messageEl.className = `form-message ${type}`;
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentUser) {
            showMessage(t('claimForm.loginFirstHtml', { registerHref: authHref('register.html'), loginHref: authHref('login.html') }), 'error');
            return;
        }
        const text = document.getElementById('claimText').value.trim();
        if (!text) {
            showMessage(escapeHtml(t('error.claim_text_required')), 'error');
            return;
        }

        submitBtn.disabled = true;
        try {
            const response = await fetch(`${API_BASE_URL}/arguments`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ text }),
            });
            const data = await response.json();
            if (!response.ok) {
                showMessage(data?.code === 'auth_required'
                    ? t('claimForm.loginFirstHtml', { registerHref: authHref('register.html'), loginHref: authHref('login.html') })
                    : escapeHtml(apiError(data, 'claimForm.error')), 'error');
                return;
            }
            form.reset();
            const id = encodeURIComponent(data.argument?.id);
            showMessage(t('claimForm.successHtml', { href: `argument.html?id=${id}`, writeHref: `argument.html?id=${id}&amp;write=any` }), 'success');
        } catch (error) {
            console.error('Claim submission error:', error);
            showMessage(escapeHtml(t('common.serverUnreachable')), 'error');
        } finally {
            submitBtn.disabled = false;
        }
    });
}

// ========================================
// Contribute: claim prioritisation
// ========================================

// Three levels, stored on the 1..5 scale the rating store already speaks.
const PRIORITY_LEVELS = [
    { score: 1, label: t('priority.low'), hint: t('priority.lowHint') },
    { score: 3, label: t('priority.medium'), hint: t('priority.mediumHint') },
    { score: 5, label: t('priority.critical'), hint: t('priority.criticalHint') },
];

function priorityTally(count, avg) {
    return count
        ? t('priority.tally', { n: count, avg: formatNumber(avg, 1) })
        : t('priority.noRatings');
}

function renderPriorityList(args) {
    const listEl = document.getElementById('priorityList');
    if (!listEl) return;

    if (!args.length) {
        listEl.innerHTML = `<p class="empty-message">${escapeHtml(t('priority.empty'))}</p>`;
        return;
    }

    // Most-contested first, but push claims this user already rated to the bottom
    // so there's always something new to weigh in on.
    const ordered = args.slice().sort((a, b) => {
        const aRated = a.importance?.my_score ? 1 : 0;
        const bRated = b.importance?.my_score ? 1 : 0;
        if (aRated !== bRated) return aRated - bRated;
        return (b.importance?.count || 0) - (a.importance?.count || 0);
    }).slice(0, 8);

    listEl.innerHTML = '';
    ordered.forEach((a) => {
        const mine = a.importance?.my_score ?? null;
        const count = a.importance?.count || 0;
        const avg = a.importance?.avg_score;

        const row = document.createElement('div');
        row.className = 'priority-row';
        row.dataset.argumentId = a.id;
        row.innerHTML = `
            <p class="priority-claim">${quotedClaimHtml(a.text)}</p>
            <div class="priority-controls" role="group" aria-label="${escapeHtml(t('priority.groupLabel'))}">
                ${PRIORITY_LEVELS.map((lvl) => `
                    <button type="button" class="priority-btn${mine === lvl.score ? ' is-selected' : ''}"
                            data-score="${lvl.score}" title="${escapeHtml(lvl.hint)}"
                            aria-pressed="${mine === lvl.score}">${escapeHtml(lvl.label)}</button>
                `).join('')}
            </div>
            <span class="priority-tally">${escapeHtml(priorityTally(count, avg))}</span>
        `;

        row.querySelectorAll('.priority-btn').forEach((btn) => {
            btn.addEventListener('click', () => submitImportanceRating(row, Number(btn.dataset.score)));
        });

        listEl.appendChild(row);
    });
}

// `toast`: report the outcome in a toast instead of the page's #priorityMessage
// (the home feed's vote cards, where a message at the top would be off-screen).
async function submitImportanceRating(row, score, { toast = false } = {}) {
    const messageEl = toast ? null : document.getElementById('priorityMessage');
    const showMessage = (text, type, html = false) => {
        if (!messageEl) return;
        if (html) messageEl.innerHTML = text;
        else messageEl.textContent = text;
        messageEl.className = `form-message ${type}`;
    };

    const argumentId = Number(row.dataset.argumentId);
    const buttons = Array.from(row.querySelectorAll('.priority-btn'));
    buttons.forEach((b) => { b.disabled = true; });

    try {
        const response = await fetch(`${API_BASE_URL}/importance-ratings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ argument_id: argumentId, score }),
        });
        const data = await response.json();

        if (!response.ok) {
            const signup = data.needs_account ? ` <a href="${authHref('register.html')}">${escapeHtml(t('common.createFreeAccount'))}</a>` : '';
            showMessage(escapeHtml(apiError(data, 'priority.error')) + signup, 'error', true);
            if (toast) {
                showToast(apiError(data, 'priority.error'), data.needs_account
                    ? { actionLabel: t('common.createAccount'), actionHref: authHref('register.html'), duration: 8000 } : {});
            }
            return;
        }

        buttons.forEach((b) => {
            const selected = Number(b.dataset.score) === score;
            b.classList.toggle('is-selected', selected);
            b.setAttribute('aria-pressed', String(selected));
        });

        const tally = row.querySelector('.priority-tally');
        const count = data.importance?.count || 0;
        const avg = data.importance?.avg_score;
        if (tally) {
            tally.textContent = priorityTally(count, avg);
        }
        if (data.pending) {
            pendingVotes = data.pending_count;
            showMessage(pendingVotesHtml(data.pending_count), 'success', true);
            if (toast) {
                showToast(t('votes.pendingToast', { n: data.pending_count }),
                    { actionLabel: t('common.createAccount'), actionHref: authHref('register.html'), duration: 8000 });
            }
        } else {
            showMessage(t('priority.thanks'), 'success');
            if (toast) showToast(t('priority.thanks'));
        }
    } catch (error) {
        console.error('Importance rating error:', error);
        showMessage(t('common.serverUnreachable'), 'error');
        if (toast) showToast(t('common.serverUnreachable'));
    } finally {
        buttons.forEach((b) => { b.disabled = false; });
    }
}

// ========================================
// Contribute: resource submission
// ========================================

function populateResourceClaimPicker(args) {
    const select = document.getElementById('resourceArgument');
    if (!select) return;

    args.slice()
        .sort((a, b) => String(a.text || '').localeCompare(String(b.text || ''), LANG))
        .forEach((a) => {
            const option = document.createElement('option');
            option.value = a.id;
            const text = String(a.text || '');
            option.textContent = text.length > 90 ? `${text.slice(0, 90).trimEnd()}…` : text;
            select.appendChild(option);
        });
}

function setupResourceForm() {
    const form = document.getElementById('resourceForm');
    if (!form) return;

    const messageEl = document.getElementById('resourceFormMessage');
    const submitBtn = document.getElementById('resourceSubmitBtn');
    const showMessage = (text, type) => {
        if (!messageEl) return;
        messageEl.textContent = text;
        messageEl.className = `form-message ${type}`;
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!currentUser) {
            showMessage(t('resourceForm.loginFirst'), 'error');
            return;
        }

        const url = document.getElementById('resourceUrl').value.trim();
        if (!url) {
            showMessage(t('resourceForm.needLink'), 'error');
            return;
        }
        if (!isWebLink(url)) {
            showMessage(t('composer.sourceInvalid'), 'error');
            return;
        }

        const argumentId = document.getElementById('resourceArgument').value;
        const payload = {
            url,
            title: document.getElementById('resourceTitle').value.trim(),
            description: document.getElementById('resourceDescription').value.trim(),
            argument_id: argumentId ? Number(argumentId) : null,
        };

        submitBtn.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/resources`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(payload),
            });
            const data = await response.json();

            if (!response.ok) {
                showMessage(apiError(data, 'resourceForm.error'), 'error');
                return;
            }

            form.reset();
            showMessage(t('resourceForm.success'), 'success');
        } catch (error) {
            console.error('Resource submission error:', error);
            showMessage(t('common.serverUnreachable'), 'error');
        } finally {
            submitBtn.disabled = false;
        }
    });
}

// ========================================
// Contribution card (profile pages)
// ========================================

function createContributionCard(contribution) {
    const card = document.createElement('div');
    card.className = 'counter-argument-card';

    const quoteText = (contribution.narrative || contribution.counter_argument || contribution.title || '').toString().trim();
    const narrativeText = (contribution.narrative || '').toString().trim();
    const counterArgumentText = (contribution.counter_argument || '').toString().trim();
    
    const tagsHTML = contribution.tags && contribution.tags.length > 0
        ? `<div class="card-tags">${contribution.tags.map(tag => 
            `<span class="tag">${escapeHtml(topicLabel(tag))}</span>`
          ).join('')}</div>`
        : '';
    
    const sourceHTML = safeHref(contribution.source)
        ? `<div class="card-source"><a href="${escapeHtml(safeHref(contribution.source))}" target="_blank" rel="noopener noreferrer">${escapeHtml(t('common.viewSource'))}</a></div>`
        : '';
    
    const dateHTML = contribution.date 
        ? `<div class="card-date">${escapeHtml(formatDate(contribution.date))}</div>`
        : '';

    const claimHTML = contribution.argument_id != null
        ? `<div class="card-source"><a href="argument.html?id=${encodeURIComponent(contribution.argument_id)}">${escapeHtml(t('contrib.viewClaim'))}</a></div>`
        : '';
    
    // Contributor info - show clickable link if user_id exists, otherwise "Anonymous contributor"
    const avatar = contribution.user_id
        ? userAvatarHtml(contribution.user_id, contribution.username, 'avatar-xs')
        : '';
    const contributorHTML = contribution.user_id && contribution.username
        ? `<a href="profile.html?id=${contribution.user_id}" class="contributor-link">${avatar}<span class="contributor-name">${escapeHtml(t('common.by', { name: contribution.username }))}</span></a>`
        : `<span class="anonymous-contributor">${escapeHtml(t('contrib.anonymous'))}</span>`;

    const detailsParts = [];
    if (narrativeText && quoteText !== narrativeText) {
        detailsParts.push(`<div class="contribution-detail"><strong>${escapeHtml(t('contrib.quote'))}</strong> ${escapeHtml(narrativeText)}</div>`);
    }
    if (counterArgumentText && quoteText !== counterArgumentText) {
        detailsParts.push(`<div class="contribution-detail"><strong>${escapeHtml(t('contrib.response'))}</strong> ${escapeHtml(counterArgumentText)}</div>`);
    }
    // A claim's title is just its text cut short: don't repeat it.
    if (contribution.title && contribution.kind !== 'claim') {
        detailsParts.push(`<div class="contribution-detail"><strong>${escapeHtml(t('contrib.title'))}</strong> ${escapeHtml(contribution.title)}</div>`);
    }
    
    // Kind chip so a claim the person added and an answer they wrote don't look alike.
    const kindMeta = { claim: TRACKER_KIND.claim, response: TRACKER_KIND.response }[contribution.kind];
    const kindHTML = kindMeta
        ? `<span class="needs-badge contribution-kind">${icon(kindMeta[0])} ${escapeHtml(t(kindMeta[1]))}</span>`
        : '';

    card.innerHTML = `
        ${kindHTML}
        <button type="button" class="contribution-quote" aria-expanded="false">
            <span class="contribution-quote-mark" aria-hidden="true">“</span>
            <span class="contribution-quote-text">${escapeHtml(quoteText)}</span>
            <span class="contribution-quote-mark" aria-hidden="true">”</span>
        </button>
        <div class="contribution-details" aria-hidden="true">
            ${detailsParts.length ? detailsParts.join('') : ''}
            ${tagsHTML}
            <div class="card-meta">
                <div class="card-contributor">${contributorHTML}</div>
                ${claimHTML}
                ${sourceHTML}
                ${dateHTML}
            </div>
        </div>
    `;

    const toggleBtn = card.querySelector('.contribution-quote');
    const detailsEl = card.querySelector('.contribution-details');
    if (toggleBtn && detailsEl) {
        const setExpanded = (expanded) => {
            card.classList.toggle('is-expanded', expanded);
            toggleBtn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
            detailsEl.setAttribute('aria-hidden', expanded ? 'false' : 'true');
        };

        toggleBtn.addEventListener('click', () => {
            const expanded = card.classList.contains('is-expanded');
            setExpanded(!expanded);
        });

        // Default collapsed
        setExpanded(false);
    }
    
    return card;
}

// ========================================
// Blog/articles Page (Resources)
// ========================================

let allBlogResources = [];

// ========================================
// Personalities Page (EVS personas)
// ========================================

// « Comment ces profils ont été construits »: opens inline at the end of the
// page. The text is fetched on first open; a failed fetch is retried on the next.
const METHODOLOGY_SRC = '/api/articles/methodologie_profils.md';
const METHODOLOGY_HASH = '#methodologie';

function setupMethodologySection() {
    const section = document.getElementById('methodologie');
    const toggle = document.getElementById('methodToggle');
    const panel = document.getElementById('methodPanel');
    const loadingEl = document.getElementById('methodLoading');
    const errorEl = document.getElementById('methodError');
    const bodyEl = document.getElementById('methodBody');
    if (!section || !toggle || !panel || !loadingEl || !errorEl || !bodyEl) return () => {};

    let loaded = false;
    let loading = false;

    async function load() {
        if (loaded || loading) return;
        loading = true;
        errorEl.hidden = true;
        loadingEl.hidden = false;
        try {
            const res = await fetch(METHODOLOGY_SRC, { headers: { 'Accept': 'text/markdown' } });
            if (!res.ok) throw new Error(`Failed to load methodology (${res.status})`);
            bodyEl.innerHTML = markdownToHtml(await res.text());
            loaded = true;
        } catch (error) {
            console.error(error);
            errorEl.textContent = t('personas.methodError');
            errorEl.hidden = false;
        } finally {
            loading = false;
            loadingEl.hidden = true;
        }
    }

    function setOpen(open) {
        toggle.setAttribute('aria-expanded', String(open));
        panel.hidden = !open;
        return open ? load() : Promise.resolve();
    }

    // Scrolls once the text is in: before that the page is too short to reach it.
    async function openFromHash() {
        if (window.location.hash !== METHODOLOGY_HASH) return;
        await setOpen(true);
        section.scrollIntoView();
    }

    toggle.addEventListener('click', () => setOpen(panel.hidden));
    window.addEventListener('hashchange', openFromHash);
    return openFromHash;
}

async function initPersonalitiesPage() {
    const openMethodologyFromHash = setupMethodologySection();
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const grid = document.getElementById('personalitiesGrid');
    const emptyMessage = document.getElementById('emptyPersonalitiesMessage');

    try {
        const response = await fetch(`${API_BASE_URL}/persona-archetypes`);
        if (!response.ok) {
            throw new Error('Failed to load personalities');
        }

        const all = await response.json();
        const allArr = Array.isArray(all) ? all : [];

        // Layer 1 — archetypes: the named, actionable profiles that pair a value
        // system with a persuasion approach.
        const archetypes = allArr.filter((p) => isActivePersona(p)
            && (p.layer === 'archetype' || (p.imported_from && p.imported_from.project === 'BonPoteJS')));
        renderArchetypesGrid(archetypes);

        if (loadingMessage) loadingMessage.style.display = 'none';
        if (errorMessage) errorMessage.style.display = 'none';

        // Population grounding: each profile's estimated share of the French
        // population (EVS 2017-2018), largest first.
        if (!grid) return;
        const shared = archetypes.filter((p) => Number.isFinite(Number(p.population_share_w)));
        if (!shared.length) {
            grid.style.display = 'none';
            if (emptyMessage) emptyMessage.style.display = 'block';
            return;
        }
        if (emptyMessage) emptyMessage.style.display = 'none';
        renderPopulationShares(grid, shared);
    } catch (error) {
        console.error('Error loading personalities:', error);
        if (loadingMessage) loadingMessage.style.display = 'none';
        if (errorMessage) {
            errorMessage.textContent = t('personas.loadError');
            errorMessage.style.display = 'block';
        }
    } finally {
        // After the cards are in, so the page no longer moves under the scroll.
        openMethodologyFromHash();
    }
}

// Population shares as a list of bars, one per profile, largest first.
function renderPopulationShares(container, personas) {
    const max = Math.max(...personas.map((p) => Number(p.population_share_w) || 0)) || 1;
    container.className = 'population-shares';
    container.style.display = '';
    container.innerHTML = personas
        .slice()
        .sort((a, b) => (Number(b.population_share_w) || 0) - (Number(a.population_share_w) || 0))
        .map((p) => {
            const share = Number(p.population_share_w) || 0;
            const color = p.accent_color ? `background:${escapeHtml(p.accent_color)}` : '';
            return `
                <a class="share-row" href="personality.html?id=${encodeURIComponent(p.id)}">
                    <span class="share-name">${escapeHtml(p.name || t('personas.fallbackName'))}</span>
                    <span class="share-bar"><span class="share-fill" style="width:${Math.round((share / max) * 100)}%;${color}"></span></span>
                    <span class="share-pct">${escapeHtml(formatPercent(share, 0))}</span>
                </a>`;
        }).join('');
}

// Display metadata for the three bipolar persuasion dimensions.
const DIMENSION_META = Object.fromEntries(
    ['epistemic_style', 'moral_foundations', 'threat_sensitivity'].map((k) => [k, {
        name: t(`dim.${k}.name`), left: t(`dim.${k}.left`), right: t(`dim.${k}.right`),
    }])
);

function renderArchetypesGrid(archetypes) {
    const grid = document.getElementById('archetypesGrid');
    if (!grid) return;
    grid.innerHTML = '';
    if (!archetypes.length) return;

    // Stable, meaningful order by id (1..5).
    archetypes
        .slice()
        .sort((a, b) => (a.id || 0) - (b.id || 0))
        .forEach((p) => grid.appendChild(createArchetypeCard(p)));
}

function bipolarBar(axisKey, dim) {
    const meta = DIMENSION_META[axisKey];
    if (!meta || !dim || typeof dim.score !== 'number') return '';
    // Map score in [-2, 2] to a marker position in [0, 100]%.
    const pct = Math.max(0, Math.min(100, ((dim.score + 2) / 4) * 100));
    const hint = dim.label ? escapeHtml(dim.label) : '';
    return `
        <div class="dim">
            <div class="dim-top">
                <span class="dim-name">${escapeHtml(meta.name)}</span>
            </div>
            <div class="dim-track"><span class="dim-marker" style="left: ${pct}%"></span></div>
            <div class="dim-poles"><span>${escapeHtml(meta.left)}</span><span>${escapeHtml(meta.right)}</span></div>
            ${hint ? `<p class="dim-hint">${hint}</p>` : ''}
        </div>
    `;
}

function createArchetypeCard(p) {
    // Concise, clickable summary. The full profile lives on personality.html.
    const card = document.createElement('a');
    card.className = 'archetype-card archetype-card-link';
    card.href = `personality.html?id=${encodeURIComponent(p.id)}`;
    if (p.accent_color) card.style.setProperty('--accent', p.accent_color);

    const emoji = p.icon_emoji ? escapeHtml(p.icon_emoji) : '🧑';
    const name = escapeHtml(p.name || t('personas.fallbackName'));
    const sublabel = p.short_label ? escapeHtml(p.short_label) : (p.description ? escapeHtml(p.description) : '');
    const values = p.value_system ? escapeHtml(p.value_system) : '';
    const engagement = p.engagement_stance ? escapeHtml(p.engagement_stance) : '';

    const persuasion = p.persuasion || {};
    const format = persuasion.format ? escapeHtml(formatLabel(persuasion.format)) : '';
    // Teaser: first sentence of the approach.
    const approach = String(persuasion.approach || '');
    const firstStop = approach.indexOf('. ');
    const teaser = firstStop > 0 ? approach.slice(0, firstStop + 1) : approach;

    card.innerHTML = `
        <header class="archetype-head">
            <span class="archetype-emoji" aria-hidden="true">${emoji}</span>
            <div class="archetype-title">
                <h4>${name}</h4>
                ${sublabel ? `<p class="archetype-sublabel">${sublabel}</p>` : ''}
            </div>
        </header>

        ${values ? `<p class="archetype-values"><span class="field-label">${escapeHtml(t('personas.valueSystem'))}</span>${values}</p>` : ''}

        <div class="archetype-tags">
            ${engagement ? `<span class="engagement-badge">${escapeHtml(t('personas.engagement', { value: p.engagement_stance }))}</span>` : ''}
            ${format ? `<span class="format-tag">${format}</span>` : ''}
        </div>

        ${teaser ? `<p class="approach-teaser"><strong>${icon('bulb')} ${escapeHtml(t('personas.howToReach'))}</strong> ${escapeHtml(teaser)}</p>` : ''}

        <span class="archetype-cta">${escapeHtml(t('personas.viewProfile'))}</span>
    `;

    return card;
}

// ========================================
// Personality detail (full page)
// ========================================

async function initPersonalityDetailPage() {
    const loading = document.getElementById('personaLoading');
    const errorEl = document.getElementById('personaError');
    const container = document.getElementById('personaProfile');

    const params = new URLSearchParams(window.location.search);
    const id = parseInt(params.get('id'), 10);
    if (!Number.isInteger(id)) {
        if (loading) loading.style.display = 'none';
        if (errorEl) { errorEl.textContent = t('persona.none'); errorEl.style.display = 'block'; }
        return;
    }

    try {
        // Claims are fetched alongside the persona so "What you'll hear" can show
        // the real ones aimed at them, ranked by community importance.
        const [res, argsRes] = await Promise.all([
            fetch(`${API_BASE_URL}/persona-archetypes`),
            fetch(`${API_BASE_URL}/arguments`),
        ]);
        if (!res.ok) throw new Error('Failed to load personalities');
        const all = await res.json();
        const list = Array.isArray(all) ? all : [];
        // Old profile links (retired groups) open the profile that replaced them.
        const newId = replacementPersonaId(list, id);
        if (newId != null) {
            params.set('id', newId);
            window.location.replace(`personality.html?${params}${window.location.hash}`);
            return;
        }
        const p = list.find((x) => isActivePersona(x) && x.id === id);

        const allArgs = argsRes.ok ? await argsRes.json() : [];
        const personaArgs = (Array.isArray(allArgs) ? allArgs : [])
            .filter((a) => (a.personas || []).some((x) => x && x.id === id));

        if (loading) loading.style.display = 'none';
        if (!p) {
            if (errorEl) { errorEl.textContent = t('persona.notFound'); errorEl.style.display = 'block'; }
            return;
        }

        document.title = t('site.pageTitle', { page: p.name || t('personas.fallbackName') });
        if (container) {
            container.style.display = 'block';
            renderPersonaProfile(p, container, personaArgs);
            const methodLink = document.getElementById('personaMethodLink');
            if (methodLink) methodLink.hidden = false;
        }

        // Prev/next among the named archetypes only.
        const archetypes = list
            .filter((x) => isActivePersona(x) && (x.layer === 'archetype' || (x.imported_from && x.imported_from.project === 'BonPoteJS')))
            .sort((a, b) => (a.id || 0) - (b.id || 0));
        renderPersonaNav(archetypes, id);
    } catch (error) {
        console.error('Error loading personality:', error);
        if (loading) loading.style.display = 'none';
        if (errorEl) { errorEl.textContent = t('persona.loadError'); errorEl.style.display = 'block'; }
    }
}

// Rank a persona's claims by how urgent the community says they are, falling
// back to activity so unrated claims still order sensibly rather than at random.
function rankByImportance(args) {
    return args.slice().sort((a, b) => {
        const ia = a.importance || {};
        const ib = b.importance || {};
        const avgA = typeof ia.avg_score === 'number' ? ia.avg_score : -1;
        const avgB = typeof ib.avg_score === 'number' ? ib.avg_score : -1;
        if (avgA !== avgB) return avgB - avgA;
        if ((ib.count || 0) !== (ia.count || 0)) return (ib.count || 0) - (ia.count || 0);
        return (b.contribution_count || 0) - (a.contribution_count || 0);
    });
}

function personaClaimsSection(persona, args) {
    const top = rankByImportance(args).slice(0, 3);
    const seeMoreHref = `index.html?persona=${encodeURIComponent(persona.id)}`;

    if (!top.length) {
        return `
            <section class="persona-section">
                <h3>${escapeHtml(t('persona.claimsTitle'))}</h3>
                <p class="empty-message">${escapeHtml(t('persona.claimsEmpty'))}</p>
            </section>`;
    }

    const cards = top.map((a) => {
        const importance = a.importance || {};
        const count = importance.count || 0;
        const avg = importance.avg_score;
        const badge = (count && typeof avg === 'number')
            ? (() => {
                const level = importanceLevel(avg);
                const avgText = formatNumber(avg, 1);
                return `<span class="needs-badge feed-importance ${level.className}" title="${escapeHtml(t('importance.title', { avg: avgText, n: count }))}">${icon('target')} ${escapeHtml(level.label)} · ${avgText}/5</span>`;
            })()
            : `<span class="needs-badge feed-importance feed-importance-unrated" title="${escapeHtml(t('importance.unratedTitle'))}">${icon('target')} ${escapeHtml(t('importance.unrated'))}</span>`;

        const contributions = a.contribution_count || 0;
        return `
            <a class="needs-card" href="argument.html?id=${encodeURIComponent(a.id)}">
                <p class="needs-quote">${quotedClaimHtml(a.text)}</p>
                <div class="needs-meta">
                    <span class="needs-badge" title="${escapeHtml(t('importance.contributionsTitle'))}">${icon('layers')} ${escapeHtml(t('importance.contributions', { n: contributions }))}</span>
                    ${badge}
                </div>
            </a>`;
    }).join('');

    return `
        <section class="persona-section">
            <div class="persona-claims-head">
                <h3>${escapeHtml(t('persona.claimsTitle'))}</h3>
                <a class="persona-claims-more" href="${seeMoreHref}">${escapeHtml(t('persona.claimsAll', { n: args.length }))}</a>
            </div>
            <p class="persona-claims-sub">${escapeHtml(t('persona.claimsLead'))}</p>
            <div class="persona-claims">${cards}</div>
        </section>`;
}

function renderPersonaProfile(p, container, personaArgs = []) {
    if (p.accent_color) container.style.setProperty('--accent', p.accent_color);

    const emoji = p.icon_emoji ? escapeHtml(p.icon_emoji) : '🧑';
    const name = escapeHtml(p.name || t('personas.fallbackName'));
    const sublabel = p.short_label ? escapeHtml(p.short_label) : (p.description ? escapeHtml(p.description) : '');
    const values = p.value_system ? escapeHtml(p.value_system) : '';

    const dims = p.dimensions || {};
    const dimsHtml = ['epistemic_style', 'moral_foundations', 'threat_sensitivity']
        .map((k) => bipolarBar(k, dims[k]))
        .join('');

    const engagement = p.engagement_stance ? escapeHtml(p.engagement_stance) : '';
    const anchoring = p.social_anchoring ? escapeHtml(p.social_anchoring) : '';

    const persuasion = p.persuasion || {};
    const approach = persuasion.approach ? escapeHtml(persuasion.approach) : '';
    const format = persuasion.format ? escapeHtml(formatLabel(persuasion.format)) : '';
    const doList = Array.isArray(persuasion.do) ? persuasion.do : [];
    const dontList = Array.isArray(persuasion.dont) ? persuasion.dont : [];
    const messengers = Array.isArray(persuasion.messengers) ? persuasion.messengers : [];

    const listHtml = (items) => items.map((x) => `<li>${escapeHtml(x)}</li>`).join('');
    const chipHtml = (items) => items.map((x) => `<span class="mini-chip">${escapeHtml(x)}</span>`).join('');

    // EVS-cluster fields (population grounding layer), shown when present.
    const share = Number(p.population_share_w);
    const sharePct = Number.isFinite(share) ? formatPercent(share, 0) : '';
    const axesLabels = Array.isArray(p.top_axes_labels) ? p.top_axes_labels.map((x) => escapeHtml(String(x))) : [];

    container.innerHTML = `
        <div class="persona-hero">
            <div class="persona-hero-main">
                <span class="persona-hero-emoji" aria-hidden="true">${emoji}</span>
                <div>
                    <h2>${name}</h2>
                    ${sublabel ? `<p class="persona-hero-sub">${sublabel}</p>` : ''}
                </div>
            </div>
            ${values ? `<p class="persona-hero-values">${values}</p>` : ''}
            ${sharePct ? `<p class="persona-hero-share">${escapeHtml(t('persona.share', { pct: sharePct }))}</p>` : ''}
        </div>

        <div class="persona-body">
            ${dimsHtml ? `
            <section class="persona-section">
                <h3>${escapeHtml(t('persona.defines'))}</h3>
                <div class="archetype-dims">${dimsHtml}</div>
                <div class="archetype-tags persona-tags">
                    ${engagement ? `<span class="engagement-badge">${escapeHtml(t('personas.engagement', { value: p.engagement_stance }))}</span>` : ''}
                    ${anchoring ? `<span class="anchoring-note"><strong>${escapeHtml(t('persona.messengers'))}</strong> ${anchoring}</span>` : ''}
                </div>
            </section>` : ''}

            ${approach ? `
            <section class="persona-section persuasion-panel">
                <h3>${icon('bulb')} ${escapeHtml(t('persona.howToChange'))}${format ? ` <span class="format-tag">${format}</span>` : ''}</h3>
                <p class="approach">${approach}</p>
                ${(doList.length || dontList.length) ? `
                <div class="do-dont">
                    ${doList.length ? `<div class="do-col"><span class="do-label">${icon('check')} ${escapeHtml(t('persona.do'))}</span><ul>${listHtml(doList)}</ul></div>` : ''}
                    ${dontList.length ? `<div class="dont-col"><span class="dont-label">${icon('ban')} ${escapeHtml(t('persona.avoid'))}</span><ul>${listHtml(dontList)}</ul></div>` : ''}
                </div>` : ''}
                ${messengers.length ? `<div class="messengers-row"><span class="field-label">${escapeHtml(t('persona.whoToBring'))}</span>${chipHtml(messengers)}</div>` : ''}
            </section>` : ''}

            ${personaClaimsSection(p, personaArgs)}

            ${axesLabels.length ? `
            <section class="persona-section">
                <h3>${escapeHtml(t('persona.axes'))}</h3>
                <div class="messengers-row">${axesLabels.map((a) => `<span class="mini-chip">${a}</span>`).join('')}</div>
            </section>` : ''}
        </div>
    `;
}

function renderPersonaNav(archetypes, currentId) {
    const nav = document.getElementById('personaNav');
    if (!nav) return;
    const idx = archetypes.findIndex((x) => x.id === currentId);
    if (idx === -1) { nav.style.display = 'none'; return; }

    const prev = idx > 0 ? archetypes[idx - 1] : null;
    const next = idx < archetypes.length - 1 ? archetypes[idx + 1] : null;

    const link = (persona, dir) => {
        if (!persona) return '<span></span>';
        const label = escapeHtml(persona.name || t('personas.fallbackName'));
        const arrow = dir === 'prev' ? '←' : '→';
        return `<a class="persona-nav-link persona-nav-${dir}" href="personality.html?id=${encodeURIComponent(persona.id)}">
            <span class="persona-nav-dir">${arrow} ${escapeHtml(t(dir === 'prev' ? 'persona.previous' : 'persona.next'))}</span>
            <span class="persona-nav-name">${label}</span>
        </a>`;
    };

    nav.innerHTML = link(prev, 'prev') + link(next, 'next');
    nav.style.display = 'flex';
}

// ========================================
// Shared: toast notifications
// ========================================

let toastTimer = null;

// `actions`: several links ([{ label, href }]), e.g. « Se connecter » and « Créer un compte ».
function showToast(message, { actionLabel = null, actionHref = null, actions = null, duration = 4000 } = {}) {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.className = 'toast';
        toast.setAttribute('role', 'status');
        toast.setAttribute('aria-live', 'polite');
        document.body.appendChild(toast);
    }

    const links = (actions || (actionLabel && actionHref ? [{ label: actionLabel, href: actionHref }] : []))
        .filter((a) => a && a.label && (a.href || a.onClick));
    const actionHtml = links.length
        ? ` <span class="toast-actions">${links.map((a, i) => (a.onClick
            ? `<button type="button" class="toast-action" data-action="${i}">${escapeHtml(a.label)}</button>`
            : `<a class="toast-action" href="${escapeHtml(a.href)}">${escapeHtml(a.label)}</a>`)).join('')}</span>`
        : '';
    toast.innerHTML = `<span>${escapeHtml(message)}</span>${actionHtml}`;
    toast.querySelectorAll('button[data-action]').forEach((btn) => {
        btn.addEventListener('click', () => {
            toast.classList.remove('is-visible');
            links[Number(btn.dataset.action)].onClick();
        });
    });
    toast.classList.add('is-visible');

    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('is-visible'), duration);
}

// ========================================
// Filter bar — one consistent controls component shared by the Home feed, the
// Resources library and the Blog list. Renders multi-select facet dropdowns, a
// sort dropdown, a search field, and a removable-chips row. It owns only UI
// state; each page owns the actual filtering via the onChange callback.
//
//   createFilterBar(mountEl, {
//     facets: [{ id, label, icon, options: [{value,label,count?}], selected: [] }],
//     sort:   { label, value, options: [{value,label}] },   // optional
//     search: { placeholder, value },                        // optional
//     onChange: (state) => {}   // state = { facets: {id:[values]}, sort, q }
//   }) -> { getState, setFacetOptions(id, options), clearAll }
// ========================================

const CHIPS_PER_FACET = 5;

const SEARCH_ICON_SVG = `
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M11 19a8 8 0 1 1 0-16 8 8 0 0 1 0 16Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M21 21l-4.35-4.35" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`;

function createFilterBar(mount, spec) {
    if (!mount) return null;

    const facets = (spec.facets || []).map((f) => ({
        id: f.id,
        label: f.label,
        icon: f.icon || '',
        options: f.options || [],
        selected: new Set((f.selected || []).map(String)),
        searchable: Boolean(f.searchable),
        searchPlaceholder: f.searchPlaceholder || t('filter.search'),
        query: '',
    }));
    const facetById = new Map(facets.map((f) => [f.id, f]));

    const sortSpec = spec.sort || null;
    let sortValue = sortSpec ? sortSpec.value : null;
    let searchValue = spec.search ? (spec.search.value || '') : '';

    mount.innerHTML = `
        <div class="filterbar">
            <div class="filterbar-facets"></div>
            <div class="filterbar-tail">
                ${sortSpec ? `
                <label class="filterbar-sort">
                    <span>${escapeHtml(sortSpec.label || t('filter.sort'))}</span>
                    <select></select>
                </label>` : ''}
                ${spec.search ? `
                <div class="filterbar-search">
                    <span class="search-icon" aria-hidden="true">${SEARCH_ICON_SVG}</span>
                    <input type="search" class="search-bar filterbar-input"
                           placeholder="${escapeHtml(spec.search.placeholder || t('filter.search'))}"
                           aria-label="${escapeHtml(spec.search.placeholder || t('filter.search'))}">
                </div>` : ''}
            </div>
        </div>
        <div class="filterbar-chips" aria-live="polite"></div>
    `;

    const facetsEl = mount.querySelector('.filterbar-facets');
    const chipsEl = mount.querySelector('.filterbar-chips');
    const sortEl = mount.querySelector('.filterbar-sort select');
    const searchEl = mount.querySelector('.filterbar-input');
    let openId = null;

    const getState = () => ({
        facets: Object.fromEntries(facets.map((f) => [f.id, [...f.selected]])),
        sort: sortValue,
        q: searchValue.trim(),
    });
    const emit = () => { if (typeof spec.onChange === 'function') spec.onChange(getState()); };
    const optionLabel = (facet, value) => {
        const opt = facet.options.find((o) => String(o.value) === String(value));
        return opt ? personaEmojiHtml(opt.emoji) + escapeHtml(opt.label) : escapeHtml(String(value));
    };
    const anyActive = () => facets.some((f) => f.selected.size) || !!searchValue.trim();

    // ---- facet buttons ----
    facets.forEach((facet) => {
        const group = document.createElement('div');
        group.className = 'filterbar-facet';
        group.dataset.facet = facet.id;
        group.innerHTML = `
            <button type="button" class="filterbar-btn" aria-haspopup="true" aria-expanded="false"></button>
            <div class="filterbar-menu" role="menu" hidden></div>`;
        facet._group = group;
        facet._btn = group.querySelector('.filterbar-btn');
        facet._menu = group.querySelector('.filterbar-menu');
        facet._btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (openId === facet.id) closeMenus();
            else { closeMenus(); openMenu(facet); }
        });
        renderButton(facet);
        facetsEl.appendChild(group);
    });

    function renderButton(facet) {
        const count = facet.selected.size ? `<span class="filterbar-count">${facet.selected.size}</span>` : '';
        facet._btn.innerHTML =
            `${facet.icon ? `<span aria-hidden="true">${facet.icon}</span> ` : ''}` +
            `${escapeHtml(facet.label)} ${count}<span class="filterbar-caret" aria-hidden="true">▾</span>`;
        facet._group.classList.toggle('has-selection', facet.selected.size > 0);
    }

    function openMenu(facet) {
        renderMenu(facet);
        facet._menu.hidden = false;
        facet._btn.setAttribute('aria-expanded', 'true');
        facet._group.classList.add('is-open');
        openId = facet.id;
        // No autofocus on touch screens: it would pop the keyboard over the list.
        if (facet.searchable && window.matchMedia('(hover: hover)').matches) {
            facet._menu.querySelector('.filterbar-menu-search')?.focus();
        }
    }

    function closeMenus() {
        facets.forEach((f) => {
            f._menu.hidden = true;
            f._btn.setAttribute('aria-expanded', 'false');
            f._group.classList.remove('is-open');
            f.query = '';
            f._menu.innerHTML = '';
        });
        openId = null;
    }

    function renderMenu(facet) {
        const menu = facet._menu;
        if (!facet.options.length) {
            menu.innerHTML = `<p class="filterbar-menu-empty">${escapeHtml(t('filter.nothing'))}</p>`;
            return;
        }
        // The search box and buttons stay in place while the list below is redrawn.
        if (!menu.querySelector('.filterbar-options')) {
            menu.classList.toggle('has-search', facet.searchable);
            menu.innerHTML = (facet.searchable ? `
                <div class="filterbar-menu-head">
                    <input type="search" class="filterbar-menu-search" placeholder="${escapeHtml(facet.searchPlaceholder)}" aria-label="${escapeHtml(facet.searchPlaceholder)}" autocomplete="off">
                    <div class="filterbar-menu-actions">
                        <button type="button" data-select-all>${escapeHtml(t('filter.selectAll'))}</button>
                        <button type="button" data-clear-selection>${escapeHtml(t('filter.clearAll'))}</button>
                    </div>
                </div>` : '') + '<div class="filterbar-options"></div>';
            if (facet.searchable) wireMenuHead(facet);
        }
        renderOptions(facet);
    }

    function visibleOptions(facet) {
        const q = normalizeSearch(facet.query);
        return q ? facet.options.filter((o) => normalizeSearch(o.label).includes(q)) : facet.options;
    }

    function renderOptions(facet) {
        const list = facet._menu.querySelector('.filterbar-options');
        const scrollTop = list.scrollTop;
        const shown = visibleOptions(facet);
        list.innerHTML = shown.length ? shown.map((o) => {
            const checked = facet.selected.has(String(o.value)) ? 'checked' : '';
            const c = (typeof o.count === 'number') ? `<span class="filterbar-opt-count">${o.count}</span>` : '';
            return `<label class="filterbar-option"><input type="checkbox" value="${escapeHtml(String(o.value))}" ${checked}><span>${personaEmojiHtml(o.emoji)}${escapeHtml(o.label)}</span>${c}</label>`;
        }).join('') : `<p class="filterbar-menu-empty">${escapeHtml(t('filter.noMatch'))}</p>`;
        list.scrollTop = scrollTop;

        list.querySelectorAll('input[type="checkbox"]').forEach((box) => {
            box.addEventListener('change', () => {
                if (box.checked) facet.selected.add(box.value);
                else facet.selected.delete(box.value);
                renderButton(facet);   // refresh count badge in place (menu stays open)
                renderChips();
                emit();
            });
        });
    }

    // « Tout sélectionner » takes what the search left visible; « Tout effacer » empties the whole facet.
    function wireMenuHead(facet) {
        const menu = facet._menu;
        menu.querySelector('.filterbar-menu-search').addEventListener('input', (e) => {
            facet.query = e.target.value;
            renderOptions(facet);
        });
        const applied = () => {
            renderButton(facet);
            renderOptions(facet);
            renderChips();
            emit();
        };
        menu.querySelector('[data-select-all]').addEventListener('click', () => {
            visibleOptions(facet).forEach((o) => facet.selected.add(String(o.value)));
            applied();
        });
        menu.querySelector('[data-clear-selection]').addEventListener('click', () => {
            facet.selected.clear();
            applied();
        });
    }

    // ---- chips ----
    function renderChips() {
        const pills = [];
        facets.forEach((facet) => {
            // Many picks (« Tout sélectionner ») read better as one chip.
            if (facet.selected.size > CHIPS_PER_FACET) {
                pills.push(`<button type="button" class="filterbar-chip" data-facet="${escapeHtml(facet.id)}" data-clear-facet="1">${escapeHtml(t('filter.chipLabel', { label: facet.label }))} ${escapeHtml(t('filter.selectedCount', { n: facet.selected.size }))} ✕</button>`);
                return;
            }
            facet.selected.forEach((v) => {
                pills.push(`<button type="button" class="filterbar-chip" data-facet="${escapeHtml(facet.id)}" data-value="${escapeHtml(String(v))}">${escapeHtml(t('filter.chipLabel', { label: facet.label }))} ${optionLabel(facet, v)} ✕</button>`);
            });
        });
        if (!pills.length && !searchValue.trim()) { chipsEl.innerHTML = ''; return; }
        chipsEl.innerHTML = pills.join('') +
            `<button type="button" class="filterbar-chip filterbar-chip-clear" data-clear="all">${escapeHtml(t('filter.clearAll'))}</button>`;

        chipsEl.querySelectorAll('[data-value]').forEach((btn) => {
            btn.addEventListener('click', () => {
                const facet = facetById.get(btn.dataset.facet);
                if (!facet) return;
                facet.selected.delete(btn.dataset.value);
                renderButton(facet);
                if (openId === facet.id) renderMenu(facet);
                renderChips();
                emit();
            });
        });
        chipsEl.querySelectorAll('[data-clear-facet]').forEach((btn) => {
            btn.addEventListener('click', () => {
                const facet = facetById.get(btn.dataset.facet);
                if (!facet) return;
                facet.selected.clear();
                renderButton(facet);
                if (openId === facet.id) renderMenu(facet);
                renderChips();
                emit();
            });
        });
        chipsEl.querySelector('[data-clear]')?.addEventListener('click', clearAll);
    }

    function clearAll() {
        facets.forEach((f) => { f.selected.clear(); renderButton(f); if (openId === f.id) renderMenu(f); });
        searchValue = '';
        if (searchEl) searchEl.value = '';
        renderChips();
        emit();
    }

    // ---- sort ----
    if (sortEl && sortSpec) {
        sortEl.innerHTML = (sortSpec.options || [])
            .map((o) => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`).join('');
        sortEl.value = sortValue;
        sortEl.addEventListener('change', () => { sortValue = sortEl.value; emit(); });
    }

    // ---- search ----
    if (searchEl) {
        searchEl.value = searchValue;
        let debounce;
        searchEl.addEventListener('input', () => {
            clearTimeout(debounce);
            debounce = setTimeout(() => { searchValue = searchEl.value; renderChips(); emit(); }, 200);
        });
    }

    // Close menus on click-away / Escape.
    document.addEventListener('click', (e) => {
        if (openId !== null && !mount.contains(e.target)) closeMenus();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && openId !== null) closeMenus();
    });

    renderChips();

    return {
        getState,
        clearAll,
        setFacetOptions(id, options) {
            const facet = facetById.get(id);
            if (!facet) return;
            facet.options = options || [];
            if (openId === id) renderMenu(facet);
            renderChips(); // labels may depend on options
        },
        // Programmatically set a facet's selection (e.g. a card click), keeping
        // the bar the single source of truth and notifying the page.
        setSelected(id, values) {
            const facet = facetById.get(id);
            if (!facet) return;
            facet.selected = new Set((values || []).map(String));
            renderButton(facet);
            if (openId === id) renderMenu(facet);
            renderChips();
            emit();
        },
    };
}

// ========================================
// Respond: narrative library (list of arguments)
// ========================================

let allArguments = [];
let feedSort = 'responses';
// "Named" personas are the conversational caricatures. The numbered EVS clusters
// ("01"…"10") are analysis placeholders and must never be offered in the UI.
function isNamedPersona(p) {
    const name = String(p?.name || '').trim();
    return isActivePersona(p) && Boolean(name) && !/^\d+$/.test(name);
}

// A retired profile's replacement: its `alias_of`, or the active profile that lists
// it in `aliases`. null when `id` is active or unknown.
function replacementPersonaId(personas, id) {
    const list = Array.isArray(personas) ? personas : [];
    if (list.some((p) => p.id === id && isActivePersona(p))) return null;
    const old = list.find((p) => p.id === id);
    const target = list.find((p) => isActivePersona(p) && (p.id === old?.alias_of || (p.aliases || []).includes(id)));
    return target ? target.id : null;
}

// Personas can be retired (e.g. the old EVS groups 01–10); never show those.
function isActivePersona(p) {
    return Boolean(p) && p.status !== 'retired';
}

let feedQuery = '';
let feedTopics = new Set();
// Persona filter: a set of ids, so several personalities can be combined.
// Seeded from ?persona=<id>[,<id>…] — the link personality pages use to show
// all their claims — and kept in the URL so a filtered feed stays shareable.
let feedPersonaIds = new Set();
let feedPersonaList = [];

const FEED_PAGE_SIZE = 12;
let feedShown = FEED_PAGE_SIZE;
// Persona id → accent colour (from the archetype list), for card avatars and chips.
let feedPersonaColors = new Map();
// The named archetypes (the five people a claim can be answered for).
let feedArchetypes = [];

// Two ways to use the home feed: 'find' an answer (the normal feed, filtered by
// who you're talking to) or 'help' out (compact cards built around one task).
// Both are kept in the URL: ?mode=help&task=vote|write|rate.
let feedMode = 'find';
let helpTask = 'vote';

// Help-out tasks. `test` says whether a claim needs this help; `signal` is the
// line above the claim; `action` is the card's one button.
const HELP_TASKS = {
    vote: {
        icon: 'target',
        test: (a) => a.importance?.my_score == null,
        signal: (a) => (a.importance?.count
            ? priorityTally(a.importance.count, a.importance.avg_score)
            : t('help.vote.none')),
    },
    write: {
        icon: 'users',
        test: (a) => missingArchetypes(a).length > 0,
        signal: (a) => {
            const missing = missingArchetypes(a);
            const names = missing.slice(0, 2).map((p) => firstName(p.name)).join(', ');
            return t('help.write.signal', { names: missing.length > 2 ? `${names} +${missing.length - 2}` : names });
        },
    },
    rate: {
        icon: 'star',
        test: (a) => Boolean(a.top_response?.text) && !a.rated,
        signal: () => t('help.rate.signal'),
    },
};

// People with no tailored answer on this claim. The list is rotated by claim id
// so suggestions spread across people rather than always naming the same one.
function missingArchetypes(a) {
    const covered = new Set((a.personas || []).map((p) => p && p.id));
    const missing = feedArchetypes.filter((p) => !covered.has(p.id));
    const k = missing.length ? (Number(a.id) || 0) % missing.length : 0;
    const rotated = missing.slice(k).concat(missing.slice(0, k));
    // People picked in « Trouver une réponse » come first (e.g. coming from « Pas encore de réponse pour Max »).
    return rotated.filter((p) => feedPersonaIds.has(p.id)).concat(rotated.filter((p) => !feedPersonaIds.has(p.id)));
}

function firstName(name) {
    return String(name || '').trim().split(/\s+/)[0] || '';
}

async function initRespondPage() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const params = new URLSearchParams(window.location.search);

    feedPersonaIds = new Set(
        (params.get('persona') || '')
            .split(',')
            .map((s) => parseInt(s, 10))
            .filter(Number.isInteger)
    );
    feedQuery = params.get('q') || '';
    feedTopics = new Set((params.get('topic') || '').split(',').filter(Boolean));
    feedMode = params.get('mode') === 'help' ? 'help' : 'find';
    if (HELP_TASKS[params.get('task')]) helpTask = params.get('task');

    setupFeedSearch();
    setupModeSwitch();
    setupHelpMore();
    syncModeUi();
    document.getElementById('feedMore')?.addEventListener('click', () => {
        feedShown += FEED_PAGE_SIZE;
        renderArgumentsList();
    });

    try {
        // Credentials so the API can say which claims this visitor already voted on.
        const response = await fetch(`${API_BASE_URL}/arguments`, { credentials: 'include' });
        if (!response.ok) throw new Error('Failed to load narratives');
        allArguments = await response.json();
        if (!Array.isArray(allArguments)) allArguments = [];

        // Full persona roster: every persona that appears on a claim, deduped,
        // plus the archetype list for names and accent colours.
        const byId = new Map();
        allArguments.forEach((a) => (a.personas || []).forEach((x) => {
            if (x && Number.isInteger(x.id) && !byId.has(x.id)) byId.set(x.id, { id: x.id, name: x.name, emoji: x.emoji });
        }));
        let allPersonas = [];
        try {
            const personaRes = await fetch(`${API_BASE_URL}/persona-archetypes`);
            const personas = personaRes.ok ? await personaRes.json() : [];
            allPersonas = Array.isArray(personas) ? personas : [];
            allPersonas.forEach((p) => {
                if (!isActivePersona(p) || !Number.isInteger(p.id)) return;
                if (p.accent_color) feedPersonaColors.set(p.id, p.accent_color);
                if (p.layer === 'archetype' && isNamedPersona(p)) feedArchetypes.push({ id: p.id, name: p.name });
                if (!byId.has(p.id)) byId.set(p.id, { id: p.id, name: p.name, emoji: p.icon_emoji });
            });
        } catch (_) { /* roster still works from claims alone */ }
        feedPersonaList = Array.from(byId.values())
            .sort((a, b) => String(a.name || '').localeCompare(String(b.name || ''), LANG));

        // Old profile ids in ?persona= (retired groups) become the profile that replaced them;
        // then drop any id that matches no known persona.
        const before = [...feedPersonaIds].join(',');
        feedPersonaIds = new Set([...feedPersonaIds]
            .map((id) => replacementPersonaId(allPersonas, id) ?? id)
            .filter((id) => byId.has(id)));
        if ([...feedPersonaIds].join(',') !== before) syncFeedUrl();

        if (loadingMessage) loadingMessage.style.display = 'none';
        renderFeedStats();

        // Offer every named persona (the archetypes), even before they have an
        // answer, plus any other named persona that appears on a claim.
        const personasWithClaims = new Set(allArguments.flatMap((a) => (a.personas || []).map((x) => x && x.id)));
        const archetypeIds = new Set(feedArchetypes.map((p) => p.id));
        renderPersonaChips(feedPersonaList.filter((p) => isNamedPersona(p)
            && (archetypeIds.has(p.id) || personasWithClaims.has(p.id))));

        renderTopicChips(topicsByUse(allArguments));

        renderArgumentsList();
    } catch (error) {
        console.error('Error loading narratives:', error);
        if (loadingMessage) loadingMessage.style.display = 'none';
        document.getElementById('feedStats')?.style.setProperty('display', 'none');
        if (errorMessage) {
            errorMessage.textContent = t('feed.loadError');
            errorMessage.style.display = 'block';
        }
    }
}

// The header search filters the feed live on the home page (elsewhere it
// submits to index.html?q=…).
function setupFeedSearch() {
    const input = document.getElementById('siteSearch');
    if (!input) return;
    input.value = feedQuery;
    input.form?.addEventListener('submit', (e) => e.preventDefault());
    let debounce;
    input.addEventListener('input', () => {
        clearTimeout(debounce);
        debounce = setTimeout(() => {
            feedQuery = input.value;
            refreshFeed();
        }, 150);
    });
}

// Filter pills: clicking a pill adds it to the selection, clicking it again
// removes it; the first pill ("Everyone" / "All topics") clears its row.
// Several people (or topics) combine with OR; people and topics with AND.
let feedPersonaChoices = [];
let feedTopicChoices = [];

function renderPersonaChips(personas) {
    feedPersonaChoices = personas;
    renderFeedFilters();
}

// Every topic on these claims, most used first.
function topicsByUse(claims) {
    const counts = new Map();
    claims.forEach((a) => (a.tags || []).forEach((tag) => tag && counts.set(tag, (counts.get(tag) || 0) + 1)));
    return [...counts.entries()].sort((x, y) => y[1] - x[1]).map(([tag]) => tag);
}

function renderTopicChips(topics) {
    feedTopicChoices = topics;
    renderFeedFilters();
}

// Key figures above the feed: totals, so they stay put while filters change.
function renderFeedStats() {
    const el = document.getElementById('feedStats');
    if (!el) return;
    const answers = allArguments.reduce((sum, a) => sum + (a.counter_argument_count || 0), 0);
    el.innerHTML = statTilesHtml([
        { n: allArguments.length, label: t('home.statClaims', { n: allArguments.length }) },
        { n: answers, label: t('home.statAnswers', { n: answers }) },
        { n: feedArchetypes.length, label: t('home.statProfiles', { n: feedArchetypes.length }) },
    ].filter((tile) => tile.n > 0));
}

function setupModeSwitch() {
    const sw = document.getElementById('modeSwitch');
    if (!sw) return;
    sw.addEventListener('click', (e) => {
        const btn = e.target.closest('[data-mode]');
        if (!btn || btn.dataset.mode === feedMode) return;
        feedMode = btn.dataset.mode;
        syncModeUi();
        refreshFeed();
    });
}

// Show the controls of the current mode and mark the active switch button.
function syncModeUi() {
    const help = feedMode === 'help';
    document.querySelectorAll('#modeSwitch [data-mode]').forEach((b) => {
        b.setAttribute('aria-pressed', String(b.dataset.mode === feedMode));
    });
    const find = document.getElementById('findControls');
    const helpEl = document.getElementById('helpControls');
    const more = document.getElementById('helpMore');
    if (find) find.hidden = help;
    if (helpEl) helpEl.hidden = !help;
    if (more) more.hidden = !help;
    document.getElementById('argumentsList')?.classList.toggle('is-help', help);
}

// « Plus de façons de contribuer → » opens the profile. Visitors have no profile yet:
// say what an account adds, with « Créer un compte » / « Se connecter ».
function setupHelpMore() {
    const el = document.getElementById('helpMore');
    if (!el || currentUser) return;
    const next = 'profile.html#contributeSection';
    el.classList.add('help-more-guest');
    el.innerHTML = `
        <span>${escapeHtml(t('help.moreGuest'))}</span>
        <span class="help-more-actions">
            <a class="btn btn-primary btn-small" href="register.html?next=${encodeURIComponent(next)}">${escapeHtml(t('common.createAccount'))}</a>
            <a class="btn btn-secondary btn-small" href="login.html?next=${encodeURIComponent(next)}">${escapeHtml(t('common.login'))}</a>
        </span>`;
}

// Help-out task picker: one button per task, with how many claims need it.
function renderHelpTasks() {
    const el = document.getElementById('helpTasks');
    if (!el || feedMode !== 'help') return;
    const pool = filteredFeedItems({ ignorePersonas: true });
    el.innerHTML = Object.entries(HELP_TASKS).map(([id, task]) => `
        <button type="button" class="help-task" data-task="${id}" aria-pressed="${helpTask === id}">
            <span class="help-task-title">${icon(task.icon)} ${escapeHtml(t(`help.${id}.title`))}</span>
            <span class="help-task-sub">${escapeHtml(t(`help.${id}.sub`))}</span>
            <span class="help-task-n">${escapeHtml(t('help.count', { n: pool.filter(task.test).length }))}</span>
        </button>`).join('');
    el.onclick = (e) => {
        const btn = e.target.closest('[data-task]');
        if (!btn || btn.dataset.task === helpTask) return;
        helpTask = btn.dataset.task;
        refreshFeed();
    };
}

function renderFeedFilters() {
    renderHelpTasks();
    const people = document.getElementById('personaChips');
    const topics = document.getElementById('topicChips');
    const anyActive = feedPersonaIds.size || feedTopics.size || feedQuery.trim();

    if (people) {
        people.innerHTML = `<button type="button" class="chip" data-all aria-pressed="${!feedPersonaIds.size}">${escapeHtml(t('feed.everyone'))}</button>` +
            feedPersonaChoices.map((p) =>
                `<button type="button" class="chip" data-value="${p.id}" aria-pressed="${feedPersonaIds.has(p.id)}">` +
                `<span class="chip-dot" style="background:${escapeHtml(personaColor(p.id))}"></span>${escapeHtml(p.name || t('common.unknown'))}</button>`
            ).join('');
        people.onclick = (e) => onFilterPick(e, feedPersonaIds, Number);
        watchRowOverflow(people);
    }
    if (topics) {
        topics.innerHTML = `<button type="button" class="topic" data-all aria-pressed="${!feedTopics.size}">${escapeHtml(t('feed.allTopics'))}</button>` +
            feedTopicChoices.map((t) =>
                `<button type="button" class="topic" data-value="${escapeHtml(t)}" aria-pressed="${feedTopics.has(t)}">${escapeHtml(topicLabel(t))}</button>`
            ).join('') +
            (anyActive ? `<button type="button" class="filter-clear" data-clear-all>${escapeHtml(t('feed.clearAll'))}</button>` : '');
        topics.onclick = (e) => {
            if (e.target.closest('[data-clear-all]')) {
                feedPersonaIds.clear();
                feedTopics.clear();
                feedQuery = '';
                const search = document.getElementById('siteSearch');
                if (search) search.value = '';
                return refreshFeed();
            }
            onFilterPick(e, feedTopics, String);
        };
    }
}

// A horizontally scrolling row fades at the edge that has more to show, so the
// last chips don't look like the end of the list on narrow screens.
function watchRowOverflow(row) {
    const update = () => {
        const max = row.scrollWidth - row.clientWidth;
        row.classList.toggle('has-more-right', row.scrollLeft < max - 2);
        row.classList.toggle('has-more-left', row.scrollLeft > 2);
    };
    if (!row.dataset.overflowWatched) {
        row.dataset.overflowWatched = '1';
        row.addEventListener('scroll', update, { passive: true });
        window.addEventListener('resize', update);
    }
    update();
}

function onFilterPick(e, selection, cast) {
    const btn = e.target.closest('button');
    if (!btn) return;
    if (btn.hasAttribute('data-all')) {
        selection.clear();
    } else if (btn.dataset.value !== undefined) {
        const v = cast(btn.dataset.value);
        if (selection.has(v)) selection.delete(v); else selection.add(v);
    } else {
        return;
    }
    refreshFeed();
}

function refreshFeed() {
    feedShown = FEED_PAGE_SIZE;
    syncFeedUrl();
    renderFeedFilters();
    renderArgumentsList();
}

// Topic tags and persuasion formats are stable codes in the data; show them in
// the interface language, falling back to the code itself.
function topicLabel(tag) {
    const code = String(tag || '').trim();
    return tOr(`topic.${code.toLowerCase()}`, capFirst(code));
}

function formatLabel(code) {
    const c = String(code || '').trim();
    return tOr(`format.${c.toLowerCase().replace(/\s+/g, '_')}`, c.replace(/_/g, ' '));
}

function capFirst(text) {
    const t = String(text || '');
    return t.charAt(0).toUpperCase() + t.slice(1);
}

function personaColor(id) {
    return feedPersonaColors.get(id) || '#6b746e';
}

// Claims matching the search and topic filters (and, in Find mode, the people picked).
function filteredFeedItems({ ignorePersonas = false } = {}) {
    const query = normalizeSearch(feedQuery);
    return allArguments.filter((a) => {
        const matchesQuery = !query || normalizeSearch(`${a.text || ''} ${a.top_response?.text || ''}`).includes(query);
        const matchesTopic = feedTopics.size === 0 || (a.tags || []).some((tag) => feedTopics.has(tag));
        const matchesPersona = ignorePersonas || feedPersonaIds.size === 0
            || (a.personas || []).some((x) => x && feedPersonaIds.has(x.id));
        return matchesQuery && matchesTopic && matchesPersona;
    });
}

function renderArgumentsList() {
    const list = document.getElementById('argumentsList');
    const emptyMessage = document.getElementById('emptyMessage');
    const moreBtn = document.getElementById('feedMore');
    if (!list) return;

    const help = feedMode === 'help';
    let items = filteredFeedItems({ ignorePersonas: help });

    if (help) {
        // Most urgent first (community importance), then the least worked-on.
        items = items.filter(HELP_TASKS[helpTask].test).sort((a, b) =>
            ((b.importance?.avg_score || 0) - (a.importance?.avg_score || 0))
            || ((a.contribution_count || 0) - (b.contribution_count || 0)));
    } else {
        items = items.slice().sort((a, b) => {
            const ca = a.contribution_count || 0;
            const cb = b.contribution_count || 0;
            if (feedSort === 'gaps') return ca - cb;
            if (feedSort === 'title') return String(a.text || '').localeCompare(String(b.text || ''), LANG);
            return cb - ca; // most active (default)
        });
    }

    list.innerHTML = '';
    if (moreBtn) moreBtn.hidden = items.length <= feedShown;

    if (!items.length) {
        if (help) {
            list.innerHTML = `<p class="feed-empty">${escapeHtml(t('help.empty'))}</p>`;
            if (emptyMessage) emptyMessage.style.display = 'none';
        } else if (emptyMessage) {
            renderFeedEmpty(emptyMessage);
            emptyMessage.style.display = 'block';
        }
        return;
    }
    if (emptyMessage) emptyMessage.style.display = 'none';

    items.slice(0, feedShown).forEach((a) => list.appendChild(help ? createHelpCard(a, helpTask) : createFeedCard(a)));
}

// Empty feed. When the only reason is the people picked (a profile with no answer
// yet), say so and point to « Contribuer → Écrire une réponse »; otherwise offer to add the claim.
function renderFeedEmpty(el) {
    if (!el.dataset.defaultHtml) el.dataset.defaultHtml = el.innerHTML;
    const blockedByPeople = feedPersonaIds.size > 0 && filteredFeedItems({ ignorePersonas: true }).length > 0;
    if (!blockedByPeople) {
        // Put the default message back (its button loses its listener, so re-wire it).
        if (el.innerHTML !== el.dataset.defaultHtml) {
            el.innerHTML = el.dataset.defaultHtml;
            el.querySelectorAll('[data-open-composer]').forEach((b) => {
                b.onclick = () => document.getElementById('feedComposerToggle')?.click();
            });
        }
        return;
    }
    const names = feedPersonaChoices.filter((p) => feedPersonaIds.has(p.id)).map((p) => p.name);
    el.innerHTML = `
        <p>${escapeHtml(t('home.emptyPersona', { names: names.join(t('home.or')) }))}</p>
        <p class="feed-empty-hint">${escapeHtml(t('home.emptyPersonaHint'))}</p>
        <button type="button" class="btn btn-primary" data-help-write>${icon('pen')} ${escapeHtml(t('home.emptyPersonaCta'))}</button>`;
    el.querySelector('[data-help-write]').onclick = () => {
        feedMode = 'help';
        helpTask = 'write';
        syncModeUi();
        refreshFeed();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
}

// Help-out card: why this claim needs help, the claim, and one action.
function createHelpCard(a, taskId) {
    const task = HELP_TASKS[taskId];
    const href = `argument.html?id=${encodeURIComponent(a.id)}`;
    let action = '';
    let extra = '';

    if (taskId === 'vote') {
        const mine = a.importance?.my_score ?? null;
        action = `
            <div class="help-vote" role="group" aria-label="${escapeHtml(t('priority.groupLabel'))}">
                ${PRIORITY_LEVELS.map((lvl) => `
                    <button type="button" class="priority-btn${mine === lvl.score ? ' is-selected' : ''}"
                            data-score="${lvl.score}" title="${escapeHtml(lvl.hint)}"
                            aria-pressed="${mine === lvl.score}">${escapeHtml(lvl.label)}</button>`).join('')}
            </div>`;
    } else if (taskId === 'write') {
        const p = missingArchetypes(a)[0];
        action = `<a class="btn btn-primary btn-small help-cta" href="${href}&amp;write=${encodeURIComponent(p.id)}">${icon('pen')} ${escapeHtml(t('help.write.cta', { name: firstName(p.name) }))}</a>`;
    } else if (taskId === 'rate') {
        extra = `<div class="claim-reply"><p>${escapeHtml(a.top_response.text)}</p></div>`;
        action = `<a class="btn btn-primary btn-small help-cta" href="${href}&amp;rate=1">★ ${escapeHtml(t('help.rate.cta'))}</a>`;
    }

    const card = document.createElement('div');
    card.className = 'claim-card help-card';
    card.dataset.argumentId = a.id;
    card.innerHTML = `
        <p class="help-signal"><span class="priority-tally">${escapeHtml(task.signal(a))}</span></p>
        <a class="claim-text help-claim" href="${href}">${quotedClaimHtml(a.text)}</a>
        ${extra}
        <div class="help-action">${action}</div>
    `;

    card.querySelectorAll('.priority-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const score = Number(btn.dataset.score);
            await submitImportanceRating(card, score, { toast: true });
            // Keep our copy in step so the task counts stay right.
            if (btn.getAttribute('aria-pressed') === 'true') {
                a.importance = { ...(a.importance || {}), my_score: score };
                renderHelpTasks();
            }
        });
    });
    return card;
}

// Importance levels mirror the Low/Medium/Critical chips on the contribute page,
// mapped back from the 1..5 average the ratings are stored on.
function importanceLevel(avgScore) {
    if (avgScore >= 4) return { label: t('priority.critical'), className: 'feed-importance-critical' };
    if (avgScore >= 2.5) return { label: t('priority.medium'), className: 'feed-importance-medium' };
    return { label: t('priority.low'), className: 'feed-importance-low' };
}

// Persona emoji, drawn as black line art by the Noto Emoji font (see .persona-emoji in styles.css).
function personaEmojiHtml(emoji) {
    return emoji ? `<span class="persona-emoji" aria-hidden="true">${escapeHtml(emoji)}</span> ` : '';
}

// Keep the feed's filters in the URL so a filtered view is shareable and
// survives a refresh.
function syncFeedUrl() {
    const url = new URL(window.location.href);
    const set = (key, value) => (value ? url.searchParams.set(key, value) : url.searchParams.delete(key));
    set('persona', [...feedPersonaIds].join(','));
    set('topic', [...feedTopics].join(','));
    set('q', feedQuery.trim());
    set('mode', feedMode === 'help' ? 'help' : '');
    set('task', feedMode === 'help' ? helpTask : '');
    window.history.replaceState({}, '', url);
}

function createFeedCard(a) {
    const count = a.counter_argument_count || 0;
    const personas = Array.isArray(a.personas) ? a.personas : [];
    const reply = a.top_response && a.top_response.text ? a.top_response : null;
    const lead = personas[0] || reply?.persona || null;

    const replyHtml = reply
        ? `<div class="claim-reply"><span class="claim-reply-label">${escapeHtml(t('card.howToAnswer'))}</span><p>${escapeHtml(reply.text)}</p></div>`
        : `<div class="claim-reply claim-reply-empty"><span class="claim-reply-label">${escapeHtml(t('card.noAnswer'))}</span><p>${escapeHtml(t('card.beFirst'))}</p></div>`;

    // Same look as the "Who are you talking to?" filter pills: dot + full name.
    const shown = personas.length ? personas : (lead ? [lead] : []);
    const pill = (p) => `<span class="claim-persona"><span class="chip-dot" style="background:${escapeHtml(personaColor(p.id))}"></span>${escapeHtml(p.name || t('common.unknown'))}</span>`;
    const who = shown.slice(0, 2).map(pill).join('') +
        (shown.length > 2 ? `<span class="claim-persona" title="${escapeHtml(shown.slice(2).map((p) => p.name).join(', '))}">+${shown.length - 2}</span>` : '');
    const answers = `<span>${escapeHtml(t('card.answers', { n: count }))}</span>`;
    const tag = (a.tags || [])[0] ? `<span class="claim-tag">${escapeHtml(topicLabel(a.tags[0]))}</span>` : '';

    const card = document.createElement('a');
    card.className = 'claim-card';
    card.href = `argument.html?id=${encodeURIComponent(a.id)}`;
    card.innerHTML = `
        <p class="claim-text">“${formatClaimText(String(a.text || '').trim().replace(/^[«"“”'\s]+|[»"“”'\s]+$/g, ''))}”</p>
        ${replyHtml}
        <div class="claim-foot">${who}${answers}${tag}</div>
    `;
    return card;
}

// ========================================
// Argument detail: responses tailored by persona
// ========================================

let argDetailState = {
    argumentId: null,
    argument: null,
    counterArguments: [],
    personas: [],
    namedPersonas: [],
};

async function initArgumentDetailPage() {
    const loading = document.getElementById('argLoading');
    const errorEl = document.getElementById('argError');
    const detail = document.getElementById('argDetail');

    const params = new URLSearchParams(window.location.search);
    const id = parseInt(params.get('id'), 10);
    if (!Number.isInteger(id)) {
        if (loading) loading.style.display = 'none';
        if (errorEl) {
            errorEl.textContent = t('claim.none');
            errorEl.style.display = 'block';
        }
        return;
    }
    argDetailState.argumentId = id;

    try {
        const [argRes, personaRes] = await Promise.all([
            // Credentials so the API can say whether this viewer may check sources.
            fetch(`${API_BASE_URL}/arguments/${id}`, { credentials: 'include' }),
            fetch(`${API_BASE_URL}/persona-archetypes`),
        ]);
        if (argRes.status === 404) {
            if (loading) loading.style.display = 'none';
            if (errorEl) {
                errorEl.className = 'not-found';
                errorEl.innerHTML = `<h2>${escapeHtml(t('claim.notFound'))}</h2><p>${t('claim.notFoundHtml')}</p>`;
                errorEl.style.display = 'block';
            }
            document.title = t('site.pageTitle', { page: t('claim.notFound') });
            return;
        }
        if (!argRes.ok) throw new Error('Failed to load narrative');

        const data = await argRes.json();
        argDetailState.argument = data.argument || {};
        argDetailState.counterArguments = Array.isArray(data.counter_arguments) ? data.counter_arguments : [];

        const personas = personaRes.ok ? await personaRes.json() : [];
        argDetailState.personas = Array.isArray(personas) ? personas : [];
        argDetailState.namedPersonas = argDetailState.personas.filter(isNamedPersona);

        if (loading) loading.style.display = 'none';
        if (detail) detail.style.display = 'block';

        renderArgumentDetail();
        initComposer();

        // Deep link from the Contribute page: ?write=<personaId> (or ?write=any) opens the composer.
        // A draft kept when the session expired comes back here, even if the link lost its ?write=.
        const write = params.get('write');
        const draft = currentUser ? takeComposerDraft(id) : null;
        if (write || draft) {
            const personaId = draft?.personaId ?? writePersonaId(write);
            openComposer(id, personaId, draft?.text || '', { draft: Boolean(draft) });
            if (draft) {
                fillComposerSources(draft.sources);
                document.getElementById('composerText')?.focus();
                showToast(t('composer.draftRestored'));
                scheduleAccountDraftSave();
            }
        }

        // Deep link from the home feed's "Rate an answer": ?rate=1 opens the first response's stars.
        if (params.get('rate')) {
            const firstCard = document.querySelector('.response-card');
            firstCard?.querySelector('.response-rate')?.click();
            firstCard?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // Deep link from the Sourcer queue: ?check=<debunkId> opens its check row.
        const checkId = params.get('check');
        const checkCa = argDetailState.counterArguments.find((c) => String(c.id) === checkId);
        const checkCard = checkCa && document.querySelector(`.response-card[data-ca-id="${checkCa.id}"]`);
        if (checkCard && checkCa.source_check?.can_check) {
            toggleCheckRow(checkCard, checkCa);
            checkCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    } catch (error) {
        console.error('Error loading narrative:', error);
        if (loading) loading.style.display = 'none';
        if (errorEl) {
            errorEl.textContent = t('claim.loadError');
            errorEl.style.display = 'block';
        }
    }
}

function renderArgumentDetail() {
    const arg = argDetailState.argument || {};
    const cas = argDetailState.counterArguments;

    document.getElementById('narrativeText').textContent = arg.text || t('claim.fallback');
    document.title = t('site.pageTitle', { page: (arg.text || t('claim.fallback')).slice(0, 50) });

    const tagsEl = document.getElementById('narrativeTags');
    tagsEl.innerHTML = (arg.tags || []).map((tag) => `<span class="tag">${escapeHtml(topicLabel(tag))}</span>`).join('');

    // "Seen with" chips: personas that already have a tailored response here.
    const seenPersonas = new Map();
    cas.forEach((ca) => {
        const p = ca.persona_archetype;
        if (p && p.id != null && isNamedPersona(p)) seenPersonas.set(p.id, p);
    });
    const seenWith = document.getElementById('seenWith');
    if (seenPersonas.size) {
        seenWith.innerHTML = `<span class="seen-with-label">${escapeHtml(t('claim.seenWith'))}</span> ` +
            Array.from(seenPersonas.values())
                .map((p) => `<span class="persona-chip">${escapeHtml(p.name || t('personas.fallbackName'))}</span>`)
                .join('');
    } else {
        seenWith.innerHTML = '';
    }

    // Persona filter (only personas present in responses) + composer persona select (all named).
    const filterEl = document.getElementById('responsePersonaFilter');
    filterEl.innerHTML = `<option value="">${escapeHtml(t('claim.anyone'))}</option>` +
        Array.from(seenPersonas.values())
            .map((p) => `<option value="${p.id}">${escapeHtml(p.name || t('personas.fallbackName'))}</option>`)
            .join('');
    filterEl.onchange = renderResponses;

    renderResponses();
    renderGaps();
}

function renderResponses() {
    const list = document.getElementById('responsesList');
    const heading = document.getElementById('responsesHeading');
    if (!list) return;

    const filterId = document.getElementById('responsePersonaFilter')?.value || '';
    let cas = argDetailState.counterArguments;
    if (filterId) {
        cas = cas.filter((ca) => ca.persona_archetype && String(ca.persona_archetype.id) === filterId);
    }

    if (heading) {
        const n = cas.length;
        heading.textContent = t('claim.responsesCount', { n });
    }

    list.innerHTML = '';
    if (!cas.length) {
        list.innerHTML = `<div class="empty-message"><p>${escapeHtml(t(filterId ? 'claim.emptyForProfile' : 'claim.emptyAll'))}</p>
            <button type="button" class="btn btn-primary" onclick="openComposer(${argDetailState.argumentId}, '${filterId}')">${escapeHtml(t('claim.writeFirst'))}</button></div>`;
        return;
    }

    cas.forEach((ca) => list.appendChild(createResponseCard(ca)));
}

function createResponseCard(ca) {
    const persona = ca.persona_archetype;
    const personaId = persona && persona.id != null ? persona.id : '';
    const personaName = persona ? (persona.name || t('personas.fallbackName')) : t('response.general');
    const label = personaLabel(persona);
    const framing = label ? ` · ${label}` : '';

    const text = String(ca.text || '');
    const isLong = text.length > 320;
    const shownText = isLong ? text.slice(0, 320) + '…' : text;

    const ratingHtml = ratingTrustHtml(ca);

    const resources = Array.isArray(ca.resources) && ca.resources.length
        ? ca.resources
        : (ca.source_url ? [{ url: ca.source_url, title: null }] : []);
    const sourceHtml = resources.length
        ? resources.map((r) => (safeHref(r.url)
            ? `<a class="trust-item source-chip" href="${escapeHtml(safeHref(r.url))}" target="_blank" rel="noopener noreferrer">${icon('clip')} ${escapeHtml(r.title || hostnameOf(r.url))}</a>`
            : `<span class="trust-item source-chip">${icon('clip')} ${escapeHtml(r.title || t('common.source'))}</span>`)
          ).join('')
        : `<span class="trust-item trust-muted">${escapeHtml(t('response.noSource'))}</span>`;

    const check = ca.source_check || {};
    const checkBtnHtml = check.can_check
        ? `<button type="button" class="btn btn-secondary btn-small response-check">${icon('search')} ${escapeHtml(t(check.my_verdict ? 'response.changeCheck' : 'response.check'))}</button>`
        : '';

    const card = document.createElement('div');
    card.className = 'response-card';
    card.dataset.caId = ca.id;
    card.dataset.text = text;
    card.dataset.source = ca.source_url || '';
    card.innerHTML = `
        <div class="response-card-head">
            <span class="persona-chip">${escapeHtml(personaName)}</span>
            <span class="response-framing">${escapeHtml(framing)}</span>
        </div>
        <p class="response-text">${escapeHtml(shownText)}</p>
        ${isLong ? `<button type="button" class="link-btn response-more">${escapeHtml(t('response.readMore'))}</button>` : ''}
        <div class="response-trust">${sourceHtml}${sourceStampHtml(check)}${ratingHtml}</div>
        <div class="response-actions">
            <button type="button" class="btn btn-secondary btn-small response-copy">${icon('copy')} ${escapeHtml(t('response.copy'))}</button>
            <button type="button" class="btn btn-secondary btn-small response-rate">★ ${escapeHtml(myRatingFor(ca, personaId) ? t('response.myRating', { n: myRatingFor(ca, personaId) }) : t('response.rate'))}</button>
            <button type="button" class="btn btn-secondary btn-small response-improve">${icon('pen')} ${escapeHtml(t('response.improve'))}</button>
            ${checkBtnHtml}
        </div>
        <div class="rate-row" style="display: none;" aria-hidden="true"></div>
        <div class="check-row" hidden></div>
    `;

    // Check source → inline verdict row (Sourcers and moderators only)
    card.querySelector('.response-check')?.addEventListener('click', () => toggleCheckRow(card, ca));

    // Read more / less
    if (isLong) {
        const moreBtn = card.querySelector('.response-more');
        const textEl = card.querySelector('.response-text');
        let expanded = false;
        moreBtn.addEventListener('click', () => {
            expanded = !expanded;
            textEl.textContent = expanded ? text : shownText;
            moreBtn.textContent = t(expanded ? 'response.readLess' : 'response.readMore');
        });
    }

    // Copy → clipboard + conversion nudge toward a field report.
    card.querySelector('.response-copy').addEventListener('click', () => {
        const urls = resources.map((r) => safeHref(r.url)).filter(Boolean);
        const toCopy = urls.length ? `${text}\n\n${t('response.copySources')}\n${urls.join('\n')}` : text;
        navigator.clipboard.writeText(toCopy).then(() => {
            showToast(t('response.copied'));
        }).catch(() => {
            showToast(t('response.copyFailed'));
        });
    });

    // Rate → inline stars
    card.querySelector('.response-rate').addEventListener('click', () => {
        const rateRow = card.querySelector('.rate-row');
        const isHidden = rateRow.style.display === 'none';
        rateRow.style.display = isHidden ? 'flex' : 'none';
        rateRow.setAttribute('aria-hidden', isHidden ? 'false' : 'true');
        if (isHidden && !rateRow.dataset.built) {
            buildRateRow(rateRow, ca, personaId, card);
            rateRow.dataset.built = '1';
        }
    });

    // Improve → composer prefilled as a variant proposal
    card.querySelector('.response-improve').addEventListener('click', () => {
        openComposer(argDetailState.argumentId, personaId, text);
    });

    return card;
}

// ---- Source-verified stamp (see backend/roles.py) ----

const SOURCE_STAMPS = {
    verified: { icon: 'check', label: t('stamp.verified'), cls: 'stamp-verified', title: t('stamp.verifiedTitle') },
    needs_work: { icon: 'x', label: t('stamp.needsWork'), cls: 'stamp-needs-work', title: t('stamp.needsWorkTitle') },
    pending: { icon: 'hourglass', label: t('stamp.pending'), cls: 'stamp-pending', title: t('stamp.pendingTitle') },
};

const SOURCE_VERDICTS = [
    { value: 'supports', label: t('verdict.supports') },
    { value: 'partial', label: t('verdict.partial') },
    { value: 'does_not_support', label: t('verdict.doesNotSupport') },
    { value: 'broken', label: t('verdict.broken') },
];

function sourceStampHtml(check) {
    const stamp = SOURCE_STAMPS[check.status];
    if (!stamp) return '';
    const label = check.status === 'pending' && check.count
        ? t('stamp.inProgress', { n: check.count })
        : stamp.label;
    return `<span class="trust-item source-stamp ${stamp.cls}" title="${escapeHtml(stamp.title)}">${icon(stamp.icon)} ${escapeHtml(label)}</span>`;
}

function toggleCheckRow(card, ca) {
    const row = card.querySelector('.check-row');
    if (!row) return;
    row.hidden = !row.hidden;
    if (row.hidden || row.dataset.built) return;
    row.dataset.built = '1';

    const mine = ca.source_check?.my_verdict || null;
    row.innerHTML = `
        <span class="rate-label">${escapeHtml(t('check.prompt'))}</span>
        <div class="check-verdicts" role="group" aria-label="${escapeHtml(t('check.verdictLabel'))}">
            ${SOURCE_VERDICTS.map((v) => `
                <button type="button" class="btn btn-secondary btn-small check-verdict" data-verdict="${v.value}"
                        aria-pressed="${v.value === mine}">${escapeHtml(v.label)}</button>`).join('')}
        </div>
        <input type="text" class="check-note" maxlength="500" aria-label="${escapeHtml(t('check.noteLabel'))}"
               placeholder="${escapeHtml(t('check.notePlaceholder'))}">
        <div class="check-submit-row">
            <button type="button" class="btn btn-primary btn-small check-submit" ${mine ? '' : 'disabled'}>${escapeHtml(t('check.submit'))}</button>
            <span class="check-hint">${escapeHtml(t('check.hint'))}</span>
        </div>
    `;

    let verdict = mine;
    row.querySelectorAll('.check-verdict').forEach((btn) => {
        btn.addEventListener('click', () => {
            verdict = btn.dataset.verdict;
            row.querySelectorAll('.check-verdict').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
            row.querySelector('.check-submit').disabled = false;
        });
    });
    row.querySelector('.check-submit').addEventListener('click', () => {
        submitSourceCheck(ca.id, verdict, row.querySelector('.check-note').value.trim(), row);
    });
}

async function submitSourceCheck(counterArgumentId, verdict, note, row) {
    if (!verdict) return;
    const btn = row.querySelector('.check-submit');
    btn.disabled = true;
    try {
        const res = await fetch(`${API_BASE_URL}/counter-arguments/${counterArgumentId}/source-checks`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ verdict, note }),
        });
        const data = await res.json();
        if (!res.ok) {
            apiErrorToast(data, 'check.error', res);
            btn.disabled = false;
            return;
        }
        const status = data.source_check?.status;
        const points = data.points_earned ? t('check.points', { n: data.points_earned }) : '';
        const msg = status === 'verified' ? t('check.verified', { points })
            : status === 'needs_work' ? t('check.needsWork', { points })
            : t('check.saved');
        showToast(msg);
        await reloadArgumentDetail();
    } catch (error) {
        console.error('Source check error:', error);
        showToast(t('check.networkError'));
        btn.disabled = false;
    }
}

// The subtitle of a profile (« La tante conservatrice »), the same everywhere.
function personaLabel(p) {
    return String(p?.short_label || p?.description || '').trim();
}

function lowerFirst(text) {
    const s = String(text || '');
    return s.charAt(0).toLocaleLowerCase(LOCALE) + s.slice(1);
}

// The viewer's own score for this answer and profile (null profile = « Tout le monde »).
function myRatingFor(ca, personaId) {
    const pid = personaId === '' || personaId == null ? null : Number(personaId);
    const entry = (ca.my_ratings || []).find((r) => (r.persona_archetype_id ?? null) === pid);
    return Number(entry ? entry.score : ca.my_score) || 0;
}

function buildRateRow(rateRow, ca, personaId, card) {
    const mine = myRatingFor(ca, personaId);
    rateRow.innerHTML = `<span class="rate-label">${escapeHtml(mine ? t('rate.promptChange', { n: mine }) : t('rate.prompt'))}</span>`;
    const starWrap = document.createElement('span');
    starWrap.className = 'rate-stars';
    for (let i = 1; i <= 5; i++) {
        const star = document.createElement('button');
        star.type = 'button';
        star.className = 'rate-star';
        star.textContent = '★';
        star.setAttribute('aria-label', t('rate.star', { n: i }));
        star.setAttribute('aria-pressed', String(i === mine));
        star.addEventListener('mouseenter', () => highlightStars(starWrap, i));
        star.addEventListener('mouseleave', () => highlightStars(starWrap, mine));
        star.addEventListener('click', () => submitRating(ca, personaId, i, rateRow, card));
        starWrap.appendChild(star);
    }
    rateRow.appendChild(starWrap);
    highlightStars(starWrap, mine);
}

function highlightStars(starWrap, upTo) {
    Array.from(starWrap.children).forEach((star, idx) => {
        star.classList.toggle('is-lit', idx < upTo);
    });
}

async function submitRating(ca, personaId, score, rateRow, card) {
    // A visitor's vote waits unattached, so the API can't say it replaced one: we know it did.
    const hadRating = myRatingFor(ca, personaId) > 0;
    try {
        const body = { counter_argument_id: ca.id, score };
        if (personaId !== '' && personaId != null) body.persona_archetype_id = Number(personaId);

        const res = await fetch(`${API_BASE_URL}/effectiveness-ratings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (res.ok) {
            rateRow.innerHTML = `<span class="rate-thanks">${escapeHtml(t(data.replaced || (data.pending && hadRating) ? 'rate.updated' : 'rate.thanks', { n: score }))}</span>`;
            delete rateRow.dataset.built;
            ca.my_score = score;
            const pid = personaId === '' || personaId == null ? null : Number(personaId);
            ca.my_ratings = (ca.my_ratings || []).filter((r) => (r.persona_archetype_id ?? null) !== pid)
                .concat({ persona_archetype_id: pid, score, pending: Boolean(data.pending) });
            const rateBtn = card?.querySelector('.response-rate');
            if (rateBtn) rateBtn.textContent = `★ ${t('response.myRating', { n: score })}`;
            // Signed-in ratings count right away: show the new average.
            const trust = card?.querySelector('.response-trust > .trust-item:last-child');
            if (trust && typeof data.avg_score === 'number') {
                ca.avg_score = data.avg_score;
                ca.rating_count = data.rating_count || 0;
                trust.outerHTML = ratingTrustHtml(ca);
            }
            if (data.pending) {
                pendingVotes = data.pending_count;
                showToast(t('votes.pendingToast', { n: data.pending_count }),
                    { actionLabel: t('common.createAccount'), actionHref: authHref('register.html'), duration: 8000 });
            } else {
                showToast(t('rate.saved'));
            }
        } else {
            apiErrorToast(data, 'rate.error', res);
        }
    } catch (error) {
        console.error('Rating error:', error);
        showToast(t('rate.networkError'));
    }
}

// Average stars and number of ratings (last item of the card's trust line).
function ratingTrustHtml(ca) {
    const avg = (typeof ca.avg_score === 'number') ? ca.avg_score : null;
    const n = ca.rating_count || 0;
    return avg != null
        ? `<span class="trust-item" title="${escapeHtml(t('response.ratingTitle', { avg: formatNumber(avg, 1), n }))}">${renderStars(avg)} <span class="rating-count">(${n})</span></span>`
        : `<span class="trust-item trust-muted">${escapeHtml(t('response.notRated'))}</span>`;
}

function renderStars(avg) {
    const rounded = Math.round(avg);
    let out = '';
    for (let i = 1; i <= 5; i++) out += (i <= rounded) ? '★' : '☆';
    return `<span class="static-stars" aria-hidden="true">${out}</span>`;
}

function hostnameOf(url) {
    try {
        return new URL(url).hostname.replace(/^www\./, '');
    } catch (e) {
        return t('common.source');
    }
}

function renderGaps() {
    const section = document.getElementById('gapsSection');
    const gapsList = document.getElementById('gapsList');
    if (!section || !gapsList) return;

    const covered = new Set(
        argDetailState.counterArguments
            .map((ca) => ca.persona_archetype && ca.persona_archetype.id)
            .filter((id) => id != null)
    );
    const gaps = argDetailState.namedPersonas.filter((p) => !covered.has(p.id));

    if (!gaps.length) {
        section.style.display = 'none';
        return;
    }
    section.style.display = 'block';
    gapsList.innerHTML = '';
    gaps.forEach((p) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'gap-chip';
        btn.innerHTML = `${icon('pen')} ${escapeHtml(t('claim.writeFor', { name: p.name }))}`;
        btn.addEventListener('click', () => openComposer(argDetailState.argumentId, p.id));
        gapsList.appendChild(btn);
    });
}

// ---- Composer ----

function initComposer() {
    const composerPersona = document.getElementById('composerPersona');
    if (composerPersona) {
        // A debunk targets exactly one persona — no "general" default.
        composerPersona.innerHTML = `<option value="" disabled selected>${escapeHtml(t('composer.choosePersona'))}</option>` +
            argDetailState.namedPersonas
                .map((p) => `<option value="${p.id}">${escapeHtml(p.name || t('personas.fallbackName'))}</option>`)
                .join('');
    }

    document.getElementById('composerAddSource')?.addEventListener('click', () => addComposerSourceInput());
    document.getElementById('composerClose')?.addEventListener('click', closeComposer);
    document.getElementById('composerCancel')?.addEventListener('click', closeComposer);
    document.getElementById('composerBackdrop')?.addEventListener('click', closeComposer);
    document.getElementById('composerForm')?.addEventListener('submit', submitComposer);
    // Changing who you answer updates the title and the tip, and moves the draft.
    composerPersona?.addEventListener('change', () => {
        syncComposerHint(composerPersona.value);
        if (composerIsBlank()) restoreAccountDraft(argDetailState.argumentId, composerPersonaId());
        else scheduleAccountDraftSave();
    });
    document.getElementById('composerForm')?.addEventListener('input', (e) => {
        if (e.target !== composerPersona) scheduleAccountDraftSave();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeComposer();
    });
    // Leaving the page (or the tab going to the background on a phone) saves right away.
    window.addEventListener('pagehide', () => saveAccountDraft({ keepalive: true }));
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') saveAccountDraft({ keepalive: true });
    });
}

function addComposerSourceInput(value = '', { focus = true } = {}) {
    const container = document.getElementById('composerSources');
    if (!container) return;
    const input = document.createElement('input');
    input.type = 'url';
    input.className = 'composer-source-input';
    input.placeholder = t('composer.sourcePlaceholder');
    input.value = value;
    container.appendChild(input);
    if (focus) input.focus();
}

function fillComposerSources(urls) {
    resetComposerSources();
    (urls || []).forEach((url, i) => {
        if (i === 0) document.querySelector('.composer-source-input').value = url;
        else addComposerSourceInput(url, { focus: false });
    });
}

function resetComposerSources() {
    const container = document.getElementById('composerSources');
    if (!container) return;
    container.innerHTML = `<input type="url" class="composer-source-input" placeholder="${escapeHtml(t('composer.sourcePlaceholder'))}">`;
}

function openComposer(argumentId, personaId = '', prefillText = '', { draft = false } = {}) {
    // Session expired with text kept for this person: the form comes back with it, ready for login.
    const kept = !currentUser && !prefillText ? keptComposerDraftFor(argumentId, personaId) : null;
    if (!currentUser && !kept) {
        // After logging in or signing up, the form opens again for the same person.
        const write = (personaId !== '' && personaId != null) ? personaId : 'any';
        showToast(t('composer.loginToast'), { actions: authToastActions({ write }), duration: 10000 });
        return;
    }
    const panel = document.getElementById('composerPanel');
    const backdrop = document.getElementById('composerBackdrop');
    const personaSelect = document.getElementById('composerPersona');
    const message = document.getElementById('composerMessage');

    if (personaSelect) personaSelect.value = (personaId !== '' && personaId != null) ? String(personaId) : '';
    document.getElementById('composerText').value = kept ? kept.text : (prefillText || '');
    resetComposerSources();
    if (kept) fillComposerSources(kept.sources);
    showComposerSourceError(false);
    if (message) { message.textContent = ''; message.className = 'form-message'; }

    syncComposerHint(personaId, Boolean(prefillText) && !draft);

    resetAccountDraftState(argumentId, draft || kept ? '' : prefillText);
    composerDraftState.open = true;
    composerDraftState.returnFocus = document.activeElement;

    if (backdrop) backdrop.style.display = 'block';
    if (panel) {
        panel.style.display = 'block';
        panel.setAttribute('aria-hidden', 'false');
        panel.classList.add('is-open');
    }
    setPageBehindComposer(true);
    document.getElementById(personaSelect && !personaSelect.value ? 'composerPersona' : 'composerText').focus();
    if (kept) {
        composerDraftState.expired = true;
        showComposerSessionExpired(kept.personaId);
    } else if (!draft) {
        restoreAccountDraft(argumentId, composerPersonaId());
    }
}

// ?write=<id> from old links can name a retired profile: answer its replacement.
function writePersonaId(write) {
    if (!/^\d+$/.test(write || '')) return '';
    const id = Number(write);
    const resolved = replacementPersonaId(argDetailState.personas, id) ?? id;
    return argDetailState.namedPersonas.some((p) => p.id === resolved) ? resolved : '';
}

// While the dialog is open, the page behind it can't be scrolled, clicked or tabbed to.
function setPageBehindComposer(covered) {
    document.documentElement.classList.toggle('composer-open', covered);
    document.querySelectorAll('body > header, body > main, body > footer').forEach((el) => { el.inert = covered; });
}

// Title and tip for the person being answered: « Vous répondez à Max Onnousment,
// le révolté anti-système. » + the profile's persuasion approach.
function syncComposerHint(personaId, improving = false) {
    const hint = document.getElementById('composerHint');
    const title = document.getElementById('composerTitle');
    const persona = argDetailState.personas.find((p) => String(p.id) === String(personaId));
    if (title) {
        title.textContent = persona && !improving
            ? t('composer.titleFor', { name: persona.name })
            : t(improving ? 'composer.titleImprove' : 'composer.title');
    }
    if (!hint) return;
    if (!persona) {
        hint.style.display = 'none';
        return;
    }
    const label = personaLabel(persona).replace(/[.\s]+$/, '');
    const intro = label ? t('composer.hint', { name: persona.name, label: lowerFirst(label) }) : persona.name;
    const approach = String(persona.persuasion?.approach || '').trim();
    hint.innerHTML = escapeHtml(intro) + (approach
        ? `<span class="composer-hint-tip"><strong>${escapeHtml(t('composer.hintApproach'))}</strong> ${escapeHtml(approach)}</span>`
        : '');
    hint.style.display = 'block';
}

// Answer drafts survive an expired session: kept for this tab, restored after login.
const COMPOSER_DRAFT_KEY = 'reinform.answerDraft';

// Nothing typed means nothing to keep (and an older copy shouldn't come back).
function saveComposerDraft(draft) {
    try {
        if (draft.text || draft.sources.length) sessionStorage.setItem(COMPOSER_DRAFT_KEY, JSON.stringify(draft));
        else sessionStorage.removeItem(COMPOSER_DRAFT_KEY);
    } catch (_) { /* private mode */ }
}

function readComposerDraft(argumentId) {
    try {
        const draft = JSON.parse(sessionStorage.getItem(COMPOSER_DRAFT_KEY) || 'null');
        return draft && draft.argumentId === argumentId ? draft : null;
    } catch (_) {
        return null;
    }
}

function takeComposerDraft(argumentId) {
    const draft = readComposerDraft(argumentId);
    if (draft) {
        try { sessionStorage.removeItem(COMPOSER_DRAFT_KEY); } catch (_) { /* private mode */ }
    }
    return draft;
}

// The kept text, if it was written for this person (« Écrire pour… » chip).
function keptComposerDraftFor(argumentId, personaId) {
    const draft = readComposerDraft(argumentId);
    const wanted = personaId === '' || personaId == null ? null : Number(personaId);
    return draft && (draft.personaId ?? null) === wanted ? draft : null;
}

// Session expired while writing: keep the text in this tab, show the visitor header,
// and offer a login that comes back here with the form open.
function composerSessionExpired(sent) {
    const state = composerDraftState;
    // The form may hold more than was sent; unchanged « improve » text reads as empty there.
    const live = state.open ? composerDraftContent() : null;
    const kept = live && (live.text || live.sources.length) ? { ...live, argumentId: state.argumentId } : sent;
    saveComposerDraft(kept);
    currentUser = null;
    updateNavForLoggedOutUser();
    if (state.open) {
        state.expired = true;
        showComposerSessionExpired(kept.personaId);
    } else {
        // Form already closed (saved on Esc): the form's own message can't be seen.
        showToast(t('composer.sessionExpired'), {
            actionLabel: t('common.login'),
            actionHref: authHrefWith('login.html', { write: kept.personaId ?? 'any' }),
            duration: 10000,
        });
    }
}

function showComposerSessionExpired(personaId) {
    const message = document.getElementById('composerMessage');
    setDraftStatus('');
    if (!message) return;
    message.innerHTML = t('composer.sessionExpiredHtml', {
        loginHref: escapeHtml(authHrefWith('login.html', { write: personaId ?? 'any' })),
    });
    message.className = 'form-message error';
    message.scrollIntoView({ block: 'nearest' });
}

// Answer drafts saved to the account: while typing (2 s pause), on close, and on
// leaving the page. One draft per claim × person; `null` is « Tout le monde ».
const DRAFT_SAVE_DELAY = 2000;
const composerDraftState = {
    open: false, frozen: false, expired: false, baseline: '', argumentId: null, lastSerial: '',
    draftId: null, savedPersonaId: null, timer: null, inflight: null, loadToken: 0, returnFocus: null,
};

function resetAccountDraftState(argumentId, baseline) {
    clearTimeout(composerDraftState.timer);
    composerDraftState.loadToken += 1;
    Object.assign(composerDraftState, {
        frozen: false, expired: false, baseline: String(baseline || '').trim(), argumentId,
        lastSerial: '', draftId: null, savedPersonaId: null,
    });
    setDraftStatus('');
}

function composerPersonaId() {
    const value = document.getElementById('composerPersona')?.value;
    return value ? Number(value) : null;
}

function composerSources() {
    return Array.from(document.querySelectorAll('.composer-source-input'))
        .map((input) => input.value.trim())
        .filter(Boolean);
}

// What would go in the draft. Unchanged text from « improve this answer » isn't theirs yet.
function composerDraftContent() {
    const value = document.getElementById('composerText').value.trim();
    const text = (composerDraftState.baseline && value === composerDraftState.baseline) ? '' : value;
    return { personaId: composerPersonaId(), text, sources: composerSources() };
}

function composerIsBlank() {
    const { text, sources } = composerDraftContent();
    return !text && !sources.length;
}

function setDraftStatus(text, isError = false) {
    const el = document.getElementById('composerDraftStatus');
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('is-error', isError);
}

function scheduleAccountDraftSave() {
    setDraftStatus('');
    clearTimeout(composerDraftState.timer);
    composerDraftState.timer = setTimeout(() => saveAccountDraft(), DRAFT_SAVE_DELAY);
}

function saveAccountDraft({ keepalive = false } = {}) {
    const state = composerDraftState;
    clearTimeout(state.timer);
    // Session gone: whatever they're still typing stays in this tab for after login.
    if (state.open && state.expired) {
        saveComposerDraft({ ...composerDraftContent(), argumentId: state.argumentId });
        return;
    }
    if (!state.open || state.frozen || !currentUser) return;

    const { personaId, text, sources } = composerDraftContent();
    const serial = JSON.stringify([personaId, text, sources]);
    if (serial === state.lastSerial) return;
    const empty = !text && !sources.length;
    if (empty && state.draftId == null) { state.lastSerial = serial; return; }
    state.lastSerial = serial;

    // Moved to another person: the draft under the previous one goes away.
    if (state.draftId != null && state.savedPersonaId !== personaId) {
        deleteDraft(state.draftId, { keepalive }).catch(() => {});
        state.draftId = null;
    }

    const argumentId = state.argumentId;
    state.inflight = (async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/me/drafts`, {
                method: 'PUT',
                keepalive,
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ argument_id: argumentId, persona_archetype_id: personaId, text, sources }),
            });
            const data = await res.json().catch(() => ({}));
            if (res.status === 401) {
                composerSessionExpired({ argumentId, personaId, text, sources });
            } else if (!res.ok) {
                setDraftStatus(apiError(data, 'composer.draftError'), true);
            } else {
                state.draftId = data.draft ? data.draft.id : null;
                state.savedPersonaId = data.draft ? personaId : null;
                if (data.draft && serial === state.lastSerial) setDraftStatus(t('composer.draftSaved'));
            }
        } catch (_) {
            state.lastSerial = ''; // try again at the next save
        }
    })();
}

function deleteDraft(id, { keepalive = false } = {}) {
    return fetch(`${API_BASE_URL}/me/drafts/${encodeURIComponent(id)}`, { method: 'DELETE', keepalive, credentials: 'include' });
}

// Sources only ever go back into text fields; safeHref still drops any odd scheme.
function draftSourceUrls(draft) {
    return (Array.isArray(draft.sources) ? draft.sources : [])
        .map((s) => safeHref(typeof s === 'string' ? s : s?.url))
        .filter(Boolean);
}

// Bring back the saved draft for this claim × person, unless they've started typing meanwhile.
async function restoreAccountDraft(argumentId, personaId) {
    const state = composerDraftState;
    if (!currentUser) return;
    const token = ++state.loadToken;
    const before = JSON.stringify(composerDraftContent());
    try {
        await state.inflight;
        const res = await fetch(`${API_BASE_URL}/me/drafts`, { credentials: 'include' });
        if (!res.ok) return;
        const { drafts } = await res.json();
        if (token !== state.loadToken || !state.open || JSON.stringify(composerDraftContent()) !== before) return;
        const draft = (drafts || []).find((d) => d.argument_id === argumentId && (d.persona_archetype_id ?? null) === personaId);
        if (!draft) return;

        document.getElementById('composerText').value = draft.text || '';
        fillComposerSources(draftSourceUrls(draft));
        state.baseline = '';
        state.draftId = draft.id;
        state.savedPersonaId = personaId;
        const { text, sources } = composerDraftContent();
        state.lastSerial = JSON.stringify([personaId, text, sources]);
        syncComposerHint(personaId ?? '');
        showToast(t('composer.draftRestored'));
    } catch (_) { /* no draft shown, nothing lost */ }
}

// The form is done with: a reload shouldn't open it again.
function removeWriteFromUrl() {
    const url = new URL(window.location.href);
    if (!url.searchParams.has('write')) return;
    url.searchParams.delete('write');
    history.replaceState(history.state, '', url);
}

function closeComposer() {
    if (!composerDraftState.open) return;
    saveAccountDraft({ keepalive: true });
    composerDraftState.open = false;
    composerDraftState.loadToken += 1;
    removeWriteFromUrl();
    const panel = document.getElementById('composerPanel');
    const backdrop = document.getElementById('composerBackdrop');
    if (panel) {
        panel.classList.remove('is-open');
        panel.style.display = 'none';
        panel.setAttribute('aria-hidden', 'true');
    }
    if (backdrop) backdrop.style.display = 'none';
    setPageBehindComposer(false);
    const back = composerDraftState.returnFocus;
    if (back && back !== document.body && back.isConnected) back.focus({ preventScroll: true });
}

// What the backend accepts as a source: an absolute http(s) address.
function isWebLink(url) {
    try {
        const u = new URL(String(url || '').trim());
        return (u.protocol === 'http:' || u.protocol === 'https:') && Boolean(u.hostname);
    } catch (_) {
        return false;
    }
}

// The API's 400 for a link that isn't http(s) (see URL_ERROR in argumentation_store.py).
function isLinkError(data) {
    return /http:\/\/ or https:\/\//.test(String(data?.error || ''));
}

function showComposerSourceError(show) {
    const el = document.getElementById('composerSourcesError');
    if (!el) return;
    el.textContent = show ? t('composer.sourceInvalid') : '';
    el.hidden = !show;
    document.querySelectorAll('.composer-source-input').forEach((input) => {
        const bad = show && input.value.trim() && !isWebLink(input.value);
        input.setAttribute('aria-invalid', bad ? 'true' : 'false');
    });
}

async function submitComposer(e) {
    e.preventDefault();
    const message = document.getElementById('composerMessage');
    const submitBtn = document.getElementById('composerSubmit');
    const text = document.getElementById('composerText').value.trim();
    const personaVal = document.getElementById('composerPersona').value;
    const sources = Array.from(document.querySelectorAll('.composer-source-input'))
        .map((input) => input.value.trim())
        .filter(Boolean);

    if (!text) {
        message.textContent = t('composer.needText');
        message.className = 'form-message error';
        return;
    }
    if (!personaVal) {
        message.textContent = t('composer.needPersona');
        message.className = 'form-message error';
        return;
    }
    showComposerSourceError(sources.some((u) => !isWebLink(u)));
    if (sources.some((u) => !isWebLink(u))) {
        message.textContent = '';
        message.className = 'form-message';
        return;
    }

    const body = {
        argument_id: argDetailState.argumentId,
        text,
        persona_archetype_id: Number(personaVal),
        sources,
    };

    submitBtn.disabled = true;
    submitBtn.textContent = t('composer.publishing');
    // No autosave from here on, so a late one can't bring the draft back after publishing.
    composerDraftState.frozen = true;
    clearTimeout(composerDraftState.timer);
    try {
        await composerDraftState.inflight;
        const res = await fetch(`${API_BASE_URL}/counter-arguments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (res.ok) {
            // The server clears the draft for this person; one kept under another
            // person or « Tout le monde » is ours to remove.
            const { draftId, savedPersonaId } = composerDraftState;
            if (draftId != null && savedPersonaId !== body.persona_archetype_id) deleteDraft(draftId).catch(() => {});
            takeComposerDraft(body.argument_id); // published: a kept copy must not come back
            closeComposer();
            showToast(t('composer.published'));
            await reloadArgumentDetail();
            return;
        }
        composerDraftState.frozen = false;
        if (isLinkError(data)) {
            showComposerSourceError(true);
        } else if (isAuthRequired(res, data)) {
            composerSessionExpired({ argumentId: body.argument_id, personaId: body.persona_archetype_id, text, sources });
        } else {
            message.textContent = apiError(data, 'composer.error');
            message.className = 'form-message error';
        }
    } catch (error) {
        console.error('Composer error:', error);
        composerDraftState.frozen = false;
        message.textContent = t('common.networkError');
        message.className = 'form-message error';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = t('composer.submit');
    }
}

async function reloadArgumentDetail() {
    try {
        const res = await fetch(`${API_BASE_URL}/arguments/${argDetailState.argumentId}`, { credentials: 'include' });
        if (!res.ok) return;
        const data = await res.json();
        argDetailState.argument = data.argument || {};
        argDetailState.counterArguments = Array.isArray(data.counter_arguments) ? data.counter_arguments : [];
        renderArgumentDetail();
    } catch (error) {
        console.error('Error reloading narrative:', error);
    }
}

const blogState = { topics: new Set(), q: '', sort: 'rating' };

async function initBlogArticlesPage() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');

    try {
        const response = await fetch(`${API_BASE_URL}/resources`);
        if (!response.ok) throw new Error('Failed to load resources');

        allBlogResources = await response.json();
        if (!Array.isArray(allBlogResources)) allBlogResources = [];

        const topicOptions = [...new Set(allBlogResources.flatMap((r) => r.tags || []).filter(Boolean))]
            .sort((a, b) => a.localeCompare(b, LANG)).map((tag) => ({ value: tag, label: topicLabel(tag) }));

        // Arriving from Profile via resources.html?tag=… pre-selects that topic.
        const requestedTag = new URLSearchParams(window.location.search).get('tag');
        const seed = requestedTag
            ? topicOptions.filter((o) => o.value.toLowerCase() === requestedTag.toLowerCase()).map((o) => o.value)
            : [];
        blogState.topics = new Set(seed);

        createFilterBar(document.getElementById('blogControls'), {
            facets: [{ id: 'topic', label: t('filter.topic'), icon: icon('tag'), options: topicOptions, selected: seed }],
            sort: {
                label: t('filter.sort'), value: blogState.sort, options: [
                    { value: 'rating', label: t('articles.sortRating') },
                    { value: 'date', label: t('articles.sortRecent') },
                    { value: 'title', label: t('articles.sortTitle') },
                ],
            },
            search: { placeholder: t('articles.search') },
            onChange: (s) => {
                blogState.topics = new Set(s.facets.topic);
                blogState.sort = s.sort;
                blogState.q = s.q;
                filterAndDisplayBlogResources();
            },
        });

        if (loadingMessage) loadingMessage.style.display = 'none';
        filterAndDisplayBlogResources();
    } catch (error) {
        console.error('Error loading resources:', error);
        if (loadingMessage) loadingMessage.style.display = 'none';
        if (errorMessage) {
            errorMessage.textContent = t('articles.loadError');
            errorMessage.style.display = 'block';
        }
    }
}

function filterAndDisplayBlogResources() {
    const searchTerms = getSearchTerms(blogState.q);

    let filtered = allBlogResources.filter((r) => {
        const matchesSearch = !searchTerms.length || itemMatchesSearchTerms(r, searchTerms);
        const matchesTopic = blogState.topics.size === 0 || (r.tags || []).some((t) => blogState.topics.has(t));
        return matchesSearch && matchesTopic;
    });

    if (blogState.sort === 'rating') {
        filtered.sort((a, b) => (b.effectiveness_rating || 0) - (a.effectiveness_rating || 0));
    } else if (blogState.sort === 'title') {
        filtered.sort((a, b) => (a.title || '').localeCompare(b.title || '', LANG));
    } else {
        filtered.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
    }

    displayMergedResources(filtered);
}

function displayMergedResources(resources) {
    const grid = document.getElementById('resourcesGrid');
    const emptyMessage = document.getElementById('emptyResourcesMessage');
    
    if (!grid) return;
    
    if (resources.length === 0) {
        grid.style.display = 'none';
        emptyMessage.style.display = 'block';
        return;
    }
    
    grid.style.display = 'grid';
    emptyMessage.style.display = 'none';
    grid.innerHTML = '';
    
    resources.forEach(resource => {
        const card = createResourceCard(resource);
        grid.appendChild(card);
    });
}

function createResourceCard(resource) {
    const card = document.createElement('div');
    card.className = 'counter-argument-card';
    
    const tagsHTML = resource.tags && resource.tags.length > 0
        ? `<div class="card-tags">${resource.tags.map(tag => 
            `<span class="tag">${escapeHtml(topicLabel(tag))}</span>`
          ).join('')}</div>`
        : '';
    
    const isInternalArticle = (() => {
        const src = (resource?.source || '').toString().trim();
        return src.startsWith('/api/articles/') && src.toLowerCase().endsWith('.md');
    })();

    const articleViewerUrl = isInternalArticle
        ? `article.html?src=${encodeURIComponent((resource.source || '').toString().trim())}`
        : '';

    const sourceHTML = isInternalArticle
        ? `<div class="card-source"><a href="${articleViewerUrl}">${escapeHtml(t('articles.read'))}</a></div>`
        : (safeHref(resource.source)
            ? `<div class="card-source"><a href="${escapeHtml(safeHref(resource.source))}" target="_blank" rel="noopener noreferrer">${escapeHtml(t('common.viewSource'))}</a></div>`
            : '');
    
    const dateHTML = resource.date 
        ? `<div class="card-date">${escapeHtml(formatDate(resource.date))}</div>`
        : '';
    
    // Author info: prefer db-backed user link; otherwise fall back to the resource's author string.
    const authorName = (resource?.author || '').toString().trim();
    const avatar = resource.user_id
        ? userAvatarHtml(resource.user_id, resource.username, 'avatar-xs')
        : '';
    const contributorHTML = resource.user_id && resource.username
        ? `<a href="profile.html?id=${resource.user_id}" class="contributor-link">${avatar}<span class="contributor-name">${escapeHtml(t('common.by', { name: resource.username }))}</span></a>`
        : (authorName
            ? `<span class="anonymous-contributor">${escapeHtml(t('common.by', { name: authorName }))}</span>`
            : `<span class="anonymous-contributor">${escapeHtml(t('articles.anonymous'))}</span>`);
    
    const titleHTML = isInternalArticle
        ? `<a class="card-title-link" href="${articleViewerUrl}"><h3 class="card-title">${escapeHtml(resource.title || t('common.untitled'))}</h3></a>`
        : `<h3 class="card-title">${escapeHtml(resource.title || t('common.untitled'))}</h3>`;

    card.innerHTML = `
        <div class="card-header">
            ${titleHTML}
            <span class="effectiveness-badge">★ ${escapeHtml(String(resource.effectiveness_rating || 0))}/10</span>
        </div>
        ${resource.narrative ? `<div class="card-narrative">“${escapeHtml(resource.narrative)}”</div>` : ''}
        <div class="card-summary">${escapeHtml(resource.summary || resource.counter_argument || '')}</div>
        ${tagsHTML}
        <div class="card-meta">
            <div class="card-contributor">${contributorHTML}</div>
            ${sourceHTML}
            ${dateHTML}
        </div>
    `;
    
    return card;
}

// ========================================
// Admin Page
// ========================================

let adminToken = null;

function initAdminPage() {
    const loginForm = document.getElementById('adminLoginForm');
    const logoutBtn = document.getElementById('logoutBtn');
    
    // Check if already logged in
    adminToken = sessionStorage.getItem('adminToken');
    if (adminToken) {
        showAdminContent();
        loadSubmissions();
        initResourceReview();
    }
    
    if (loginForm) {
        loginForm.addEventListener('submit', handleAdminLogin);
    }
    
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleAdminLogout);
    }
}

async function handleAdminLogin(e) {
    e.preventDefault();
    
    const password = document.getElementById('adminPassword').value;
    const loginMessage = document.getElementById('loginMessage');
    const loginBtn = document.querySelector('#adminLoginForm button[type="submit"]');
    
    loginBtn.disabled = true;
    loginBtn.textContent = t('common.loggingIn');
    
    try {
        const response = await fetch(`${API_BASE_URL}/admin/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            adminToken = data.token;
            sessionStorage.setItem('adminToken', adminToken);
            showAdminContent();
            loadSubmissions();
            initResourceReview();
        } else {
            loginMessage.textContent = apiError(data, 'admin.invalidPassword');
            loginMessage.className = 'form-message error';
        }
    } catch (error) {
        console.error('Login error:', error);
        loginMessage.textContent = t('common.networkError');
        loginMessage.className = 'form-message error';
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = t('common.login');
    }
}

function handleAdminLogout() {
    // Best-effort: tell the server to invalidate this token, then clear locally.
    if (adminToken) {
        fetch(`${API_BASE_URL}/admin/logout`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${adminToken}` }
        }).catch(() => {});
    }
    adminToken = null;
    sessionStorage.removeItem('adminToken');
    document.querySelector('.admin-login').style.display = 'block';
    document.querySelector('.admin-content').classList.remove('active');
}

// The admin token expired (API 401 admin_auth_required): back to the password form.
function adminSessionExpired() {
    handleAdminLogout();
    const msg = document.getElementById('loginMessage');
    if (msg) {
        msg.textContent = t('error.admin_auth_required');
        msg.className = 'form-message error';
    }
}

function showAdminContent() {
    document.querySelector('.admin-login').style.display = 'none';
    document.querySelector('.admin-content').classList.add('active');
}

async function loadSubmissions() {
    const submissionsList = document.getElementById('submissionsList');
    const loadingMessage = document.querySelector('.admin-content .loading-message');
    
    if (!submissionsList) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/admin/submissions`, {
            headers: {
                'Authorization': `Bearer ${adminToken}`
            }
        });
        
        if (response.status === 401) return adminSessionExpired();
        if (!response.ok) {
            throw new Error('Failed to load submissions');
        }
        
        const submissions = await response.json();
        
        loadingMessage.style.display = 'none';
        
        if (submissions.length === 0) {
            submissionsList.innerHTML = `<p class="empty-message">${escapeHtml(t('admin.submissionsEmpty'))}</p>`;
            return;
        }
        
        // Sort by date, newest first, pending first
        submissions.sort((a, b) => {
            if (a.status === 'pending' && b.status !== 'pending') return -1;
            if (a.status !== 'pending' && b.status === 'pending') return 1;
            return new Date(b.submitted_at) - new Date(a.submitted_at);
        });
        
        submissionsList.innerHTML = submissions.map(s => createSubmissionCard(s)).join('');
        
        // Add event listeners for approve/reject buttons
        attachSubmissionEventListeners();
        
    } catch (error) {
        console.error('Error loading submissions:', error);
        loadingMessage.textContent = t('admin.submissionsLoadError');
    }
}

function createSubmissionCard(submission) {
    const date = escapeHtml(formatDateTime(submission.submitted_at));
    const statusClass = submission.status || 'pending';
    const statusLabel = { pending: 'admin.statusPending', approved: 'admin.statusApproved', rejected: 'admin.statusRejected' }[statusClass];
    const e = (key, vars) => escapeHtml(t(key, vars));

    const urls = (Array.isArray(submission.sources) && submission.sources.length)
        ? submission.sources
        : (submission.source ? [submission.source] : []);
    const sources = urls.length
        ? urls.map((u) => (safeHref(u)
            ? `<a href="${escapeHtml(safeHref(u))}" target="_blank" rel="noopener noreferrer">${escapeHtml(u)}</a>`
            : escapeHtml(u))).join('<br>')
        : e('admin.notProvided');

    const author = submission.user_id && submission.username
        ? `<a href="profile.html?id=${encodeURIComponent(submission.user_id)}">${escapeHtml(submission.username)}</a>`
        : e('contrib.anonymous');

    const claimLink = submission.argument_id != null
        ? `<a href="argument.html?id=${encodeURIComponent(submission.argument_id)}">${escapeHtml(submission.narrative || '')}</a>`
        : escapeHtml(submission.narrative || '');
    const newClaim = submission.argument_is_new ? ` <span class="submission-new">${e('admin.newClaim')}</span>` : '';

    // Tags are the only thing an admin adds on approval (they go on the claim).
    const actions = submission.status === 'pending' ? `
        <div class="submission-actions">
            <button class="btn btn-primary btn-approve" data-id="${submission.id}">${e('admin.approve')}</button>
            <button class="btn btn-secondary btn-reject" data-id="${submission.id}">${e('admin.reject')}</button>
        </div>
        <div class="approval-form" id="approval-form-${submission.id}">
            <div class="form-group">
                <label for="approval-tags-${submission.id}">${e('admin.fieldTags')}</label>
                <input type="text" class="approval-tags" id="approval-tags-${submission.id}"
                       value="${escapeHtml((submission.tags || []).join(', '))}" placeholder="${e('admin.fieldTagsPlaceholder')}" />
            </div>
            <div class="form-actions">
                <button class="btn btn-primary btn-confirm-approve" data-id="${submission.id}">${e('admin.confirmApproval')}</button>
                <button class="btn btn-secondary btn-cancel-approve" data-id="${submission.id}">${e('common.cancel')}</button>
            </div>
        </div>
    ` : '';

    return `
        <div class="submission-card ${statusClass}">
            <div class="submission-header">
                <strong>${e('admin.submissionNumber', { id: submission.id })}</strong>
                <span class="submission-status ${statusClass}">${statusLabel ? e(statusLabel) : escapeHtml(capitalizeWords(statusClass))}</span>
            </div>
            <div class="submission-content">
                <div class="submission-field">
                    <strong>${e('admin.claim')}</strong>
                    <p>${claimLink}${newClaim}</p>
                </div>
                <div class="submission-field">
                    <strong>${e('admin.response')}</strong>
                    <p>${escapeHtml(submission.counter_argument || '')}</p>
                </div>
                <div class="submission-field">
                    <strong>${e('admin.source')}</strong>
                    <p>${sources}</p>
                </div>
                <div class="submission-field">
                    <strong>${e('admin.author')}</strong>
                    <p>${author}</p>
                </div>
                <div class="submission-field">
                    <strong>${e('admin.submitted')}</strong>
                    <p>${date}</p>
                </div>
            </div>
            ${actions}
        </div>
    `;
}

function attachSubmissionEventListeners() {
    // Approve buttons - show form
    document.querySelectorAll('.btn-approve').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const submissionId = e.target.dataset.id;
            const form = document.getElementById(`approval-form-${submissionId}`);
            form.classList.add('active');
        });
    });
    
    // Cancel approve buttons - hide form
    document.querySelectorAll('.btn-cancel-approve').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const submissionId = e.target.dataset.id;
            const form = document.getElementById(`approval-form-${submissionId}`);
            form.classList.remove('active');
        });
    });
    
    // Confirm approve buttons
    document.querySelectorAll('.btn-confirm-approve').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const submissionId = e.target.dataset.id;
            const form = document.getElementById(`approval-form-${submissionId}`);
            
            const tagsInput = form.querySelector('.approval-tags').value.trim();
            const tags = tagsInput ? tagsInput.split(',').map((tag) => tag.trim()).filter(Boolean) : [];

            await approveSubmission(submissionId, { tags });
        });
    });
    
    // Reject buttons
    document.querySelectorAll('.btn-reject').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const submissionId = e.target.dataset.id;
            if (confirm(t('admin.confirmReject'))) {
                await rejectSubmission(submissionId);
            }
        });
    });
}

async function approveSubmission(submissionId, data) {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/approve-submission/${submissionId}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${adminToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            alert(t('admin.approved'));
            loadSubmissions(); // Reload submissions
        } else {
            const error = await response.json();
            alert(t('admin.approveFailed', { error: error.error || t('admin.unknownError') }));
        }
    } catch (error) {
        console.error('Error approving submission:', error);
        alert(t('common.networkError'));
    }
}

async function rejectSubmission(submissionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/reject-submission/${submissionId}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${adminToken}`
            }
        });
        
        if (response.ok) {
            alert(t('admin.rejected'));
            loadSubmissions(); // Reload submissions
        } else {
            const error = await response.json();
            alert(t('admin.rejectFailed', { error: error.error || t('admin.unknownError') }));
        }
    } catch (error) {
        console.error('Error rejecting submission:', error);
        alert(t('common.networkError'));
    }
}

// ========================================
// Avatar Helpers
// ========================================

function avatarUrl(userId) {
    return `${API_BASE_URL}/user/${userId}/avatar`;
}

// Round profile picture for a user, with a first-letter fallback rendered
// underneath the image. The <img> removes itself on failure (404 = no picture
// uploaded), which cleanly reveals the fallback with no broken-image flash.
// Returns raw HTML — innerHTML contexts only.
function userAvatarHtml(userId, username, sizeClass = 'avatar-sm') {
    if (!userId) return '';
    const initial = ((username || '').trim().charAt(0) || '?').toUpperCase();
    return `
        <span class="avatar ${sizeClass}" aria-hidden="true">
            <span class="avatar-initials">${escapeHtml(initial)}</span>
            <img src="${avatarUrl(userId)}" alt="" loading="lazy" onerror="this.remove()">
        </span>`;
}

// ========================================
// Utility Functions
// ========================================

// A link from user data, or null if it isn't a plain web link. Only http(s)
// (absolute, or relative to this site) may go in an href: this rules out
// `javascript:`, `data:` and similar schemes that would run on click.
function safeHref(url) {
    const raw = String(url ?? '').trim();
    if (!raw) return null;
    try {
        const u = new URL(raw, window.location.href);
        return (u.protocol === 'http:' || u.protocol === 'https:') ? raw : null;
    } catch (_) {
        return null;
    }
}

// Lower-case and strip accents, so "écolo", "ecolo" and "Écolo" all match.
function normalizeSearch(text) {
    return String(text || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Claim text often stacks long slash-joined runs
// ("incendies/inondations/sécheresses/canicules") that browsers treat as one
// unbreakable word and overflow the card. Escape first, then add a zero-width
// break opportunity after each slash so it can wrap — without visibly changing
// the text. Returns HTML, so use only in innerHTML contexts.
function formatClaimText(text) {
    return escapeHtml(String(text || '')).replace(/\//g, '/<wbr>');
}

// Claim text between curly quotes, dropping any quotes already around it.
function quotedClaimHtml(text) {
    return `“${formatClaimText(String(text || '').trim().replace(/^[«"“”'\s]+|[»"“”'\s]+$/g, ''))}”`;
}

const __searchTextCache = new WeakMap();

function getSearchTerms(query) {
    const normalized = normalizeSearch(query);
    if (!normalized) return [];
    return normalized.split(/\s+/).filter(Boolean);
}

function buildSearchText(value, visited) {
    if (value === null || value === undefined) return '';
    const valueType = typeof value;

    if (valueType === 'string' || valueType === 'number' || valueType === 'boolean') {
        return String(value);
    }

    if (valueType === 'function') return '';

    if (Array.isArray(value)) {
        return value.map(v => buildSearchText(v, visited)).join(' ');
    }

    if (valueType === 'object') {
        if (!visited) visited = new WeakSet();
        if (visited.has(value)) return '';
        visited.add(value);

        // Prefer values over keys; include keys only if they help discovery
        return Object.entries(value)
            .filter(([key]) => key !== '__searchText')
            .map(([key, val]) => `${key} ${buildSearchText(val, visited)}`)
            .join(' ');
    }

    return '';
}

function getCachedSearchText(item) {
    if (!item || typeof item !== 'object') {
        return normalizeSearch(buildSearchText(item));
    }

    const cached = __searchTextCache.get(item);
    if (cached) return cached;

    const computed = normalizeSearch(buildSearchText(item));
    __searchTextCache.set(item, computed);
    return computed;
}

function itemMatchesSearchTerms(item, terms) {
    if (!terms || !terms.length) return true;
    const haystack = getCachedSearchText(item);
    return terms.every(term => haystack.includes(term));
}

function capitalizeWords(str) {
    // Unicode-aware, so accented words stay intact ("énergie" → "Énergie", not "éNergie").
    return String(str).replace(/(^|[^\p{L}\p{N}'’])(\p{L})/gu, (m, before, letter) => before + letter.toUpperCase());
}

// ========================================
// Resource Library Page
// ========================================

// Filters live in one object so every control shares a single render path.
// The page opens on infographics — this panel replaced the old Gallery, and
// landing on the visuals keeps that entry point. "All types" is one click away.
const LIBRARY_DEFAULT_TYPE = 'infographic';

const LIBRARY_TYPE_ICONS = {
    article: 'file',
    study: 'flask',
    report: 'book',
    infographic: 'chart',
    video: 'video',
    podcast: 'headphones',
    dataset: 'trend',
    website: 'globe',
};

// Type names come from the API in English; the catalog has them by type id.
function libraryTypeLabel(type, fallback) {
    return tOr(`libraryType.${type}`, fallback || capitalizeWords(String(type || '')));
}

// Client-side filter state: sets so every facet is multi-select, matching the
// Home feed. The library is small enough to fetch once and filter in the page.
let allLibraryResources = [];
const libraryState = {
    type: new Set([LIBRARY_DEFAULT_TYPE]),
    topic: new Set(),
    publisher: new Set(),
    q: '',
    sort: 'used',
};
let libraryBar = null;

async function initLibraryPage() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');

    if (loadingMessage) loadingMessage.style.display = 'block';
    if (errorMessage) errorMessage.style.display = 'none';

    try {
        // Fetch the whole reviewed library once; filter/sort/search happen here.
        const response = await fetch(`${API_BASE_URL}/library`);
        if (!response.ok) throw new Error(`Request failed: ${response.status}`);
        const data = await response.json();
        allLibraryResources = Array.isArray(data.resources) ? data.resources : [];

        // A type defaulted in state may not exist in the data — drop it if so,
        // so the page doesn't open on an empty, confusing view.
        const presentTypes = new Set(allLibraryResources.map((r) => r.type));
        libraryState.type = new Set([...libraryState.type].filter((t) => presentTypes.has(t)));

        libraryBar = createFilterBar(document.getElementById('libraryControls'), {
            facets: [
                { id: 'type', label: t('filter.type'), icon: icon('file'), options: libraryFacetOptions('type'), selected: [...libraryState.type] },
                { id: 'topic', label: t('filter.topic'), icon: icon('tag'), options: libraryFacetOptions('topic'), searchable: true, searchPlaceholder: t('filter.searchTopics') },
                { id: 'publisher', label: t('filter.publisher'), icon: icon('building'), options: libraryFacetOptions('publisher') },
            ],
            sort: {
                label: t('filter.sort'), value: libraryState.sort, options: [
                    { value: 'used', label: t('library.sortCited') },
                    { value: 'recent', label: t('library.sortRecent') },
                    { value: 'title', label: t('library.sortTitle') },
                ],
            },
            search: { placeholder: t('library.search') },
            onChange: (s) => {
                libraryState.type = new Set(s.facets.type);
                libraryState.topic = new Set(s.facets.topic);
                libraryState.publisher = new Set(s.facets.publisher);
                libraryState.sort = s.sort;
                libraryState.q = s.q;
                renderLibrary();
            },
        });

        if (loadingMessage) loadingMessage.style.display = 'none';
        renderLibrary();
    } catch (error) {
        console.error('Library error:', error);
        if (loadingMessage) loadingMessage.style.display = 'none';
        if (errorMessage) {
            errorMessage.textContent = t('library.loadError');
            errorMessage.style.display = 'block';
        }
    }
}

// A resource's value(s) for a given facet.
function libraryFacetValues(r, facet) {
    if (facet === 'type') return [r.type];
    if (facet === 'publisher') return [r.publisher];
    return r.topics || []; // topic
}

// Distinct options for a facet, counted against the OTHER active facets so the
// counts a facet advertises always lead somewhere.
function libraryFacetOptions(facet) {
    const base = filterLibrary({ except: facet });
    const counts = new Map();
    base.forEach((r) => libraryFacetValues(r, facet).forEach((v) => {
        if (v) counts.set(v, (counts.get(v) || 0) + 1);
    }));
    // Keep any currently-selected value visible even at count 0.
    libraryState[facet].forEach((v) => { if (!counts.has(v)) counts.set(v, 0); });

    const label = (v) => facet === 'type' ? libraryTypeLabel(v) : (facet === 'topic' ? topicLabel(v) : v);
    return [...counts.entries()]
        .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]), LANG))
        .map(([value, count]) => ({ value, label: label(value), count }));
}

// Apply the active facets/search; `except` skips one facet (for its own counts).
function filterLibrary({ except = null } = {}) {
    const q = normalizeSearch(libraryState.q);
    return allLibraryResources.filter((r) => {
        for (const facet of ['type', 'topic', 'publisher']) {
            if (facet === except) continue;
            const sel = libraryState[facet];
            if (sel.size && !libraryFacetValues(r, facet).some((v) => sel.has(v))) return false;
        }
        if (q) {
            const hay = normalizeSearch(`${r.title || ''} ${r.description || ''} ${r.publisher || ''} ${(r.topics || []).join(' ')} ${(r.topics || []).map(topicLabel).join(' ')}`);
            if (!hay.includes(q)) return false;
        }
        return true;
    });
}

function renderLibrary() {
    const emptyMessage = document.getElementById('emptyLibraryMessage');
    let resources = filterLibrary();

    if (libraryState.sort === 'recent') {
        resources.sort((a, b) => String(b.submitted_at || '').localeCompare(String(a.submitted_at || '')));
    } else if (libraryState.sort === 'title') {
        resources.sort((a, b) => String(a.title || '').localeCompare(String(b.title || ''), LANG));
    } else {
        resources.sort((a, b) => (b.used_in_responses || 0) - (a.used_in_responses || 0)
            || String(a.title || '').localeCompare(String(b.title || ''), LANG));
    }

    // Refresh facet option counts to reflect the new selection.
    if (libraryBar) {
        ['type', 'topic', 'publisher'].forEach((f) => libraryBar.setFacetOptions(f, libraryFacetOptions(f)));
    }
    renderLibraryStats(resources);
    renderLibraryGrid(resources);
    if (emptyMessage) emptyMessage.style.display = resources.length ? 'none' : 'block';
}

function renderLibraryStats(resources) {
    const statsEl = document.getElementById('libraryStats');
    if (!statsEl) return;

    const publishers = new Set(resources.map((r) => r.publisher).filter(Boolean));
    const cited = resources.filter((r) => (r.used_in_responses || 0) > 0).length;

    // Tiles describe what's on screen, not the whole library — the page opens
    // filtered, so a total here would understate the collection.
    const tiles = [
        { n: resources.length, label: t('library.statShown', { n: resources.length }) },
        { n: publishers.size, label: t('library.statPublishers', { n: publishers.size }) },
        { n: cited, label: t('library.statCited', { n: cited }) },
    ];
    statsEl.innerHTML = statTilesHtml(tiles);
}

// Big number + small label, shared by the library and the home feed.
function statTilesHtml(tiles) {
    return tiles
        .map((tile) => `<div class="stat-tile"><span class="stat-n">${tile.n}</span><span class="stat-label">${escapeHtml(tile.label)}</span></div>`)
        .join('');
}

// Focus the library on a single facet value clicked from a card. The bar owns
// the selection, so route through it — its onChange re-runs renderLibrary.
function libraryToggleFacet(facet, value) {
    if (!value || !libraryBar) return;
    libraryBar.setSelected(facet, [value]);
}

function renderLibraryGrid(resources) {
    const grid = document.getElementById('libraryGrid');
    if (!grid) return;

    grid.innerHTML = '';
    resources.forEach((r) => {
        const typeIcon = icon(LIBRARY_TYPE_ICONS[r.type] || 'book');
        const topics = (r.topics || []).slice(0, 4)
            .map((t) => `<button type="button" class="tag library-topic" data-topic="${escapeHtml(t)}">${escapeHtml(topicLabel(t))}</button>`)
            .join('');

        const preview = safeHref(r.preview_image)
            ? `<img class="library-preview" src="${escapeHtml(safeHref(r.preview_image))}" alt="" loading="lazy">`
            : '';

        const cited = r.used_in_responses
            ? `<span class="library-cited">${escapeHtml(t('library.cited', { n: r.used_in_responses }))}</span>`
            : '';

        const submitter = r.submitted_by?.username
            ? `<span class="library-submitter">${escapeHtml(t('library.sharedBy', { name: r.submitted_by.username }))}</span>`
            : '';

        const context = safeHref(r.context_url)
            ? `<a class="library-context" href="${escapeHtml(safeHref(r.context_url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(t('library.context'))}</a>`
            : '';

        const card = document.createElement('article');
        card.className = 'library-card';
        card.innerHTML = `
            ${preview}
            <div class="library-card-body">
                <div class="library-card-head">
                    <button type="button" class="library-type-badge" data-type="${escapeHtml(r.type)}">
                        ${typeIcon} ${escapeHtml(libraryTypeLabel(r.type, r.type_label))}
                    </button>
                    <button type="button" class="library-publisher" data-publisher="${escapeHtml(r.publisher)}">${escapeHtml(r.publisher)}</button>
                </div>
                <h3 class="library-title">
                    ${safeHref(r.url)
                        ? `<a href="${escapeHtml(safeHref(r.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title || r.url)}</a>`
                        : escapeHtml(r.title || r.url || '')}
                </h3>
                ${r.description ? `<p class="library-description">${escapeHtml(r.description)}</p>` : ''}
                ${topics ? `<div class="card-tags">${topics}</div>` : ''}
                <div class="library-card-meta">${cited}${submitter}${context}</div>
            </div>
        `;

        card.querySelectorAll('[data-topic]').forEach((btn) => {
            btn.addEventListener('click', () => libraryToggleFacet('topic', btn.dataset.topic));
        });
        card.querySelector('[data-publisher]')?.addEventListener('click', () => libraryToggleFacet('publisher', r.publisher));
        card.querySelector('[data-type]')?.addEventListener('click', () => libraryToggleFacet('type', r.type));

        grid.appendChild(card);
    });
}

// ========================================
// Admin: resource review queue
// ========================================

let adminLibraryTypes = [];

function initResourceReview() {
    const statusFilter = document.getElementById('reviewStatusFilter');
    if (statusFilter && !statusFilter.dataset.bound) {
        statusFilter.dataset.bound = 'true';
        statusFilter.addEventListener('change', loadResourceReviewQueue);
    }
    loadResourceReviewQueue();
}

async function loadResourceReviewQueue() {
    const queue = document.getElementById('reviewQueue');
    const countsEl = document.getElementById('reviewCounts');
    if (!queue) return;

    const status = document.getElementById('reviewStatusFilter')?.value ?? 'pending';
    queue.innerHTML = `<p class="loading-message">${escapeHtml(t('admin.reviewLoading'))}</p>`;

    try {
        const params = status ? `?status=${encodeURIComponent(status)}` : '';
        const response = await fetch(`${API_BASE_URL}/admin/library${params}`, {
            headers: { 'Authorization': `Bearer ${adminToken}` },
        });
        if (response.status === 401) return adminSessionExpired();
        if (!response.ok) throw new Error(`Request failed: ${response.status}`);
        const data = await response.json();

        adminLibraryTypes = data.types || [];

        if (countsEl) {
            const counts = data.counts || {};
            countsEl.textContent = t('admin.reviewCounts', { pending: counts.pending || 0, approved: counts.approved || 0, rejected: counts.rejected || 0 });
        }

        renderResourceReviewQueue(data.resources || []);
    } catch (error) {
        console.error('Review queue error:', error);
        queue.innerHTML = `<p class="error-message">${escapeHtml(t('admin.reviewLoadError'))}</p>`;
    }
}

function renderResourceReviewQueue(resources) {
    const queue = document.getElementById('reviewQueue');
    if (!queue) return;

    if (!resources.length) {
        queue.innerHTML = `<p class="empty-message">${escapeHtml(t('admin.reviewEmpty'))}</p>`;
        return;
    }

    queue.innerHTML = '';
    // Cap the page: approved is in the hundreds and the queue is a work surface,
    // not a browsing view — that's what the public library page is for.
    resources.slice(0, 60).forEach((r) => {
        const status = r.review?.status || 'pending';
        const reviewers = (r.review?.reviewers || []).map((x) => x.name).join(', ');
        const provenance = r.review?.by_provenance
            ? t('admin.byProvenance', { origins: (r.origins || []).join(', ') })
            : (reviewers ? t('admin.reviewedBy', { names: reviewers }) : t('admin.neverReviewed'));
        const statusKey = { pending: 'admin.filterPending', approved: 'admin.statusApproved', rejected: 'admin.statusRejected' }[status];

        const typeOptions = adminLibraryTypes
            .map((ty) => `<option value="${escapeHtml(ty.value)}"${ty.value === r.type ? ' selected' : ''}>${escapeHtml(libraryTypeLabel(ty.value, ty.label))}</option>`)
            .join('');

        const item = document.createElement('div');
        item.className = 'review-item';
        item.innerHTML = `
            <div class="review-item-head">
                <span class="review-status review-status-${escapeHtml(status)}">${escapeHtml(statusKey ? t(statusKey) : status)}</span>
                ${safeHref(r.url)
                    ? `<a href="${escapeHtml(safeHref(r.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title || r.url)}</a>`
                    : escapeHtml(r.title || r.url || '')}
                <span class="review-meta">${escapeHtml(r.publisher || '')}</span>
            </div>
            <p class="review-meta">${escapeHtml(r.url)}</p>
            <div class="review-fields">
                <label>${escapeHtml(t('admin.reviewTitle'))}
                    <input type="text" data-field="title" value="${escapeHtml(r.title || '')}">
                </label>
                <label>${escapeHtml(t('admin.reviewType'))}
                    <select data-field="type">${typeOptions}</select>
                </label>
                <label>${escapeHtml(t('admin.reviewTopics'))}
                    <input type="text" data-field="topics" value="${escapeHtml((r.topics || []).join(', '))}">
                </label>
                <label>${escapeHtml(t('admin.reviewDescription'))}
                    <input type="text" data-field="description" value="${escapeHtml(r.description || '')}">
                </label>
            </div>
            <div class="review-actions">
                <button type="button" class="btn btn-primary" data-action="approved">${escapeHtml(t('admin.approve'))}</button>
                <button type="button" class="btn btn-secondary" data-action="rejected">${escapeHtml(t('admin.reject'))}</button>
            </div>
            <p class="review-meta">${escapeHtml(t('admin.reviewCited', { n: r.used_in_responses || 0, provenance }))}</p>
        `;

        item.querySelectorAll('[data-action]').forEach((btn) => {
            btn.addEventListener('click', () => submitResourceReview(item, r.url, btn.dataset.action));
        });

        queue.appendChild(item);
    });
}

async function submitResourceReview(item, url, status) {
    const messageEl = document.getElementById('reviewMessage');
    const showMessage = (text, type) => {
        if (!messageEl) return;
        messageEl.textContent = text;
        messageEl.className = `form-message ${type}`;
    };

    const field = (name) => item.querySelector(`[data-field="${name}"]`)?.value.trim() ?? '';
    const topics = field('topics').split(',').map((t) => t.trim()).filter(Boolean);

    const buttons = Array.from(item.querySelectorAll('[data-action]'));
    buttons.forEach((b) => { b.disabled = true; });

    try {
        const response = await fetch(`${API_BASE_URL}/admin/library/review`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${adminToken}`,
            },
            body: JSON.stringify({
                url,
                status,
                type: field('type'),
                topics,
                title: field('title'),
                description: field('description'),
            }),
        });
        const data = await response.json();

        if (!response.ok) {
            showMessage(apiError(data, 'admin.reviewError'), 'error');
            return;
        }

        showMessage(t(status === 'approved' ? 'admin.reviewApproved' : 'admin.reviewRejected'), 'success');
        loadResourceReviewQueue();
    } catch (error) {
        console.error('Resource review error:', error);
        showMessage(t('common.serverUnreachable'), 'error');
    } finally {
        buttons.forEach((b) => { b.disabled = false; });
    }
}
