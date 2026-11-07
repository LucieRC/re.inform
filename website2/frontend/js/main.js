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

function ensureMultiTagDropdown(selectEl, { placeholder = 'All topics' } = {}) {
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
    triggerEl.textContent = `${selected.length} topics`;
}

// ========================================
// Navigation panel (off-canvas below 1100px — see styles.css)
// ========================================

function setupNavPanel() {
    const toggle = document.querySelector('.mobile-menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    if (!toggle || !navLinks) return;

    // One backdrop per page, created here so every page gets it without
    // repeating markup in a dozen HTML files.
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
    const wide = window.matchMedia('(min-width: 1101px)');
    const syncToWidth = () => {
        if (wide.matches && isOpen()) setOpen(false);
    };
    wide.addEventListener('change', syncToWidth);
}

// Expose the sticky header's height as --header-height so sticky bars below it
// (the filter bar) stop right under the header instead of sliding beneath it.
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

    // Check authentication state first
    await checkAuthState();
    
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
            updateNavForLoggedOutUser();
        }
    } catch (error) {
        console.error('Error checking auth state:', error);
        currentUser = null;
        updateNavForLoggedOutUser();
    }
}

function updateNavForLoggedInUser() {
    const authLinks = document.getElementById('authLinks');
    if (!authLinks) return;

    const roleRaw = (currentUser?.role || '').toString().trim();
    const roleNormalized = roleRaw.toLowerCase();
    const shouldShowRoleBadge = roleRaw && roleNormalized !== 'registered';
    const roleBadgeHtml = shouldShowRoleBadge
        ? `<span class="role-badge ${roleNormalized}">${roleRaw}</span>`
        : '';

    const usernameEscaped = escapeHtml(currentUser?.username || 'Profile');

    // Real profile picture when one is uploaded; otherwise the default icon.
    const avatarMarkup = currentUser?.avatar_url
        ? `<img class="profile-image" src="${currentUser.avatar_url}" alt="" aria-hidden="true" onerror="this.classList.add('profile-image-missing')">`
        : '';

    authLinks.innerHTML = `
        <div class="user-menu">
            <a href="profile.html" class="user-info" aria-label="Profile (${usernameEscaped})" title="${usernameEscaped}">
                <span class="profile-icon ${currentUser?.avatar_url ? 'profile-icon-avatar' : ''}" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M20 21a8 8 0 0 0-16 0" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M12 13a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    ${avatarMarkup}
                </span>
                ${roleBadgeHtml}
            </a>
            <a href="#" onclick="logout(); return false;" class="btn btn-secondary" style="padding: 0.5rem 1rem;">Logout</a>
        </div>
    `;
}

function updateNavForLoggedOutUser() {
    const authLinks = document.getElementById('authLinks');
    if (!authLinks) return;
    
    authLinks.innerHTML = `
        <a href="${authHref()}">Log in</a>
    `;
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

function initLoginPage() {
    const form = document.getElementById('loginForm');
    if (!form) return;

    // Keep the return destination when switching between login and register.
    document.querySelectorAll('a[href="register.html"]').forEach((a) => { a.href = `register.html${window.location.search}`; });
    
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
        
        loginBtn.disabled = true;
        loginBtn.textContent = 'Logging in...';
        
        try {
            const response = await fetch(`${API_BASE_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, password, remember })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                formMessage.textContent = 'Login successful! Redirecting...';
                formMessage.className = 'form-message success';
                setTimeout(() => {
                    window.location.href = postAuthDestination();
                }, 1000);
            } else {
                formMessage.textContent = data.error || 'Login failed';
                formMessage.className = 'form-message error';
            }
        } catch (error) {
            console.error('Login error:', error);
            formMessage.textContent = 'Network error. Please try again.';
            formMessage.className = 'form-message error';
        } finally {
            loginBtn.disabled = false;
            loginBtn.textContent = 'Login';
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
        
        // Validate passwords match
        if (password !== confirmPassword) {
            formMessage.textContent = 'Passwords do not match';
            formMessage.className = 'form-message error';
            return;
        }
        
        registerBtn.disabled = true;
        registerBtn.textContent = 'Creating account...';
        
        try {
            const response = await fetch(`${API_BASE_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, email, password })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                formMessage.textContent = 'Account created! Redirecting...';
                formMessage.className = 'form-message success';
                setTimeout(() => {
                    window.location.href = postAuthDestination();
                }, 1000);
            } else {
                formMessage.textContent = data.error || 'Registration failed';
                formMessage.className = 'form-message error';
            }
        } catch (error) {
            console.error('Registration error:', error);
            formMessage.textContent = 'Network error. Please try again.';
            formMessage.className = 'form-message error';
        } finally {
            registerBtn.disabled = false;
            registerBtn.textContent = 'Create account';
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
            window.location.href = 'login.html';
            return;
        }
        profileUserId = currentUser.id;
        isOwnProfile = true;
    }
    
    setupProfileControls();

    // Load profile data
    await loadProfileData();

    // If own profile, load following sections
    if (isOwnProfile) {
        document.getElementById('levelBar').style.display = 'block';
        document.getElementById('recommendationsSection').style.display = 'block';
        document.getElementById('followingActivitySection').style.display = 'block';
        document.getElementById('statusTrackerSection').style.display = 'block';
        document.getElementById('tagsFollowingSection').style.display = 'block';
        document.getElementById('followingSection').style.display = 'block';
        document.getElementById('contributionsTitle').textContent = 'Your contributions';
        document.getElementById('rolesSection').style.display = 'block';
        await Promise.all([
            loadMyRoles(),
            loadRecommendations(),
            loadFollowingActivity(),
            loadFollowedTags(),
            loadFollowingProfiles(),
            loadContributionStatus(),
        ]);
    } else {
        document.getElementById('contributionsTitle').textContent = 'Contributions';
        // Show follow button if logged in and viewing someone else
        if (currentUser) {
            document.getElementById('followButtonContainer').style.display = 'block';
        }
    }
}

// Wire up the profile page's buttons and modal (no inline handlers, so keyboard
// and focus behave and the markup stays clean).
function setupProfileControls() {
    document.getElementById('followButton')?.addEventListener('click', toggleFollow);
    document.getElementById('addTagsBtn')?.addEventListener('click', openTagSelector);
    setupAvatarControls();

    const modal = document.getElementById('tagSelectorModal');
    if (modal && !modal.dataset.bound) {
        modal.dataset.bound = 'true';
        modal.querySelectorAll('[data-close-tag-selector]').forEach((btn) => {
            btn.addEventListener('click', closeTagSelector);
        });
        // Backdrop click and Escape close the dialog, like the argument composer.
        modal.addEventListener('click', (e) => { if (e.target === modal) closeTagSelector(); });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display !== 'none') closeTagSelector();
        });
    }
}

// ---- Level bar (reputation → progression, per the gamification design) ----
const LEVELS = [
    { name: 'Observer', min: 0 },
    { name: 'Engager', min: 51 },
    { name: 'Strategist', min: 201 },
    { name: 'Researcher', min: 501 },
    { name: 'Community Leader', min: 1001 },
];

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
        nextEl.textContent = `${next.min - p} pts to ${next.name}`;
    } else {
        pct = 100;
        nextEl.textContent = 'Top level reached';
    }
    pct = Math.max(0, Math.min(100, pct));
    fill.style.width = `${pct}%`;
    track.setAttribute('aria-valuenow', String(pct));
    track.setAttribute('aria-valuetext', next ? `${current.name}, ${pct}% to ${next.name}` : current.name);
}

// ---- Contributor roles (function, not rights: see GAMIFICATION_DESIGN.md) ----

function renderRoleChips(roles) {
    const el = document.getElementById('profileRoles');
    if (!el) return;
    el.innerHTML = roles.map((r) => {
        const confirmed = r.tier === 'confirmed';
        const title = `${confirmed ? 'Confirmed' : 'Apprentice'} ${r.name}`;
        return `<span class="role-chip${confirmed ? ' is-confirmed' : ''}" title="${escapeHtml(title)}">
            ${icon(r.icon)}${escapeHtml(r.name)}${confirmed ? ` ${icon('check')}` : ''}</span>`;
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
        container.innerHTML = '<p class="empty-message">Could not load roles.</p>';
    }
}

function renderMyRoles(catalog, mine) {
    const container = document.getElementById('rolesContent');
    const held = new Map(mine.map((r) => [r.role, r]));

    container.innerHTML = catalog.map((c) => {
        const r = held.get(c.role);
        let body;
        if (!r) {
            body = `<p>${escapeHtml(c.duty)}</p>`;
        } else if (r.tier === 'confirmed') {
            body = `<p>${escapeHtml(c.power)}</p>`;
        } else if (r.progress) {
            const pct = Math.min(100, Math.round((r.progress.done / r.progress.needed) * 100));
            body = `<p>${escapeHtml(c.duty)}</p>
                <div class="role-progress">
                    <div class="level-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"
                         aria-label="Progress to confirmed ${escapeHtml(c.name)}">
                        <span class="level-fill" style="width: ${pct}%"></span>
                    </div>
                    <span class="role-progress-label">${r.progress.done} / ${r.progress.needed} ${escapeHtml(r.progress.unit)}</span>
                </div>`;
        } else {
            body = `<p>${escapeHtml(c.power)}</p>`;
        }
        const privacy = r && !c.public ? '<p>Only you can see this role.</p>' : '';
        const tier = r ? `<span class="role-tier${r.tier === 'confirmed' ? ' tier-confirmed' : ''}">${r.tier === 'confirmed' ? 'Confirmed' : 'Apprentice'}</span>` : '';
        const action = r
            ? `<button type="button" class="link-btn" data-role-drop="${c.role}">Step down</button>`
            : `<button type="button" class="btn btn-secondary btn-small" data-role-take="${c.role}">Take this role</button>`;
        return `
            <div class="role-card${r ? ' is-held' : ''}">
                <div class="role-card-head">
                    ${icon(c.icon)}
                    <span class="role-card-name">${escapeHtml(c.name)}</span>
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
            showToast(data.error || 'Could not update your role.');
            return;
        }
        showToast(method === 'POST'
            ? 'Role taken. Your work will confirm you in it.'
            : 'You stepped down. Your past work still counts if you come back.');
        await Promise.all([loadMyRoles(), loadProfileData()]);
    } catch (error) {
        console.error('Role change error:', error);
        showToast('Network error while updating your role.');
    }
}

async function loadSourceQueue() {
    const container = document.getElementById('sourceQueueContent');
    if (!container) return;
    try {
        const res = await fetch(`${API_BASE_URL}/me/source-check-queue?limit=3`, { credentials: 'include' });
        if (!res.ok) throw new Error('Failed to load queue');
        const items = (await res.json()).items || [];
        if (!items.length) {
            container.innerHTML = '<p class="empty-message">Nothing waiting for you right now. Thank you!</p>';
            return;
        }
        container.innerHTML = items.map((it) => {
            const hosts = (it.resources || []).map((r) => hostnameOf(r.url)).join(', ');
            const checks = it.check_count
                ? `${it.check_count} check${it.check_count === 1 ? '' : 's'} so far`
                : 'No check yet';
            return `
                <a class="queue-item" href="argument.html?id=${encodeURIComponent(it.argument_id)}&check=${encodeURIComponent(it.counter_argument_id)}">
                    <span class="queue-claim">“${escapeHtml(it.argument_text || '')}”</span>
                    <span class="queue-text">${escapeHtml(it.text || '')}</span>
                    <span class="queue-meta">${it.persona ? `For ${escapeHtml(it.persona)} · ` : ''}${icon('clip')} ${escapeHtml(hosts)} · ${checks}</span>
                </a>`;
        }).join('');
    } catch (error) {
        console.error('Source queue error:', error);
        container.innerHTML = '<p class="empty-message">Could not load the queue.</p>';
    }
}

// ---- Contribution status tracker ----
const STATUS_META = {
    pending:  { icon: icon('hourglass'), label: 'Pending review', cls: 'status-pending' },
    approved: { icon: icon('check'), label: 'Approved',        cls: 'status-approved' },
    rejected: { icon: icon('x'), label: 'Needs work',      cls: 'status-rejected' },
};

async function loadContributionStatus() {
    const container = document.getElementById('statusTrackerContent');
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE_URL}/me/contributions`, { credentials: 'include' });
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        const data = await res.json();
        renderContributionStatus(data.items || [], data.counts || {});
    } catch (error) {
        console.error('Status tracker error:', error);
        container.innerHTML = '<p class="empty-message">Couldn\'t load your submissions right now.</p>';
    }
}

function renderContributionStatus(items, counts) {
    const container = document.getElementById('statusTrackerContent');
    if (!container) return;

    if (!items.length) {
        container.innerHTML = `
            <p class="empty-message">You haven't submitted anything yet.
            <a href="contribute.html">Answer a claim or share a source →</a></p>`;
        return;
    }

    const summary = ['pending', 'approved', 'rejected']
        .filter((s) => counts[s])
        .map((s) => `<span class="status-chip ${STATUS_META[s].cls}">${STATUS_META[s].icon} ${counts[s]} ${STATUS_META[s].label.toLowerCase()}</span>`)
        .join('');

    const rows = items.map((it) => {
        const meta = STATUS_META[it.status] || STATUS_META.pending;
        const kind = it.kind === 'source'
            ? `<span class="tracker-kind">${icon('book')} Source</span>`
            : `<span class="tracker-kind">${icon('message')} Response</span>`;
        const when = it.submitted_at
            ? `<time datetime="${escapeHtml(it.submitted_at)}">${new Date(it.submitted_at).toLocaleDateString()}</time>`
            : '—';
        const link = it.url
            ? `<a href="${escapeHtml(it.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(it.title)}</a>`
            : escapeHtml(it.title);
        return `
            <tr>
                <td data-label="Type">${kind}</td>
                <td data-label="What">${link}</td>
                <td data-label="Sent">${when}</td>
                <td data-label="Status"><span class="status-chip ${meta.cls}">${meta.icon} ${escapeHtml(meta.label)}</span></td>
            </tr>`;
    }).join('');

    container.innerHTML = `
        ${summary ? `<div class="status-summary">${summary}</div>` : ''}
        <div class="table-scroll">
            <table class="status-table">
                <thead>
                    <tr><th scope="col">Type</th><th scope="col">What</th><th scope="col">Sent</th><th scope="col">Status</th></tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

// ---- Recommended for you (claims matched to followed tags) ----
async function loadRecommendations() {
    const container = document.getElementById('recommendationsContent');
    if (!container) return;

    try {
        const [tagsRes, argsRes] = await Promise.all([
            fetch(`${API_BASE_URL}/user/${currentUser.id}/followed-tags`, { credentials: 'include' }),
            // Credentialed so the API can flag claims we authored or already answered.
            fetch(`${API_BASE_URL}/arguments`, { credentials: 'include' }),
        ]);
        const followed = tagsRes.ok ? (await tagsRes.json()).followed_tags.map((t) => t.tag) : [];
        const args = argsRes.ok ? await argsRes.json() : [];
        renderRecommendations(Array.isArray(args) ? args : [], followed);
    } catch (error) {
        console.error('Recommendations error:', error);
        container.innerHTML = '<p class="empty-message">Couldn\'t load recommendations right now.</p>';
    }
}

function renderRecommendations(args, followedTags) {
    const container = document.getElementById('recommendationsContent');
    if (!container) return;

    // No declared interests yet → invite the user to pick some (progressive
    // profiling: this single action powers the whole section).
    if (!followedTags.length) {
        container.innerHTML = `
            <p class="empty-message">Tell us which topics you care about and we'll suggest claims to work on.
            <button type="button" class="btn btn-secondary btn-small" id="recAddTags">+ Choose topics</button></p>`;
        document.getElementById('recAddTags')?.addEventListener('click', openTagSelector);
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
        container.innerHTML = `
            <p class="empty-message">No open claims in your topics right now — nice work.
            <a href="index.html">Browse all claims →</a></p>`;
        return;
    }

    container.innerHTML = '';
    scored.forEach(({ a, matched }) => {
        const count = a.counter_argument_count || 0;
        const why = `Because you follow ${matched.slice(0, 2).map((t) => `#${escapeHtml(t)}`).join(' ')}`;
        const urgencyNote = count === 0
            ? `<span class="needs-badge needs-badge-hot">${icon('zap')} No response yet</span>`
            : `<span class="needs-badge">${count} response${count === 1 ? '' : 's'}</span>`;

        const card = document.createElement('a');
        card.className = 'needs-card';
        card.href = `argument.html?id=${encodeURIComponent(a.id)}`;
        card.innerHTML = `
            <p class="rec-why">${icon('target')} ${why}</p>
            <p class="needs-quote">"${formatClaimText(a.text)}"</p>
            <div class="needs-meta">${urgencyNote}</div>
            <span class="needs-cta">${count === 0 ? `${icon('pen')} Be the first to respond →` : `${icon('message')} Add your response →`}</span>`;
        container.appendChild(card);
    });
}

// ---- Activity from people you follow ----
function timeAgo(iso) {
    const then = new Date(iso).getTime();
    if (!then) return '';
    const secs = Math.max(1, Math.floor((Date.now() - then) / 1000));
    const units = [['year', 31536000], ['month', 2592000], ['week', 604800], ['day', 86400], ['hour', 3600], ['minute', 60]];
    for (const [name, s] of units) {
        const v = Math.floor(secs / s);
        if (v >= 1) return `${v} ${name}${v === 1 ? '' : 's'} ago`;
    }
    return 'just now';
}

async function loadFollowingActivity() {
    const container = document.getElementById('followingActivityContent');
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE_URL}/me/following-activity`, { credentials: 'include' });
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        const data = await res.json();
        renderFollowingActivity(data.items || []);
    } catch (error) {
        console.error('Following activity error:', error);
        container.innerHTML = '<p class="empty-message">Couldn\'t load activity right now.</p>';
    }
}

const ACTIVITY_ICON = { response: icon('message'), source: icon('book') };

function renderFollowingActivity(items) {
    const container = document.getElementById('followingActivityContent');
    if (!container) return;

    if (!items.length) {
        container.innerHTML = `
            <p class="empty-message">No recent activity from people you follow.
            <a href="index.html">Find people to follow →</a></p>`;
        return;
    }

    container.innerHTML = items.map((it) => {
        const kindIcon = ACTIVITY_ICON[it.kind] || '•';
        const who = `<a href="profile.html?id=${encodeURIComponent(it.user.id)}" class="activity-user">${escapeHtml(it.user.username)}</a>`;
        const what = it.href
            ? `<a href="${escapeHtml(it.href)}"${it.kind === 'source' ? ' target="_blank" rel="noopener noreferrer"' : ''}>"${escapeHtml(it.title)}"</a>`
            : `"${escapeHtml(it.title)}"`;
        const when = it.at ? `<time datetime="${escapeHtml(it.at)}" class="activity-time">${escapeHtml(timeAgo(it.at))}</time>` : '';
        return `
            <div class="activity-item">
                <span class="activity-icon" aria-hidden="true">${kindIcon}</span>
                <p class="activity-text">${who} ${escapeHtml(it.verb)} ${what}</p>
                ${when}
            </div>`;
    }).join('');
}

async function loadProfileData() {
    try {
        const response = await fetch(`${API_BASE_URL}/user/${profileUserId}`, {
            credentials: 'include'
        });
        
        if (!response.ok) {
            throw new Error('Failed to load profile');
        }
        
        const data = await response.json();
        
        // Update profile header
        document.getElementById('profileUsername').textContent = data.user.username;
        document.getElementById('profileRole').textContent = data.user.role;
        document.getElementById('profileRole').className = `role-badge ${data.user.role}`;
        document.getElementById('profileMemberSince').innerHTML =
            `Member since <time datetime="${escapeHtml(data.user.created_at)}">${escapeHtml(new Date(data.user.created_at).toLocaleDateString())}</time>`;

        // Profile picture (works for own and other profiles)
        renderProfileAvatar(data.user);
        
        // Update stats
        document.getElementById('reputationScore').textContent = data.user.reputation_score;
        document.getElementById('contributionCount').textContent = data.contribution_count;
        document.getElementById('followingCount').textContent = data.following_count;
        document.getElementById('followersCount').textContent = data.followers_count;

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
        document.title = `${data.user.username} - re.inform`;
        
    } catch (error) {
        console.error('Error loading profile:', error);
        document.getElementById('profileUsername').textContent = 'User not found';
    }
}

function updateFollowButton() {
    const btn = document.getElementById('followButton');
    if (!btn) return;
    
    if (isFollowingProfile) {
        btn.textContent = 'Unfollow';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
    } else {
        btn.textContent = 'Follow';
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
    if (user.avatar_url) {
        if (!img) {
            img = document.createElement('img');
            img.id = 'profileAvatarImg';
            img.alt = '';
            img.onerror = () => img.remove();
            wrap.appendChild(img);
        }
        img.src = user.avatar_url;
    } else if (img) {
        img.remove();
    }

    // Upload controls only make sense on your own profile (needs login anyway).
    const actions = document.getElementById('avatarActions');
    if (actions) {
        actions.style.display = (isOwnProfile && currentUser) ? 'block' : 'none';
    }
}

function setupAvatarControls() {
    const changeBtn = document.getElementById('changeAvatarBtn');
    const fileInput = document.getElementById('avatarInput');
    const errorEl = document.getElementById('avatarError');
    if (!changeBtn || !fileInput) return;

    changeBtn.addEventListener('click', () => {
        if (!currentUser) {
            window.location.href = 'login.html';
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

        const originalText = changeBtn.textContent;
        changeBtn.disabled = true;
        changeBtn.textContent = 'Uploading…';

        try {
            const res = await fetch(`${API_BASE_URL}/me/avatar`, {
                method: 'POST',
                credentials: 'include',
                body: form
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                errorEl.textContent = data.error || 'Upload failed.';
                errorEl.style.display = 'block';
                return;
            }

            if (currentUser) currentUser.avatar_url = data.avatar_url;
            renderProfileAvatar({
                username: currentUser?.username || '',
                avatar_url: data.avatar_url
            });
        } catch (error) {
            console.error('Avatar upload error:', error);
            errorEl.textContent = 'Upload failed. Please try again.';
            errorEl.style.display = 'block';
        } finally {
            changeBtn.disabled = false;
            changeBtn.textContent = originalText;
        }
    });
}

async function toggleFollow() {
    if (!currentUser) {
        window.location.href = 'login.html';
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
        container.innerHTML = '<p class="empty-message">No endorsements yet.</p>';
        return;
    }
    
    container.innerHTML = endorsements.map(e => `
        <div class="endorsement-card">
            <div class="endorser">
                <a href="profile.html?id=${e.endorser_id}" class="contributor-link">${e.endorser_username}</a>
            </div>
            <div class="reason">"${e.reason}"</div>
            <div class="date">${new Date(e.created_at).toLocaleDateString()}</div>
        </div>
    `).join('');
}

function displayProfileContributions(contributions) {
    const container = document.getElementById('contributionsList');
    
    if (contributions.length === 0) {
        const message = isOwnProfile 
            ? 'No contributions yet. <a href="contribute.html">Write your first response!</a>'
            : 'No contributions yet.';
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

async function loadFollowedTags() {
    const container = document.getElementById('followedTagsContent');
    
    try {
        const response = await fetch(`${API_BASE_URL}/user/${currentUser.id}/followed-tags`, {
            credentials: 'include'
        });
        
        if (!response.ok) throw new Error('Failed to load followed tags');
        
        const data = await response.json();
        
        if (data.followed_tags.length === 0) {
            container.innerHTML = '<p class="empty-message">You\'re not following any tags yet. Click "Add tags" to get started!</p>';
            return;
        }
        
        container.innerHTML = '';
        data.followed_tags.forEach(({ tag }) => {
            const wrapper = document.createElement('span');
            wrapper.className = 'followed-tag';

            const link = document.createElement('a');
            link.className = 'followed-tag-link';
            link.href = `resources.html?tag=${encodeURIComponent(tag)}`;
            link.textContent = `#${tag}`;

            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'tag-remove';
            removeBtn.title = 'Unfollow';
            removeBtn.textContent = '×';
            removeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                unfollowTag(tag);
            });

            wrapper.appendChild(link);
            wrapper.appendChild(removeBtn);
            container.appendChild(wrapper);
        });
        
    } catch (error) {
        console.error('Error loading followed tags:', error);
        container.innerHTML = '<p class="error-message">Failed to load tags.</p>';
    }
}

async function unfollowTag(tag) {
    try {
        const response = await fetch(`${API_BASE_URL}/user/${currentUser.id}/followed-tags/${tag}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
        if (response.ok) {
            await loadFollowedTags();
        }
    } catch (error) {
        console.error('Error unfollowing tag:', error);
    }
}

async function followTag(tag) {
    try {
        const response = await fetch(`${API_BASE_URL}/user/${currentUser.id}/followed-tags`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ tag: tag })
        });
        
        if (response.ok) {
            await loadFollowedTags();
            // Update the tag selector UI
            const tagEl = document.querySelector(`[data-tag="${tag}"]`);
            if (tagEl) {
                tagEl.classList.add('selected');
            }
        }
    } catch (error) {
        console.error('Error following tag:', error);
    }
}

async function openTagSelector() {
    const modal = document.getElementById('tagSelectorModal');
    const grid = document.getElementById('tagSelectorGrid');

    // Remember what opened the dialog so focus can return there on close.
    tagSelectorReturnFocus = document.activeElement;

    // Load all available tags
    try {
        const [tagsResponse, followedResponse] = await Promise.all([
            fetch(`${API_BASE_URL}/tags`),
            fetch(`${API_BASE_URL}/user/${currentUser.id}/followed-tags`, { credentials: 'include' })
        ]);

        const tagsData = await tagsResponse.json();
        const followedData = await followedResponse.json();
        const followedSet = new Set(followedData.followed_tags.map(t => t.tag));

        grid.innerHTML = tagsData.tags.map(tag => `
            <button
                type="button"
                class="tag-selector-item ${followedSet.has(tag) ? 'selected' : ''}"
                data-tag="${escapeHtml(tag)}"
                aria-pressed="${followedSet.has(tag) ? 'true' : 'false'}"
            >
                #${escapeHtml(tag)}
            </button>
        `).join('');

        // Delegated handler instead of inline onclick (keeps markup clean and
        // works for keyboard activation).
        if (!grid.dataset.bound) {
            grid.dataset.bound = 'true';
            grid.addEventListener('click', (e) => {
                const btn = e.target.closest('.tag-selector-item');
                if (btn) toggleTagFollow(btn.dataset.tag);
            });
        }
    } catch (error) {
        console.error('Error loading tags:', error);
        grid.innerHTML = '<p class="error-message">Failed to load tags.</p>';
    }

    modal.style.display = 'flex';
    // Move focus into the dialog for keyboard/screen-reader users.
    modal.querySelector('.modal-close')?.focus();
}

let tagSelectorReturnFocus = null;

function closeTagSelector() {
    document.getElementById('tagSelectorModal').style.display = 'none';
    // Return focus to whatever opened the dialog.
    if (tagSelectorReturnFocus && typeof tagSelectorReturnFocus.focus === 'function') {
        tagSelectorReturnFocus.focus();
    }
    tagSelectorReturnFocus = null;
}

async function toggleTagFollow(tag) {
    const tagEl = document.querySelector(`.tag-selector-item[data-tag="${tag}"]`);
    const isSelected = tagEl.classList.contains('selected');

    if (isSelected) {
        await unfollowTag(tag);
        tagEl.classList.remove('selected');
    } else {
        await followTag(tag);
        tagEl.classList.add('selected');
    }
    tagEl.setAttribute('aria-pressed', String(!isSelected));
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
        
        if (!response.ok) throw new Error('Failed to load following');
        
        const data = await response.json();
        
        if (data.following.length === 0) {
            container.innerHTML = '<p class="empty-message">You\'re not following anyone yet. Visit contributor profiles to follow them!</p>';
            return;
        }
        
        container.innerHTML = data.following.map(user => `
            <div class="user-card">
                <a href="profile.html?id=${encodeURIComponent(user.id)}" class="user-card-link">
                    <div class="user-card-name">${escapeHtml(user.username)}</div>
                    <span class="role-badge ${escapeHtml(user.role)}">${escapeHtml(user.role)}</span>
                </a>
                <button type="button" class="btn btn-secondary btn-small" data-unfollow="${encodeURIComponent(user.id)}">Unfollow</button>
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
        container.innerHTML = '<p class="error-message">Failed to load following.</p>';
    }
}

async function unfollowUser(userId) {
    try {
        const response = await fetch(`${API_BASE_URL}/user/${userId}/follow`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
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
    { id: 'quick', icon: 'clock', title: 'I have a few minutes', who: 'Anyone — no expertise needed', tasks: ['prioritize', 'rate', 'claim'] },
    { id: 'talk', icon: 'message', title: 'I discuss these topics with people around me', who: 'Family, friends, colleagues, online', tasks: ['claim', 'answer', 'personas'] },
    { id: 'science', icon: 'flask', title: "I'm comfortable with science and sources", who: 'Researcher, student, fact-checker', tasks: ['review', 'resource', 'answer'] },
    { id: 'write', icon: 'pen', title: 'I write, teach or communicate', who: 'Journalist, teacher, activist', tasks: ['answer', 'improve', 'personas'] },
    { id: 'visual', icon: 'palette', title: 'I design or illustrate', who: 'Graphic designer, illustrator', tasks: ['visual', 'personas'] },
    { id: 'tech', icon: 'code', title: 'I code or work with data', who: 'Developer, data scientist', tasks: ['code', 'data'] },
];

// `open` reveals a workspace on this page; `href` (or `hrefFrom`, computed once
// claims are loaded) leads to where the work happens.
const CONTRIBUTOR_TASKS = {
    prioritize: { icon: 'target', title: 'Vote on the claims you hear most', desc: 'One tap per claim. It decides what gets answered first.', time: '1 min', account: true, open: 'ws-prioritize' },
    rate: { icon: 'star', title: 'Rate an answer', desc: 'Read a response and say whether it would convince the person it targets.', time: '30 sec', account: true, hrefFrom: 'unrated' },
    claim: { icon: 'plus', title: "Add a claim you've heard", desc: 'Just the sentence. No source or answer needed.', time: '1 min', account: true, open: 'ws-claim' },
    answer: { icon: 'message', title: 'Answer a claim for a specific person', desc: 'Pick a claim and a persona that has no answer yet. A few sentences are enough.', time: '5 min', account: true, open: 'ws-answer' },
    personas: { icon: 'users', title: 'Get to know the personas', desc: 'Five kinds of people you may talk to, and what moves each of them.', time: '5 min', account: false, href: 'personalities.html' },
    review: { icon: 'search', title: 'Fact-check the answers to a claim', desc: 'Check the sources, rate the answers, and propose a corrected version if something is off.', time: '10 min', account: true, hrefFrom: 'review' },
    resource: { icon: 'book', title: 'Share a study, article or report', desc: 'Add evidence to the shared library so anyone can cite it.', time: '2 min', account: true, open: 'ws-resource' },
    improve: { icon: 'pen', title: 'Make an existing answer more convincing', desc: 'Open a claim and use “Propose an improved response” on an answer that sounds too dry or too preachy.', time: '10 min', account: true, hrefFrom: 'review' },
    visual: { icon: 'image', title: 'Turn an answer into a visual', desc: 'Infographics travel further than text. Make one, then share its link in the library.', time: '1 h+', account: true, open: 'ws-resource' },
    code: { icon: 'code', title: 'Improve the website', desc: 'Flask + plain JavaScript. Pick an open issue or propose your own on GitHub.', time: 'Flexible', account: false, href: `${REPO_URL}/issues`, external: true },
    data: { icon: 'chart', title: 'Work on the persona research', desc: 'Personas are built from European Values Study survey data with a Python pipeline. Help refine the method.', time: 'Flexible', account: false, href: REPO_URL, external: true },
};

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
    const t = CONTRIBUTOR_TASKS[taskId];
    const el = document.createElement(t.open ? 'button' : 'a');
    el.className = `help-path${isFirst ? ' help-path-primary' : ''}`;
    el.dataset.task = taskId;
    if (t.open) {
        el.type = 'button';
        el.setAttribute('aria-controls', t.open);
        el.setAttribute('aria-expanded', 'false');
        el.addEventListener('click', () => openContributorTask(taskId));
    } else {
        el.href = taskHref(t);
        if (t.external) { el.target = '_blank'; el.rel = 'noopener noreferrer'; }
    }
    const accountTag = t.account && !currentUser ? ' · free account' : '';
    el.innerHTML = `
        <span class="help-path-icon">${icon(t.icon)}</span>
        <span class="help-path-body">
            <span class="help-path-title">${escapeHtml(t.title)}</span>
            <span class="help-path-desc">${escapeHtml(t.desc)}</span>
        </span>
        <span class="help-path-effort">${escapeHtml(t.time + accountTag)}</span>
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
        hint.innerHTML = currentUser ? '' : `
            <p class="onb-account onb-account-inline">You'll need a free account to save this.
                <a href="${authHref('register.html')}">Create one</a> or <a href="${authHref('login.html')}">log in</a> — you'll come back right here.</p>`;
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
    const params = new URLSearchParams();
    if (contributeState.profileId) params.set('as', contributeState.profileId);
    if (contributeState.taskId) params.set('task', contributeState.taskId);
    const qs = params.toString();
    history.replaceState(null, '', `${window.location.pathname}${qs ? `?${qs}` : ''}`);
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
        contributeState.archetypes = (Array.isArray(personas) ? personas : []).filter((p) => p && p.layer === 'archetype');
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
        el.innerHTML = '<p class="empty-message">Every claim has an answer for every persona right now. <a href="index.html">Browse the claims</a> to improve one.</p>';
        return;
    }

    el.innerHTML = gaps.map(({ a, missing }) => `
        <div class="needs-card">
            <p class="needs-quote">"${formatClaimText(a.text)}"</p>
            <div class="onb-gap-links">
                ${missing.slice(0, 3).map((p) => `
                    <a class="gap-chip" href="argument.html?id=${encodeURIComponent(a.id)}&amp;write=${encodeURIComponent(p.id)}">${icon('pen')} For ${escapeHtml(p.name)}</a>
                `).join('')}
            </div>
        </div>
    `).join('') + '<p class="onb-more"><a href="index.html">See all claims →</a></p>';
}

// Home feed: the "Heard a claim…" prompt opens the claim form in place.
function setupFeedComposer() {
    const toggle = document.getElementById('feedComposerToggle');
    const form = document.getElementById('feedComposer');
    if (!toggle || !form) return;
    const messageEl = document.getElementById('claimFormMessage');

    const open = () => {
        toggle.hidden = true;
        toggle.setAttribute('aria-expanded', 'true');
        form.hidden = false;
        // Logged out: say so up front rather than after they've typed.
        if (!currentUser && !messageEl.textContent) {
            messageEl.innerHTML = `You'll need a free account to post. <a href="${authHref('register.html')}">Create one</a> or <a href="${authHref('login.html')}">log in</a> — you'll come back right here.`;
            messageEl.className = 'form-message';
        }
        form.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        document.getElementById('claimText').focus({ preventScroll: true });
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
    // writing so the draft stays visible; otherwise fold back to the prompt.
    const textArea = document.getElementById('claimText');
    const collapseIfEmpty = (target) => {
        if (form.hidden || form.contains(target) || toggle.contains(target)) return;
        if (textArea.value.trim()) return;
        close({ refocus: false });
    };
    document.addEventListener('pointerdown', (e) => collapseIfEmpty(e.target));
    form.addEventListener('focusout', (e) => {
        if (e.relatedTarget) collapseIfEmpty(e.relatedTarget);
    });

    toggle.addEventListener('click', open);
    document.getElementById('feedComposerCancel').addEventListener('click', () => close());
    document.querySelectorAll('[data-open-composer]').forEach((b) => b.addEventListener('click', open));
    setupClaimForm('feedComposer');
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
            showMessage(`Please <a href="${authHref('register.html')}">create a free account</a> or <a href="${authHref('login.html')}">log in</a> first.`, 'error');
            return;
        }
        const text = document.getElementById('claimText').value.trim();
        if (!text) return;

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
                showMessage(escapeHtml(data.error || 'Could not add the claim.'), 'error');
                return;
            }
            form.reset();
            const id = encodeURIComponent(data.argument?.id);
            showMessage(`Thanks, the claim is online. <a href="argument.html?id=${id}">See it</a> or <a href="argument.html?id=${id}&amp;write=any">write a first answer →</a>`, 'success');
        } catch (error) {
            console.error('Claim submission error:', error);
            showMessage('Could not reach the server. Please try again.', 'error');
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
    { score: 1, label: 'Low', hint: 'Rarely comes up' },
    { score: 3, label: 'Medium', hint: 'Worth covering' },
    { score: 5, label: 'Critical', hint: 'Debunk this first' },
];

function renderPriorityList(args) {
    const listEl = document.getElementById('priorityList');
    if (!listEl) return;

    if (!args.length) {
        listEl.innerHTML = '<p class="empty-message">No claims to rate yet.</p>';
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
            <p class="priority-claim">"${formatClaimText(a.text)}"</p>
            <div class="priority-controls" role="group" aria-label="How urgent is this to debunk?">
                ${PRIORITY_LEVELS.map((lvl) => `
                    <button type="button" class="priority-btn${mine === lvl.score ? ' is-selected' : ''}"
                            data-score="${lvl.score}" title="${escapeHtml(lvl.hint)}"
                            aria-pressed="${mine === lvl.score}">${escapeHtml(lvl.label)}</button>
                `).join('')}
            </div>
            <span class="priority-tally">${
                count
                    ? `${count} rating${count === 1 ? '' : 's'} · avg ${Number(avg).toFixed(1)}/5`
                    : 'No ratings yet'
            }</span>
        `;

        row.querySelectorAll('.priority-btn').forEach((btn) => {
            btn.addEventListener('click', () => submitImportanceRating(row, Number(btn.dataset.score)));
        });

        listEl.appendChild(row);
    });
}

async function submitImportanceRating(row, score) {
    const messageEl = document.getElementById('priorityMessage');
    const showMessage = (text, type) => {
        if (!messageEl) return;
        messageEl.textContent = text;
        messageEl.className = `form-message ${type}`;
    };

    if (!currentUser) {
        showMessage('Please log in to rate which claims matter most.', 'error');
        return;
    }

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
            showMessage(data.error || 'Could not save your rating.', 'error');
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
            tally.textContent = count
                ? `${count} rating${count === 1 ? '' : 's'} · avg ${Number(avg).toFixed(1)}/5`
                : 'No ratings yet';
        }
        showMessage('Thanks — your rating helps set the priorities.', 'success');
    } catch (error) {
        console.error('Importance rating error:', error);
        showMessage('Could not reach the server. Please try again.', 'error');
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
        .sort((a, b) => String(a.text || '').localeCompare(String(b.text || '')))
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
            showMessage('Please log in to share a resource.', 'error');
            return;
        }

        const url = document.getElementById('resourceUrl').value.trim();
        if (!url) {
            showMessage('Please provide a link.', 'error');
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
                showMessage(data.error || 'Could not save the resource.', 'error');
                return;
            }

            form.reset();
            showMessage('Thanks — your resource is now in the shared library.', 'success');
        } catch (error) {
            console.error('Resource submission error:', error);
            showMessage('Could not reach the server. Please try again.', 'error');
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
            `<span class="tag">${capitalizeWords(tag)}</span>`
          ).join('')}</div>`
        : '';
    
    const sourceHTML = contribution.source 
        ? `<div class="card-source"><a href="${contribution.source}" target="_blank" rel="noopener noreferrer">View source</a></div>`
        : '';
    
    const dateHTML = contribution.date 
        ? `<div class="card-date">${new Date(contribution.date).toLocaleDateString()}</div>`
        : '';
    
    // Contributor info - show clickable link if user_id exists, otherwise "Anonymous contributor"
    const avatar = contribution.user_id
        ? userAvatarHtml(contribution.user_id, contribution.username, 'avatar-xs')
        : '';
    const contributorHTML = contribution.user_id && contribution.username
        ? `<a href="profile.html?id=${contribution.user_id}" class="contributor-link">${avatar}<span class="contributor-name">By ${contribution.username}</span></a>`
        : '<span class="anonymous-contributor">Anonymous contributor</span>';

    const detailsParts = [];
    if (narrativeText && quoteText !== narrativeText) {
        detailsParts.push(`<div class="contribution-detail"><strong>Quote:</strong> ${escapeHtml(narrativeText)}</div>`);
    }
    if (counterArgumentText && quoteText !== counterArgumentText) {
        detailsParts.push(`<div class="contribution-detail"><strong>Response:</strong> ${escapeHtml(counterArgumentText)}</div>`);
    }
    if (contribution.title) {
        detailsParts.push(`<div class="contribution-detail"><strong>Title:</strong> ${escapeHtml(contribution.title)}</div>`);
    }
    
    card.innerHTML = `
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

async function initPersonalitiesPage() {
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
        const archetypes = allArr.filter((p) =>
            p && (p.layer === 'archetype' || (p.imported_from && p.imported_from.project === 'BonPoteJS'))
        );
        renderArchetypesGrid(archetypes);

        // Layer 2 — EVS clusters: population grounding.
        const evs = allArr
            .filter((p) => p && p.imported_from && p.imported_from.project === 'EVS');

        evs.sort((a, b) => (Number(b.population_share_w) || 0) - (Number(a.population_share_w) || 0));

        if (loadingMessage) loadingMessage.style.display = 'none';
        if (errorMessage) errorMessage.style.display = 'none';

        if (!grid) return;
        grid.innerHTML = '';

        if (!evs.length) {
            grid.style.display = 'none';
            if (emptyMessage) emptyMessage.style.display = 'block';
            return;
        }

        grid.style.display = 'grid';
        if (emptyMessage) emptyMessage.style.display = 'none';

        function detailsHtml(p) {
            const name = escapeHtml(String(p?.name || p?.persona_code || 'Personality'));
            const share = Number(p?.population_share_w);
            const sharePct = Number.isFinite(share) ? `${(share * 100).toFixed(1)}%` : '';

            const valueSystem = p?.value_system ? escapeHtml(String(p.value_system)) : '';
            const persuasionFormat = p?.persuasion_format ? escapeHtml(String(p.persuasion_format)) : '';
            const keywords = Array.isArray(p?.top_axes_labels)
                ? p.top_axes_labels
                    .map((x) => escapeHtml(String(x)))
                    .filter((x) => x)
                    .slice(0, 6)
                : [];

            return `
                <div class="personality-card-header">
                    <div class="personality-card-name">${name}</div>
                    <div class="personality-icon-mini">
                        ${p?.icon
                            ? `<img class="personality-icon-mini-img" src="${escapeHtml(String(p.icon))}" alt="" loading="lazy" />`
                            : `<span class="personality-icon-mini-fallback">${escapeHtml(p?.persona_code || 'P')}</span>`
                        }
                    </div>
                </div>
                <div class="personality-detail-lines">
                    ${sharePct ? `<div class="personality-detail"><strong>Weight:</strong> ${escapeHtml(sharePct)}</div>` : ''}
                    ${valueSystem ? `<div class="personality-detail"><strong>Value system:</strong> ${valueSystem}</div>` : ''}
                    ${persuasionFormat ? `<div class="personality-detail"><strong>♡</strong> ${persuasionFormat}</div>` : ''}
                    ${keywords.length ? `<div class="personality-detail"><strong>Description:</strong> ${keywords.join(', ')}</div>` : ''}
                </div>
            `;
        }

        evs.forEach((p) => {
            const iconCard = document.createElement('button');
            iconCard.type = 'button';
            iconCard.className = 'counter-argument-card personality-icon-card';

            const name = String(p?.name || p?.persona_code || 'Personality');
            const code = p?.persona_code ? String(p.persona_code) : '';
            const label = `${name}`;

            const iconSrc = p?.icon ? escapeHtml(String(p.icon)) : '';
            iconCard.setAttribute('aria-label', `Toggle details for ${label}`);
            iconCard.setAttribute('aria-expanded', 'false');

            iconCard.innerHTML = `
                <div class="personality-card-inner" aria-hidden="false">
                    <div class="personality-card-flip">
                        <div class="personality-card-face personality-card-front">
                            ${iconSrc
                                ? `<img class="personality-icon-only" src="${iconSrc}" alt="" loading="lazy" />`
                                : `<span class="personality-icon-fallback">${escapeHtml(code || 'P')}</span>`
                            }
                        </div>
                        <div class="personality-card-face personality-card-back">
                            <div class="personality-card-details" aria-hidden="true"></div>
                        </div>
                    </div>
                </div>
            `;

            const detailsEl = iconCard.querySelector('.personality-card-details');
            if (detailsEl) {
                detailsEl.innerHTML = detailsHtml(p);
            }

            iconCard.addEventListener('click', () => {
                const expanded = iconCard.classList.contains('is-expanded');
                iconCard.classList.toggle('is-expanded');
                iconCard.setAttribute('aria-expanded', expanded ? 'false' : 'true');
                if (detailsEl) detailsEl.setAttribute('aria-hidden', expanded ? 'true' : 'false');
            });
            grid.appendChild(iconCard);
        });
    } catch (error) {
        console.error('Error loading personalities:', error);
        if (loadingMessage) loadingMessage.style.display = 'none';
        if (errorMessage) {
            errorMessage.textContent = 'Failed to load personalities. Please try again later.';
            errorMessage.style.display = 'block';
        }
    }
}

// Display metadata for the three bipolar persuasion dimensions.
const DIMENSION_META = {
    epistemic_style:    { name: 'Rapport au savoir',        left: 'Données & institutions', right: 'Intuition & alternatives' },
    moral_foundations:  { name: 'Fondements moraux',        left: 'Soin & justice',         right: 'Loyauté & tradition' },
    threat_sensitivity: { name: 'Rapport au changement',    left: 'Ouvert au changement',   right: 'Défense du statu quo' },
};

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
    const name = escapeHtml(p.name || 'Archetype');
    const sublabel = p.short_label ? escapeHtml(p.short_label) : (p.description ? escapeHtml(p.description) : '');
    const values = p.value_system ? escapeHtml(p.value_system) : '';
    const engagement = p.engagement_stance ? escapeHtml(p.engagement_stance) : '';

    const persuasion = p.persuasion || {};
    const format = persuasion.format ? escapeHtml(persuasion.format.replace(/_/g, ' ')) : '';
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

        ${values ? `<p class="archetype-values"><span class="field-label">Value system</span>${values}</p>` : ''}

        <div class="archetype-tags">
            ${engagement ? `<span class="engagement-badge">Engagement: ${engagement}</span>` : ''}
            ${format ? `<span class="format-tag">${format}</span>` : ''}
        </div>

        ${teaser ? `<p class="approach-teaser"><strong>${icon('bulb')} How to reach them:</strong> ${escapeHtml(teaser)}</p>` : ''}

        <span class="archetype-cta">View full profile →</span>
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
        if (errorEl) { errorEl.textContent = 'No personality selected.'; errorEl.style.display = 'block'; }
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
        const p = list.find((x) => x && x.id === id);

        const allArgs = argsRes.ok ? await argsRes.json() : [];
        const personaArgs = (Array.isArray(allArgs) ? allArgs : [])
            .filter((a) => (a.personas || []).some((x) => x && x.id === id));

        if (loading) loading.style.display = 'none';
        if (!p) {
            if (errorEl) { errorEl.textContent = 'Personality not found.'; errorEl.style.display = 'block'; }
            return;
        }

        document.title = `${p.name || 'Personality'} - re.inform`;
        if (container) {
            container.style.display = 'block';
            renderPersonaProfile(p, container, personaArgs);
        }

        // Prev/next among the named archetypes only.
        const archetypes = list
            .filter((x) => x && (x.layer === 'archetype' || (x.imported_from && x.imported_from.project === 'BonPoteJS')))
            .sort((a, b) => (a.id || 0) - (b.id || 0));
        renderPersonaNav(archetypes, id);
    } catch (error) {
        console.error('Error loading personality:', error);
        if (loading) loading.style.display = 'none';
        if (errorEl) { errorEl.textContent = 'Failed to load this personality. Please try again later.'; errorEl.style.display = 'block'; }
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
                <h3>What you'll hear</h3>
                <p class="empty-message">No claims are linked to this personality yet.</p>
            </section>`;
    }

    const cards = top.map((a) => {
        const importance = a.importance || {};
        const count = importance.count || 0;
        const avg = importance.avg_score;
        const badge = (count && typeof avg === 'number')
            ? (() => {
                const level = importanceLevel(avg);
                return `<span class="needs-badge feed-importance ${level.className}" title="Importance: ${avg.toFixed(1)}/5 from ${count} rating${count === 1 ? '' : 's'}">${icon('target')} ${level.label} · ${avg.toFixed(1)}/5</span>`;
            })()
            : `<span class="needs-badge feed-importance feed-importance-unrated" title="Nobody has rated how urgent this is yet">${icon('target')} Importance unrated</span>`;

        const contributions = a.contribution_count || 0;
        return `
            <a class="needs-card" href="argument.html?id=${encodeURIComponent(a.id)}">
                <p class="needs-quote">"${formatClaimText(a.text)}"</p>
                <div class="needs-meta">
                    <span class="needs-badge" title="Debunks, ratings and resources combined">${icon('layers')} ${contributions} contribution${contributions === 1 ? '' : 's'}</span>
                    ${badge}
                </div>
            </a>`;
    }).join('');

    return `
        <section class="persona-section">
            <div class="persona-claims-head">
                <h3>What you'll hear</h3>
                <a class="persona-claims-more" href="${seeMoreHref}">See all ${args.length} claim${args.length === 1 ? '' : 's'} →</a>
            </div>
            <p class="persona-claims-sub">The claims aimed at this personality, most important to debunk first.</p>
            <div class="persona-claims">${cards}</div>
        </section>`;
}

function renderPersonaProfile(p, container, personaArgs = []) {
    if (p.accent_color) container.style.setProperty('--accent', p.accent_color);

    const emoji = p.icon_emoji ? escapeHtml(p.icon_emoji) : '🧑';
    const name = escapeHtml(p.name || 'Personality');
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
    const format = persuasion.format ? escapeHtml(persuasion.format.replace(/_/g, ' ')) : '';
    const doList = Array.isArray(persuasion.do) ? persuasion.do : [];
    const dontList = Array.isArray(persuasion.dont) ? persuasion.dont : [];
    const messengers = Array.isArray(persuasion.messengers) ? persuasion.messengers : [];

    const listHtml = (items) => items.map((x) => `<li>${escapeHtml(x)}</li>`).join('');
    const chipHtml = (items) => items.map((x) => `<span class="mini-chip">${escapeHtml(x)}</span>`).join('');

    // EVS-cluster fields (population grounding layer), shown when present.
    const share = Number(p.population_share_w);
    const sharePct = Number.isFinite(share) ? `${(share * 100).toFixed(1)}%` : '';
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
            ${sharePct ? `<p class="persona-hero-share">Estimated ${sharePct} of the French population (EVS 2018).</p>` : ''}
        </div>

        <div class="persona-body">
            ${dimsHtml ? `
            <section class="persona-section">
                <h3>What defines them</h3>
                <div class="archetype-dims">${dimsHtml}</div>
                <div class="archetype-tags persona-tags">
                    ${engagement ? `<span class="engagement-badge">Engagement: ${engagement}</span>` : ''}
                    ${anchoring ? `<span class="anchoring-note"><strong>Trusted messengers:</strong> ${anchoring}</span>` : ''}
                </div>
            </section>` : ''}

            ${approach ? `
            <section class="persona-section persuasion-panel">
                <h3>${icon('bulb')} How to change their mind${format ? ` <span class="format-tag">${format}</span>` : ''}</h3>
                <p class="approach">${approach}</p>
                ${(doList.length || dontList.length) ? `
                <div class="do-dont">
                    ${doList.length ? `<div class="do-col"><span class="do-label">${icon('check')} Do</span><ul>${listHtml(doList)}</ul></div>` : ''}
                    ${dontList.length ? `<div class="dont-col"><span class="dont-label">${icon('ban')} Avoid</span><ul>${listHtml(dontList)}</ul></div>` : ''}
                </div>` : ''}
                ${messengers.length ? `<div class="messengers-row"><span class="field-label">Who to bring</span>${chipHtml(messengers)}</div>` : ''}
            </section>` : ''}

            ${personaClaimsSection(p, personaArgs)}

            ${axesLabels.length ? `
            <section class="persona-section">
                <h3>Most distinguishing value axes</h3>
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
        const label = escapeHtml(persona.name || 'Personality');
        const arrow = dir === 'prev' ? '←' : '→';
        return `<a class="persona-nav-link persona-nav-${dir}" href="personality.html?id=${encodeURIComponent(persona.id)}">
            <span class="persona-nav-dir">${arrow} ${dir === 'prev' ? 'Previous' : 'Next'}</span>
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

function showToast(message, { actionLabel = null, actionHref = null, duration = 4000 } = {}) {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.className = 'toast';
        toast.setAttribute('role', 'status');
        toast.setAttribute('aria-live', 'polite');
        document.body.appendChild(toast);
    }

    const actionHtml = actionLabel && actionHref
        ? ` <a class="toast-action" href="${escapeHtml(actionHref)}">${escapeHtml(actionLabel)}</a>`
        : '';
    toast.innerHTML = `<span>${escapeHtml(message)}</span>${actionHtml}`;
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
                    <span>${escapeHtml(sortSpec.label || 'Sort')}</span>
                    <select></select>
                </label>` : ''}
                ${spec.search ? `
                <div class="filterbar-search">
                    <span class="search-icon" aria-hidden="true">${SEARCH_ICON_SVG}</span>
                    <input type="search" class="search-bar filterbar-input"
                           placeholder="${escapeHtml(spec.search.placeholder || 'Search…')}"
                           aria-label="${escapeHtml(spec.search.placeholder || 'Search')}">
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
    }

    function closeMenus() {
        facets.forEach((f) => {
            f._menu.hidden = true;
            f._btn.setAttribute('aria-expanded', 'false');
            f._group.classList.remove('is-open');
        });
        openId = null;
    }

    function renderMenu(facet) {
        if (!facet.options.length) {
            facet._menu.innerHTML = '<p class="filterbar-menu-empty">Nothing to filter by yet.</p>';
            return;
        }
        facet._menu.innerHTML = '<div class="filterbar-options">' + facet.options.map((o) => {
            const checked = facet.selected.has(String(o.value)) ? 'checked' : '';
            const c = (typeof o.count === 'number') ? `<span class="filterbar-opt-count">${o.count}</span>` : '';
            return `<label class="filterbar-option"><input type="checkbox" value="${escapeHtml(String(o.value))}" ${checked}><span>${personaEmojiHtml(o.emoji)}${escapeHtml(o.label)}</span>${c}</label>`;
        }).join('') + '</div>';

        facet._menu.querySelectorAll('input[type="checkbox"]').forEach((box) => {
            box.addEventListener('change', () => {
                if (box.checked) facet.selected.add(box.value);
                else facet.selected.delete(box.value);
                renderButton(facet);   // refresh count badge in place (menu stays open)
                renderChips();
                emit();
            });
        });
    }

    // ---- chips ----
    function renderChips() {
        const pills = [];
        facets.forEach((facet) => {
            facet.selected.forEach((v) => {
                pills.push(`<button type="button" class="filterbar-chip" data-facet="${escapeHtml(facet.id)}" data-value="${escapeHtml(String(v))}">${escapeHtml(facet.label)}: ${optionLabel(facet, v)} ✕</button>`);
            });
        });
        if (!pills.length && !searchValue.trim()) { chipsEl.innerHTML = ''; return; }
        chipsEl.innerHTML = pills.join('') +
            '<button type="button" class="filterbar-chip filterbar-chip-clear" data-clear="all">Clear all</button>';

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
    return Boolean(name) && !/^\d+$/.test(name);
}

let feedQuery = '';
let feedTopics = new Set();
// Persona filter: a set of ids, so several personalities can be combined.
// Seeded from ?persona=<id>[,<id>…] — the link personality pages use to show
// all their claims — and kept in the URL so a filtered feed stays shareable.
let feedPersonaIds = new Set();
let feedPersonaList = [];

async function initRespondPage() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');

    feedPersonaIds = new Set(
        (new URLSearchParams(window.location.search).get('persona') || '')
            .split(',')
            .map((s) => parseInt(s, 10))
            .filter(Number.isInteger)
    );

    try {
        const response = await fetch(`${API_BASE_URL}/arguments`);
        if (!response.ok) throw new Error('Failed to load narratives');
        allArguments = await response.json();
        if (!Array.isArray(allArguments)) allArguments = [];

        // Full persona roster: every persona that appears on a claim, deduped,
        // plus the archetype list as a fallback for names.
        const byId = new Map();
        allArguments.forEach((a) => (a.personas || []).forEach((x) => {
            if (x && Number.isInteger(x.id) && !byId.has(x.id)) byId.set(x.id, { id: x.id, name: x.name, emoji: x.emoji });
        }));
        try {
            const personaRes = await fetch(`${API_BASE_URL}/persona-archetypes`);
            const personas = personaRes.ok ? await personaRes.json() : [];
            (Array.isArray(personas) ? personas : []).forEach((p) => {
                if (p && Number.isInteger(p.id) && !byId.has(p.id)) {
                    byId.set(p.id, { id: p.id, name: p.name, emoji: p.icon_emoji });
                }
            });
        } catch (_) { /* roster still works from claims alone */ }
        feedPersonaList = Array.from(byId.values())
            .sort((a, b) => String(a.name || '').localeCompare(String(b.name || '')));

        // Drop any seeded id that matches no known persona.
        feedPersonaIds = new Set([...feedPersonaIds].filter((id) => byId.has(id)));

        if (loadingMessage) loadingMessage.style.display = 'none';

        const topicOptions = [...new Set(allArguments.flatMap((a) => a.tags || []).filter(Boolean))]
            .sort().map((t) => ({ value: t, label: capitalizeWords(t) }));

        // Only offer personas that actually appear on a claim — filtering by one
        // with no claims would only ever empty the feed.
        const personasWithClaims = new Set(allArguments.flatMap((a) => (a.personas || []).map((x) => x && x.id)));
        const personaOptions = feedPersonaList
            .filter((p) => personasWithClaims.has(p.id) && isNamedPersona(p))
            .map((p) => ({ value: String(p.id), label: p.name || 'Unknown', emoji: p.emoji }));

        createFilterBar(document.getElementById('feedControls'), {
            facets: [
                { id: 'persona', label: 'Personality', icon: icon('users'), options: personaOptions, selected: [...feedPersonaIds].map(String) },
                { id: 'topic', label: 'Topic', icon: icon('tag'), options: topicOptions },
            ],
            sort: {
                label: 'Sort', value: feedSort, options: [
                    { value: 'responses', label: 'Most active' },
                    { value: 'gaps', label: 'Needs help' },
                    { value: 'title', label: 'A–Z' },
                ],
            },
            search: { placeholder: 'Search a claim you heard…' },
            onChange: (s) => {
                feedPersonaIds = new Set(s.facets.persona.map(Number));
                feedTopics = new Set(s.facets.topic);
                feedSort = s.sort;
                feedQuery = s.q;
                syncPersonaUrl();
                renderArgumentsList();
            },
        });

        renderArgumentsList();
    } catch (error) {
        console.error('Error loading narratives:', error);
        if (loadingMessage) loadingMessage.style.display = 'none';
        if (errorMessage) {
            errorMessage.textContent = 'Failed to load the feed. Please try again later.';
            errorMessage.style.display = 'block';
        }
    }
}

function renderArgumentsList() {
    const list = document.getElementById('argumentsList');
    const emptyMessage = document.getElementById('emptyMessage');
    if (!list) return;

    const query = feedQuery.trim().toLowerCase();

    let items = allArguments.filter((a) => {
        const text = String(a.text || '').toLowerCase();
        const matchesQuery = !query || text.includes(query);
        const matchesTopic = feedTopics.size === 0 || (a.tags || []).some((t) => feedTopics.has(t));
        const matchesPersona = feedPersonaIds.size === 0
            || (a.personas || []).some((x) => x && feedPersonaIds.has(x.id));
        return matchesQuery && matchesTopic && matchesPersona;
    });

    items = items.slice().sort((a, b) => {
        const ca = a.contribution_count || 0;
        const cb = b.contribution_count || 0;
        if (feedSort === 'gaps') return ca - cb;
        if (feedSort === 'title') return String(a.text || '').localeCompare(String(b.text || ''));
        return cb - ca; // most active (default)
    });

    list.innerHTML = '';

    if (!items.length) {
        if (emptyMessage) emptyMessage.style.display = 'block';
        return;
    }
    if (emptyMessage) emptyMessage.style.display = 'none';

    items.forEach((a) => list.appendChild(createFeedCard(a)));
}

// Importance levels mirror the Low/Medium/Critical chips on the contribute page,
// mapped back from the 1..5 average the ratings are stored on.
function importanceLevel(avgScore) {
    if (avgScore >= 4) return { label: 'Critical', className: 'feed-importance-critical' };
    if (avgScore >= 2.5) return { label: 'Medium', className: 'feed-importance-medium' };
    return { label: 'Low', className: 'feed-importance-low' };
}

// Persona emoji, drawn as black line art by the Noto Emoji font (see .persona-emoji in styles.css).
function personaEmojiHtml(emoji) {
    return emoji ? `<span class="persona-emoji" aria-hidden="true">${escapeHtml(emoji)}</span> ` : '';
}

// Keep ?persona=… in sync with the current selection so the feed is shareable
// and survives a refresh.
function syncPersonaUrl() {
    const url = new URL(window.location.href);
    if (feedPersonaIds.size) {
        url.searchParams.set('persona', [...feedPersonaIds].join(','));
    } else {
        url.searchParams.delete('persona');
    }
    window.history.replaceState({}, '', url);
}

function createFeedCard(a) {
    // Uses the compact "Needs your help" card template, plus persona emojis.
    const id = encodeURIComponent(a.id);
    const href = `argument.html?id=${id}`;
    const count = a.counter_argument_count || 0;
    const contributions = a.contribution_count || 0;
    const personas = Array.isArray(a.personas) ? a.personas : [];

    // Affected personalities: emoji + family name (spans — the whole card is a link).
    const personaChips = personas.slice(0, 3).map((p) => {
        const parts = (p.name || '').trim().split(/\s+/);
        const family = escapeHtml(parts.length > 1 ? parts.slice(1).join(' ') : (parts[0] || 'Profile'));
        const em = personaEmojiHtml(p.emoji);
        return `<span class="feed-persona-mini" title="${escapeHtml(p.name || '')}">${em}${family}</span>`;
    }).join('');
    const moreChip = personas.length > 3 ? `<span class="feed-persona-more">+${personas.length - 3}</span>` : '';
    const personasHtml = personaChips
        ? `<div class="needs-personas">${personaChips}${moreChip}</div>`
        : '';

    const badge = count === 0
        ? `<span class="needs-badge needs-badge-hot">${icon('zap')} No response yet</span>`
        : `<span class="needs-badge" title="Debunks, ratings and resources combined">${icon('layers')} ${contributions} contribution${contributions === 1 ? '' : 's'}</span>`;

    // How urgent the community says this is to debunk — from the importance
    // ratings cast on the contribute page.
    const importance = a.importance || {};
    const ratingCount = importance.count || 0;
    const avg = importance.avg_score;
    let importanceHtml;
    if (ratingCount && typeof avg === 'number') {
        const level = importanceLevel(avg);
        importanceHtml = `<span class="needs-badge feed-importance ${level.className}" title="Importance: ${avg.toFixed(1)}/5 from ${ratingCount} rating${ratingCount === 1 ? '' : 's'}">${icon('target')} ${level.label} · ${avg.toFixed(1)}/5</span>`;
    } else {
        importanceHtml = `<span class="needs-badge feed-importance feed-importance-unrated" title="Nobody has rated how urgent this is yet">${icon('target')} Importance unrated</span>`;
    }

    const card = document.createElement('a');
    card.className = 'needs-card';
    card.href = href;
    card.innerHTML = `
        <p class="needs-quote">"${formatClaimText(a.text)}"</p>
        <div class="needs-meta">
            ${badge}
            ${importanceHtml}
        </div>
        ${personasHtml}
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
            errorEl.textContent = 'No claim selected.';
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
        const write = params.get('write');
        if (write) openComposer(id, /^\d+$/.test(write) ? Number(write) : '');

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
            errorEl.textContent = 'Failed to load this claim. Please try again later.';
            errorEl.style.display = 'block';
        }
    }
}

function renderArgumentDetail() {
    const arg = argDetailState.argument || {};
    const cas = argDetailState.counterArguments;

    document.getElementById('narrativeText').textContent = arg.text || 'Claim';
    document.title = `${(arg.text || 'Claim').slice(0, 50)} - re.inform`;

    const tagsEl = document.getElementById('narrativeTags');
    tagsEl.innerHTML = (arg.tags || []).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join('');

    // "Seen with" chips: personas that already have a tailored response here.
    const seenPersonas = new Map();
    cas.forEach((ca) => {
        const p = ca.persona_archetype;
        if (p && p.id != null && isNamedPersona(p)) seenPersonas.set(p.id, p);
    });
    const seenWith = document.getElementById('seenWith');
    if (seenPersonas.size) {
        seenWith.innerHTML = `<span class="seen-with-label">Responses for:</span> ` +
            Array.from(seenPersonas.values())
                .map((p) => `<span class="persona-chip">${escapeHtml(p.name || 'Profile')}</span>`)
                .join('');
    } else {
        seenWith.innerHTML = '';
    }

    // Persona filter (only personas present in responses) + composer persona select (all named).
    const filterEl = document.getElementById('responsePersonaFilter');
    filterEl.innerHTML = '<option value="">Anyone</option>' +
        Array.from(seenPersonas.values())
            .map((p) => `<option value="${p.id}">${escapeHtml(p.name || 'Profile')}</option>`)
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
        heading.textContent = `${n} response${n === 1 ? '' : 's'}`;
    }

    list.innerHTML = '';
    if (!cas.length) {
        list.innerHTML = `<div class="empty-message"><p>No response for this profile yet.</p>
            <button type="button" class="btn btn-primary" onclick="openComposer(${argDetailState.argumentId}, '${filterId}')">Write the first one</button></div>`;
        return;
    }

    cas.forEach((ca) => list.appendChild(createResponseCard(ca)));
}

function createResponseCard(ca) {
    const persona = ca.persona_archetype;
    const personaId = persona && persona.id != null ? persona.id : '';
    const personaName = persona ? (persona.name || 'Profile') : 'General';
    const framing = persona && persona.description ? ` · ${persona.description}` : '';

    const text = String(ca.text || '');
    const isLong = text.length > 320;
    const shownText = isLong ? text.slice(0, 320) + '…' : text;

    const avg = (typeof ca.avg_score === 'number') ? ca.avg_score : null;
    const ratingCount = ca.rating_count || 0;
    const ratingHtml = avg != null
        ? `<span class="trust-item" title="${avg.toFixed(1)} / 5 from ${ratingCount} rating${ratingCount === 1 ? '' : 's'}">${renderStars(avg)} <span class="rating-count">(${ratingCount})</span></span>`
        : `<span class="trust-item trust-muted">Not rated yet</span>`;

    const resources = Array.isArray(ca.resources) && ca.resources.length
        ? ca.resources
        : (ca.source_url ? [{ url: ca.source_url, title: null }] : []);
    const sourceHtml = resources.length
        ? resources.map((r) =>
            `<a class="trust-item source-chip" href="${escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer">${icon('clip')} ${escapeHtml(r.title || hostnameOf(r.url))}</a>`
          ).join('')
        : `<span class="trust-item trust-muted">No source</span>`;

    const check = ca.source_check || {};
    const checkBtnHtml = check.can_check
        ? `<button type="button" class="btn btn-secondary btn-small response-check">${icon('search')} ${check.my_verdict ? 'Change your check' : 'Check source'}</button>`
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
        ${isLong ? `<button type="button" class="link-btn response-more">Read more</button>` : ''}
        <div class="response-trust">${sourceHtml}${sourceStampHtml(check)}${ratingHtml}</div>
        <div class="response-actions">
            <button type="button" class="btn btn-secondary btn-small response-copy">${icon('copy')} Copy</button>
            <button type="button" class="btn btn-secondary btn-small response-rate">★ Rate</button>
            <button type="button" class="btn btn-secondary btn-small response-improve">${icon('pen')} Improve</button>
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
            moreBtn.textContent = expanded ? 'Read less' : 'Read more';
        });
    }

    // Copy → clipboard + conversion nudge toward a field report.
    card.querySelector('.response-copy').addEventListener('click', () => {
        const urls = resources.map((r) => r.url).filter(Boolean);
        const toCopy = urls.length ? `${text}\n\nSources:\n${urls.join('\n')}` : text;
        navigator.clipboard.writeText(toCopy).then(() => {
            showToast('Copied! Tell us if it worked in a real conversation.');
        }).catch(() => {
            showToast('Could not copy automatically — select and copy the text.');
        });
    });

    // Rate → inline stars
    card.querySelector('.response-rate').addEventListener('click', () => {
        const rateRow = card.querySelector('.rate-row');
        const isHidden = rateRow.style.display === 'none';
        rateRow.style.display = isHidden ? 'flex' : 'none';
        rateRow.setAttribute('aria-hidden', isHidden ? 'false' : 'true');
        if (isHidden && !rateRow.dataset.built) {
            buildRateRow(rateRow, ca.id, personaId);
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
    verified: { icon: 'check', label: 'Source verified', cls: 'stamp-verified',
        title: 'Sourcers checked that the cited sources support this response.' },
    needs_work: { icon: 'x', label: 'Source needs work', cls: 'stamp-needs-work',
        title: 'Sourcers found the cited sources don’t fully support this response.' },
    pending: { icon: 'hourglass', label: 'Source not verified yet', cls: 'stamp-pending',
        title: 'No Sourcer has confirmed these sources yet.' },
};

const SOURCE_VERDICTS = [
    { value: 'supports', label: 'Yes, they back it up' },
    { value: 'partial', label: 'Only partly' },
    { value: 'does_not_support', label: 'No' },
    { value: 'broken', label: 'Link broken' },
];

function sourceStampHtml(check) {
    const stamp = SOURCE_STAMPS[check.status];
    if (!stamp) return '';
    const label = check.status === 'pending' && check.count
        ? `Source check in progress (${check.count})`
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
        <span class="rate-label">Open the sources above. Do they back up this response?</span>
        <div class="check-verdicts" role="group" aria-label="Your verdict">
            ${SOURCE_VERDICTS.map((v) => `
                <button type="button" class="btn btn-secondary btn-small check-verdict" data-verdict="${v.value}"
                        aria-pressed="${v.value === mine}">${escapeHtml(v.label)}</button>`).join('')}
        </div>
        <input type="text" class="check-note" maxlength="500" aria-label="Note for the author (optional)"
               placeholder="Optional: what’s missing or wrong?">
        <div class="check-submit-row">
            <button type="button" class="btn btn-primary btn-small check-submit" ${mine ? '' : 'disabled'}>Submit check</button>
            <span class="check-hint">Your check counts toward the stamp. Points come once the verdict is settled.</span>
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
            showToast(data.error || 'Could not save your check.');
            btn.disabled = false;
            return;
        }
        const status = data.source_check?.status;
        const points = data.points_earned ? ` +${data.points_earned} reputation.` : '';
        const msg = status === 'verified' ? `Source verified.${points}`
            : status === 'needs_work' ? `Flagged: sources need work.${points}`
            : 'Check saved. One more settles it.';
        showToast(msg);
        await reloadArgumentDetail();
    } catch (error) {
        console.error('Source check error:', error);
        showToast('Network error while saving your check.');
        btn.disabled = false;
    }
}

function buildRateRow(rateRow, counterArgumentId, personaId) {
    rateRow.innerHTML = '<span class="rate-label">Would this work?</span>';
    const starWrap = document.createElement('span');
    starWrap.className = 'rate-stars';
    for (let i = 1; i <= 5; i++) {
        const star = document.createElement('button');
        star.type = 'button';
        star.className = 'rate-star';
        star.textContent = '★';
        star.setAttribute('aria-label', `${i} star${i === 1 ? '' : 's'}`);
        star.addEventListener('mouseenter', () => highlightStars(starWrap, i));
        star.addEventListener('mouseleave', () => highlightStars(starWrap, 0));
        star.addEventListener('click', () => submitRating(counterArgumentId, personaId, i, rateRow));
        starWrap.appendChild(star);
    }
    rateRow.appendChild(starWrap);
}

function highlightStars(starWrap, upTo) {
    Array.from(starWrap.children).forEach((star, idx) => {
        star.classList.toggle('is-lit', idx < upTo);
    });
}

async function submitRating(counterArgumentId, personaId, score, rateRow) {
    if (!currentUser) {
        showToast('Log in to rate a response.', { actionLabel: 'Log in', actionHref: authHref() });
        return;
    }
    try {
        const body = { counter_argument_id: counterArgumentId, score };
        if (personaId !== '' && personaId != null) body.persona_archetype_id = Number(personaId);

        const res = await fetch(`${API_BASE_URL}/effectiveness-ratings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (res.ok) {
            rateRow.innerHTML = `<span class="rate-thanks">Thanks — rated ${score}★</span>`;
            showToast('Rating saved. Have you tried this one for real?');
        } else {
            showToast(data.error || 'Could not save rating.');
        }
    } catch (error) {
        console.error('Rating error:', error);
        showToast('Network error while saving rating.');
    }
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
        return 'Source';
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
        btn.innerHTML = `${icon('pen')} Write for ${escapeHtml(p.name)}`;
        btn.addEventListener('click', () => openComposer(argDetailState.argumentId, p.id));
        gapsList.appendChild(btn);
    });
}

// ---- Composer ----

function initComposer() {
    const composerPersona = document.getElementById('composerPersona');
    if (composerPersona) {
        // A debunk targets exactly one persona — no "general" default.
        composerPersona.innerHTML = '<option value="" disabled selected>Choose a personality…</option>' +
            argDetailState.namedPersonas
                .map((p) => `<option value="${p.id}">${escapeHtml(p.name || 'Profile')}</option>`)
                .join('');
    }

    document.getElementById('composerAddSource')?.addEventListener('click', addComposerSourceInput);
    document.getElementById('composerClose')?.addEventListener('click', closeComposer);
    document.getElementById('composerCancel')?.addEventListener('click', closeComposer);
    document.getElementById('composerBackdrop')?.addEventListener('click', closeComposer);
    document.getElementById('composerForm')?.addEventListener('submit', submitComposer);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeComposer();
    });
}

function addComposerSourceInput(value = '') {
    const container = document.getElementById('composerSources');
    if (!container) return;
    const input = document.createElement('input');
    input.type = 'url';
    input.className = 'composer-source-input';
    input.placeholder = 'https://example.com/evidence';
    input.value = value;
    container.appendChild(input);
    input.focus();
}

function resetComposerSources() {
    const container = document.getElementById('composerSources');
    if (!container) return;
    container.innerHTML = '<input type="url" class="composer-source-input" placeholder="https://example.com/evidence">';
}

function openComposer(argumentId, personaId = '', prefillText = '') {
    if (!currentUser) {
        showToast('Log in to contribute a response.', { actionLabel: 'Log in', actionHref: authHref() });
        return;
    }
    const panel = document.getElementById('composerPanel');
    const backdrop = document.getElementById('composerBackdrop');
    const personaSelect = document.getElementById('composerPersona');
    const hint = document.getElementById('composerHint');
    const title = document.getElementById('composerTitle');
    const message = document.getElementById('composerMessage');

    if (personaSelect) personaSelect.value = (personaId !== '' && personaId != null) ? String(personaId) : '';
    document.getElementById('composerText').value = prefillText || '';
    resetComposerSources();
    if (message) { message.textContent = ''; message.className = 'form-message'; }

    const persona = argDetailState.personas.find((p) => String(p.id) === String(personaId));
    if (persona && persona.description) {
        title.textContent = `Write a response for ${persona.name}`;
        hint.textContent = `Tip — ${persona.name}: ${persona.description}. Speak to what they value.`;
        hint.style.display = 'block';
    } else {
        title.textContent = prefillText ? 'Propose an improved response' : 'Write a response';
        hint.style.display = 'none';
    }

    if (backdrop) backdrop.style.display = 'block';
    if (panel) {
        panel.style.display = 'block';
        panel.setAttribute('aria-hidden', 'false');
        panel.classList.add('is-open');
    }
    document.getElementById('composerText').focus();
}

function closeComposer() {
    const panel = document.getElementById('composerPanel');
    const backdrop = document.getElementById('composerBackdrop');
    if (panel) {
        panel.classList.remove('is-open');
        panel.style.display = 'none';
        panel.setAttribute('aria-hidden', 'true');
    }
    if (backdrop) backdrop.style.display = 'none';
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
        message.textContent = 'Please write a response.';
        message.className = 'form-message error';
        return;
    }
    if (!personaVal) {
        message.textContent = 'Please choose who this response is for.';
        message.className = 'form-message error';
        return;
    }

    const body = {
        argument_id: argDetailState.argumentId,
        text,
        persona_archetype_id: Number(personaVal),
        sources,
    };

    submitBtn.disabled = true;
    submitBtn.textContent = 'Publishing…';
    try {
        const res = await fetch(`${API_BASE_URL}/counter-arguments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (res.ok) {
            closeComposer();
            showToast('Response published — thank you for contributing!');
            await reloadArgumentDetail();
        } else {
            message.textContent = data.error || 'Could not publish response.';
            message.className = 'form-message error';
        }
    } catch (error) {
        console.error('Composer error:', error);
        message.textContent = 'Network error. Please try again.';
        message.className = 'form-message error';
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Publish for review';
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
            .sort().map((t) => ({ value: t, label: capitalizeWords(t) }));

        // Arriving from Profile via resources.html?tag=… pre-selects that topic.
        const requestedTag = new URLSearchParams(window.location.search).get('tag');
        const seed = requestedTag
            ? topicOptions.filter((o) => o.value.toLowerCase() === requestedTag.toLowerCase()).map((o) => o.value)
            : [];
        blogState.topics = new Set(seed);

        createFilterBar(document.getElementById('blogControls'), {
            facets: [{ id: 'topic', label: 'Topic', icon: icon('tag'), options: topicOptions, selected: seed }],
            sort: {
                label: 'Sort', value: blogState.sort, options: [
                    { value: 'rating', label: 'Effectiveness' },
                    { value: 'date', label: 'Most recent' },
                    { value: 'title', label: 'A–Z' },
                ],
            },
            search: { placeholder: 'Search articles…' },
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
            errorMessage.textContent = 'Failed to load resources. Please try again later.';
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
        filtered.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
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
            `<span class="tag">${capitalizeWords(tag)}</span>`
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
        ? `<div class="card-source"><a href="${articleViewerUrl}">Read article</a></div>`
        : (resource.source 
            ? `<div class="card-source"><a href="${resource.source}" target="_blank" rel="noopener noreferrer">View source</a></div>`
            : '');
    
    const dateHTML = resource.date 
        ? `<div class="card-date">${new Date(resource.date).toLocaleDateString()}</div>`
        : '';
    
    // Author info: prefer db-backed user link; otherwise fall back to the resource's author string.
    const authorName = (resource?.author || '').toString().trim();
    const avatar = resource.user_id
        ? userAvatarHtml(resource.user_id, resource.username, 'avatar-xs')
        : '';
    const contributorHTML = resource.user_id && resource.username
        ? `<a href="profile.html?id=${resource.user_id}" class="contributor-link">${avatar}<span class="contributor-name">By ${resource.username}</span></a>`
        : (authorName
            ? `<span class="anonymous-contributor">By ${authorName}</span>`
            : '<span class="anonymous-contributor">By Anonymous</span>');
    
    const titleHTML = isInternalArticle
        ? `<a class="card-title-link" href="${articleViewerUrl}"><h3 class="card-title">${resource.title || 'Untitled'}</h3></a>`
        : `<h3 class="card-title">${resource.title || 'Untitled'}</h3>`;

    card.innerHTML = `
        <div class="card-header">
            ${titleHTML}
            <span class="effectiveness-badge">★ ${resource.effectiveness_rating || 0}/10</span>
        </div>
        ${resource.narrative ? `<div class="card-narrative">"${resource.narrative}"</div>` : ''}
        <div class="card-summary">${resource.summary || resource.counter_argument || ''}</div>
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
    loginBtn.textContent = 'Logging in...';
    
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
            loginMessage.textContent = data.error || 'Invalid password';
            loginMessage.className = 'form-message error';
        }
    } catch (error) {
        console.error('Login error:', error);
        loginMessage.textContent = 'Network error. Please try again.';
        loginMessage.className = 'form-message error';
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = 'Login';
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
        
        if (!response.ok) {
            throw new Error('Failed to load submissions');
        }
        
        const submissions = await response.json();
        
        loadingMessage.style.display = 'none';
        
        if (submissions.length === 0) {
            submissionsList.innerHTML = '<p class="empty-message">No submissions yet.</p>';
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
        loadingMessage.textContent = 'Failed to load submissions.';
    }
}

function createSubmissionCard(submission) {
    const date = new Date(submission.submitted_at).toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    
    const statusClass = submission.status || 'pending';
    const source = submission.source ? escapeHtml(submission.source) : 'N/A';
    
    const actions = submission.status === 'pending' ? `
        <div class="submission-actions">
            <button class="btn btn-primary btn-approve" data-id="${submission.id}">Approve</button>
            <button class="btn btn-secondary btn-reject" data-id="${submission.id}">Reject</button>
        </div>
        <div class="approval-form" id="approval-form-${submission.id}">
            <h4>Approval Details</h4>
            <div class="form-group">
                <label>Title:</label>
                <input type="text" class="approval-title" placeholder="Counter-Argument Title" />
            </div>
            <div class="form-group">
                <label>Effectiveness Rating (0-10):</label>
                <input type="number" class="approval-rating" min="0" max="10" value="5" />
            </div>
            <div class="form-group">
                <label>Tags (comma-separated):</label>
                <input type="text" class="approval-tags" placeholder="e.g., immigration, economy, climate" />
            </div>
            <div class="form-actions">
                <button class="btn btn-primary btn-confirm-approve" data-id="${submission.id}">Confirm Approval</button>
                <button class="btn btn-secondary btn-cancel-approve" data-id="${submission.id}">Cancel</button>
            </div>
        </div>
    ` : '';
    
    return `
        <div class="submission-card ${statusClass}">
            <div class="submission-header">
                <strong>Submission #${submission.id}</strong>
                <span class="submission-status ${statusClass}">${capitalizeWords(statusClass)}</span>
            </div>
            <div class="submission-content">
                <div class="submission-field">
                    <strong>Narrative:</strong>
                    <p>${escapeHtml(submission.narrative)}</p>
                </div>
                <div class="submission-field">
                    <strong>Counter-Argument:</strong>
                    <p>${escapeHtml(submission.counter_argument)}</p>
                </div>
                <div class="submission-field">
                    <strong>Source:</strong>
                    <p>${source}</p>
                </div>
                <div class="submission-field">
                    <strong>Submitted:</strong>
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
            
            const title = form.querySelector('.approval-title').value.trim();
            const rating = parseInt(form.querySelector('.approval-rating').value);
            const tagsInput = form.querySelector('.approval-tags').value.trim();
            const tags = tagsInput ? tagsInput.split(',').map(t => t.trim()) : [];
            
            if (!title) {
                alert('Please enter a title');
                return;
            }
            
            await approveSubmission(submissionId, { title, effectiveness_rating: rating, tags });
        });
    });
    
    // Reject buttons
    document.querySelectorAll('.btn-reject').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const submissionId = e.target.dataset.id;
            if (confirm('Are you sure you want to reject this submission?')) {
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
            alert('Submission approved successfully!');
            loadSubmissions(); // Reload submissions
        } else {
            const error = await response.json();
            alert('Failed to approve submission: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error approving submission:', error);
        alert('Network error. Please try again.');
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
            alert('Submission rejected.');
            loadSubmissions(); // Reload submissions
        } else {
            const error = await response.json();
            alert('Failed to reject submission: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error rejecting submission:', error);
        alert('Network error. Please try again.');
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

const __searchTextCache = new WeakMap();

function getSearchTerms(query) {
    const normalized = String(query || '').toLowerCase().trim();
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
        return buildSearchText(item).toLowerCase();
    }

    const cached = __searchTextCache.get(item);
    if (cached) return cached;

    const computed = buildSearchText(item).toLowerCase();
    __searchTextCache.set(item, computed);
    return computed;
}

function itemMatchesSearchTerms(item, terms) {
    if (!terms || !terms.length) return true;
    const haystack = getCachedSearchText(item);
    return terms.every(term => haystack.includes(term));
}

function capitalizeWords(str) {
    return str.replace(/\b\w/g, char => char.toUpperCase());
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
                { id: 'type', label: 'Type', icon: icon('file'), options: libraryFacetOptions('type'), selected: [...libraryState.type] },
                { id: 'topic', label: 'Topic', icon: icon('tag'), options: libraryFacetOptions('topic') },
                { id: 'publisher', label: 'Publisher', icon: icon('building'), options: libraryFacetOptions('publisher') },
            ],
            sort: {
                label: 'Sort', value: libraryState.sort, options: [
                    { value: 'used', label: 'Most cited' },
                    { value: 'recent', label: 'Most recent' },
                    { value: 'title', label: 'A–Z' },
                ],
            },
            search: { placeholder: 'Search resources…' },
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
            errorMessage.textContent = 'Could not load the resource library. Please try again.';
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

    const label = (v) => facet === 'type' ? capitalizeWords(v) : v;
    return [...counts.entries()]
        .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
        .map(([value, count]) => ({ value, label: label(value), count }));
}

// Apply the active facets/search; `except` skips one facet (for its own counts).
function filterLibrary({ except = null } = {}) {
    const q = libraryState.q.trim().toLowerCase();
    return allLibraryResources.filter((r) => {
        for (const facet of ['type', 'topic', 'publisher']) {
            if (facet === except) continue;
            const sel = libraryState[facet];
            if (sel.size && !libraryFacetValues(r, facet).some((v) => sel.has(v))) return false;
        }
        if (q) {
            const hay = `${r.title || ''} ${r.description || ''} ${r.publisher || ''} ${(r.topics || []).join(' ')}`.toLowerCase();
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
        resources.sort((a, b) => String(a.title || '').localeCompare(String(b.title || '')));
    } else {
        resources.sort((a, b) => (b.used_in_responses || 0) - (a.used_in_responses || 0)
            || String(a.title || '').localeCompare(String(b.title || '')));
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
        { n: resources.length, label: 'resources shown' },
        { n: publishers.size, label: 'publishers' },
        { n: cited, label: 'cited in responses' },
    ];
    statsEl.innerHTML = tiles
        .map((t) => `<div class="stat-tile"><span class="stat-n">${t.n}</span><span class="stat-label">${escapeHtml(t.label)}</span></div>`)
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
            .map((t) => `<button type="button" class="tag library-topic" data-topic="${escapeHtml(t)}">#${escapeHtml(t)}</button>`)
            .join('');

        const preview = r.preview_image
            ? `<img class="library-preview" src="${escapeHtml(r.preview_image)}" alt="" loading="lazy">`
            : '';

        const cited = r.used_in_responses
            ? `<span class="library-cited">Cited in ${r.used_in_responses} response${r.used_in_responses === 1 ? '' : 's'}</span>`
            : '';

        const submitter = r.submitted_by?.username
            ? `<span class="library-submitter">Shared by ${escapeHtml(r.submitted_by.username)}</span>`
            : '';

        const context = r.context_url
            ? `<a class="library-context" href="${escapeHtml(r.context_url)}" target="_blank" rel="noopener noreferrer">Context ↗</a>`
            : '';

        const card = document.createElement('article');
        card.className = 'library-card';
        card.innerHTML = `
            ${preview}
            <div class="library-card-body">
                <div class="library-card-head">
                    <button type="button" class="library-type-badge" data-type="${escapeHtml(r.type)}">
                        ${typeIcon} ${escapeHtml(r.type_label || r.type)}
                    </button>
                    <button type="button" class="library-publisher" data-publisher="${escapeHtml(r.publisher)}">${escapeHtml(r.publisher)}</button>
                </div>
                <h3 class="library-title">
                    <a href="${escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title || r.url)}</a>
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
    queue.innerHTML = '<p class="loading-message">Loading resources…</p>';

    try {
        const params = status ? `?status=${encodeURIComponent(status)}` : '';
        const response = await fetch(`${API_BASE_URL}/admin/library${params}`, {
            headers: { 'Authorization': `Bearer ${adminToken}` },
        });
        if (!response.ok) throw new Error(`Request failed: ${response.status}`);
        const data = await response.json();

        adminLibraryTypes = data.types || [];

        if (countsEl) {
            const counts = data.counts || {};
            countsEl.textContent = `${counts.pending || 0} awaiting review · ${counts.approved || 0} approved · ${counts.rejected || 0} rejected`;
        }

        renderResourceReviewQueue(data.resources || []);
    } catch (error) {
        console.error('Review queue error:', error);
        queue.innerHTML = '<p class="error-message">Could not load resources for review.</p>';
    }
}

function renderResourceReviewQueue(resources) {
    const queue = document.getElementById('reviewQueue');
    if (!queue) return;

    if (!resources.length) {
        queue.innerHTML = '<p class="empty-message">Nothing here — the queue is clear.</p>';
        return;
    }

    queue.innerHTML = '';
    // Cap the page: approved is in the hundreds and the queue is a work surface,
    // not a browsing view — that's what the public library page is for.
    resources.slice(0, 60).forEach((r) => {
        const status = r.review?.status || 'pending';
        const reviewers = (r.review?.reviewers || []).map((x) => x.name).join(', ');
        const provenance = r.review?.by_provenance
            ? `auto-approved by provenance (${escapeHtml((r.origins || []).join(', '))})`
            : (reviewers ? `reviewed by ${escapeHtml(reviewers)}` : 'never reviewed');

        const typeOptions = adminLibraryTypes
            .map((t) => `<option value="${escapeHtml(t.value)}"${t.value === r.type ? ' selected' : ''}>${escapeHtml(t.label)}</option>`)
            .join('');

        const item = document.createElement('div');
        item.className = 'review-item';
        item.innerHTML = `
            <div class="review-item-head">
                <span class="review-status review-status-${escapeHtml(status)}">${escapeHtml(status)}</span>
                <a href="${escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title || r.url)}</a>
                <span class="review-meta">${escapeHtml(r.publisher || '')}</span>
            </div>
            <p class="review-meta">${escapeHtml(r.url)}</p>
            <div class="review-fields">
                <label>Title
                    <input type="text" data-field="title" value="${escapeHtml(r.title || '')}">
                </label>
                <label>Type
                    <select data-field="type">${typeOptions}</select>
                </label>
                <label>Topics (comma-separated)
                    <input type="text" data-field="topics" value="${escapeHtml((r.topics || []).join(', '))}">
                </label>
                <label>Description
                    <input type="text" data-field="description" value="${escapeHtml(r.description || '')}">
                </label>
            </div>
            <div class="review-actions">
                <button type="button" class="btn btn-primary" data-action="approved">Approve</button>
                <button type="button" class="btn btn-secondary" data-action="rejected">Reject</button>
            </div>
            <p class="review-meta">Cited in ${r.used_in_responses || 0} response(s) · ${provenance}</p>
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
            showMessage(data.error || 'Could not save the review.', 'error');
            return;
        }

        showMessage(`Resource ${status}.`, 'success');
        loadResourceReviewQueue();
    } catch (error) {
        console.error('Resource review error:', error);
        showMessage('Could not reach the server. Please try again.', 'error');
    } finally {
        buttons.forEach((b) => { b.disabled = false; });
    }
}
