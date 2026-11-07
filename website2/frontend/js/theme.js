// ========================================
// re.inform - Theme (Light / Night / System)
// ========================================

const THEME_STORAGE_KEY = 're_inform_color_scheme';

function getStoredPreference() {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    if (value === 'light' || value === 'dark' || value === 'system') return value;
    return 'system';
}

function getEffectiveScheme(preference) {
    if (preference === 'dark') return 'dark';
    if (preference === 'light') return 'light';

    // system
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
}

function applyColorScheme(preference) {
    const scheme = getEffectiveScheme(preference);

    document.documentElement.setAttribute('data-color-scheme', scheme);

    // Helps built-in form controls match the theme.
    document.documentElement.style.colorScheme = scheme;

    // Keep the chosen preference available for debugging / potential future use.
    document.documentElement.setAttribute('data-color-scheme-preference', preference);
}

function setPreference(preference) {
    localStorage.setItem(THEME_STORAGE_KEY, preference);
    applyColorScheme(preference);
}

// Apply ASAP (before DOMContentLoaded) to minimize theme flash.
try {
    applyColorScheme(getStoredPreference());
} catch (e) {
    // If storage is unavailable (private mode / blocked), default to system.
    applyColorScheme('system');
}

// Keep in sync when system theme changes while in "system" mode.
const mediaQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
function onSystemSchemeChange() {
    const preference = getStoredPreference();
    if (preference === 'system') applyColorScheme(preference);
}

if (mediaQuery) {
    if (typeof mediaQuery.addEventListener === 'function') {
        mediaQuery.addEventListener('change', onSystemSchemeChange);
    } else if (typeof mediaQuery.addListener === 'function') {
        // Safari < 14
        mediaQuery.addListener(onSystemSchemeChange);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const preference = getStoredPreference();

    // New UI: icon button group
    const buttons = Array.from(document.querySelectorAll('[data-color-scheme-option]'));
    if (buttons.length) {
        const syncButtons = () => {
            const current = getStoredPreference();
            buttons.forEach((btn) => {
                const value = btn.getAttribute('data-color-scheme-option');
                btn.classList.toggle('is-active', value === current);
                btn.setAttribute('aria-pressed', value === current ? 'true' : 'false');
            });
        };

        syncButtons();

        buttons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const next = btn.getAttribute('data-color-scheme-option');
                if (next !== 'light' && next !== 'dark' && next !== 'system') return;
                setPreference(next);
                syncButtons();
            });
        });

        // If system theme changes and user is in system mode, keep UI in sync.
        if (mediaQuery) {
            const resync = () => syncButtons();
            if (typeof mediaQuery.addEventListener === 'function') {
                mediaQuery.addEventListener('change', resync);
            } else if (typeof mediaQuery.addListener === 'function') {
                mediaQuery.addListener(resync);
            }
        }
    }

    // Header UI: one button that flips between light and dark.
    document.querySelectorAll('[data-color-scheme-toggle]').forEach((btn) => {
        const label = () => btn.setAttribute('aria-label',
            document.documentElement.getAttribute('data-color-scheme') === 'dark'
                ? t('theme.toLight') : t('theme.toDark'));
        label();
        btn.addEventListener('click', () => {
            const isDark = document.documentElement.getAttribute('data-color-scheme') === 'dark';
            setPreference(isDark ? 'light' : 'dark');
            label();
        });
    });

    // Legacy UI: select dropdown (kept for compatibility)
    const select = document.getElementById('colorSchemeSelect');
    if (select) {
        select.value = preference;
        select.addEventListener('change', () => {
            const next = select.value;
            if (next !== 'light' && next !== 'dark' && next !== 'system') return;
            setPreference(next);
        });
    }
});
