// Shared data helpers for the homepage design prototypes (throwaway).
const Proto = {
  async load() {
    const [claims, personas] = await Promise.all([
      fetch('/api/arguments', { credentials: 'include' }).then(r => r.json()),
      fetch('/api/persona-archetypes').then(r => r.json()),
    ]);
    const named = personas.filter(p => p.layer !== 'evs_cluster');
    return { claims, personas: named };
  },
  esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  },
  // Strip stray surrounding quotes from scraped claim text.
  clean(s) {
    return String(s || '').trim().replace(/^[«"“”'\s]+|[»"“”'\s]+$/g, '');
  },
  firstName(name) {
    return String(name || '').split(' ')[0];
  },
  initials(name) {
    return String(name || '?').split(' ').map(w => w[0]).slice(0, 2).join('').toUpperCase();
  },
  colorFor(personas, id) {
    return (personas.find(p => p.id === id) || {}).accent_color || '#29683e';
  },
  filter(claims, { q = '', persona = null, tag = null }) {
    const needle = q.trim().toLowerCase();
    return claims.filter(c =>
      (!needle || c.text.toLowerCase().includes(needle) || (c.top_response?.text || '').toLowerCase().includes(needle)) &&
      (!persona || c.personas.some(p => p.id === persona)) &&
      (!tag || c.tags.includes(tag))
    );
  },
  tags(claims) {
    const counts = {};
    claims.forEach(c => c.tags.forEach(t => (counts[t] = (counts[t] || 0) + 1)));
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([t]) => t);
  },
  // Theme: one button cycling light/dark, shared key with the real site.
  initTheme(btn) {
    const KEY = 're_inform_color_scheme';
    const apply = v => {
      const dark = v === 'dark' || (v !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
      document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    };
    let pref = 'system';
    try { pref = localStorage.getItem(KEY) || 'system'; } catch (e) {}
    apply(pref);
    btn?.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(KEY, next); } catch (e) {}
      apply(next);
    });
  },
};
