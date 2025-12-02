// ========================================
// re.inform - Main JavaScript
// ========================================

// Configuration
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:5001/api' 
    : '/api';

console.log('API Base URL:', API_BASE_URL);

// ========================================
// Mobile Navigation Toggle
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }
    
    // Initialize page-specific functionality
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    
    if (currentPage.includes('contribute')) {
        initContributePage();
    } else if (currentPage.includes('contributions')) {
        initContributionsPage();
    } else if (currentPage.includes('resources')) {
        initResourcesPage();
    } else if (currentPage.includes('admin')) {
        initAdminPage();
    }
});

// ========================================
// Contribution Form
// ========================================

function initContributePage() {
    const form = document.getElementById('contributionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resetBtn = document.getElementById('resetBtn');
    const formMessage = document.getElementById('formMessage');
    
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get form data
        const selectedTags = Array.from(document.querySelectorAll('input[name="tags"]:checked'))
            .map(checkbox => checkbox.value);
        
        const formData = {
            narrative: document.getElementById('narrative').value.trim(),
            counter_argument: document.getElementById('counterArgument').value.trim(),
            source: document.getElementById('source').value.trim(),
            tags: selectedTags
        };
        
        // Basic validation
        if (!formData.source) {
            showFormMessage('Please provide a source/reference.', 'error');
            return;
        }
        
        // Disable submit button
        submitBtn.disabled = true;
        submitBtn.textContent = 'Submitting...';
        
        try {
            const response = await fetch(`${API_BASE_URL}/submit-contribution`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showFormMessage(data.message || 'Contribution submitted successfully!', 'success');
                form.reset();
                
                // Scroll to message
                formMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                showFormMessage(data.error || 'Failed to submit contribution. Please try again.', 'error');
            }
        } catch (error) {
            console.error('Submission error:', error);
            showFormMessage('Network error. Please check your connection and try again.', 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Submit Contribution';
        }
    });
    
    resetBtn.addEventListener('click', () => {
        form.reset();
        hideFormMessage();
    });
}

function showFormMessage(message, type) {
    const formMessage = document.getElementById('formMessage');
    formMessage.textContent = message;
    formMessage.className = `form-message ${type}`;
}

function hideFormMessage() {
    const formMessage = document.getElementById('formMessage');
    formMessage.className = 'form-message';
}

// ========================================
// Contributions Page
// ========================================

let allContributions = [];
let allContributionTags = new Set();

async function initContributionsPage() {
    await loadContributions();
    
    const sortSelect = document.getElementById('sortBy');
    const tagFilter = document.getElementById('tagFilter');
    
    if (sortSelect) {
        sortSelect.addEventListener('change', filterAndDisplayContributions);
    }
    
    if (tagFilter) {
        tagFilter.addEventListener('change', filterAndDisplayContributions);
    }
}

async function loadContributions() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const grid = document.getElementById('counterArgumentsGrid');
    
    try {
        const response = await fetch(`${API_BASE_URL}/contributions`);
        
        if (!response.ok) {
            throw new Error('Failed to load contributions');
        }
        
        allContributions = await response.json();
        
        // Extract all unique tags
        allContributions.forEach(c => {
            if (c.tags && Array.isArray(c.tags)) {
                c.tags.forEach(tag => allContributionTags.add(tag));
            }
        });
        
        // Populate tag filter
        populateContributionTagFilter();
        
        // Hide loading message
        loadingMessage.style.display = 'none';
        
        // Display contributions
        filterAndDisplayContributions();
        
    } catch (error) {
        console.error('Error loading contributions:', error);
        loadingMessage.style.display = 'none';
        errorMessage.textContent = 'Failed to load contributions. Please try again later.';
        errorMessage.style.display = 'block';
    }
}

function populateContributionTagFilter() {
    const tagFilter = document.getElementById('tagFilter');
    if (!tagFilter) return;
    
    // Clear existing options (except "All Topics")
    tagFilter.innerHTML = '<option value="">All topics</option>';
    
    // Add tag options
    Array.from(allContributionTags).sort().forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = capitalizeWords(tag);
        tagFilter.appendChild(option);
    });
}

function filterAndDisplayContributions() {
    const sortBy = document.getElementById('sortBy')?.value || 'date';
    const tagFilter = document.getElementById('tagFilter')?.value || '';
    
    let filtered = [...allContributions];
    
    // Filter by tag
    if (tagFilter) {
        filtered = filtered.filter(c => c.tags && c.tags.includes(tagFilter));
    }
    
    // Sort
    if (sortBy === 'title') {
        filtered.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
    } else { // date
        filtered.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
    }
    
    // Display
    displayContributions(filtered);
}

function displayContributions(contributions) {
    const grid = document.getElementById('counterArgumentsGrid');
    const emptyMessage = document.getElementById('emptyMessage');
    
    if (!grid) return;
    
    if (contributions.length === 0) {
        grid.style.display = 'none';
        emptyMessage.style.display = 'block';
        return;
    }
    
    grid.style.display = 'grid';
    emptyMessage.style.display = 'none';
    grid.innerHTML = '';
    
    contributions.forEach(contribution => {
        const card = createContributionCard(contribution);
        grid.appendChild(card);
    });
}

function createContributionCard(contribution) {
    const card = document.createElement('div');
    card.className = 'counter-argument-card';
    
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
    
    card.innerHTML = `
        <div class="card-header">
            <h3 class="card-title">${contribution.title || 'Untitled'}</h3>
        </div>
        ${contribution.narrative ? `<div class="card-narrative">"${contribution.narrative}"</div>` : ''}
        <div class="card-summary">${contribution.counter_argument || ''}</div>
        ${tagsHTML}
        <div class="card-meta">
            ${sourceHTML}
            ${dateHTML}
        </div>
    `;
    
    return card;
}

// ========================================
// Resources Page (Merged with Contributions)
// ========================================

let allResources = [];
let allContributions = [];
let allMergedTags = new Set();

async function initResourcesPage() {
    console.log('*** initResourcesPage called ***');
    await loadMergedContent();
    
    const sortSelect = document.getElementById('sortBy');
    const tagFilter = document.getElementById('tagFilter');
    
    if (sortSelect) {
        sortSelect.addEventListener('change', filterAndDisplayMergedContent);
    }
    
    if (tagFilter) {
        tagFilter.addEventListener('change', filterAndDisplayMergedContent);
    }
}

async function loadMergedContent() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    
    console.log('Starting to load merged content...');
    
    try {
        // Load both resources and contributions
        console.log('Fetching from:', `${API_BASE_URL}/resources`, `${API_BASE_URL}/contributions`);
        
        const [resourcesResponse, contributionsResponse] = await Promise.all([
            fetch(`${API_BASE_URL}/resources`),
            fetch(`${API_BASE_URL}/contributions`)
        ]);
        
        console.log('Resources response:', resourcesResponse.status);
        console.log('Contributions response:', contributionsResponse.status);
        
        if (!resourcesResponse.ok || !contributionsResponse.ok) {
            throw new Error('Failed to load content');
        }
        
        allResources = await resourcesResponse.json();
        allContributions = await contributionsResponse.json();
        
        console.log('Loaded resources:', allResources.length);
        console.log('Loaded contributions:', allContributions.length);
        
        // Extract all unique tags from both
        [...allResources, ...allContributions].forEach(item => {
            if (item.tags && Array.isArray(item.tags)) {
                item.tags.forEach(tag => allMergedTags.add(tag));
            }
        });
        
        // Populate tag filter
        populateMergedTagFilter();
        
        // Hide loading message
        loadingMessage.style.display = 'none';
        
        // Display both sections
        filterAndDisplayMergedContent();
        
    } catch (error) {
        console.error('Error loading content:', error);
        loadingMessage.style.display = 'none';
        errorMessage.textContent = 'Failed to load content. Please try again later.';
        errorMessage.style.display = 'block';
    }
}

function populateMergedTagFilter() {
    const tagFilter = document.getElementById('tagFilter');
    if (!tagFilter) return;
    
    // Clear existing options (except "All Topics")
    tagFilter.innerHTML = '<option value="">All topics</option>';
    
    // Add tag options
    Array.from(allMergedTags).sort().forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = capitalizeWords(tag);
        tagFilter.appendChild(option);
    });
}

function filterAndDisplayMergedContent() {
    const sortBy = document.getElementById('sortBy')?.value || 'rating';
    const tagFilter = document.getElementById('tagFilter')?.value || '';
    
    // Filter resources
    let filteredResources = [...allResources];
    if (tagFilter) {
        filteredResources = filteredResources.filter(r => r.tags && r.tags.includes(tagFilter));
    }
    
    // Filter contributions
    let filteredContributions = [...allContributions];
    if (tagFilter) {
        filteredContributions = filteredContributions.filter(c => c.tags && c.tags.includes(tagFilter));
    }
    
    // Sort resources
    if (sortBy === 'rating') {
        filteredResources.sort((a, b) => (b.effectiveness_rating || 0) - (a.effectiveness_rating || 0));
    } else if (sortBy === 'title') {
        filteredResources.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
    } else { // date
        filteredResources.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
    }
    
    // Sort contributions
    if (sortBy === 'title') {
        filteredContributions.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
    } else { // date (rating doesn't apply to contributions)
        filteredContributions.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
    }
    
    // Display both sections
    displayMergedResources(filteredResources);
    displayMergedContributions(filteredContributions);
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

function displayMergedContributions(contributions) {
    const grid = document.getElementById('contributionsGrid');
    const emptyMessage = document.getElementById('emptyContributionsMessage');
    
    if (!grid) return;
    
    if (contributions.length === 0) {
        grid.style.display = 'none';
        emptyMessage.style.display = 'block';
        return;
    }
    
    grid.style.display = 'grid';
    emptyMessage.style.display = 'none';
    grid.innerHTML = '';
    
    contributions.forEach(contribution => {
        const card = createContributionCard(contribution);
        grid.appendChild(card);
    });
}

// ========================================
// Old Resources Page (deprecated - keeping for backward compatibility)
// ========================================

let allResourcesOld = [];
let allResourceTags = new Set();

async function initResourcesPageOld() {
    await loadResources();
    
    const sortSelect = document.getElementById('sortBy');
    const tagFilter = document.getElementById('tagFilter');
    
    if (sortSelect) {
        sortSelect.addEventListener('change', filterAndDisplayResources);
    }
    
    if (tagFilter) {
        tagFilter.addEventListener('change', filterAndDisplayResources);
    }
}

async function loadResources() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const grid = document.getElementById('counterArgumentsGrid');
    
    try {
        const response = await fetch(`${API_BASE_URL}/resources`);
        
        if (!response.ok) {
            throw new Error('Failed to load resources');
        }
        
        allResources = await response.json();
        
        // Extract all unique tags
        allResources.forEach(r => {
            if (r.tags && Array.isArray(r.tags)) {
                r.tags.forEach(tag => allResourceTags.add(tag));
            }
        });
        
        // Populate tag filter
        populateResourceTagFilter();
        
        // Hide loading message
        loadingMessage.style.display = 'none';
        
        // Display resources
        filterAndDisplayResources();
        
    } catch (error) {
        console.error('Error loading resources:', error);
        loadingMessage.style.display = 'none';
        errorMessage.textContent = 'Failed to load resources. Please try again later.';
        errorMessage.style.display = 'block';
    }
}

function populateResourceTagFilter() {
    const tagFilter = document.getElementById('tagFilter');
    if (!tagFilter) return;
    
    // Clear existing options (except "All Topics")
    tagFilter.innerHTML = '<option value="">All topics</option>';
    
    // Add tag options
    Array.from(allResourceTags).sort().forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = capitalizeWords(tag);
        tagFilter.appendChild(option);
    });
}

function filterAndDisplayResources() {
    const sortBy = document.getElementById('sortBy')?.value || 'rating';
    const tagFilter = document.getElementById('tagFilter')?.value || '';
    
    let filtered = [...allResources];
    
    // Filter by tag
    if (tagFilter) {
        filtered = filtered.filter(r => r.tags && r.tags.includes(tagFilter));
    }
    
    // Sort
    if (sortBy === 'rating') {
        filtered.sort((a, b) => (b.effectiveness_rating || 0) - (a.effectiveness_rating || 0));
    } else if (sortBy === 'title') {
        filtered.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
    } else { // date
        filtered.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
    }
    
    // Display
    displayResources(filtered);
}

function displayResources(resources) {
    const grid = document.getElementById('counterArgumentsGrid');
    const emptyMessage = document.getElementById('emptyMessage');
    
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
    
    const sourceHTML = resource.source 
        ? `<div class="card-source"><a href="${resource.source}" target="_blank" rel="noopener noreferrer">View source</a></div>`
        : '';
    
    const dateHTML = resource.date 
        ? `<div class="card-date">${new Date(resource.date).toLocaleDateString()}</div>`
        : '';
    
    card.innerHTML = `
        <div class="card-header">
            <h3 class="card-title">${resource.title || 'Untitled'}</h3>
            <span class="effectiveness-badge">★ ${resource.effectiveness_rating || 0}/10</span>
        </div>
        ${resource.narrative ? `<div class="card-narrative">"${resource.narrative}"</div>` : ''}
        <div class="card-summary">${resource.summary || resource.counter_argument || ''}</div>
        ${tagsHTML}
        <div class="card-meta">
            ${sourceHTML}
            ${dateHTML}
        </div>
    `;
    
    return card;
}

// ========================================
// Old Explore Page (deprecated - keeping for backward compatibility)
// ========================================

let allCounterArguments = [];
let allTags = new Set();

async function initExplorePage() {
    await loadCounterArguments();
    
    const sortSelect = document.getElementById('sortBy');
    const tagFilter = document.getElementById('tagFilter');
    
    if (sortSelect) {
        sortSelect.addEventListener('change', filterAndDisplayCounterArguments);
    }
    
    if (tagFilter) {
        tagFilter.addEventListener('change', filterAndDisplayCounterArguments);
    }
}

async function loadCounterArguments() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const grid = document.getElementById('counterArgumentsGrid');
    
    try {
        const response = await fetch(`${API_BASE_URL}/counter-arguments`);
        
        if (!response.ok) {
            throw new Error('Failed to load counter-arguments');
        }
        
        allCounterArguments = await response.json();
        
        // Extract all unique tags
        allCounterArguments.forEach(ca => {
            if (ca.tags && Array.isArray(ca.tags)) {
                ca.tags.forEach(tag => allTags.add(tag));
            }
        });
        
        // Populate tag filter
        populateTagFilter();
        
        // Hide loading message
        loadingMessage.style.display = 'none';
        
        // Display counter-arguments
        filterAndDisplayCounterArguments();
        
    } catch (error) {
        console.error('Error loading counter-arguments:', error);
        loadingMessage.style.display = 'none';
        errorMessage.textContent = 'Failed to load counter-arguments. Please try again later.';
        errorMessage.style.display = 'block';
    }
}

function populateTagFilter() {
    const tagFilter = document.getElementById('tagFilter');
    if (!tagFilter) return;
    
    // Clear existing options (except "All Topics")
    tagFilter.innerHTML = '<option value="">All Topics</option>';
    
    // Add tag options
    Array.from(allTags).sort().forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = capitalizeWords(tag);
        tagFilter.appendChild(option);
    });
}

function filterAndDisplayCounterArguments() {
    const sortBy = document.getElementById('sortBy')?.value || 'rating';
    const tagFilter = document.getElementById('tagFilter')?.value || '';
    
    let filtered = [...allCounterArguments];
    
    // Filter by tag
    if (tagFilter) {
        filtered = filtered.filter(ca => 
            ca.tags && ca.tags.includes(tagFilter)
        );
    }
    
    // Sort
    if (sortBy === 'rating') {
        filtered.sort((a, b) => (b.effectiveness_rating || 0) - (a.effectiveness_rating || 0));
    } else if (sortBy === 'date') {
        filtered.sort((a, b) => new Date(b.date) - new Date(a.date));
    } else if (sortBy === 'title') {
        filtered.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
    }
    
    displayCounterArguments(filtered);
}

function displayCounterArguments(counterArguments) {
    const grid = document.getElementById('counterArgumentsGrid');
    const emptyMessage = document.getElementById('emptyMessage');
    
    if (!grid) return;
    
    if (counterArguments.length === 0) {
        grid.style.display = 'none';
        emptyMessage.style.display = 'block';
        return;
    }
    
    grid.style.display = 'grid';
    emptyMessage.style.display = 'none';
    
    grid.innerHTML = counterArguments.map(ca => createCounterArgumentCard(ca)).join('');
}

function createCounterArgumentCard(ca) {
    const date = ca.date ? new Date(ca.date).toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    }) : 'N/A';
    
    const tags = ca.tags && ca.tags.length > 0
        ? ca.tags.map(tag => `<span class="tag">${capitalizeWords(tag)}</span>`).join('')
        : '';
    
    const source = ca.source 
        ? `<div class="card-source"><a href="${ca.source}" target="_blank" rel="noopener noreferrer">View Source</a></div>`
        : '<div class="card-source">No source</div>';
    
    return `
        <div class="counter-argument-card">
            <div class="card-header">
                <h3 class="card-title">${escapeHtml(ca.title)}</h3>
                <span class="effectiveness-badge">${ca.effectiveness_rating || 0}/10</span>
            </div>
            <div class="card-narrative">
                <strong>Addresses:</strong> ${escapeHtml(ca.narrative)}
            </div>
            <div class="card-summary">
                ${escapeHtml(ca.summary)}
            </div>
            <div class="card-tags">
                ${tags}
            </div>
            <div class="card-meta">
                ${source}
                <div class="card-date">${date}</div>
            </div>
        </div>
    `;
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
// Utility Functions
// ========================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function capitalizeWords(str) {
    return str.replace(/\b\w/g, char => char.toUpperCase());
}
