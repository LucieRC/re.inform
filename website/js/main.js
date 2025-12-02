// Main JavaScript for re.inform website

document.addEventListener('DOMContentLoaded', function() {
    // Mobile navigation toggle
    const createMobileNav = () => {
        const header = document.querySelector('header');
        const nav = document.querySelector('nav');
        
        if (header && nav && window.innerWidth <= 768) {
            const mobileToggle = document.createElement('div');
            mobileToggle.className = 'mobile-nav-toggle';
            mobileToggle.innerHTML = '☰';
            mobileToggle.addEventListener('click', function() {
                nav.classList.toggle('active');
                this.innerHTML = nav.classList.contains('active') ? '✕' : '☰';
            });
            
            // Only add if it doesn't exist already
            if (!document.querySelector('.mobile-nav-toggle')) {
                header.insertBefore(mobileToggle, nav);
                // Add styles
                document.head.insertAdjacentHTML('beforeend', `
                <style>
                    @media (max-width: 768px) {
                        nav {
                            display: none;
                        }
                        nav.active {
                            display: block;
                            width: 100%;
                        }
                        .mobile-nav-toggle {
                            display: block;
                            font-size: 1.5rem;
                            cursor: pointer;
                        }
                    }
                </style>
                `);
            }
        }
    };

    // Call on load and resize
    createMobileNav();
    window.addEventListener('resize', createMobileNav);

    // Translation functionality
    let translations = {};
    let currentLanguage = 'en';

    // Load translations
    const loadTranslations = async (lang) => {
        try {
            const response = await fetch(`translations/${lang}.json`);
            if (!response.ok) {
                throw new Error('Translation file not found');
            }
            return await response.json();
        } catch (error) {
            console.error('Failed to load translations:', error);
            return null;
        }
    };

    // Apply translations to the page
    const applyTranslations = () => {
        if (!translations) return;
        
        // Navigation
        document.querySelectorAll('nav ul li a').forEach((link, index) => {
            const key = link.getAttribute('href').replace('.html', '').replace('index', 'home');
            if (translations.nav[key]) {
                link.textContent = translations.nav[key];
            }
        });
        
        // Header
        document.querySelectorAll('.discord-button a').forEach(button => {
            button.textContent = translations.header.joinDiscord;
        });
        
        // Hero section
        const heroSection = document.querySelector('.hero');
        if (heroSection) {
            const heroTitle = heroSection.querySelector('h1');
            const heroSubtitle = heroSection.querySelector('.subtitle');
            const heroButtons = heroSection.querySelectorAll('.hero-buttons a');
            
            if (heroTitle) heroTitle.textContent = translations.hero.title;
            if (heroSubtitle) heroSubtitle.textContent = translations.hero.subtitle;
            if (heroButtons.length >= 2) {
                heroButtons[0].textContent = translations.hero.joinButton;
                heroButtons[1].textContent = translations.hero.learnButton;
            }
        }
        
        // Problem statement
        const problemSection = document.querySelector('.problem-statement');
        if (problemSection) {
            const problemTitle = problemSection.querySelector('h2');
            const problemDesc = problemSection.querySelector('p');
            
            if (problemTitle) problemTitle.textContent = translations.problem.title;
            if (problemDesc) problemDesc.textContent = translations.problem.description;
        }
        
        // Approach section
        const approachSection = document.querySelector('.solution-overview');
        if (approachSection) {
            const approachTitle = approachSection.querySelector('h2');
            const steps = approachSection.querySelectorAll('.methodology-step');
            
            if (approachTitle) approachTitle.textContent = translations.approach.title;
            if (steps.length >= 3) {
                steps[0].querySelector('h3').textContent = translations.approach.step1Title;
                steps[0].querySelector('p').textContent = translations.approach.step1Desc;
                steps[1].querySelector('h3').textContent = translations.approach.step2Title;
                steps[1].querySelector('p').textContent = translations.approach.step2Desc;
                steps[2].querySelector('h3').textContent = translations.approach.step3Title;
                steps[2].querySelector('p').textContent = translations.approach.step3Desc;
            }
        }
        
        // Resources section
        const resourcesSection = document.querySelector('.featured-resources');
        if (resourcesSection) {
            const resourcesTitle = resourcesSection.querySelector('h2');
            const resources = resourcesSection.querySelectorAll('.resource-card');
            
            if (resourcesTitle) resourcesTitle.textContent = translations.resources.title;
            if (resources.length >= 3) {
                resources[0].querySelector('h3').textContent = translations.resources.resource1Title;
                resources[0].querySelector('p').textContent = translations.resources.resource1Desc;
                resources[0].querySelector('.read-more').textContent = translations.resources.readMore;
                
                resources[1].querySelector('h3').textContent = translations.resources.resource2Title;
                resources[1].querySelector('p').textContent = translations.resources.resource2Desc;
                resources[1].querySelector('.read-more').textContent = translations.resources.readMore;
                
                resources[2].querySelector('h3').textContent = translations.resources.resource3Title;
                resources[2].querySelector('p').textContent = translations.resources.resource3Desc;
                resources[2].querySelector('.read-more').textContent = translations.resources.readMore;
            }
        }
        
        // CTA section
        const ctaSection = document.querySelector('.cta');
        if (ctaSection) {
            const ctaTitle = ctaSection.querySelector('h2');
            const ctaDesc = ctaSection.querySelector('p');
            const ctaButtons = ctaSection.querySelectorAll('.cta-buttons a');
            
            if (ctaTitle) ctaTitle.textContent = translations.cta.title;
            if (ctaDesc) ctaDesc.textContent = translations.cta.description;
            if (ctaButtons.length >= 2) {
                ctaButtons[0].textContent = translations.cta.joinButton;
                ctaButtons[1].textContent = translations.cta.feedbackButton;
            }
        }
        
        // Footer
        const footer = document.querySelector('footer');
        if (footer) {
            const footerTagline = footer.querySelector('.footer-column p');
            const footerHeadings = footer.querySelectorAll('.footer-column h3');
            const footerCopyright = footer.querySelector('.copyright p');
            
            if (footerTagline) footerTagline.textContent = translations.footer.tagline;
            if (footerHeadings.length >= 3) {
                footerHeadings[1].textContent = translations.footer.pages;
                footerHeadings[2].textContent = translations.footer.connect;
            }
            if (footerCopyright) footerCopyright.textContent = translations.footer.copyright;
        }
        
        // Feedback float
        const feedbackFloat = document.querySelector('.feedback-float a');
        if (feedbackFloat) feedbackFloat.textContent = translations.nav.feedback;
        
        // Update HTML lang attribute
        document.documentElement.lang = currentLanguage;
    };

    // Language selector functionality
    const languageSelect = document.getElementById('language-select');
    if (languageSelect) {
        // Set initial language based on browser preference or default
        currentLanguage = localStorage.getItem('preferredLanguage') || 'en';
        languageSelect.value = currentLanguage;
        
        // Initial load of translations
        loadTranslations(currentLanguage).then(data => {
            translations = data;
            applyTranslations();
        });
        
        languageSelect.addEventListener('change', async function() {
            const selectedLanguage = this.value;
            currentLanguage = selectedLanguage;
            
            // Save preference
            localStorage.setItem('preferredLanguage', selectedLanguage);
            
            // Load and apply new translations
            translations = await loadTranslations(selectedLanguage);
            applyTranslations();
        });
    }

    // Form validation for feedback and contact forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            let valid = true;
            const requiredInputs = form.querySelectorAll('[required]');
            
            requiredInputs.forEach(input => {
                if (!input.value.trim()) {
                    valid = false;
                    input.classList.add('error');
                    
                    // Create or update error message
                    let errorMsg = input.nextElementSibling;
                    if (!errorMsg || !errorMsg.classList.contains('error-message')) {
                        errorMsg = document.createElement('div');
                        errorMsg.className = 'error-message';
                        input.parentNode.insertBefore(errorMsg, input.nextSibling);
                    }
                    errorMsg.textContent = 'This field is required';
                } else {
                    input.classList.remove('error');
                    const errorMsg = input.nextElementSibling;
                    if (errorMsg && errorMsg.classList.contains('error-message')) {
                        errorMsg.textContent = '';
                    }
                }
            });
            
            if (!valid) {
                e.preventDefault();
            } else {
                // For demo purposes, prevent actual submission and show success message
                e.preventDefault();
                form.innerHTML = '<div class="success-message"><h3>Thank you!</h3><p>Your submission has been received.</p></div>';
            }
        });
    });

    // Resource filtering functionality
    const resourceFilter = document.querySelector('.resource-filter');
    if (resourceFilter) {
        const filterButtons = resourceFilter.querySelectorAll('button');
        const resourceItems = document.querySelectorAll('.resource-item');
        
        filterButtons.forEach(button => {
            button.addEventListener('click', function() {
                const filter = this.getAttribute('data-filter');
                
                // Update active button
                filterButtons.forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                
                // Filter resources
                resourceItems.forEach(item => {
                    if (filter === 'all' || item.classList.contains(filter)) {
                        item.style.display = 'block';
                    } else {
                        item.style.display = 'none';
                    }
                });
            });
        });
    }
    
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            if(this.getAttribute('href') !== '#') {
                e.preventDefault();
                
                const target = document.querySelector(this.getAttribute('href'));
                if(target) {
                    target.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
    
    // Animate elements on scroll
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.methodology-step, .resource-card, .contribute-card');
        
        elements.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            if(elementTop < windowHeight - 100) {
                if(!element.classList.contains('animated')) {
                    element.classList.add('animated');
                }
            }
        });
    };
    
    // Add animation CSS
    document.head.insertAdjacentHTML('beforeend', `
    <style>
        .methodology-step, .resource-card, .contribute-card {
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.6s ease, transform 0.6s ease;
        }
        
        .methodology-step.animated, .resource-card.animated, .contribute-card.animated {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
    `);
    
    // Call once on load and then on scroll
    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll();
});
