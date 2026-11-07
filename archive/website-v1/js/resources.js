document.addEventListener('DOMContentLoaded', function() {
    // Resource filtering functionality
    const filterButtons = document.querySelectorAll('.resource-filter button');
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
    
    // No modal functionality - links will navigate normally to the resource pages
});