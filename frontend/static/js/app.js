// Custom JavaScript for Adaptive Portfolio Engine

// Utility functions
function formatCurrency(value) {
    return `₹${value.toLocaleString('en-IN')}`;
}

function formatPercentage(value) {
    return `${(value * 100).toFixed(2)}%`;
}

// Form validation
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const startYear = parseInt(document.getElementById('start_year')?.value);
            const endYear = parseInt(document.getElementById('end_year')?.value);
            
            if (startYear && endYear && endYear <= startYear) {
                event.preventDefault();
                alert('End year must be greater than start year!');
                return false;
            }
        });
    });
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Tooltip initialization (if using Bootstrap tooltips)
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});

// Auto-hide alerts after 5 seconds
setTimeout(function() {
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        const bsAlert = new bootstrap.Alert(alert);
        bsAlert.close();
    });
}, 5000);
