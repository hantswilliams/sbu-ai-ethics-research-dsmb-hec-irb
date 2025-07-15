// Function to handle form validation
document.addEventListener('DOMContentLoaded', function() {
    // Get all forms that need validation
    const forms = document.querySelectorAll('form');
    
    // Add validation check on submit
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Highlight required fields
                const requiredFields = form.querySelectorAll('[required]');
                requiredFields.forEach(field => {
                    if (!field.value) {
                        field.classList.add('error');
                    } else {
                        field.classList.remove('error');
                    }
                });
                
                // Show alert
                alert('Please fill out all required fields.');
            }
        });
    });
    
    // Clear error styling on input
    document.querySelectorAll('input, textarea').forEach(input => {
        input.addEventListener('input', function() {
            if (this.hasAttribute('required')) {
                if (this.value) {
                    this.classList.remove('error');
                }
            }
        });
    });
    
    // Add confirmation on submission
    const evaluationForm = document.querySelector('.evaluation-form form');
    if (evaluationForm) {
        evaluationForm.addEventListener('submit', function(event) {
            // Only confirm if all required fields are filled
            if (evaluationForm.checkValidity()) {
                const confirmed = confirm('Are you sure you want to submit this evaluation? You cannot change it later.');
                if (!confirmed) {
                    event.preventDefault();
                }
            }
        });
    }
});

// Function to format the case and response text
function formatText() {
    const caseText = document.querySelector('.case-text');
    const responseText = document.querySelector('.response-text');
    
    if (caseText) {
        caseText.innerHTML = caseText.innerHTML.replace(/\n/g, '<br>');
    }
    
    if (responseText) {
        responseText.innerHTML = responseText.innerHTML.replace(/\n/g, '<br>');
    }
}

// Call text formatting on page load
window.onload = function() {
    formatText();
};
