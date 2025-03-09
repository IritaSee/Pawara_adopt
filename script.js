document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('adoptionForm');
    const uploadBoxes = document.querySelectorAll('.upload-box');
    const fileInputs = document.querySelectorAll('.file-input');

    // Handle drag and drop
    uploadBoxes.forEach(box => {
        box.addEventListener('dragover', (e) => {
            e.preventDefault();
            box.classList.add('dragover');
        });

        box.addEventListener('dragleave', (e) => {
            e.preventDefault();
            box.classList.remove('dragover');
        });

        box.addEventListener('drop', (e) => {
            e.preventDefault();
            box.classList.remove('dragover');
            const input = box.querySelector('.file-input');
            const files = e.dataTransfer.files;
            if (files.length) {
                handleFileSelect(input, files[0]);
            }
        });
    });

    // Handle file selection
    fileInputs.forEach(input => {
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            handleFileSelect(input, file);
        });
    });

    function handleFileSelect(input, file) {
        const uploadBox = input.closest('.upload-box');
        const fileInfo = uploadBox.querySelector('.file-info');
        const errorMessage = uploadBox.querySelector('.error-message');
        const progressBar = uploadBox.querySelector('.progress-bar');
        const progress = uploadBox.querySelector('.progress');

        // Reset previous state
        errorMessage.style.display = 'none';
        fileInfo.classList.remove('active');
        progressBar.style.display = 'none';
        progress.style.width = '0%';

        // Validate file
        const maxSize = input.dataset.maxSize;
        const allowedTypes = ['image/jpeg', 'image/png', 'application/pdf'];
        
        if (!allowedTypes.includes(file.type)) {
            errorMessage.textContent = 'Format file tidak sesuai. Gunakan JPG, PNG, atau PDF.';
            errorMessage.style.display = 'block';
            return;
        }

        if (file.size > maxSize * 1024 * 1024) {
            errorMessage.textContent = `Ukuran file melebihi ${maxSize}MB`;
            errorMessage.style.display = 'block';
            return;
        }

        // Show file info
        fileInfo.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)}MB)`;
        fileInfo.classList.add('active');

        // Simulate upload
        progressBar.style.display = 'block';
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 100) {
                clearInterval(interval);
            } else {
                width += 5;
                progress.style.width = width + '%';
            }
        }, 100);
    }

    // Handle form submission
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        // Here you would typically send the form data to your server
        alert('Form berhasil dikirim!');
    });

    // Handle cancel button
    document.getElementById('cancelButton').addEventListener('click', () => {
        if (confirm('Apakah Anda yakin ingin membatalkan proses adopsi?')) {
            window.location.href = '/'; // Replace with your cancel URL
        }
    });
});
function toggleChatPopup() {
    const popup = document.getElementById('chatPopupContainer');
    if (popup.style.display === 'block') {
        popup.style.display = 'none';
    } else {
        popup.style.display = 'block';
    }
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (message !== '') {
        const chatBody = document.getElementById('chatBody');
        
        // Create user message
        const userMsg = document.createElement('div');
        userMsg.className = 'chat-message user-message';
        userMsg.textContent = message;
        chatBody.appendChild(userMsg);
        
        // Clear input
        input.value = '';
        
        // Auto scroll to bottom
        chatBody.scrollTop = chatBody.scrollHeight;
        
        // Simulate AI response after a short delay
        setTimeout(() => {
            const aiMsg = document.createElement('div');
            aiMsg.className = 'chat-message ai-message';
            aiMsg.textContent = 'Terima kasih atas pertanyaan Anda. Saya, Pawara, akan membantu menjawabnya segera.';
            chatBody.appendChild(aiMsg);
            chatBody.scrollTop = chatBody.scrollHeight;
        }, 1000);
    }
}

// Allow sending message with Enter key
document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            let isValid = true;
            
            // Simple email validation
            if (!isValidEmail(email)) {
                showError('email', 'Email tidak valid');
                isValid = false;
            } else {
                removeError('email');
            }
            
            // Password validation
            if (password.length < 8) {
                showError('password', 'Password minimal 8 karakter');
                isValid = false;
            } else {
                removeError('password');
            }
            
            // If everything is valid, proceed with form submission
            if (isValid) {
                // Here you would typically send the data to your server
                console.log('Login form submitted', { email, password });
                
                // Simulate successful login (replace with actual API call)
                simulateLogin();
            }
        });
    }
    
    // Form validation for Registration
    const registerForm = document.getElementById('registerForm');
    
    if (registerForm) {
        registerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const nama = document.getElementById('nama').value;
            const email = document.getElementById('email').value;
            const noTelp = document.getElementById('noTelp').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            const terms = document.getElementById('terms').checked;
            
            let isValid = true;
            
            // Name validation
            if (nama.trim() === '') {
                showError('nama', 'Nama tidak boleh kosong');
                isValid = false;
            } else {
                removeError('nama');
            }
            
            // Email validation
            if (!isValidEmail(email)) {
                showError('email', 'Email tidak valid');
                isValid = false;
            } else {
                removeError('email');
            }
            
            // Phone validation
            if (!isValidPhone(noTelp)) {
                showError('noTelp', 'Nomor telepon tidak valid');
                isValid = false;
            } else {
                removeError('noTelp');
            }
            
            // Password validation
            if (password.length < 8) {
                showError('password', 'Password minimal 8 karakter');
                isValid = false;
            } else if (!/\d/.test(password) || !/[a-zA-Z]/.test(password)) {
                showError('password', 'Password harus berisi huruf dan angka');
                isValid = false;
            } else {
                removeError('password');
            }
            
            // Confirm password validation
            if (password !== confirmPassword) {
                showError('confirmPassword', 'Password tidak cocok');
                isValid = false;
            } else {
                removeError('confirmPassword');
            }
            
            // Terms validation
            if (!terms) {
                showError('terms', 'Anda harus menyetujui syarat dan ketentuan');
                isValid = false;
            } else {
                removeError('terms');
            }
            
            // If everything is valid, proceed with form submission
            if (isValid) {
                // Here you would typically send the data to your server
                console.log('Registration form submitted', {
                    nama,
                    email,
                    noTelp,
                    password
                });
                
                // Simulate successful registration (replace with actual API call)
                simulateRegistration();
            }
        });
    }
    
    // Helper functions
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    function isValidPhone(phone) {
        const phoneRegex = /^(\+62|62|0)8[1-9][0-9]{6,9}$/;
        return phoneRegex.test(phone);
    }
    
    function showError(fieldId, message) {
        const field = document.getElementById(fieldId);
        let errorElement = field.nextElementSibling;
        
        if (!errorElement || !errorElement.classList.contains('error-message')) {
            errorElement = document.createElement('div');
            errorElement.className = 'error-message';
            errorElement.style.color = 'var(--error-color)';
            errorElement.style.fontSize = '12px';
            errorElement.style.marginTop = '5px';
            field.parentNode.insertBefore(errorElement, field.nextSibling);
        }
        
        errorElement.textContent = message;
        field.style.borderColor = 'var(--error-color)';
    }
    
    function removeError(fieldId) {
        const field = document.getElementById(fieldId);
        const errorElement = field.nextElementSibling;
        
        if (errorElement && errorElement.classList.contains('error-message')) {
            errorElement.remove();
        }
        
        field.style.borderColor = 'var(--border-color)';
    }
    
    function simulateLogin() {
        // Show loading state
        const loginButton = document.querySelector('#loginForm button[type="submit"]');
        const originalText = loginButton.textContent;
        loginButton.textContent = 'Memproses...';
        loginButton.disabled = true;
        
        // Simulate API request delay
        setTimeout(() => {
            // Redirect to dashboard or home page
            window.location.href = 'beranda.html';
        }, 1500);
    }
    
    function simulateRegistration() {
        // Show loading state
        const registerButton = document.querySelector('#registerForm button[type="submit"]');
        const originalText = registerButton.textContent;
        registerButton.textContent = 'Mendaftarkan...';
        registerButton.disabled = true;
        
        // Simulate API request delay
        setTimeout(() => {
            // Show success message or redirect
            const authContainer = document.querySelector('.auth-container');
            
            // Create success message
            const successDiv = document.createElement('div');
            successDiv.className = 'success-message';
            successDiv.style.backgroundColor = 'rgba(56, 142, 60, 0.1)';
            successDiv.style.color = 'var(--success-color)';
            successDiv.style.padding = '15px';
            successDiv.style.borderRadius = '4px';
            successDiv.style.marginBottom = '20px';
            successDiv.style.textAlign = 'center';
            
            successDiv.innerHTML = `
                <h3 style="margin-bottom: 10px; color: var(--success-color);">Pendaftaran Berhasil!</h3>
                <p>Akun Anda berhasil dibuat. Silakan periksa email Anda untuk verifikasi.</p>
                <p style="margin-top: 15px;">
                    <a href="login.html" class="btn btn-primary">Login Sekarang</a>
                </p>
            `;
            
            // Replace form with success message
            registerForm.style.display = 'none';
            authContainer.insertBefore(successDiv, registerForm);
            
            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
            
            // Reset button state (for demo purposes)
            registerButton.textContent = originalText;
            registerButton.disabled = false;
        }, 2000);
    }
});