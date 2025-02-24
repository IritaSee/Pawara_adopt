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