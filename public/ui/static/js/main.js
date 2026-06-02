const demoBanner = document.querySelector('.demo-banner');

function updatePreview(file) {
    if (!file) return;

    demoBanner.classList.add('show');

    const reader = new FileReader();
    reader.onload = e => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
    };
    reader.readAsDataURL(file);
}