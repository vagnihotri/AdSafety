const uploadCard = document.getElementById('upload-card');
const imageInput = document.getElementById('image-input');
const uploadContent = document.getElementById('upload-content');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const evaluateBtn = document.getElementById('evaluate-btn');
const loadingSpinner = document.getElementById('loading-spinner');
const btnText = document.querySelector('.btn-text');
const resultsSection = document.getElementById('results-section');

let selectedFile = null;

// Drag and drop handlers
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadCard.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadCard.addEventListener(eventName, () => uploadCard.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadCard.addEventListener(eventName, () => uploadCard.classList.remove('dragover'), false);
});

uploadCard.addEventListener('drop', handleDrop, false);
uploadCard.addEventListener('click', () => {
    if (!selectedFile) imageInput.click();
});
imageInput.addEventListener('change', handleFiles, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files: files } });
}

function handleFiles(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            evaluateBtn.disabled = false;
            resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    imageInput.value = '';
    imagePreview.src = '';
    previewContainer.classList.add('hidden');
    uploadContent.classList.remove('hidden');
    evaluateBtn.disabled = true;
    resultsSection.classList.add('hidden');
}

evaluateBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI Loading state
    evaluateBtn.disabled = true;
    btnText.textContent = 'Evaluating...';
    loadingSpinner.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
            resetBtn();
        }
    } catch (error) {
        alert('Network error occurred.');
        resetBtn();
    }
});

function resetBtn() {
    evaluateBtn.disabled = false;
    btnText.textContent = 'Evaluate Image';
    loadingSpinner.classList.add('hidden');
}

function displayResults(data) {
    resetBtn();
    resultsSection.classList.remove('hidden');
    
    // Set Verdict
    const badge = document.getElementById('verdict-badge');
    badge.textContent = data.verdict;
    badge.className = 'verdict-badge ' + (
        data.verdict === 'AUTOMATED BLOCK' ? 'block' :
        data.verdict === 'HUMAN REVIEW QUEUE' ? 'review' : 'approve'
    );
    
    // Set Main Metrics
    document.getElementById('violation-score').textContent = data.violation_score.toFixed(3);
    document.getElementById('predicted-class').textContent = data.predicted_class;
    document.getElementById('confidence-subtext').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    
    // Animate Progress Bar
    setTimeout(() => {
        const progress = document.getElementById('score-progress');
        progress.style.width = `${Math.min(data.violation_score * 100, 100)}%`;
        
        if (data.verdict === 'AUTOMATED BLOCK') progress.style.backgroundColor = 'var(--danger-color)';
        else if (data.verdict === 'HUMAN REVIEW QUEUE') progress.style.backgroundColor = 'var(--warning-color)';
        else progress.style.backgroundColor = 'var(--success-color)';
    }, 100);
    
    // Populate Breakdown
    const breakdownList = document.getElementById('breakdown-list');
    breakdownList.innerHTML = '';
    
    const sortedBreakdown = Object.entries(data.breakdown)
        .sort(([,a], [,b]) => b - a);
        
    sortedBreakdown.forEach(([cls, prob]) => {
        const li = document.createElement('li');
        li.className = 'breakdown-item';
        li.innerHTML = `
            <span class="breakdown-name">${cls}</span>
            <span class="breakdown-val">${(prob * 100).toFixed(2)}%</span>
        `;
        breakdownList.appendChild(li);
    });
}
