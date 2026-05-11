const fileInput = document.getElementById('videoFile');
const fileNameDisplay = document.getElementById('file-name-display');

fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = fileInput.files[0].name;
    } else {
        fileNameDisplay.textContent = "Click to Select a Video (MP4)";
    }
});

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    if (fileInput.files.length === 0) {
        alert("Please select a video file first.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const statusMessage = document.getElementById('statusMessage');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const submitBtn = document.getElementById('submitBtn');

    statusMessage.textContent = "Uploading & initializing... Please wait.";
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    document.getElementById('dashboard').style.display = 'none';
    document.getElementById('exports').style.display = 'none';
    document.getElementById('videoPlayerSection').style.display = 'none';
    submitBtn.disabled = true;

    // reset to centered layout on new upload iteration
    document.getElementById('mainContainer').className = 'centered-layout';

    try {
        const response = await fetch('/api/process-video', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            const taskId = data.task_id;

            // Poll for progress
            const interval = setInterval(async () => {
                try {
                    const progRes = await fetch(`/api/progress/${taskId}`);
                    const progData = await progRes.json();

                    if (progData.status === 'processing' || progData.status === 'starting') {
                        statusMessage.textContent = `Processing: ${progData.progress}%`;
                        progressBar.style.width = `${progData.progress}%`;
                    } else if (progData.status === 'completed') {
                        clearInterval(interval);
                        statusMessage.textContent = "Processing complete!";
                        progressBar.style.width = '100%';
                        submitBtn.disabled = false;

                        // Switch to two column layout now that we have results
                        document.getElementById('mainContainer').className = 'two-column-layout';

                        document.getElementById('dashboard').style.display = 'block';
                        document.getElementById('avgConfidence').textContent = `${progData.metrics.average_confidence}%`;

                        document.getElementById('videoPlayerSection').style.display = 'block';
                        const outputVideo = document.getElementById('outputVideo');
                        outputVideo.src = progData.exports.mp4_url;
                        outputVideo.load();

                        document.getElementById('exports').style.display = 'block';
                        document.getElementById('btnJson').onclick = () => window.location.href = progData.exports.json_url;
                        document.getElementById('btnCsv').onclick = () => window.location.href = progData.exports.csv_url;
                        document.getElementById('btnMp4').onclick = () => window.location.href = progData.exports.mp4_url;
                    } else if (progData.status === 'failed') {
                        clearInterval(interval);
                        statusMessage.textContent = `Error: ${progData.error}`;
                        submitBtn.disabled = false;
                    }
                } catch (err) {
                    console.error("Polling error", err);
                }
            }, 1000); // Check every second

        } else {
            const error = await response.json();
            statusMessage.textContent = `Error: ${error.detail}`;
            submitBtn.disabled = false;
        }
    } catch (e) {
        statusMessage.textContent = `Error: ${e.message}`;
        submitBtn.disabled = false;
    }
});