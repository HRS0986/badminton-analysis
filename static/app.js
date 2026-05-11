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
    document.getElementById('movementDashboard').style.display = 'none';
    document.getElementById('exports').style.display = 'none';
    document.getElementById('videoPlayerSection').style.display = 'none';
    document.getElementById('movementVideoSection').style.display = 'none';
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
                        outputVideo.src = progData.exports.mp4_url + '?t=' + new Date().getTime();
                        outputVideo.load();

                        document.getElementById('movementDashboard').style.display = 'block';
                        if (progData.movement_metrics) {
                            document.getElementById('totalDist').textContent = progData.movement_metrics.total_distance_covered;
                            document.getElementById('avgSpeed').textContent = progData.movement_metrics.average_speed;
                            document.getElementById('maxSpeed').textContent = progData.movement_metrics.max_speed;
                            document.getElementById('movementEff').textContent = progData.movement_metrics.movement_efficiency;
                            document.getElementById('courtCoverage').textContent = `${progData.movement_metrics.court_coverage_percentage}%`;
                            document.getElementById('jumpCount').textContent = progData.movement_metrics.jump_count;
                            document.getElementById('avgRecoveryTime').textContent = progData.movement_metrics.average_recovery_time;
                            document.getElementById('poseStability').textContent = progData.movement_metrics.pose_stability_score;
                        }

                        document.getElementById('movementVideoSection').style.display = 'block';
                        const outputMovementVideo = document.getElementById('outputMovementVideo');
                        outputMovementVideo.src = progData.exports.movement_mp4_url + '?t=' + new Date().getTime();
                        outputMovementVideo.load();

                        document.getElementById('exports').style.display = 'block';
                        document.getElementById('btnJson').onclick = () => window.location.href = progData.exports.json_url;
                        document.getElementById('btnCsv').onclick = () => window.location.href = progData.exports.csv_url;
                        document.getElementById('btnMp4').onclick = () => window.location.href = progData.exports.mp4_url;
                        document.getElementById('btnMovementCsv').onclick = () => window.location.href = progData.exports.movement_csv_url;
                        document.getElementById('btnMovementMp4').onclick = () => window.location.href = progData.exports.movement_mp4_url;
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