

/* ── Element refs ────────────────────────────────────────────────────────── */
const fileInput       = document.getElementById('videoFile');
const uploadArea      = document.getElementById('uploadArea');
const selectedFile    = document.getElementById('selectedFile');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const submitBtn       = document.getElementById('submitBtn');
const submitBtnText   = document.getElementById('submitBtnText');
const submitBtnIcon   = document.getElementById('submitBtnIcon');

const progressSection = document.getElementById('progressSection');
const progressBar     = document.getElementById('progressBar');
const progressStatus  = document.getElementById('progressStatus');
const progressPct     = document.getElementById('progressPct');

const statusBadge = document.getElementById('statusBadge');
const statusDot   = document.getElementById('statusDot');
const statusText  = document.getElementById('statusText');

/* ── Status helper ───────────────────────────────────────────────────────── */
function setStatus(state, message) {
  statusBadge.className = `status-badge ${state}`;
  statusDot.className   = `status-dot${state === 'working' ? ' pulse' : ''}`;
  statusText.textContent = message;
}

/* ── File drag & drop ────────────────────────────────────────────────────── */
uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});

['dragleave', 'dragend'].forEach(evt =>
  uploadArea.addEventListener(evt, () => uploadArea.classList.remove('drag-over'))
);

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    fileInput.files = dt.files;
    showSelectedFile(files[0].name);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    showSelectedFile(fileInput.files[0].name);
  }
});

function showSelectedFile(name) {
  fileNameDisplay.textContent = name;
  selectedFile.style.display  = 'flex';
  setStatus('idle', 'File selected');
}

/* ── Form submit ─────────────────────────────────────────────────────────── */
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  if (fileInput.files.length === 0) {
    setStatus('error', 'Please select a file first');
    return;
  }

  const file     = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  // Reset results panels
  ['dashboard', 'movementDashboard', 'exports', 'videoPlayerSection'].forEach(id => {
    document.getElementById(id).style.display = 'none';
  });
  document.getElementById('mainContainer').className = 'centered-layout';
  document.getElementById('mainContainer').querySelector('.right-column').style.display = 'none';

  // UI — uploading state
  submitBtn.disabled      = true;
  submitBtnText.textContent = 'Processing…';
  submitBtnIcon.textContent = '⏳';
  progressSection.style.display = 'block';
  progressBar.style.width = '0%';
  progressStatus.textContent = 'Uploading…';
  progressPct.textContent = '0%';
  setStatus('working', 'Uploading video…');

  try {
    const response = await fetch('/api/process-video', { method: 'POST', body: formData });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const { task_id } = await response.json();
    setStatus('working', 'Inference running…');
    progressStatus.textContent = 'Running pose detection…';

    // ── SSE (Server-Sent Events) ──────────────────────────────────────────
    const eventSource = new EventSource(`/api/progress-stream/${task_id}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === 'processing' || data.status === 'starting') {
          const pct = data.progress ?? 0;
          progressBar.style.width   = `${pct}%`;
          progressPct.textContent   = `${pct}%`;
          if (pct < 50) {
            progressStatus.textContent = 'Detecting poses…';
          } else if (pct < 90) {
            progressStatus.textContent = 'Analysing movement…';
          } else {
            progressStatus.textContent = 'Rendering videos simultaneously...';
          }

        } else if (data.status === 'completed') {
          eventSource.close();
          progressBar.style.width   = '100%';
          progressPct.textContent   = '100%';
          progressStatus.textContent = 'Complete!';
          setStatus('success', 'Analysis complete');

          submitBtn.disabled        = false;
          submitBtnText.textContent = 'Analyse Again';
          submitBtnIcon.textContent = '⚡';

          // Switch to two-column
          document.getElementById('mainContainer').className = 'two-column-layout';
          const rightCol = document.getElementById('mainContainer').querySelector('.right-column');
          rightCol.style.display = 'flex';

          // ── Pose detection section ───────────────────────────────────
          const dash = document.getElementById('dashboard');
          dash.style.display = 'block';
          if (data.metrics) {
            document.getElementById('avgConfidence').textContent =
              data.metrics.average_confidence != null
                ? `${data.metrics.average_confidence}%`
                : '—';
          }

          // ── Videos ──────────────────────────────────────────────────
          const videoSection = document.getElementById('videoPlayerSection');
          videoSection.style.display = 'block';

          const poseVid = document.getElementById('outputVideo');
          poseVid.src   = (data.exports.mp4_url) + '?t=' + Date.now();
          poseVid.load();

          const mvVid = document.getElementById('outputMovementVideo');
          mvVid.src   = (data.exports.movement_mp4_url) + '?t=' + Date.now();
          mvVid.load();



          // ── Movement metrics ─────────────────────────────────────────
          const mvDash = document.getElementById('movementDashboard');
          mvDash.style.display = 'block';
          const mm = data.movement_metrics;
          if (mm) {
            document.getElementById('totalDist').textContent      = mm.total_distance_covered ?? '—';
            document.getElementById('avgSpeed').textContent       = mm.average_speed ?? '—';
            document.getElementById('maxSpeed').textContent       = mm.max_speed ?? '—';
            document.getElementById('movementEff').textContent    = mm.movement_efficiency ?? '—';
            document.getElementById('courtCoverage').textContent  =
              mm.court_coverage_percentage != null ? `${mm.court_coverage_percentage}` : '—';
            document.getElementById('jumpCount').textContent       = mm.jump_count ?? '—';
            document.getElementById('avgRecoveryTime').textContent = mm.average_recovery_time ?? '—';
            document.getElementById('poseStability').textContent   = mm.pose_stability_score ?? '—';
          }

          // ── Exports ──────────────────────────────────────────────────
          const exDiv = document.getElementById('exports');
          exDiv.style.display = 'block';
          document.getElementById('btnJson').onclick        = () => window.location.href = data.exports.json_url;
          document.getElementById('btnCsv').onclick         = () => window.location.href = data.exports.csv_url;
          document.getElementById('btnMp4').onclick         = () => window.location.href = data.exports.mp4_url;
          document.getElementById('btnMovementCsv').onclick = () => window.location.href = data.exports.movement_csv_url;
          document.getElementById('btnMovementMp4').onclick = () => window.location.href = data.exports.movement_mp4_url;

        } else if (data.status === 'failed') {
          eventSource.close();
          setStatus('error', `Error: ${data.error || 'Processing failed'}`);
          progressStatus.textContent = 'Processing failed';
          submitBtn.disabled      = false;
          submitBtnText.textContent = 'Retry';
          submitBtnIcon.textContent = '🔄';
        }
    };

    eventSource.onerror = (err) => {
      eventSource.close();
      setStatus('error', 'Lost connection to server');
    };

  } catch (err) {
    setStatus('error', err.message);
    submitBtn.disabled        = false;
    submitBtnText.textContent = 'Retry';
    submitBtnIcon.textContent = '🔄';
  }
});