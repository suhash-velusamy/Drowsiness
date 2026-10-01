/**
 * DriverGuard AI - Frontend Dashboard Controller
 * Real-time Telemetry, Web Audio Synthesizer Alarm, and Chart.js integration
 */

// Global App State
const state = {
    isRunning: false,
    pollInterval: null,
    audioEnabled: true,
    lastAlarmTime: 0,
    audioCtx: null
};

// Web Audio API Alarm Synthesizer
class AlarmSynthesizer {
    constructor() {
        this.audioCtx = null;
        this.isPlaying = false;
    }

    init() {
        if (!this.audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContext();
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
    }

    playAlarmPulse() {
        if (this.isPlaying) return;
        this.init();

        try {
            const now = this.audioCtx.currentTime;
            this.isPlaying = true;

            // Two-tone high-pitch urgency alarm (1500Hz -> 2200Hz)
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();

            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(1400, now);
            osc.frequency.exponentialRampToValueAtTime(2200, now + 0.15);
            osc.frequency.setValueAtTime(1400, now + 0.2);
            osc.frequency.exponentialRampToValueAtTime(2200, now + 0.35);

            gain.gain.setValueAtTime(0.3, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.4);

            osc.connect(gain);
            gain.connect(this.audioCtx.destination);

            osc.start(now);
            osc.stop(now + 0.4);

            setTimeout(() => {
                this.isPlaying = false;
            }, 450);
        } catch (e) {
            console.error("Audio error:", e);
            this.isPlaying = false;
        }
    }
}

const alarmSynth = new AlarmSynthesizer();

// Initialize Chart.js Telemetry Chart
let telemetryChart;
const MAX_CHART_POINTS = 35;
const chartLabels = Array(MAX_CHART_POINTS).fill('');
const earData = Array(MAX_CHART_POINTS).fill(0.3);
const marData = Array(MAX_CHART_POINTS).fill(0.1);
const threshData = Array(MAX_CHART_POINTS).fill(0.20);

function initChart() {
    const ctx = document.getElementById('telemetryChart').getContext('2d');
    
    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartLabels,
            datasets: [
                {
                    label: 'Eye Aspect Ratio (EAR)',
                    data: earData,
                    borderColor: '#00d26a',
                    backgroundColor: 'rgba(0, 210, 106, 0.08)',
                    fill: true,
                    borderWidth: 2.5,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Mouth Ratio (MAR)',
                    data: marData,
                    borderColor: '#00b4d8',
                    borderDash: [4, 4],
                    borderWidth: 1.5,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'EAR Alert Limit (0.20)',
                    data: threshData,
                    borderColor: 'rgba(255, 51, 102, 0.7)',
                    borderDash: [6, 6],
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    min: 0.0,
                    max: 0.8,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8b9bb4', font: { family: 'JetBrains Mono', size: 10 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f0f4f8', font: { family: 'Inter', size: 11 }, boxWidth: 14 }
                }
            }
        }
    });
}

// DOM Elements
const videoFeed = document.getElementById('video-feed');
const videoPlaceholder = document.getElementById('video-placeholder');
const dangerFlash = document.getElementById('danger-flash');
const toggleCameraBtn = document.getElementById('toggle-camera-btn');
const startStreamBtn = document.getElementById('start-stream-btn');
const cameraBtnText = document.getElementById('camera-btn-text');
const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const fpsBadge = document.getElementById('fps-badge');
const tripTimer = document.getElementById('trip-timer');
const headerSafetyScore = document.getElementById('header-safety-score');
const scorePill = document.getElementById('score-pill');

// Metric Elements
const earValue = document.getElementById('ear-value');
const earProgress = document.getElementById('ear-progress');
const earStatusBadge = document.getElementById('ear-status-badge');
const marValue = document.getElementById('mar-value');
const marProgress = document.getElementById('mar-progress');
const marStatusBadge = document.getElementById('mar-status-badge');
const blinksRate = document.getElementById('blinks-rate');
const totalBlinks = document.getElementById('total-blinks');
const drowsyIncidentsCount = document.getElementById('drowsy-incidents-count');
const yawnIncidentsCount = document.getElementById('yawn-incidents-count');
const incidentCountBadge = document.getElementById('incident-count-badge');

// HUD Elements
const hudEar = document.getElementById('hud-ear');
const hudMar = document.getElementById('hud-mar');
const hudClosure = document.getElementById('hud-closure');

// Settings Sliders
const earSlider = document.getElementById('ear-slider');
const earSliderVal = document.getElementById('ear-slider-val');
const earThreshDisplay = document.getElementById('ear-thresh-display');
const delaySlider = document.getElementById('delay-slider');
const delaySliderVal = document.getElementById('delay-slider-val');
const marSlider = document.getElementById('mar-slider');
const marSliderVal = document.getElementById('mar-slider-val');
const marThreshDisplay = document.getElementById('mar-thresh-display');
const audioToggle = document.getElementById('audio-toggle');

// Incidents Table
const incidentsTbody = document.getElementById('incidents-tbody');

// Modal Elements
const tripModal = document.getElementById('trip-modal');
const endTripBtn = document.getElementById('end-trip-btn');
const closeModalBtn = document.getElementById('close-modal-btn');
const newTripBtn = document.getElementById('new-trip-btn');
const modalScore = document.getElementById('modal-score');
const modalDuration = document.getElementById('modal-duration');
const modalBlinks = document.getElementById('modal-blinks');
const modalDrowsy = document.getElementById('modal-drowsy');
const modalYawns = document.getElementById('modal-yawns');

// Camera Controls
async function toggleCamera() {
    alarmSynth.init(); // Unlock AudioContext on user interaction
    try {
        const response = await fetch('/api/camera/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: state.isRunning ? 'stop' : 'start' })
        });
        const data = await response.json();
        updateCameraState(data.is_running);
    } catch (e) {
        console.error("Camera toggle error:", e);
    }
}

function updateCameraState(running) {
    state.isRunning = running;
    if (running) {
        videoFeed.src = '/video_feed?' + new Date().getTime();
        videoFeed.classList.remove('hidden');
        videoPlaceholder.classList.add('hidden');
        cameraBtnText.textContent = 'Stop Vision';
        toggleCameraBtn.classList.remove('btn-primary');
        toggleCameraBtn.classList.add('btn-secondary');
        startPolling();
    } else {
        videoFeed.src = '';
        videoFeed.classList.add('hidden');
        videoPlaceholder.classList.remove('hidden');
        cameraBtnText.textContent = 'Start Vision';
        toggleCameraBtn.classList.remove('btn-secondary');
        toggleCameraBtn.classList.add('btn-primary');
        dangerFlash.classList.remove('active');
        statusPill.className = 'status-pill status-offline';
        statusText.textContent = 'OFFLINE';
    }
}

// Telemetry Polling Loop
function startPolling() {
    if (state.pollInterval) clearInterval(state.pollInterval);
    state.pollInterval = setInterval(fetchTelemetry, 150);
}

async function fetchTelemetry() {
    if (!state.isRunning) return;

    try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        renderTelemetry(data);
    } catch (e) {
        console.warn("Telemetry fetch error:", e);
    }
}

function renderTelemetry(data) {
    // 1. Status Pill & Danger Flash
    statusPill.className = 'status-pill';
    if (data.is_drowsy) {
        statusPill.classList.add('status-drowsy');
        statusText.textContent = 'DROWSY WARNING';
        dangerFlash.classList.add('active');
        if (state.audioEnabled) {
            alarmSynth.playAlarmPulse();
        }
    } else if (data.is_yawning) {
        statusPill.classList.add('status-warning');
        statusText.textContent = 'YAWNING DETECTED';
        dangerFlash.classList.remove('active');
    } else if (data.face_detected) {
        statusPill.classList.add('status-alert');
        statusText.textContent = 'ACTIVE & ALERT';
        dangerFlash.classList.remove('active');
    } else {
        statusPill.classList.add('status-offline');
        statusText.textContent = 'NO FACE DETECTED';
        dangerFlash.classList.remove('active');
    }

    // 2. Telemetry Cards & Values
    fpsBadge.textContent = `${data.fps} FPS`;
    tripTimer.textContent = data.trip_duration;
    
    // Safety score
    headerSafetyScore.textContent = `${data.safety_score}%`;
    if (data.safety_score < 75) {
        scorePill.style.color = '#ff3366';
        scorePill.style.borderColor = 'rgba(255, 51, 102, 0.4)';
    } else {
        scorePill.style.color = '#00d26a';
        scorePill.style.borderColor = 'rgba(0, 210, 106, 0.35)';
    }

    // EAR
    earValue.textContent = data.ear.toFixed(2);
    const earPct = Math.min(100, Math.max(0, (data.ear / 0.45) * 100));
    earProgress.style.width = `${earPct}%`;
    if (data.ear < data.ear_threshold) {
        earStatusBadge.textContent = 'Closed';
        earStatusBadge.style.color = '#ff3366';
    } else {
        earStatusBadge.textContent = 'Open';
        earStatusBadge.style.color = '#00d26a';
    }

    // MAR
    marValue.textContent = data.mar.toFixed(2);
    const marPct = Math.min(100, Math.max(0, (data.mar / 1.0) * 100));
    marProgress.style.width = `${marPct}%`;
    if (data.mar > data.mar_threshold) {
        marStatusBadge.textContent = 'Yawning';
        marStatusBadge.style.color = '#ffb703';
    } else {
        marStatusBadge.textContent = 'Normal';
        marStatusBadge.style.color = '#00b4d8';
    }

    // Blinks & Incidents
    blinksRate.textContent = data.blinks_per_minute;
    totalBlinks.textContent = data.total_blinks;
    drowsyIncidentsCount.textContent = data.drowsy_incidents_count;
    yawnIncidentsCount.textContent = data.yawn_incidents_count;
    incidentCountBadge.textContent = `${data.drowsy_incidents_count + data.yawn_incidents_count} Events`;

    // Video HUD Tags
    hudEar.textContent = `EAR: ${data.ear.toFixed(2)}`;
    hudMar.textContent = `MAR: ${data.mar.toFixed(2)}`;
    hudClosure.textContent = `Closed: ${data.closed_duration.toFixed(1)}s`;

    // 3. Update Live Telemetry Chart
    if (telemetryChart) {
        earData.shift();
        earData.push(data.ear);
        marData.shift();
        marData.push(data.mar);
        threshData.shift();
        threshData.push(data.ear_threshold);
        telemetryChart.update('none');
    }
}

// Fetch and Render Incidents Log
async function fetchIncidents() {
    try {
        const res = await fetch('/api/incidents');
        const data = await res.json();
        renderIncidents(data.incidents || []);
    } catch (e) {
        console.warn("Incidents fetch error:", e);
    }
}

function renderIncidents(incidents) {
    if (!incidents || incidents.length === 0) {
        incidentsTbody.innerHTML = `
            <tr class="no-data-row">
                <td colspan="5">No fatigue incidents recorded. System monitoring is clear.</td>
            </tr>`;
        return;
    }

    incidentsTbody.innerHTML = incidents.slice(0, 15).map(item => `
        <tr>
            <td style="font-family: var(--font-mono); font-size: 0.8rem;">${item.timestamp}</td>
            <td><strong>${item.type}</strong></td>
            <td class="${item.severity === 'HIGH' ? 'severity-high' : 'severity-medium'}">${item.severity}</td>
            <td style="color: var(--text-muted);">${item.details}</td>
            <td style="font-family: var(--font-mono); color: var(--accent-blue);">${item.ear.toFixed(2)}</td>
        </tr>
    `).join('');
}

// Settings Synchronization
async function syncSettings() {
    const payload = {
        ear_threshold: parseFloat(earSlider.value),
        drowsy_time_limit: parseFloat(delaySlider.value),
        mar_threshold: parseFloat(marSlider.value),
        audio_enabled: audioToggle.checked
    };

    state.audioEnabled = audioToggle.checked;
    earSliderVal.textContent = payload.ear_threshold.toFixed(2);
    earThreshDisplay.textContent = payload.ear_threshold.toFixed(2);
    delaySliderVal.textContent = `${payload.drowsy_time_limit.toFixed(1)}s`;
    marSliderVal.textContent = payload.mar_threshold.toFixed(2);
    marThreshDisplay.textContent = payload.mar_threshold.toFixed(2);

    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    } catch (e) {
        console.error("Settings sync error:", e);
    }
}

// Reset Stats / Reset Trip
async function resetTripStats() {
    try {
        await fetch('/api/trip/reset', { method: 'POST' });
        fetchIncidents();
        fetchTelemetry();
    } catch (e) {
        console.error("Reset trip error:", e);
    }
}

// Event Listeners Setup
function setupEventListeners() {
    toggleCameraBtn.addEventListener('click', toggleCamera);
    startStreamBtn.addEventListener('click', toggleCamera);
    document.getElementById('reset-trip-btn').addEventListener('click', resetTripStats);
    
    // Test Alarm Button
    document.getElementById('test-alarm-btn').addEventListener('click', () => {
        alarmSynth.playAlarmPulse();
    });

    // Sensitivity Sliders
    earSlider.addEventListener('input', syncSettings);
    delaySlider.addEventListener('input', syncSettings);
    marSlider.addEventListener('input', syncSettings);
    audioToggle.addEventListener('change', syncSettings);

    // Reset Defaults
    document.getElementById('reset-settings-btn').addEventListener('click', () => {
        earSlider.value = 0.20;
        delaySlider.value = 0.8;
        marSlider.value = 0.65;
        audioToggle.checked = true;
        syncSettings();
    });

    // End Trip Modal
    endTripBtn.addEventListener('click', async () => {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        
        modalScore.textContent = `${data.safety_score}%`;
        modalDuration.textContent = data.trip_duration;
        modalBlinks.textContent = data.total_blinks;
        modalDrowsy.textContent = data.drowsy_incidents_count;
        modalYawns.textContent = data.yawn_incidents_count;

        tripModal.classList.remove('hidden');
    });

    closeModalBtn.addEventListener('click', () => {
        tripModal.classList.add('hidden');
    });

    newTripBtn.addEventListener('click', () => {
        tripModal.classList.add('hidden');
        resetTripStats();
    });

    // Auto-refresh incidents table every 2.5s
    setInterval(fetchIncidents, 2500);
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setupEventListeners();
    fetchIncidents();
    syncSettings();
});
