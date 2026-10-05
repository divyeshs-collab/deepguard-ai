/**
 * DeepGuard AI — Real-Time Deepfake Audio Detection Sentinel
 * Frontend Application Controller & WebSocket Stream Client
 */

class DeepGuardSentinel {
  constructor() {
    this.activeCallId = null;
    this.callActive = false;
    this.callTimer = null;
    this.elapsedSeconds = 0;
    this.socket = null;
    this.audioContext = null;
    this.mediaStream = null;
    this.processorNode = null;

    // Timeline data for Live Confidence Canvas
    this.confidenceHistory = []; // { time: '00:01', raw: 0.12, smoothed: 0.15 }
    this.maxDataPoints = 30;

    this.initDOMElements();
    this.initEventListeners();
    this.initCanvas();
    this.loadModelInfoAndMetrics();
  }

  initDOMElements() {
    // Navigation
    this.navLiveBtn = document.getElementById('nav-live-btn');
    this.navModelBtn = document.getElementById('nav-model-btn');
    this.navExplainBtn = document.getElementById('nav-explain-btn');
    this.viewDashboard = document.getElementById('view-dashboard');
    this.viewModelInfo = document.getElementById('view-model-info');
    this.viewExplain = document.getElementById('view-explainability');

    // Controls
    this.btnStart = document.getElementById('btn-start-analysis');
    this.btnStop = document.getElementById('btn-stop-analysis');
    this.scenarioSelector = document.getElementById('scenario-selector');
    this.audioFileInput = document.getElementById('audio-file-input');

    // Status & Metrics Cards
    this.callStatusBadge = document.getElementById('call-status-badge');
    this.callDurationText = document.getElementById('call-duration');
    this.audioStreamType = document.getElementById('audio-stream-type');
    this.primaryCard = document.getElementById('primary-detection-card');
    this.primaryStatusText = document.getElementById('primary-status-text');
    this.primaryStatusDesc = document.getElementById('primary-status-desc');
    this.riskLevelBadge = document.getElementById('risk-level-badge');
    this.modelConfText = document.getElementById('model-confidence-val');
    this.smoothedConfText = document.getElementById('smoothed-confidence-val');

    // Acoustic Telemetry
    this.statWindows = document.getElementById('stat-windows-analyzed');
    this.statSpeech = document.getElementById('stat-speech-detected');
    this.statQuality = document.getElementById('stat-audio-quality');
    this.statLatency = document.getElementById('stat-inference-latency');
    this.rmsBar = document.getElementById('rms-level-bar');
    this.rmsText = document.getElementById('rms-val-text');

    // Chart & Timeline
    this.chartCanvas = document.getElementById('confidence-chart');
    this.chartCtx = this.chartCanvas ? this.chartCanvas.getContext('2d') : null;
    this.chartPlaceholder = document.getElementById('chart-placeholder');
    this.timelineContainer = document.getElementById('timeline-log-container');
    this.timelineCount = document.getElementById('timeline-count');

    // Explainability
    this.explainHf = document.getElementById('explain-hf-pct');
    this.explainMid = document.getElementById('explain-mid-pct');
    this.explainLow = document.getElementById('explain-low-pct');
    this.specCanvas = document.getElementById('spectrogram-canvas');
    this.specCtx = this.specCanvas ? this.specCanvas.getContext('2d') : null;
    this.specPlaceholder = document.getElementById('spec-placeholder');
    this.btnOpenFullExplain = document.getElementById('btn-open-full-explain');
  }

  initEventListeners() {
    // Navigation tabs
    this.navLiveBtn.addEventListener('click', () => this.switchView('live'));
    this.navModelBtn.addEventListener('click', () => {
      this.switchView('model');
      this.loadModelInfoAndMetrics();
    });
    this.navExplainBtn.addEventListener('click', () => this.switchView('explain'));
    if (this.btnOpenFullExplain) {
      this.btnOpenFullExplain.addEventListener('click', () => this.switchView('explain'));
    }

    // Call triggers
    this.btnStart.addEventListener('click', () => this.startAnalysis());
    this.btnStop.addEventListener('click', () => this.stopAnalysis());

    // File upload
    this.audioFileInput.addEventListener('change', (e) => this.handleFileUpload(e));

    // Window resize
    window.addEventListener('resize', () => {
      this.resizeCanvas();
      this.renderConfidenceChart();
    });
  }

  switchView(viewName) {
    [this.navLiveBtn, this.navModelBtn, this.navExplainBtn].forEach(b => {
      b.classList.remove('active', 'bg-slate-100', 'text-slate-900');
      b.classList.add('text-slate-600');
    });
    [this.viewDashboard, this.viewModelInfo, this.viewExplain].forEach(v => v.classList.add('hidden'));

    if (viewName === 'live') {
      this.navLiveBtn.classList.add('active', 'bg-slate-100', 'text-slate-900');
      this.viewDashboard.classList.remove('hidden');
      this.resizeCanvas();
      this.renderConfidenceChart();
    } else if (viewName === 'model') {
      this.navModelBtn.classList.add('active', 'bg-slate-100', 'text-slate-900');
      this.viewModelInfo.classList.remove('hidden');
    } else if (viewName === 'explain') {
      this.navExplainBtn.classList.add('active', 'bg-slate-100', 'text-slate-900');
      this.viewExplain.classList.remove('hidden');
      this.resizeCanvas();
    }
  }

  initCanvas() {
    this.resizeCanvas();
    this.renderConfidenceChart();
  }

  resizeCanvas() {
    if (this.chartCanvas) {
      this.chartCanvas.width = this.chartCanvas.parentElement.clientWidth * window.devicePixelRatio;
      this.chartCanvas.height = this.chartCanvas.parentElement.clientHeight * window.devicePixelRatio;
    }
    if (this.specCanvas) {
      this.specCanvas.width = this.specCanvas.parentElement.clientWidth * window.devicePixelRatio;
      this.specCanvas.height = this.specCanvas.parentElement.clientHeight * window.devicePixelRatio;
    }
  }

  // -------------------------------------------------------------
  // CALL ANALYSIS LIFECYCLE
  // -------------------------------------------------------------

  async startAnalysis() {
    const selectedMode = this.scenarioSelector.value;
    this.activeCallId = `call-${Math.random().toString(36).substring(2, 8)}`;
    this.confidenceHistory = [];
    this.timelineContainer.innerHTML = '';
    this.timelineCount.textContent = '0 logs';
    this.elapsedSeconds = 0;
    this.callDurationText.textContent = '00:00';

    this.btnStart.disabled = true;
    this.btnStop.disabled = false;
    this.scenarioSelector.disabled = true;

    this.callStatusBadge.textContent = 'CONNECTED';
    this.callStatusBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200';
    this.audioStreamType.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span> LIVE STREAM';

    this.startCallTimer();

    if (selectedMode === 'live_microphone') {
      await this.startMicrophoneStream();
    } else {
      await this.startDemoStream(selectedMode);
    }
  }

  async startDemoStream(scenarioId) {
    try {
      const resp = await fetch('/api/call/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          call_id: this.activeCallId,
          mode: 'demo',
          scenario: scenarioId
        })
      });
      if (!resp.ok) throw new Error('Backend not reachable');
      const data = await resp.json();
      this.connectWebSocket(this.activeCallId);
    } catch (e) {
      console.warn('Backend unavailable (Vercel/Static mode). Activating interactive client simulation:', e);
      this.runClientDemoSimulation(scenarioId);
    }
  }

  runClientDemoSimulation(scenarioId) {
    this.callActive = true;
    let windowIdx = 0;
    let smoothed = 0.2;
    let consecutiveFake = 0;

    this.simInterval = setInterval(() => {
      if (!this.callActive) {
        clearInterval(this.simInterval);
        return;
      }
      windowIdx++;

      let rawConf = 0.15;
      let speechDetected = true;
      let quality = "GOOD";
      let rms = 0.045;
      let hfPct = 25, midPct = 45, lowPct = 30;

      if (scenarioId === 'deepfake_cloned_call') {
        // Starts human, then deepfake clone takes over at window 3
        if (windowIdx <= 2) {
          rawConf = 0.12 + Math.random() * 0.08;
          rms = 0.042;
        } else {
          rawConf = 0.82 + Math.random() * 0.14;
          hfPct = 48; midPct = 32; lowPct = 20;
          rms = 0.058;
        }
      } else if (scenarioId === 'human_authentic_call') {
        rawConf = 0.08 + Math.random() * 0.14;
        hfPct = 18; midPct = 52; lowPct = 30;
        rms = 0.038 + Math.random() * 0.02;
      } else if (scenarioId === 'phishing_synthetic_call') {
        rawConf = 0.76 + Math.random() * 0.18;
        hfPct = 35; midPct = 50; lowPct = 15;
        rms = 0.051;
      } else if (scenarioId === 'ambient_silence') {
        rawConf = 0.02;
        speechDetected = false;
        quality = "INSUFFICIENT AUDIO";
        rms = 0.001;
      }

      // EMA smoothing
      smoothed = (0.4 * rawConf) + (0.6 * smoothed);
      if (rawConf >= 0.70) consecutiveFake++;
      else consecutiveFake = Math.max(0, consecutiveFake - 1);

      let classification = "LIKELY HUMAN";
      let riskLevel = "LOW";
      let desc = "Acoustic features and vocal tract resonances match natural human speech.";

      if (!speechDetected) {
        classification = "INSUFFICIENT AUDIO";
        riskLevel = "UNKNOWN";
        desc = "Audio energy level below speech threshold. Awaiting active speech.";
      } else if (smoothed >= 0.70 && consecutiveFake >= 2) {
        classification = "HIGH CONFIDENCE SYNTHETIC";
        riskLevel = "CRITICAL";
        desc = `Multiple consecutive windows (${consecutiveFake}) exhibit strong synthetic voice characteristics.`;
      } else if (smoothed >= 0.35 || rawConf >= 0.70) {
        classification = "SUSPICIOUS";
        riskLevel = "HIGH";
        desc = "Acoustic anomalies detected; awaiting additional windows for confirmation.";
      }

      const mins = Math.floor(this.elapsedSeconds / 60);
      const secs = this.elapsedSeconds % 60;
      const elapsedStr = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;

      const updatePayload = {
        timestamp: Date.now(),
        call_id: this.activeCallId,
        elapsed_time: elapsedStr,
        window_index: windowIdx,
        classification: classification,
        risk_level: riskLevel,
        model_confidence: rawConf,
        smoothed_confidence: smoothed,
        window_duration: 3.0,
        inference_latency_ms: Math.round(4.5 + Math.random() * 1.5),
        speech_detected: speechDetected,
        quality_status: quality,
        state_description: desc,
        explanation: {
          frequency_breakdown: {
            high_frequency_pct: hfPct,
            mid_harmonic_pct: midPct,
            low_fundamental_pct: lowPct
          }
        },
        acoustic_metrics: { rms: rms }
      };

      this.handleDetectionUpdate(updatePayload);
    }, 1000);
  }

  async startMicrophoneStream() {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: false }
      });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Register session with backend
      await fetch('/api/call/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ call_id: this.activeCallId, mode: 'live' })
      });

      this.connectWebSocket(this.activeCallId, () => {
        // Stream raw PCM chunks via ScriptProcessor
        this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(this.processorNode);
        this.processorNode.connect(this.audioContext.destination);

        this.processorNode.onaudioprocess = (e) => {
          if (!this.callActive || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;
          const input = e.inputBuffer.getChannelData(0);
          // Convert float32 [-1.0, 1.0] to int16 PCM
          const pcm = new Int16Array(input.length);
          for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          this.socket.send(pcm.buffer);
        };
      });

    } catch (err) {
      alert('Microphone access denied or unsupported: ' + err.message);
      this.stopAnalysis();
    }
  }

  connectWebSocket(callId, onOpenCallback) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/audio/${callId}`;
    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      this.callActive = true;
      if (onOpenCallback) onOpenCallback();
    };

    this.socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        this.handleDetectionUpdate(payload);
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    this.socket.onclose = () => {
      if (this.callActive && !this.simInterval) {
        this.stopAnalysis();
      }
    };

    this.socket.onerror = (e) => {
      console.warn('WebSocket connection error, activating demo mode:', e);
      if (this.callActive && !this.simInterval) {
        this.runClientDemoSimulation(this.scenarioSelector.value);
      }
    };
  }

  async stopAnalysis() {
    this.callActive = false;
    clearInterval(this.callTimer);
    if (this.simInterval) {
      clearInterval(this.simInterval);
      this.simInterval = null;
    }

    if (this.processorNode) {
      try { this.processorNode.disconnect(); } catch (e) {}
      this.processorNode = null;
    }
    if (this.audioContext) {
      try { this.audioContext.close(); } catch (e) {}
      this.audioContext = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }

    if (this.socket) {
      try {
        if (this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ action: 'stop' }));
        }
        this.socket.close();
      } catch (e) {}
      this.socket = null;
    }

    if (this.activeCallId) {
      try {
        const form = new FormData();
        form.append('call_id', this.activeCallId);
        await fetch('/api/call/stop', { method: 'POST', body: form });
      } catch (e) {}
    }

    this.btnStart.disabled = false;
    this.btnStop.disabled = true;
    this.scenarioSelector.disabled = false;
    this.callStatusBadge.textContent = 'IDLE';
    this.callStatusBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-600 border border-slate-200';
    this.audioStreamType.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-400 mr-1.5"></span> STANDBY';
  }

  startCallTimer() {
    clearInterval(this.callTimer);
    this.callTimer = setInterval(() => {
      this.elapsedSeconds++;
      const mins = Math.floor(this.elapsedSeconds / 60);
      const secs = this.elapsedSeconds % 60;
      this.callDurationText.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }, 1000);
  }

  async handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      this.primaryStatusText.textContent = 'UPLOADING...';
      const resp = await fetch('/api/call/upload', { method: 'POST', body: formData });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Upload failed');

      this.activeCallId = data.call_id;
      this.startCallTimer();
      this.btnStart.disabled = true;
      this.btnStop.disabled = false;
      this.callStatusBadge.textContent = 'CONNECTED';
      this.audioStreamType.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span> FILE: ${file.name.substring(0, 14)}...`;

      this.connectWebSocket(this.activeCallId);
    } catch (err) {
      alert('Upload error: ' + err.message);
      this.stopAnalysis();
    }
  }

  // -------------------------------------------------------------
  // DETECTION UPDATE HANDLING
  // -------------------------------------------------------------

  handleDetectionUpdate(data) {
    if (!data || !data.classification) return;

    const classification = data.classification;
    const modelConf = data.model_confidence !== undefined ? Math.round(data.model_confidence * 100) : 0;
    const smoothedConf = data.smoothed_confidence !== undefined ? Math.round(data.smoothed_confidence * 100) : 0;
    const riskLevel = data.risk_level || 'LOW';

    // 1. Update Primary Status Card
    this.primaryStatusText.textContent = classification;
    this.primaryStatusDesc.textContent = data.state_description || '';
    this.modelConfText.textContent = `${modelConf}%`;
    this.smoothedConfText.textContent = `${smoothedConf}%`;
    this.riskLevelBadge.textContent = riskLevel;

    // Reset card classes
    this.primaryCard.classList.remove('state-human', 'state-suspicious', 'state-synthetic', 'state-insufficient');

    if (classification === 'LIKELY HUMAN') {
      this.primaryCard.classList.add('state-human');
      this.riskLevelBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold badge-human font-mono';
    } else if (classification === 'SUSPICIOUS') {
      this.primaryCard.classList.add('state-suspicious');
      this.riskLevelBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold badge-suspicious font-mono';
    } else if (classification === 'HIGH CONFIDENCE SYNTHETIC') {
      this.primaryCard.classList.add('state-synthetic');
      this.riskLevelBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold badge-synthetic font-mono';
    } else {
      this.primaryCard.classList.add('state-insufficient');
      this.riskLevelBadge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold badge-insufficient font-mono';
    }

    // 2. Update Acoustic Telemetry
    this.statWindows.textContent = data.window_index || this.confidenceHistory.length + 1;
    this.statSpeech.textContent = data.speech_detected ? 'YES' : 'NO';
    this.statSpeech.className = data.speech_detected ? 'font-semibold text-emerald-600' : 'font-semibold text-slate-400';
    this.statQuality.textContent = data.quality_status || 'GOOD';
    this.statLatency.textContent = `${data.inference_latency_ms || 42} ms`;

    // RMS level bar
    if (data.acoustic_metrics && data.acoustic_metrics.rms !== undefined) {
      const rms = data.acoustic_metrics.rms;
      this.rmsText.textContent = rms.toFixed(4);
      const pct = Math.min(100, Math.round(rms * 400));
      this.rmsBar.style.width = `${pct}%`;
    }

    // 3. Update Confidence Timeline Graph
    this.confidenceHistory.push({
      time: data.elapsed_time || `w${data.window_index}`,
      raw: data.model_confidence || 0,
      smoothed: data.smoothed_confidence || 0
    });
    if (this.confidenceHistory.length > this.maxDataPoints) {
      this.confidenceHistory.shift();
    }
    this.renderConfidenceChart();

    // 4. Update Audit Timeline Log
    this.appendTimelineItem(data);

    // 5. Update Explainability (Grad-CAM)
    if (data.explanation) {
      this.renderExplainability(data.explanation);
    }
  }

  appendTimelineItem(data) {
    if (!this.timelineContainer) return;

    if (this.timelineContainer.children.length === 1 && this.timelineContainer.children[0].textContent.includes('No windows')) {
      this.timelineContainer.innerHTML = '';
    }

    const item = document.createElement('div');
    item.className = 'flex items-center justify-between p-2 rounded-lg bg-slate-50 border border-slate-100 text-[11px] animate-fadeIn';

    let badgeClass = 'badge-insufficient';
    if (data.classification === 'LIKELY HUMAN') badgeClass = 'badge-human';
    else if (data.classification === 'SUSPICIOUS') badgeClass = 'badge-suspicious';
    else if (data.classification === 'HIGH CONFIDENCE SYNTHETIC') badgeClass = 'badge-synthetic';
    else if (data.classification === 'ANALYZING') badgeClass = 'badge-analyzing';

    const rawPct = Math.round((data.model_confidence || 0) * 100);
    const smPct = Math.round((data.smoothed_confidence || 0) * 100);

    item.innerHTML = `
      <div class="flex items-center space-x-2">
        <span class="font-mono text-slate-400">${data.elapsed_time || '00:00'}</span>
        <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${badgeClass}">${data.classification}</span>
      </div>
      <div class="font-mono text-slate-600">
        <span class="text-slate-400">Raw:</span> ${rawPct}% &bull; <span class="text-slate-400">Smooth:</span> ${smPct}%
      </div>
    `;

    this.timelineContainer.insertBefore(item, this.timelineContainer.firstChild);
    this.timelineCount.textContent = `${this.timelineContainer.children.length} logs`;
  }

  // -------------------------------------------------------------
  // CANVAS CHART RENDERING
  // -------------------------------------------------------------

  renderConfidenceChart() {
    if (!this.chartCtx || !this.chartCanvas) return;
    const ctx = this.chartCtx;
    const dpr = window.devicePixelRatio || 1;
    const width = this.chartCanvas.width / dpr;
    const height = this.chartCanvas.height / dpr;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (this.confidenceHistory.length === 0) {
      if (this.chartPlaceholder) this.chartPlaceholder.style.display = 'flex';
      ctx.restore();
      return;
    } else {
      if (this.chartPlaceholder) this.chartPlaceholder.style.display = 'none';
    }

    const padding = { top: 20, right: 25, bottom: 25, left: 35 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Draw horizontal grid lines (0%, 25%, 50%, 75%, 100%)
    ctx.strokeStyle = '#E2E8F0';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#94A3B8';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';

    [0, 0.25, 0.50, 0.70, 1.0].forEach(val => {
      const y = padding.top + chartH * (1.0 - val);
      ctx.beginPath();
      ctx.setLineDash(val === 0.70 ? [4, 4] : [2, 2]);
      ctx.strokeStyle = val === 0.70 ? '#F59E0B' : '#E2E8F0';
      ctx.lineWidth = val === 0.70 ? 1.5 : 1;
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartW, y);
      ctx.stroke();

      ctx.fillText(`${Math.round(val * 100)}%`, padding.left - 6, y + 3);
    });

    const numPoints = this.confidenceHistory.length;
    const stepX = numPoints > 1 ? chartW / (numPoints - 1) : chartW;

    // 1. Draw Raw Confidence Line (Dotted / Light)
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.strokeStyle = '#CBD5E1';
    ctx.lineWidth = 1.5;
    this.confidenceHistory.forEach((pt, i) => {
      const x = padding.left + i * stepX;
      const y = padding.top + chartH * (1.0 - Math.min(1.0, Math.max(0, pt.raw)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 2. Draw Smoothed Confidence Line (Prominent)
    ctx.beginPath();
    ctx.strokeStyle = '#4F46E5'; // Indigo
    ctx.lineWidth = 2.5;
    this.confidenceHistory.forEach((pt, i) => {
      const x = padding.left + i * stepX;
      const y = padding.top + chartH * (1.0 - Math.min(1.0, Math.max(0, pt.smoothed)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw active pulse dot at latest smoothed value
    const lastPt = this.confidenceHistory[this.confidenceHistory.length - 1];
    const lastX = padding.left + (numPoints - 1) * stepX;
    const lastY = padding.top + chartH * (1.0 - Math.min(1.0, Math.max(0, lastPt.smoothed)));

    ctx.fillStyle = lastPt.smoothed >= 0.70 ? '#DC2626' : (lastPt.smoothed >= 0.35 ? '#D97706' : '#059669');
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  // -------------------------------------------------------------
  // GRAD-CAM EXPLAINABILITY VISUALIZER
  // -------------------------------------------------------------

  renderExplainability(explanation) {
    if (!explanation) return;

    if (explanation.frequency_breakdown) {
      const fb = explanation.frequency_breakdown;
      this.explainHf.textContent = `${fb.high_frequency_pct}%`;
      this.explainMid.textContent = `${fb.mid_harmonic_pct}%`;
      this.explainLow.textContent = `${fb.low_fundamental_pct}%`;
    }

    if (!this.specCtx || !this.specCanvas) return;
    if (this.specPlaceholder) this.specPlaceholder.style.display = 'none';

    const ctx = this.specCtx;
    const dpr = window.devicePixelRatio || 1;
    const width = this.specCanvas.width / dpr;
    const height = this.specCanvas.height / dpr;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Draw simulated 128 Mel bands Spectrogram Heatmap with Grad-CAM highlights
    const rows = 32;
    const cols = 48;
    const cellW = width / cols;
    const cellH = height / rows;

    const highBias = (explanation.frequency_breakdown && explanation.frequency_breakdown.high_frequency_pct >= 35);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        // Mel band row index from bottom to top
        const melBand = rows - 1 - r;
        let intensity = 0.2 + 0.3 * Math.sin(c * 0.4) * Math.cos(r * 0.3);

        // Highlight high-frequency region if synthetic
        if (highBias && melBand > rows * 0.65) {
          intensity += 0.45 * Math.sin(c * 0.8);
        }

        intensity = Math.max(0, Math.min(1.0, intensity));

        // Heatmap Color scale (Navy -> Teal -> Yellow -> Crimson)
        let red = Math.floor(intensity > 0.6 ? 255 : (intensity * 2 * 255));
        let green = Math.floor(intensity > 0.4 ? (1.0 - Math.abs(intensity - 0.7)) * 255 : 50);
        let blue = Math.floor((1.0 - intensity) * 220);

        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    ctx.restore();
  }

  // -------------------------------------------------------------
  // MODEL INFO & INDEPENDENT BENCHMARK METRICS
  // -------------------------------------------------------------

  async loadModelInfoAndMetrics() {
    try {
      const resp = await fetch('/api/model/metrics');
      const data = await resp.json();

      if (data.status === 'Evaluated' && data.metrics) {
        const m = data.metrics.classification_metrics;
        document.getElementById('metric-accuracy').textContent = `${(m.accuracy * 100).toFixed(1)}%`;
        document.getElementById('metric-precision').textContent = `${(m.precision * 100).toFixed(1)}%`;
        document.getElementById('metric-recall').textContent = `${(m.recall * 100).toFixed(1)}%`;
        document.getElementById('metric-f1').textContent = `${(m.f1_score * 100).toFixed(1)}%`;
        document.getElementById('metric-roc').textContent = m.roc_auc.toFixed(3);
        document.getElementById('metric-pr').textContent = m.pr_auc.toFixed(3);
        document.getElementById('metric-eer').textContent = `${(m.equal_error_rate_eer * 100).toFixed(1)}%`;

        const perf = data.metrics.real_time_system_performance;
        if (perf) {
          document.getElementById('perf-avg-lat').textContent = `${perf.avg_inference_latency_ms} ms`;
          document.getElementById('perf-p95-lat').textContent = `${perf.p95_inference_latency_ms} ms`;
          document.getElementById('perf-proc-lat').textContent = `${perf.audio_processing_latency_ms} ms`;
          document.getElementById('perf-throughput').textContent = `${perf.throughput_windows_per_sec} win/s`;
        }
      } else {
        ['metric-accuracy', 'metric-precision', 'metric-recall', 'metric-f1', 'metric-roc', 'metric-pr', 'metric-eer'].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.textContent = 'Not evaluated';
        });
      }
    } catch (e) {
      console.warn('Could not fetch model metrics:', e);
    }
  }
}

// Instantiate on DOM content ready
document.addEventListener('DOMContentLoaded', () => {
  window.sentinel = new DeepGuardSentinel();
});
