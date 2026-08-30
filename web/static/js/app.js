/**
 * DRDO Problem Statement 26052: Adaptive Noise Cancellation & PWA Controller
 * Handles PWA Field Recorder, IndexedDB persistence, Sync Engine UI,
 * Web Audio visualizer, and live model metric telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global Application State
  const AppState = {
    activeTab: 'tab-landing',
    activeModelId: 'baseline_811k',
    activeClipId: 'op_5_10_00',
    activeChannel: 'enhanced',
    isPlaying: false,
    clips: [],
    models: [],
    customAudioBuffer: null,
    deferredInstallPrompt: null,

    // Field Recorder State
    isFieldRecording: false,
    recMediaRecorder: null,
    recChunks: [],
    recStartTime: 0,
    recTimerInterval: null,
    recAudioContext: null,
    recAnalyser: null,
    recAnimFrame: null,

    // Custom Audio Mic State
    recordedChunks: [],
    mediaRecorder: null,
    isRecording: false
  };

  // Audio Elements
  const audioClean = document.getElementById('audioElementClean');
  const audioNoisy = document.getElementById('audioElementNoisy');
  const audioEnhanced = document.getElementById('audioElementEnhanced');

  // Header Elements
  const globalModelSelect = document.getElementById('globalModelSelect');
  const hwDeviceText = document.getElementById('hwDeviceText');
  const activeStreamTitle = document.getElementById('activeStreamTitle');
  const canvasChannelLabel = document.getElementById('canvasChannelLabel');
  const offlineBanner = document.getElementById('offlineBanner');
  const offlineBannerText = document.getElementById('offlineBannerText');
  const btnPwaHeaderInstall = document.getElementById('btnPwaHeaderInstall');
  const btnHeroPwaInstall = document.getElementById('btnHeroPwaInstall');

  // Transport Controls
  const btnPlayPause = document.getElementById('btnPlayPause');
  const playIcon = document.getElementById('playIcon');
  const btnStop = document.getElementById('btnStop');
  const timeDisplay = document.getElementById('timeDisplay');
  const audioScrubber = document.getElementById('audioScrubber');
  const audioVolume = document.getElementById('audioVolume');
  const btnReEnhanceActive = document.getElementById('btnReEnhanceActive');

  // Channel Switcher Buttons
  const btnChannelClean = document.getElementById('btnChannelClean');
  const btnChannelNoisy = document.getElementById('btnChannelNoisy');
  const btnChannelEnhanced = document.getElementById('btnChannelEnhanced');

  // Metric Display Tiles
  const gaugeValOutSnr = document.getElementById('gaugeValOutSnr');
  const gaugeValInSnr = document.getElementById('gaugeValInSnr');
  const badgeSnr = document.getElementById('badgeSnr');
  const gaugeValGain = document.getElementById('gaugeValGain');
  const gaugeValStoi = document.getElementById('gaugeValStoi');
  const gaugeValInStoi = document.getElementById('gaugeValInStoi');
  const badgeStoi = document.getElementById('badgeStoi');
  const gaugeValLatency = document.getElementById('gaugeValLatency');
  const gaugeValRtf = document.getElementById('gaugeValRtf');
  const telemetryModelSubtitle = document.getElementById('telemetryModelSubtitle');

  // Field Recorder DOM Elements
  const btnRecordGiant = document.getElementById('btnRecordGiant');
  const recTimerDisplay = document.getElementById('recTimerDisplay');
  const liveMeterCanvas = document.getElementById('liveMeterCanvas');
  const micDeviceSelect = document.getElementById('micDeviceSelect');
  const recStatusText = document.getElementById('recStatusText');
  const recordingActiveBanner = document.getElementById('recordingActiveBanner');

  // Library & Sync DOM Elements
  const libraryTableBody = document.getElementById('libraryTableBody');
  const libraryCount = document.getElementById('libraryCount');
  const syncQueueBadge = document.getElementById('syncQueueBadge');
  const syncQueueDepthVal = document.getElementById('syncQueueDepthVal');
  const syncLastTimeVal = document.getElementById('syncLastTimeVal');
  const btnSyncNow = document.getElementById('btnSyncNow');
  const syncEngineStatusText = document.getElementById('syncEngineStatusText');
  const syncLogTableBody = document.getElementById('syncLogTableBody');

  // Settings & Modals DOM Elements
  const inputEndpointUrl = document.getElementById('inputEndpointUrl');
  const btnSaveEndpoint = document.getElementById('btnSaveEndpoint');
  const storageStatusText = document.getElementById('storageStatusText');
  const storageProgressBar = document.getElementById('storageProgressBar');
  const btnClearLocalData = document.getElementById('btnClearLocalData');
  const micPermissionModal = document.getElementById('micPermissionModal');
  const btnGrantMicPerm = document.getElementById('btnGrantMicPerm');
  const btnCancelMicPerm = document.getElementById('btnCancelMicPerm');
  const clearDataModal = document.getElementById('clearDataModal');
  const btnConfirmClearData = document.getElementById('btnConfirmClearData');
  const btnCancelClearData = document.getElementById('btnCancelClearData');

  // Waveform Visualizer
  const visualizer = new AudioVisualizer('oscilloscopeCanvas');
  visualizer.clearCanvas();

  // =========================================================================
  // 1. Navigation Tabs Switching
  // =========================================================================
  const navTabs = document.querySelectorAll('.tab-button');
  const tabPanels = document.querySelectorAll('.tab-panel');

  window.switchTab = function(targetTab) {
    AppState.activeTab = targetTab;

    navTabs.forEach(t => {
      if (t.dataset.tab === targetTab) t.classList.add('active');
      else t.classList.remove('active');
    });

    tabPanels.forEach(p => {
      if (p.id === targetTab) p.classList.add('active');
      else p.classList.remove('active');
    });

    if (targetTab === 'tab-library') renderLibraryUI();
    if (targetTab === 'tab-sync') updateSyncPanelUI();
    if (targetTab === 'tab-settings') updateStorageMonitorUI();
  };

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      window.switchTab(tab.dataset.tab);
    });
  });

  // =========================================================================
  // 2. Network Status Monitoring
  // =========================================================================
  function updateNetworkStatus() {
    if (navigator.onLine) {
      offlineBanner.classList.add('is-online');
      offlineBanner.classList.remove('hidden');
      offlineBannerText.textContent = '⚡ Network Online — Auto-sync active';
    } else {
      offlineBanner.classList.remove('is-online');
      offlineBanner.classList.remove('hidden');
      offlineBannerText.textContent = '📡 Offline Mode — All recordings saved locally to IndexedDB';
    }
  }

  window.addEventListener('online', updateNetworkStatus);
  window.addEventListener('offline', updateNetworkStatus);
  updateNetworkStatus();

  // =========================================================================
  // 3. PWA Installation Prompt
  // =========================================================================
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    AppState.deferredInstallPrompt = e;
    btnPwaHeaderInstall.classList.remove('hidden');
    btnHeroPwaInstall.classList.remove('hidden');
  });

  async function triggerPwaInstall() {
    if (!AppState.deferredInstallPrompt) return;
    AppState.deferredInstallPrompt.prompt();
    const { outcome } = await AppState.deferredInstallPrompt.userChoice;
    console.log('[PWA] User install choice outcome:', outcome);
    AppState.deferredInstallPrompt = null;
    btnPwaHeaderInstall.classList.add('hidden');
    btnHeroPwaInstall.classList.add('hidden');
  }

  btnPwaHeaderInstall.addEventListener('click', triggerPwaInstall);
  btnHeroPwaInstall.addEventListener('click', triggerPwaInstall);

  // =========================================================================
  // 4. Models Initialization
  // =========================================================================
  async function initModels() {
    try {
      const res = await fetch('/api/models');
      const models = await res.json();
      AppState.models = models;

      globalModelSelect.innerHTML = '';
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = `${m.name} (${(m.parameters / 1000).toFixed(0)}K params)`;
        if (m.recommended) {
          opt.textContent += ' (Recommended)';
          opt.selected = true;
          AppState.activeModelId = m.id;
        }
        globalModelSelect.appendChild(opt);
      });
    } catch (e) {
      console.warn('[Server] Model catalog API offline:', e);
      hwDeviceText.textContent = 'Offline Mode';
    }
  }

  globalModelSelect.addEventListener('change', (e) => {
    AppState.activeModelId = e.target.value;
    const m = AppState.models.find(mod => mod.id === AppState.activeModelId);
    if (m) {
      telemetryModelSubtitle.textContent = `${m.name} (${(m.parameters / 1000).toFixed(0)}K parameters)`;
    }
  });

  // =========================================================================
  // 5. Field Recorder Logic (Offline-First)
  // =========================================================================
  async function initMicDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioMics = devices.filter(d => d.kind === 'audioinput');

      micDeviceSelect.innerHTML = '';
      audioMics.forEach((mic, index) => {
        const opt = document.createElement('option');
        opt.value = mic.deviceId;
        opt.textContent = mic.label || `Microphone ${index + 1}`;
        micDeviceSelect.appendChild(opt);
      });
    } catch (err) {
      console.warn('[Mic] Device enumeration notice:', err);
    }
  }
  initMicDevices();

  function startLiveMeterVisualizer(stream) {
    try {
      AppState.recAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = AppState.recAudioContext.createMediaStreamSource(stream);
      AppState.recAnalyser = AppState.recAudioContext.createAnalyser();
      AppState.recAnalyser.fftSize = 256;
      source.connect(AppState.recAnalyser);

      const ctx = liveMeterCanvas.getContext('2d');
      const dataArray = new Uint8Array(AppState.recAnalyser.frequencyBinCount);

      function drawMeter() {
        if (!AppState.isFieldRecording) return;
        AppState.recAnimFrame = requestAnimationFrame(drawMeter);

        AppState.recAnalyser.getByteFrequencyData(dataArray);
        ctx.clearRect(0, 0, liveMeterCanvas.width, liveMeterCanvas.height);

        const barWidth = (liveMeterCanvas.width / dataArray.length) * 2.5;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
          const barHeight = (dataArray[i] / 255) * liveMeterCanvas.height;
          ctx.fillStyle = '#5b9dff';
          ctx.fillRect(x, liveMeterCanvas.height - barHeight, barWidth, barHeight);
          x += barWidth + 1;
        }
      }
      drawMeter();
    } catch (e) {
      console.warn('[Visualizer] Live meter setup error:', e);
    }
  }

  function stopLiveMeterVisualizer() {
    if (AppState.recAnimFrame) cancelAnimationFrame(AppState.recAnimFrame);
    if (AppState.recAudioContext) {
      AppState.recAudioContext.close().catch(() => {});
    }
    const ctx = liveMeterCanvas.getContext('2d');
    ctx.clearRect(0, 0, liveMeterCanvas.width, liveMeterCanvas.height);
  }

  async function startFieldRecording() {
    try {
      const deviceId = micDeviceSelect.value;
      const constraints = {
        audio: deviceId ? { deviceId: { exact: deviceId } } : true
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      initMicDevices(); // Refresh labels now that permission granted

      AppState.recChunks = [];
      AppState.recMediaRecorder = new MediaRecorder(stream);

      AppState.recMediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) AppState.recChunks.push(e.data);
      };

      AppState.recMediaRecorder.onstop = async () => {
        const durationSec = (Date.now() - AppState.recStartTime) / 1000;
        const mimeType = AppState.recMediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(AppState.recChunks, { type: mimeType });

        stream.getTracks().forEach(t => t.stop());
        stopLiveMeterVisualizer();

        recStatusText.textContent = 'Saving recording to local IndexedDB...';

        try {
          // Save recording locally to IndexedDB
          const record = await window.ancDB.saveRecording({
            id: self.crypto.randomUUID(),
            blob: blob,
            mimeType: mimeType,
            durationSec: durationSec,
            sampleRate: 16000,
            createdAt: new Date().toISOString(),
            syncState: 'local'
          });

          recStatusText.textContent = `Recording saved locally (${(blob.size / 1024).toFixed(1)} KB). Queueing sync...`;

          // Trigger sync engine
          window.syncEngine.triggerSync('field_recording_finished');

          // Refresh UI
          renderLibraryUI();
          updateSyncPanelUI();
          updateStorageMonitorUI();

        } catch (err) {
          alert('IndexedDB Storage Error: ' + err.message);
          recStatusText.textContent = 'Failed to save recording locally!';
        }
      };

      AppState.recMediaRecorder.start(250);
      AppState.isFieldRecording = true;
      AppState.recStartTime = Date.now();

      // UI state updates
      btnRecordGiant.classList.add('is-recording');
      recordingActiveBanner.classList.remove('hidden');
      recStatusText.textContent = 'Recording live microphone audio...';

      // Start Timer
      AppState.recTimerInterval = setInterval(() => {
        const elapsedMs = Date.now() - AppState.recStartTime;
        const totalSec = Math.floor(elapsedMs / 1000);
        const mins = String(Math.floor(totalSec / 60)).padStart(2, '0');
        const secs = String(totalSec % 60).padStart(2, '0');
        const ms = String(Math.floor((elapsedMs % 1000) / 100));
        recTimerDisplay.textContent = `${mins}:${secs}.${ms}`;
      }, 100);

      startLiveMeterVisualizer(stream);

    } catch (e) {
      console.error('[Mic] Permission or capture error:', e);
      micPermissionModal.classList.remove('hidden');
    }
  }

  function stopFieldRecording() {
    if (AppState.recMediaRecorder && AppState.isFieldRecording) {
      AppState.recMediaRecorder.stop();
      AppState.isFieldRecording = false;

      clearInterval(AppState.recTimerInterval);
      btnRecordGiant.classList.remove('is-recording');
      recordingActiveBanner.classList.add('hidden');
      recTimerDisplay.textContent = '00:00.0';
    }
  }

  btnRecordGiant.addEventListener('click', () => {
    if (!AppState.isFieldRecording) {
      startFieldRecording();
    } else {
      stopFieldRecording();
    }
  });

  btnGrantMicPerm.addEventListener('click', () => {
    micPermissionModal.classList.add('hidden');
    startFieldRecording();
  });

  btnCancelMicPerm.addEventListener('click', () => {
    micPermissionModal.classList.add('hidden');
  });

  // =========================================================================
  // 6. Library Component Logic
  // =========================================================================
  async function renderLibraryUI() {
    try {
      const items = await window.ancDB.getAllRecordings();
      libraryCount.textContent = items.length;

      if (items.length === 0) {
        libraryTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No field recordings stored in IndexedDB.</td></tr>`;
        return;
      }

      libraryTableBody.innerHTML = '';
      items.forEach(rec => {
        const tr = document.createElement('tr');

        const dateStr = new Date(rec.createdAt).toLocaleString();
        const durStr = `${rec.durationSec.toFixed(1)}s`;
        const formatStr = rec.mimeType.includes('webm') ? 'WEBM 16kHz' : 'WAV 16kHz';

        tr.innerHTML = `
          <td>
            <span class="chip-sync ${rec.syncState}">
              ${rec.syncState}
            </span>
          </td>
          <td>${dateStr}</td>
          <td>${durStr}</td>
          <td>${formatStr}</td>
          <td>
            <div class="inline-actions">
              <button class="btn-secondary btn-sm btn-play-rec" data-id="${rec.id}">▶ Play</button>
              <button class="btn-secondary btn-sm btn-delete-rec" data-id="${rec.id}" style="color: var(--color-danger);">Delete</button>
              <button class="btn-primary btn-sm btn-enhance-rec" data-id="${rec.id}">
                Enhance <span class="badge-online-required">Online Required</span>
              </button>
            </div>
          </td>
        `;
        libraryTableBody.appendChild(tr);
      });

      // Play local recording handler
      document.querySelectorAll('.btn-play-rec').forEach(btn => {
        btn.addEventListener('click', async () => {
          const rec = await window.ancDB.getRecording(btn.dataset.id);
          if (rec && rec.blob) {
            stopAudio();
            const url = URL.createObjectURL(rec.blob);
            audioEnhanced.src = url;
            audioNoisy.src = url;
            audioClean.src = url;
            activeStreamTitle.textContent = `Field Recording (${rec.id.slice(0, 8)})`;
            setChannel('enhanced');
            playAudio();
          }
        });
      });

      // Delete local recording handler
      document.querySelectorAll('.btn-delete-rec').forEach(btn => {
        btn.addEventListener('click', async () => {
          await window.ancDB.deleteRecording(btn.dataset.id);
          renderLibraryUI();
          updateSyncPanelUI();
          updateStorageMonitorUI();
        });
      });

      // Optional Enhancement Processing Handler (Online Connection Explicit)
      document.querySelectorAll('.btn-enhance-rec').forEach(btn => {
        btn.addEventListener('click', async () => {
          if (!navigator.onLine) {
            alert('⚠️ Connectivity Required: Running the ANC neural network model requires a connection to the server backend.');
            return;
          }

          const rec = await window.ancDB.getRecording(btn.dataset.id);
          if (!rec || !rec.blob) return;

          btn.disabled = true;
          btn.textContent = 'Processing...';

          try {
            const b64 = await window.syncEngine.blobToBase64(rec.blob);
            const res = await fetch('/api/upload_and_process', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                audio_data: b64,
                model_id: AppState.activeModelId
              })
            });

            const data = await res.json();
            if (data.enhanced_audio_b64) {
              stopAudio();
              audioNoisy.src = b64;
              audioEnhanced.src = data.enhanced_audio_b64;
              activeStreamTitle.textContent = `Enhanced Field Recording (${rec.id.slice(0, 8)})`;
              if (data.metrics) updateTelemetryUI(data.metrics);
              setChannel('enhanced');
              playAudio();
            }
          } catch (err) {
            alert('Enhancement processing error: ' + err.message);
          } finally {
            btn.disabled = false;
            btn.innerHTML = `Enhance <span class="badge-online-required">Online Required</span>`;
          }
        });
      });

    } catch (e) {
      console.error('[Library] Render error:', e);
    }
  }

  // =========================================================================
  // 7. Sync Panel Logic
  // =========================================================================
  async function updateSyncPanelUI() {
    try {
      const items = await window.ancDB.getAllRecordings();
      const unsynced = items.filter(i => i.syncState !== 'synced');

      syncQueueBadge.textContent = unsynced.length;
      syncQueueDepthVal.textContent = unsynced.length;

      const lastTime = await window.ancDB.getMeta('lastSyncTime', null);
      syncLastTimeVal.textContent = lastTime ? new Date(lastTime).toLocaleTimeString() : 'Never';

      syncEngineStatusText.textContent = window.syncEngine.isSyncing ? 'Syncing queue to server...' : (navigator.onLine ? 'Idle (Online)' : 'Offline');

      if (items.length === 0) {
        syncLogTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No items in queue log.</td></tr>`;
        return;
      }

      syncLogTableBody.innerHTML = '';
      items.forEach(rec => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><code>${rec.id.slice(0, 13)}...</code></td>
          <td><span class="chip-sync ${rec.syncState}">${rec.syncState}</span></td>
          <td>${rec.attempts || 0}</td>
          <td style="color: var(--text-muted); font-size: 12px;">${rec.lastError || 'None'}</td>
        `;
        syncLogTableBody.appendChild(tr);
      });
    } catch (e) {
      console.error('[SyncUI] Update error:', e);
    }
  }

  btnSyncNow.addEventListener('click', () => {
    window.syncEngine.triggerSync('manual_button');
  });

  window.syncEngine.subscribe(() => {
    updateSyncPanelUI();
    renderLibraryUI();
  });

  // =========================================================================
  // 8. Settings & Storage Monitor Logic
  // =========================================================================
  async function initSettings() {
    const savedEndpoint = await window.ancDB.getMeta('endpointUrl', '/api/recordings');
    inputEndpointUrl.value = savedEndpoint;
  }

  btnSaveEndpoint.addEventListener('click', async () => {
    await window.ancDB.setMeta('endpointUrl', inputEndpointUrl.value.trim());
    alert('Sync Endpoint saved successfully!');
  });

  async function updateStorageMonitorUI() {
    const quota = await window.ancDB.checkStorageQuota();
    storageStatusText.textContent = `Storage Used: ${quota.usageMB} MB / ${quota.quotaMB} MB (${quota.percentUsed}% used, ${quota.freeMB} MB free)`;

    storageProgressBar.style.width = `${Math.max(2, quota.percentUsed)}%`;
    storageProgressBar.className = 'storage-bar-inner';
    if (quota.percentUsed > 70) storageProgressBar.classList.add('warn');
    if (quota.percentUsed > 90) storageProgressBar.classList.add('danger');
  }

  btnClearLocalData.addEventListener('click', () => {
    clearDataModal.classList.remove('hidden');
  });

  btnCancelClearData.addEventListener('click', () => {
    clearDataModal.classList.add('hidden');
  });

  btnConfirmClearData.addEventListener('click', async () => {
    await window.ancDB.clearAllRecordings();
    clearDataModal.classList.add('hidden');
    renderLibraryUI();
    updateSyncPanelUI();
    updateStorageMonitorUI();
    alert('All local IndexedDB recordings purged.');
  });

  // =========================================================================
  // 9. Audio Playback & Scrubber Controls (Preserved)
  // =========================================================================
  function setChannel(ch) {
    AppState.activeChannel = ch;
    [btnChannelClean, btnChannelNoisy, btnChannelEnhanced].forEach(b => b.classList.remove('active'));

    if (ch === 'clean') {
      btnChannelClean.classList.add('active');
      canvasChannelLabel.textContent = 'Clean Speech';
      canvasChannelLabel.className = 'channel-indicator-text tag-clean';
    } else if (ch === 'noisy') {
      btnChannelNoisy.classList.add('active');
      canvasChannelLabel.textContent = 'Noisy Input';
      canvasChannelLabel.className = 'channel-indicator-text tag-noisy';
    } else {
      btnChannelEnhanced.classList.add('active');
      canvasChannelLabel.textContent = 'Cleaned up';
      canvasChannelLabel.className = 'channel-indicator-text tag-enhanced';
    }
    syncPlaybackChannels();
  }

  btnChannelClean.addEventListener('click', () => setChannel('clean'));
  btnChannelNoisy.addEventListener('click', () => setChannel('noisy'));
  btnChannelEnhanced.addEventListener('click', () => setChannel('enhanced'));

  function getActiveAudioElement() {
    if (AppState.activeChannel === 'clean') return audioClean;
    if (AppState.activeChannel === 'noisy') return audioNoisy;
    return audioEnhanced;
  }

  function syncPlaybackChannels() {
    const active = getActiveAudioElement();
    const currTime = active.currentTime || 0;
    const vol = parseFloat(audioVolume.value);

    [audioClean, audioNoisy, audioEnhanced].forEach(a => {
      a.volume = 0;
      try { a.currentTime = currTime; } catch (e) {}
    });
    active.volume = vol;
  }

  function playAudio() {
    syncPlaybackChannels();
    const active = getActiveAudioElement();
    active.play().then(() => {
      AppState.isPlaying = true;
      playIcon.textContent = '⏸';
      visualizer.connectAudioElement(active);
      visualizer.start();
    }).catch(e => console.warn('Play error:', e));
  }

  function pauseAudio() {
    [audioClean, audioNoisy, audioEnhanced].forEach(a => a.pause());
    AppState.isPlaying = false;
    playIcon.textContent = '▶';
    visualizer.stop();
  }

  function stopAudio() {
    pauseAudio();
    [audioClean, audioNoisy, audioEnhanced].forEach(a => {
      try { a.currentTime = 0; } catch (e) {}
    });
    audioScrubber.value = 0;
    timeDisplay.textContent = '0:00 / 0:00';
  }

  btnPlayPause.addEventListener('click', () => {
    if (AppState.isPlaying) pauseAudio();
    else playAudio();
  });

  btnStop.addEventListener('click', stopAudio);

  audioVolume.addEventListener('input', (e) => {
    getActiveAudioElement().volume = parseFloat(e.target.value);
  });

  audioScrubber.addEventListener('input', (e) => {
    const active = getActiveAudioElement();
    if (active.duration) {
      const targetTime = (parseFloat(e.target.value) / 100) * active.duration;
      [audioClean, audioNoisy, audioEnhanced].forEach(a => {
        try { a.currentTime = targetTime; } catch (err) {}
      });
    }
  });

  getActiveAudioElement().addEventListener('timeupdate', () => {
    const active = getActiveAudioElement();
    if (active.duration) {
      const pct = (active.currentTime / active.duration) * 100;
      audioScrubber.value = pct;

      const cMin = Math.floor(active.currentTime / 60);
      const cSec = String(Math.floor(active.currentTime % 60)).padStart(2, '0');
      const dMin = Math.floor(active.duration / 60);
      const dSec = String(Math.floor(active.duration % 60)).padStart(2, '0');
      timeDisplay.textContent = `${cMin}:${cSec} / ${dMin}:${dSec}`;
    }
  });

  // =========================================================================
  // 10. Test Clips & Scorecard Loaders (Preserved)
  // =========================================================================
  async function initClipsCatalog() {
    try {
      const res = await fetch('/api/clips');
      const clips = await res.json();
      AppState.clips = clips;
      renderClipsTable(clips);
      if (clips.length > 0) loadClip(clips[0].id);
    } catch (e) {
      console.warn('[Server] Listening clips catalog unavailable offline:', e);
    }
  }

  function renderClipsTable(clips) {
    const tbody = document.getElementById('clipsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    clips.forEach(c => {
      const tr = document.createElement('tr');
      tr.dataset.id = c.id;
      tr.dataset.bucket = c.bucket;
      if (c.id === AppState.activeClipId) tr.classList.add('selected');

      const gain = c.out_snr_db - c.measured_in_snr_db;
      tr.innerHTML = `
        <td><span class="status-dot dot-ready"></span></td>
        <td><code>${c.id}</code></td>
        <td>${c.bucket}</td>
        <td>${c.noise_source}</td>
        <td>${c.speech_source}</td>
        <td>${c.measured_in_snr_db.toFixed(1)} dB</td>
        <td class="font-mono text-primary">${c.out_snr_db.toFixed(1)} dB</td>
        <td class="font-mono text-success">+${gain.toFixed(1)} dB</td>
        <td><button class="btn-secondary btn-sm" onclick="event.stopPropagation(); loadClip('${c.id}')">Load</button></td>
      `;
      tr.addEventListener('click', () => loadClip(c.id));
      tbody.appendChild(tr);
    });
  }

  window.loadClip = async function(clipId) {
    AppState.activeClipId = clipId;
    document.querySelectorAll('#clipsTableBody tr').forEach(r => {
      r.classList.toggle('selected', r.dataset.id === clipId);
    });

    audioClean.src = `/api/audio/clip?id=${clipId}&type=clean`;
    audioNoisy.src = `/api/audio/clip?id=${clipId}&type=noisy`;
    
    // Process clip with backend
    try {
      const res = await fetch('/api/process_clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clip_id: clipId,
          model_id: AppState.activeModelId
        })
      });
      const data = await res.json();
      if (data.enhanced_audio_b64) {
        audioEnhanced.src = data.enhanced_audio_b64;
        if (data.metrics) updateTelemetryUI(data.metrics);
      }
    } catch (e) {
      console.warn('[Processing] Clip processing offline:', e);
    }

    activeStreamTitle.textContent = `Clip ${clipId}`;
    setChannel('enhanced');
  };

  async function initDRDOScorecard() {
    try {
      const res = await fetch('/api/drdo_scorecard');
      const scorecard = await res.json();
      const container = document.getElementById('scorecardTargetCards');
      if (!container) return;
      container.innerHTML = '';

      scorecard.targets.forEach(t => {
        const isPass = t.status === 'PASSED' || t.status === 'PASSED_PROXY';
        const cardClass = isPass ? 'item-pass' : 'item-short';
        const valClass = isPass ? 'pass' : 'short';

        const card = document.createElement('div');
        card.className = `scorecard-item ${cardClass}`;
        card.innerHTML = `
          <div class="scorecard-item-header">
            <span class="scorecard-metric-title">${t.metric}</span>
            <span class="scorecard-target-text">Target: ${t.drdo_target}</span>
          </div>
          <div class="scorecard-val ${valClass}">${t.achieved}</div>
          <div class="scorecard-comment">${t.verdict}</div>
        `;
        container.appendChild(card);
      });
    } catch (e) {
      console.warn('[Server] Scorecard API offline:', e);
    }
  }

  function updateTelemetryUI(metrics) {
    if (!metrics) return;
    if (metrics.output_snr !== undefined) {
      gaugeValOutSnr.textContent = metrics.output_snr.toFixed(1) + ' dB';
      gaugeValInSnr.textContent = `${metrics.input_snr.toFixed(1)} dB`;
      badgeSnr.className = `metric-badge ${metrics.drdo_snr_pass ? 'badge-met' : 'badge-unmet'}`;
      badgeSnr.textContent = metrics.drdo_snr_pass ? 'PASSED' : 'SHORT (>15dB)';
    }
    if (metrics.snr_gain !== undefined) {
      gaugeValGain.textContent = (metrics.snr_gain >= 0 ? `+${metrics.snr_gain.toFixed(1)}` : metrics.snr_gain.toFixed(1)) + ' dB';
    }
    if (metrics.output_stoi !== undefined) {
      gaugeValStoi.textContent = metrics.output_stoi.toFixed(3);
      if (metrics.input_stoi !== undefined) {
        gaugeValInStoi.textContent = metrics.input_stoi.toFixed(3);
      }
      badgeStoi.className = `metric-badge ${metrics.drdo_stoi_pass ? 'badge-met' : 'badge-unmet'}`;
      badgeStoi.textContent = metrics.drdo_stoi_pass ? 'PASSED' : 'BELOW SPEC';
    }
    if (metrics.latency_ms !== undefined) {
      gaugeValLatency.textContent = metrics.latency_ms.toFixed(3) + ' ms';
    }
    if (metrics.rtf !== undefined) {
      gaugeValRtf.textContent = metrics.rtf.toFixed(4);
    }
  }

  // =========================================================================
  // 11. 24/7 Voice Logger Operator Console Logic
  // =========================================================================
  const voiceloggerTableBody = document.getElementById('voiceloggerTableBody');
  const auditLogTableBody = document.getElementById('auditLogTableBody');
  const chainStatusValue = document.getElementById('chainStatusValue');
  const chainTotalEntriesVal = document.getElementById('chainTotalEntriesVal');
  const loggerAlertBadge = document.getElementById('loggerAlertBadge');
  const loggerSosBanner = document.getElementById('loggerSosBanner');
  const btnVerifyHashChain = document.getElementById('btnVerifyHashChain');
  const btnRefreshLoggerData = document.getElementById('btnRefreshLoggerData');

  async function renderVoiceLoggerDashboard() {
    try {
      const res = await fetch('/api/recordings');
      const recs = await res.json();
      if (chainTotalEntriesVal) chainTotalEntriesVal.textContent = recs.length;

      if (recs.length === 0) {
        if (voiceloggerTableBody) voiceloggerTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No logbook entries synced yet.</td></tr>`;
      } else if (voiceloggerTableBody) {
        voiceloggerTableBody.innerHTML = '';
        recs.forEach(rec => {
          const tr = document.createElement('tr');
          const dateStr = new Date(rec.createdAt).toLocaleString();
          const classif = rec.classification || 'NORMAL';
          const chipClass = classif === 'SOS' ? 'failed' : (classif === 'PROTECTED' ? 'queued' : 'synced');

          tr.innerHTML = `
            <td><span class="chip-sync ${chipClass}">${classif}</span></td>
            <td>${dateStr}</td>
            <td>${rec.durationSec.toFixed(1)}s</td>
            <td>${(rec.vadSpeechSec || rec.durationSec).toFixed(1)}s</td>
            <td><code>${(rec.sha256 || 'N/A').slice(0, 12)}...</code></td>
            <td style="color: var(--color-primary); font-family: var(--font-mono);">${rec.levelReductionDb !== null && rec.levelReductionDb !== undefined ? (rec.levelReductionDb >= 0 ? '+' : '') + rec.levelReductionDb.toFixed(2) + ' dB' : 'Not Measured'}</td>
            <td>
              <button class="btn-secondary btn-sm btn-play-server-rec" data-id="${rec.id}">▶ Play</button>
            </td>
          `;
          voiceloggerTableBody.appendChild(tr);
        });

        document.querySelectorAll('.btn-play-server-rec').forEach(btn => {
          btn.addEventListener('click', () => {
            const url = `/api/recordings/${btn.dataset.id}/audio`;
            stopAudio();
            audioEnhanced.src = url;
            audioNoisy.src = url;
            activeStreamTitle.textContent = `Server Logbook Entry (${btn.dataset.id.slice(0, 8)})`;
            setChannel('enhanced');
            playAudio();
          });
        });
      }

      fetchAlertsAndAuditLogs();
      verifyHashChain();
    } catch (e) {
      console.warn('[Logger] Server recordings fetch offline:', e);
    }
  }

  async function fetchAlertsAndAuditLogs() {
    try {
      const resAlerts = await fetch('/api/alerts');
      const alerts = await resAlerts.json();
      if (loggerAlertBadge) loggerAlertBadge.textContent = alerts.length;

      if (alerts.length > 0 && loggerSosBanner) {
        loggerSosBanner.classList.remove('hidden');
      } else if (loggerSosBanner) {
        loggerSosBanner.classList.add('hidden');
      }

      const resAudit = await fetch('/api/audit_logs');
      const logs = await resAudit.json();

      if (auditLogTableBody) {
        if (logs.length === 0) {
          auditLogTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No audit log events recorded.</td></tr>`;
        } else {
          auditLogTableBody.innerHTML = '';
          logs.forEach(l => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><strong style="color: var(--color-primary);">${l.eventType}</strong></td>
              <td>${new Date(l.timestamp).toLocaleString()}</td>
              <td>${l.details}</td>
              <td><code>${(l.sha256 || 'N/A').slice(0, 10)}...</code></td>
            `;
            auditLogTableBody.appendChild(tr);
          });
        }
      }
    } catch (e) {
      console.warn('[Logger] Alerts/Audit logs offline:', e);
    }
  }

  async function verifyHashChain() {
    try {
      const res = await fetch('/api/hash_chain/verify');
      const status = await res.json();

      if (chainStatusValue) {
        if (status.valid) {
          chainStatusValue.textContent = 'VALID';
          chainStatusValue.style.color = 'var(--color-success)';
        } else {
          chainStatusValue.textContent = `TAMPERED (${status.brokenAtId ? status.brokenAtId.slice(0, 8) : 'Error'})`;
          chainStatusValue.style.color = 'var(--color-danger)';
        }
      }
    } catch (e) {
      if (chainStatusValue) {
        chainStatusValue.textContent = 'UNVERIFIED';
        chainStatusValue.style.color = 'var(--color-warn)';
      }
    }
  }

  if (btnVerifyHashChain) btnVerifyHashChain.addEventListener('click', verifyHashChain);
  if (btnRefreshLoggerData) btnRefreshLoggerData.addEventListener('click', renderVoiceLoggerDashboard);

  // Initial Load
  initModels();
  initClipsCatalog();
  initDRDOScorecard();
  initSettings();
  renderLibraryUI();
  updateSyncPanelUI();
  updateStorageMonitorUI();
  renderVoiceLoggerDashboard();
  window.syncEngine.init();
});
