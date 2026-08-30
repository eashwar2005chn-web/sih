/**
 * DRDO PS 26052: Real-Time WebAudio Spectrum & Oscilloscope Visualizer
 * High-frame-rate time-domain waveforms and FFT frequency spectrums on Glass UI Canvas.
 */

class AudioVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.audioCtx = null;
    this.analyser = null;
    this.sourceNodes = {};
    this.activeChannel = 'enhanced';
    this.dataArray = null;
    this.timeDomainArray = null;
    this.bufferLength = 0;
    this.animationId = null;
    this.isRunning = false;
    
    this.initCanvasSize();
    window.addEventListener('resize', () => this.initCanvasSize());
  }

  initCanvasSize() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * (window.devicePixelRatio || 1);
    this.canvas.height = rect.height * (window.devicePixelRatio || 1);
  }

  ensureAudioContext() {
    if (!this.audioCtx) {
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtxClass();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 512;
      this.analyser.smoothingTimeConstant = 0.8;
      this.bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(this.bufferLength);
      this.timeDomainArray = new Uint8Array(this.bufferLength);

      // Connect HTML5 audio elements
      ['clean', 'noisy', 'enhanced'].forEach(channel => {
        const audioEl = document.getElementById(`audioElement${channel.charAt(0).toUpperCase() + channel.slice(1)}`);
        if (audioEl) {
          try {
            const source = this.audioCtx.createMediaElementSource(audioEl);
            source.connect(this.analyser);
            this.analyser.connect(this.audioCtx.destination);
            this.sourceNodes[channel] = source;
          } catch (e) {
            console.warn(`[Visualizer] Source init for ${channel}:`, e);
          }
        }
      });
    }

    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  setChannel(channelName) {
    this.activeChannel = channelName;
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.renderLoop();
  }

  stop() {
    this.isRunning = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    this.clearCanvas();
  }

  clearCanvas() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.fillStyle = '#0f172a';
    this.ctx.fillRect(0, 0, w, h);
    
    // Draw subtle grid line
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();
    this.ctx.moveTo(0, h / 2);
    this.ctx.lineTo(w, h / 2);
    this.ctx.stroke();
  }

  renderLoop() {
    if (!this.isRunning) return;
    this.animationId = requestAnimationFrame(() => this.renderLoop());

    if (!this.analyser) {
      this.clearCanvas();
      return;
    }

    this.analyser.getByteFrequencyData(this.dataArray);
    this.analyser.getByteTimeDomainData(this.timeDomainArray);

    const w = this.canvas.width;
    const h = this.canvas.height;

    // Dark slate background with trail
    this.ctx.fillStyle = 'rgba(15, 23, 42, 0.35)';
    this.ctx.fillRect(0, 0, w, h);

    // Color scheme by active channel
    let primaryColor = '#3b82f6';
    let secondaryColor = 'rgba(59, 130, 246, 0.18)';
    if (this.activeChannel === 'clean') {
      primaryColor = '#10b981';
      secondaryColor = 'rgba(16, 185, 129, 0.18)';
    } else if (this.activeChannel === 'noisy') {
      primaryColor = '#f43f5e';
      secondaryColor = 'rgba(244, 63, 94, 0.18)';
    }

    // 1. Draw FFT Frequency Spectrum (Background Bars)
    const barWidth = (w / this.bufferLength) * 2.2;
    let x = 0;

    for (let i = 0; i < this.bufferLength; i++) {
      const barHeight = (this.dataArray[i] / 255) * (h * 0.75);

      const grad = this.ctx.createLinearGradient(0, h, 0, h - barHeight);
      grad.addColorStop(0, secondaryColor);
      grad.addColorStop(1, primaryColor);

      this.ctx.fillStyle = grad;
      this.ctx.fillRect(x, h - barHeight, barWidth - 1, barHeight);
      x += barWidth;
      if (x > w) break;
    }

    // 2. Draw Time-Domain Waveform (Glowing Foreground Line)
    this.ctx.lineWidth = 2.2 * (window.devicePixelRatio || 1);
    this.ctx.strokeStyle = primaryColor;
    this.ctx.shadowBlur = 8;
    this.ctx.shadowColor = primaryColor;

    this.ctx.beginPath();
    const sliceWidth = w / this.bufferLength;
    let posX = 0;

    for (let i = 0; i < this.bufferLength; i++) {
      const v = this.timeDomainArray[i] / 128.0;
      const posY = (v * h) / 2;

      if (i === 0) {
        this.ctx.moveTo(posX, posY);
      } else {
        this.ctx.lineTo(posX, posY);
      }
      posX += sliceWidth;
    }

    this.ctx.stroke();
    this.ctx.shadowBlur = 0; // Reset shadow
  }
}

window.AudioVisualizer = AudioVisualizer;
