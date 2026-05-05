export type RecorderState = "idle" | "listening" | "processing";

export interface RecorderStartOptions {
  onWaveform?: (level: number) => void;
  onSilence?: () => void;
  silenceMs?: number;
}

export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private animationFrame: number | null = null;
  private chunks: Blob[] = [];
  private state: RecorderState = "idle";
  private silenceStart = 0;

  get currentState(): RecorderState {
    return this.state;
  }

  async start(options: RecorderStartOptions = {}): Promise<void> {
    if (this.state !== "idle") return;

    const silenceMs = Math.max(1000, options.silenceMs ?? 1500);
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: "audio/webm" });
    this.chunks = [];

    this.audioContext = new AudioContext();
    const source = this.audioContext.createMediaStreamSource(this.stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 512;
    source.connect(this.analyser);
    this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);

    this.mediaRecorder.ondataavailable = (ev: BlobEvent) => {
      if (ev.data && ev.data.size > 0) this.chunks.push(ev.data);
    };

    this.state = "listening";
    this.mediaRecorder.start(200);

    const loop = () => {
      if (!this.analyser || !this.dataArray || this.state !== "listening") return;
      this.analyser.getByteFrequencyData(this.dataArray);
      const avg = this.dataArray.reduce((sum, v) => sum + v, 0) / this.dataArray.length;
      const level = Math.min(1, avg / 64);
      options.onWaveform?.(level);

      if (avg < 8) {
        if (this.silenceStart === 0) this.silenceStart = performance.now();
        if (performance.now() - this.silenceStart >= silenceMs) {
          options.onSilence?.();
        }
      } else {
        this.silenceStart = 0;
      }

      this.animationFrame = requestAnimationFrame(loop);
    };

    this.animationFrame = requestAnimationFrame(loop);
  }

  async stop(): Promise<Blob> {
    if (!this.mediaRecorder || this.state !== "listening") {
      throw new Error("Recorder is not active");
    }

    this.state = "processing";

    const blob = await new Promise<Blob>((resolve) => {
      const recorder = this.mediaRecorder as MediaRecorder;
      recorder.onstop = () => {
        const audioBlob = new Blob(this.chunks, { type: "audio/webm" });
        resolve(audioBlob);
      };
      recorder.stop();
    });

    this.cleanup();
    this.state = "idle";
    return blob;
  }

  cancel(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }
    this.cleanup();
    this.state = "idle";
  }

  private cleanup(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.audioContext) {
      this.audioContext.close().catch(() => undefined);
      this.audioContext = null;
    }

    this.analyser = null;
    this.dataArray = null;
    this.mediaRecorder = null;
    this.chunks = [];
    this.silenceStart = 0;
  }
}
