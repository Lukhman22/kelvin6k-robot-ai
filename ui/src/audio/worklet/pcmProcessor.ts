/**
 * AudioWorklet Processor for Kelvin6K.
 *
 * Runs on the dedicated Web Audio rendering thread.
 * Downsamples native microphone audio (e.g. 48kHz or 44.1kHz) to mono 16kHz 16-bit PCM
 * and transfers buffers to the main thread / ring buffer without blocking UI rendering.
 */

export const PCM_PROCESSOR_NAME = "kelvin-pcm-processor";

export interface PcmWorkletMessage {
  type: "pcm";
  buffer: ArrayBuffer;
  sampleCount: number;
  sampleRate: number;
}

/**
 * Self-contained AudioWorklet source string.
 * Using an inline Blob URL ensures reliable loading in Vite development, production bundles,
 * and test runners without relying on external static server MIME mappings.
 */
export const PCM_PROCESSOR_CODE = `
class KelvinPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.sourceRate = sampleRate; // Global sampleRate in AudioWorkletGlobalScope
    this.ratio = this.sourceRate / this.targetRate;
    this.resamplePhase = 0;
    
    // Internal accumulation buffer (1024 samples = 64ms of 16kHz audio)
    this.bufferSize = 1024;
    this.pcmBuffer = new Int16Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelCount = input.length;
    const frameCount = input[0].length; // Typically 128 frames

    // Downsample and mix to mono
    let sourceIdx = this.resamplePhase;
    while (sourceIdx < frameCount) {
      const idx = Math.floor(sourceIdx);
      const frac = sourceIdx - idx;
      const nextIdx = Math.min(idx + 1, frameCount - 1);

      // Mix channels to mono
      let sample = 0;
      let nextSample = 0;
      for (let c = 0; c < channelCount; c += 1) {
        sample += input[c][idx];
        nextSample += input[c][nextIdx];
      }
      sample /= channelCount;
      nextSample /= channelCount;

      // Linear interpolation
      const interpolated = sample + (nextSample - sample) * frac;

      // Clamp to [-1.0, 1.0] and convert to 16-bit PCM
      const clamped = Math.max(-1.0, Math.min(1.0, interpolated));
      const pcm16 = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);

      this.pcmBuffer[this.bufferIndex++] = pcm16;

      // When buffer is full, transfer it to main thread
      if (this.bufferIndex >= this.bufferSize) {
        this.port.postMessage(
          {
            type: "pcm",
            buffer: this.pcmBuffer.buffer,
            sampleCount: this.bufferIndex,
            sampleRate: this.targetRate,
          },
          [this.pcmBuffer.buffer]
        );
        this.pcmBuffer = new Int16Array(this.bufferSize);
        this.bufferIndex = 0;
      }

      sourceIdx += this.ratio;
    }

    this.resamplePhase = sourceIdx - frameCount;
    return true;
  }
}

registerProcessor('${PCM_PROCESSOR_NAME}', KelvinPcmProcessor);
`;

/** Creates an object URL containing the registered PCM processor script. */
export function createPcmProcessorBlobUrl(): string {
  const blob = new Blob([PCM_PROCESSOR_CODE], { type: "application/javascript" });
  return URL.createObjectURL(blob);
}
