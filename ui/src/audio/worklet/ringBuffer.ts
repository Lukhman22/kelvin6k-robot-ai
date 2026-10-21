/**
 * High-performance circular audio ring buffer for Kelvin6K.
 *
 * Maintains a fixed pre-allocated window of 16 kHz mono 16-bit PCM audio
 * (default 2000 ms = 32,000 samples).
 *
 * Supports continuous writing from an AudioWorklet without memory re-allocations,
 * and lock-free chronological slicing with wrap-around handling.
 */

export const DEFAULT_RING_SAMPLE_RATE = 16000;
export const DEFAULT_RING_DURATION_MS = 2000;
export const DEFAULT_RING_CAPACITY = (DEFAULT_RING_SAMPLE_RATE * DEFAULT_RING_DURATION_MS) / 1000; // 32,000 samples

/** Converts normalized Float32 samples (-1.0 to +1.0) into signed 16-bit PCM integers. */
export function floatTo16BitPcm(float32: Float32Array): Int16Array {
  const pcm16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const s = Math.max(-1.0, Math.min(1.0, float32[i]));
    pcm16[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
  }
  return pcm16;
}

/** Converts signed 16-bit PCM integers to normalized Float32 samples (-1.0 to +1.0). */
export function pcm16ToFloat(int16: Int16Array): Float32Array {
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) {
    const s = int16[i];
    float32[i] = s < 0 ? s / 32768.0 : s / 32767.0;
  }
  return float32;
}

export class AudioRingBuffer {
  readonly capacity: number;
  readonly sampleRate: number;
  private readonly buffer: Int16Array;
  private writeIndex: number = 0;
  private samplesWrittenTotal: number = 0;

  constructor(capacity: number = DEFAULT_RING_CAPACITY, sampleRate: number = DEFAULT_RING_SAMPLE_RATE) {
    if (capacity <= 0) {
      throw new Error(`RingBuffer capacity must be positive, got ${capacity}`);
    }
    this.capacity = capacity;
    this.sampleRate = sampleRate;
    this.buffer = new Int16Array(capacity);
  }

  /**
   * Writes PCM samples into the circular buffer.
   * If Float32Array is passed, it is converted to 16-bit PCM automatically.
   */
  write(samples: Int16Array | Float32Array): number {
    if (samples.length === 0) return 0;

    const pcm = samples instanceof Int16Array ? samples : floatTo16BitPcm(samples);
    const count = pcm.length;

    for (let i = 0; i < count; i += 1) {
      this.buffer[this.writeIndex] = pcm[i];
      this.writeIndex = (this.writeIndex + 1) % this.capacity;
    }

    this.samplesWrittenTotal += count;
    return count;
  }

  /** Returns number of valid samples currently available in the buffer. */
  get available(): number {
    return Math.min(this.samplesWrittenTotal, this.capacity);
  }

  /** Total number of samples ever written to this buffer. */
  get totalWritten(): number {
    return this.samplesWrittenTotal;
  }

  /** Duration of buffered audio in milliseconds. */
  get durationMs(): number {
    return (this.available / this.sampleRate) * 1000;
  }

  /** Current raw write index. */
  get currentWriteIndex(): number {
    return this.writeIndex;
  }

  /**
   * Reads the most recent `count` samples from the buffer, ending `offsetFromLatest` samples
   * before the latest written sample.
   *
   * Example: read(16000) returns the last 1.0 second of audio.
   */
  read(count: number, offsetFromLatest: number = 0): Int16Array {
    const available = this.available;
    if (count <= 0 || available === 0) {
      return new Int16Array(0);
    }

    const safeOffset = Math.max(0, offsetFromLatest);
    if (safeOffset >= available) {
      return new Int16Array(0);
    }

    const actualCount = Math.min(count, available - safeOffset);
    const output = new Int16Array(actualCount);

    // Calculate start position inside circular buffer
    let readIdx = (this.writeIndex - safeOffset - actualCount) % this.capacity;
    if (readIdx < 0) {
      readIdx += this.capacity;
    }

    for (let i = 0; i < actualCount; i += 1) {
      output[i] = this.buffer[readIdx];
      readIdx = (readIdx + 1) % this.capacity;
    }

    return output;
  }

  /** Reads all currently available audio chronologically. */
  readAll(): Int16Array {
    return this.read(this.available, 0);
  }

  /** Resets write pointers and clears data. */
  clear(): void {
    this.buffer.fill(0);
    this.writeIndex = 0;
    this.samplesWrittenTotal = 0;
  }
}
