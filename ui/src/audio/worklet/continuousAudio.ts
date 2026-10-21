/**
 * Continuous Audio Pipeline Foundation for Kelvin6K.
 *
 * Phase 1 Foundation Only:
 * Connects an active MediaStream to an AudioWorkletNode that downsamples audio
 * to mono 16 kHz 16-bit PCM and streams it into a 2000 ms circular AudioRingBuffer.
 *
 * SAFETY / FEATURE FLAG:
 * CONTINUOUS_AUDIO_FOUNDATION defaults to FALSE.
 * When FALSE, the foundation is completely dormant and does not interfere
 * with the working Kelvin6K voice assistant, wake-word, or STT/TTS pipeline.
 */

import { useEffect, useRef, useState } from "react";
import { createPcmProcessorBlobUrl, PCM_PROCESSOR_NAME, type PcmWorkletMessage } from "./pcmProcessor";
import { AudioRingBuffer, DEFAULT_RING_CAPACITY, DEFAULT_RING_SAMPLE_RATE } from "./ringBuffer";

/**
 * Phase 1 Feature Flag.
 * Default is strictly FALSE. When FALSE, existing Kelvin6K voice pipeline
 * functions exactly as before without any changes.
 */
export const CONTINUOUS_AUDIO_FOUNDATION = false;

export class ContinuousAudioPipeline {
  private audioContext: AudioContext | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private readonly ringBuffer: AudioRingBuffer;
  private blobUrl: string | null = null;
  private running: boolean = false;

  constructor(capacity = DEFAULT_RING_CAPACITY, sampleRate = DEFAULT_RING_SAMPLE_RATE) {
    this.ringBuffer = new AudioRingBuffer(capacity, sampleRate);
  }

  get isRunning(): boolean {
    return this.running;
  }

  get buffer(): AudioRingBuffer {
    return this.ringBuffer;
  }

  /**
   * Initializes the AudioWorklet and pipes stream audio into the ring buffer.
   * Does NOT send audio to the backend.
   */
  async start(stream: MediaStream, existingContext?: AudioContext | null): Promise<boolean> {
    if (this.running) return true;
    if (!stream || !stream.active || stream.getAudioTracks().length === 0) {
      console.warn("[AUDIO_FOUNDATION] Cannot start: provided stream is inactive or has no tracks.");
      return false;
    }

    try {
      const AudioCtor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtor) return false;

      this.audioContext = existingContext ?? new AudioCtor();
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      // Load PCM processor into AudioWorklet
      if (!this.blobUrl) {
        this.blobUrl = createPcmProcessorBlobUrl();
      }
      await this.audioContext.audioWorklet.addModule(this.blobUrl);

      // Create nodes
      this.sourceNode = this.audioContext.createMediaStreamSource(stream);
      this.workletNode = new AudioWorkletNode(this.audioContext, PCM_PROCESSOR_NAME);

      // Receive 16kHz PCM chunks from worklet thread and store in ring buffer
      this.workletNode.port.onmessage = (event: MessageEvent<PcmWorkletMessage>) => {
        if (event.data?.type === "pcm" && event.data.buffer) {
          const pcm16 = new Int16Array(event.data.buffer);
          this.ringBuffer.write(pcm16);
        }
      };

      this.sourceNode.connect(this.workletNode);
      this.running = true;
      console.info(
        `[AUDIO_FOUNDATION] Continuous audio worklet started (in: ${this.audioContext.sampleRate}Hz -> ring: 16000Hz PCM, 2000ms capacity)`
      );
      return true;
    } catch (err) {
      console.error("[AUDIO_FOUNDATION] Failed to start continuous audio worklet:", err);
      this.stop();
      return false;
    }
  }

  /** Retrieves the most recent audio from the circular buffer. */
  getRecentPcm(durationMs: number): Int16Array {
    const samplesNeeded = Math.round((durationMs / 1000) * this.ringBuffer.sampleRate);
    return this.ringBuffer.read(samplesNeeded, 0);
  }

  /** Stops the worklet and disconnects audio nodes without stopping the underlying MediaStream. */
  stop(): void {
    if (!this.running && !this.sourceNode && !this.workletNode) return;

    try {
      this.sourceNode?.disconnect();
      this.workletNode?.disconnect();
      if (this.workletNode) {
        this.workletNode.port.onmessage = null;
      }
    } catch {
      /* ignore disconnection errors */
    }

    this.sourceNode = null;
    this.workletNode = null;
    this.running = false;
    this.ringBuffer.clear();

    if (this.blobUrl) {
      URL.revokeObjectURL(this.blobUrl);
      this.blobUrl = null;
    }
    console.info("[AUDIO_FOUNDATION] Continuous audio worklet stopped.");
  }
}

/**
 * Optional React Hook to safely attach the continuous audio pipeline foundation.
 * When enabled === false (the default), this is completely dormant.
 */
export function useContinuousAudioFoundation(
  stream: MediaStream | null,
  enabled: boolean = CONTINUOUS_AUDIO_FOUNDATION
) {
  const pipelineRef = useRef<ContinuousAudioPipeline | null>(null);
  const [active, setActive] = useState(false);
  const [samplesBuffered, setSamplesBuffered] = useState(0);

  useEffect(() => {
    if (!enabled || !stream) {
      if (pipelineRef.current?.isRunning) {
        pipelineRef.current.stop();
        setActive(false);
      }
      return;
    }

    if (!pipelineRef.current) {
      pipelineRef.current = new ContinuousAudioPipeline();
    }
    const pipeline = pipelineRef.current;

    let mounted = true;
    void pipeline.start(stream).then((ok) => {
      if (mounted && ok) {
        setActive(true);
      }
    });

    const interval = window.setInterval(() => {
      if (mounted && pipeline.isRunning) {
        setSamplesBuffered(pipeline.buffer.available);
      }
    }, 500);

    return () => {
      mounted = false;
      window.clearInterval(interval);
      pipeline.stop();
      setActive(false);
    };
  }, [stream, enabled]);

  return {
    pipeline: pipelineRef.current,
    active,
    samplesBuffered,
    capacity: DEFAULT_RING_CAPACITY,
  };
}
