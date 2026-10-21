/**
 * Microphone diagnostic: finds a configuration that actually delivers audio.
 *
 * The key measurement is NOT the level meter -- it is the count of non-zero
 * PCM samples. A track can be open, enabled and live while every sample it
 * produces is exactly 0.0, which is what "muted" means at the Windows level.
 * Counting real samples tells those two situations apart with certainty.
 *
 * `muted` is also watched over time rather than read once: Chrome commonly
 * reports muted=true for the first few hundred milliseconds after a device is
 * opened, before the driver starts producing audio. Reading it immediately
 * gives a false alarm.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { MIC_CONFIGS, saveMicChoice, type MicConfig } from "../audio/micConfigs";
import { claimMicrophone, registerMicHolder } from "../audio/micLock";

/** How long each configuration is listened to before judging it. */
const PROBE_MS = 2500;

export interface MicDevice {
  deviceId: string;
  label: string;
}

export interface ProbeResult {
  configId: string;
  label: string;
  reason: string;
  opened: boolean;
  error?: string;
  trackLabel?: string;
  actualDeviceId?: string;
  channelCount?: number | string;
  sampleRate?: number | string;
  readyState?: string;
  enabled?: boolean;
  /** Muted when the probe finished -- after giving the driver time to start. */
  mutedAtEnd?: boolean;
  /** True if the track was ever seen muted, even briefly. */
  everMuted?: boolean;
  /** The decisive number: how many samples were not exactly zero. */
  nonZeroSamples: number;
  totalSamples: number;
  peak: number;
  verdict: "AUDIO OK" | "SILENT" | "FAILED TO OPEN";
}

interface LiveInfo {
  trackLabel: string;
  actualDeviceId: string;
  channelCount: number | string;
  sampleRate: number | string;
  readyState: string;
  enabled: boolean;
  muted: boolean;
  contextState: string;
  nonZeroSamples: number;
  configLabel: string;
}



export function useMicDiagnostic() {
  const [devices, setDevices] = useState<MicDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [results, setResults] = useState<ProbeResult[]>([]);
  const [probing, setProbing] = useState(false);
  const [probingLabel, setProbingLabel] = useState("");
  const [winner, setWinner] = useState<ProbeResult | null>(null);

  const [monitoring, setMonitoring] = useState(false);
  const [level, setLevel] = useState(0);
  const [peak, setPeak] = useState(0);
  const [liveInfo, setLiveInfo] = useState<LiveInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<number | null>(null);
  const peakRef = useRef(0);
  const nonZeroRef = useRef(0);

  const refreshDevices = useCallback(async (): Promise<MicDevice[]> => {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const mics = all
        .filter((d) => d.kind === "audioinput")
        .map((d) => ({ deviceId: d.deviceId, label: d.label || "Microphone (allow access to see the name)" }));
      setDevices(mics);
      console.info(`[MIC DIAG] ${mics.length} audioinput device(s):`);
      mics.forEach((m) => console.info(`[MIC DIAG]   ${m.deviceId.slice(0, 16)}…  "${m.label}"`));
      return mics;
    } catch (err) {
      console.warn("[MIC DIAG] enumerateDevices failed:", err);
      return [];
    }
  }, []);

  useEffect(() => {
    void refreshDevices();
  }, [refreshDevices]);

  const stopMonitor = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    try {
      void ctxRef.current?.close();
    } catch {
      /* already closed */
    }
    ctxRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setMonitoring(false);
    setLevel(0);
  }, []);

  /** Opens one configuration, listens for PROBE_MS, and reports what arrived. */
  const probeOne = useCallback(
    async (config: MicConfig, deviceId: string): Promise<ProbeResult> => {
      const base: ProbeResult = {
        configId: config.id,
        label: config.label,
        reason: config.reason,
        opened: false,
        nonZeroSamples: 0,
        totalSamples: 0,
        peak: 0,
        verdict: "FAILED TO OPEN",
      };

      const constraints = config.build(deviceId || undefined);
      console.info(`[MIC DIAG] --- ${config.label} ---`);
      console.info("[MIC DIAG] constraints:", JSON.stringify(constraints));

      const { getSharedAudioContext, unlockAudioContext } = await import("../audio/sharedContext");
      const ctx = getSharedAudioContext()!;
      unlockAudioContext();
      if (ctx.state === "suspended") await ctx.resume().catch(() => {});

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
      } catch (err) {
        base.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
        console.warn("[MIC DIAG] getUserMedia failed:", base.error);
        return base;
      }

      const track = stream.getAudioTracks()[0];
      const settings = track?.getSettings?.() ?? {};
      let everMuted = !!track?.muted;
      const onMute = () => {
        everMuted = true;
        console.warn("[MIC DIAG] track fired 'mute'");
      };
      const onUnmute = () => console.info("[MIC DIAG] track fired 'unmute'");
      track?.addEventListener("mute", onMute);
      track?.addEventListener("unmute", onUnmute);

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      // Keep the graph pulled without playing the microphone back into the room.
      const silence = ctx.createGain();
      silence.gain.value = 0;
      analyser.connect(silence);
      silence.connect(ctx.destination);

      const buffer = new Float32Array(analyser.fftSize);
      let nonZero = 0;
      let total = 0;
      let peakSeen = 0;

      await new Promise<void>((resolve) => {
        const started = Date.now();
        const id = window.setInterval(() => {
          analyser.getFloatTimeDomainData(buffer);
          for (let i = 0; i < buffer.length; i += 1) {
            const sample = Math.abs(buffer[i]);
            if (sample > 0) nonZero += 1;
            if (sample > peakSeen) peakSeen = sample;
          }
          total += buffer.length;
          if (Date.now() - started >= PROBE_MS) {
            window.clearInterval(id);
            resolve();
          }
        }, 50);
      });

      const mutedAtEnd = !!track?.muted;

      const result: ProbeResult = {
        ...base,
        opened: true,
        trackLabel: track?.label,
        actualDeviceId: String(settings.deviceId ?? "(not reported)"),
        channelCount: settings.channelCount ?? "(not reported)",
        sampleRate: settings.sampleRate ?? ctx.sampleRate,
        readyState: track?.readyState,
        enabled: track?.enabled,
        mutedAtEnd,
        everMuted,
        nonZeroSamples: nonZero,
        totalSamples: total,
        peak: peakSeen,
        verdict: nonZero > 0 ? "AUDIO OK" : "SILENT",
      };

      console.info(
        `[MIC DIAG] result: ${result.verdict}  nonZero=${nonZero}/${total}  ` +
          `peak=${peakSeen.toFixed(4)}  muted=${mutedAtEnd}  everMuted=${everMuted}  ` +
          `channels=${result.channelCount}  rate=${result.sampleRate}`,
      );

      track?.removeEventListener("mute", onMute);
      track?.removeEventListener("unmute", onUnmute);
      track?.stop();
      stream.getTracks().forEach((t) => t.stop());
      try {
        // Do not close shared context
      } catch {
        /* ignore */
      }

      return result;
    },
    [],
  );

  /** Tries every configuration in turn and remembers the first that works. */
  const runFullDiagnostic = useCallback(async () => {
    stopMonitor();
    claimMicrophone(stopMonitor);
    setError(null);
    setResults([]);
    setWinner(null);
    setProbing(true);

    console.info("========== [MIC DIAG] FULL DIAGNOSTIC ==========");
    const mics = devices.length ? devices : await refreshDevices();
    const deviceId = selectedDeviceId || mics[0]?.deviceId || "";
    console.info("[MIC DIAG] Target device:", deviceId || "(browser default)");

    const collected: ProbeResult[] = [];
    for (const config of MIC_CONFIGS) {
      setProbingLabel(config.label);
      // Speak during the probe: each one listens for 2.5 seconds.
      const result = await probeOne(config, deviceId);
      collected.push(result);
      setResults([...collected]);
    }

    // Labels are only revealed once permission has been granted.
    void refreshDevices();

    const best = collected.find((r) => r.verdict === "AUDIO OK") ?? null;
    setWinner(best);
    if (best) {
      saveMicChoice({ configId: best.configId, deviceId });
      console.info(`[MIC DIAG] WINNER: ${best.label} — saved and will be used for recording.`);
    } else {
      console.warn("[MIC DIAG] No configuration produced audio.");
    }

    setProbing(false);
    setProbingLabel("");
  }, [devices, selectedDeviceId, probeOne, refreshDevices, stopMonitor]);

  /** Continuous level monitor using one specific configuration. */
  const startMonitor = useCallback(
    async (configId?: string) => {
      stopMonitor();
      claimMicrophone(stopMonitor);
      setError(null);
      peakRef.current = 0;
      nonZeroRef.current = 0;
      setPeak(0);

      const config =
        MIC_CONFIGS.find((c) => c.id === (configId ?? winner?.configId)) ?? MIC_CONFIGS[1];
      const constraints = config.build(selectedDeviceId || undefined);

      const { getSharedAudioContext, unlockAudioContext } = await import("../audio/sharedContext");
      const ctx = getSharedAudioContext()!;
      unlockAudioContext();
      if (ctx.state === "suspended") await ctx.resume().catch(() => {});
      ctxRef.current = ctx;

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
      } catch (err) {
        const name = err instanceof Error ? err.name : "";
        setError(
          name === "NotAllowedError"
            ? "Microphone permission denied. Click the padlock in the address bar, set Microphone to Allow, then reload."
            : name === "NotReadableError"
              ? "Windows would not open this microphone — another program has it. Close any Python script, Zoom, or Teams and retry."
              : `Could not open the microphone: ${err instanceof Error ? err.message : String(err)}`,
        );
        return;
      }

      streamRef.current = stream;
      const track = stream.getAudioTracks()[0];
      const settings = track?.getSettings?.() ?? {};

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      const silence = ctx.createGain();
      silence.gain.value = 0;
      analyser.connect(silence);
      silence.connect(ctx.destination);

      const buffer = new Float32Array(analyser.fftSize);
      timerRef.current = window.setInterval(() => {
        analyser.getFloatTimeDomainData(buffer);
        let sum = 0;
        for (let i = 0; i < buffer.length; i += 1) {
          sum += buffer[i] * buffer[i];
          if (buffer[i] !== 0) nonZeroRef.current += 1;
        }
        const rms = Math.sqrt(sum / buffer.length);
        setLevel(rms);
        peakRef.current = Math.max(peakRef.current, rms);
        setPeak(peakRef.current);
        // Re-read every tick: `muted` changes over time, it is not a fixed fact.
        setLiveInfo({
          trackLabel: track?.label ?? "(unknown)",
          actualDeviceId: String(settings.deviceId ?? "(not reported)"),
          channelCount: settings.channelCount ?? "(not reported)",
          sampleRate: settings.sampleRate ?? ctx.sampleRate,
          readyState: track?.readyState ?? "?",
          enabled: !!track?.enabled,
          muted: !!track?.muted,
          contextState: ctx.state,
          nonZeroSamples: nonZeroRef.current,
          configLabel: config.label,
        });
      }, 100);

      setMonitoring(true);
    },
    [selectedDeviceId, winner, stopMonitor],
  );

  // Let the recorder ask this monitor to release the microphone when the
  // operator presses START LISTENING, so only one stream is ever open.
  useEffect(() => registerMicHolder(stopMonitor), [stopMonitor]);

  useEffect(() => () => stopMonitor(), [stopMonitor]);

  return {
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    refreshDevices,
    results,
    probing,
    probingLabel,
    winner,
    runFullDiagnostic,
    monitoring,
    startMonitor,
    stopMonitor,
    level,
    peak,
    liveInfo,
    error,
  };
}
