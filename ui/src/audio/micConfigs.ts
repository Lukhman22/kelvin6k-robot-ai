/**
 * Candidate microphone configurations, and the winner is remembered.
 *
 * Different Windows audio drivers accept different constraint combinations.
 * A USB microphone can deliver a MUTED track under one set of constraints and
 * perfect audio under another, so rather than guessing, the diagnostic opens
 * the microphone once per configuration, counts how many non-zero audio
 * samples actually arrive, and the first configuration that produces real
 * sound is stored and used from then on.
 */

export interface MicConfig {
  id: string;
  label: string;
  /** Why this configuration is worth trying. Shown in the results table. */
  reason: string;
  build: (deviceId?: string) => MediaTrackConstraints | true;
}

export const MIC_CONFIGS: MicConfig[] = [
  {
    id: "browser-default",
    label: "Browser default — audio: true",
    reason: "No constraints at all. If only this works, a constraint is the problem.",
    build: () => true,
  },
  {
    id: "device-raw-mono-48k",
    label: "Selected device, mono 48 kHz, auto-gain enabled",
    reason: "The configuration best suited to far-field speech recognition.",
    build: (deviceId) => ({
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      channelCount: 1,
      sampleRate: 48000,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: true,
    }),
  },
  {
    id: "device-raw",
    label: "Selected device, auto-gain enabled, driver picks format",
    reason: "Some drivers mute the track when a channel count or rate is forced.",
    build: (deviceId) => ({
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: true,
    }),
  },
  {
    id: "device-processed",
    label: "Selected device, Windows processing ON",
    reason: "A few drivers only deliver audio through the processing pipeline.",
    build: (deviceId) => ({
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    }),
  },
  {
    id: "device-stereo",
    label: "Selected device, stereo, no processing",
    reason: "This microphone reports 2 channels; this matches it exactly.",
    build: (deviceId) => ({
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      channelCount: 2,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    }),
  },
  {
    id: "device-plain",
    label: "Selected device, nothing else specified",
    reason: "The simplest possible request for this specific device.",
    build: (deviceId) => (deviceId ? { deviceId: { exact: deviceId } } : true),
  },
];

const STORAGE_KEY = "kelvin6k.micConfig";

export interface SavedMicChoice {
  configId: string;
  deviceId: string;
}

export function loadMicChoice(): SavedMicChoice | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as SavedMicChoice) : null;
  } catch {
    return null; // private windows and blocked storage must not break capture
  }
}

export function saveMicChoice(choice: SavedMicChoice): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(choice));
  } catch {
    /* not fatal -- the choice simply isn't remembered next time */
  }
}

/** Builds the constraints for a saved choice, falling back to a safe default. */
export function constraintsFor(configId: string, deviceId?: string): MediaTrackConstraints | true {
  const config = MIC_CONFIGS.find((c) => c.id === configId) ?? MIC_CONFIGS[1];
  const built = config.build(deviceId);

  // A chosen device ALWAYS wins. The "browser default" configuration builds
  // plain `true`, which discards the deviceId and opens whatever Windows calls
  // the default -- often the silent Communications endpoint. That is how the
  // diagnostic (which honours the dropdown) could hear audio while
  // START LISTENING (which used this) recorded silence from another device.
  if (deviceId && built === true) {
    return { deviceId: { exact: deviceId } };
  }
  if (deviceId && typeof built === "object" && !built.deviceId) {
    return { ...built, deviceId: { exact: deviceId } };
  }
  return built;
}
