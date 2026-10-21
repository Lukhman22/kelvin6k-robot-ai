const AudioCtor = window.AudioContext ?? (window as any).webkitAudioContext;

// Create it lazily only upon request to avoid auto-suspend issues in Safari
let sharedAudioContext: AudioContext | null = null;

export function getSharedAudioContext(): AudioContext | null {
  if (!sharedAudioContext && AudioCtor) {
    sharedAudioContext = new AudioCtor();
  }
  return sharedAudioContext;
}

export function unlockAudioContext() {
  const ctx = getSharedAudioContext();
  if (ctx && ctx.state === "suspended") {
    ctx.resume().catch(() => {});
  }
}

if (typeof document !== "undefined") {
  const unlock = () => {
    unlockAudioContext();
    document.removeEventListener("click", unlock);
    document.removeEventListener("touchstart", unlock);
    document.removeEventListener("keydown", unlock);
  };
  document.addEventListener("click", unlock);
  document.addEventListener("touchstart", unlock);
  document.addEventListener("keydown", unlock);
}
