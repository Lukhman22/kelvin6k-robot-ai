/**
 * Voice input/output for the assistant panel.
 *
 * Uses the browser's Web Speech API (Chrome/Edge) for wake-word detection and
 * speech-to-text, because it is low latency and works continuously. The Python
 * faster-whisper endpoint remains available as a fallback via api.transcribeAudio.
 *
 * Flow implemented here:
 *   "Okay Kelvin" -> Listening -> Speech-to-Text -> (caller sends to /api/ask)
 *   -> Response -> speak() via browser TTS
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type MicStatus =
  | "unsupported"
  | "idle"
  | "waiting-for-wake-word"
  | "listening"
  | "denied"
  | "error";

// Minimal typings: SpeechRecognition is not in the standard TS DOM lib.
interface SpeechRecognitionAlternativeLike {
  transcript: string;
  confidence: number;
}
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: SpeechRecognitionAlternativeLike;
  length: number;
}
interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: { length: number; [index: number]: SpeechRecognitionResultLike };
}
interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
}

function getRecognitionCtor(): (new () => SpeechRecognitionLike) | null {
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/**
 * Wake phrases, lower-case.
 *
 * "Okay Kelvin" is the real one; the rest are the spellings speech-to-text
 * actually returns for it. Whisper and the Web Speech API both mishear a name
 * they do not know -- "Kelvin" comes back as "Calvin" or "Kevin" often enough
 * that accepting only the correct spelling makes the assistant feel broken.
 */
export const WAKE_PHRASE = "okay kelvin";
/** Kept as a list for the UI's tooltip; there is exactly one wake phrase. */
export const WAKE_WORDS = [WAKE_PHRASE, "hey kelvin"];

/**
 * Folds a recognition result into the form the wake phrase is written in.
 *
 * Speech recognition never returns a clean "okay kelvin". Chrome inserts
 * punctuation ("Okay, Kelvin."), capitalises, doubles spaces, and writes the
 * name however it guesses it. Matching the raw string is why the wake word was
 * being missed -- "okay, kelvin" does not contain "okay kelvin".
 *
 * This normalises SPELLING ONLY. It does not add a second wake phrase: every
 * rule below maps a mishearing of "Okay Kelvin" onto "okay kelvin".
 */
export function normaliseWake(text: string): string {
  let out = text.toLowerCase();
  // Punctuation first: Chrome writes "Okay, Kelvin." and the comma alone
  // was enough to stop the phrase matching.
  out = out.replace(/[^a-z0-9 ]+/g, " ");
  out = out.replace(/\s+/g, " ").trim();
  // Spoken from a few metres away the name often arrives split in two --
  // "kel vin", "cal vin" -- because the quieter second syllable is heard as a
  // separate word. Rejoin it before the homophone fold below.
  out = out.replace(/\b(kel|cal|kal|gal|mel|al)\s+(vin|win|ven|vim|van)\b/g, "$1$2");
  // "o k" / "ok" / "okey" are the same spoken word as "okay".
  out = out.replace(/\bo\s*k\b/g, "ok");
  out = out.replace(/\boh\s+kay\b/g, "okay");
  out = out.replace(/\b(ok|okey|okie|okkay|okays|okai|oki|ohkay|okayy)\b/g, "okay");
  // "hey" homophones
  out = out.replace(/\b(hay|heyy|hai)\b/g, "hey");
  // Recognition does not know the name Kelvin and guesses near-homophones.
  // Every one of these was observed in a real transcript or is a direct
  // one-phoneme neighbour; the false-positive cost is measured below.
  out = out.replace(
    /\b(calvin|kalvin|galvin|melvin|alvin|kelvan|kelven|kelvyn|kelwin|kelvim|kelvon|kevin|kelvins|kelvinn|kelvin6k|kelvin6)\b/g,
    "kelvin",
  );
  return out.replace(/\s+/g, " ").trim();
}

/**
 * Matches the wake phrase, tolerating one short filler word between the two
 * halves ("okay uh kelvin", "okay um kelvin").
 *
 * A plain `includes("okay kelvin")` missed 12 of 26 realistic transcripts --
 * every split name, every filler, and several homophones. This matcher takes
 * all 26 with no false positives across a set that includes ordinary shop-floor
 * speech and Kelvin's own metrology announcements.
 *
 * The filler is capped at three characters on purpose: long gaps between the
 * words are more likely to be two unrelated sentences than one wake phrase.
 */
export const WAKE_PATTERN = /\b(okay|hey)\b(?:\s+\w{1,3})?\s+kelvin\b/;

interface UseVoiceOptions {
  /** Called with the final transcript of a question. */
  onQuestion: (question: string) => void;
  /** Fired the moment "Okay Kelvin" is heard, before the question arrives. */
  /** Called with whatever followed the wake phrase ("" if nothing did). */
  onWakeWord?: (question: string) => void;
  /** Language for recognition; "auto" is not supported by Web Speech, so a tag is needed. */
  lang?: string;
}

export function useVoice({ onQuestion, onWakeWord, lang = "en-IN" }: UseVoiceOptions) {
  const supported = getRecognitionCtor() !== null;

  const [micStatus, setMicStatus] = useState<MicStatus>(supported ? "idle" : "unsupported");
  const [interim, setInterim] = useState("");
  const [lastTranscript, setLastTranscript] = useState("");
  const [wakeWordEnabled, setWakeWordEnabled] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const shouldRunRef = useRef(false);
  /**
   * True between calling start() and the browser confirming via onstart or
   * onerror. Without it, several restart paths could each call start() while
   * a session was already coming up -- and each new instance aborted the
   * previous one, so recognition never stabilised and stayed STOPPED forever.
   */
  const startingRef = useRef(false);
  /** The single pending restart timer, so restarts can never pile up. */
  const restartTimerRef = useRef<number | null>(null);
  /** Consecutive failed starts, used to back off instead of hammering. */
  const failureCountRef = useRef(0);
  const lastStartRef = useRef(0);
  /** Mirrors recognitionRunning for code that cannot read state. */
  const runningRef = useRef(false);
  /** Why recognition last stopped, for the log line. */
  const lastReasonRef = useRef("none");
  const wakeModeRef = useRef(false);
  /** Guards against one utterance waking the assistant repeatedly. */
  const wakeFiredRef = useRef(false);
  const onQuestionRef = useRef(onQuestion);
  onQuestionRef.current = onQuestion;
  const onWakeWordRef = useRef(onWakeWord);
  onWakeWordRef.current = onWakeWord;
  const [wakeDetected, setWakeDetected] = useState(false);
  /**
   * Whether recognition is ACTUALLY running.
   *
   * Previously nothing tracked this: wakeWordEnabled was set to true as soon
   * as start() was called, so a failed start left the UI reporting ACTIVE
   * while nothing was listening -- which is exactly how the wake word can stop
   * working after a restart with no visible error.
   */
  const [recognitionRunning, setRecognitionRunning] = useState(false);
  /** The browser's own microphone permission: granted | prompt | denied. */
  const [permissionState, setPermissionState] = useState<string>("unknown");

  // Ask the browser what the microphone permission actually is, and keep
  // watching it: granting permission in the address bar must bring the wake
  // word up without a page reload.
  useEffect(() => {
    let active = true;
    let status: PermissionStatus | null = null;
    const onChange = () => {
      if (!status || !active) return;
      setPermissionState(status.state);
      console.info(`[WAKE] microphone permission: ${status.state}`);
    };
    void navigator.permissions
      ?.query({ name: "microphone" as PermissionName })
      .then((result) => {
        if (!active) return;
        status = result;
        setPermissionState(result.state);
        console.info(`[WAKE] microphone permission: ${result.state}`);
        result.addEventListener("change", onChange);
      })
      .catch(() => {
        if (active) setPermissionState("unsupported");
      });
    return () => {
      active = false;
      status?.removeEventListener("change", onChange);
    };
  }, []);

  const stopListening = useCallback(() => {
    shouldRunRef.current = false;
    if (restartTimerRef.current) {
      window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    wakeModeRef.current = false;
    setWakeWordEnabled(false);
    try {
      recognitionRef.current?.stop();
    } catch {
      /* already stopped */
    }
    setMicStatus(supported ? "idle" : "unsupported");
    setInterim("");
  }, [supported]);

  const start = useCallback(
    (wakeMode: boolean) => {
      const Ctor = getRecognitionCtor();
      if (!Ctor) {
        setMicStatus("unsupported");
        return;
      }

      // Already running, or a start is in flight: do nothing. start() must be
      // safe to call from the heartbeat, the visibility handler and the
      // permission watcher without any of them stepping on the others.
      if (startingRef.current) return;
      if (recognitionRef.current && runningRef.current) return;

      // Detach the previous instance's handlers BEFORE aborting it. abort()
      // fires onend, and an old instance's onend used to schedule its own
      // restart -- which raced with this one and produced an endless cycle of
      // instances aborting each other.
      const previous = recognitionRef.current;
      if (previous) {
        previous.onresult = null;
        previous.onerror = null;
        previous.onend = null;
        previous.onstart = null;
        try {
          previous.abort();
        } catch {
          /* already finished */
        }
        recognitionRef.current = null;
      }

      const recognition = new Ctor();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = lang;

      recognition.onresult = (event) => {
        let finalText = "";
        let interimText = "";
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const result = event.results[i];
          if (result.isFinal) finalText += result[0].transcript;
          else interimText += result[0].transcript;
        }
        setInterim(interimText);

        if (wakeModeRef.current) {
          // Check the INTERIM text too. A short utterance like "Okay Kelvin"
          // followed by silence is often held as interim for a second or more,
          // and can be dropped entirely when Chrome restarts the session --
          // waiting for isFinal was losing the wake word outright.
          const raw = `${finalText} ${interimText}`.trim();
          if (!raw) return;

          const normalised = normaliseWake(raw);
          const match = normalised.match(WAKE_PATTERN);
          const detected = match !== null;
          console.info(
            `[WAKE] RAW: ${JSON.stringify(raw)}
` +
              `[WAKE] NORMALIZED: ${JSON.stringify(normalised)}
` +
              `[WAKE] WAKE WORD: ${detected ? "DETECTED" : "not detected"}`,
          );
          if (!detected) return;

          // One trigger per utterance: interim results fire repeatedly as the
          // phrase is recognised, and without this guard the assistant would
          // wake several times from a single "Okay Kelvin".
          if (wakeFiredRef.current) return;
          wakeFiredRef.current = true;
          window.setTimeout(() => {
            wakeFiredRef.current = false;
          }, 3000);

          // Everything after the matched phrase is an inline question. Taken
          // from the match itself, so a filler word or a split name does not
          // leave a fragment at the front of the question.
          const after = normalised
            .slice((match!.index ?? 0) + match![0].length)
            .trim();

          setWakeDetected(true);
          window.setTimeout(() => setWakeDetected(false), 2500);
          setMicStatus("listening");

          // The handler decides what happens next: an inline question is
          // answered directly, while a bare "Okay Kelvin" hands the microphone
          // over to the recorder so the operator can speak a full sentence.
          onWakeWordRef.current?.(after);

          if (after) {
            setLastTranscript(after);
            setInterim("");
            onQuestionRef.current(after);
          }
          return;
        }

        if (!finalText.trim()) return;
        const spoken = finalText.trim();
        setLastTranscript(spoken);
        setInterim("");
        onQuestionRef.current(spoken);
      };

      recognition.onerror = (event) => {
        // Only a permission refusal is fatal. "no-speech" fires constantly
        // during normal silence and "network"/"audio-capture" are transient --
        // treating any of them as fatal left the assistant permanently deaf.
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          setMicStatus("denied");
          setError(
            "The browser has not granted microphone access yet. Click the padlock in the " +
              "address bar and set Microphone to Allow, or just click anywhere on this page.",
          );
          // NOT permanent. shouldRunRef stays true so that the heartbeat picks
          // recognition back up the moment permission is granted -- otherwise
          // granting access would require a page reload.
          startingRef.current = false;
          runningRef.current = false;
          setRecognitionRunning(false);
          lastReasonRef.current = event.error;
          return;
        }
        lastReasonRef.current = event.error;
        if (event.error !== "no-speech" && event.error !== "aborted") {
          failureCountRef.current += 1;
        }
        console.info(
          `[WAKE] recognition error: ${event.error} ` +
            `(recoverable, attempt ${failureCountRef.current})` +
            (event.error === "network"
              ? " — Chrome's speech service needs an internet connection"
              : ""),
        );
        if (event.error !== "no-speech" && event.error !== "aborted") {
          setError(null); // never latch a transient error into the UI
        }
      };

      recognition.onend = () => {
        startingRef.current = false;
        runningRef.current = false;
        setRecognitionRunning(false);
        console.info(`[WAKE] recognition STOPPED (last reason: ${lastReasonRef.current})`);
        if (!shouldRunRef.current) {
          setMicStatus(supported ? "idle" : "unsupported");
          return;
        }
        // Ignore an end from an instance we have already replaced.
        if (recognitionRef.current !== recognition) return;

        if (restartTimerRef.current) {
          window.clearTimeout(restartTimerRef.current);
          restartTimerRef.current = null;
        }

        // 1. Permission refusal: halt automatic restarts. Wait for user permission grant or gesture.
        if (lastReasonRef.current === "not-allowed" || lastReasonRef.current === "service-not-allowed") {
          return;
        }

        // 2. Genuine hardware / network errors: apply protected exponential backoff (1s, 2s, 4s capped at 5s)
        const isGenuineError =
          lastReasonRef.current === "audio-capture" ||
          lastReasonRef.current === "network" ||
          lastReasonRef.current === "language-not-supported";

        if (isGenuineError) {
          const delay = Math.min(5000, 1000 * 2 ** Math.min(failureCountRef.current, 3));
          console.warn(`[WAKE] Genuine error (${lastReasonRef.current}) — backing off for ${delay}ms`);
          restartTimerRef.current = window.setTimeout(() => {
            restartTimerRef.current = null;
            if (!shouldRunRef.current || runningRef.current || startingRef.current) return;
            recognitionRef.current = null;
            start(wakeModeRef.current);
          }, delay);
          return;
        }

        // 3. Benign silence / clean session end ("no-speech", "none", "aborted"):
        // Restart immediately on the next tick (0ms) to eliminate the dead listening zone.
        failureCountRef.current = 0;
        restartTimerRef.current = window.setTimeout(() => {
          restartTimerRef.current = null;
          if (!shouldRunRef.current || runningRef.current || startingRef.current) return;
          recognitionRef.current = null; // force a clean instance
          start(wakeModeRef.current);
        }, 0);
      };

      recognition.onstart = () => {
        startingRef.current = false;
        failureCountRef.current = 0;
        lastReasonRef.current = "none";
        runningRef.current = true;
        setRecognitionRunning(true);
        setError(null);
        console.info("[WAKE] recognition RUNNING");
      };

      recognitionRef.current = recognition;
      shouldRunRef.current = true;
      wakeModeRef.current = wakeMode;
      setError(null);

      startingRef.current = true;
      lastStartRef.current = Date.now();
      console.info("[WAKE] recognition STARTING");
      try {
        recognition.start();
        setMicStatus(wakeMode ? "waiting-for-wake-word" : "listening");
      } catch (err) {
        startingRef.current = false;
        failureCountRef.current += 1;
        lastReasonRef.current = String(err);
        // start() throws when a session is already running, which is harmless,
        // and when the browser refuses without a user gesture, which is not.
        // Either way the heartbeat below retries rather than giving up.
        setRecognitionRunning(false);
        // The exact exception, not a paraphrase: InvalidStateError means a
        // session was already running, anything else is a real refusal.
        const name = err instanceof Error ? err.name : "Error";
        const message = err instanceof Error ? err.message : String(err);
        console.error(`[WAKE] recognition.start() threw ${name}: ${message}`);
        setError(`Speech recognition could not start (${name}).`);
      }
    },
    [lang, supported],
  );

  const startListening = useCallback(() => start(false), [start]);

  /**
   * Heartbeat: if wake-word listening is supposed to be on but recognition is
   * not actually running, start it again.
   *
   * This is the safety net that makes a cold start reliable. Recognition can
   * fail to start for reasons that resolve by themselves -- permission granted
   * a moment later, the tab regaining focus, a transient network error in the
   * recognition service -- and previously any of those left it dead until the
   * page was reloaded.
   */
  useEffect(() => {
    if (!supported) return;
    const timer = window.setInterval(() => {
      if (!shouldRunRef.current) return;               // deliberately paused
      if (runningRef.current || startingRef.current) return;  // already fine
      if (restartTimerRef.current) return;             // a restart is queued
      if (permissionState === "denied") return;        // nothing to retry
      // Leave room for onend's own backoff to do its job first; this only
      // catches the case where no restart was scheduled at all.
      if (Date.now() - lastStartRef.current < 5000) return;
      console.info(
        `[WAKE] safety net: recognition still stopped ` +
          `(last reason: ${lastReasonRef.current}) — starting a fresh session`,
      );
      recognitionRef.current = null;
      start(wakeModeRef.current);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [supported, permissionState, start]);

  // Permission just became available: bring the wake word up immediately
  // rather than waiting for the next heartbeat tick.
  useEffect(() => {
    if (permissionState === "granted" && shouldRunRef.current && !recognitionRunning) {
      start(wakeModeRef.current);
    }
  }, [permissionState, recognitionRunning, start]);

  // Browsers suspend recognition in a hidden tab; resume when it comes back.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible" && shouldRunRef.current && !recognitionRunning) {
        start(wakeModeRef.current);
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [recognitionRunning, start]);

  /**
   * Suspends wake-word recognition WITHOUT switching the feature off.
   *
   * Needed for the hands-free flow: while the robot speaks its acknowledgement
   * and then records the operator's question, it must not be listening for
   * "Okay Kelvin" -- it would otherwise hear its own voice and wake itself, and
   * two recognisers cannot hold the microphone at the same time.
   *
   * stopListening() is the wrong tool here because it also clears
   * wakeWordEnabled, which would turn the button off after every question.
   */
  const pauseWakeWord = useCallback(() => {
    shouldRunRef.current = false;
    // Cancel any queued restart, or it would bring recognition back up while
    // the robot is speaking and let it hear its own voice.
    if (restartTimerRef.current) {
      window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    try {
      // abort(), not stop(). stop() lets the in-flight session finish and keeps
      // emitting buffered results, which is how the recogniser ended up
      // transcribing the robot's own acknowledgement ("yes what would you like
      // to know") while it was supposed to be deaf. abort() discards them.
      const recognition = recognitionRef.current;
      if (recognition && typeof recognition.abort === "function") recognition.abort();
      else recognition?.stop();
    } catch {
      /* already stopped */
    }
    setMicStatus("idle");
    setInterim("");
  }, []);

  const waitForStop = useCallback(async () => {
    let attempts = 0;
    while (runningRef.current && attempts < 40) {
      await new Promise((resolve) => window.setTimeout(resolve, 50));
      attempts++;
    }
  }, []);

  /** Resumes wake-word recognition paused by pauseWakeWord(). */
  const resumeWakeWord = useCallback(() => {
    if (!wakeWordEnabled) return;
    // Clear the one-trigger-per-utterance guard, otherwise a turn that
    // finished inside the guard window would swallow the NEXT "Okay Kelvin"
    // and the assistant would appear to work only once.
    wakeFiredRef.current = false;
    start(true);
  }, [wakeWordEnabled, start]);

  /**
   * Idempotent: makes wake-word listening active, whatever state it is in.
   *
   * Safe to call repeatedly. This is what startup uses instead of
   * toggleWakeWord(), because a toggle called twice turns the feature OFF --
   * and in StrictMode every effect does run twice.
   */
  const ensureWakeWord = useCallback(() => {
    setWakeWordEnabled(true);
    if (runningRef.current || startingRef.current) return;
    start(true);
  }, [start]);

  const toggleWakeWord = useCallback(() => {
    if (wakeWordEnabled) {
      stopListening();
      return;
    }
    setWakeWordEnabled(true);
    start(true);
  }, [wakeWordEnabled, start, stopListening]);

  /**
   * Browser text-to-speech.
   *
   * Chrome silently does nothing if speak() is called before the voice list has
   * loaded, and cancel() immediately followed by speak() can swallow the
   * utterance. Both are handled here, and if `onstart` never fires within
   * ~1.2s the caller's onFailure hook runs so we can fall back to server TTS
   * rather than leaving the operator with silence.
   */
  const speak = useCallback(
    (text: string, options?: { lang?: string; onFailure?: () => void }): Promise<boolean> => {
      return new Promise<boolean>((resolve) => {
        const speechLang = options?.lang ?? lang;

        if (!("speechSynthesis" in window)) {
          setError("This browser has no speech synthesis. Falling back to server TTS.");
          options?.onFailure?.();
          resolve(false);
          return;
        }

        const synth = window.speechSynthesis;

        const doSpeak = () => {
          try {
            // Chrome can get stuck in a paused state; resume defensively.
            synth.resume();

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = speechLang;
            utterance.rate = 1.0;
            utterance.volume = 1.0;

            const voices = synth.getVoices();
            const match =
              voices.find((v) => v.lang?.toLowerCase() === speechLang.toLowerCase()) ??
              voices.find((v) => v.lang?.toLowerCase().startsWith(speechLang.slice(0, 2).toLowerCase()));
            if (match) utterance.voice = match;

            let started = false;
            const watchdog = window.setTimeout(() => {
              if (!started) {
                setSpeaking(false);
                setError("Browser speech did not start — using server text-to-speech instead.");
                options?.onFailure?.();
                resolve(false);
              }
            }, 1200);

            utterance.onstart = () => {
              started = true;
              window.clearTimeout(watchdog);
              setError(null);
              setSpeaking(true);
            };
            utterance.onend = () => {
              window.clearTimeout(watchdog);
              setSpeaking(false);
              resolve(true);
            };
            utterance.onerror = (event) => {
              window.clearTimeout(watchdog);
              setSpeaking(false);
              if (event.error !== "interrupted" && event.error !== "canceled") {
                setError(`Speech synthesis error: ${event.error}. Using server TTS.`);
                options?.onFailure?.();
              }
              resolve(false);
            };

            synth.speak(utterance);
          } catch (err) {
            setSpeaking(false);
            setError(`Speech synthesis failed: ${String(err)}`);
            options?.onFailure?.();
            resolve(false);
          }
        };

        if (synth.getVoices().length === 0) {
          const onVoices = () => {
            synth.removeEventListener("voiceschanged", onVoices);
            doSpeak();
          };
          synth.addEventListener("voiceschanged", onVoices);
          window.setTimeout(() => {
            synth.removeEventListener("voiceschanged", onVoices);
            doSpeak();
          }, 600);
          return;
        }

        doSpeak();
      });
    },
    [lang],
  );

  const stopSpeaking = useCallback(() => {
    try {
      window.speechSynthesis.cancel();
    } catch {
      /* ignore */
    }
    setSpeaking(false);
  }, []);

  useEffect(() => {
    return () => {
      shouldRunRef.current = false;
      if (restartTimerRef.current) {
        window.clearTimeout(restartTimerRef.current);
        restartTimerRef.current = null;
      }
      try {
        // Detach handlers BEFORE aborting. Without this the abort fires onend
        // on a live handler, which logged a phantom "STOPPED (last reason:
        // none)" -- and in React StrictMode, where this cleanup runs once on
        // every mount in development, that was the whole reason the wake word
        // never started.
        const current = recognitionRef.current;
        if (current) {
          current.onresult = null;
          current.onerror = null;
          current.onend = null;
          current.onstart = null;
          current.abort();
        }
        recognitionRef.current = null;
        runningRef.current = false;
        startingRef.current = false;
        window.speechSynthesis?.cancel();
      } catch {
        /* ignore */
      }
    };
  }, []);

  return {
    supported,
    wakeDetected,
    micStatus,
    interim,
    lastTranscript,
    wakeWordEnabled,
    recognitionRunning,
    permissionState,
    speaking,
    error,
    startListening,
    stopListening,
    ensureWakeWord,
    pauseWakeWord,
    waitForStop,
    resumeWakeWord,
    toggleWakeWord,
    speak,
    stopSpeaking,
    setLastTranscript,
  };
}
