/**
 * Plays assistant answers as audio in the BROWSER.
 *
 * Architecture (explicit, per the backend's /api/tts/audio):
 *     Python TTS -> audio bytes -> HTTP -> HTMLAudioElement -> laptop speaker
 *
 * The SPEAKING state is driven by the element's real `play`/`ended` events, so
 * the UI only claims to be speaking while audio is genuinely playing.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchSpeechAudio } from "../api";

export type PlaybackState = "idle" | "generating" | "playing" | "error";

export function useAudioPlayer() {
  const [playbackState, setPlaybackState] = useState<PlaybackState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [engine, setEngine] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null);

  const cleanup = useCallback(() => {
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
  }, []);

  /**
   * Resolver for the speak() call currently in flight.
   *
   * Without this, stopping playback stranded the caller forever: stop() calls
   * audio.pause(), which fires neither `ended` nor `error`, so the promise
   * speak() returned never settled. Since speak() itself calls stop() first,
   * two overlapping announcements left the FIRST await hanging -- and the
   * caller of that await was the voice turn, which then never reached its
   * finally: `busy` stayed true, the UI stayed on THINKING, and the wake word
   * was never resumed.
   */
  const pendingRef = useRef<((finished: boolean) => void) | null>(null);

  const settle = useCallback((finished: boolean) => {
    const resolve = pendingRef.current;
    pendingRef.current = null;
    resolve?.(finished);
  }, []);

  const stop = useCallback(() => {
    try {
      audioRef.current?.pause();
    } catch {
      /* ignore */
    }
    cleanup();
    // Release anyone waiting on the playback that was just cancelled.
    settle(false);
    setPlaybackState("idle");
  }, [cleanup, settle]);

  /** Generates speech on the backend and plays it here. Resolves when finished. */
  const speak = useCallback(
    async (text: string): Promise<boolean> => {
      const trimmed = (text ?? "").trim();
      if (!trimmed) return false;

      stop();          // also settles any playback this one supersedes
      setError(null);
      setPlaybackState("generating");

      let blob: Blob;
      try {
        const result = await fetchSpeechAudio(trimmed);
        blob = result.blob;
        setEngine(result.engine);
      } catch (err) {
        setPlaybackState("error");
        setError(
          err instanceof Error
            ? `Speech generation failed: ${err.message}`
            : String(err),
        );
        return false;
      }

      if (blob.size < 512) {
        setPlaybackState("error");
        setError("The backend returned an empty audio file.");
        return false;
      }

      const url = URL.createObjectURL(blob);
      urlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;

      return new Promise<boolean>((resolve) => {
        // Registered so stop(), or a later speak(), can settle this promise
        // instead of abandoning the caller.
        pendingRef.current = resolve;
        audio.onplay = () => setPlaybackState("playing");
        audio.onended = () => {
          cleanup();
          setPlaybackState("idle");
          settle(true);
        };
        audio.onerror = () => {
          cleanup();
          setPlaybackState("error");
          setError("The browser could not play the audio returned by the backend.");
          settle(false);
        };

        audio.play().catch((err: unknown) => {
          cleanup();
          setPlaybackState("error");
          // Chrome blocks audio until the page has had a user gesture.
          const name = err instanceof Error ? err.name : "";
          setError(
            name === "NotAllowedError"
              ? "The browser blocked audio playback. Click anywhere on the page once, then try again."
              : `Playback failed: ${err instanceof Error ? err.message : String(err)}`,
          );
          settle(false);
        });
      });
    },
    [stop, cleanup, settle],
  );

  useEffect(() => () => { stop(); }, [stop]);

  return {
    speak,
    stop,
    playbackState,
    speaking: playbackState === "playing",
    generating: playbackState === "generating",
    error,
    engine,
  };
}
