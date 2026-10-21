/**
 * Ensures only ONE part of the app holds the microphone at a time.
 *
 * The diagnostic's LIVE METER and START LISTENING each call getUserMedia
 * separately, so leaving the meter running and then starting a recording opened
 * two streams on the same device. A USB microphone that Windows hands out in
 * exclusive mode gives the second caller a track that is open but permanently
 * MUTED -- the recording then hears nothing while the meter next to it happily
 * shows signal, which is a genuinely baffling thing to debug.
 *
 * Whoever is about to open the microphone calls claim(); everyone else releases.
 */

type Release = () => void;

const holders = new Set<Release>();

/** Registers a release callback; call the returned function when finished. */
export function registerMicHolder(release: Release): () => void {
  holders.add(release);
  return () => holders.delete(release);
}

/** Asks every other holder to let go before this caller opens the microphone. */
export function claimMicrophone(claimant: Release | null = null): void {
  for (const release of holders) {
    if (release !== claimant) {
      try {
        release();
      } catch {
        /* a failing release must not stop the claim */
      }
    }
  }
}
