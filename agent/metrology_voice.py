"""
Decides WHEN the metrology result is worth announcing, and in what words.

This module speaks nothing. It returns a sentence or None, and the existing
Voice Assistant in the browser does the speaking through its existing
speakAnswerText() path -- so there is one voice system, one TTS engine, and one
SPEAKING state, not a second assistant talking over the first.

The camera produces a result several times a second. Announcing each one would
be unusable, so an announcement is released only when:

  * the status CHANGED (PASS -> UNDER-WIDTH, UNDER-WIDTH -> PASS, ...), or
  * the status has held steady for METROLOGY_VOICE_STABLE_SEC and the cooldown
    since the last announcement has elapsed,

and never more often than METROLOGY_VOICE_ALERT_COOLDOWN_SEC. A status flapping
between two values is not announced at all until it settles, because the
stability window restarts on every change of value.

The wording is built from the SAME MetrologyResult the overlay draws, so the
operator can never hear one number and read another.
"""

import time
from dataclasses import dataclass
from typing import Optional

from agent import config
from agent.logging_setup import get_logger
from agent.metrology import INVALID, OVER_WIDTH, PASS, UNDER_WIDTH, MetrologyResult

log = get_logger("METROLOGY-VOICE")


@dataclass
class Announcement:
    """One thing worth saying. `id` increments so the UI can tell a new
    announcement from a repeat of the one it already spoke."""

    id: int
    text: str
    status: str
    average_width_mm: Optional[float]
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "average_width_mm": self.average_width_mm,
            "timestamp": self.timestamp,
        }


def _spoken_mm(value: float) -> str:
    """45.7 -> "45.7 millimeters". One decimal: the measurement is not precise
    enough to justify more, and a long decimal is unusable spoken aloud."""
    return f"{value:.1f} millimeters"


def compose(result: MetrologyResult) -> Optional[str]:
    """
    The sentence for this result, or None when there is nothing sensible to say.

    Never states a width the result does not have.
    """
    low = config.metrology_range_min()
    high = config.metrology_range_max()
    band = f"Required range is {low:.0f} to {high:.0f} millimeters."

    if result.status == INVALID:
        return "Unable to reliably measure layer width."

    if result.average_width_mm is None:
        return None

    width = _spoken_mm(result.average_width_mm)

    if result.status == PASS:
        return f"Layer width is correct. Average width is {width}."
    if result.status == UNDER_WIDTH:
        return f"Warning. Layer width is below the required range. Average width is {width}. {band}"
    if result.status == OVER_WIDTH:
        return f"Warning. Layer width is above the required range. Average width is {width}. {band}"
    return None


class AnnouncementGate:
    """
    Holds the announcement state between frames.

    One instance lives on the camera service. It is called with every
    measurement and returns an Announcement only when one is due.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._last_announced_status: Optional[str] = None
        self._last_announced_at = 0.0
        self._candidate_status: Optional[str] = None
        self._candidate_since = 0.0
        self.latest: Optional[Announcement] = None

    def reset(self) -> None:
        """Forget the history -- used when the camera drops, so the first
        measurement after it returns is announced rather than suppressed."""
        self._last_announced_status = None
        self._candidate_status = None
        self._candidate_since = 0.0

    def _cooldown_for(self, status: str) -> float:
        if status == INVALID:
            return config.METROLOGY_VOICE_INVALID_COOLDOWN_SEC
        return config.METROLOGY_VOICE_ALERT_COOLDOWN_SEC

    def consider(self, result: MetrologyResult, now: Optional[float] = None) -> Optional[Announcement]:
        """
        Returns an Announcement when this result is worth speaking, else None.
        """
        now = time.time() if now is None else now
        status = result.status

        # A new status restarts the stability window. This is what stops a
        # measurement flapping across the tolerance edge from being announced
        # twice a second.
        if status != self._candidate_status:
            self._candidate_status = status
            self._candidate_since = now

        changed = status != self._last_announced_status
        stable_for = now - self._candidate_since
        since_last = now - self._last_announced_at
        cooldown = self._cooldown_for(status)

        if changed:
            # A transition is the most useful moment to speak, but it still has
            # to have held briefly -- otherwise one noisy frame triggers it.
            if stable_for < config.METROLOGY_VOICE_STABLE_SEC:
                return None
        else:
            # Same status as last time: only repeat after the cooldown, and
            # only for a state the operator needs reminding about.
            if status == PASS:
                return None
            if since_last < cooldown:
                return None

        # NOTE: a genuine status TRANSITION is deliberately NOT held back by the
        # cooldown. An earlier version did that and swallowed two of the four
        # announcements in a 114-second test -- including the UNDER-WIDTH ->
        # PASS recovery, which is exactly what the operator needs to hear. The
        # stability window above is what prevents flapping; the cooldown's job
        # is to stop REPEATS of an unchanged status, not to hide real changes.

        text = compose(result)
        if not text:
            return None

        self._counter += 1
        self._last_announced_status = status
        self._last_announced_at = now
        announcement = Announcement(
            id=self._counter,
            text=text,
            status=status,
            average_width_mm=result.average_width_mm,
            timestamp=now,
        )
        self.latest = announcement
        log.info("[METROLOGY VOICE] %s", text)
        return announcement
