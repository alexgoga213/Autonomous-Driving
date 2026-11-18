"""Driving-decision layer: turns per-frame detections into a stable driving command.

Raw detector output flickers: a sign may be detected in one frame, missed in the next,
or mis-classified for a single frame. Acting on that directly would make a vehicle brake
and accelerate erratically. This module adds:

* class-specific confidence thresholds: safety-critical classes (red light, stop) need
  less evidence than permissive ones (green light), because missing a red light is worse
  than braking for a false one;
* a size filter that ignores tiny, far-away detections;
* temporal confirmation: a class must appear in at least `min_hits` of the last `window`
  frames before it is trusted (this removes single-frame false positives);
* a priority-based state machine: red light > stop sign > green light > cruise;
* a latched mandatory stop: after a Stop sign the vehicle stays stopped for
  `stop_hold_frames`, and the same sign does not trigger again while it stays in view;
* a persistent speed limit that is remembered after the sign leaves the frame.

The module is pure Python (no torch / OpenCV), so it can be unit-tested and reused with any detector.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum

RED_LIGHT = "Red Light"
GREEN_LIGHT = "Green Light"
STOP_SIGN = "Stop"
_SPEED_RE = re.compile(r"^Speed Limit (\d+)$")


class Action(str, Enum):
    STOP = "STOP"
    PROCEED = "PROCEED"
    CRUISE = "CRUISE"  # nothing new: keep driving within the current speed limit


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class DrivingDecision:
    action: Action
    speed_limit: int | None  # km/h, None if no limit seen yet
    target_speed: int | None  # 0 when stopping, otherwise the speed limit
    reason: str
    confirmed: frozenset[str] = frozenset()

    def __str__(self) -> str:
        limit = f"{self.speed_limit} km/h" if self.speed_limit is not None else "-"
        return f"{self.action.value:<8} | limit {limit:<8} | {self.reason}"


@dataclass
class DecisionConfig:
    default_min_confidence: float = 0.45
    class_min_confidence: dict[str, float] = field(
        default_factory=lambda: {RED_LIGHT: 0.30, STOP_SIGN: 0.35, GREEN_LIGHT: 0.50}
    )
    min_relative_area: float = 0.0004  # box area / frame area; filters far-away objects
    window: int = 8  # frames of history used for confirmation
    min_hits: int = 3  # frames (inside the window) a class must appear in
    stop_hold_frames: int = 30  # ~1 s at 30 FPS

    def __post_init__(self) -> None:
        if not 1 <= self.min_hits <= self.window:
            raise ValueError("need 1 <= min_hits <= window")
        if self.stop_hold_frames < 0:
            raise ValueError("stop_hold_frames must be >= 0")

    def threshold_for(self, label: str) -> float:
        return self.class_min_confidence.get(label, self.default_min_confidence)


def parse_speed_limit(label: str) -> int | None:
    """'Speed Limit 50' -> 50, anything else -> None."""
    match = _SPEED_RE.match(label.strip())
    return int(match.group(1)) if match else None


class DecisionEngine:
    """Stateful: call `update()` once per frame, in order."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        self.reset()

    def reset(self) -> None:
        self._history: deque[frozenset[str]] = deque(maxlen=self.config.window)
        self.speed_limit: int | None = None
        self._stop_hold_remaining = 0
        self._stop_latched = False
        self.frame_index = 0

    # ------------------------------------------------------------------ filtering
    def _is_valid(self, det: Detection, frame_area: float | None) -> bool:
        if det.confidence < self.config.threshold_for(det.label):
            return False
        if frame_area and det.area / frame_area < self.config.min_relative_area:
            return False
        return True

    def _confirmed_labels(self) -> tuple[frozenset[str], Counter[str]]:
        hits: Counter[str] = Counter()
        for labels in self._history:
            hits.update(labels)
        confirmed = frozenset(lbl for lbl, n in hits.items() if n >= self.config.min_hits)
        return confirmed, hits

    # ------------------------------------------------------------------ main step
    def update(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int] | None = None,
    ) -> DrivingDecision:
        """Process one frame. `frame_size` is (width, height), used by the size filter."""
        self.frame_index += 1
        frame_area = float(frame_size[0] * frame_size[1]) if frame_size else None

        frame_labels = frozenset(d.label for d in detections if self._is_valid(d, frame_area))
        self._history.append(frame_labels)
        confirmed, hits = self._confirmed_labels()

        self._update_speed_limit(confirmed, hits)
        action, reason = self._decide_action(confirmed)

        target = 0 if action is Action.STOP else self.speed_limit
        return DrivingDecision(action, self.speed_limit, target, reason, confirmed)

    def _update_speed_limit(self, confirmed: frozenset[str], hits: Counter[str]) -> None:
        candidates = [(hits[lbl], parse_speed_limit(lbl)) for lbl in confirmed]
        candidates = [(n, v) for n, v in candidates if v is not None]
        if not candidates:
            return
        # Most consistently seen sign wins; on a tie take the LOWER limit (safer).
        self.speed_limit = max(candidates, key=lambda nv: (nv[0], -nv[1]))[1]

    def _decide_action(self, confirmed: frozenset[str]) -> tuple[Action, str]:
        stop_visible = STOP_SIGN in confirmed
        if not stop_visible:
            self._stop_latched = False  # sign left the view: the next Stop sign may trigger again
        elif not self._stop_latched:
            self._stop_latched = True
            self._stop_hold_remaining = self.config.stop_hold_frames

        if RED_LIGHT in confirmed:
            if GREEN_LIGHT in confirmed:
                return Action.STOP, "conflicting lights detected: stopping for safety"
            return Action.STOP, "red light"

        if self._stop_hold_remaining > 0:
            self._stop_hold_remaining -= 1
            return Action.STOP, f"mandatory stop ({self._stop_hold_remaining} frames left)"

        if GREEN_LIGHT in confirmed:
            return Action.PROCEED, "green light"

        if self.speed_limit is not None:
            return Action.CRUISE, f"keeping speed limit {self.speed_limit} km/h"
        return Action.CRUISE, "no traffic control detected"
