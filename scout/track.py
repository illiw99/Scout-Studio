"""A small multi-object tracker over cached detections.

Measuring speed from one frame to the next is noisy: in a crowded scrum the
nearest box is not always the same player, and a single missed detection reads
as a teleport.  Keeping identities over time and smoothing each player's
velocity turns a spiky per-frame estimate into something a threshold can be set
against, and it is what any later per-player analysis would need anyway.

Deliberately simple - constant-velocity prediction, greedy association, no
appearance model - because the camera is fixed and the sampling interval short.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field, replace
import math

import numpy as np


@dataclass
class Track:
    id: int
    pos: np.ndarray          # foot point, pixels
    vel: np.ndarray          # pixels / second, smoothed
    h: float                 # apparent height, pixels
    age: int = 0             # frames confirmed
    missed: int = 0
    #: Decaying maximum of recent speed.  Someone who ran at some point in the
    #: last few seconds is a player; someone who never does is a spectator
    #: standing on the touchline, and the two are otherwise indistinguishable.
    recent_max: float = 0.0
    history: list = _field(default_factory=list)

    @property
    def speed(self) -> float:
        """Speed in body-heights per second, comparable near and far."""
        return float(np.linalg.norm(self.vel) / max(self.h, 1.0))


class Tracker:
    def __init__(self, dt: float, gate: float = 1.5, smooth: float = 0.6,
                 max_missed: int = 3, keep_history: bool = False,
                 recent_halflife: float = 8.0):
        self.dt = dt
        self.gate = gate              # in body heights
        self.smooth = smooth          # EMA weight on the new measurement
        self.max_missed = max_missed
        self.keep_history = keep_history
        self.decay = 0.5 ** (dt / recent_halflife)
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, pts: np.ndarray, hs: np.ndarray) -> list[Track]:
        # Predict where each existing track should be now.
        for t in self.tracks:
            t.pos = t.pos + t.vel * self.dt

        if len(pts) and self.tracks:
            tp = np.array([t.pos for t in self.tracks])
            th = np.array([t.h for t in self.tracks])
            d = np.linalg.norm(pts[:, None, :] - tp[None, :, :], axis=2)
            limit = self.gate * np.maximum(hs[:, None], th[None, :])
            d = np.where(d <= limit, d, np.inf)

            taken_t: set[int] = set()
            taken_d: set[int] = set()
            # Cheapest pairs first: an isolated player is matched before the
            # ambiguous ones in a pile.
            order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
            for i, j in order:
                if not np.isfinite(d[i, j]):
                    break
                if int(i) in taken_d or int(j) in taken_t:
                    continue
                taken_d.add(int(i)); taken_t.add(int(j))
                tr = self.tracks[j]
                meas_v = (pts[i] - tr.pos) / self.dt + tr.vel
                tr.vel = self.smooth * meas_v + (1 - self.smooth) * tr.vel
                tr.pos = pts[i]
                tr.h = 0.7 * tr.h + 0.3 * hs[i]
                tr.age += 1
                tr.missed = 0
                if self.keep_history:
                    tr.history.append((float(pts[i][0]), float(pts[i][1])))
        else:
            taken_d = set()
            taken_t = set()

        for j, t in enumerate(self.tracks):
            if j not in taken_t:
                t.missed += 1
                t.vel *= 0.6          # unseen: decay toward a stop
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        for i in range(len(pts)):
            if i not in taken_d:
                self.tracks.append(Track(self._next, pts[i].copy(),
                                         np.zeros(2), float(hs[i])))
                self._next += 1

        for t in self.tracks:
            t.recent_max = max(t.speed, t.recent_max * self.decay)

        return [t for t in self.tracks if t.missed == 0]

    def reset(self) -> None:
        self.tracks.clear()


class TimeScaledTracker(Tracker):
    """Preserve the 5 Hz settings in units of elapsed time at sparse sampling rates."""
    def __init__(self,dt):
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite tracker time step required')
        super().__init__(dt=dt,gate=1.5*max(1.,5*dt),smooth=1-.4**(5*dt),
                         max_missed=max(1,math.ceil(.6/dt-1e-9)))
        self.required=max(1,math.ceil(.4/dt-1e-9))

    def update(self,pts,hs):
        live=super().update(pts,hs)
        # Features require age>=2. Adapt that view without mutating track ages.
        return [replace(track,age=2) if self.required<=track.age<2 else track for track in live]
