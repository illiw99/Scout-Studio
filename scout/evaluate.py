"""Scoring detected plays against a hand-marked Shotcut project.

Matching is by overlap in time: a detection and a marked play correspond when
their intervals overlap at all, and each is used once, best overlap first.  The
point of the numbers is to make tuning honest - a change that finds two more
plays but invents five is not an improvement, and only a score says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from .groundtruth import Marked
from .segment import Play


@dataclass
class Match:
    gt: Marked | None
    det: Play | None

    @property
    def kind(self) -> str:
        if self.gt is not None and self.det is not None:
            return "hit"
        return "missed" if self.det is None else "false"

    @property
    def overlap(self) -> float:
        if self.gt is None or self.det is None:
            return 0.0
        return max(0.0, min(self.gt.end, self.det.end)
                   - max(self.gt.start, self.det.start))


@dataclass
class Score:
    matches: list[Match]

    @property
    def hits(self) -> list[Match]:
        return [m for m in self.matches if m.kind == "hit"]

    @property
    def missed(self) -> list[Match]:
        return [m for m in self.matches if m.kind == "missed"]

    @property
    def false_pos(self) -> list[Match]:
        return [m for m in self.matches if m.kind == "false"]

    @property
    def recall(self) -> float:
        n = len(self.hits) + len(self.missed)
        return len(self.hits) / n if n else 0.0

    @property
    def precision(self) -> float:
        n = len(self.hits) + len(self.false_pos)
        return len(self.hits) / n if n else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> str:
        return (f"recall {self.recall:5.1%} ({len(self.hits)}/"
                f"{len(self.hits) + len(self.missed)})   "
                f"precision {self.precision:5.1%} "
                f"({len(self.hits)}/{len(self.hits) + len(self.false_pos)})   "
                f"F1 {self.f1:5.1%}")


def score(gt: list[Marked], det: list[Play]) -> Score:
    pairs = []
    for g in gt:
        for d in det:
            ov = min(g.end, d.end) - max(g.start, d.start)
            if ov > 0:
                pairs.append((ov, g.index, d.index, g, d))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    used_g: set[int] = set()
    used_d: set[int] = set()
    matches: list[Match] = []
    for ov, gi, di, g, d in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi); used_d.add(di)
        matches.append(Match(g, d))
    matches += [Match(g, None) for g in gt if g.index not in used_g]
    matches += [Match(None, d) for d in det if d.index not in used_d]
    matches.sort(key=lambda m: (m.gt.start if m.gt else m.det.start))
    return Score(matches)
