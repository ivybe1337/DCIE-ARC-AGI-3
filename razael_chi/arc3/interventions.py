"""Object-relative causal interventions for parameterized ARC actions."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np


OperatorSignature = tuple[int, int, int, int, int]
InstanceKey = tuple[OperatorSignature, int, int]


@dataclass(frozen=True)
class ClickTarget:
    """A visible intervention target and its translation-invariant type."""

    coordinate: tuple[int, int]
    signature: OperatorSignature

    @property
    def instance_key(self) -> InstanceKey:
        return (self.signature, self.coordinate[0], self.coordinate[1])


@dataclass
class _Evidence:
    attempts: int = 0
    effects: int = 0
    progress: int = 0
    failures: int = 0


def extract_click_targets(frame: Any) -> tuple[ClickTarget, ...]:
    """Compile one center intervention per visible single-color component."""

    value = np.asarray(frame.frame, dtype=np.int16)
    if value.ndim != 3 or min(value.shape) <= 0 or not np.isfinite(value).all():
        raise ValueError("ARC frame must be a finite non-empty [layers, height, width] grid")
    image = value[-1]
    height, width = image.shape
    status_rows = max(1, int(round(height * 0.05)))
    world_height = max(1, height - status_rows)
    world = image[:world_height]
    counts = Counter(int(cell) for cell in world.reshape(-1))
    background = min(counts, key=lambda color: (-counts[color], color))
    remaining = {
        (row, column)
        for row in range(world_height)
        for column in range(width)
        if int(world[row, column]) != background
    }
    components: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    while remaining:
        start = min(remaining)
        color = int(world[start])
        pending = deque([start])
        remaining.remove(start)
        cells: list[tuple[int, int]] = []
        while pending:
            cell = pending.popleft()
            cells.append(cell)
            row, column = cell
            for neighbor in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if neighbor in remaining and int(world[neighbor]) == color:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        components.append((color, tuple(cells)))

    targets: list[ClickTarget] = []
    for color, cells in components:
        rows = [cell[0] for cell in cells]
        columns = [cell[1] for cell in cells]
        box_height = max(rows) - min(rows) + 1
        box_width = max(columns) - min(columns) + 1
        fill = int(round(1000 * len(cells) / (box_height * box_width)))
        center_row = sum(rows) / len(rows)
        center_column = sum(columns) / len(columns)
        selected = min(
            cells,
            key=lambda cell: (
                abs(cell[0] - center_row) + abs(cell[1] - center_column),
                cell,
            ),
        )
        targets.append(
            ClickTarget(
                coordinate=(selected[1], selected[0]),
                signature=(color, len(cells), box_height, box_width, fill),
            )
        )
    targets.sort(
        key=lambda target: (
            counts[target.signature[0]],
            target.signature[1],
            target.signature,
            target.coordinate,
        )
    )
    return tuple(targets)


class CausalClickPlanner:
    """Explore and reuse typed click effects over a deterministic state graph."""

    def __init__(self) -> None:
        self.targets: dict[str, tuple[ClickTarget, ...]] = {}
        self.transitions: dict[tuple[str, InstanceKey], str] = {}
        self.evidence: dict[OperatorSignature, _Evidence] = {}
        self.instance_attempts: Counter[InstanceKey] = Counter()

    def register_state(self, state: str, targets: tuple[ClickTarget, ...]) -> None:
        self.targets[state] = targets

    def observe(
        self,
        *,
        before_state: str,
        target: ClickTarget,
        after_state: str,
        progress_delta: int,
        failed: bool,
    ) -> None:
        self.transitions[(before_state, target.instance_key)] = after_state
        self.instance_attempts[target.instance_key] += 1
        evidence = self.evidence.setdefault(target.signature, _Evidence())
        evidence.attempts += 1
        evidence.effects += int(after_state != before_state)
        evidence.progress += max(0, int(progress_delta))
        evidence.failures += int(failed)

    def _utility(self, target: ClickTarget) -> float:
        evidence = self.evidence.get(target.signature, _Evidence())
        effect_probability = (evidence.effects + 1.0) / (evidence.attempts + 2.0)
        exploration = 0.25 / sqrt(evidence.attempts + 1.0)
        return (
            100.0 * evidence.progress
            + effect_probability
            + exploration
            - 2.0 * evidence.failures
            - float(self.instance_attempts[target.instance_key])
        )

    def _untried(self, state: str) -> tuple[ClickTarget, ...]:
        return tuple(
            target
            for target in self.targets.get(state, ())
            if (state, target.instance_key) not in self.transitions
        )

    def _route(self, state: str) -> InstanceKey | None:
        pending = deque([(state, None)])
        visited = {state}
        while pending:
            current, first = pending.popleft()
            if current != state and self._untried(current):
                return first
            for (source, operator), destination in sorted(
                self.transitions.items(), key=lambda item: item[0]
            ):
                if source != current or destination in visited:
                    continue
                visited.add(destination)
                pending.append((destination, operator if first is None else first))
        return None

    def choose(
        self, state: str, targets: tuple[ClickTarget, ...]
    ) -> ClickTarget | None:
        self.register_state(state, targets)
        untried = self._untried(state)
        if untried:
            return min(
                untried,
                key=lambda target: (-self._utility(target), target.instance_key),
            )
        route = self._route(state)
        if route is not None:
            return next(
                (target for target in targets if target.instance_key == route), None
            )
        effective = tuple(
            target
            for target in targets
            if self.evidence.get(target.signature, _Evidence()).effects > 0
        )
        if not effective:
            return None
        return min(
            effective,
            key=lambda target: (-self._utility(target), target.instance_key),
        )
