"""Pixel-derived causal navigation for simple ARC action spaces."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, deque
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NavigationScene:
    actor: tuple[float, float]
    attractors: tuple[tuple[float, float], ...]
    objects: tuple[tuple[int, tuple[float, float]], ...] = ()


def extract_navigation_scene(frame: Any) -> NavigationScene | None:
    value = np.asarray(frame.frame, dtype=np.int16)
    if value.ndim != 3 or value.shape[1] < 4 or value.shape[2] < 4:
        return None
    status_rows = max(1, int(round(value.shape[1] * 0.05)))
    image = value[-1, :-status_rows]
    counts = Counter(int(cell) for cell in image.reshape(-1))
    if len(counts) < 3:
        return None
    field_colors = {color for color, _ in counts.most_common(2)}
    remaining = {
        (row, column)
        for row in range(image.shape[0])
        for column in range(image.shape[1])
        if int(image[row, column]) not in field_colors
    }
    components: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending = deque([start])
        cells = []
        while pending:
            row, column = pending.popleft()
            cells.append((row, column))
            for neighbor in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        if not any(
            row in {0, image.shape[0] - 1} or column in {0, image.shape[1] - 1}
            for row, column in cells
        ):
            components.append(tuple(cells))
    if len(components) < 2:
        return None
    components.sort(key=lambda cells: (len(cells), min(cells)))

    def center(cells: tuple[tuple[int, int], ...]) -> tuple[float, float]:
        return (
            float(sum(column for _, column in cells) / len(cells)),
            float(sum(row for row, _ in cells) / len(cells)),
        )

    actor = center(components[0])
    attractors = tuple(center(cells) for cells in components[1:])
    objects = tuple((len(cells), center(cells)) for cells in components)
    return NavigationScene(actor=actor, attractors=attractors, objects=objects)


class CausalNavigator:
    def __init__(self) -> None:
        self.attempted: set[int] = set()
        self.vectors: dict[int, np.ndarray] = {}
        self.counts: Counter[int] = Counter()
        self.actor_size: int | None = None
        self.target_size: int | None = None

    def observe(
        self,
        action_id: int,
        before: NavigationScene | None,
        after: NavigationScene | None,
    ) -> None:
        action_id = int(action_id)
        self.attempted.add(action_id)
        if before is None or after is None:
            return
        displacement = None
        if before.objects and after.objects:
            after_by_size: dict[int, list[tuple[float, float]]] = {}
            for size, center in after.objects:
                after_by_size.setdefault(size, []).append(center)
            moved = []
            for size, center in before.objects:
                matches = after_by_size.get(size, [])
                if len(matches) != 1:
                    continue
                delta = np.asarray(matches[0], dtype=np.float64) - np.asarray(
                    center, dtype=np.float64
                )
                distance = float(np.linalg.norm(delta))
                if 1e-6 < distance <= 8.0:
                    moved.append((distance, size, delta))
            if self.actor_size is None and moved:
                _, self.actor_size, displacement = max(moved)
            elif self.actor_size is not None:
                match = next((delta for _, size, delta in moved if size == self.actor_size), None)
                displacement = match
        if displacement is None and not before.objects:
            displacement = np.asarray(after.actor, dtype=np.float64) - np.asarray(
                before.actor, dtype=np.float64
            )
        if displacement is None:
            return
        if not np.isfinite(displacement).all() or np.linalg.norm(displacement) > 8.0:
            return
        count = self.counts[action_id]
        previous = self.vectors.get(action_id, np.zeros(2, dtype=np.float64))
        self.vectors[action_id] = (previous * count + displacement) / (count + 1)
        self.counts[action_id] += 1

    def choose(self, scene: NavigationScene, actions: tuple[int, ...]) -> int | None:
        legal = tuple(sorted(set(int(action) for action in actions)))
        unseen = [action for action in legal if action not in self.attempted]
        if unseen:
            return unseen[0]
        if not scene.attractors:
            return None
        actor_point = scene.actor
        attractors = scene.attractors
        target_point = None
        if self.actor_size is not None and scene.objects:
            actor_match = next(
                (center for size, center in scene.objects if size == self.actor_size),
                None,
            )
            if actor_match is not None:
                actor_point = actor_match
                target_objects = tuple(
                    (size, center)
                    for size, center in scene.objects
                    if size != self.actor_size
                )
                if self.target_size is not None:
                    matching_targets = tuple(
                        center for size, center in target_objects if size == self.target_size
                    )
                    if matching_targets:
                        target_point = min(
                            matching_targets,
                            key=lambda point: float(
                                np.linalg.norm(np.asarray(point) - np.asarray(actor_point))
                            ),
                        )
                    else:
                        self.target_size = None
                if target_point is None and target_objects:
                    target_size, target_point = min(
                        target_objects,
                        key=lambda item: float(
                            np.linalg.norm(np.asarray(item[1]) - np.asarray(actor_point))
                        ),
                    )
                    self.target_size = target_size
                attractors = tuple(center for _, center in target_objects)
        if not attractors:
            return None
        actor = np.asarray(actor_point, dtype=np.float64)
        target = target_point or min(
            attractors, key=lambda point: float(np.linalg.norm(np.asarray(point) - actor))
        )
        target_vector = np.asarray(target, dtype=np.float64)
        candidates = []
        for action in legal:
            vector = self.vectors.get(action)
            if vector is None or np.linalg.norm(vector) < 1e-6:
                continue
            distance = float(np.linalg.norm(actor + vector - target_vector))
            candidates.append((distance, action))
        return min(candidates)[1] if candidates else None
