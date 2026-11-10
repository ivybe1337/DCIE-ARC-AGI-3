"""Deterministic frame features and click candidates for ARC-AGI-3."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from typing import Any

import numpy as np
from numpy.typing import NDArray


FEATURE_DIM = 64


def _array(frame: Any) -> NDArray[np.int16]:
    value = np.asarray(frame.frame, dtype=np.int16)
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] == 0 or value.shape[2] == 0:
        raise ValueError("ARC frame must be a non-empty rectangular [layers, height, width] grid")
    if not np.isfinite(value).all():
        raise ValueError("ARC frame contains non-finite values")
    return value


def frame_fingerprint(frame: Any) -> str:
    value = _array(frame)
    payload = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "cells": value.reshape(-1).tolist(),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def causal_frame_fingerprint(frame: Any) -> str:
    value = _array(frame)
    status_rows = max(1, int(round(value.shape[1] * 0.05)))
    world = value[:, :-status_rows, :] if value.shape[1] > status_rows else value
    payload = {
        "shape": list(world.shape),
        "dtype": str(world.dtype),
        "cells": world.reshape(-1).tolist(),
        "levels_completed": int(getattr(frame, "levels_completed", 0)),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_frame(frame: Any) -> NDArray[np.float64]:
    value = _array(frame)
    layers, height, width = value.shape
    features = np.zeros(FEATURE_DIM, dtype=np.float64)
    row_scale = max(1, height - 1)
    column_scale = max(1, width - 1)
    for color in range(10):
        positions = np.argwhere(value == color)
        base = color * 5
        features[base] = positions.shape[0] / value.size
        if positions.size:
            features[base + 1] = positions[:, 1].mean() / row_scale
            features[base + 2] = positions[:, 2].mean() / column_scale
            features[base + 3] = positions[:, 1].std() / row_scale
            features[base + 4] = positions[:, 2].std() / column_scale

    counts = np.bincount(np.clip(value.reshape(-1), 0, 9), minlength=10).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    horizontal = (value[:, :, 1:] != value[:, :, :-1]).mean() if width > 1 else 0.0
    vertical = (value[:, 1:, :] != value[:, :-1, :]).mean() if height > 1 else 0.0
    border = np.concatenate((value[:, 0, :].reshape(-1), value[:, -1, :].reshape(-1)))
    if height > 2:
        border = np.concatenate((border, value[:, 1:-1, 0].reshape(-1), value[:, 1:-1, -1].reshape(-1)))
    modal = int(np.argmax(counts))
    features[50:] = (
        min(layers, 8) / 8.0,
        min(height, 64) / 64.0,
        min(width, 64) / 64.0,
        float((value != modal).mean()),
        float(-(probabilities * np.log(probabilities)).sum() / np.log(10.0)),
        float(horizontal),
        float(vertical),
        float(np.count_nonzero(counts) / 10.0),
        float(value.mean() / 9.0),
        float(value.std() / 9.0),
        float(value.min() / 9.0),
        float(value.max() / 9.0),
        float((border != modal).mean()),
        1.0,
    )
    if not np.isfinite(features).all():
        raise FloatingPointError("non-finite ARC frame features")
    return features


def salient_coordinates(frame: Any) -> tuple[tuple[int, int], ...]:
    """Return deterministic `(x, y)` click candidates from visible components."""
    value = _array(frame)
    image = value[-1]
    height, width = image.shape
    counts = Counter(int(cell) for cell in image.reshape(-1))
    background = min(counts, key=lambda color: (-counts[color], color))
    remaining = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if int(image[row, column]) != background
    }
    components: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    while remaining:
        start = min(remaining)
        color = int(image[start])
        pending = deque([start])
        remaining.remove(start)
        cells = []
        while pending:
            row, column = pending.popleft()
            cells.append((row, column))
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if neighbor in remaining and int(image[neighbor]) == color:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        components.append((color, tuple(sorted(cells))))

    ordered: list[tuple[int, int]] = []
    components.sort(key=lambda item: (counts[item[0]], len(item[1]), item[0], item[1]))
    for _, cells in components:
        center_row = sum(row for row, _ in cells) / len(cells)
        center_column = sum(column for _, column in cells) / len(cells)
        ranked = sorted(
            cells,
            key=lambda cell: (
                abs(cell[0] - center_row) + abs(cell[1] - center_column),
                cell,
            ),
        )
        ordered.append((ranked[0][1], ranked[0][0]))
        for row, column in (cells[0], cells[-1]):
            ordered.append((column, row))

    if not ordered:
        ordered.append((width // 2, height // 2))
    for y_fraction in (0.25, 0.5, 0.75):
        for x_fraction in (0.25, 0.5, 0.75):
            ordered.append((min(width - 1, int(width * x_fraction)), min(height - 1, int(height * y_fraction))))
    for index in range(64):
        ordered.append(((17 + 37 * index) % width, (11 + 23 * index) % height))
    return tuple(dict.fromkeys(ordered))
