"""Online action-conditioned Rotor codebook world state for ARC-AGI-3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .features import FEATURE_DIM


OPERATOR_FEATURE_DIM = 4


def _normalize(value: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else value.copy()


@dataclass(frozen=True)
class RotorWorldConfig:
    width: int = 64
    codebook_size: int = 512
    actions: int = 7
    blocks: int = 4
    topk: int = 8
    beta: float = 8.0
    merge_similarity: float = 0.97
    operator_merge_distance: float = 0.15
    operator_retrieval_distance: float = 0.30
    learning_rate: float = 0.35
    seed: int = 735

    def __post_init__(self) -> None:
        if self.width < 2 or self.width % 2:
            raise ValueError("Rotor world width must be an even integer >= 2")
        if self.codebook_size < 1:
            raise ValueError("codebook_size must be positive")
        if self.actions < 1 or self.blocks < 1:
            raise ValueError("actions and blocks must be positive")
        if not 1 <= self.topk <= self.codebook_size:
            raise ValueError("topk must be within codebook capacity")
        if self.operator_merge_distance < 0.0:
            raise ValueError("operator_merge_distance must be non-negative")
        if self.operator_retrieval_distance < self.operator_merge_distance:
            raise ValueError("operator retrieval distance must cover the merge distance")


@dataclass(frozen=True)
class RotorPrediction:
    next_features: NDArray[np.float64]
    residual: NDArray[np.float64]
    progress: float
    failure: float
    uncertainty: float
    used_codes: int


class AdaptiveRotorWorld:
    """Isometric history state plus online associative transition codebooks."""

    def __init__(self, config: RotorWorldConfig) -> None:
        self.config = config
        generator = np.random.default_rng(config.seed)
        self.projection = generator.normal(
            0.0, 1.0 / np.sqrt(FEATURE_DIM), size=(FEATURE_DIM, config.width)
        )
        self.operator_projection = generator.normal(
            0.0,
            1.0 / np.sqrt(OPERATOR_FEATURE_DIM),
            size=(OPERATOR_FEATURE_DIM, config.width),
        )
        self.angles = generator.uniform(
            -np.pi / 2,
            np.pi / 2,
            size=(config.actions, config.blocks, config.width // 2),
        )
        self.keys = np.zeros((config.actions, config.codebook_size, config.width), dtype=np.float64)
        self.operator_keys = np.zeros(
            (config.actions, config.codebook_size, OPERATOR_FEATURE_DIM), dtype=np.float64
        )
        self.residuals = np.zeros(
            (config.actions, config.codebook_size, FEATURE_DIM), dtype=np.float64
        )
        self.progress = np.zeros((config.actions, config.codebook_size), dtype=np.float64)
        self.failures = np.zeros((config.actions, config.codebook_size), dtype=np.float64)
        self.counts = np.zeros((config.actions, config.codebook_size), dtype=np.int64)
        self.state = np.zeros(config.width, dtype=np.float64)
        self.state[0] = 1.0

    def reset_state(self) -> None:
        self.state.fill(0.0)
        self.state[0] = 1.0

    def _action_index(self, action_id: int) -> int:
        if not 1 <= int(action_id) <= self.config.actions:
            raise ValueError(f"unsupported ARC action id: {action_id}")
        return int(action_id) - 1

    def _rotate(self, state: NDArray[np.float64], action_index: int) -> NDArray[np.float64]:
        output = state.copy()
        for block in range(self.config.blocks):
            permutation = (np.arange(self.config.width) + block) % self.config.width
            inverse = np.argsort(permutation)
            paired = output[permutation]
            left, right = paired[0::2], paired[1::2]
            cosine = np.cos(self.angles[action_index, block])
            sine = np.sin(self.angles[action_index, block])
            rotated = np.empty_like(paired)
            rotated[0::2] = cosine * left - sine * right
            rotated[1::2] = sine * left + cosine * right
            output = rotated[inverse]
        return output

    def _context(
        self,
        action_index: int,
        frame_features: NDArray[np.float64],
        operator_features: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if frame_features.shape != (FEATURE_DIM,):
            raise ValueError(f"frame_features must have shape ({FEATURE_DIM},)")
        if operator_features is None:
            operator_features = np.zeros(OPERATOR_FEATURE_DIM, dtype=np.float64)
        if operator_features.shape != (OPERATOR_FEATURE_DIM,):
            raise ValueError(
                f"operator_features must have shape ({OPERATOR_FEATURE_DIM},)"
            )
        event = _normalize(frame_features @ self.projection)
        operator = _normalize(operator_features @ self.operator_projection)
        return _normalize(self._rotate(self.state, action_index) + event + operator)

    def predict(
        self,
        action_id: int,
        frame_features: NDArray[np.float64],
        *,
        operator_features: NDArray[np.float64] | None = None,
    ) -> RotorPrediction:
        action_index = self._action_index(action_id)
        used = np.flatnonzero(self.counts[action_index] > 0)
        if used.size == 0:
            return RotorPrediction(
                next_features=frame_features.copy(),
                residual=np.zeros(FEATURE_DIM, dtype=np.float64),
                progress=0.0,
                failure=0.0,
                uncertainty=1.0,
                used_codes=0,
            )
        if operator_features is None:
            operator_features = np.zeros(OPERATOR_FEATURE_DIM, dtype=np.float64)
        context = self._context(action_index, frame_features, operator_features)
        operator_distances = np.linalg.norm(
            self.operator_keys[action_index, used] - operator_features,
            axis=1,
        )
        local = operator_distances <= self.config.operator_retrieval_distance
        if not np.any(local):
            return RotorPrediction(
                next_features=frame_features.copy(),
                residual=np.zeros(FEATURE_DIM, dtype=np.float64),
                progress=0.0,
                failure=0.0,
                uncertainty=1.0,
                used_codes=int(used.size),
            )
        used = used[local]
        operator_distances = operator_distances[local]
        similarities = self.keys[action_index, used] @ context - operator_distances
        count = min(self.config.topk, used.size)
        selected_order = np.argsort(similarities)[-count:]
        selected = used[selected_order]
        logits = self.config.beta * similarities[selected_order]
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        residual = weights @ self.residuals[action_index, selected]
        progress = float(weights @ self.progress[action_index, selected])
        failure = float(weights @ self.failures[action_index, selected])
        operator_confidence = float(
            np.clip(
                1.0 - operator_distances.min() / self.config.operator_retrieval_distance,
                0.0,
                1.0,
            )
        )
        residual *= operator_confidence
        progress *= operator_confidence
        failure *= operator_confidence
        nearest_similarity = float(similarities[selected_order[-1]])
        evidence = float(self.counts[action_index, selected].sum())
        uncertainty = float(np.clip(0.5 * (1.0 - nearest_similarity) + 1.0 / np.sqrt(1.0 + evidence), 0.0, 1.0))
        prediction = frame_features + residual
        if not np.isfinite(prediction).all():
            raise FloatingPointError("non-finite Rotor world prediction")
        return RotorPrediction(prediction, residual, progress, failure, uncertainty, int(used.size))

    def observe(
        self,
        *,
        action_id: int,
        operator_features: NDArray[np.float64] | None = None,
        before_features: NDArray[np.float64],
        after_features: NDArray[np.float64],
        progress_delta: float,
        failed: bool,
    ) -> None:
        action_index = self._action_index(action_id)
        if operator_features is None:
            operator_features = np.zeros(OPERATOR_FEATURE_DIM, dtype=np.float64)
        context = self._context(action_index, before_features, operator_features)
        used = np.flatnonzero(self.counts[action_index] > 0)
        slot: int
        if used.size:
            context_similarities = self.keys[action_index, used] @ context
            operator_distances = np.linalg.norm(
                self.operator_keys[action_index, used] - operator_features,
                axis=1,
            )
            nearest_position = int(np.argmax(context_similarities - operator_distances))
            nearest = int(used[nearest_position])
            slot = (
                nearest
                if context_similarities[nearest_position] >= self.config.merge_similarity
                and operator_distances[nearest_position] <= self.config.operator_merge_distance
                else -1
            )
        else:
            slot = -1
        if slot < 0:
            empty = np.flatnonzero(self.counts[action_index] == 0)
            slot = int(empty[0]) if empty.size else int(np.argmin(self.counts[action_index]))
            self.keys[action_index, slot] = context
            self.operator_keys[action_index, slot] = operator_features
            self.residuals[action_index, slot] = after_features - before_features
            self.progress[action_index, slot] = float(progress_delta)
            self.failures[action_index, slot] = float(failed)
            self.counts[action_index, slot] = 1
        else:
            rate = self.config.learning_rate / np.sqrt(1.0 + self.counts[action_index, slot])
            self.keys[action_index, slot] = _normalize((1.0 - rate) * self.keys[action_index, slot] + rate * context)
            self.operator_keys[action_index, slot] = (
                (1.0 - rate) * self.operator_keys[action_index, slot]
                + rate * operator_features
            )
            self.residuals[action_index, slot] = (
                (1.0 - rate) * self.residuals[action_index, slot]
                + rate * (after_features - before_features)
            )
            self.progress[action_index, slot] = (
                (1.0 - rate) * self.progress[action_index, slot] + rate * progress_delta
            )
            self.failures[action_index, slot] = (
                (1.0 - rate) * self.failures[action_index, slot] + rate * float(failed)
            )
            self.counts[action_index, slot] += 1
        event = _normalize(after_features @ self.projection)
        self.state = _normalize(self._rotate(self.state, action_index) + event)

    def action_observations(self, action_id: int) -> int:
        action_index = self._action_index(action_id)
        return int(self.counts[action_index].sum())
