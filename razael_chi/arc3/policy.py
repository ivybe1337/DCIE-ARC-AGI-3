"""Official-policy-compatible ARC-AGI-3 action selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import (
    causal_frame_fingerprint,
    encode_frame,
    frame_fingerprint,
    salient_coordinates,
)
from .planning import TransitionFrontier
from .interventions import CausalClickPlanner, ClickTarget, extract_click_targets
from .navigation import CausalNavigator, NavigationScene, extract_navigation_scene
from .world import AdaptiveRotorWorld, RotorWorldConfig


@dataclass
class _PendingTransition:
    action_id: int
    operator_features: np.ndarray
    features: np.ndarray
    causal_fingerprint: str
    simple: bool
    navigation_scene: NavigationScene | None
    click_target: ClickTarget | None
    levels_completed: int


class RotorARC3Policy:
    """Online Rotor policy that consumes only official observations/actions."""

    def __init__(
        self,
        *,
        codebook_size: int = 512,
        width: int = 64,
        seed: int = 735,
        action_type: Any | None = None,
        use_salience: bool = True,
        use_navigation: bool = False,
        use_causal_clicks: bool = False,
        use_failure_adaptive_clicks: bool = False,
    ) -> None:
        if action_type is None:
            from arcengine import GameAction

            action_type = GameAction
        self.action_type = action_type
        self.use_salience = use_salience
        self.use_navigation = use_navigation
        self.use_causal_clicks = use_causal_clicks
        self.use_failure_adaptive_clicks = use_failure_adaptive_clicks
        self.click_failure_epoch = 0
        self.seed = seed
        self.world = AdaptiveRotorWorld(
            RotorWorldConfig(width=width, codebook_size=codebook_size, seed=seed)
        )
        self.pending: _PendingTransition | None = None
        self.episode_key: tuple[str, str] | None = None
        self.tried_coordinates: dict[tuple[str, int], set[tuple[int, int]]] = {}
        self.global_tried_coordinates: dict[int, set[tuple[int, int]]] = {}
        self.state_action_counts: dict[tuple[str, int], int] = {}
        self.frontier = TransitionFrontier()
        self.navigator = CausalNavigator()
        self.click_planner = CausalClickPlanner()
        self.frontier_game_id: str | None = None
        self.blocked_simple_actions: set[tuple[str, int]] = set()

    def _state_name(self, frame: Any) -> str:
        return str(getattr(frame.state, "name", frame.state))

    def _advance_click_regime(self) -> None:
        self.click_failure_epoch = min(2, self.click_failure_epoch + 1)
        self.world = AdaptiveRotorWorld(self.world.config)
        self.tried_coordinates.clear()
        self.global_tried_coordinates.clear()
        self.state_action_counts.clear()

    def _reset_action(self) -> Any:
        return self.action_type.RESET

    def _observe_pending(self, frame: Any, features: np.ndarray) -> None:
        if self.pending is None:
            return
        state_name = self._state_name(frame)
        self.world.observe(
            action_id=self.pending.action_id,
            operator_features=self.pending.operator_features,
            before_features=self.pending.features,
            after_features=features,
            progress_delta=float(frame.levels_completed - self.pending.levels_completed),
            failed=state_name == "GAME_OVER",
        )
        if self.pending.simple:
            after_causal_fingerprint = causal_frame_fingerprint(frame)
            self.frontier.record_transition(
                self.pending.causal_fingerprint,
                self.pending.action_id,
                after_causal_fingerprint,
            )
            if after_causal_fingerprint == self.pending.causal_fingerprint:
                self.blocked_simple_actions.add(
                    (self.pending.causal_fingerprint, self.pending.action_id)
                )
            if self.use_navigation:
                self.navigator.observe(
                    self.pending.action_id,
                    self.pending.navigation_scene,
                    extract_navigation_scene(frame),
                )
        elif not self.pending.simple:
            after_causal_fingerprint = causal_frame_fingerprint(frame)
            progress_delta = int(
                frame.levels_completed - self.pending.levels_completed
            )
            if self.pending.click_target is not None:
                self.click_planner.observe(
                    before_state=self.pending.causal_fingerprint,
                    target=self.pending.click_target,
                    after_state=after_causal_fingerprint,
                    progress_delta=progress_delta,
                    failed=state_name == "GAME_OVER",
                )
            if self.use_failure_adaptive_clicks:
                if state_name == "GAME_OVER":
                    self._advance_click_regime()
        self.pending = None

    def _score_action(self, action_id: int, features: np.ndarray) -> float:
        prediction = self.world.predict(action_id, features)
        effect = float(np.linalg.norm(prediction.residual))
        coverage_bonus = 1.0 / np.sqrt(1.0 + self.world.action_observations(action_id))
        return (
            4.0 * prediction.progress
            + 0.35 * effect
            + 0.75 * prediction.uncertainty
            + 0.75 * coverage_bonus
            - 3.0 * prediction.failure
        )

    def _state_tie_break(self, fingerprint: str, action_id: int) -> float:
        digest = hashlib.sha256(f"{self.seed}:{fingerprint}:{action_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def _coordinate_tie_break(
        self,
        fingerprint: str,
        action_id: int,
        coordinate: tuple[int, int],
    ) -> float:
        digest = hashlib.sha256(
            f"{self.seed}:{fingerprint}:{action_id}:{coordinate[0]}:{coordinate[1]}".encode()
        ).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def _score_coordinate(
        self,
        *,
        action_id: int,
        features: np.ndarray,
        fingerprint: str,
        coordinate: tuple[int, int],
    ) -> float:
        prediction = self.world.predict(
            action_id,
            features,
            operator_features=self._operator_features(coordinate),
        )
        tried = coordinate in self.tried_coordinates.get((fingerprint, action_id), set())
        globally_tried = coordinate in self.global_tried_coordinates.get(action_id, set())
        effect = float(np.linalg.norm(prediction.residual))
        return (
            4.0 * prediction.progress
            + 0.35 * effect
            + 0.75 * prediction.uncertainty
            + (0.75 if not tried else 0.0)
            + (0.25 if not globally_tried else 0.0)
            - 3.0 * prediction.failure
            + 1e-6 * self._coordinate_tie_break(fingerprint, action_id, coordinate)
        )

    @staticmethod
    def _operator_features(coordinate: tuple[int, int]) -> np.ndarray:
        x = float(coordinate[0]) / 63.0
        y = float(coordinate[1]) / 63.0
        return np.array([x, y, x * x, y * y], dtype=np.float64)

    def _configure_complex_action(
        self, action: Any, frame: Any
    ) -> tuple[np.ndarray, ClickTarget | None]:
        if self.use_failure_adaptive_clicks and self.click_failure_epoch == 0:
            fingerprint = frame_fingerprint(frame)
            key = (fingerprint, int(action.value))
            tried = self.tried_coordinates.setdefault(key, set())
            candidates = salient_coordinates(frame)
            coordinate = next(
                (candidate for candidate in candidates if candidate not in tried),
                candidates[0],
            )
            tried.add(coordinate)
            self.global_tried_coordinates.setdefault(int(action.value), set()).add(
                coordinate
            )
            action.set_data(
                {"game_id": frame.game_id, "x": coordinate[0], "y": coordinate[1]}
            )
            return self._operator_features(coordinate), None
        if self.use_causal_clicks and (
            not self.use_failure_adaptive_clicks or self.click_failure_epoch >= 2
        ):
            state = causal_frame_fingerprint(frame)
            targets = extract_click_targets(frame)
            target = self.click_planner.choose(state, targets)
            if target is not None:
                coordinate = target.coordinate
                action.set_data(
                    {"game_id": frame.game_id, "x": coordinate[0], "y": coordinate[1]}
                )
                return self._operator_features(coordinate), target
        if not self.use_salience:
            coordinate = (32, 32)
            action.set_data({"game_id": frame.game_id, "x": 32, "y": 32})
            return self._operator_features(coordinate), None
        fingerprint = frame_fingerprint(frame)
        key = (fingerprint, int(action.value))
        tried = self.tried_coordinates.setdefault(key, set())
        globally_tried = self.global_tried_coordinates.setdefault(int(action.value), set())
        candidates = salient_coordinates(frame)
        features = encode_frame(frame)
        coordinate = max(
            candidates,
            key=lambda candidate: self._score_coordinate(
                action_id=int(action.value),
                features=features,
                fingerprint=fingerprint,
                coordinate=candidate,
            ),
        )
        tried.add(coordinate)
        globally_tried.add(coordinate)
        action.set_data({"game_id": frame.game_id, "x": coordinate[0], "y": coordinate[1]})
        return self._operator_features(coordinate), None

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        del frames
        state_name = self._state_name(latest_frame)
        game_id = str(latest_frame.game_id)
        if self.frontier_game_id != game_id:
            self.frontier = TransitionFrontier()
            self.navigator = CausalNavigator()
            self.click_planner = CausalClickPlanner()
            self.click_failure_epoch = 0
            self.blocked_simple_actions.clear()
            self.frontier_game_id = game_id
        key = (str(latest_frame.game_id), str(getattr(latest_frame, "guid", "")))
        if self.episode_key != key or bool(getattr(latest_frame, "full_reset", False)):
            self.world.reset_state()
            self.pending = None
            self.tried_coordinates.clear()
            self.global_tried_coordinates.clear()
            self.state_action_counts.clear()
            self.episode_key = key

        features = encode_frame(latest_frame)
        self._observe_pending(latest_frame, features)
        if state_name in {"NOT_PLAYED", "GAME_OVER"}:
            return self._reset_action()

        actions = [
            self.action_type.from_id(action_id)
            for action_id in latest_frame.available_actions
            if self.action_type.from_id(action_id) is not self._reset_action()
        ]
        if not actions:
            raise RuntimeError("latest ARC frame has no legal non-reset actions")
        fingerprint = frame_fingerprint(latest_frame)
        simple = all(not action.is_complex() for action in actions)
        selected = None
        causal_fingerprint = causal_frame_fingerprint(latest_frame)
        navigation_scene = extract_navigation_scene(latest_frame) if simple else None
        if simple:
            self.frontier.record_state(
                causal_fingerprint,
                tuple(int(action.value) for action in actions),
            )
            planned_action = (
                self.navigator.choose(
                    navigation_scene,
                    tuple(
                        int(action.value)
                        for action in actions
                        if (causal_fingerprint, int(action.value))
                        not in self.blocked_simple_actions
                    ),
                )
                if self.use_navigation and navigation_scene is not None
                else None
            )
            if planned_action is None:
                planned_action = self.frontier.next_action(causal_fingerprint)
            selected = next(
                (action for action in actions if int(action.value) == planned_action),
                None,
            )
        if selected is None:
            ranked = sorted(
                actions,
                key=lambda action: (
                    -(
                        self._score_action(int(action.value), features)
                        + 0.75 / np.sqrt(1.0 + self.state_action_counts.get((fingerprint, int(action.value)), 0))
                        + 1e-6 * self._state_tie_break(fingerprint, int(action.value))
                    ),
                    action.name,
                ),
            )
            selected = ranked[0]
        state_action_key = (fingerprint, int(selected.value))
        self.state_action_counts[state_action_key] = self.state_action_counts.get(state_action_key, 0) + 1
        operator_features = np.zeros(4, dtype=np.float64)
        click_target = None
        if selected.is_complex():
            operator_features, click_target = self._configure_complex_action(
                selected, latest_frame
            )
        self.pending = _PendingTransition(
            action_id=int(selected.value),
            operator_features=operator_features,
            features=features.copy(),
            causal_fingerprint=causal_fingerprint,
            simple=not selected.is_complex(),
            navigation_scene=navigation_scene,
            click_target=click_target,
            levels_completed=int(latest_frame.levels_completed),
        )
        return selected
