"""Deterministic causal transition frontier for online ARC control."""

from __future__ import annotations

from collections import deque


class TransitionFrontier:
    def __init__(self) -> None:
        self.available: dict[str, tuple[int, ...]] = {}
        self.transitions: dict[tuple[str, int], str] = {}
        self.attempted: set[tuple[str, int]] = set()

    def record_state(self, state: str, available_actions: tuple[int, ...]) -> None:
        self.available[state] = tuple(sorted(set(available_actions)))

    def record_transition(self, before: str, action_id: int, after: str) -> None:
        self.transitions[(before, int(action_id))] = after
        self.attempted.add((before, int(action_id)))

    def next_action(self, state: str) -> int | None:
        for action_id in self.available.get(state, ()):
            if (state, action_id) not in self.attempted:
                return action_id

        pending = deque([(state, None)])
        visited = {state}
        while pending:
            current, first_action = pending.popleft()
            if current != state:
                for action_id in self.available.get(current, ()):
                    if (current, action_id) not in self.attempted:
                        return first_action
            for action_id in self.available.get(current, ()):
                following = self.transitions.get((current, action_id))
                if following is None or following in visited:
                    continue
                visited.add(following)
                pending.append(
                    (following, action_id if first_action is None else first_action)
                )
        return None
