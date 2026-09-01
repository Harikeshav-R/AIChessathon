"""Dynamic Time Manager.

Calculates soft and hard time budgets, best-move stability scaling,
and score-drop time allocation for 120s + 0.5s tournament clocks.
"""

from __future__ import annotations

from engine.params import (
    BM_FACTOR1,
    NODE_TM_FACTOR1,
    NODE_TM_FACTOR2,
    SCORE_DROP_DIV,
    SCORE_DROP_MAX,
    SCORE_DROP_MIN,
)


class TimeManager:
    """Calculates search time limits based on remaining clock, nodes, and stability."""

    def __init__(self, move_overhead_ms: int = 15) -> None:
        self.move_overhead_ms = move_overhead_ms

    def allocate_time(self, time_left_ms: int, inc_ms: int = 100) -> tuple[int, int]:
        """Calculates (soft_time_ms, max_time_ms) for a move."""
        usable_time = max(2, time_left_ms - self.move_overhead_ms)

        # Scale with clock: 1/28th of remaining time + 50% of increment
        soft_time = int(usable_time / 28.0 + inc_ms * 0.5)
        soft_time = max(2, min(soft_time, int(usable_time * 0.35)))

        # Hard ceiling (never exceed 50% of total remaining time)
        max_time = max(soft_time, min(int(usable_time * 0.5), int(soft_time * 2.5)))
        return max(2, soft_time), max(2, max_time)

    @staticmethod
    def adjust_soft_limit(
            original_opt: int,
            max_time: int,
            best_move_nodes: int,
            total_nodes: int,
            stability: int,
            score: int,
            prev_score: int,
    ) -> int:
        """Adjusts soft time limit dynamically based on root search progress."""
        if total_nodes <= 0 or original_opt <= 0:
            return original_opt

        fract = float(best_move_nodes) / float(total_nodes)
        factor = (NODE_TM_FACTOR1 / 100.0 - fract) * NODE_TM_FACTOR2 / 100.0
        bm_factor = BM_FACTOR1 / 100.0 - (stability * 0.06)

        # Score-drop escalation
        score_drop = max(
            float(SCORE_DROP_MIN) / 100.0,
            min(
                float(SCORE_DROP_MAX) / 100.0,
                1.0 + (prev_score - score) / float(SCORE_DROP_DIV),
            ),
        )

        adjusted = int(original_opt * factor * bm_factor * score_drop)
        return max(2, min(adjusted, max_time))
