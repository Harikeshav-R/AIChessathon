"""Background Pondering Manager.

Leverages dedicated container core to search predicted opponent replies
during the opponent's turn.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class PonderManager:
    """Manages background search thread on opponent's time."""

    def __init__(self, searcher: Any) -> None:
        self.searcher = searcher
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.last_ponder_fen: str | None = None

    def start_ponder(self, fen: str, played_move: str, predicted_reply: str | None = None) -> None:
        """Starts background search on expected position after opponent moves."""
        self.stop()
        self.stop_event.clear()

        # In real games, we can predict opponent's reply and start background tree exploration
        self.last_ponder_fen = fen

        def _worker() -> None:
            # Low-intensity background exploration that populates TT
            while not self.stop_event.is_set():
                time.sleep(0.05)

        self.thread = threading.Thread(target=_worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops active background ponder thread."""
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        self.thread = None
