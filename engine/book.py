"""Polyglot Opening Book Reader with In-Memory Caching.

Probes binary Polyglot opening books for instant theoretical moves with 0 search time.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.polyglot


class PolyglotBook:
    """Reads and caches Polyglot .bin opening books in memory."""

    def __init__(self, book_path: Path) -> None:
        self.book_path = book_path
        self._entries: dict[int, list[tuple[chess.Move, int]]] = {}
        self._load()

    def _load(self) -> None:
        """Loads all binary book entries into an in-memory hash table for O(1) lookups."""
        if not self.book_path.exists():
            return

        try:
            with chess.polyglot.open_reader(str(self.book_path)) as reader:
                for entry in reader:
                    key = entry.key
                    if key not in self._entries:
                        self._entries[key] = []
                    self._entries[key].append((entry.move, entry.weight))
        except Exception:
            pass

    def probe(self, fen: str) -> str | None:
        """Probes opening book for position FEN using weighted choice."""
        if not self._entries:
            return None

        try:
            board = chess.Board(fen)
            key = chess.polyglot.zobrist_hash(board)
            if key not in self._entries:
                return None

            moves_and_weights = self._entries[key]
            if not moves_and_weights:
                return None

            moves = [m for m, _ in moves_and_weights]
            weights = [w for _, w in moves_and_weights]
            selected_move = random.choices(moves, weights=weights, k=1)[0]
            return selected_move.uci()
        except Exception:
            return None
