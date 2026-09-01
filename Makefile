SHELL := /bin/bash

.PHONY: setup play arena zip gate benchmark

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package --include engine

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

benchmark:
	uv run python -m tools.benchmark_stockfish $(if $(DEPTH),--depth $(DEPTH),--sweep-depths 1,2,4,6,8) $(if $(OPENINGS),--openings $(OPENINGS))
