"""Stockfish Automated Benchmarking & Elo Rating Suite.

Plays balanced test matches against Stockfish at configurable depths, skill levels,
or UCI Elo ratings, computing performance metrics, W/D/L scores, and estimated Elo ratings.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

from agent import get_move

STOCKFISH_SEARCH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "stockfish",
]


def find_stockfish_binary() -> str:
    """Finds the local Stockfish binary executable."""
    for path in STOCKFISH_SEARCH_PATHS:
        resolved = shutil.which(path) or (path if Path(path).exists() else None)
        if resolved and Path(resolved).is_file():
            return resolved
    raise FileNotFoundError(
        "Stockfish binary not found. Please install via `brew install stockfish` or specify path."
    )


# Standard opening positions for balanced 2-game pairing tests
OPENING_POSITIONS = [
    chess.STARTING_FEN,
    # Italian Game
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Sicilian Defense (Open)
    "rnbqkb1r/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4",
    # French Defense (Winawer)
    "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    # Caro-Kann Defense (Advance)
    "rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    # Queen's Gambit Declined
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # King's Indian Defense
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP2BPPP/R1BQK1NR w KQ - 2 6",
    # Ruy Lopez
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    # English Opening (Symmetrical)
    "r1bqkbnr/pp1ppppp/2n5/2p5/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3",
    # Nimzo-Indian Defense
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
]


def play_single_game(
        engine_is_white: bool,
        start_fen: str,
        stockfish_engine: chess.engine.SimpleEngine,
        sf_limit: chess.engine.Limit,
        base_time_ms: int,
        inc_time_ms: int,
        ply_cap: int = 150,
) -> tuple[str, str, float, float]:
    """Plays a single game between our Engine and Stockfish.

    Returns (winner, termination_reason, engine_total_time_s, sf_total_time_s).
    """
    board = chess.Board(start_fen)
    engine_clock = float(base_time_ms)
    sf_clock = float(base_time_ms)

    engine_time_used = 0.0
    sf_time_used = 0.0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        turn = board.turn
        is_engine_turn = (turn == chess.WHITE and engine_is_white) or (
                turn == chess.BLACK and not engine_is_white
        )

        if is_engine_turn:
            t0 = time.perf_counter()
            move_uci = get_move(board.fen(), int(engine_clock))
            elapsed = (time.perf_counter() - t0) * 1000.0
            engine_clock = engine_clock - elapsed + inc_time_ms
            engine_time_used += elapsed / 1000.0

            if engine_clock < 0:
                loser = "black" if engine_is_white else "white"
                return (loser, "flag", engine_time_used, sf_time_used)

            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    loser = "black" if engine_is_white else "white"
                    return (loser, "illegal_move", engine_time_used, sf_time_used)
            except Exception:
                loser = "black" if engine_is_white else "white"
                return (loser, "illegal_format", engine_time_used, sf_time_used)

            board.push(move)
        else:
            t0 = time.perf_counter()
            if sf_limit.depth is not None:
                curr_limit = sf_limit
            else:
                w_clock = (engine_clock if engine_is_white else sf_clock) / 1000.0
                b_clock = (sf_clock if engine_is_white else engine_clock) / 1000.0
                curr_limit = chess.engine.Limit(
                    white_clock=max(0.01, w_clock),
                    black_clock=max(0.01, b_clock),
                    white_inc=inc_time_ms / 1000.0,
                    black_inc=inc_time_ms / 1000.0,
                )
            result = stockfish_engine.play(board, curr_limit)
            elapsed = (time.perf_counter() - t0) * 1000.0
            sf_clock = sf_clock - elapsed + inc_time_ms
            sf_time_used += elapsed / 1000.0

            if sf_clock < 0:
                winner = "white" if engine_is_white else "black"
                return (winner, "flag", engine_time_used, sf_time_used)

            if result.move is None or result.move not in board.legal_moves:
                winner = "white" if engine_is_white else "black"
                return (winner, "stockfish_error", engine_time_used, sf_time_used)

            board.push(result.move)

    # Game conclusion
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        piece_vals = [
            (chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
            (chess.ROOK, 5), (chess.QUEEN, 9),
        ]
        diff = sum(
            (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))) * val
            for pt, val in piece_vals
        )
        if diff > 1:
            res = "white"
        elif diff < -1:
            res = "black"
        else:
            res = "draw"
        return res, "adjudication", engine_time_used, sf_time_used

    winner = (
        "draw"
        if outcome.winner is None
        else ("white" if outcome.winner == chess.WHITE else "black")
    )
    return winner, outcome.termination.name.lower(), engine_time_used, sf_time_used


def calculate_elo_diff(wins: int, draws: int, losses: int) -> tuple[float, float]:
    """Calculates Elo difference and 95% error margin from W/D/L."""
    total = wins + draws + losses
    if total == 0:
        return 0.0, 0.0

    score = (wins + 0.5 * draws) / float(total)
    clamped_score = max(0.01, min(0.99, score))
    elo_diff = -400.0 * math.log10(1.0 / clamped_score - 1.0)

    # Standard error approximation
    dev = math.sqrt(
        (wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2)
        / max(1, total - 1)
    )
    denom = math.log(10) * clamped_score * (1.0 - clamped_score)
    error_margin = 1.96 * dev / math.sqrt(total) * 400.0 / denom

    return elo_diff, error_margin


def run_benchmark(
        stockfish_path: str,
        target_depth: int | None = None,
        target_skill: int | None = None,
        target_elo: int | None = None,
        games_per_pairing: int = 2,
        base_time_ms: int = 10000,
        inc_time_ms: int = 100,
        max_openings: int = 5,
) -> dict[str, Any]:
    """Runs a round-robin benchmark match against a specific Stockfish configuration."""
    sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Configure Stockfish limit mode
    sf_limit = chess.engine.Limit()
    config_desc = ""

    if target_depth is not None:
        sf_limit.depth = target_depth
        config_desc = f"Stockfish (Depth {target_depth})"
    elif target_elo is not None:
        sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo})
        config_desc = f"Stockfish (Elo {target_elo})"
    elif target_skill is not None:
        sf_engine.configure({"Skill Level": target_skill})
        config_desc = f"Stockfish (Skill Level {target_skill})"
    else:
        sf_limit.depth = 6
        config_desc = "Stockfish (Depth 6)"

    openings = OPENING_POSITIONS[:max_openings]
    total_games = len(openings) * games_per_pairing

    print("\n=======================================================")
    print(f"  Benchmarking against: {config_desc}")
    print(f"  Total Games: {total_games} ({len(openings)} openings x 2 colors)")
    print(f"  Time Control: {base_time_ms / 1000:.1f}s + {inc_time_ms / 1000:.1f}s")
    print("=======================================================\n")

    wins = draws = losses = 0
    game_idx = 0
    total_engine_time = 0.0

    for fen in openings:
        # Pairing 1: Engine is White
        game_idx += 1
        winner, term, e_time, _ = play_single_game(
            engine_is_white=True,
            start_fen=fen,
            stockfish_engine=sf_engine,
            sf_limit=sf_limit,
            base_time_ms=base_time_ms,
            inc_time_ms=inc_time_ms,
        )
        total_engine_time += e_time
        if winner == "white":
            wins += 1
            res_str = "WIN (+1)"
        elif winner == "draw":
            draws += 1
            res_str = "DRAW (=)"
        else:
            losses += 1
            res_str = "LOSS (-1)"
        print(f"Game {game_idx:02d}/{total_games:02d} [White]: {res_str} by {term} ({e_time:.1f}s)")

        # Pairing 2: Engine is Black
        game_idx += 1
        winner, term, e_time, _ = play_single_game(
            engine_is_white=False,
            start_fen=fen,
            stockfish_engine=sf_engine,
            sf_limit=sf_limit,
            base_time_ms=base_time_ms,
            inc_time_ms=inc_time_ms,
        )
        total_engine_time += e_time
        if winner == "black":
            wins += 1
            res_str = "WIN (+1)"
        elif winner == "draw":
            draws += 1
            res_str = "DRAW (=)"
        else:
            losses += 1
            res_str = "LOSS (-1)"
        print(f"Game {game_idx:02d}/{total_games:02d} [Black]: {res_str} by {term} ({e_time:.1f}s)")

    sf_engine.quit()

    score_pct = (wins + 0.5 * draws) / total_games * 100.0
    elo_diff, error = calculate_elo_diff(wins, draws, losses)

    print(f"\n--- Match Result vs {config_desc} ---")
    print(f"Score: +{wins} ={draws} -{losses} ({score_pct:.1f}%)")
    print(f"Elo Advantage: {elo_diff:+.1f} ± {error:.1f}")
    print(f"Avg Time Per Game: {total_engine_time / total_games:.2f}s\n")

    return {
        "config": config_desc,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": score_pct,
        "elo_diff": elo_diff,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Engine against Stockfish")
    parser.add_argument("--stockfish", type=str, default=None, help="Path to stockfish binary")
    parser.add_argument("--depth", type=int, default=None, help="Target Stockfish depth")
    parser.add_argument("--skill", type=int, default=None, help="Target Stockfish skill (0..20)")
    parser.add_argument("--elo", type=int, default=None, help="Target Stockfish Elo (1320..3190)")
    parser.add_argument(
        "--sweep-depths",
        type=str,
        default=None,
        help="Comma-separated depths to sweep (e.g. 2,4,6,8,10)",
    )
    parser.add_argument(
        "--sweep-elos",
        type=str,
        default=None,
        help="Comma-separated UCI Elo levels to sweep (e.g. 1500,1800,2000,2200)",
    )
    parser.add_argument(
        "--openings", type=int, default=5, help="Number of openings (each played twice)"
    )
    parser.add_argument("--base-ms", type=int, default=5000, help="Base time per game in ms")
    parser.add_argument("--inc-ms", type=int, default=100, help="Increment time per move in ms")
    args = parser.parse_args()

    sf_path = args.stockfish or find_stockfish_binary()
    print(f"Using Stockfish binary: {sf_path}")

    if args.sweep_depths:
        depths = [int(d.strip()) for d in args.sweep_depths.split(",")]
        summary_results = []
        for d in depths:
            res = run_benchmark(
                stockfish_path=sf_path,
                target_depth=d,
                base_time_ms=args.base_ms,
                inc_time_ms=args.inc_ms,
                max_openings=args.openings,
            )
            summary_results.append(res)

        print("\n=======================================================")
        print("                 BENCHMARK SUMMARY TABLE                ")
        print("=======================================================")
        print(f"{'Opponent':<28} | {'Score':<10} | {'Win Rate':<10} | {'Elo Diff':<12}")
        print("-" * 68)
        for r in summary_results:
            score_str = f"+{r['wins']} ={r['draws']} -{r['losses']}"
            diff_str = f"{r['elo_diff']:>+6.1f} ± {r['error']:.0f}"
            print(
                f"{r['config']:<28} | {score_str:<10} | {r['score_pct']:>6.1f}%   | {diff_str}"
            )
        print("=======================================================\n")
    elif args.sweep_elos:
        elos = [int(e.strip()) for e in args.sweep_elos.split(",")]
        summary_results = []
        for e in elos:
            res = run_benchmark(
                stockfish_path=sf_path,
                target_elo=e,
                base_time_ms=args.base_ms,
                inc_time_ms=args.inc_ms,
                max_openings=args.openings,
            )
            summary_results.append(res)

        print("\n=======================================================")
        print("                 BENCHMARK SUMMARY TABLE                ")
        print("=======================================================")
        print(f"{'Opponent':<28} | {'Score':<10} | {'Win Rate':<10} | {'Elo Diff':<12} | {'Estimated Elo':<14}")
        print("-" * 84)
        for idx, r in enumerate(summary_results):
            score_str = f"+{r['wins']} ={r['draws']} -{r['losses']}"
            diff_str = f"{r['elo_diff']:>+6.1f} ± {r['error']:.0f}"
            est_elo = elos[idx] + r['elo_diff']
            print(
                f"{r['config']:<28} | {score_str:<10} | {r['score_pct']:>6.1f}%   | {diff_str} | ~{est_elo:.0f} Elo"
            )
        print("=======================================================\n")
    else:
        run_benchmark(
            stockfish_path=sf_path,
            target_depth=args.depth,
            target_skill=args.skill,
            target_elo=args.elo,
            base_time_ms=args.base_ms,
            inc_time_ms=args.inc_ms,
            max_openings=args.openings,
        )


if __name__ == "__main__":
    main()
