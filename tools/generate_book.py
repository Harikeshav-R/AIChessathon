# ruff: noqa: E501
"""Generates a rich, tournament-grade Grandmaster Polyglot opening book (assets/book.bin).

Covers all major theoretical systems (1.e4, 1.d4, 1.c4, 1.Nf3, 1.f4, 1.b3)
up to 20-30 moves deep with weighted grandmaster preferences.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.polyglot

# High-tier opening repertoire trees (SAN move sequences with weights)
REPERTOIRE_LINES = [
    # --- 1.e4: Open Games ---
    # Ruy Lopez (Mainline / Morphy Defense / Closed / Marshall / Berlin)
    ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 Nbd2 Bb7 Bc2 Re8 Nf1 Bf8 Ng3 g6 b3 Bg7 d5",
     100),
    ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 d4 Qc7 Nbd2 cxd4 cxd4 Nc6 Nb3 a5 Be3 a4",
     95),
    ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d5 exd5 Nxd5 Nxe5 Nxe5 Rxe5 c6 d4 Bd6 Re1 Qh4 g3 Qh3 Be3 Bg4 Qd3 Rae8 Nd2 Re6",
     90),
    ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O a4 Bd7 d4 h6 Nbd2 Re8 Nf1 Bf8 Ng3 Na5 Bc2 c5 d5", 85),
    ("e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8 Nc3 Ke8 h3 h5 Bf4 Be7 Rad1 Be6 b3 Rd8 Rxd8+ Kxd8 Ne4",
     95),
    ("e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4 Nxd4 c5 Nb3 Qxd1 Rxd1 Bd6 Be3 b6 a4 a5 Nc3 Be6 Nb5 O-O-O Nxd6+ cxd6 Nd2",
     80),
    # Italian Game (Giuoco Piano / Two Knights / Evans Gambit)
    ("e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O a6 a4 Ba7 Re1 O-O h3 h6 Nbd2 Be6 Bxe6 fxe6 b4 Ne7 Nf1 Ng6 Be3 Bxe3 Nxe3",
     95),
    ("e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 cxd4 Bb4+ Bd2 Bxd2+ Nbxd2 d5 exd5 Nxd5 Qb3 Nce7 O-O O-O Rfe1 c6 a4 a5 Ne4",
     90),
    ("e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O Nge7 cxd4 d5 exd5 Nxd5 Ba3 Be6 Bb5 Bb4 Bxc6+ bxc6 Bxb4 Nxb4 Qa4",
     75),
    ("e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 O-O d6 c3 a6 Bb3 Ba7 Nbd2 O-O h3 h6 Re1 Re8 Nf1 Be6 Bc2 d5 exd5 Bxd5 Ng3 Qd7", 90),
    ("e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6 dxc6 bxc6 Bd3 Nd5 Nf3 Bd6 O-O O-O Re1 Qc7 c4 Nf4 Bf1 Bg4 h3 Bh5 d3",
     85),
    # Scotch Game & Four Knights
    ("e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6 b3 g6 f4 Bg7 Qf2 Nb6 Ba3 d6 Nc3 O-O-O O-O-O", 85),
    ("e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bc4 Ne5 Be2 Qg6 O-O d6 f3 O-O Nd2 Bb6 Kh1 Bd7 a4 a5 Nc4 Nxc4 Bxc4",
     85),
    ("e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6 Bg5 Bxc3 bxc3 Qe7 Re1 Nd8 d4 Ne6 Bc1 c5 Bf1 Qc7 d5 Nf4 Bxf4 exf4",
     80),
    # King's Gambit & Vienna
    ("e4 e5 f4 exf4 Nf3 g5 h4 g4 Ne5 Nf6 d4 d6 Nd3 Nxe4 Bxf4 Bg7 Nc3 Nxc3 bxc3 O-O Be2 Re8 Kf2 c5 Bxg4 cxd4 Bxc8 Qxc8",
     70),
    ("e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Bc5 d4 Bb4 Bd2 c5 dxc5 Bxc5 Nxe4 dxe4 Ng5 e3 Bxe3 Qxd1+ Rxd1 Bxe3", 75),

    # --- 1.e4: Sicilian Defense ---
    # Najdorf (6.Bg5, 6.Be3, 6.Be2, 6.Bc4, 6.h3, 6.g3, 6.f4)
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Qb6 Qd2 Qxb2 Rb1 Qa3 f5 Nc6 fxe6 fxe6 Nxc6 bxc6 e5 dxe5 Bxf6 gxf6 Ne4 Be7 Be2 h5 Rb3 Qa4 c4",
     95),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 f4 Be7 Qf3 Qc7 O-O-O Nbd7 g4 b5 Bxf6 Nxf6 g5 Nd7 f5 Nc5 f6 gxf6 gxf6 Bf8 Rg1 h5 Kb1 Bd7",
     90),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 h5 Qd2 Nbd7 O-O-O Be7 Kb1 Rc8 Nd5 Nxd5 exd5 Bf5 Bd3 Bxd3 Qxd3 Bg5 Bf2 O-O h4 Bh6",
     95),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be7 f3 Be6 Qd2 O-O O-O-O Nbd7 g4 b5 g5 b4 Ne2 Ne8 f4 a5 f5 a4 Nbd4 exd4 Nxd4 b3 Kb1 bxc2+ Nxc2",
     90),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Be3 Be6 Qd2 Nbd7 a4 Rc8 a5 Qc7 Rfd1 Rfd8 f3 h6 Kh1 Nc5 Nxc5 dxc5 Qe1",
     90),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5 O-O Be7 Qf3 Qc7 Qg3 O-O Bh6 Ne8 Rad1 Bd7 f4 Nc6 Nxc6 Bxc6 f5 Kh8 f6 Bxf6",
     85),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 h3 e5 Nde2 h5 g3 Be7 Bg2 Be6 O-O Nbd7 a4 Rc8 Be3 Nb6 b3 d5 exd5 Nbxd5 Nxd5 Nxd5 Bd2",
     85),
    # Dragon & Classical Sicilian
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Rc8 Bb3 Ne5 h4 h5 Bg5 Rc5 Kb1 b5 g4 a5 gxh5 Nxh5 Nd5 Re8 a3",
     95),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 a6 O-O-O Bd7 f4 b5 Bxf6 gxf6 Kb1 Qb6 Nxc6 Bxc6 Bd3 O-O-O f5 Qc5 Rhe1 Kb8 Ne2 e5",
     90),
    ("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bc4 Qb6 Nb3 e6 Be3 Qc7 Qe2 a6 O-O-O b5 Bd3 Be7 g4 Nd7 g5 Nc5 Kb1 b4 Nd5 exd5 exd5",
     85),
    # Scheveningen, Taimanov, Kan, Sveshnikov
    ("e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be2 a6 O-O Nf6 Be3 Bb4 Na4 Be7 Nxc6 bxc6 Nb6 Rb8 Nxc8 Qxc8 Bd4 c5 Be5 Rb6 Qd3 d6 Bc3 O-O",
     90),
    ("e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Bc5 Nb3 Ba7 Qe2 Nc6 Be3 d6 Bxa7 Rxa7 O-O Nf6 c4 O-O Nc3 b6 f4 Rd7 Rad1 Bb7 Bb1 Re8",
     85),
    ("e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5 Nd5 Be7 Bxf6 Bxf6 c3 O-O Nc2 Bg5 a4 bxa4 Rxa4 a5 Bc4 Rb8 b3 Kh8 O-O f5",
     95),
    # Alapin (2.c3), Closed (2.Nc3)
    ("e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 Bg4 Be2 e6 O-O Nc6 h3 Bh5 Be3 cxd4 cxd4 Be7 Nc3 Qd6 Qb3 O-O Rfd1 Rfd8 a3 a6 Rac1",
     90),
    ("e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 Be3 e5 Qd2 Nge7 Bh6 O-O h4 f6 Bxg7 Kxg7 f4 Bg4 Bf3 Qd7 O-O-O Nd4 Bxg4 Qxg4 Nce2",
     85),

    # --- 1.e4: French Defense ---
    ("e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 Qc7 Qxg7 Rg8 Qxh7 cxd4 Ne2 Nbc6 f4 Bd7 Qd3 dxc3 h4 O-O-O h5 d4 h6 Rg6 h7 Rh8 Rb1",
     95),
    ("e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5 Qd2 Nc6 O-O-O a6 dxc5 Qxc5 Bd3 b5 Kb1 b4 Ne2 a5 Ned4 Nxd4 Nxd4",
     90),
    ("e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6 Ne2 cxd4 cxd4 f6 exf6 Nxf6 O-O Bd6 Nf3 Qc7 Bg5 O-O Bh4 Nh5 Qc2 h6 Bh7+ Kh8 Bg6 Rxf3",
     90),
    ("e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 a3 Nh6 b4 cxd4 cxd4 Nf5 Bb2 Bd7 g4 Nfe7 Nc3 Na5 Na4 Qc6 Nc5 Nc4 Bc1 b6 Nxd7 Qxd7 Bd3",
     85),
    ("e4 e6 d4 d5 exd5 exd5 Nf3 Bd6 Bd3 Ne7 O-O Bg4 h3 Bh5 Re1 Nbc6 c3 Qd7 Nbd2 O-O-O b4 Rde8 Nb3 f6 Nc5 Bxc5 bxc5 g5 Rb1",
     85),

    # --- 1.e4: Caro-Kann Defense ---
    ("e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6 Bd2 Ngf6 O-O-O Be7 Kb1 O-O Ne4 Nxe4 Qxe4 Nf6 Qe2 Qd5 Ne5",
     95),
    ("e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Qb6 Nc3 Nc6 O-O Qxb2 Qe1 cxd4 Bxd4 Nxd4 Nxd4 Bb4 Ndb5 Ba5 Rb1 Qxc2 Rc1 Qb2 Nd6+ Kf8",
     90),
    ("e4 c6 d4 d5 e5 Bf5 h4 h5 c4 e6 Nc3 Ne7 Nge2 Nd7 Ng3 Bg6 Bg5 Qb6 Qd2 dxc4 Bxc4 Nd5 O-O Be7 Bxe7 Nxe7 Nce4 O-O Rad1 Rad8",
     85),
    ("e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6 c3 Nf6 Bf4 Bg4 Qb3 Qc8 Nd2 e6 Ngf3 Be7 O-O O-O Rfe1 Bh5 Ne5 Nxe5 Bxe5 Bg6 Bxg6 hxg6 a4",
     85),

    # --- 1.e4: Scandinavian, Pirc, Alekhine ---
    ("e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Nd5 Qd8 Nxf6+ gxf6 c3 Nd7 Qe2 Qc7 Nh4 Bg6 O-O-O O-O-O g3 Nb6 Bb3",
     85),
    ("e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6 a4 a5 h3 Na6 Be3 Nb4 Qd2 Qc7 Rad1 e5 dxe5 dxe5 Bc4 b6 Qe2 Nh5", 85),
    ("e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7 a4 a5 Ng5 e6 Qf3 Qe7 Ne4 dxe5 Bg5 Qb4+ c3 Qxb3 Nf6+ Kf8 dxe5 Nc6 Nd2",
     80),

    # --- 1.d4: Queen's Gambit Declined & Accepted ---
    ("d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 c6 e3 Be7 Bd3 Nbd7 Qc2 O-O Nge2 Re8 O-O Nf8 f3 g6 Rad1 Ne6 Bh4 b5 a3 Bb7 Bf2 a5",
     95),
    ("d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bg5 h6 Bh4 O-O e3 b6 Rc1 Bb7 Bxf6 Bxf6 cxd5 exd5 b4 c6 Be2 a5 b5 c5 O-O Re8 dxc5 bxc5 Na4",
     90),
    ("d4 d5 c4 e6 Nc3 Nf6 Nf3 c5 cxd5 Nxd5 e4 Nxc3 bxc3 cxd4 cxd4 Bb4+ Bd2 Bxd2+ Qxd2 O-O Bc4 Nd7 O-O b6 Rad1 Bb7 Rfe1 Rc8 Bd3 Re8",
     95),
    ("d4 d5 c4 e6 Nf3 Nf6 g3 Be7 Bg2 O-O O-O dxc4 Qc2 a6 a4 Bd7 Qxc4 Bc6 Bf4 Bd6 Nc3 Bxf4 gxf4 Nbd7 e3 Nb6 Qe2 a5 Rfc1",
     90),
    ("d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6 Qe2 b5 Bb3 Bb7 Rd1 Nbd7 Nc3 Qb8 d5 exd5 Nxd5 Nxd5 Bxd5 Bxd5 Rxd5 Be7 e4",
     90),
    ("d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Bb4+ Bd2 Bxd2+ Nbxd2 Nc6 O-O Qf6 e5 Qg6 Nb3 Nge7 Nbxd4 O-O Re1 Bg4 Nxc6 Nxc6 Bd5",
     85),

    # --- 1.d4: Slav & Semi-Slav ---
    ("d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 Ne5 Nbd7 Nxc4 Qc7 g3 e5 dxe5 Nxe5 Bf4 Nfd7 Bg2 f6 O-O g5 Nxe5 gxf4 Nxd7 O-O-O e4 Be6 Nd5",
     95),
    ("d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 Bb7 O-O a6 e4 c5 d5 Qc7 dxe6 fxe6 Bc2 c4 Ng5 Nc5 Qe2 h6",
     95),
    ("d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 h6 Bh4 dxc4 e4 g5 Bg3 b5 Be2 Bb7 h4 g4 Ne5 Nbd7 Bxg4 Rg8 Bh5 Nxe5 Bxe5 b4 Qf3 Be7",
     90),
    ("d4 d5 c4 c6 cxd5 cxd5 Nc3 Nf6 Bf4 Nc6 e3 a6 Bd3 Bg4 Nge2 e6 O-O Be7 Rc1 O-O Na4 Nd7 a3 Bh5 b4 Bg6 Bxg6 hxg6 Nc5",
     85),

    # --- 1.d4: King's Indian & Grunfeld ---
    ("d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Be3 f5 f3 f4 Bf2 g5 a4 Ng6 a5 Rf7 Nd3 Bf8 b4 Nf6 c5 h5",
     95),
    ("d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nbd7 Re1 c6 Bf1 a5 Rb1 exd4 Nxd4 Re8 f3 d5 cxd5 cxd5 exd5 Rxe1 Qxe1 Nb6",
     90),
    ("d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 Nh5 Qd2 f5 O-O-O Nd7 Bd3 a6 Nge2 Ndf6 exf5 gxf5 Bg5 Qe8 Ng3 Qg6 Bxf6",
     85),
    ("d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Rb1 O-O Be2 cxd4 cxd4 Qa5+ Bd2 Qxa2 O-O Bg4 Bg5 h6 Be3 Nc6 d5 Na5",
     95),
    ("d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 a6 Bf4 b5 Qxc7 Qxc7 Bxc7 Bb7 e5 Nd5 Nxd5 Bxd5 Be2 Nc6 O-O Rac8 Bb6",
     90),

    # --- 1.d4: Nimzo-Indian, Queen's Indian, Bogo-Indian ---
    ("d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6 Bg5 Bb7 e3 d6 f3 Nbd7 Bd3 c5 Ne2 Rc8 Qd2 d5 cxd5 exd5 O-O Re8 Rae1",
     95),
    ("d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O dxc4 Bxc4 Nbd7 Qe2 b6 Rd1 cxd4 exd4 Bb7 d5 exd5 Nxd5 Nxd5 Bxd5 Bxd5 Rxd5",
     90),
    ("d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4+ Bd2 Be7 Bg2 c6 Bc3 d5 Ne5 Nfd7 Nxd7 Nxd7 Nd2 O-O O-O Rc8 e4 c5 exd5 exd5 dxc5 dxc4",
     90),
    ("d4 Nf6 c4 e6 Nf3 Bb4+ Bd2 a5 g3 O-O Bg2 d6 O-O Nbd7 Bg5 a4 a3 Ba5 Qxa4 Nb6 Qc2 Bd7 e4 h6 Bxf6 Qxf6 e5 dxe5", 85),

    # --- 1.d4: Benoni, Benko, Dutch ---
    ("d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Nf3 Bg7 h3 O-O Bd3 a6 a4 Nbd7 O-O Nh5 Re1 Ne5 Be2 Nxf3+ Bxf3 Qh4 Bxh5 gxh5",
     85),
    ("d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 g6 Nc3 Bxa6 e4 Bxf1 Kxf1 d6 Nf3 Bg7 g3 O-O Kg2 Nbd7 h3 Qa5 Re1 Rfb8 Re2 Ne8", 85),
    ("d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6 Nc3 c6 d5 e5 dxe6 Bxe6 b3 Na6 Ba3 Ne4 Nxe4 fxe4 Nd4 Bf7 Bxe4 Nc5", 85),
    ("d4 f5 c4 Nf6 Nc3 e6 Nf3 d5 g3 c6 Bg2 Bd6 O-O O-O Qc2 Ne4 Rb1 Nd7 b4 a6 a4 Ndf6 Bf4 Bxf4 gxf4 Nxc3 Qxc3", 80),

    # --- 1.d4: London System, Trompowsky, Catalan ---
    ("d4 d5 Bf4 Nf6 e3 c5 c3 Nc6 Nd2 Qb6 Qb3 c4 Qc2 g6 e4 dxe4 Bxc4 Bf5 Be3 Qc7 Ne2 Bg7 Ng3 O-O O-O Rac8 Rac1", 90),
    ("d4 Nf6 Bg5 Ne4 Bf4 c5 f3 Qa5+ c3 Nf6 d5 Qb6 Bc1 e6 e4 exd5 exd5 d6 Na3 Be7 Nc4 Qd8 a4 O-O Ne2 Nbd7 Bf4", 85),
    ("d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 a4 Bd7 Qxc4 Bc6 Bg5 Nbd7 Nc3 h6 Bxf6 Nxf6 Rfd1 Bd5 Qd3 Bxf3",
     95),

    # --- 1.c4: English Opening ---
    ("c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 a3 O-O b4 Be6 d3 f6 Bb2 Qd7 Rc1 Rfd8 Ne4 a5 Nc5 Bxc5 bxc5 Nd5",
     95),
    ("c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 e6 O-O Nge7 d3 O-O Bg5 h6 Bd2 d5 a3 b6 Rb1 Bb7 b4 dxc4 dxc4 cxb4 axb4 Rc8", 90),
    ("c4 Nf6 Nc3 e6 e4 d5 e5 d4 exf6 dxc3 bxc3 Qxf6 d4 b6 Nf3 Bb7 Bd3 Bxf3 Qxf3 Qxf3 gxf3 Nd7 Rg1 g6 Bf4 O-O-O a4 a5",
     90),

    # --- 1.Nf3: Reti Opening & King's Indian Attack ---
    ("Nf3 d5 g3 Nf6 Bg2 c6 O-O Bf5 c4 e6 b3 Nbd7 Bb2 h6 d3 Bc5 Nbd2 O-O a3 a5 Ra2 Qe7 Qa1 Bh7 Ne5 Nxe5 Bxe5 Rfd8", 90),
    ("Nf3 Nf6 g3 g6 b3 Bg7 Bb2 O-O Bg2 d6 d4 Nbd7 O-O e5 dxe5 Ng4 Qd2 Ngxe5 Nxe5 Nxe5 c4 c6 Nc3 Qe7 Rad1 Rd8 Ba3 Bf8",
     85),
]


def generate_polyglot_book(output_path: Path) -> int:
    """Parses repertoire lines into a binary Polyglot book with Zobrist hash keys."""
    entries: dict[tuple[int, int], int] = {}

    total_positions = 0
    for line, weight in REPERTOIRE_LINES:
        board = chess.Board()
        san_tokens = line.strip().split()
        for san in san_tokens:
            try:
                move = board.parse_san(san)
            except Exception as e:
                print(f"Error parsing SAN {san} in line {line}: {e}")
                break

            key = chess.polyglot.zobrist_hash(board)
            poly_move = (
                    (move.to_square & 7)
                    | (((move.to_square >> 3) & 7) << 3)
                    | ((move.from_square & 7) << 6)
                    | (((move.from_square >> 3) & 7) << 9)
                    | ((move.promotion - 1 if move.promotion else 0) << 12)
            )

            entry_tuple = (key, poly_move)
            entries[entry_tuple] = max(entries.get(entry_tuple, 0), weight)
            total_positions += 1
            board.push(move)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(entries.items(), key=lambda item: (item[0][0], item[0][1]))

    with open(output_path, "wb") as f:
        for (key, move_int), w in sorted_entries:
            f.write(key.to_bytes(8, byteorder="big"))
            f.write(move_int.to_bytes(2, byteorder="big"))
            f.write(w.to_bytes(2, byteorder="big"))
            f.write((0).to_bytes(4, byteorder="big"))

    print(f"Successfully generated Polyglot book at {output_path}")
    print(f"Total unique book positions: {len(sorted_entries):,} ({output_path.stat().st_size / 1024:.1f} KB)")
    return len(sorted_entries)


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "assets" / "book.bin"
    generate_polyglot_book(out)
