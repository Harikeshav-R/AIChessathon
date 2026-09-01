"""Search Tuning Parameters and Logarithmic LMR Table.

Contains all pruning thresholds, reduction formulas, extension margins,
and the precomputed Late Move Reduction (LMR) table.
"""

from __future__ import annotations

import math

import numpy as np

# --- Search Pruning & Reduction Constants ---
NMP_MIN_DEPTH: int = 3
NMP_BASE: int = 4
NMP_DEPTH_DIV: int = 5
NMP_EVAL_DIV: int = 175

RFP_MARGIN: int = 85
RFP_MAX_DEPTH: int = 10

PROBCUT_MARGIN: int = 250

LMR_BASE: int = 4
LMR_RATIO: int = 20
LMR_MIN_DEPTH: int = 3

LMP_BASE: int = 3
LMP_DEPTH: int = 5

SE_DEPTH: int = 5
SE_DOUBLE_EXT_MARGIN: int = 18
SE_TRIPLE_EXT_MARGIN: int = 126

FP_DEPTH: int = 4
FP_MARGIN: int = 90

IIR_MIN_DEPTH: int = 4

SEE_PRUNING_DEPTH: int = 4
SEE_QUIET_MARGIN: int = -50
SEE_NOISY_MARGIN: int = -90

HIST_BONUS: int = 291
HIST_MAX: int = 2476
HIST_DIV: int = 9818
CORR_WEIGHT: int = 25
ASP_START_WINDOW: int = 20

NODE_TM_FACTOR1: int = 149
NODE_TM_FACTOR2: int = 177
BM_FACTOR1: int = 152
SCORE_DROP_DIV: int = 540
SCORE_DROP_MIN: int = 90
SCORE_DROP_MAX: int = 118

# --- Precomputed Logarithmic LMR Table ---
MAX_DEPTH: int = 128
MAX_MOVES: int = 256
LMR_TABLE = np.zeros((MAX_DEPTH, MAX_MOVES), dtype=np.int32)


def _init_lmr_table() -> None:
    base = float(LMR_BASE) / 10.0
    scale = 10.0 / float(LMR_RATIO)
    for d in range(1, MAX_DEPTH):
        log_d = math.log(float(d))
        for m in range(1, MAX_MOVES):
            log_m = math.log(float(m))
            reduction = base + log_d * log_m * scale
            LMR_TABLE[d, m] = int(reduction)


_init_lmr_table()
