"""Quantizes PyTorch float32 HalfKA models to int16 weights.

Applies quantization scales QA=255, QB=64, SCALE=400.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from training.model import HalfKANNUE

QA = 255
QB = 64
SCALE = 400.0


def quantize_single_net(model: HalfKANNUE) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Quantizes model weights into int16 numpy arrays."""
    # Feature layer: (1024, 768) float -> (768, 1024) int16
    feat_w = model.feature_transform.weight.detach().cpu().numpy().T
    feat_b = model.feature_transform.bias.detach().cpu().numpy()

    q_feat_w = np.clip(np.round(feat_w * QA), -32768, 32767).astype(np.int16)
    q_feat_b = np.clip(np.round(feat_b * QA), -32768, 32767).astype(np.int16)

    # Output layer: (1, 2048) float -> (2048,) int16
    out_w = model.output_layer.weight.detach().cpu().numpy().flatten()
    out_b = model.output_layer.bias.detach().cpu().item()

    q_out_w = np.clip(np.round(out_w * QB), -32768, 32767).astype(np.int16)
    # output bias scaled by QA * QB / SCALE
    q_out_b = int(np.clip(np.round(out_b * (QA * QB) / SCALE), -32768, 32767))

    return q_feat_w, q_feat_b, q_out_w, q_out_b


def export_3nets(
        mg_model_path: Path,
        eg_model_path: Path,
        sac_model_path: Path,
        output_npz: Path,
) -> None:
    """Combines 3 PyTorch checkpoints into weights_3nets.npz."""
    models = []
    for p in (mg_model_path, eg_model_path, sac_model_path):
        m = HalfKANNUE()
        if p.exists():
            state = torch.load(p, map_location="cpu")
            m.load_state_dict(state)
        models.append(m)

    feat_w_list = []
    feat_b_list = []
    out_w_list = []
    out_b_list = []

    for m in models:
        fw, fb, ow, ob = quantize_single_net(m)
        feat_w_list.append(fw)
        feat_b_list.append(fb)
        out_w_list.append(ow)
        out_b_list.append(ob)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        feature_weights=np.stack(feat_w_list, axis=0),
        feature_biases=np.stack(feat_b_list, axis=0),
        output_weights=np.stack(out_w_list, axis=0),
        output_biases=np.array(out_b_list, dtype=np.int16),
    )
    print(f"Successfully exported 3-net quantized weights to {output_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export quantized 3-net NNUE weights")
    parser.add_argument("--mg", type=str, default="checkpoints/mg.pt")
    parser.add_argument("--eg", type=str, default="checkpoints/eg.pt")
    parser.add_argument("--sac", type=str, default="checkpoints/sac.pt")
    parser.add_argument("--output", type=str, default="weights/weights_3nets.npz")
    args = parser.parse_args()

    export_3nets(Path(args.mg), Path(args.eg), Path(args.sac), Path(args.output))
