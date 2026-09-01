"""PyTorch NNUE Training Pipeline with Metal (MPS) Acceleration.

Trains HalfKA perspective network with blended WDL + MSE Lambda loss,
AdamW optimizer, and CosineAnnealingLR scheduling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.model import HalfKANNUE
from training.nnue_dataset import NNUEDataset, collate_fn


def select_device() -> torch.device:
    """Selects Apple Silicon Metal (MPS) GPU if available, else CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lambda_loss(
        pred_cp: torch.Tensor,
        target_cp: torch.Tensor,
        target_wdl: torch.Tensor,
        lambda_param: float = 0.5,
) -> torch.Tensor:
    """Blended WDL cross-entropy + centipawn MSE loss."""
    pred_wdl = torch.sigmoid(pred_cp / 400.0)
    wdl_loss = nn.functional.binary_cross_entropy(pred_wdl, target_wdl)
    mse_loss = nn.functional.mse_loss(pred_cp, target_cp) / (400.0 ** 2)
    return (1.0 - lambda_param) * wdl_loss + lambda_param * mse_loss


def train_phase_net(
        fens: list[str],
        evals: list[int],
        output_path: Path,
        epochs: int = 10,
        batch_size: int = 4096,
        lr: float = 1e-3,
) -> None:
    device = select_device()
    print(f"Training on device: {device} with {len(fens):,} positions...")

    dataset = NNUEDataset(fens, evals)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = HalfKANNUE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in loader:
            w_feat = batch["white_features"].to(device)
            b_feat = batch["black_features"].to(device)
            stm = batch["stm"].to(device)
            tgt_cp = batch["eval_cp"].to(device)
            tgt_wdl = batch["wdl"].to(device)

            optimizer.zero_grad()
            preds = model(w_feat, b_feat, stm)
            loss = lambda_loss(preds, tgt_cp, tgt_wdl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batches)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.5f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HalfKA NNUE net")
    parser.add_argument("--output", type=str, default="checkpoints/model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    # Synthetic demo training
    dummy_fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 1000
    dummy_evals = [20] * 1000
    train_phase_net(
        dummy_fens,
        dummy_evals,
        Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
