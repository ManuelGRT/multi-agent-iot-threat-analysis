# src/latent/alignment.py
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from pytorch_metric_learning import losses, miners
except ImportError as exc:  # pragma: no cover - optional training dependency
    torch = None
    nn = None
    losses = None
    miners = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_training_deps() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError("Install the 'training' extra to use latent alignment") from _IMPORT_ERROR


if nn is not None:
    class FusedProjector(nn.Module):
        def __init__(self, in_dim: int, out_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, out_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return nn.functional.normalize(self.net(x), p=2, dim=-1)

    miner = miners.MultiSimilarityMiner()
    contrastive_loss = losses.NTXentLoss(temperature=0.07)
else:
    class FusedProjector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_training_deps()

    miner = None
    contrastive_loss = None


def contrastive_step(embeddings: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    _require_training_deps()
    hard_pairs = miner(embeddings, labels)
    return contrastive_loss(embeddings, labels, hard_pairs)
