from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import math
from typing import Optional

import torch
from torch.utils.data import DataLoader


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gene_embeddings_tsv(gene_emb_tsv: str) -> Tuple[List[str], torch.Tensor]:
    df = pd.read_csv(gene_emb_tsv, sep="\t")
    genes = df["gene"].astype(str).tolist()
    mat = df.drop(columns=["gene"]).to_numpy().astype("float32")
    n_genes, d_mat = mat.shape
    print(f"Loaded gene embeddings for {n_genes} genes and {d_mat} dimensions")
    return genes, torch.from_numpy(mat)


def intersect_genes_in_embedding_order(
    genes_from_emb: List[str],
    genes_expr: List[str],
) -> List[str]:
    expr_set = set(genes_expr)
    genes_common = [g for g in genes_from_emb if g in expr_set]
    if len(genes_common) == 0:
        raise ValueError("No overlapping genes between embeddings and expression data.")
    print(f"Using {len(genes_common)} genes in common between embeddings and expression")
    return genes_common


def subset_embeddings(
    genes_from_emb: List[str],
    emb_matrix: torch.Tensor,
    genes_common: List[str],
) -> torch.Tensor:
    idx_map = {g: i for i, g in enumerate(genes_from_emb)}
    idx = [idx_map[g] for g in genes_common]
    return emb_matrix[idx, :]


def save_vae_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    genes_common: List[str],
    config_snapshot: Dict[str, Any],
    gene_emb_source: str,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "genes_common": genes_common,
        "config": config_snapshot,
        "gene_emb_source": gene_emb_source,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_vae_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    incompat = model.load_state_dict(ckpt["model_state_dict"], strict=bool(strict))
    ckpt["missing_keys"] = list(getattr(incompat, "missing_keys", []))
    ckpt["unexpected_keys"] = list(getattr(incompat, "unexpected_keys", []))
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt

def inv_softplus(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Inverse of softplus for y>0: x = log(exp(y) - 1).
    Numerically stable for small/large y.
    """
    y = torch.clamp(y, min=eps)
    # For large y, exp(y) can overflow; use y + log1p(-exp(-y))
    return y + torch.log1p(-torch.exp(-y))


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


@torch.no_grad()
def estimate_gene_means(
    dataset,
    max_cells: int = 5000,
    batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Estimate per-gene mean counts from up to max_cells (first max_cells in dataset order).
    Returns (G,) float tensor on CPU.
    """
    n = min(max_cells, len(dataset))
    if n <= 0:
        raise ValueError("Dataset is empty; cannot estimate gene means.")

    # Build a deterministic DataLoader over first n cells
    # (assumes dataset order corresponds to adata order; shuffle=False)
    # We'll stop after covering n items.
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    sum_x = None
    total_cells = 0

    for batch in dl:
        x = batch["x_expr"]  # expected shape (B, G)
        b = x.shape[0]
        take = min(b, n - total_cells)
        if take <= 0:
            break

        x = x[:take]

        # Basic sanity: finite & non-negative
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite values found in x_expr during mean estimation.")
        if (x < 0).any():
            raise ValueError("Negative values found in x_expr during mean estimation (ZINB expects counts).")

        x = x.float()

        if sum_x is None:
            sum_x = x.sum(dim=0)
        else:
            sum_x += x.sum(dim=0)

        total_cells += take
        if total_cells >= n:
            break

    if sum_x is None or total_cells == 0:
        raise ValueError("Failed to accumulate gene means (no data batches).")

    mean_x = sum_x / float(total_cells)
    return mean_x.cpu()

@torch.no_grad()
def estimate_gene_means(
    dataset,
    max_cells: int = 5000,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: Optional[torch.device] = None,
    require_integer: bool = True,
) -> torch.Tensor:
    """
    Estimate per-gene mean counts from up to `max_cells` cells of a dataset.

    Assumptions:
      - dataset[i] returns a dict with key "x_expr"
      - x_expr is shape (G,) per item
      - DataLoader collates to (B, G)
      - Values are non-negative counts (may be float dtype but should be integer-valued)

    Returns:
      mean_x: (G,) float32 tensor on CPU
    """
    n = min(int(max_cells), len(dataset))
    if n <= 0:
        raise ValueError("Dataset is empty; cannot estimate gene means.")

    # Deterministic order; do NOT shuffle for reproducibility.
    # Use VAE collate to handle variable-length token_gene_idx tensors (transformer cache mode).
    from cellinguist.data.dataloaders import collate_vae_batch

    dl = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_vae_batch,
    )

    sum_x: Optional[torch.Tensor] = None
    total_cells = 0

    for batch in dl:
        x = batch["x_expr"]  # (B, G)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        bsz = x.shape[0]
        take = min(bsz, n - total_cells)
        if take <= 0:
            break

        x = x[:take]

        # Move to device if requested (usually CPU is fine; this is cheap)
        if device is not None:
            x = x.to(device, non_blocking=True)

        # Sanity checks (high-yield for diagnosing CLI vs interactive differences)
        if not torch.isfinite(x).all():
            bad = x[~torch.isfinite(x)]
            raise ValueError(f"Non-finite values in x_expr during mean estimation. Example: {bad[:5]}")

        if (x < 0).any():
            mn = float(x.min().item())
            raise ValueError(f"Negative values found in x_expr (min={mn}). ZINB expects non-negative counts.")

        if require_integer and not torch.allclose(x, torch.round(x)):
            frac = x - torch.round(x)
            idx = torch.nonzero(frac != 0, as_tuple=False)
            example = x[idx[0, 0], idx[0, 1]].item() if idx.numel() > 0 else None
            raise ValueError(
                "Non-integer values detected in x_expr during mean estimation "
                f"(example={example}). Are you accidentally using normalized/log data instead of counts? "
                "If batch_correction_method is active, pass require_integer=False."
            )

        x = x.float()

        if sum_x is None:
            sum_x = x.sum(dim=0)
        else:
            sum_x += x.sum(dim=0)

        total_cells += take
        if total_cells >= n:
            break

    if sum_x is None or total_cells == 0:
        raise ValueError("Failed to accumulate gene means (no batches processed).")

    mean_x = sum_x / float(total_cells)
    return mean_x.detach().cpu()
