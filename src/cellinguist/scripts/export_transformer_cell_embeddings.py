from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    GeneVAE,
    TransformerCellEncoder,
    ZINBExpressionDecoder,
)
from cellinguist.utils.vae_io import load_vae_checkpoint


def build_transformer_vae_from_checkpoint(
    checkpoint_path: str,
    n_genes: int,
    n_conditions: int | None,
    perturbation_dim: int | None,
    perturb_emb_dim: int,
    device: torch.device,
    max_tokens_per_cell_override: int | None = None,
    min_expr_for_token_override: float | None = None,
) -> tuple[GeneVAE, dict]:
    # Note: perturbation_dim/perturb_emb_dim are passed to the decoder only.
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu")
    train_cfg = ckpt_raw.get("config", {})
    encoder_type = str(train_cfg.get("encoder_type", "cbow")).lower()
    if encoder_type != "transformer":
        raise ValueError(
            f"Checkpoint encoder_type is '{encoder_type}', expected 'transformer'."
        )

    latent_dim = int(train_cfg.get("latent_dim", 32))
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    n_hidden_layers = int(train_cfg.get("n_hidden_layers", 2))
    cond_emb_dim = int(train_cfg.get("cond_emb_dim", 16))
    input_transform = str(train_cfg.get("input_transform", "log1p"))
    use_library_size_covariate = bool(train_cfg.get("use_library_size_covariate", False))
    library_size_covariate_eps = float(train_cfg.get("library_size_covariate_eps", 1e-8))

    max_tokens_per_cell = train_cfg.get("max_tokens_per_cell", None)
    if max_tokens_per_cell_override is not None:
        max_tokens_per_cell = int(max_tokens_per_cell_override)
    min_expr_for_token = float(train_cfg.get("min_expr_for_token", 0.0))
    if min_expr_for_token_override is not None:
        min_expr_for_token = float(min_expr_for_token_override)

    encoder = TransformerCellEncoder(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_conditions=n_conditions,
        cond_emb_dim=cond_emb_dim,
        input_transform=input_transform,
        transformer_d_model=int(train_cfg.get("transformer_d_model", 256)),
        transformer_n_heads=int(train_cfg.get("transformer_n_heads", 8)),
        transformer_n_layers=int(train_cfg.get("transformer_n_layers", 4)),
        transformer_ff_mult=int(train_cfg.get("transformer_ff_mult", 4)),
        transformer_dropout=float(train_cfg.get("transformer_dropout", 0.0)),
        token_mlp_hidden_dim=int(train_cfg.get("token_mlp_hidden_dim", 256)),
        token_mlp_layers=int(train_cfg.get("token_mlp_layers", 2)),
        max_tokens_per_cell=max_tokens_per_cell,
        min_expr_for_token=min_expr_for_token,
    )
    decoder = ZINBExpressionDecoder(
        n_genes=n_genes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_conditions=n_conditions,
        cond_emb_dim=cond_emb_dim,
        perturbation_dim=perturbation_dim,
        perturb_emb_dim=perturb_emb_dim,
        use_library_size_covariate=use_library_size_covariate,
        library_size_covariate_eps=library_size_covariate_eps,
    )
    model = GeneVAE(encoder, decoder).to(device)
    ckpt = load_vae_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
        strict=False,
    )
    dropped_adv_keys = [
        k for k in ckpt.get("unexpected_keys", []) if str(k).startswith("batch_adversary.")
    ]
    if dropped_adv_keys:
        print(
            f"[export] INFO: ignored {len(dropped_adv_keys)} batch_adversary keys "
            "from checkpoint during export load."
        )
    model.eval()
    return model, ckpt_raw


def export_transformer_cell_embeddings(
    adata_path: str,
    checkpoint_path: str,
    out_tsv_gz: str,
    gene_key: str = "gene",
    layer: str | None = None,
    cond_key: str | None = None,
    batch_key: str | None = None,
    max_cells: int | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
    backed: bool = True,
    max_tokens_per_cell_override: int | None = None,
    min_expr_for_token_override: float | None = None,
) -> None:
    run_device = torch.device(device)
    if batch_key is not None and cond_key is not None and batch_key != cond_key:
        raise ValueError(
            f"Both batch_key='{batch_key}' and cond_key='{cond_key}' were provided, but differ."
        )
    effective_batch_key = batch_key if batch_key is not None else cond_key

    ckpt_raw = torch.load(checkpoint_path, map_location="cpu")
    genes_common = ckpt_raw.get("genes_common", None)
    if genes_common is None:
        raise ValueError("Checkpoint does not contain 'genes_common'.")
    train_cfg = ckpt_raw.get("config", {})
    perturbation_mode = str(train_cfg.get("perturbation_mode", "none")).lower()
    if perturbation_mode == "categorical" and effective_batch_key is None:
        raise ValueError("categorical perturbation export requires batch_key/cond_key.")
    cytokine_keys = train_cfg.get("cytokine_keys", None)
    cytokine_transform = str(train_cfg.get("cytokine_transform", "log1p"))
    cytokine_missing_policy = str(train_cfg.get("cytokine_missing_policy", "error"))
    batch_correction_method = str(train_cfg.get("batch_correction_method", "none")).lower()
    batch_correction_eps = float(train_cfg.get("batch_correction_eps", 1e-8))
    batch_correction_clip_min = float(train_cfg.get("batch_correction_clip_min", 0.1))
    batch_correction_clip_max = float(train_cfg.get("batch_correction_clip_max", 10.0))
    perturb_emb_dim = int(train_cfg.get("perturb_emb_dim", 32))

    ds = SingleCellVAEDataset(
        adata_or_path=adata_path,
        gene_key=gene_key,
        layer=layer,
        cond_key=effective_batch_key,
        batch_key=effective_batch_key,
        batch_correction_method=batch_correction_method,
        batch_correction_eps=batch_correction_eps,
        batch_correction_clip_min=batch_correction_clip_min,
        batch_correction_clip_max=batch_correction_clip_max,
        perturbation_mode=perturbation_mode,
        cytokine_keys=cytokine_keys,
        cytokine_transform=cytokine_transform,
        cytokine_missing_policy=cytokine_missing_policy,
        gene_order=genes_common,
        transform="none",
        backed=backed,
    )

    n_conditions = len(ds.batch_categories) if ds.batch_categories is not None else None
    perturbation_dim = ds.n_perturb_features if perturbation_mode == "cytokine_vector" else None
    model, _ = build_transformer_vae_from_checkpoint(
        checkpoint_path=checkpoint_path,
        n_genes=ds.n_genes,
        n_conditions=n_conditions,
        perturbation_dim=perturbation_dim,
        perturb_emb_dim=perturb_emb_dim,
        device=run_device,
        max_tokens_per_cell_override=max_tokens_per_cell_override,
        min_expr_for_token_override=min_expr_for_token_override,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(run_device.type == "cuda"),
    )

    if max_cells is None:
        max_cells = len(ds)
    else:
        max_cells = min(int(max_cells), len(ds))

    out_path = Path(out_tsv_gz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = 0
    wrote_header = False

    with torch.inference_mode(), gzip.open(out_path, "wt") as f_out:
        for batch in dl:
            if seen >= max_cells:
                break

            x = batch["x_expr"].to(run_device, non_blocking=True)
            batch_idx = batch.get("batch_idx", None)
            if batch_idx is None:
                batch_idx = batch.get("cond_idx", None)
            if batch_idx is not None:
                batch_idx = batch_idx.to(run_device, non_blocking=True)
            perturb_vec = batch.get("perturb_vec", None)
            if perturb_vec is not None:
                perturb_vec = perturb_vec.to(run_device, non_blocking=True)

            mu_z, _ = model.encode(x, batch_idx, perturb_vec=perturb_vec)

            emb_np = mu_z.detach().cpu().numpy()
            bsz = emb_np.shape[0]
            take = min(bsz, max_cells - seen)
            if take <= 0:
                break

            cell_ids = ds.obs_names[seen : seen + take].tolist()
            df_batch = pd.DataFrame(
                emb_np[:take],
                columns=[f"dim_{i+1}" for i in range(emb_np.shape[1])],
            )
            df_batch.insert(0, "cell_id", cell_ids)
            df_batch.to_csv(f_out, sep="\t", index=False, header=not wrote_header)
            wrote_header = True
            seen += take

    print(f"[export] Wrote Transformer cell embeddings to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Transformer VAE cell embeddings (mu_z) from a VAE checkpoint."
    )
    parser.add_argument("--adata", required=True, help="Path to input .h5ad file.")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint (.ckpt).")
    parser.add_argument("--out", required=True, help="Output .tsv.gz path for cell embeddings.")
    parser.add_argument("--gene-key", default="gene", help="Gene column in adata.var (default: gene).")
    parser.add_argument("--layer", default=None, help="Expression layer in adata.layers (default: X).")
    parser.add_argument("--cond-key", default=None, help="Condition column in adata.obs.")
    parser.add_argument("--batch-key", default=None, help="Batch column in adata.obs (preferred alias).")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Maximum cells to export (default: all cells).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4).")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    parser.add_argument(
        "--max-tokens-per-cell-override",
        type=int,
        default=None,
        help="Optional override to cap tokens per cell at export time for lower memory usage.",
    )
    parser.add_argument(
        "--min-expr-for-token-override",
        type=float,
        default=None,
        help="Optional override for tokenization threshold at export time.",
    )
    parser.add_argument(
        "--backed",
        dest="backed",
        action="store_true",
        help="Read AnnData in backed mode (default).",
    )
    parser.add_argument(
        "--no-backed",
        dest="backed",
        action="store_false",
        help="Read AnnData fully into memory.",
    )
    parser.set_defaults(backed=True)
    args = parser.parse_args()

    export_transformer_cell_embeddings(
        adata_path=args.adata,
        checkpoint_path=args.checkpoint,
        out_tsv_gz=args.out,
        gene_key=args.gene_key,
        layer=args.layer,
        cond_key=args.cond_key,
        batch_key=args.batch_key,
        max_cells=args.max_cells,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        backed=args.backed,
        max_tokens_per_cell_override=args.max_tokens_per_cell_override,
        min_expr_for_token_override=args.min_expr_for_token_override,
    )


if __name__ == "__main__":
    main()
