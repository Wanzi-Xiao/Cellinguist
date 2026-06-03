from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from cellinguist.config import GRNConstructConfig, load_yaml
from cellinguist.data.dataloaders import collate_vae_batch
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.models.vae import (
    CBOWCellEncoder,
    GeneVAE,
    PerceiverCellEncoder,
    TransformerCellEncoder,
    ZINBExpressionDecoder,
)
from cellinguist.utils.vae_io import (
    load_gene_embeddings_tsv,
    load_vae_checkpoint,
    subset_embeddings,
)


def _resolve_batch_key(batch_key: Optional[str], cond_key: Optional[str]) -> Optional[str]:
    if batch_key is not None and cond_key is not None and batch_key != cond_key:
        raise ValueError(
            f"Both batch_key='{batch_key}' and cond_key='{cond_key}' were provided, but differ."
        )
    return batch_key if batch_key is not None else cond_key


def _safe_context_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("_") or "context"


def _maybe_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return None


def _load_tf_list(path: str) -> list[str]:
    vals: list[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(s)
    if not vals:
        raise ValueError("tf_list_path is empty.")
    # Stable deduplicate.
    seen = set()
    out = []
    for x in vals:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _build_model_from_checkpoint(
    cfg: GRNConstructConfig,
    ckpt_raw: dict,
    ds: SingleCellVAEDataset,
    device: torch.device,
) -> tuple[GeneVAE, dict, str]:
    train_cfg = ckpt_raw.get("config", {})
    encoder_type = str(train_cfg.get("encoder_type", "cbow")).lower()
    if encoder_type not in {"cbow", "perceiver", "transformer"}:
        raise ValueError(f"Unsupported encoder_type in checkpoint: {encoder_type}")

    n_genes = ds.n_genes
    n_conditions = len(ds.batch_categories) if ds.batch_categories is not None else None
    perturbation_dim = ds.n_perturb_features if train_cfg.get("perturbation_mode", "none") == "cytokine_vector" else None

    latent_dim = int(train_cfg.get("latent_dim", 32))
    hidden_dim = int(train_cfg.get("hidden_dim", 256))
    n_hidden_layers = int(train_cfg.get("n_hidden_layers", 2))
    cond_emb_dim = int(train_cfg.get("cond_emb_dim", 16))
    input_transform = str(train_cfg.get("input_transform", "log1p"))
    freeze_gene_embeddings = bool(train_cfg.get("freeze_gene_embeddings", True))
    perturb_emb_dim = int(train_cfg.get("perturb_emb_dim", 32))

    if encoder_type == "cbow":
        emb_path = ckpt_raw.get("gene_emb_source", None)
        if not emb_path:
            raise ValueError(
                "CBOW checkpoint requires gene_emb_source in checkpoint to build model for GRN."
            )
        genes_from_emb, emb_full = load_gene_embeddings_tsv(emb_path)
        emb = subset_embeddings(genes_from_emb, emb_full, ckpt_raw["genes_common"])
        encoder = CBOWCellEncoder(
            gene_embeddings=emb,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cond_emb_dim,
            freeze_gene_embeddings=freeze_gene_embeddings,
            input_transform=input_transform,
        )
    elif encoder_type == "perceiver":
        encoder = PerceiverCellEncoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cond_emb_dim,
            input_transform=input_transform,
            library_norm=str(train_cfg.get("library_norm", "size_factor")),
            library_norm_target_sum=float(train_cfg.get("library_norm_target_sum", 1e4)),
            library_norm_eps=float(train_cfg.get("library_norm_eps", 1e-8)),
            perceiver_d_model=int(train_cfg.get("perceiver_d_model", 256)),
            perceiver_num_latents=int(train_cfg.get("perceiver_num_latents", 64)),
            perceiver_num_cross_attn_heads=int(train_cfg.get("perceiver_num_cross_attn_heads", 8)),
            perceiver_num_self_attn_heads=int(train_cfg.get("perceiver_num_self_attn_heads", 8)),
            perceiver_num_self_attn_layers=int(train_cfg.get("perceiver_num_self_attn_layers", 4)),
            perceiver_ff_mult=int(train_cfg.get("perceiver_ff_mult", 4)),
            perceiver_dropout=float(train_cfg.get("perceiver_dropout", 0.0)),
        )
    else:
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
            max_tokens_per_cell=train_cfg.get("max_tokens_per_cell", None),
            min_expr_for_token=float(train_cfg.get("min_expr_for_token", 0.0)),
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
        use_library_size_covariate=bool(train_cfg.get("use_library_size_covariate", False)),
        library_size_covariate_eps=float(train_cfg.get("library_size_covariate_eps", 1e-8)),
    )
    model = GeneVAE(encoder, decoder).to(device)
    ckpt = load_vae_checkpoint(
        cfg.checkpoint_path,
        model,
        optimizer=None,
        map_location=device,
        strict=False,
    )
    dropped_adv = [
        k for k in ckpt.get("unexpected_keys", []) if str(k).startswith("batch_adversary.")
    ]
    if dropped_adv:
        print(
            f"[construct-grn] INFO: ignored {len(dropped_adv)} batch_adversary keys "
            "while loading checkpoint."
        )

    model.eval()
    return model, train_cfg, encoder_type


def _row_rank_normalize(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.shape[1] <= 1:
        return out
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(values.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(values.shape[1], dtype=np.int64)[None, :]
    out = ranks.astype(np.float32) / float(values.shape[1] - 1)
    return out


def _robust_scale_rows(values: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.size == 0:
        return out
    q05 = np.quantile(values, 0.05, axis=1, keepdims=True)
    q95 = np.quantile(values, 0.95, axis=1, keepdims=True)
    out = (values - q05) / (q95 - q05 + float(eps))
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def _extract_gene_embedding_matrix(model: GeneVAE) -> Optional[np.ndarray]:
    enc = model.encoder
    emb_mod = getattr(enc, "gene_embedding", None)
    if emb_mod is None:
        return None
    weight = getattr(emb_mod, "weight", None)
    if weight is None:
        return None
    arr = weight.detach().cpu().numpy().astype(np.float32, copy=False)
    return arr


def _load_prior_matrix(
    prior_edges_tsv: Optional[str],
    tf_name_to_i: dict[str, int],
    gene_name_to_i: dict[str, int],
    n_tf: int,
    n_genes: int,
) -> np.ndarray:
    prior = np.zeros((n_tf, n_genes), dtype=np.float32)
    if not prior_edges_tsv:
        return prior
    df = pd.read_csv(prior_edges_tsv, sep="\t")
    needed = {"tf", "target", "prior_weight"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(
            "prior_edges_tsv must contain columns: tf, target, prior_weight"
        )
    for row in df.itertuples(index=False):
        tf = str(getattr(row, "tf"))
        tg = str(getattr(row, "target"))
        w = float(getattr(row, "prior_weight"))
        i = tf_name_to_i.get(tf)
        j = gene_name_to_i.get(tg)
        if i is None or j is None:
            continue
        prior[i, j] = float(np.clip(w, 0.0, 1.0))
    return prior


def _make_loader(
    ds: SingleCellVAEDataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    use_all = indices.shape[0] == len(ds) and np.all(indices == np.arange(len(ds), dtype=np.int64))
    if use_all:
        src = ds
    else:
        src = Subset(ds, indices.tolist())
    return DataLoader(
        src,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_vae_batch,
    )


def _decode_mu(
    model: GeneVAE,
    z: torch.Tensor,
    batch_idx: Optional[torch.Tensor],
    libsize: Optional[torch.Tensor],
    perturb_vec: Optional[torch.Tensor],
) -> torch.Tensor:
    dec_out = model.decoder(z, batch_idx, libsize=libsize, perturb_vec=perturb_vec)
    if isinstance(dec_out, tuple):
        return dec_out[0]
    return dec_out


def _compute_components(
    model: GeneVAE,
    ds: SingleCellVAEDataset,
    indices: np.ndarray,
    tf_indices: np.ndarray,
    cfg: GRNConstructConfig,
    encoder_type: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    t_start = time.time()
    n_tf = int(tf_indices.shape[0])
    n_genes = int(ds.n_genes)

    pin_memory = cfg.device.startswith("cuda")
    dl = _make_loader(
        ds=ds,
        indices=indices,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
    )

    attention_available = (
        encoder_type == "transformer" and hasattr(model.encoder, "forward_with_attention")
    )

    attn_sum = np.zeros((n_tf, n_genes), dtype=np.float64)
    attn_count = np.zeros((n_tf, n_genes), dtype=np.float64)

    effect_sum = np.zeros((n_tf, n_genes), dtype=np.float64)
    effect_cells = np.zeros((n_tf,), dtype=np.int64)
    pos_count = np.zeros((n_tf, n_genes), dtype=np.int64)
    neg_count = np.zeros((n_tf, n_genes), dtype=np.int64)

    baseline_s = 0.0
    perturb_s = 0.0

    with torch.inference_mode():
        for batch in dl:
            x = batch["x_expr"].to(cfg.device, non_blocking=True)
            libsize = batch.get("libsize", None)
            if libsize is not None:
                libsize = libsize.to(cfg.device, non_blocking=True)

            batch_idx = batch.get("batch_idx", None)
            if batch_idx is None:
                batch_idx = batch.get("cond_idx", None)
            if batch_idx is not None:
                batch_idx = batch_idx.to(cfg.device, non_blocking=True)

            perturb_vec = batch.get("perturb_vec", None)
            if perturb_vec is not None:
                perturb_vec = perturb_vec.to(cfg.device, non_blocking=True)

            token_gene_idx = batch.get("token_gene_idx", None)
            token_gene_mask = batch.get("token_gene_mask", None)
            if token_gene_idx is not None:
                token_gene_idx = token_gene_idx.to(cfg.device, non_blocking=True)
            if token_gene_mask is not None:
                token_gene_mask = token_gene_mask.to(cfg.device, non_blocking=True)

            t0 = time.time()
            extras = None
            if attention_available:
                mu_z, _, extras = model.encoder.forward_with_attention(
                    x_expr=x,
                    cond_idx=batch_idx,
                    perturb_vec=perturb_vec,
                    token_gene_idx=token_gene_idx,
                    token_gene_mask=token_gene_mask,
                )
            else:
                mu_z, _ = model.encode(
                    x,
                    batch_idx,
                    perturb_vec=perturb_vec,
                    token_gene_idx=token_gene_idx,
                    token_gene_mask=token_gene_mask,
                )

            mu_base = _decode_mu(
                model=model,
                z=mu_z,
                batch_idx=batch_idx,
                libsize=libsize,
                perturb_vec=perturb_vec,
            )
            baseline_s += (time.time() - t0)

            if attention_available and extras is not None and "attn_weights" in extras:
                # attn_weights: (B, L_layers, H, S, S)
                attn_w = extras["attn_weights"]
                tok_idx = extras["token_gene_idx"]
                tok_mask = extras["token_gene_mask"]

                attn_mean = attn_w.mean(dim=1).mean(dim=1)  # (B, S, S)
                for b in range(attn_mean.shape[0]):
                    ids = tok_idx[b]
                    valid = tok_mask[b]
                    valid_pos = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                    if valid_pos.numel() == 0:
                        continue
                    gene_ids = ids[valid_pos].detach().cpu().numpy().astype(np.int64, copy=False)
                    rowmat = attn_mean[b]
                    for ti, g_tf in enumerate(tf_indices):
                        tf_pos = torch.nonzero(valid & (ids == int(g_tf)), as_tuple=False).squeeze(-1)
                        if tf_pos.numel() == 0:
                            continue
                        row = rowmat[tf_pos].mean(dim=0)
                        vals = row[valid_pos].detach().cpu().numpy().astype(np.float64, copy=False)
                        np.add.at(attn_sum[ti], gene_ids, vals)
                        np.add.at(attn_count[ti], gene_ids, 1.0)

            t1 = time.time()
            for ti, g_tf in enumerate(tf_indices):
                eligible = x[:, int(g_tf)] > 0
                n_eligible = int(eligible.sum().item())
                if n_eligible <= 0:
                    continue

                x_pert = x.clone()
                base_tf = x_pert[:, int(g_tf)]
                delta = torch.maximum(
                    torch.full_like(base_tf, float(cfg.perturb_min_abs)),
                    float(cfg.perturb_frac) * torch.maximum(base_tf, torch.ones_like(base_tf)),
                )
                x_pert[:, int(g_tf)] = base_tf + delta

                mu_z_p, _ = model.encode(
                    x_pert,
                    batch_idx,
                    perturb_vec=perturb_vec,
                    token_gene_idx=token_gene_idx,
                    token_gene_mask=token_gene_mask,
                )
                mu_pert = _decode_mu(
                    model=model,
                    z=mu_z_p,
                    batch_idx=batch_idx,
                    libsize=libsize,
                    perturb_vec=perturb_vec,
                )

                d = torch.log1p(mu_pert) - torch.log1p(mu_base)
                d = d[eligible]

                effect_sum[ti] += d.sum(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
                effect_cells[ti] += int(d.shape[0])

                pos_count[ti] += (
                    (d > float(cfg.sign_eps)).sum(dim=0).detach().cpu().numpy().astype(np.int64, copy=False)
                )
                neg_count[ti] += (
                    (d < -float(cfg.sign_eps)).sum(dim=0).detach().cpu().numpy().astype(np.int64, copy=False)
                )

            perturb_s += (time.time() - t1)

    effect_cells_safe = np.maximum(effect_cells[:, None], 1)
    effect_size = (effect_sum / effect_cells_safe).astype(np.float32, copy=False)
    p_pos = (pos_count / effect_cells_safe).astype(np.float32, copy=False)
    p_neg = (neg_count / effect_cells_safe).astype(np.float32, copy=False)

    sign_code = np.zeros_like(effect_size, dtype=np.int8)
    act = (effect_size > float(cfg.sign_eps)) & (p_pos >= float(cfg.sign_consistency_min))
    rep = (effect_size < -float(cfg.sign_eps)) & (p_neg >= float(cfg.sign_consistency_min))
    sign_code[act] = 1
    sign_code[rep] = -1

    if attention_available:
        with np.errstate(divide="ignore", invalid="ignore"):
            a_raw = np.divide(
                attn_sum,
                np.maximum(attn_count, 1.0),
                out=np.zeros_like(attn_sum, dtype=np.float64),
            ).astype(np.float32, copy=False)
        a_score = _robust_scale_rows(a_raw, eps=float(cfg.eps))
    else:
        a_score = np.zeros((n_tf, n_genes), dtype=np.float32)

    d_raw = np.abs(effect_size)
    d_score = _robust_scale_rows(d_raw, eps=float(cfg.eps))

    timings = {
        "baseline_forward_s": float(baseline_s),
        "perturb_s": float(perturb_s),
        "components_total_s": float(time.time() - t_start),
        "attention_available": bool(attention_available),
    }

    components = {
        "A": a_score,
        "D": d_score,
        "effect_size": effect_size,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "sign_code": sign_code,
        "n_cells_used": int(indices.shape[0]),
    }
    return components, timings


def _build_embedding_score(model: GeneVAE, tf_indices: np.ndarray, n_genes: int) -> np.ndarray:
    emb = _extract_gene_embedding_matrix(model)
    if emb is None:
        return np.zeros((tf_indices.shape[0], n_genes), dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb_n = emb / norms
    cos = emb_n[tf_indices] @ emb_n.T
    e_raw = (cos + 1.0) / 2.0
    return _row_rank_normalize(e_raw.astype(np.float32, copy=False))


def _select_edges(
    score: np.ndarray,
    effect_size: np.ndarray,
    tf_indices: np.ndarray,
    cfg: GRNConstructConfig,
) -> np.ndarray:
    n_tf, n_genes = score.shape
    sel = (score >= float(cfg.score_min)) & (np.abs(effect_size) >= float(cfg.min_abs_effect))

    if not bool(cfg.allow_self_edges):
        for ti, g_tf in enumerate(tf_indices):
            sel[ti, int(g_tf)] = False

    top_k = int(cfg.top_k_per_tf)
    if top_k > 0 and top_k < n_genes:
        for ti in range(n_tf):
            idx = np.where(sel[ti])[0]
            if idx.size <= top_k:
                continue
            row = score[ti, idx]
            keep_local = np.argsort(row)[-top_k:]
            keep = idx[keep_local]
            new_row = np.zeros((n_genes,), dtype=bool)
            new_row[keep] = True
            sel[ti] = new_row
    return sel


def _write_network_outputs(
    out_dir: Path,
    context_name: str,
    genes: list[str],
    tf_names: list[str],
    tf_indices: np.ndarray,
    score: np.ndarray,
    components: dict[str, np.ndarray],
    prior: np.ndarray,
    selected: np.ndarray,
    bootstrap_freq: np.ndarray,
    n_cells_used: int,
) -> tuple[int, float, float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    sign_code = components["sign_code"]
    effect_size = components["effect_size"]

    np.savez_compressed(
        out_dir / "edge_components.npz",
        A=components["A"],
        E=components["E"],
        D=components["D"],
        P=prior,
        S=score,
        effect_size=effect_size,
        sign_code=sign_code,
        tf_names=np.asarray(tf_names, dtype=object),
        gene_names=np.asarray(genes, dtype=object),
    )

    nodes = pd.DataFrame({"gene": genes})
    is_tf = np.zeros((len(genes),), dtype=np.int8)
    is_tf[np.asarray(tf_indices, dtype=np.int64)] = 1
    nodes["is_tf"] = is_tf
    with gzip.open(out_dir / "nodes.tsv.gz", "wt") as f:
        nodes.to_csv(f, sep="\t", index=False)

    rows = []
    for ti, tf in enumerate(tf_names):
        tg_idx = np.where(selected[ti])[0]
        for gj in tg_idx.tolist():
            freq = float(bootstrap_freq[ti, gj])
            if freq >= 0.80:
                conf = "high"
            elif freq >= 0.50:
                conf = "medium"
            else:
                conf = "low"

            code = int(sign_code[ti, gj])
            if code > 0:
                sign = "activation"
            elif code < 0:
                sign = "repression"
            else:
                sign = "ambiguous"

            rows.append(
                {
                    "tf": tf,
                    "target": genes[gj],
                    "score": float(score[ti, gj]),
                    "effect_size": float(effect_size[ti, gj]),
                    "sign": sign,
                    "attention_score": float(components["A"][ti, gj]),
                    "embedding_score": float(components["E"][ti, gj]),
                    "decoder_score": float(components["D"][ti, gj]),
                    "prior_score": float(prior[ti, gj]),
                    "bootstrap_freq": freq,
                    "confidence": conf,
                    "n_cells_used": int(n_cells_used),
                    "context": context_name,
                }
            )

    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.sort_values(["tf", "score"], ascending=[True, False], kind="mergesort")
    with gzip.open(out_dir / "edges.tsv.gz", "wt") as f:
        edges.to_csv(f, sep="\t", index=False)

    if edges.empty:
        frac_act = 0.0
        frac_rep = 0.0
        frac_amb = 0.0
    else:
        frac_act = float((edges["sign"] == "activation").mean())
        frac_rep = float((edges["sign"] == "repression").mean())
        frac_amb = float((edges["sign"] == "ambiguous").mean())

    return int(len(edges)), frac_act, frac_rep, frac_amb


def _run_single_network(
    context_name: str,
    out_dir: Path,
    model: GeneVAE,
    ds: SingleCellVAEDataset,
    indices: np.ndarray,
    tf_names: list[str],
    tf_indices: np.ndarray,
    gene_name_to_i: dict[str, int],
    cfg: GRNConstructConfig,
    encoder_type: str,
) -> dict:
    t_start = time.time()

    components, timing_components = _compute_components(
        model=model,
        ds=ds,
        indices=indices,
        tf_indices=tf_indices,
        cfg=cfg,
        encoder_type=encoder_type,
    )

    e_score = _build_embedding_score(model=model, tf_indices=tf_indices, n_genes=ds.n_genes)
    components["E"] = e_score

    tf_name_to_i = {t: i for i, t in enumerate(tf_names)}
    prior = _load_prior_matrix(
        prior_edges_tsv=cfg.prior_edges_tsv,
        tf_name_to_i=tf_name_to_i,
        gene_name_to_i=gene_name_to_i,
        n_tf=len(tf_names),
        n_genes=ds.n_genes,
    )

    wa = float(cfg.w_a)
    we = float(cfg.w_e)
    wd = float(cfg.w_d)
    wp = float(cfg.w_p)
    if not bool(timing_components["attention_available"]):
        wa = 0.0
        s = we + wd + wp
        if s <= 0:
            raise ValueError("At least one of w_e, w_d, w_p must be positive when attention is unavailable.")
        we, wd, wp = we / s, wd / s, wp / s

    score = (
        wa * components["A"]
        + we * components["E"]
        + wd * components["D"]
        + wp * prior
    ).astype(np.float32, copy=False)

    selected = _select_edges(
        score=score,
        effect_size=components["effect_size"],
        tf_indices=tf_indices,
        cfg=cfg,
    )

    bootstrap_start = time.time()
    if int(cfg.bootstrap_iters) > 0:
        counts = np.zeros_like(score, dtype=np.int64)
        n = int(indices.shape[0])
        n_draw = max(1, int(round(float(cfg.bootstrap_cell_frac) * n)))
        rng = np.random.default_rng(int(cfg.seed) + 991)
        for b in range(int(cfg.bootstrap_iters)):
            sampled = rng.choice(indices, size=n_draw, replace=True).astype(np.int64)
            boot_components, _ = _compute_components(
                model=model,
                ds=ds,
                indices=sampled,
                tf_indices=tf_indices,
                cfg=cfg,
                encoder_type=encoder_type,
            )
            boot_components["E"] = e_score
            boot_score = (
                wa * boot_components["A"]
                + we * boot_components["E"]
                + wd * boot_components["D"]
                + wp * prior
            ).astype(np.float32, copy=False)
            boot_selected = _select_edges(
                score=boot_score,
                effect_size=boot_components["effect_size"],
                tf_indices=tf_indices,
                cfg=cfg,
            )
            counts += boot_selected.astype(np.int64)
            print(
                f"[construct-grn] context={context_name} bootstrap {b + 1}/{int(cfg.bootstrap_iters)} complete",
                flush=True,
            )
        bootstrap_freq = (counts / float(cfg.bootstrap_iters)).astype(np.float32, copy=False)
    else:
        bootstrap_freq = selected.astype(np.float32)

    edge_count, frac_act, frac_rep, frac_amb = _write_network_outputs(
        out_dir=out_dir,
        context_name=context_name,
        genes=ds.gene_order,
        tf_names=tf_names,
        tf_indices=tf_indices,
        score=score,
        components=components,
        prior=prior,
        selected=selected,
        bootstrap_freq=bootstrap_freq,
        n_cells_used=int(indices.shape[0]),
    )

    qc = {
        "context": context_name,
        "n_cells_total": int(ds.n_cells),
        "n_cells_used": int(indices.shape[0]),
        "n_genes": int(ds.n_genes),
        "n_tfs_matched": int(tf_indices.shape[0]),
        "edge_count_final": int(edge_count),
        "fraction_activation": float(frac_act),
        "fraction_repression": float(frac_rep),
        "fraction_ambiguous": float(frac_amb),
        "median_bootstrap_freq": float(np.median(bootstrap_freq[selected])) if edge_count > 0 else 0.0,
        "attention_available": bool(timing_components["attention_available"]),
        "baseline_forward_s": float(timing_components["baseline_forward_s"]),
        "perturb_s": float(timing_components["perturb_s"]),
        "bootstrap_s": float(time.time() - bootstrap_start),
        "total_s": float(time.time() - t_start),
    }

    with (out_dir / "qc_metrics.json").open("w") as f:
        json.dump(qc, f, indent=2, sort_keys=True)

    return qc


def construct_grn(cfg: GRNConstructConfig) -> None:
    t0 = time.time()
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    device = torch.device(cfg.device)
    ckpt_raw = torch.load(cfg.checkpoint_path, map_location="cpu")
    genes_common = ckpt_raw.get("genes_common", None)
    if genes_common is None:
        raise ValueError("Checkpoint does not contain 'genes_common'.")

    train_cfg = ckpt_raw.get("config", {})
    perturbation_mode = str(train_cfg.get("perturbation_mode", "none")).lower()
    cytokine_keys = train_cfg.get("cytokine_keys", None)
    cytokine_transform = str(train_cfg.get("cytokine_transform", "log1p")).lower()
    cytokine_missing_policy = str(train_cfg.get("cytokine_missing_policy", "error")).lower()

    effective_batch_key = _resolve_batch_key(cfg.batch_key, cfg.cond_key)
    if perturbation_mode == "categorical" and effective_batch_key is None:
        raise ValueError("categorical perturbation checkpoint requires batch_key/cond_key.")

    batch_correction_method = str(train_cfg.get("batch_correction_method", "none")).lower()
    batch_correction_eps = float(train_cfg.get("batch_correction_eps", 1e-8))
    batch_correction_clip_min = float(train_cfg.get("batch_correction_clip_min", 0.1))
    batch_correction_clip_max = float(train_cfg.get("batch_correction_clip_max", 10.0))

    encoder_type = str(train_cfg.get("encoder_type", "cbow")).lower()
    use_transformer_in_memory_precompute = (
        encoder_type == "transformer"
        and (not cfg.token_index_cache_dir)
        and bool(cfg.transformer_precompute_token_indices)
    )

    ds = SingleCellVAEDataset(
        adata_or_path=cfg.adata_path,
        gene_key=cfg.gene_key,
        layer=cfg.layer,
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
        backed=bool(cfg.backed),
        precompute_token_gene_indices=use_transformer_in_memory_precompute,
        token_min_expr=float(train_cfg.get("min_expr_for_token", 0.0)),
        token_max_genes=train_cfg.get("max_tokens_per_cell", None),
        token_index_cache_dir=(cfg.token_index_cache_dir if encoder_type == "transformer" else None),
        token_index_cache_require=(encoder_type == "transformer" and bool(cfg.token_index_cache_require)),
    )

    model, train_cfg_resolved, encoder_type = _build_model_from_checkpoint(
        cfg=cfg,
        ckpt_raw=ckpt_raw,
        ds=ds,
        device=device,
    )

    tf_requested = _load_tf_list(cfg.tf_list_path)
    gene_name_to_i = {g: i for i, g in enumerate(ds.gene_order)}

    tf_names: list[str] = []
    tf_indices: list[int] = []
    tf_missing: list[str] = []
    for tf in tf_requested:
        idx = gene_name_to_i.get(tf)
        if idx is None:
            tf_missing.append(tf)
            continue
        tf_names.append(tf)
        tf_indices.append(int(idx))

    if not tf_indices:
        raise ValueError("No TFs from tf_list_path were matched to checkpoint genes.")

    tf_indices_np = np.asarray(tf_indices, dtype=np.int64)

    all_idx = np.arange(len(ds), dtype=np.int64)
    if cfg.max_cells is not None and int(cfg.max_cells) < len(ds):
        rng = np.random.default_rng(int(cfg.max_cells_seed))
        selected_idx = np.sort(
            rng.choice(all_idx, size=int(cfg.max_cells), replace=False).astype(np.int64)
        )
    else:
        selected_idx = all_idx

    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[construct-grn] encoder_type={encoder_type} n_cells={len(ds)} "
        f"selected_cells={selected_idx.shape[0]} n_genes={ds.n_genes} n_tfs={len(tf_indices)}",
        flush=True,
    )

    qc_all = []
    global_dir = out_root / "global"
    qc_global = _run_single_network(
        context_name="global",
        out_dir=global_dir,
        model=model,
        ds=ds,
        indices=selected_idx,
        tf_names=tf_names,
        tf_indices=tf_indices_np,
        gene_name_to_i=gene_name_to_i,
        cfg=cfg,
        encoder_type=encoder_type,
    )
    qc_all.append(qc_global)

    if cfg.context_key:
        obs = ds.adata.obs
        if cfg.context_key not in obs.columns:
            raise ValueError(
                f"context_key '{cfg.context_key}' not found in adata.obs. "
                f"Available columns: {list(obs.columns)}"
            )
        ctx_vals_all = obs[cfg.context_key].astype(str).to_numpy()
        ctx_vals_sel = ctx_vals_all[selected_idx]
        unique_ctx = pd.unique(ctx_vals_sel)
        for ctx in unique_ctx.tolist():
            local_mask = (ctx_vals_sel == ctx)
            local_idx = selected_idx[local_mask]
            if local_idx.shape[0] < int(cfg.min_cells_per_context):
                continue
            ctx_dir = out_root / "contexts" / _safe_context_name(ctx)
            qc_ctx = _run_single_network(
                context_name=str(ctx),
                out_dir=ctx_dir,
                model=model,
                ds=ds,
                indices=local_idx,
                tf_names=tf_names,
                tf_indices=tf_indices_np,
                gene_name_to_i=gene_name_to_i,
                cfg=cfg,
                encoder_type=encoder_type,
            )
            qc_all.append(qc_ctx)

    run_metadata = {
        "config": asdict(cfg),
        "train_config": train_cfg_resolved,
        "encoder_type": encoder_type,
        "genes_common_n": len(genes_common),
        "n_tfs_requested": len(tf_requested),
        "n_tfs_matched": len(tf_names),
        "missing_tfs": tf_missing,
        "git_commit": _maybe_git_commit(),
        "run_total_s": float(time.time() - t0),
    }
    with (out_root / "run_metadata.json").open("w") as f:
        json.dump(run_metadata, f, indent=2, sort_keys=True)

    with (out_root / "qc_metrics.json").open("w") as f:
        json.dump({"contexts": qc_all}, f, indent=2, sort_keys=True)

    print(f"[construct-grn] Wrote outputs to: {out_root}")


def run_from_config(config_path: str) -> None:
    d = load_yaml(config_path)
    allowed = {f.name for f in fields(GRNConstructConfig)}
    unknown = sorted(k for k in d.keys() if k not in allowed)
    if unknown:
        print(f"[construct-grn] WARNING: ignoring unknown config keys: {', '.join(unknown)}")
    cfg = GRNConstructConfig(**{k: v for k, v in d.items() if k in allowed})
    construct_grn(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Construct TF->target GRN from VAE checkpoint.")
    ap.add_argument("--config", required=True, help="Path to grn construction YAML config.")
    args = ap.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
