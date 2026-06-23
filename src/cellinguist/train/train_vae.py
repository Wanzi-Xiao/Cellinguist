from __future__ import annotations

import argparse
import csv
import datetime
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from cellinguist.config import VAETrainConfig, load_yaml
from cellinguist.data.datasets import SingleCellVAEDataset
from cellinguist.data.dataloaders import collate_vae_batch
from cellinguist.models.vae import (
    CBOWCellEncoder,
    PerceiverCellEncoder,
    TransformerCellEncoder,
    ZINBExpressionDecoder,
    BatchAdversary,
    GeneVAE,
    zinb_negative_log_likelihood,
    kl_divergence_normal,
    expression_contrastive_metric_loss,
)
from cellinguist.utils.vae_io import (
    set_seed,
    load_gene_embeddings_tsv,
    intersect_genes_in_embedding_order,
    subset_embeddings,
    save_vae_checkpoint,
    load_vae_checkpoint,
    estimate_gene_means,
    inv_softplus,
    logit,
)
from cellinguist.utils.perturbation_split import build_cytokine_combo_split


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _early_log(msg: str) -> None:
    print(f"[ddp-setup] {msg}", flush=True)


def _setup_distributed(cfg_device: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return torch.device(cfg_device), 0, 1, False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    master_addr = os.environ.get("MASTER_ADDR", "unset")
    master_port = os.environ.get("MASTER_PORT", "unset")
    _early_log(
        f"pre-init rank={rank} local_rank={local_rank} world_size={world_size} "
        f"master={master_addr}:{master_port}"
    )

    if not torch.cuda.is_available():
        _early_log("CUDA unavailable; using gloo backend.")
        backend = "gloo"
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        backend = os.environ.get("CELLINGUIST_DDP_BACKEND", "nccl").lower()
        device = torch.device("cuda", local_rank)
        _early_log(
            f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')} "
            f"device_count={torch.cuda.device_count()} backend={backend}"
        )

    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(minutes=5),
        )
        _early_log(f"init_process_group ok rank={dist.get_rank()} world_size={dist.get_world_size()}")
    except Exception:
        _early_log("init_process_group failed with exception:")
        traceback.print_exc()
        raise

    return device, dist.get_rank(), dist.get_world_size(), True


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _log(rank: int, msg: str) -> None:
    print(f"[rank {rank}] {msg}", flush=True)


def _resolve_batch_key(batch_key: Optional[str], cond_key: Optional[str]) -> Optional[str]:
    if batch_key is not None and cond_key is not None and batch_key != cond_key:
        raise ValueError(
            f"Both batch_key='{batch_key}' and cond_key='{cond_key}' were provided, but differ."
        )
    return batch_key if batch_key is not None else cond_key


def _resolve_loss_csv_path(cfg: VAETrainConfig) -> str:
    if cfg.loss_csv_path:
        return str(cfg.loss_csv_path)
    return str(Path(cfg.checkpoint_dir) / f"{cfg.run_name}_losses.csv")


def _append_epoch_loss_row(
    path: str,
    *,
    epoch: int,
    train_loss: float,
    train_recon: float,
    train_kl: float,
    train_metric: float,
    train_adv: float,
    val_recon: Optional[float],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not out_path.exists()) or out_path.stat().st_size == 0
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_recon",
                "train_kl",
                "train_metric",
                "train_adv",
                "val_recon",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_recon": float(train_recon),
                "train_kl": float(train_kl),
                "train_metric": float(train_metric),
                "train_adv": float(train_adv),
                "val_recon": ("" if val_recon is None else float(val_recon)),
            }
        )


def train_vae(cfg: VAETrainConfig) -> str:
    device, rank, world_size, is_ddp = _setup_distributed(cfg.device)

    try:
        _log(
            rank,
            f"startup: ddp={is_ddp} world_size={world_size} device={device} "
            f"cfg_num_workers={cfg.num_workers} resume_from={cfg.resume_from}",
        )
        seed = None if cfg.seed is None else int(cfg.seed) + rank
        set_seed(seed)
        _log(rank, f"seed set to {seed}")

        is_main = rank == 0
        encoder_type = str(cfg.encoder_type).lower()
        if encoder_type not in {"cbow", "perceiver", "transformer"}:
            raise ValueError(
                f"Unsupported encoder_type: {cfg.encoder_type}. Use 'cbow', 'perceiver', or 'transformer'."
            )
        _log(rank, f"encoder_type={encoder_type}")
        if encoder_type == "transformer" and cfg.token_index_cache_require and not cfg.token_index_cache_dir:
            raise ValueError(
                "encoder_type='transformer' with token_index_cache_require=True requires token_index_cache_dir."
            )
        if cfg.use_metric_loss:
            if cfg.metric_loss_weight < 0:
                raise ValueError("metric_loss_weight must be >= 0.")
            if cfg.metric_temperature <= 0:
                raise ValueError("metric_temperature must be > 0.")
            if cfg.metric_k_pos < 1:
                raise ValueError("metric_k_pos must be >= 1.")
            if cfg.metric_expr_transform not in {"log1p", "none"}:
                raise ValueError(
                    f"Unsupported metric_expr_transform: {cfg.metric_expr_transform}"
                )
        if cfg.runin_batches < 0:
            raise ValueError("runin_batches must be >= 0.")
        if cfg.runin_kl_weight < 0:
            raise ValueError("runin_kl_weight must be >= 0.")
        if cfg.runin_metric_weight < 0:
            raise ValueError("runin_metric_weight must be >= 0.")
        if cfg.batch_invariance_method not in {"none", "adversarial"}:
            raise ValueError("batch_invariance_method must be 'none' or 'adversarial'.")
        if cfg.batch_correction_method not in {"none", "mean_scale"}:
            raise ValueError("batch_correction_method must be 'none' or 'mean_scale'.")
        if cfg.batch_correction_eps <= 0:
            raise ValueError("batch_correction_eps must be > 0.")
        if cfg.batch_correction_clip_min <= 0:
            raise ValueError("batch_correction_clip_min must be > 0.")
        if cfg.batch_correction_clip_max < cfg.batch_correction_clip_min:
            raise ValueError(
                "batch_correction_clip_max must be >= batch_correction_clip_min."
            )
        if cfg.batch_invariance_weight < 0:
            raise ValueError("batch_invariance_weight must be >= 0.")
        if cfg.batch_adv_grl_lambda <= 0:
            raise ValueError("batch_adv_grl_lambda must be > 0.")
        if cfg.batch_adv_hidden_dim <= 0:
            raise ValueError("batch_adv_hidden_dim must be > 0.")
        if cfg.batch_adv_n_hidden_layers < 0:
            raise ValueError("batch_adv_n_hidden_layers must be >= 0.")
        if cfg.batch_invariance_warmup_epochs < 0:
            raise ValueError("batch_invariance_warmup_epochs must be >= 0.")
        if cfg.perturbation_mode not in {"none", "categorical", "cytokine_vector"}:
            raise ValueError("perturbation_mode must be one of: none, categorical, cytokine_vector.")
        if cfg.cytokine_transform not in {"none", "log1p", "zscore"}:
            raise ValueError("cytokine_transform must be one of: none, log1p, zscore.")
        if cfg.cytokine_missing_policy not in {"error", "fill_zero"}:
            raise ValueError("cytokine_missing_policy must be one of: error, fill_zero.")
        if cfg.perturb_emb_dim <= 0:
            raise ValueError("perturb_emb_dim must be > 0.")
        if (
            cfg.perturbation_mode == "cytokine_vector"
            and not bool(cfg.perturb_condition_encoder)
            and not bool(cfg.perturb_condition_decoder)
        ):
            raise ValueError(
                "cytokine_vector mode requires perturb_condition_encoder or "
                "perturb_condition_decoder to be enabled."
            )
        if cfg.cytokine_holdout_min_active < 2:
            raise ValueError("cytokine_holdout_min_active must be >= 2.")

        effective_batch_key = _resolve_batch_key(cfg.batch_key, cfg.cond_key)
        if cfg.perturbation_mode == "categorical" and effective_batch_key is None:
            raise ValueError(
                "perturbation_mode='categorical' requires batch_key/cond_key."
            )
        if cfg.batch_correction_method != "none" and effective_batch_key is None:
            raise ValueError(
                "batch_correction_method requires batch_key/cond_key."
            )

        if encoder_type == "cbow":
            if not cfg.gene_emb_tsv:
                raise ValueError("gene_emb_tsv must be provided when encoder_type='cbow'.")

            genes_from_emb, emb_full = load_gene_embeddings_tsv(cfg.gene_emb_tsv)
            ds_probe = SingleCellVAEDataset(
                adata_or_path=cfg.adata_path,
                gene_key=cfg.gene_key,
                layer=cfg.layer,
                cond_key=effective_batch_key,
                batch_key=effective_batch_key,
                batch_correction_method=cfg.batch_correction_method,
                batch_correction_eps=cfg.batch_correction_eps,
                batch_correction_clip_min=cfg.batch_correction_clip_min,
                batch_correction_clip_max=cfg.batch_correction_clip_max,
                perturbation_mode=cfg.perturbation_mode,
                cytokine_keys=cfg.cytokine_keys,
                cytokine_transform=cfg.cytokine_transform,
                cytokine_missing_policy=cfg.cytokine_missing_policy,
                transform="none",
                backed=cfg.backed,
            )
            genes_expr = ds_probe.gene_order
            if is_main:
                print(f"Expression dataset: {len(genes_expr)} genes before intersection")

            genes_common = intersect_genes_in_embedding_order(genes_from_emb, genes_expr)
            emb = subset_embeddings(genes_from_emb, emb_full, genes_common)

            vae_dataset = SingleCellVAEDataset(
                adata_or_path=cfg.adata_path,
                gene_key=cfg.gene_key,
                layer=cfg.layer,
                cond_key=effective_batch_key,
                batch_key=effective_batch_key,
                batch_correction_method=cfg.batch_correction_method,
                batch_correction_eps=cfg.batch_correction_eps,
                batch_correction_clip_min=cfg.batch_correction_clip_min,
                batch_correction_clip_max=cfg.batch_correction_clip_max,
                perturbation_mode=cfg.perturbation_mode,
                cytokine_keys=cfg.cytokine_keys,
                cytokine_transform=cfg.cytokine_transform,
                cytokine_missing_policy=cfg.cytokine_missing_policy,
                gene_order=genes_common,
                transform="none",
                backed=cfg.backed,
            )
            n_cells, n_genes = vae_dataset.n_cells, vae_dataset.n_genes
            if is_main:
                print(f"VAE dataset after alignment: {n_cells} cells, {n_genes} genes")
            assert n_genes == emb.shape[0]
            gene_emb_source = cfg.gene_emb_tsv
        else:
            if cfg.use_library_size_covariate and str(cfg.library_norm).lower() != "none" and is_main:
                print(
                    "[VAE] WARNING: use_library_size_covariate=True with library_norm!='none'. "
                    "For a pure covariate strategy, set library_norm='none'."
                )
            transformer_cache_dir = ""
            if encoder_type == "transformer":
                transformer_cache_dir = str(cfg.token_index_cache_dir or "")
            use_transformer_in_memory_precompute = (
                encoder_type == "transformer"
                and (not transformer_cache_dir)
                and cfg.transformer_precompute_token_indices
            )

            vae_dataset = SingleCellVAEDataset(
                adata_or_path=cfg.adata_path,
                gene_key=cfg.gene_key,
                layer=cfg.layer,
                cond_key=effective_batch_key,
                batch_key=effective_batch_key,
                batch_correction_method=cfg.batch_correction_method,
                batch_correction_eps=cfg.batch_correction_eps,
                batch_correction_clip_min=cfg.batch_correction_clip_min,
                batch_correction_clip_max=cfg.batch_correction_clip_max,
                perturbation_mode=cfg.perturbation_mode,
                cytokine_keys=cfg.cytokine_keys,
                cytokine_transform=cfg.cytokine_transform,
                cytokine_missing_policy=cfg.cytokine_missing_policy,
                transform="none",
                backed=cfg.backed,
                precompute_token_gene_indices=use_transformer_in_memory_precompute,
                token_min_expr=cfg.min_expr_for_token,
                token_max_genes=cfg.max_tokens_per_cell,
                token_index_cache_dir=(transformer_cache_dir if encoder_type == "transformer" else None),
                token_index_cache_require=(
                    encoder_type == "transformer" and bool(cfg.token_index_cache_require)
                ),
            )
            genes_common = vae_dataset.gene_order
            n_cells, n_genes = vae_dataset.n_cells, vae_dataset.n_genes
            if is_main:
                print(f"VAE dataset: {n_cells} cells, {n_genes} genes ({encoder_type} encoder)")
            emb = None
            gene_emb_source = ""
        _log(rank, f"dataset ready: n_cells={n_cells} n_genes={n_genes}")

        n_conditions = (
            len(vae_dataset.batch_categories) if vae_dataset.batch_categories is not None else None
        )
        perturbation_dim = (
            int(vae_dataset.n_perturb_features) if cfg.perturbation_mode == "cytokine_vector" else None
        )
        if cfg.perturbation_mode == "cytokine_vector" and (perturbation_dim is None or perturbation_dim <= 0):
            raise ValueError("cytokine_vector mode requires non-empty cytokine_keys.")
        if is_main:
            print(
                f"[VAE] perturbation_mode={cfg.perturbation_mode} "
                f"n_perturb_features={perturbation_dim or 0} "
                f"cytokine_keys={vae_dataset.cytokine_keys if hasattr(vae_dataset, 'cytokine_keys') else []}"
            )

        if encoder_type == "cbow":
            encoder = CBOWCellEncoder(
                gene_embeddings=emb,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                n_hidden_layers=cfg.n_hidden_layers,
                n_conditions=n_conditions,
                cond_emb_dim=cfg.cond_emb_dim,
                perturbation_dim=perturbation_dim,
                perturb_emb_dim=cfg.perturb_emb_dim,
                perturb_condition_encoder=cfg.perturb_condition_encoder,
                freeze_gene_embeddings=cfg.freeze_gene_embeddings,
                input_transform=cfg.input_transform,
            )
        elif encoder_type == "perceiver":
            encoder = PerceiverCellEncoder(
                n_genes=n_genes,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                n_hidden_layers=cfg.n_hidden_layers,
                n_conditions=n_conditions,
                cond_emb_dim=cfg.cond_emb_dim,
                perturbation_dim=perturbation_dim,
                perturb_emb_dim=cfg.perturb_emb_dim,
                perturb_condition_encoder=cfg.perturb_condition_encoder,
                input_transform=cfg.input_transform,
                library_norm=cfg.library_norm,
                library_norm_target_sum=cfg.library_norm_target_sum,
                library_norm_eps=cfg.library_norm_eps,
                perceiver_d_model=cfg.perceiver_d_model,
                perceiver_num_latents=cfg.perceiver_num_latents,
                perceiver_num_cross_attn_heads=cfg.perceiver_num_cross_attn_heads,
                perceiver_num_self_attn_heads=cfg.perceiver_num_self_attn_heads,
                perceiver_num_self_attn_layers=cfg.perceiver_num_self_attn_layers,
                perceiver_ff_mult=cfg.perceiver_ff_mult,
                perceiver_dropout=cfg.perceiver_dropout,
            )
        else:
            encoder = TransformerCellEncoder(
                n_genes=n_genes,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                n_hidden_layers=cfg.n_hidden_layers,
                n_conditions=n_conditions,
                cond_emb_dim=cfg.cond_emb_dim,
                perturbation_dim=perturbation_dim,
                perturb_emb_dim=cfg.perturb_emb_dim,
                perturb_condition_encoder=cfg.perturb_condition_encoder,
                input_transform=cfg.input_transform,
                transformer_d_model=cfg.transformer_d_model,
                transformer_n_heads=cfg.transformer_n_heads,
                transformer_n_layers=cfg.transformer_n_layers,
                transformer_ff_mult=cfg.transformer_ff_mult,
                transformer_dropout=cfg.transformer_dropout,
                token_mlp_hidden_dim=cfg.token_mlp_hidden_dim,
                token_mlp_layers=cfg.token_mlp_layers,
                max_tokens_per_cell=cfg.max_tokens_per_cell,
                min_expr_for_token=cfg.min_expr_for_token,
            )
        decoder = ZINBExpressionDecoder(
            n_genes=n_genes,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            n_hidden_layers=cfg.n_hidden_layers,
            n_conditions=n_conditions,
            cond_emb_dim=cfg.cond_emb_dim,
            perturbation_dim=perturbation_dim,
            perturb_emb_dim=cfg.perturb_emb_dim,
            perturb_condition_decoder=cfg.perturb_condition_decoder,
            use_library_size_covariate=cfg.use_library_size_covariate,
            library_size_covariate_eps=cfg.library_size_covariate_eps,
        )
        batch_adversary = None
        use_adv = cfg.batch_invariance_method == "adversarial"
        if use_adv:
            if n_conditions is None or n_conditions < 2:
                raise ValueError(
                    "Adversarial batch invariance requires a batch/cond key with at least 2 categories."
                )
            batch_adversary = BatchAdversary(
                latent_dim=cfg.latent_dim,
                n_batches=n_conditions,
                hidden_dim=cfg.batch_adv_hidden_dim,
                n_hidden_layers=cfg.batch_adv_n_hidden_layers,
            )
        model: torch.nn.Module = GeneVAE(encoder, decoder, batch_adversary=batch_adversary).to(device)
        _log(rank, "model built")

        if is_ddp:
            model = DDP(model, device_ids=[device.index], output_device=device.index)
            _log(rank, "DDP wrapper initialized")

        raw_model = _unwrap(model)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        start_epoch = 0
        if cfg.resume_from:
            ckpt = load_vae_checkpoint(
                cfg.resume_from,
                raw_model,
                optimizer,
                map_location=device,
                strict=True,
            )
            start_epoch = int(ckpt.get("epoch", -1)) + 1

            ckpt_genes = ckpt.get("genes_common", None)
            if ckpt_genes is not None and ckpt_genes != genes_common:
                raise ValueError("genes_common mismatch between checkpoint and current data/embeddings.")
        _log(rank, f"checkpoint state ready: start_epoch={start_epoch}")

        train_dataset = vae_dataset
        val_dataset = None
        if cfg.perturbation_mode == "cytokine_vector":
            split = build_cytokine_combo_split(
                perturb_matrix=vae_dataset.get_perturb_matrix(),
                cytokine_keys=vae_dataset.cytokine_keys,
                min_active_for_holdout=cfg.cytokine_holdout_min_active,
            )
            if split["val_indices"]:
                train_dataset = Subset(vae_dataset, split["train_indices"])
                val_dataset = Subset(vae_dataset, split["val_indices"])
            if is_main:
                print(
                    "[VAE] cytokine split summary: "
                    f"train_cells={split['n_train_cells']} val_cells={split['n_val_cells']} "
                    f"n_train_signatures={split['n_train_signatures']} "
                    f"n_val_signatures={split['n_val_signatures']}"
                )

        sampler = None
        if is_ddp:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            _log(rank, "distributed sampler ready")

        # Safer DDP default: avoid worker subprocesses unless explicitly requested >0.
        effective_num_workers = int(cfg.num_workers)
        if is_ddp and effective_num_workers <= 0:
            effective_num_workers = 0

        dl = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=effective_num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=effective_num_workers > 0,
            collate_fn=collate_vae_batch,
        )
        _log(rank, f"dataloader ready: batch_size={cfg.batch_size} num_workers={effective_num_workers}")

        val_dl = None
        if val_dataset is not None:
            val_dl = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=effective_num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=effective_num_workers > 0,
                collate_fn=collate_vae_batch,
            )

        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_last = str(ckpt_dir / f"{cfg.run_name}_last.ckpt")
        loss_csv_path = _resolve_loss_csv_path(cfg)
        if is_main and not cfg.resume_from:
            loss_csv_file = Path(loss_csv_path)
            if loss_csv_file.exists():
                loss_csv_file.unlink()
        _log(rank, f"checkpoint path={ckpt_last}")

        if not cfg.resume_from:
            if hasattr(raw_model.decoder, "log_theta"):
                theta_init = torch.full((n_genes,), float(cfg.decoder_theta_init), dtype=torch.float32)
                raw_model.decoder.log_theta.data = inv_softplus(theta_init).to(raw_model.decoder.log_theta.data.device)

            if hasattr(raw_model.decoder, "mlp_pi"):
                pi_linear = raw_model.decoder.mlp_pi.net[-1]
                pi_linear.bias.data.fill_(logit(cfg.decoder_pi_init))
                torch.nn.init.zeros_(pi_linear.weight)

            if hasattr(raw_model.decoder, "mlp_mu"):
                mu_linear = raw_model.decoder.mlp_mu.net[-1]

                if cfg.decoder_mu_init == "data_mean":
                    if is_main:
                        mean_x = estimate_gene_means(
                            vae_dataset,
                            max_cells=cfg.decoder_init_n_cells,
                            batch_size=cfg.decoder_init_batch_size,
                            num_workers=cfg.decoder_init_num_workers,
                            device=None,
                            require_integer=(cfg.batch_correction_method == "none"),
                        )
                        mean_x = torch.clamp(
                            mean_x,
                            min=cfg.decoder_mu_init_eps,
                            max=cfg.decoder_mu_init_cap,
                        )
                    else:
                        mean_x = torch.empty((n_genes,), dtype=torch.float32, device=device)

                    if is_ddp:
                        # NCCL does not support CPU tensors for collectives.
                        if is_main:
                            mean_x = mean_x.to(device=device, dtype=torch.float32, non_blocking=True)
                        dist.broadcast(mean_x, src=0)
                    elif mean_x.device != device:
                        mean_x = mean_x.to(device=device, dtype=torch.float32, non_blocking=True)

                    mu_bias = inv_softplus(mean_x).to(mu_linear.bias.device)
                    mu_linear.bias.data.copy_(mu_bias)
                    torch.nn.init.zeros_(mu_linear.weight)

                elif cfg.decoder_mu_init == "constant":
                    mean0 = torch.tensor(cfg.decoder_mu_init_constant)
                    mean0 = torch.clamp(mean0, min=cfg.decoder_mu_init_eps, max=cfg.decoder_mu_init_cap)
                    b0 = inv_softplus(mean0).item()
                    mu_linear.bias.data.fill_(b0)
                    torch.nn.init.zeros_(mu_linear.weight)

        if is_ddp:
            for p in raw_model.parameters():
                dist.broadcast(p.data, src=0)
            for b in raw_model.buffers():
                dist.broadcast(b.data, src=0)
            _log(rank, "initial parameter/buffer broadcast complete")

        model.train()
        _log(rank, f"training loop start: epochs={cfg.epochs} start_epoch={start_epoch}")

        if start_epoch == 0 and cfg.runin_batches > 0:
            if sampler is not None:
                sampler.set_epoch(0)
            runin_total = 0.0
            runin_recon = 0.0
            runin_kl = 0.0
            runin_metric = 0.0
            runin_nb = 0
            max_runin_batches = min(int(cfg.runin_batches), len(dl))
            _log(
                rank,
                f"run-in start: batches={max_runin_batches} kl_w={cfg.runin_kl_weight} "
                f"metric_w={cfg.runin_metric_weight}",
            )
            for batch in dl:
                if runin_nb >= max_runin_batches:
                    break

                x = batch["x_expr"].to(device, non_blocking=True)
                libsize = batch.get("libsize", None)
                if libsize is None:
                    libsize = x.sum(dim=1)
                libsize = libsize.to(device, non_blocking=True)
                batch_idx = batch.get("batch_idx", None)
                if batch_idx is None:
                    batch_idx = batch.get("cond_idx", None)
                if batch_idx is not None:
                    batch_idx = batch_idx.to(device, non_blocking=True)
                perturb_vec = batch.get("perturb_vec", None)
                if perturb_vec is not None:
                    perturb_vec = perturb_vec.to(device, non_blocking=True)

                token_gene_idx = batch.get("token_gene_idx", None)
                token_gene_mask = batch.get("token_gene_mask", None)
                if token_gene_idx is not None:
                    token_gene_idx = token_gene_idx.to(device, non_blocking=True)
                if token_gene_mask is not None:
                    token_gene_mask = token_gene_mask.to(device, non_blocking=True)

                recon_out, mu_z, logvar_z = model(
                    x,
                    batch_idx,
                    libsize=libsize,
                    perturb_vec=perturb_vec,
                    token_gene_idx=token_gene_idx,
                    token_gene_mask=token_gene_mask,
                )
                mu, theta, pi = recon_out

                recon = zinb_negative_log_likelihood(x, mu, theta, pi, reduction="mean")
                kl = kl_divergence_normal(mu_z, logvar_z, reduction="mean")
                metric = x.new_zeros(())
                if cfg.use_metric_loss and cfg.runin_metric_weight > 0:
                    metric = expression_contrastive_metric_loss(
                        x_expr=x,
                        z_latent=mu_z,
                        expr_transform=cfg.metric_expr_transform,
                        temperature=cfg.metric_temperature,
                        k_pos=cfg.metric_k_pos,
                    )
                loss = recon + cfg.runin_kl_weight * kl + cfg.runin_metric_weight * metric

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

                runin_total += float(loss.item())
                runin_recon += float(recon.item())
                runin_kl += float(kl.item())
                runin_metric += float(metric.item())
                runin_nb += 1

            if is_ddp:
                runin_stats = torch.tensor(
                    [runin_total, runin_recon, runin_kl, runin_metric, float(runin_nb)],
                    device=device,
                )
                dist.all_reduce(runin_stats, op=dist.ReduceOp.SUM)
                runin_total = float(runin_stats[0].item())
                runin_recon = float(runin_stats[1].item())
                runin_kl = float(runin_stats[2].item())
                runin_metric = float(runin_stats[3].item())
                runin_nb = int(runin_stats[4].item())

            if is_main and runin_nb > 0:
                denom = max(runin_nb, 1)
                print(
                    f"[VAE] Run-in ({runin_nb} batches) "
                    f"loss={runin_total/denom:.4f} "
                    f"recon={runin_recon/denom:.4f} "
                    f"kl={runin_kl/denom:.4f} "
                    f"metric={runin_metric/denom:.4f}"
                )

        for epoch in range(start_epoch, cfg.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            total = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_metric = 0.0
            total_adv = 0.0
            nb = 0

            for batch in dl:
                x = batch["x_expr"].to(device, non_blocking=True)
                libsize = batch.get("libsize", None)
                if libsize is None:
                    libsize = x.sum(dim=1)
                libsize = libsize.to(device, non_blocking=True)
                batch_idx = batch.get("batch_idx", None)
                if batch_idx is None:
                    batch_idx = batch.get("cond_idx", None)
                if batch_idx is not None:
                    batch_idx = batch_idx.to(device, non_blocking=True)
                perturb_vec = batch.get("perturb_vec", None)
                if perturb_vec is not None:
                    perturb_vec = perturb_vec.to(device, non_blocking=True)

                token_gene_idx = batch.get("token_gene_idx", None)
                token_gene_mask = batch.get("token_gene_mask", None)
                if token_gene_idx is not None:
                    token_gene_idx = token_gene_idx.to(device, non_blocking=True)
                if token_gene_mask is not None:
                    token_gene_mask = token_gene_mask.to(device, non_blocking=True)

                recon_out, mu_z, logvar_z = model(
                    x,
                    batch_idx,
                    libsize=libsize,
                    perturb_vec=perturb_vec,
                    token_gene_idx=token_gene_idx,
                    token_gene_mask=token_gene_mask,
                )
                mu, theta, pi = recon_out

                recon = zinb_negative_log_likelihood(x, mu, theta, pi, reduction="mean")
                kl = kl_divergence_normal(mu_z, logvar_z, reduction="mean")
                metric = x.new_zeros(())
                if cfg.use_metric_loss and cfg.metric_loss_weight > 0:
                    metric = expression_contrastive_metric_loss(
                        x_expr=x,
                        z_latent=mu_z,
                        expr_transform=cfg.metric_expr_transform,
                        temperature=cfg.metric_temperature,
                        k_pos=cfg.metric_k_pos,
                    )
                adv = x.new_zeros(())
                if (
                    use_adv
                    and cfg.batch_invariance_weight > 0
                    and batch_idx is not None
                    and epoch >= cfg.batch_invariance_warmup_epochs
                ):
                    logits = raw_model.predict_batch_logits(
                        mu_z, grl_lambda=cfg.batch_adv_grl_lambda
                    )
                    adv = F.cross_entropy(logits, batch_idx)
                loss = (
                    recon
                    + cfg.kl_weight * kl
                    + cfg.metric_loss_weight * metric
                    + cfg.batch_invariance_weight * adv
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

                total += float(loss.item())
                total_recon += float(recon.item())
                total_kl += float(kl.item())
                total_metric += float(metric.item())
                total_adv += float(adv.item())
                nb += 1

            if is_ddp:
                loss_stats = torch.tensor(
                    [total, total_recon, total_kl, total_metric, total_adv, float(nb)],
                    device=device,
                )
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                total = float(loss_stats[0].item())
                total_recon = float(loss_stats[1].item())
                total_kl = float(loss_stats[2].item())
                total_metric = float(loss_stats[3].item())
                total_adv = float(loss_stats[4].item())
                nb = int(loss_stats[5].item())

            val_recon = 0.0
            val_nb = 0
            if val_dl is not None:
                model.eval()
                with torch.no_grad():
                    for batch in val_dl:
                        x = batch["x_expr"].to(device, non_blocking=True)
                        libsize = batch.get("libsize", None)
                        if libsize is None:
                            libsize = x.sum(dim=1)
                        libsize = libsize.to(device, non_blocking=True)
                        batch_idx = batch.get("batch_idx", None)
                        if batch_idx is None:
                            batch_idx = batch.get("cond_idx", None)
                        if batch_idx is not None:
                            batch_idx = batch_idx.to(device, non_blocking=True)
                        perturb_vec = batch.get("perturb_vec", None)
                        if perturb_vec is not None:
                            perturb_vec = perturb_vec.to(device, non_blocking=True)

                        token_gene_idx = batch.get("token_gene_idx", None)
                        token_gene_mask = batch.get("token_gene_mask", None)
                        if token_gene_idx is not None:
                            token_gene_idx = token_gene_idx.to(device, non_blocking=True)
                        if token_gene_mask is not None:
                            token_gene_mask = token_gene_mask.to(device, non_blocking=True)

                        recon_out, _, _ = model(
                            x,
                            batch_idx,
                            libsize=libsize,
                            perturb_vec=perturb_vec,
                            token_gene_idx=token_gene_idx,
                            token_gene_mask=token_gene_mask,
                        )
                        mu, theta, pi = recon_out
                        recon = zinb_negative_log_likelihood(x, mu, theta, pi, reduction="mean")
                        val_recon += float(recon.item())
                        val_nb += 1
                model.train()

            if is_ddp and val_dl is not None:
                val_stats = torch.tensor([val_recon, float(val_nb)], device=device)
                dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
                val_recon = float(val_stats[0].item())
                val_nb = int(val_stats[1].item())

            if is_main:
                denom = max(nb, 1)
                train_loss_epoch = total / denom
                train_recon_epoch = total_recon / denom
                train_kl_epoch = total_kl / denom
                train_metric_epoch = total_metric / denom
                train_adv_epoch = total_adv / denom
                val_recon_epoch = (val_recon / max(val_nb, 1)) if val_nb > 0 else None
                val_msg = ""
                if val_recon_epoch is not None:
                    val_msg = f" val_recon={val_recon_epoch:.4f}"
                print(
                    f"[VAE] Epoch {epoch+1}/{cfg.epochs} "
                    f"loss={train_loss_epoch:.4f} "
                    f"recon={train_recon_epoch:.4f} "
                    f"kl={train_kl_epoch:.4f} "
                    f"metric={train_metric_epoch:.4f} "
                    f"adv={train_adv_epoch:.4f}"
                    f"{val_msg}"
                )
                _append_epoch_loss_row(
                    loss_csv_path,
                    epoch=epoch + 1,
                    train_loss=train_loss_epoch,
                    train_recon=train_recon_epoch,
                    train_kl=train_kl_epoch,
                    train_metric=train_metric_epoch,
                    train_adv=train_adv_epoch,
                    val_recon=val_recon_epoch,
                )

                if cfg.save_every > 0 and ((epoch + 1) % cfg.save_every == 0):
                    save_vae_checkpoint(
                        ckpt_last,
                        model=raw_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        genes_common=genes_common,
                        config_snapshot=asdict(cfg),
                        gene_emb_source=gene_emb_source,
                    )

        if is_main:
            save_vae_checkpoint(
                ckpt_last,
                model=raw_model,
                optimizer=optimizer,
                epoch=cfg.epochs - 1,
                genes_common=genes_common,
                config_snapshot=asdict(cfg),
                gene_emb_source=gene_emb_source,
            )

        _log(rank, "train_vae completed")
        return ckpt_last
    except Exception:
        _log(rank, "fatal exception in train_vae")
        traceback.print_exc()
        raise
    finally:
        if _is_distributed():
            _log(rank, "destroying process group")
            dist.destroy_process_group()


def run_vae_training_from_config(config_path: str) -> None:
    d = load_yaml(config_path)
    cfg = VAETrainConfig(**d)
    train_vae(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Traine VAE from config file.")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file."
    )
    args = ap.parse_args()
    run_vae_training_from_config(args.config)


if __name__ == "__main__":
    main()
