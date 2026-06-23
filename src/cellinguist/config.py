from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List

import json

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# CBOW config
# ---------------------------------------------------------------------------

@dataclass
class CBOWConfig:
    """
    Configuration for CBOW training.

    Attributes
    ----------
    emb_dim : int
        Embedding dimension for token embeddings.
    window_size : int
        Context window radius. Total context size = 2 * window_size.
    num_negatives : int
        Number of negative samples per (target, context) pair.
    num_workers : int
        Number of cpus for dataloader.
    batch_size : int
        Minibatch size for training.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    device : str
        Device string, e.g. "cpu" or "cuda".
    use_separate_output : bool
        If True, use a separate output embedding for targets/negatives
        (classic word2vec). If False, tie input and output embeddings.
    lr_scheduler : str
        Type of LR scheduler: "none" or "step".
    lr_step_size : int
        Step size (in epochs) for StepLR, if used.
    lr_gamma : float
        Multiplicative factor of learning rate decay for StepLR.
    """

    emb_dim: int = 128
    window_size: int = 5
    num_negatives: int = 10
    num_workers: int = 12
    batch_size: int = 1024
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    use_separate_output: bool = True

    lr_scheduler: str = "none"  # or "step"
    lr_step_size: int = 10
    lr_gamma: float = 0.5

    samples_per_cell: int = 1


# ---------------------------------------------------------------------------
# VAE Config
# ---------------------------------------------------------------------------

@dataclass
class VAETrainConfig:
    adata_path: str
    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    batch_key: Optional[str] = None  # Preferred alias for nuisance batch covariate.
    batch_correction_method: str = "none"  # "none" or "mean_scale"
    batch_correction_eps: float = 1e-8
    batch_correction_clip_min: float = 0.1
    batch_correction_clip_max: float = 10.0
    backed: bool = True

    # Used only when encoder_type == "cbow".
    gene_emb_tsv: str = ""
    encoder_type: str = "transformer"  # "transformer" or "perceiver" or "cbow"

    latent_dim: int = 32
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    cond_emb_dim: int = 16
    input_transform: str = "log1p"
    library_norm: str = "size_factor"  # "size_factor" or "none" (Perceiver encoder)
    library_norm_target_sum: float = 1e4
    library_norm_eps: float = 1e-8
    use_library_size_covariate: bool = False  # Decoder-side log1p(library_size) covariate
    library_size_covariate_eps: float = 1e-8
    freeze_gene_embeddings: bool = True
    perceiver_d_model: int = 256
    perceiver_num_latents: int = 64
    perceiver_num_cross_attn_heads: int = 8
    perceiver_num_self_attn_heads: int = 8
    perceiver_num_self_attn_layers: int = 4
    perceiver_ff_mult: int = 4
    perceiver_dropout: float = 0.0

    transformer_d_model: int = 256
    transformer_n_heads: int = 8
    transformer_n_layers: int = 4
    transformer_ff_mult: int = 4
    transformer_dropout: float = 0.0
    token_mlp_hidden_dim: int = 256
    token_mlp_layers: int = 2
    max_tokens_per_cell: Optional[int] = None
    min_expr_for_token: float = 0.0
    transformer_precompute_token_indices: bool = True
    token_index_cache_dir: str = ""
    token_index_cache_require: bool = True

    kl_weight: float = 1.0
    use_metric_loss: bool = False
    metric_loss_weight: float = 0.1
    metric_expr_transform: str = "log1p"  # "log1p" or "none"
    metric_margin: float = 0.2  # deprecated for contrastive loss, kept for backward compatibility
    metric_temperature: float = 0.1
    metric_k_pos: int = 5
    metric_k_neg: int = 20  # deprecated for contrastive loss, kept for backward compatibility
    runin_batches: int = 0
    runin_kl_weight: float = 0.0
    runin_metric_weight: float = 0.0
    batch_invariance_method: str = "none"  # "none" or "adversarial"
    batch_invariance_weight: float = 0.0
    batch_adv_grl_lambda: float = 1.0
    batch_adv_hidden_dim: int = 128
    batch_adv_n_hidden_layers: int = 1
    batch_invariance_warmup_epochs: int = 0
    perturbation_mode: str = "none"  # "none" | "categorical" | "cytokine_vector"
    cytokine_keys: Optional[List[str]] = None
    cytokine_transform: str = "log1p"  # "none" | "log1p" | "zscore"
    cytokine_missing_policy: str = "error"  # "error" | "fill_zero"
    perturb_emb_dim: int = 32
    perturb_condition_encoder: bool = True
    perturb_condition_decoder: bool = True
    cytokine_holdout_min_active: int = 2

    lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 50
    num_workers: int = 4
    device: str = "cuda"
    grad_clip_norm: float = 1.0
    seed: Optional[int] = 0

    checkpoint_dir: str = "checkpoints"
    run_name: str = "vae_run"
    loss_csv_path: str = ""
    resume_from: Optional[str] = None
    save_every: int = 1
    debug_steps: int = 0

    decoder_theta_init: float = 5.0          # initial theta (dispersion), gene-wise
    decoder_pi_init: float = 0.9             # initial dropout prob pi (ZI prob)
    decoder_mu_init: str = "data_mean"       # "data_mean" or "constant" or "none"
    decoder_mu_init_constant: float = 0.2    # used if decoder_mu_init == "constant"
    decoder_mu_init_cap: float = 10.0        # cap for gene-wise mean init
    decoder_mu_init_eps: float = 1e-4        # lower bound for mean init

    decoder_init_n_cells: int = 5000         # number of cells to estimate gene means from
    decoder_init_batch_size: int = 256       # batch size for mean-estimation pass
    decoder_init_num_workers: int = 0   

@dataclass
class VAEExportConfig:
    adata_path: str
    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    batch_key: Optional[str] = None
    perturbation_mode: str = "none"  # "none" | "categorical" | "cytokine_vector"
    cytokine_keys: Optional[List[str]] = None
    cytokine_transform: str = "log1p"  # "none" | "log1p" | "zscore"
    cytokine_missing_policy: str = "error"  # "error" | "fill_zero"
    perturb_emb_dim: int = 32
    perturb_condition_encoder: bool = True
    perturb_condition_decoder: bool = True
    counterfactual_override_path: Optional[str] = None

    # Used only when checkpoint/config indicates encoder_type == "cbow".
    gene_emb_tsv: str = ""
    checkpoint_path: str = ""
    out_pred_tsv_gz: str = "pred_mu.tsv.gz"

    max_cells: Optional[int] = None
    max_cells_seed: Optional[int] = None
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"
    backed: bool = True
    token_index_cache_dir: str = ""
    token_index_cache_require: bool = False
    transformer_precompute_token_indices: bool = True


@dataclass
class CytokineTreatmentPredictionConfig:
    adata_path: str
    checkpoint_path: str
    counterfactual_override_path: str
    out_dir: str

    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    batch_key: Optional[str] = None

    perturbation_mode: Optional[str] = None
    cytokine_keys: Optional[List[str]] = None
    cytokine_transform: Optional[str] = None
    cytokine_missing_policy: Optional[str] = None
    perturb_emb_dim: Optional[int] = None
    perturb_condition_encoder: Optional[bool] = None
    perturb_condition_decoder: Optional[bool] = None

    gene_emb_tsv: str = ""

    max_cells: Optional[int] = None
    max_cells_seed: Optional[int] = None
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"
    backed: bool = True
    token_index_cache_dir: str = ""
    token_index_cache_require: bool = False
    transformer_precompute_token_indices: bool = True


@dataclass
class GRNConstructConfig:
    adata_path: str
    checkpoint_path: str
    tf_list_path: str
    out_dir: str

    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    batch_key: Optional[str] = None
    context_key: Optional[str] = None
    min_cells_per_context: int = 300

    max_cells: Optional[int] = None
    max_cells_seed: int = 17
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda"
    backed: bool = True

    token_index_cache_dir: str = ""
    token_index_cache_require: bool = False
    transformer_precompute_token_indices: bool = True

    score_min: float = 0.35
    top_k_per_tf: int = 50
    min_abs_effect: float = 0.01
    allow_self_edges: bool = False

    perturb_frac: float = 0.10
    perturb_min_abs: float = 0.25
    sign_eps: float = 1e-3
    sign_consistency_min: float = 0.60

    w_a: float = 0.25
    w_e: float = 0.15
    w_d: float = 0.55
    w_p: float = 0.05

    prior_edges_tsv: Optional[str] = None
    bootstrap_iters: int = 0
    bootstrap_cell_frac: float = 0.80

    eps: float = 1e-8
    seed: int = 0


@dataclass
class PhenotypeTrainConfig:
    adata_path: str
    vae_checkpoint_path: str
    sample_key: str

    phenotype_key: Optional[str] = None
    phenotype_tsv_path: Optional[str] = None
    phenotype_tsv_sample_col: str = "sample_id"
    phenotype_tsv_value_col: str = "phenotype"

    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    batch_key: Optional[str] = None
    cell_type_key: Optional[str] = None
    backed: bool = True

    batch_correction_method: Optional[str] = None
    batch_correction_eps: Optional[float] = None
    batch_correction_clip_min: Optional[float] = None
    batch_correction_clip_max: Optional[float] = None
    perturbation_mode: Optional[str] = None
    cytokine_keys: Optional[List[str]] = None
    cytokine_transform: Optional[str] = None
    cytokine_missing_policy: Optional[str] = None
    perturb_emb_dim: Optional[int] = None

    freeze_vae_encoder: bool = True
    max_cells_per_sample: Optional[int] = None
    cell_subsample_mode: str = "random"  # "random" | "head"

    aggregator_hidden_dim: int = 128
    aggregator_latent_dim: int = 64
    predictor_hidden_dim: int = 128
    predictor_n_hidden_layers: int = 1
    dropout: float = 0.1

    batch_size_samples: int = 8
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.2
    min_train_samples: int = 2

    num_workers: int = 0
    device: str = "cuda"
    seed: int = 0

    checkpoint_dir: str = "checkpoints"
    run_name: str = "phenotype_model"
    resume_from: Optional[str] = None


@dataclass
class PhenotypePredictConfig:
    adata_path: str
    vae_checkpoint_path: str
    phenotype_checkpoint_path: str
    out_dir: str
    sample_key: str

    phenotype_key: Optional[str] = None
    phenotype_tsv_path: Optional[str] = None
    phenotype_tsv_sample_col: str = "sample_id"
    phenotype_tsv_value_col: str = "phenotype"

    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    batch_key: Optional[str] = None
    cell_type_key: Optional[str] = None
    backed: bool = True

    batch_correction_method: Optional[str] = None
    batch_correction_eps: Optional[float] = None
    batch_correction_clip_min: Optional[float] = None
    batch_correction_clip_max: Optional[float] = None
    perturbation_mode: Optional[str] = None
    cytokine_keys: Optional[List[str]] = None
    cytokine_transform: Optional[str] = None
    cytokine_missing_policy: Optional[str] = None
    perturb_emb_dim: Optional[int] = None

    max_cells_per_sample: Optional[int] = None
    cell_subsample_mode: str = "head"
    batch_size_samples: int = 8
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 0

    explanation_top_k: int = 5
    latent_cluster_count: int = 8


@dataclass
class PhenotypeCounterfactualConfig:
    adata_path: str
    vae_checkpoint_path: str
    phenotype_checkpoint_path: str
    intervention_path: str
    out_dir: str
    sample_key: str

    phenotype_key: Optional[str] = None
    phenotype_tsv_path: Optional[str] = None
    phenotype_tsv_sample_col: str = "sample_id"
    phenotype_tsv_value_col: str = "phenotype"

    gene_key: str = "gene"
    layer: Optional[str] = None
    cond_key: Optional[str] = None
    batch_key: Optional[str] = None
    cell_type_key: Optional[str] = None
    backed: bool = True

    batch_correction_method: Optional[str] = None
    batch_correction_eps: Optional[float] = None
    batch_correction_clip_min: Optional[float] = None
    batch_correction_clip_max: Optional[float] = None
    perturbation_mode: Optional[str] = None
    cytokine_keys: Optional[List[str]] = None
    cytokine_transform: Optional[str] = None
    cytokine_missing_policy: Optional[str] = None
    perturb_emb_dim: Optional[int] = None

    max_cells_per_sample: Optional[int] = None
    cell_subsample_mode: str = "head"
    batch_size_samples: int = 8
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 0

    explanation_top_k: int = 5
    latent_cluster_count: int = 8


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration file for CBOW training.

    The file can be YAML (.yml, .yaml) or JSON (.json).
    Returns a dict expected to contain (optionally) the keys:
      - "cbow":   CBOWConfig-related kwargs
      - "data":   data-related settings (paths, gene_key, etc.)
      - "output": output settings (e.g., embeddings_path)

    Any missing sections are filled with empty dicts.

    Parameters
    ----------
    path : str
        Path to a YAML or JSON config file.

    Returns
    -------
    cfg : Dict[str, Any]
        Parsed configuration dictionary with at least the keys:
        "cbow", "data", and "output".
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path_obj.suffix.lower()

    if suffix in {".yml", ".yaml"}:
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install with `pip install pyyaml`."
            )
        with path_obj.open("r") as f:
            cfg = yaml.safe_load(f)
    elif suffix == ".json":
        with path_obj.open("r") as f:
            cfg = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension '{suffix}'. "
            "Use .yml, .yaml, or .json."
        )

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping/dict.")

    # Ensure the expected top-level sections exist
    cfg.setdefault("cbow", {})
    cfg.setdefault("data", {})
    cfg.setdefault("output", {})

    return cfg
