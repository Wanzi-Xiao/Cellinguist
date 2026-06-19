"""
Initialize a new cytokine experiment run directory.

Creates the full directory scaffold and all sbatch/config files following
the project's standard experiment layout (run_XXX_description_YYMMDD/).

Usage:
    python -m cellinguist.scripts.init_cytokine_run \\
        --runs-dir    /ix1/acillo/wax11/21_cellinguist_260312/02_transformer_runs \\
        --raw-adata   /ix1/acillo/wax11/21_cellinguist_260312/01_input/cytokine_dict_ser_sub_full_genes_ad_251118.h5ad \\
        --cellinguist-dir /path/to/Cellinguist \\
        --conda-env   pytorch_250107 \\
        [--run-num 004]
        [--description full_genes_cytokine]

After running, submit everything with:
    cd {runs_dir}/run_004_full_genes_cytokine_YYMMDD
    bash scripts/run_all.sh
"""
from __future__ import annotations

import argparse
import re
import textwrap
from datetime import datetime
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_run_num(runs_dir: Path) -> int:
    nums = []
    for d in runs_dir.iterdir():
        m = re.match(r"run_(\d+)_", d.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def _write(path: Path, content: str, run_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"))
    print(f"  [init] wrote {path.relative_to(run_dir.parent)}")


# ─────────────────────────────────────────────────────────────────────────────
# sbatch generators
# Log naming convention: r{run_num:03d}_{step}_%j.{out|err}  (matches existing runs)
# ─────────────────────────────────────────────────────────────────────────────

def _sbatch_preprocess(
    run_num: int,
    run_dir: Path,
    preprocessed_dir: Path,
    raw_adata: str,
    run_name: str,
    cellinguist_dir: str,
    python: str,
) -> str:
    preprocessed_h5ad = preprocessed_dir / "cytokine_ad_preprocessed.h5ad"
    keys_file         = preprocessed_dir / "cytokine_ad_preprocessed.cytokine_keys.txt"
    train_cfg         = run_dir / "configs" / "vae_train.yml"
    log               = run_dir / "logs" / f"r{run_num:03d}_preprocess_%j"
    return f"""\
#!/bin/bash
#SBATCH --job-name=r{run_num:03d}_preprocess
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output={log}.out
#SBATCH --error={log}.err

PYTHON="{python}"
set -euo pipefail
cd {cellinguist_dir}

PREPROCESSED_H5AD="{preprocessed_h5ad}"
KEYS_FILE="{keys_file}"
TRAIN_CFG="{train_cfg}"

if [ -f "$PREPROCESSED_H5AD" ]; then
    echo "[preprocess] Preprocessed adata already exists — skipping."
else
    echo "[preprocess] Running preprocessing..."
    python -m cellinguist.scripts.prepare_cytokine_adata \\
        "{raw_adata}" \\
        "$PREPROCESSED_H5AD" \\
        --cytokine-key cytokine \\
        --control PBS \\
        --na-label NA
fi

# Patch configs/vae_train.yml with discovered cytokine_keys
echo "[preprocess] Patching $TRAIN_CFG with cytokine_keys..."
python - <<'PYEOF'
import yaml, sys
keys_file = "{keys_file}"
train_cfg  = "{train_cfg}"
with open(keys_file) as f:
    data = yaml.safe_load(f)
keys = [str(k) for k in data["cytokine_keys"]]
with open(train_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["cytokine_keys"] = keys
with open(train_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(f"[preprocess] Patched with {{len(keys)}} cytokine keys: {{keys[:4]}}...")
PYEOF

echo "[preprocess] Done."
"""


def _sbatch_token_cache(
    run_num: int,
    run_dir: Path,
    preprocessed_dir: Path,
    run_name: str,
    cellinguist_dir: str,
    conda_env: str,
    max_tokens: int | None,
    min_expr: float,
    gene_key: str,
    layer: str | None,
    num_workers: int,
) -> str:
    preprocessed_h5ad = preprocessed_dir / "cytokine_ad_preprocessed.h5ad"
    token_cache_dir   = run_dir / "token_cache"
    log               = run_dir / "logs" / f"r{run_num:03d}_precompute_%j"
    max_tokens_arg    = f"--max-tokens-per-cell {max_tokens}" if max_tokens else ""
    layer_arg         = f"--layer {layer}" if layer else ""
    return f"""\
#!/bin/bash
#SBATCH --job-name=r{run_num:03d}_precompute
#SBATCH --partition=htc
#SBATCH --cpus-per-task={num_workers}
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output={log}.out
#SBATCH --error={log}.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {conda_env}
cd {cellinguist_dir}

python -m cellinguist.scripts.precompute_transformer_token_indices \\
    --adata         "{preprocessed_h5ad}" \\
    --out-dir       "{token_cache_dir}" \\
    --gene-key      {gene_key} \\
    --min-expr-for-token {min_expr} \\
    --num-workers   {num_workers} \\
    {max_tokens_arg} {layer_arg}

echo "[precompute] Token cache written to {token_cache_dir}"
"""


def _sbatch_train(
    run_num: int,
    run_dir: Path,
    run_name: str,
    cellinguist_dir: str,
    conda_env: str,
) -> str:
    train_cfg = run_dir / "configs" / "vae_train.yml"
    log       = run_dir / "logs" / f"r{run_num:03d}_train_%j"
    return f"""\
#!/bin/bash
#SBATCH --job-name=r{run_num:03d}_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output={log}.out
#SBATCH --error={log}.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {conda_env}
cd {cellinguist_dir}

echo "[train] Started at $(date)"
echo "[train] Node: $SLURMD_NODENAME"
echo "[train] GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m cellinguist.train.train_vae --config "{train_cfg}"

echo "[train] Finished at $(date)"
"""


def _sbatch_predict(
    run_num: int,
    run_dir: Path,
    preprocessed_dir: Path,
    run_name: str,
    cellinguist_dir: str,
    conda_env: str,
    predict_mode: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> str:
    preprocessed_h5ad = preprocessed_dir / "cytokine_ad_preprocessed.h5ad"
    keys_file         = preprocessed_dir / "cytokine_ad_preprocessed.cytokine_keys.txt"
    checkpoint_dir    = run_dir / "checkpoints"
    token_cache_dir   = run_dir / "token_cache"
    overrides_dir     = run_dir / "predictions" / "overrides"
    predictions_dir   = run_dir / "predictions"
    log               = run_dir / "logs" / f"r{run_num:03d}_predict_%j"
    return f"""\
#!/bin/bash
#SBATCH --job-name=r{run_num:03d}_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output={log}.out
#SBATCH --error={log}.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {conda_env}
cd {cellinguist_dir}

echo "[predict] Started at $(date)"
echo "[predict] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Locate best checkpoint
CKPT=$(python - <<'PYEOF'
import glob, sys
from pathlib import Path
ckpt_dir  = "{checkpoint_dir}"
run_name  = "{run_name}"
last = Path(ckpt_dir) / f"{{run_name}}_last.ckpt"
if last.exists():
    print(last); sys.exit(0)
candidates = sorted(glob.glob(str(Path(ckpt_dir) / f"{{run_name}}_epoch*.ckpt")))
if candidates:
    print(candidates[-1]); sys.exit(0)
print("ERROR: no checkpoint found", file=sys.stderr); sys.exit(1)
PYEOF
)
echo "[predict] Checkpoint: $CKPT"

# Read cytokine keys
mapfile -t CYTOKINES < <(python -c "
import yaml
with open('{keys_file}') as f:
    d = yaml.safe_load(f)
for k in d['cytokine_keys']:
    print(k)
")
echo "[predict] ${{#CYTOKINES[@]}} cytokines to predict"

mkdir -p "{overrides_dir}"

for CYTOKINE in "${{CYTOKINES[@]}}"; do
    echo "[predict] ── $CYTOKINE ──"
    OVERRIDE_TSV="{overrides_dir}/${{CYTOKINE}}_{predict_mode}.tsv"
    PRED_DIR="{predictions_dir}/$CYTOKINE"

    python -m cellinguist.scripts.make_cytokine_override_tsv \\
        "{preprocessed_h5ad}" "$OVERRIDE_TSV" \\
        --cytokine-keys-file "{keys_file}" \\
        --mode {predict_mode} \\
        --target-cytokine "$CYTOKINE"

    python -c "
from cellinguist.config import CytokineTreatmentPredictionConfig
from cellinguist.scripts.predict_cytokine_treatment import predict_cytokine_treatment
import sys
cfg = CytokineTreatmentPredictionConfig(
    adata_path='{preprocessed_h5ad}',
    checkpoint_path=sys.argv[1],
    counterfactual_override_path=sys.argv[2],
    out_dir=sys.argv[3],
    gene_key='gene',
    batch_key='processing_batch',
    batch_size={batch_size},
    num_workers={num_workers},
    device='{device}',
    backed=True,
    token_index_cache_dir='{token_cache_dir}',
    token_index_cache_require=False,
    transformer_precompute_token_indices=True,
)
predict_cytokine_treatment(cfg)
" "$CKPT" "$OVERRIDE_TSV" "$PRED_DIR"

    echo "[predict] $CYTOKINE → $PRED_DIR"
done

echo "[predict] All done at $(date)"
"""


def _run_all_sh(run_num: int, run_dir: Path) -> str:
    return f"""\
#!/bin/bash
# Submit all cytokine pipeline jobs with SLURM dependencies.
# Run from the experiment directory:
#   cd {run_dir}
#   bash scripts/run_all.sh

set -euo pipefail
cd "{run_dir}"

echo "Submitting pipeline: {run_dir.name}"

JID0=$(sbatch --parsable scripts/00_preprocess.sbatch)
echo "  00_preprocess:         $JID0"

JID1=$(sbatch --parsable --dependency=afterok:$JID0 scripts/01_precompute_tokens.sbatch)
echo "  01_precompute_tokens:  $JID1"

JID2=$(sbatch --parsable --dependency=afterok:$JID1 scripts/02_train_vae.sbatch)
echo "  02_train_vae:          $JID2"

JID3=$(sbatch --parsable --dependency=afterok:$JID2 scripts/03_predict_cytokines.sbatch)
echo "  03_predict_cytokines:  $JID3"

echo ""
echo "Chain: $JID0 → $JID1 → $JID2 → $JID3"
echo "Logs:  {run_dir}/logs/"
echo "Watch: squeue -u $USER"
"""


def _vae_train_yml(
    preprocessed_h5ad: Path,
    run_dir: Path,
    run_name: str,
    args: argparse.Namespace,
) -> dict:
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    return {
        "adata_path": str(preprocessed_h5ad),
        "gene_key": "gene",
        "layer": None,
        "batch_key": "processing_batch",
        "cond_key": None,
        "batch_correction_method": "none",
        "batch_correction_eps": 1e-8,
        "batch_correction_clip_min": 0.1,
        "batch_correction_clip_max": 10.0,
        "backed": True,
        "encoder_type": "transformer",
        "gene_emb_tsv": "",
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "n_hidden_layers": args.n_hidden_layers,
        "cond_emb_dim": 16,
        "input_transform": "log1p",
        "library_norm": "size_factor",
        "library_norm_target_sum": 10000.0,
        "library_norm_eps": 1e-8,
        "use_library_size_covariate": False,
        "library_size_covariate_eps": 1e-8,
        "freeze_gene_embeddings": True,
        "transformer_d_model": args.transformer_d_model,
        "transformer_n_heads": args.transformer_n_heads,
        "transformer_n_layers": args.transformer_n_layers,
        "transformer_ff_mult": 4,
        "transformer_dropout": 0.0,
        "token_mlp_hidden_dim": 256,
        "token_mlp_layers": 2,
        "max_tokens_per_cell": max_tokens,
        "min_expr_for_token": 0.0,
        "transformer_precompute_token_indices": True,
        "token_index_cache_dir": str(run_dir / "token_cache"),
        "token_index_cache_require": True,
        "perturbation_mode": "cytokine_vector",
        "cytokine_keys": None,          # patched by 00_preprocess.sbatch
        "cytokine_transform": "log1p",
        "cytokine_missing_policy": "fill_zero",
        "perturb_emb_dim": 32,
        "perturb_condition_encoder": False,
        "perturb_condition_decoder": True,
        "cytokine_holdout_min_active": 2,
        "kl_weight": 1.0,
        "use_metric_loss": False,
        "metric_loss_weight": 0.1,
        "metric_expr_transform": "log1p",
        "metric_margin": 0.2,
        "metric_temperature": 0.1,
        "metric_k_pos": 5,
        "metric_k_neg": 20,
        "runin_batches": 0,
        "runin_kl_weight": 0.0,
        "runin_metric_weight": 0.0,
        "batch_invariance_method": "none",
        "batch_invariance_weight": 0.0,
        "batch_adv_grl_lambda": 1.0,
        "batch_adv_hidden_dim": 128,
        "batch_adv_n_hidden_layers": 1,
        "batch_invariance_warmup_epochs": 0,
        "lr": args.lr,
        "weight_decay": 0.0,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "device": args.device,
        "grad_clip_norm": 1.0,
        "seed": 0,
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "run_name": run_name,
        "resume_from": None,
        "save_every": 5,
        "loss_csv_path": str(run_dir / "checkpoints" / f"{run_name}_loss_curve.csv"),
        "decoder_theta_init": 5.0,
        "decoder_pi_init": 0.9,
        "decoder_mu_init": "data_mean",
        "decoder_mu_init_constant": 0.2,
        "decoder_mu_init_cap": 10.0,
        "decoder_mu_init_eps": 1e-4,
        "decoder_init_n_cells": 5000,
        "decoder_init_batch_size": 256,
        "decoder_init_num_workers": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scaffold a new cytokine experiment run directory."
    )
    ap.add_argument("--runs-dir",       required=True)
    ap.add_argument("--raw-adata",      required=True)
    ap.add_argument("--cellinguist-dir",required=True)
    ap.add_argument("--conda-env",      default="pytorch_250107")
    ap.add_argument("--run-num",        type=int, default=None,
                    help="Run number (auto-detected if omitted)")
    ap.add_argument("--description",    default="full_genes_cytokine")
    ap.add_argument("--run-name",       default=None)
    ap.add_argument("--predict-mode",   default="all_treated",
                    choices=["all_treated", "pbs_only", "zero"])
    ap.add_argument("--epochs",              type=int,   default=50)
    ap.add_argument("--batch-size",          type=int,   default=256)
    ap.add_argument("--lr",                  type=float, default=3e-4)
    ap.add_argument("--latent-dim",          type=int,   default=32)
    ap.add_argument("--hidden-dim",          type=int,   default=256)
    ap.add_argument("--n-hidden-layers",     type=int,   default=2)
    ap.add_argument("--transformer-d-model", type=int,   default=256)
    ap.add_argument("--transformer-n-heads", type=int,   default=8)
    ap.add_argument("--transformer-n-layers",type=int,   default=4)
    ap.add_argument("--max-tokens",          type=int,   default=2000,
                    help="Max tokens per cell (0 = uncapped)")
    ap.add_argument("--num-workers",         type=int,   default=8)
    ap.add_argument("--device",              default="cuda")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_num     = args.run_num if args.run_num is not None else _next_run_num(runs_dir)
    date_str    = datetime.now().strftime("%y%m%d")
    run_dir     = runs_dir / f"run_{run_num:03d}_{args.description}_{date_str}"

    if run_dir.exists():
        print(f"[init] ERROR: {run_dir} already exists.")
        raise SystemExit(1)

    run_name        = args.run_name or f"{args.description}_{run_num:03d}"
    max_tokens      = args.max_tokens if args.max_tokens > 0 else None
    preprocessed_dir  = runs_dir.parent / "01_preprocessed"
    preprocessed_h5ad = preprocessed_dir / "cytokine_ad_preprocessed.h5ad"

    print(f"\n[init] Experiment:        {run_dir.name}")
    print(f"[init] Run dir:           {run_dir}")
    print(f"[init] Shared preprocess: {preprocessed_dir}")
    print(f"[init] Run name:          {run_name}")
    print(f"[init] Log prefix:        r{run_num:03d}_{{step}}_{{jobid}}")
    print()

    # Create directories
    for sub in ["configs", "checkpoints", "logs", "scripts",
                "token_cache", "predictions", "figs", "metrics"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # configs/vae_train.yml
    train_cfg_path = run_dir / "configs" / "vae_train.yml"
    with train_cfg_path.open("w") as f:
        yaml.safe_dump(
            _vae_train_yml(preprocessed_h5ad, run_dir, run_name, args),
            f, sort_keys=False, allow_unicode=True,
        )
    print(f"  [init] wrote {run_dir.name}/configs/vae_train.yml")

    # sbatch scripts
    _write(run_dir / "scripts" / "00_preprocess.sbatch",
           _sbatch_preprocess(run_num, run_dir, preprocessed_dir, args.raw_adata,
                              run_name, args.cellinguist_dir, args.conda_env),
           run_dir)
    _write(run_dir / "scripts" / "01_precompute_tokens.sbatch",
           _sbatch_token_cache(run_num, run_dir, preprocessed_dir, run_name,
                               args.cellinguist_dir, args.conda_env,
                               max_tokens, 0.0, "gene", None, args.num_workers),
           run_dir)
    _write(run_dir / "scripts" / "02_train_vae.sbatch",
           _sbatch_train(run_num, run_dir, run_name, args.cellinguist_dir, args.conda_env),
           run_dir)
    _write(run_dir / "scripts" / "03_predict_cytokines.sbatch",
           _sbatch_predict(run_num, run_dir, preprocessed_dir, run_name,
                           args.cellinguist_dir, args.conda_env, args.predict_mode,
                           args.batch_size, args.num_workers, args.device),
           run_dir)
    _write(run_dir / "scripts" / "run_all.sh",
           _run_all_sh(run_num, run_dir),
           run_dir)

    print(f"\n[init] Done. Run dir: {run_dir}")
    print(f"[init] To launch:\n\n    cd {run_dir}\n    bash scripts/run_all.sh\n")


if __name__ == "__main__":
    main()
