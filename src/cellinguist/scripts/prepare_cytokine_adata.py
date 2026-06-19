"""
Preprocess the cytokine h5ad for cytokine_vector VAE training.

Steps:
  1. Filter out cells where obs['cytokine'] == 'NA'
  2. Add one binary indicator column per non-PBS cytokine (sanitized name)
     PBS cells → all zeros; treated cells → 1 in their cytokine's column
  3. Write a clean h5ad to output_path
  4. Print a YAML snippet of cytokine_keys ready to paste into vae_train_cytokine.yml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


CONTROL_LABEL = "PBS"
NA_LABEL = "NA"


def sanitize_col(name: str) -> str:
    """Convert a cytokine display name to a safe pandas/YAML column name."""
    name = (
        name.replace("α", "alpha")
        .replace("β", "beta")
        .replace("γ", "gamma")
        .replace("δ", "delta")
        .replace("ε", "epsilon")
        .replace("ζ", "zeta")
        .replace("η", "eta")
        .replace("θ", "theta")
        .replace("κ", "kappa")
        .replace("λ", "lambda")
        .replace("μ", "mu")
        .replace("ρ", "rho")
        .replace("σ", "sigma")
        .replace("τ", "tau")
        .replace("ω", "omega")
    )
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess cytokine h5ad: filter NA, add binary cytokine columns."
    )
    ap.add_argument("input_path", help="Path to raw cytokine .h5ad file")
    ap.add_argument("output_path", help="Path for preprocessed .h5ad output")
    ap.add_argument(
        "--cytokine-key",
        default="cytokine",
        help="obs column with cytokine treatment labels (default: 'cytokine')",
    )
    ap.add_argument(
        "--control",
        default=CONTROL_LABEL,
        help=f"Control label in cytokine column (default: '{CONTROL_LABEL}')",
    )
    ap.add_argument(
        "--na-label",
        default=NA_LABEL,
        help=f"Label to filter out as NA (default: '{NA_LABEL}')",
    )
    args = ap.parse_args()

    input_path = args.input_path
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Reading {input_path} (backed mode)...")
    adata = ad.read_h5ad(input_path, backed="r")
    print(f"[prepare] Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # --- 1. Filter NA cells ---
    cytokine_series = adata.obs[args.cytokine_key].astype(str)
    na_mask = cytokine_series == args.na_label
    n_na = int(na_mask.sum())
    keep_mask = ~na_mask
    print(f"[prepare] Filtering {n_na} NA cells → {int(keep_mask.sum())} cells remain")

    # Materialize filtered adata (reads X into memory — needs ~2-4 GB RAM)
    print("[prepare] Materializing expression matrix (this may take a moment)...")
    adata_filtered = adata[keep_mask].to_memory()
    if adata.isbacked:
        adata.file.close()

    n_cells = adata_filtered.n_obs
    print(f"[prepare] Filtered adata: {n_cells} cells × {adata_filtered.n_vars} genes")

    # --- 2. Build binary cytokine indicator columns ---
    cytokine_labels = adata_filtered.obs[args.cytokine_key].astype(str).to_numpy()
    unique_labels = sorted(set(cytokine_labels))
    treatment_labels = [lbl for lbl in unique_labels if lbl != args.control]

    print(f"[prepare] Found {len(treatment_labels)} unique cytokine treatments (excluding PBS):")
    for lbl in treatment_labels:
        count = int((cytokine_labels == lbl).sum())
        print(f"  {lbl!r:30s}  n={count}")

    # Sanitize names and check for collisions
    sanitized = [sanitize_col(lbl) for lbl in treatment_labels]
    if len(set(sanitized)) != len(sanitized):
        seen: dict[str, list[str]] = {}
        for orig, san in zip(treatment_labels, sanitized):
            seen.setdefault(san, []).append(orig)
        collisions = {k: v for k, v in seen.items() if len(v) > 1}
        print(
            f"[prepare] ERROR: name sanitization produced collisions: {collisions}",
            file=sys.stderr,
        )
        sys.exit(1)

    existing_cols = set(adata_filtered.obs.columns)
    for san in sanitized:
        if san in existing_cols:
            print(
                f"[prepare] ERROR: sanitized column '{san}' already exists in obs. "
                "Rename the conflicting column before running this script.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Add indicator columns
    print(f"[prepare] Adding {len(treatment_labels)} binary cytokine indicator columns...")
    for orig, san in zip(treatment_labels, sanitized):
        indicator = (cytokine_labels == orig).astype(np.float32)
        adata_filtered.obs[san] = indicator

    n_pbs = int((cytokine_labels == args.control).sum())
    print(f"[prepare] PBS control cells (all-zero vector): {n_pbs}")

    # --- 3. Write output ---
    print(f"[prepare] Writing preprocessed h5ad to: {output_path}")
    adata_filtered.write_h5ad(str(output_path), compression="gzip")
    print(f"[prepare] Done. {adata_filtered.n_obs} cells written.")

    # --- 4. Print YAML snippet ---
    print("\n" + "=" * 60)
    print("Paste the following cytokine_keys block into vae_train_cytokine.yml:")
    print("=" * 60)
    print("cytokine_keys:")
    for san in sanitized:
        print(f"  - \"{san}\"")
    print("=" * 60)

    # Also write a .txt file next to the output for convenience
    keys_path = output_path.with_suffix("").with_suffix(".cytokine_keys.txt")
    with keys_path.open("w") as f:
        f.write("cytokine_keys:\n")
        for san in sanitized:
            f.write(f'  - "{san}"\n')
    print(f"[prepare] Cytokine keys also saved to: {keys_path}")


if __name__ == "__main__":
    main()
