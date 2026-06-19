"""
Create a counterfactual override TSV for predict_cytokine_treatment.py.

The override TSV has one row per cell and one column per cytokine key.
It specifies what cytokine vector to inject for each cell during inference,
replacing whatever that cell actually received during training.

Common use cases:
  --mode all_treated:  set every cell to 1 for --target-cytokine (simulate universal treatment)
  --mode pbs_only:     only include PBS cells, set to 1 for --target-cytokine
  --mode zero:         set all cells to 0 (simulate PBS / no treatment)

Note: the output TSV columns must match cytokine_keys exactly (same order).
      Use the .cytokine_keys.txt file produced by prepare_cytokine_adata.py.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml


def _load_cytokine_keys(keys_file: str) -> list[str]:
    with open(keys_file) as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "cytokine_keys" in data:
        return [str(k) for k in data["cytokine_keys"]]
    if isinstance(data, list):
        return [str(k) for k in data]
    raise ValueError(
        f"Could not parse cytokine_keys from {keys_file}. "
        "Expected a YAML file with a 'cytokine_keys' list."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a counterfactual override TSV for cytokine prediction."
    )
    ap.add_argument(
        "adata_path",
        help="Preprocessed h5ad (output of prepare_cytokine_adata.py)",
    )
    ap.add_argument("output_tsv", help="Output path for the override TSV (may end in .tsv or .tsv.gz)")
    ap.add_argument(
        "--cytokine-keys-file",
        required=True,
        help="Path to the .cytokine_keys.txt file produced by prepare_cytokine_adata.py",
    )
    ap.add_argument(
        "--mode",
        choices=["all_treated", "pbs_only", "zero"],
        default="all_treated",
        help=(
            "Which cells to include and what vector to set:\n"
            "  all_treated — all cells, set target cytokine = 1, rest = 0\n"
            "  pbs_only    — only PBS cells, set target cytokine = 1\n"
            "  zero        — all cells, all cytokines = 0 (baseline)"
        ),
    )
    ap.add_argument(
        "--target-cytokine",
        default=None,
        help="Sanitized cytokine column name to set to 1.0 (required for modes all_treated and pbs_only)",
    )
    ap.add_argument(
        "--cytokine-key",
        default="cytokine",
        help="obs column with original cytokine labels (used for pbs_only filter, default: 'cytokine')",
    )
    ap.add_argument(
        "--control",
        default="PBS",
        help="Control label in original cytokine column (default: 'PBS')",
    )
    ap.add_argument(
        "--cell-type",
        default=None,
        help="If set, restrict to cells of this cell type (matches adata.obs['cell_types'])",
    )
    ap.add_argument(
        "--cell-type-key",
        default="cell_types",
        help="obs column for cell types (default: 'cell_types')",
    )
    args = ap.parse_args()

    if args.mode in ("all_treated", "pbs_only") and args.target_cytokine is None:
        ap.error("--target-cytokine is required for modes all_treated and pbs_only")

    cytokine_keys = _load_cytokine_keys(args.cytokine_keys_file)
    print(f"[override] Loaded {len(cytokine_keys)} cytokine keys")

    if args.target_cytokine is not None and args.target_cytokine not in cytokine_keys:
        raise ValueError(
            f"--target-cytokine '{args.target_cytokine}' not found in cytokine_keys. "
            f"Available: {cytokine_keys}"
        )

    print(f"[override] Reading cell IDs from {args.adata_path}...")
    adata = ad.read_h5ad(args.adata_path, backed="r")
    obs = adata.obs.copy()
    if adata.isbacked:
        adata.file.close()

    # --- Cell selection ---
    mask = np.ones(len(obs), dtype=bool)

    if args.mode == "pbs_only":
        pbs_mask = obs[args.cytokine_key].astype(str) == args.control
        mask &= pbs_mask.to_numpy()
        print(f"[override] pbs_only: {int(mask.sum())} PBS cells selected")

    if args.cell_type is not None:
        ct_mask = obs[args.cell_type_key].astype(str) == args.cell_type
        mask &= ct_mask.to_numpy()
        print(f"[override] cell_type filter '{args.cell_type}': {int(mask.sum())} cells remain")

    selected_obs = obs[mask]
    cell_ids = selected_obs.index.astype(str).tolist()
    n_cells = len(cell_ids)
    print(f"[override] Building override TSV for {n_cells} cells, mode='{args.mode}'")

    # --- Build override matrix ---
    n_cytokines = len(cytokine_keys)
    mat = np.zeros((n_cells, n_cytokines), dtype=np.float32)

    if args.mode != "zero" and args.target_cytokine is not None:
        target_idx = cytokine_keys.index(args.target_cytokine)
        mat[:, target_idx] = 1.0
        print(f"[override] Setting '{args.target_cytokine}' = 1.0 for all {n_cells} cells")

    df = pd.DataFrame(mat, columns=cytokine_keys)
    df.insert(0, "cell_id", cell_ids)

    # --- Write ---
    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if str(out_path).endswith(".gz"):
        with gzip.open(str(out_path), "wt") as f:
            df.to_csv(f, sep="\t", index=False)
    else:
        df.to_csv(str(out_path), sep="\t", index=False)

    print(f"[override] Wrote {n_cells} × {n_cytokines + 1} override TSV to: {out_path}")
    print(f"[override] Columns: cell_id + {cytokine_keys[:3]}{'...' if len(cytokine_keys) > 3 else ''}")


if __name__ == "__main__":
    main()
