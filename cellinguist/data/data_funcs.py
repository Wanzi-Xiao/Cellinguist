import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def bin_expression_counts(counts: np.ndarray, num_expression_bins: int, reserved_tokens_count: int):
    """
    Bins raw gene expression counts into discrete tokens.
    Only counts > 1 are considered.
    The resulting bin indices are shifted by +1 so that valid bins lie in [1, n_bins],
    leaving IDs above n_bins for special tokens.
    
    Args:
        counts (np.ndarray): 1D array of raw counts for expressed genes.
        n_bins (int): Number of bins.
        
    Returns:
        List[int]: Binned expression tokens.
    """
    # Filter: only counts > 1 (assume counts is already a 1D array for expressed genes)
    valid_counts = counts[counts > 1]
    if valid_counts.size == 0:
        return []
    
    # Compute quantile-based bin edges (n_bins+1 edges yields n_bins bins)
    bin_edges = np.percentile(valid_counts, np.linspace(0, 100, num_expression_bins + 1))
    
    # In degenerate case (all valid counts identical), assign them all to bin 1.
    if np.all(valid_counts == valid_counts[0]):
        binned = np.ones_like(valid_counts, dtype=int)
    else:
        # np.digitize returns bins in 0..(n_bins-1); shift by 1 so bins are in 1..n_bins.
        binned = np.digitize(valid_counts, bin_edges[1:-1], right=True)
        # Shift by RESERVED_TOKENS_COUNT so that tokens are in [3, n_bins+2].
        binned = binned + reserved_tokens_count
    return binned.tolist()

def bin_expression_counts_full(row: np.ndarray, num_expression_bins: int, reserved_tokens_count: int):
    """
    Bins the raw expression counts for a full cell (all genes) into discrete tokens.
    Unlike the above function, we do not filter out counts <= 1.
    All genes receive a binned token, which is then shifted by RESERVED_TOKENS_COUNT.
    """
    bin_edges = np.percentile(row, np.linspace(0, 100, num_expression_bins + 1))
    if np.all(row == row[0]):
        binned = np.ones_like(row, dtype=int)
    else:
        binned = np.digitize(row, bin_edges[1:-1], right=True)
    return (binned + reserved_tokens_count).tolist()

class SingleCellDatasetUnified(Dataset):
    """
    Expects an expression matrix of shape (num_cells, num_genes) and an optional array of condition labels.
    For each cell:
      - Only genes with raw counts > 1 are kept for masked prediction.
      - Gene IDs (for masked prediction) are taken as the column indices (shifted).
      - The corresponding raw counts are binned using bin_expression_counts.
      - A whole genome target is computed for every gene.
      - The number of expressed genes (library size) is computed and binned using global quantile binning.
    """
    def __init__(self, expression_matrix: np.ndarray, num_library_bins: int, num_expression_bins: int, reserved_tokens_count: int, condition_labels: np.ndarray = None, domain_labels: np.ndarray = None):
        self.expression_matrix = np.atleast_2d(np.asarray(expression_matrix))
        if condition_labels is not None:
            self.condition_labels = np.atleast_1d(np.asarray(condition_labels))
        else:
            self.condition_labels = None
        if domain_labels is not None:
            self.domain_labels = np.atleast_1d(np.asarray(domain_labels))
        self.num_library_bins = num_library_bins
        self.num_expression_bins = num_expression_bins
        self.reserved_tokens_count = reserved_tokens_count
        # Precompute library sizes for all cells
        library_sizes = []
        for i in range(self.expression_matrix.shape[0]):
            row = self.expression_matrix[i, :]
            expressed_indices = np.where(row > 1)[0]
            library_sizes.append(len(expressed_indices))
        library_sizes = np.array(library_sizes)
        # Compute global quantile bin edges for library sizes.
        # If all library sizes are the same, we'll simply use a single bin.
        if np.all(library_sizes == library_sizes[0]):
            self.library_bin_edges = None  # Will assign all to bin 0
        else:
            # Create bin edges for the desired number of bins.
            self.library_bin_edges = np.percentile(library_sizes, np.linspace(0, 100, self.num_library_bins + 1))
        self.reserved_tokens_count = reserved_tokens_count
    def __len__(self):
        return self.expression_matrix.shape[0]
    def __getitem__(self, idx):
        row = self.expression_matrix[idx, :]  # (num_genes,)
        # For masked prediction: select expressed gene indices (count > 1)
        expressed_indices = np.where(row > 1)[0]
        if expressed_indices.size == 0:
            gene_ids = []
            expression_tokens = []
        else:
            # Shift gene IDs by RESERVED_TOKENS_COUNT.
            gene_ids = (expressed_indices + self.reserved_tokens_count).tolist()
            raw_counts = row[expressed_indices]
            expression_tokens = bin_expression_counts(raw_counts, num_expression_bins=self.num_expression_bins, reserved_tokens_count=self.reserved_tokens_count)
        # For whole genome prediction:
        whole_genome_target = bin_expression_counts_full(row, num_expression_bins=self.num_expression_bins, reserved_tokens_count=self.reserved_tokens_count)
        # Compute library size (number of expressed genes)
        library_size = len(expressed_indices)
        # Global binning: use precomputed global edges if available
        if self.library_bin_edges is None:
            library_bin = 0
        else:
            # np.digitize returns a bin index in the range [0, num_library_bins-1].
            library_bin = int(np.digitize(library_size, self.library_bin_edges[1:-1], right=True))
        # Get condition label if provided; otherwise default to 0.
        if self.condition_labels is not None:
            condition = int(self.condition_labels[idx].item()) if isinstance(self.condition_labels[idx], np.generic) else int(self.condition_labels[idx])
        else:
            condition = 0
        # Get domain label if provided; otherwise default to 0.
        if self.domain_labels is not None:
            domain = int(self.domain_labels[idx].item()) if isinstance(self.domain_labels[idx], np.generic) else int(self.domain_labels[idx])
        else:
            domain = 0
        return {
            "gene_ids": gene_ids,                 # For masked prediction.
            "expression_tokens": expression_tokens,  # For masked prediction.
            "whole_genome_target": whole_genome_target,  # For whole genome prediction.
            "condition": condition,               # Cell-level condition label.
            "library_size": library_bin,           # Global library size bin.
            "domain": domain,          # Domain label for gradient reversal.
        }
    
def collate_fn_unified(samples, cls_token_id: int, pad_token_id: int):
    """
    Collate function to combine a list of samples into a batch.
    For each sample:
      - Prepend the [CLS] token to the gene_ids and expression_tokens sequences.
      - Pad all sequences to the maximum length in the batch using the [PAD] token.
      - Collect condition labels into a tensor.
    """
    gene_ids_list = []
    expr_tokens_list = []
    whole_genome_targets = []
    conditions = []
    library_sizes = []
    domain = []
    for sample in samples:
        # Prepend [CLS] to gene_ids.
        gene_ids = [cls_token_id] + sample["gene_ids"]
        gene_ids_list.append(torch.tensor(gene_ids, dtype=torch.long))
        # Prepend [CLS] to expression_tokens.
        expr_tokens = [cls_token_id] + sample["expression_tokens"]
        expr_tokens_list.append(torch.tensor(expr_tokens, dtype=torch.long))
        cond = sample.get("condition", 0)
        if cond is None:
            cond = 0
        conditions.append(cond)
        dom = sample.get("domain", 0)
        if dom is None:
            dom = 0
        domain.append(dom)
        whole_genome_targets.append(
            torch.tensor(sample["whole_genome_target"], dtype=torch.long)
        )
        lib_size = sample.get("library_size")
        library_sizes.append(lib_size)
    padded_gene_ids = pad_sequence(gene_ids_list, batch_first=True, padding_value=pad_token_id)
    padded_expr_tokens = pad_sequence(expr_tokens_list, batch_first=True, padding_value=pad_token_id)
    conditions = torch.tensor(conditions, dtype=torch.long)
    domain = torch.tensor(domain, dtype=torch.long)
    # Stack whole genome targets; they are assumed to be the same length (num_genes).
    whole_genome_targets = torch.stack(whole_genome_targets, dim=0)
    library_bins = torch.tensor(library_sizes, dtype=torch.long)
    return {
        "gene_ids": padded_gene_ids,             # (batch_size, max_seq_len)
        "expression_tokens": padded_expr_tokens,   # (batch_size, max_seq_len)
        "whole_genome_target": whole_genome_targets,  # (batch_size, num_genes)
        "condition": conditions,                  # (batch_size,)
        "library_size": library_bins,              # (batch_size,) -- binned library size
        "domain": domain # domain for gradient reversal
    }
