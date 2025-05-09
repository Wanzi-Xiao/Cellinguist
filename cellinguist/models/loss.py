import torch
import torch.nn.functional as F

def compute_similarity_loss(cell_embeddings, threshold=0.8):
    """
    Computes a loss that encourages pairs of cell embeddings with cosine similarity above a threshold
    to be even more similar. This loss is defined as the average of (1 - cosine_similarity) for all
    pairs with similarity greater than the threshold.
    
    Args:
        cell_embeddings (torch.Tensor): Tensor of shape (batch_size, d_model) for cell embeddings.
        threshold (float): The cosine similarity threshold for selecting positive pairs.
    
    Returns:
        torch.Tensor: The similarity loss (a scalar).
    """
    # Normalize embeddings so that cosine similarity is equivalent to dot product.
    normalized = F.normalize(cell_embeddings, p=2, dim=1)
    # Compute the pairwise cosine similarity matrix (batch_size x batch_size).
    cos_sim = torch.mm(normalized, normalized.t())
    batch_size = cell_embeddings.size(0)
    # Create a mask for the upper triangular matrix, excluding the diagonal.
    triu_mask = torch.triu(torch.ones(batch_size, batch_size, dtype=torch.bool, device=cell_embeddings.device), diagonal=1)
    # Only use positive pairs
    cos_sim = F.relu(cos_sim)
    # Select pairs with cosine similarity above the threshold.
    positive_pairs = (cos_sim > threshold) & triu_mask
    if positive_pairs.sum() == 0:
        # If no pair exceeds the threshold, return 0 loss.
        return torch.tensor(0.0, device=cell_embeddings.device, requires_grad=True)
    # For selected pairs, loss is defined as (1 - cosine_similarity).
    loss = torch.mean(1 - (cos_sim - threshold) ** 2)
    return loss

def mse_loss_for_expression(logits: torch.Tensor, target: torch.Tensor, ignore_index: int):
    """
    Computes an MSE loss for a prediction task where the model outputs logits over discrete bins,
    and we wish to compute the expected value of the distribution.

    Args:
        logits (torch.Tensor): Tensor of shape (B, L, EXPRESSION_VOCAB_SIZE).
        target (torch.Tensor): Tensor of shape (B, L) containing target bin indices.
        ignore_index (int): Token value to ignore (e.g. PAD_TOKEN_ID).
    
    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    # Convert logits to probabilities along the last dimension.
    probs = F.softmax(logits, dim=-1)  # (B, L, EXPRESSION_VOCAB_SIZE)
    # Create a vector of bin indices. These are our "labels" in continuous form.
    # For instance, if EXPRESSION_VOCAB_SIZE is 131, then bins will be [0, 1, 2, ..., 130].
    # (If you reserved certain indices for special tokens, you may want to shift these values accordingly.)
    bins = torch.arange(0, logits.size(-1), device=logits.device).float()  # (EXPRESSION_VOCAB_SIZE,)
    # Compute the expected value: weighted sum of bins using the probabilities.
    # This yields a tensor of shape (B, L).
    expected = torch.sum(probs * bins, dim=-1)
    # Convert target to float.
    target_float = target.float()
    # Create a mask to ignore positions with ignore_index.
    valid_mask = target != ignore_index
    # Compute mean squared error only over valid positions.
    loss = F.mse_loss(expected[valid_mask], target_float[valid_mask])
    return loss