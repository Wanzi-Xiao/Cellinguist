import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
from .loss import compute_similarity_loss, mse_loss_for_expression
import torch.nn.functional as F

class TokenEmbeddingLayer(nn.Module):
    def __init__(self,
                 gene_vocab_size: int,
                 expression_vocab_size: int,
                 condition_vocab_size: int,
                 pad_token_id: int,
                 library_vocab_size: int,  # For library size bins.
                 embedding_dim: int = 256,
                 max_seq_len: int = 512,
                 use_positional: bool = True
                 ):
        """
        This layer sums embeddings for gene IDs, expression tokens, condition tokens,
        and also adds an embedding for the library size covariate to the [CLS] token.
        """
        super(TokenEmbeddingLayer, self).__init__()
        self.gene_embeddings = nn.Embedding(gene_vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.expression_embeddings = nn.Embedding(expression_vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.use_condition = condition_vocab_size is not None
        if self.use_condition:
            # We expect a single condition token per sample.
            self.condition_embeddings = nn.Embedding(condition_vocab_size, embedding_dim)
        self.library_embeddings = nn.Embedding(library_vocab_size, embedding_dim)
        self.use_positional = use_positional
        if use_positional:
            self.positional_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.mix_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(embedding_dim)
        )
    def forward(self, gene_ids: torch.Tensor, expression_tokens: torch.Tensor,
                condition_tokens: torch.Tensor, library_size: torch.Tensor):
        B, L = gene_ids.shape
        D = self.gene_embeddings.embedding_dim
        gene_embed = self.gene_embeddings(gene_ids)
        expr_embed = self.expression_embeddings(expression_tokens)
        g_e_cat = torch.cat([gene_embed, expr_embed],dim=-1)
        combined = self.mix_layer(g_e_cat)
        if self.use_positional:
            batch_size, seq_length = gene_ids.size()
            positions = torch.arange(seq_length, device=gene_ids.device).unsqueeze(0).expand(batch_size, seq_length)
            pos_embed = self.positional_embeddings(positions)
            combined = combined + pos_embed
        # If condition embeddings are used, add them only to the [CLS] token.
        cls = combined[:, 0, :]
        if self.use_condition:
            if condition_tokens is None:
                # If no condition tokens are provided, use a zero tensor.
                cond_embed = torch.zeros(gene_ids.size(0), gene_embed.size(-1), device=gene_ids.device)
            else:
                cond_embed = self.condition_embeddings(condition_tokens)
            cls = cls + cond_embed
        # Add library size embedding to [CLS] token.
        lib_embed = self.library_embeddings(library_size)  # (batch_size, embedding_dim)
        cls = cls + lib_embed
        # Add CLS back
        combined = torch.cat([cls.unsqueeze(1), combined[:, 1:, :]], dim=1)
        return combined

def generate_causal_mask(seq_len: int, device: torch.device):
    """
    Generates a causal mask for a sequence of length `seq_len`.
    Each position can attend only to itself and previous tokens.
    
    Args:
        seq_len (int): The sequence length.
        device (torch.device): The device to create the mask on.
        
    Returns:
        torch.Tensor: A tensor of shape (seq_len, seq_len) with -inf for masked positions.
    """
    # Create a mask with -inf in the upper triangle (excluding the main diagonal)
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(device)
    return mask

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, causal: bool = False):
        """
        A Transformer encoder layer that uses Flash Attention via flash_attn_func.
        
        Args:
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
            causal (bool): Whether to use causal masking.
        """
        super(FlashTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        # Linear projections for Q, K, and V.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # Output projection.
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Feedforward network.
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x, key_padding_mask=None):
        batch_size, seq_length, d_model = x.size()
        d_head = d_model // self.nhead  # dimension per head
        # Project the inputs to Q, K, and V.
        Q = self.q_proj(x)  # (B, seq_length, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Reshape Q, K, V to (B, seq_length, nhead, d_head).
        Q = Q.view(batch_size, seq_length, self.nhead, d_head)
        K = K.view(batch_size, seq_length, self.nhead, d_head)
        V = V.view(batch_size, seq_length, self.nhead, d_head)
        # Option 1: Explicitly cast to half precision.
        Q = Q.half()
        K = K.half()
        V = V.half()
    # Option 2: Use autocast (comment out the explicit casting above and wrap with autocast)
    # with torch.cuda.amp.autocast(dtype=torch.float16):
    #     attn_output = flash_attn_func(Q, K, V, dropout_p=self.dropout.p, causal=self.causal)
        # Here we use explicit casting:
        attn_output = flash_attn_func(Q, K, V, dropout_p=self.dropout.p, causal=self.causal)
        # The output of flash_attn_func is in fp16; cast back to original precision if needed.
        attn_output = attn_output.to(x.dtype)
        # Reshape back to (B, seq_length, d_model)
        attn_output = attn_output.view(batch_size, self.nhead, seq_length, d_head)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, d_model)
        attn_output = self.out_proj(attn_output)
        # Residual connection and normalization.
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)
        return x
    
class MaskedGeneExpressionPredictionHead(nn.Module):
    def __init__(self, d_model: int, expression_vocab_size: int):
        """
        Args:
            d_model (int): The transformer embedding dimension.
            expression_vocab_size (int): The number of expression bins (size of the expression token vocabulary).
        """
        super(MaskedGeneExpressionPredictionHead, self).__init__()
        # A simple linear projection from the transformer output to the expression token logits.
        self.fc = nn.Linear(d_model, expression_vocab_size)
    def forward(self, token_outputs: torch.Tensor):
        """
        Args:
            token_outputs (torch.Tensor): Output embeddings for tokens to be predicted.
                                           Shape: (batch_size, seq_length, d_model)
        
        Returns:
            torch.Tensor: Logits for expression bin prediction.
                          Shape: (batch_size, seq_length, expression_vocab_size)
        """
        logits = self.fc(token_outputs)
        return logits

class WholeGenomeExpressionPredictionHead(nn.Module):
    def __init__(self, d_model: int, total_gene_count: int, expression_vocab_size: int):
        """
        Predicts binned expression values for every gene.
        Args:
            d_model: Transformer embedding dimension.
            total_gene_count: The total number of genes (i.e., number of columns in the expression matrix).
            expression_vocab_size: Vocabulary size for expression tokens.
        """
        super(WholeGenomeExpressionPredictionHead, self).__init__()
        self.fc = nn.Linear(d_model, total_gene_count * expression_vocab_size)
        self.total_gene_count = total_gene_count
        self.expression_vocab_size = expression_vocab_size
    def forward(self, cls_token: torch.Tensor):
        """
        Args:
            cls_token: Tensor of shape (batch_size, d_model) extracted from the [CLS] token.
        Returns:
            Tensor of shape (batch_size, total_gene_count, expression_vocab_size) with logits.
        """
        logits = self.fc(cls_token)
        logits = logits.view(-1, self.total_gene_count, self.expression_vocab_size)
        return logits
    
class MaskedGeneIDPredictionHead(nn.Module):
    def __init__(self, d_model: int, gene_vocab_size: int):
        """
        Args:
            d_model (int): The transformer embedding dimension.
            gene_vocab_size (int): The number of gene IDs in your vocabulary
                                   (including any reserved tokens).
        """
        super(MaskedGeneIDPredictionHead, self).__init__()
        # Linear projection from transformer hidden dim to gene‐ID logits
        self.fc = nn.Linear(d_model, gene_vocab_size)
    def forward(self, token_outputs: torch.Tensor):
        """
        Args:
            token_outputs (torch.Tensor): Output embeddings for each token
                                          Shape: (batch_size, seq_length, d_model)
        
        Returns:
            torch.Tensor: Logits for gene‐ID prediction at each position.
                          Shape: (batch_size, seq_length, gene_vocab_size)
        """
        return self.fc(token_outputs)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # Forward pass: return the input unchanged.
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient by multiplying with -lambda.
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains):
        """
        Args:
            input_dim (int): Dimension of the cell embeddings (e.g. d_model, such as 512).
            hidden_dim (int): Hidden layer dimension.
            num_domains (int): Number of domain classes (or sample-specific labels).
        """
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains)
        )
    def forward(self, x):
        return self.classifier(x)
    
class FullModel(nn.Module):
    def __init__(self, token_embedding_layer, encoder_layers, masked_head, whole_genome_head, domain_classifier, gene_id_head, lambda_value):
        super(FullModel, self).__init__()
        self.token_embedding_layer = token_embedding_layer
        self.encoder_layers = encoder_layers  # assume this is an nn.ModuleList
        self.masked_head = masked_head
        self.whole_genome_head = whole_genome_head
        self.domain_classifier = domain_classifier
        self.lambda_value = lambda_value
        self.gene_id_head = gene_id_head
    def forward(self, batch):
        """
        Expects batch to be a dictionary with keys:
         - 'gene_ids': (batch_size, seq_len)
         - 'expression_tokens': (batch_size, seq_len)
         - 'condition': (batch_size,)
         - 'library_size': (batch_size,)
         - 'whole_genome_target': (batch_size, num_genes)
        """
        gene_ids = batch["gene_ids"]
        expression_tokens = batch["expression_tokens"]
        condition = batch["condition"]
        library_size = batch["library_size"]
        whole_genome_target = batch["whole_genome_target"]  # not used in forward, only for loss
        # Compute token embeddings.
        x = self.token_embedding_layer(gene_ids, expression_tokens, condition, library_size)
        # Pass through each encoder layer.
        for layer in self.encoder_layers:
            x = layer(x)
        # Masked prediction logits.
        masked_logits = self.masked_head(x)
        # Whole genome prediction: extract [CLS] token.
        cls_token = x[:, 0, :]  # (batch_size, d_model)
        whole_genome_logits = self.whole_genome_head(cls_token)
        # Gene id prediction
        gene_id_logits = self.gene_id_head(x)
        # Domain classification 
        reversed_embeddings = grad_reverse(cls_token, lambda_=self.lambda_value)
        # Forward pass through the domain classifer
        domain_preds = self.domain_classifier(reversed_embeddings)
        return masked_logits, whole_genome_logits, cls_token, domain_preds, gene_id_logits
    
def get_random_mask_positions(token_tensor, mask_token_id: int, mask_fraction=0.2):
    """
    For each token in token_tensor, randomly select a fraction (mask_fraction)
    of tokens that are not already masked. Returns a boolean mask tensor of the same shape.
    """
    # Only consider tokens that are not already masked.
    not_masked = (token_tensor != mask_token_id)
    random_tensor = torch.rand(token_tensor.shape, device=token_tensor.device)
    mask_positions = (random_tensor < mask_fraction) & not_masked
    return mask_positions

def train_epoch_ddp(dataloader, ddp_model, optimizer, device, mask_token_id: int, pad_token_id: int, num_iterative_steps=3, mask_fraction=0.2):
    """
    Trains the model for one epoch using an iterative masking strategy.
    
    Args:
        dataloader: An iterable DataLoader yielding batches of gene_ids, expression_tokens, and condition_tokens.
        token_embedding_layer: The embedding module.
        transformer_encoder: The autoregressive transformer encoder.
        masked_head: The head predicting masked gene expression tokens.
        optimizer: The optimizer.
        device: torch.device (e.g., 'cuda' or 'cpu').
        num_iterative_steps: Number of iterative prediction steps per batch.
        mask_fraction: Fraction of tokens to mask in each iterative step.
    """
    ddp_model.train()
    total_loss = 0.0
    num_batches = 0 
    for batch in dataloader:
        # Move each part of the batch to the device.
        for key in batch:
            batch[key] = batch[key].to(device)
        # Create an input copy where the selected positions are replaced by MASK_TOKEN_ID
        current_expression_tokens = batch['expression_tokens'].clone()
        current_gene_id_tokens = batch["gene_ids"].clone()
        optimizer.zero_grad()
        for step in range(num_iterative_steps): 
            # Update the batch dictionary with the current expression tokens.
            mask_positions = get_random_mask_positions(current_expression_tokens, mask_fraction)
            # (The rest of the batch remains unchanged.)
            input_expression_tokens = current_expression_tokens
            input_gene_id_tokens = current_gene_id_tokens
            batch['expression_tokens'][mask_positions] = mask_token_id
            batch["gene_ids"][mask_positions] = mask_token_id
            # Forward pass: Embed tokens, process with the transformer encoder,
            # and obtain prediction logits for the expression tokens.
            masked_logits, whole_genome_logits, cls_token, domain_preds, gene_id_logits = ddp_model(batch)
            # Compute loss only on the masked positions
            if mask_positions.sum() > 0:
                loss_masked = mse_loss_for_expression(masked_logits[mask_positions], input_expression_tokens[mask_positions], ignore_index=pad_token_id)
                loss_gene_id = F.cross_entropy(gene_id_logits[mask_positions], input_gene_id_tokens[mask_positions],ignore_index=pad_token_id)
            else:
                loss_masked = torch.tensor(0.0, device=device)
                loss_gene_id = torch.tensor(0.0, device=device)
            # Compute loss on genome 
            loss_genome = mse_loss_for_expression(whole_genome_logits, batch["whole_genome_target"], ignore_index=pad_token_id)
            loss_similarity = compute_similarity_loss(cls_token, threshold = 0.95)
            # Compute a domain classification loss (e.g., cross-entropy) using the domain labels:
            loss_domain = F.cross_entropy(domain_preds, batch['domain'])
            # Combine losses
            loss = loss_masked*0.01 + loss_genome*0.01 + loss_similarity*10 + loss_domain + loss_gene_id*10
            loss.backward()
            optimizer.step()
            # Teacher forcing: Update the current expression tokens at masked positions with the model's predictions
            with torch.no_grad():
                probabilities = F.softmax(masked_logits, dim=-1)
                # Get confidence scores (max probability) for each token.
                confidence, _ = probabilities.max(dim=-1)  # shape: (batch_size, seq_length)
                # Among the masked positions, compute the threshold corresponding to the top 20% highest confidence.
                # (That is, we want tokens whose confidence is in the top 20% of the masked positions.)
                if mask_positions.sum() > 0:
                    threshold = torch.quantile(confidence[mask_positions], 0.8)
                else:
                    threshold = 1.0  # fallback (shouldn't happen if mask has any True)
                # Create an update mask: only update tokens that were masked AND whose confidence is >= threshold.
                update_mask = mask_positions & (confidence >= threshold)
                # Get the predicted tokens from the logits.
                predicted_tokens = masked_logits.argmax(dim=-1)
                # Update only those positions in current_expression_tokens where update_mask is True.
                current_expression_tokens[update_mask] = predicted_tokens[update_mask]
                # Detach to clear the computation graph for the next iteration.
                current_expression_tokens = current_expression_tokens.detach()
        total_loss += loss.item()
        num_batches += 1
    # Print loss 
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, loss_masked, loss_genome, loss_similarity, loss_domain, loss_gene_id
