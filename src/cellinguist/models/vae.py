from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        layers = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, float(lambda_))


class BatchAdversary(nn.Module):
    """
    Predict batch labels from latent z using a small MLP.
    Used with gradient reversal so encoder learns batch-invariant features.
    """

    def __init__(
        self,
        latent_dim: int,
        n_batches: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.classifier = MLP(
            input_dim=int(latent_dim),
            output_dim=int(n_batches),
            hidden_dim=int(hidden_dim),
            n_hidden_layers=int(n_hidden_layers),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


class PerturbationProjector(nn.Module):
    """
    Projects a cytokine perturbation feature vector into a dense embedding.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, perturb_vec: torch.Tensor) -> torch.Tensor:
        if perturb_vec.ndim != 2 or perturb_vec.shape[1] != self.input_dim:
            raise ValueError(
                f"perturb_vec must have shape (B, {self.input_dim}), got {tuple(perturb_vec.shape)}."
            )
        return self.net(perturb_vec)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = int(d_model * ff_mult)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(latents)
        kv = self.norm_kv(inputs)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        latents = latents + attn_out
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.norm_attn(x)
        attn_out, _ = self.attn(
            qkv,
            qkv,
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


# ---------------------------------------------------------------------------
# CBOWCellEncoder: expression-weighted CBOW + MLP to (mu, logvar)
# ---------------------------------------------------------------------------

class CBOWCellEncoder(nn.Module):
    """
    Cell encoder that:
      1) Uses expression-weighted CBOW to get a cell-level representation:
         h_cell = X @ E, where X is (B, G) expression and E is (G, d_gene_emb).
      2) Optionally concatenates a condition embedding.
      3) Passes through an MLP to produce (mu, logvar) for a VAE.

    X should be some transformed expression (e.g. log1p-normalized counts),
    and the decoder's reconstruction target should match that transform.
    """

    def __init__(
        self,
        gene_embeddings: torch.Tensor,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
        perturbation_dim: Optional[int] = None,
        perturb_emb_dim: int = 32,
        perturb_condition_encoder: bool = True,
        freeze_gene_embeddings: bool = True,
        input_transform: str = "log1p",   # "log1p" or "none"
    ) -> None:
        super().__init__()

        # gene_embeddings: (n_genes, d_gene)
        n_genes, d_gene = gene_embeddings.shape
        self.n_genes = n_genes
        self.d_gene = d_gene
        self.latent_dim = latent_dim

        # Wrap CBOW gene embeddings in an Embedding-like module
        # We'll use matmul directly, but keep as Parameter/Buffer for convenience.
        self.gene_embedding = nn.Embedding(
            num_embeddings=n_genes,
            embedding_dim=d_gene,
        )
        self.gene_embedding.weight.data.copy_(gene_embeddings)
        if freeze_gene_embeddings:
            self.gene_embedding.weight.requires_grad_(False)

        # Condition embedding (optional)
        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        self.perturb_condition_encoder = bool(perturb_condition_encoder)
        if perturbation_dim is not None and self.perturb_condition_encoder:
            self.perturb_projector = PerturbationProjector(
                input_dim=int(perturbation_dim),
                output_dim=int(perturb_emb_dim),
            )
            perturb_input_dim = int(perturb_emb_dim)
        else:
            self.perturb_projector = None
            perturb_input_dim = 0

        encoder_input_dim = d_gene + cond_input_dim + perturb_input_dim

        # MLP to latent parameters
        self.mlp_mu = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
        self.mlp_logvar = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

        self.input_transform = input_transform

    def forward(self, x_expr, cond_idx=None, perturb_vec=None, **kwargs):
        B, G = x_expr.shape
        assert G == self.n_genes

        # 1) Check raw input
        if torch.isnan(x_expr).any() or torch.isinf(x_expr).any():
            print("NaNs/Infs in x_expr!")

        # 2) Apply transform for encoder
        if self.input_transform == "log1p":
            x_enc = torch.log1p(x_expr)
        elif self.input_transform == "none":
            x_enc = x_expr
        else:
            raise ValueError(f"Unsupported input_transform: {self.input_transform}")

        if torch.isnan(x_enc).any() or torch.isinf(x_enc).any():
            print("NaNs/Infs after log1p in x_enc")

        # 3) Expression-weighted CBOW
        E = self.gene_embedding.weight  # (G, d_gene)
        if torch.isnan(E).any() or torch.isinf(E).any():
            print("NaNs/Infs in gene_embeddings!")

        h_cell = x_enc @ E  # (B, d_gene)

        if torch.isnan(h_cell).any() or torch.isinf(h_cell).any():
            print("NaNs/Infs after matmul (h_cell)")

        # 4) Optional condition embedding
        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([h_cell, c], dim=-1)
        else:
            h_in = h_cell

        if self.perturb_projector is not None:
            if perturb_vec is None:
                raise ValueError("perturb_vec is required when perturbation_dim is configured.")
            p = self.perturb_projector(perturb_vec.to(dtype=h_in.dtype, device=h_in.device))
            h_in = torch.cat([h_in, p], dim=-1)

        if torch.isnan(h_in).any() or torch.isinf(h_in).any():
            print("NaNs/Infs in h_in BEFORE MLP")

        # 5) Check MLP parameters
        for name, p in self.mlp_mu.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print("NaNs/Infs in mlp_mu parameter:", name)
        for name, p in self.mlp_logvar.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print("NaNs/Infs in mlp_logvar parameter:", name)

        mu = self.mlp_mu(h_in)
        logvar = self.mlp_logvar(h_in)

        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            print("NaNs in mu/logvar AFTER MLP")

        return mu, logvar


class PerceiverCellEncoder(nn.Module):
    """
    Perceiver-style cell encoder:
      1) Builds per-gene input tokens from expression + learned gene-id embeddings.
      2) Uses latent queries with cross-attention to genes.
      3) Applies latent self-attention blocks.
      4) Pools latents and maps to (mu, logvar) for the VAE posterior.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
        perturbation_dim: Optional[int] = None,
        perturb_emb_dim: int = 32,
        perturb_condition_encoder: bool = True,
        input_transform: str = "log1p",
        library_norm: str = "size_factor",
        library_norm_target_sum: float = 1e4,
        library_norm_eps: float = 1e-8,
        perceiver_d_model: int = 256,
        perceiver_num_latents: int = 64,
        perceiver_num_cross_attn_heads: int = 8,
        perceiver_num_self_attn_heads: int = 8,
        perceiver_num_self_attn_layers: int = 4,
        perceiver_ff_mult: int = 4,
        perceiver_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.input_transform = input_transform
        self.library_norm = str(library_norm).lower()
        self.library_norm_target_sum = float(library_norm_target_sum)
        self.library_norm_eps = float(library_norm_eps)
        if self.library_norm not in {"size_factor", "none"}:
            raise ValueError(f"Unsupported library_norm: {library_norm}")
        if self.library_norm_target_sum <= 0:
            raise ValueError("library_norm_target_sum must be > 0.")
        if self.library_norm_eps <= 0:
            raise ValueError("library_norm_eps must be > 0.")

        d_model = int(perceiver_d_model)
        n_latents = int(perceiver_num_latents)

        if d_model % int(perceiver_num_cross_attn_heads) != 0:
            raise ValueError("perceiver_d_model must be divisible by perceiver_num_cross_attn_heads.")
        if d_model % int(perceiver_num_self_attn_heads) != 0:
            raise ValueError("perceiver_d_model must be divisible by perceiver_num_self_attn_heads.")

        self.expr_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gene_embedding = nn.Embedding(self.n_genes, d_model)
        self.register_buffer("gene_indices", torch.arange(self.n_genes, dtype=torch.long))

        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.cross_attn = CrossAttentionBlock(
            d_model=d_model,
            n_heads=int(perceiver_num_cross_attn_heads),
            ff_mult=int(perceiver_ff_mult),
            dropout=float(perceiver_dropout),
        )
        self.self_attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    d_model=d_model,
                    n_heads=int(perceiver_num_self_attn_heads),
                    ff_mult=int(perceiver_ff_mult),
                    dropout=float(perceiver_dropout),
                )
                for _ in range(int(perceiver_num_self_attn_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        self.perturb_condition_encoder = bool(perturb_condition_encoder)
        if perturbation_dim is not None and self.perturb_condition_encoder:
            self.perturb_projector = PerturbationProjector(
                input_dim=int(perturbation_dim),
                output_dim=int(perturb_emb_dim),
            )
            perturb_input_dim = int(perturb_emb_dim)
        else:
            self.perturb_projector = None
            perturb_input_dim = 0

        encoder_input_dim = d_model + cond_input_dim + perturb_input_dim
        self.mlp_mu = MLP(
            input_dim=encoder_input_dim,
            output_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
        self.mlp_logvar = MLP(
            input_dim=encoder_input_dim,
            output_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_genes = x_expr.shape
        if n_genes != self.n_genes:
            raise ValueError(
                f"Expected {self.n_genes} genes, but got {n_genes}. "
                "Dataset gene order/shape does not match encoder setup."
            )

        if self.library_norm == "size_factor":
            libsize = x_expr.sum(dim=1, keepdim=True).clamp_min(self.library_norm_eps)
            x_norm = x_expr * (self.library_norm_target_sum / libsize)
            x_enc = torch.log1p(x_norm)
        else:
            if self.input_transform == "log1p":
                x_enc = torch.log1p(x_expr)
            elif self.input_transform == "none":
                x_enc = x_expr
            else:
                raise ValueError(f"Unsupported input_transform: {self.input_transform}")

        expr_tokens = self.expr_projection(x_enc.unsqueeze(-1))  # (B, G, D)
        gene_tokens = self.gene_embedding(self.gene_indices).unsqueeze(0)  # (1, G, D)
        inputs = expr_tokens + gene_tokens

        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)  # (B, L, D)
        latents = self.cross_attn(latents, inputs)
        for block in self.self_attn_blocks:
            latents = block(latents)

        h_cell = self.final_norm(latents.mean(dim=1))  # (B, D)

        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([h_cell, c], dim=-1)
        else:
            h_in = h_cell

        if self.perturb_projector is not None:
            if perturb_vec is None:
                raise ValueError("perturb_vec is required when perturbation_dim is configured.")
            p = self.perturb_projector(perturb_vec.to(dtype=h_in.dtype, device=h_in.device))
            h_in = torch.cat([h_in, p], dim=-1)

        mu = self.mlp_mu(h_in)
        logvar = self.mlp_logvar(h_in)
        return mu, logvar


class TransformerCellEncoder(nn.Module):
    """
    Transformer cell encoder:
      1) Build variable-length per-cell tokens from expressed genes only.
      2) Token = MLP([gene_id_embedding, expression_scalar]).
      3) Prepend a learned CLS token and run Transformer self-attention.
      4) Use CLS hidden state as pooled cell representation.
      5) Map to (mu, logvar) for VAE posterior.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
        perturbation_dim: Optional[int] = None,
        perturb_emb_dim: int = 32,
        perturb_condition_encoder: bool = True,
        input_transform: str = "log1p",
        transformer_d_model: int = 256,
        transformer_n_heads: int = 8,
        transformer_n_layers: int = 4,
        transformer_ff_mult: int = 4,
        transformer_dropout: float = 0.0,
        token_mlp_hidden_dim: int = 256,
        token_mlp_layers: int = 2,
        max_tokens_per_cell: Optional[int] = None,
        min_expr_for_token: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.latent_dim = int(latent_dim)
        self.input_transform = str(input_transform)
        if self.input_transform not in {"log1p", "none"}:
            raise ValueError(f"Unsupported input_transform: {input_transform}")
        self.max_tokens_per_cell = None if max_tokens_per_cell is None else int(max_tokens_per_cell)
        if self.max_tokens_per_cell is not None and self.max_tokens_per_cell <= 0:
            raise ValueError("max_tokens_per_cell must be > 0 when provided.")
        self.min_expr_for_token = float(min_expr_for_token)

        d_model = int(transformer_d_model)
        n_heads = int(transformer_n_heads)
        if d_model % n_heads != 0:
            raise ValueError("transformer_d_model must be divisible by transformer_n_heads.")
        if int(transformer_n_layers) <= 0:
            raise ValueError("transformer_n_layers must be > 0.")
        if int(token_mlp_layers) < 0:
            raise ValueError("token_mlp_layers must be >= 0.")

        self.gene_embedding = nn.Embedding(self.n_genes, d_model)
        self.token_mlp = MLP(
            input_dim=d_model + 1,
            output_dim=d_model,
            hidden_dim=int(token_mlp_hidden_dim),
            n_hidden_layers=int(token_mlp_layers),
            activation=nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Embedding(self.n_genes + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * int(transformer_ff_mult),
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=int(transformer_n_layers),
        )
        self.final_norm = nn.LayerNorm(d_model)

        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        self.perturb_condition_encoder = bool(perturb_condition_encoder)
        if perturbation_dim is not None and self.perturb_condition_encoder:
            self.perturb_projector = PerturbationProjector(
                input_dim=int(perturbation_dim),
                output_dim=int(perturb_emb_dim),
            )
            perturb_input_dim = int(perturb_emb_dim)
        else:
            self.perturb_projector = None
            perturb_input_dim = 0

        encoder_input_dim = d_model + cond_input_dim + perturb_input_dim
        self.mlp_mu = MLP(
            input_dim=encoder_input_dim,
            output_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )
        self.mlp_logvar = MLP(
            input_dim=encoder_input_dim,
            output_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def _expr_for_tokens(self, x_expr: torch.Tensor) -> torch.Tensor:
        if self.input_transform == "log1p":
            return torch.log1p(x_expr.clamp_min(0.0))
        if self.input_transform == "none":
            return x_expr
        raise ValueError(f"Unsupported input_transform: {self.input_transform}")

    def _build_token_batch(
        self,
        x_expr: torch.Tensor,
        token_gene_idx: Optional[torch.Tensor] = None,
        token_gene_mask: Optional[torch.Tensor] = None,
        return_token_metadata: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_genes = x_expr.shape
        if n_genes != self.n_genes:
            raise ValueError(
                f"Expected {self.n_genes} genes, but got {n_genes}. "
                "Dataset gene order/shape does not match encoder setup."
            )

        x_tok = self._expr_for_tokens(x_expr)
        d_model = self.gene_embedding.embedding_dim

        if token_gene_idx is None:
            selected_indices: list[torch.Tensor] = []
            max_seq_len = 1  # CLS only fallback

            for i in range(bsz):
                vals_raw = x_expr[i]
                keep = vals_raw > self.min_expr_for_token
                idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
                if idx.numel() > 0 and self.max_tokens_per_cell is not None and idx.numel() > self.max_tokens_per_cell:
                    vals_keep = vals_raw[idx]
                    topk = torch.topk(vals_keep, k=self.max_tokens_per_cell, largest=True).indices
                    idx = idx[topk]
                selected_indices.append(idx)
                seq_len = int(idx.numel()) + 1
                if seq_len > max_seq_len:
                    max_seq_len = seq_len

            tokens = x_expr.new_zeros((bsz, max_seq_len, d_model))
            key_padding_mask = torch.ones((bsz, max_seq_len), dtype=torch.bool, device=x_expr.device)
            token_gene_idx_full = torch.full(
                (bsz, max_seq_len),
                fill_value=-1,
                dtype=torch.long,
                device=x_expr.device,
            )
            token_gene_mask_full = torch.zeros(
                (bsz, max_seq_len),
                dtype=torch.bool,
                device=x_expr.device,
            )

            cls = self.cls_token.to(dtype=tokens.dtype, device=tokens.device).squeeze(0).squeeze(0)
            cls_pos = self.pos_embedding(
                torch.tensor([0], dtype=torch.long, device=x_expr.device)
            ).squeeze(0).to(dtype=tokens.dtype, device=tokens.device)

            for i, idx in enumerate(selected_indices):
                tokens[i, 0, :] = cls + cls_pos
                key_padding_mask[i, 0] = False
                if idx.numel() == 0:
                    continue

                expr_vals = x_tok[i, idx].unsqueeze(-1).to(dtype=tokens.dtype)
                gene_emb = self.gene_embedding(idx)
                gene_pos = self.pos_embedding(idx + 1)
                token_in = torch.cat([gene_emb, expr_vals], dim=-1)
                token_vec = self.token_mlp(token_in) + gene_pos
                seq_len = int(idx.numel())
                tokens[i, 1 : seq_len + 1, :] = token_vec
                key_padding_mask[i, 1 : seq_len + 1] = False
                token_gene_idx_full[i, 1 : seq_len + 1] = idx
                token_gene_mask_full[i, 1 : seq_len + 1] = True

            if return_token_metadata:
                return tokens, key_padding_mask, token_gene_idx_full, token_gene_mask_full
            return tokens, key_padding_mask

        token_gene_idx = token_gene_idx.to(device=x_expr.device, dtype=torch.long)
        if token_gene_mask is None:
            token_gene_mask = torch.ones_like(token_gene_idx, dtype=torch.bool, device=x_expr.device)
        else:
            token_gene_mask = token_gene_mask.to(device=x_expr.device, dtype=torch.bool)

        if token_gene_idx.ndim != 2 or token_gene_mask.ndim != 2:
            raise ValueError("token_gene_idx and token_gene_mask must have shape (B, L).")
        if token_gene_idx.shape != token_gene_mask.shape:
            raise ValueError("token_gene_idx and token_gene_mask must have the same shape.")
        if token_gene_idx.shape[0] != bsz:
            raise ValueError("token_gene_idx batch dimension must match x_expr batch size.")

        max_l = int(token_gene_idx.shape[1])
        if max_l == 0:
            tokens = x_expr.new_zeros((bsz, 1, d_model))
            key_padding_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=x_expr.device)
            cls = self.cls_token.to(dtype=tokens.dtype, device=tokens.device)
            cls_pos = self.pos_embedding(
                torch.tensor([0], dtype=torch.long, device=x_expr.device)
            ).view(1, 1, -1).to(dtype=tokens.dtype, device=tokens.device)
            tokens[:, :1, :] = cls + cls_pos
            if return_token_metadata:
                token_gene_idx_full = torch.full(
                    (bsz, 1),
                    fill_value=-1,
                    dtype=torch.long,
                    device=x_expr.device,
                )
                token_gene_mask_full = torch.zeros((bsz, 1), dtype=torch.bool, device=x_expr.device)
                return tokens, key_padding_mask, token_gene_idx_full, token_gene_mask_full
            return tokens, key_padding_mask

        idx_safe = token_gene_idx.clamp(min=0, max=self.n_genes - 1)
        expr_vals = torch.gather(x_tok, 1, idx_safe).unsqueeze(-1)
        gene_emb = self.gene_embedding(idx_safe)
        pos_emb = self.pos_embedding(idx_safe + 1)
        token_in = torch.cat([gene_emb, expr_vals.to(dtype=gene_emb.dtype)], dim=-1)

        flat = token_in.view(-1, token_in.shape[-1])
        token_vec = self.token_mlp(flat).view(bsz, max_l, d_model) + pos_emb
        token_vec = token_vec * token_gene_mask.unsqueeze(-1).to(dtype=token_vec.dtype)

        tokens = x_expr.new_zeros((bsz, max_l + 1, d_model))
        key_padding_mask = torch.ones((bsz, max_l + 1), dtype=torch.bool, device=x_expr.device)
        cls = self.cls_token.to(dtype=tokens.dtype, device=tokens.device)
        cls_pos = self.pos_embedding(
            torch.tensor([0], dtype=torch.long, device=x_expr.device)
        ).view(1, 1, -1).to(dtype=tokens.dtype, device=tokens.device)
        tokens[:, :1, :] = cls + cls_pos
        tokens[:, 1:, :] = token_vec
        key_padding_mask[:, 0] = False
        key_padding_mask[:, 1:] = ~token_gene_mask
        if return_token_metadata:
            token_gene_idx_full = torch.full(
                (bsz, max_l + 1),
                fill_value=-1,
                dtype=torch.long,
                device=x_expr.device,
            )
            token_gene_idx_full[:, 1:] = idx_safe
            token_gene_mask_full = torch.zeros((bsz, max_l + 1), dtype=torch.bool, device=x_expr.device)
            token_gene_mask_full[:, 1:] = token_gene_mask
            return tokens, key_padding_mask, token_gene_idx_full, token_gene_mask_full
        return tokens, key_padding_mask

    def _run_transformer(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not return_attention:
            if self.activation_checkpointing:
                from torch.utils.checkpoint import checkpoint as ckpt
                hidden = tokens
                for layer in self.transformer.layers:
                    hidden = ckpt(layer, hidden, None, key_padding_mask, use_reentrant=False)
                if self.transformer.norm is not None:
                    hidden = self.transformer.norm(hidden)
            else:
                hidden = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
            return hidden, None

        hidden = tokens
        attn_by_layer: list[torch.Tensor] = []
        for layer in self.transformer.layers:
            if layer.norm_first:
                qkv = layer.norm1(hidden)
                attn_out, attn_w = layer.self_attn(
                    qkv,
                    qkv,
                    qkv,
                    attn_mask=None,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                hidden = hidden + layer.dropout1(attn_out)
                ff_in = layer.norm2(hidden)
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_in))))
                hidden = hidden + layer.dropout2(ff)
            else:
                attn_out, attn_w = layer.self_attn(
                    hidden,
                    hidden,
                    hidden,
                    attn_mask=None,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                hidden = layer.norm1(hidden + layer.dropout1(attn_out))
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))
                hidden = layer.norm2(hidden + layer.dropout2(ff))
            attn_by_layer.append(attn_w)

        if self.transformer.norm is not None:
            hidden = self.transformer.norm(hidden)

        if len(attn_by_layer) == 0:
            return hidden, None
        return hidden, torch.stack(attn_by_layer, dim=1)

    def forward(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
        token_gene_idx: Optional[torch.Tensor] = None,
        token_gene_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, key_padding_mask = self._build_token_batch(
            x_expr,
            token_gene_idx=token_gene_idx,
            token_gene_mask=token_gene_mask,
        )
        hidden, _ = self._run_transformer(
            tokens=tokens,
            key_padding_mask=key_padding_mask,
            return_attention=False,
        )
        h_cell = self.final_norm(hidden[:, 0, :])

        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([h_cell, c], dim=-1)
        else:
            h_in = h_cell

        if self.perturb_projector is not None:
            if perturb_vec is None:
                raise ValueError("perturb_vec is required when perturbation_dim is configured.")
            p = self.perturb_projector(perturb_vec.to(dtype=h_in.dtype, device=h_in.device))
            h_in = torch.cat([h_in, p], dim=-1)

        mu = self.mlp_mu(h_in)
        logvar = self.mlp_logvar(h_in)
        return mu, logvar

    def forward_with_attention(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
        token_gene_idx: Optional[torch.Tensor] = None,
        token_gene_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        tokens, key_padding_mask, token_gene_idx_full, token_gene_mask_full = self._build_token_batch(
            x_expr,
            token_gene_idx=token_gene_idx,
            token_gene_mask=token_gene_mask,
            return_token_metadata=True,
        )
        hidden, attn_weights = self._run_transformer(
            tokens=tokens,
            key_padding_mask=key_padding_mask,
            return_attention=True,
        )
        h_cell = self.final_norm(hidden[:, 0, :])

        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([h_cell, c], dim=-1)
        else:
            h_in = h_cell

        if self.perturb_projector is not None:
            if perturb_vec is None:
                raise ValueError("perturb_vec is required when perturbation_dim is configured.")
            p = self.perturb_projector(perturb_vec.to(dtype=h_in.dtype, device=h_in.device))
            h_in = torch.cat([h_in, p], dim=-1)

        mu = self.mlp_mu(h_in)
        logvar = self.mlp_logvar(h_in)
        extras = {
            "token_gene_idx": token_gene_idx_full,
            "token_gene_mask": token_gene_mask_full,
            "key_padding_mask": key_padding_mask,
        }
        if attn_weights is not None:
            extras["attn_weights"] = attn_weights
        return mu, logvar, extras

# ---------------------------------------------------------------------------
# ExpressionDecoder: z (+ cond) -> reconstructed expression
# ---------------------------------------------------------------------------

class ExpressionDecoder(nn.Module):
    """
    Simple MLP decoder for expression.

    Given latent z (and optional condition embedding), outputs a reconstruction
    of x_expr. Here we use a Gaussian/MSE-style decoder, so we just output
    a single (B, G) matrix of reconstructed expression.

    If you'd like NB/ZINB later, you can change this to output mu/theta(/pi).
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
        perturbation_dim: Optional[int] = None,
        perturb_emb_dim: int = 32,
        perturb_condition_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        self.perturb_condition_decoder = bool(perturb_condition_decoder)
        if perturbation_dim is not None and self.perturb_condition_decoder:
            self.perturb_projector = PerturbationProjector(
                input_dim=int(perturbation_dim),
                output_dim=int(perturb_emb_dim),
            )
            perturb_input_dim = int(perturb_emb_dim)
        else:
            self.perturb_projector = None
            perturb_input_dim = 0

        decoder_input_dim = latent_dim + cond_input_dim + perturb_input_dim

        self.mlp = MLP(
            input_dim=decoder_input_dim,
            output_dim=n_genes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(
        self,
        z: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        libsize: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)
        cond_idx : Optional[(B,) LongTensor]

        Returns
        -------
        recon_x : (B, n_genes)
            Reconstructed expression (same transform as x_expr).
        """
        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([z, c], dim=-1)
        else:
            h_in = z

        if self.perturb_projector is not None:
            if perturb_vec is None:
                raise ValueError("perturb_vec is required when perturbation_dim is configured.")
            p = self.perturb_projector(perturb_vec.to(dtype=h_in.dtype, device=h_in.device))
            h_in = torch.cat([h_in, p], dim=-1)

        recon_x = self.mlp(h_in)
        return recon_x


# ---------------------------------------------------------------------------
# GeneVAE wrapper
# ---------------------------------------------------------------------------

class GeneVAE(nn.Module):
    """
    VAE that uses CBOWCellEncoder + ExpressionDecoder.

    Forward:
        - encode x_expr (+ cond) -> (mu, logvar)
        - reparameterize -> z
        - decode z (+ cond) -> recon_x
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        batch_adversary: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_adversary = batch_adversary

    def encode(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
        **encoder_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x_expr, cond_idx, perturb_vec=perturb_vec, **encoder_kwargs)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        libsize: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(z, cond_idx, libsize=libsize, perturb_vec=perturb_vec)

    def forward(
        self,
        x_expr: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        libsize: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
        **encoder_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        recon_x : (B, n_genes)
        mu : (B, latent_dim)
        logvar : (B, latent_dim)
        """
        mu, logvar = self.encode(
            x_expr,
            cond_idx,
            perturb_vec=perturb_vec,
            **encoder_kwargs,
        )
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, cond_idx, libsize=libsize, perturb_vec=perturb_vec)
        return recon_x, mu, logvar

    def predict_batch_logits(
        self,
        z: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> torch.Tensor:
        if self.batch_adversary is None:
            raise RuntimeError("batch_adversary is not configured on this model.")
        z_rev = gradient_reverse(z, lambda_=grl_lambda)
        return self.batch_adversary(z_rev)

# ---------------------------------------------------------------------------
# ZINB decoder
# ---------------------------------------------------------------------------

class ZINBExpressionDecoder(nn.Module):
    """
    ZINB decoder: given z (+ optional cond), outputs
      - mu    : mean counts per gene, shape (B, G), > 0
      - theta : inverse dispersion per gene, shape (B, G) or (1, G), > 0
      - pi    : dropout probability per gene, shape (B, G), in (0, 1)

    For simplicity, we:
      - predict mu and pi via MLP,
      - keep theta as a gene-wise parameter (broadcast across batch).
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_conditions: Optional[int] = None,
        cond_emb_dim: int = 16,
        perturbation_dim: Optional[int] = None,
        perturb_emb_dim: int = 32,
        perturb_condition_decoder: bool = True,
        use_library_size_covariate: bool = False,
        library_size_covariate_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.use_library_size_covariate = bool(use_library_size_covariate)
        self.library_size_covariate_eps = float(library_size_covariate_eps)
        if self.library_size_covariate_eps <= 0:
            raise ValueError("library_size_covariate_eps must be > 0.")

        # Optional condition embedding
        if n_conditions is not None:
            self.cond_embedding = nn.Embedding(n_conditions, cond_emb_dim)
            cond_input_dim = cond_emb_dim
        else:
            self.cond_embedding = None
            cond_input_dim = 0

        self.perturb_condition_decoder = bool(perturb_condition_decoder)
        if perturbation_dim is not None and self.perturb_condition_decoder:
            self.perturb_projector = PerturbationProjector(
                input_dim=int(perturbation_dim),
                output_dim=int(perturb_emb_dim),
            )
            perturb_input_dim = int(perturb_emb_dim)
        else:
            self.perturb_projector = None
            perturb_input_dim = 0

        decoder_input_dim = (
            latent_dim
            + cond_input_dim
            + perturb_input_dim
            + (1 if self.use_library_size_covariate else 0)
        )

        # MLP for mu (pre-activation)
        self.mlp_mu = MLP(
            input_dim=decoder_input_dim,
            output_dim=n_genes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

        # MLP for pi (dropout logit)
        self.mlp_pi = MLP(
            input_dim=decoder_input_dim,
            output_dim=n_genes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
        )

        # Gene-wise inverse dispersion parameter (learned)
        # We'll apply softplus to ensure positivity.
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        z: torch.Tensor,
        cond_idx: Optional[torch.Tensor] = None,
        libsize: Optional[torch.Tensor] = None,
        perturb_vec: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : (B, latent_dim)
        cond_idx : Optional[(B,) LongTensor]

        Returns
        -------
        mu : (B, G)  > 0
        theta : (B, G)  > 0
        pi : (B, G)  in (0, 1)
        """
        if self.cond_embedding is not None and cond_idx is not None:
            c = self.cond_embedding(cond_idx)
            h_in = torch.cat([z, c], dim=-1)
        else:
            h_in = z

        if self.perturb_projector is not None:
            if perturb_vec is None:
                raise ValueError("perturb_vec is required when perturbation_dim is configured.")
            p = self.perturb_projector(perturb_vec.to(dtype=h_in.dtype, device=h_in.device))
            h_in = torch.cat([h_in, p], dim=-1)

        if self.use_library_size_covariate:
            if libsize is None:
                raise ValueError("libsize must be provided when use_library_size_covariate=True.")
            if libsize.ndim == 1:
                libsize = libsize.unsqueeze(-1)
            elif libsize.ndim != 2 or libsize.shape[1] != 1:
                raise ValueError("libsize must have shape (B,) or (B, 1).")
            libsize_feat = torch.log1p(libsize.to(dtype=z.dtype, device=z.device).clamp_min(0.0))
            h_in = torch.cat([h_in, libsize_feat], dim=-1)

        mu_logit = self.mlp_mu(h_in)   # (B, G)
        pi_logit = self.mlp_pi(h_in)   # (B, G)

        # Ensure positivity
        mu = F.softplus(mu_logit) + 1e-8          # mean counts
        theta = F.softplus(self.log_theta) + 1e-8  # (G,)
        theta = theta.unsqueeze(0).expand_as(mu)   # (B, G)

        # Dropout probability
        pi = torch.sigmoid(pi_logit)              # (B, G)

        return mu, theta, pi

# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def kl_divergence_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) for diagonal Gaussian, p(z)=N(0, I).

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    # (B, D)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # (B,)
    kl = kl.sum(dim=-1)

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def gaussian_reconstruction_loss(
    recon_x: torch.Tensor,
    x_true: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Simple MSE loss between reconstructed and true expression.

    recon_x and x_true should be on the same scale (e.g. log1p normalized).
    """
    mse = F.mse_loss(recon_x, x_true, reduction="none")  # (B, G)
    mse = mse.sum(dim=-1)  # (B,)
    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    elif reduction == "none":
        return mse
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def zinb_negative_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    ZINB negative log-likelihood per cell:

    x    : observed counts,   (B, G)
    mu   : mean parameter,    (B, G)
    theta: inv. dispersion,   (B, G)
    pi   : dropout prob,      (B, G)
    """
    # x, mu, theta, pi must be non-negative / bounded appropriately
    # Ensure numerical stability
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)
    pi = pi.clamp(min=eps, max=1 - eps)
    x = x.clamp(min=0.0)

    # log NB pmf
    # lgamma(theta + x) - lgamma(theta) - lgamma(x + 1)
    t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1.0)

    log_theta = torch.log(theta + eps)
    log_mu = torch.log(mu + eps)
    log_theta_mu = torch.log(theta + mu + eps)

    t2 = theta * (log_theta - log_theta_mu)
    t3 = x * (log_mu - log_theta_mu)
    log_nb = t1 + t2 + t3            # log NB(x | mu, theta)

    # NB probability of zero
    # when x = 0, NB(0) = (theta / (theta + mu))^theta
    log_nb_zero = theta * (log_theta - log_theta_mu)

    # Mix with zero-inflation
    # For x == 0:
    #   log p(x=0) = log( pi + (1 - pi) * exp(log_nb_zero) )
    # For x > 0:
    #   log p(x)   = log(1 - pi) + log_nb
    is_zero = (x < eps)

    log_prob_zero = torch.log(
        pi + (1.0 - pi) * torch.exp(log_nb_zero) + eps
    )

    log_prob_nonzero = torch.log(1.0 - pi + eps) + log_nb

    log_prob = torch.where(is_zero, log_prob_zero, log_prob_nonzero)

    nll = -log_prob.sum(dim=-1)  # sum over genes -> (B,)

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction == "none":
        return nll
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def expression_contrastive_metric_loss(
    x_expr: torch.Tensor,
    z_latent: torch.Tensor,
    expr_transform: str = "log1p",
    temperature: float = 0.1,
    k_pos: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Batch-wise contrastive metric loss (InfoNCE-style).

    For each anchor, positives are selected as top-k nearest neighbors in
    expression-profile space. All non-self cells in the batch act as
    denominator candidates in latent space.
    """
    bsz = int(x_expr.shape[0])
    if bsz < 3:
        return z_latent.new_zeros(())
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    if expr_transform == "log1p":
        x_feat = torch.log1p(x_expr.clamp_min(0.0))
    elif expr_transform == "none":
        x_feat = x_expr
    else:
        raise ValueError(f"Unsupported expr_transform: {expr_transform}")

    x_norm = F.normalize(x_feat, p=2, dim=1, eps=eps)
    expr_sim = x_norm @ x_norm.t()  # (B, B)
    eye_mask = torch.eye(bsz, device=expr_sim.device, dtype=torch.bool)

    # Build a binary positive mask from expression nearest neighbors.
    expr_sim_pos = expr_sim.masked_fill(eye_mask, float("-inf"))
    k_pos_eff = max(1, min(int(k_pos), bsz - 1))
    pos_idx = torch.topk(expr_sim_pos, k=k_pos_eff, dim=1, largest=True).indices
    pos_mask = torch.zeros((bsz, bsz), device=expr_sim.device, dtype=torch.bool)
    pos_mask.scatter_(1, pos_idx, True)
    pos_mask = pos_mask | pos_mask.t()
    pos_mask = pos_mask & (~eye_mask)

    # Contrastive logits in latent space.
    z_norm = F.normalize(z_latent, p=2, dim=1, eps=eps)
    logits = (z_norm @ z_norm.t()) / float(temperature)  # (B, B)
    logits = logits.masked_fill(eye_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not bool(valid.any()):
        return z_latent.new_zeros(())

    pos_log_prob_sum = (log_prob.masked_fill(~pos_mask, 0.0)).sum(dim=1)
    loss_i = -pos_log_prob_sum[valid] / pos_counts[valid].to(log_prob.dtype)
    return loss_i.mean()
