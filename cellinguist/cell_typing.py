import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.sparse import csr_matrix
import numpy as np
import anndata as ad
import torch.distributed as dist
from cellinguist.models.loss import mse_loss_for_expression, compute_similarity_loss
from cellinguist.data.data_funcs import SingleCellDatasetUnified, collate_fn_unified, SingleCellDatasetCellTyping, collate_fn_celltypes
from cellinguist.models.base_model import TokenEmbeddingLayer, FlashTransformerEncoderLayer, MaskedGeneExpressionPredictionHead, WholeGenomeExpressionPredictionHead, DomainClassifier, FullModel, get_random_mask_positions, train_epoch_ddp, MaskedGeneIDPredictionHead
from cellinguist.models.cell_type_model import CellinguistForCellType
import scipy.sparse

def main():
     ## Parse input arguments
    parser = argparse.ArgumentParser(description="Function to extract gene embedding from a trained model.")
    parser.add_argument("--input_anndata", type=str, required=True, help="Path to anndata file.")
    parser.add_argument("--in_model", type=str, required=True, help="Path to trained model, should end in pth extension.")
    parser.add_argument("--domain_for_grad_rev", type=str, help="Optional column name of metadata containing the domain info (such as sequencing batch) for gradient reversal.")
    parser.add_argument("--condition_data", type=str, help="Optional column name of metadata containing condition info.")
    parser.add_argument("--reserved_cls_token", type=int, default=0, help="Reserved token for CLS.")
    parser.add_argument("--reserved_pad_token", type=int, default=1, help="Reserved token for padding.")
    parser.add_argument("--reserved_mask_token", type=int, default=2, help="Reserved token for masking.")
    parser.add_argument("--num_expression_bins", type=int, default=128, help="Number of bins for expression data.")
    parser.add_argument("--num_library_bins", type=int, default=20, help="Number of bins for library size normalization.")

    args = parser.parse_args()

    ## Load anndata
    dat = ad.read_h5ad(args.input_anndata)
    if scipy.sparse.issparse(dat.X):
        dense_matrix = dat.X.toarray()  # type: ignore
    else:
        dense_matrix = np.asarray(dat.X)

    ## Set gene ids and vocab size
    gene_ids = dat.var.index.to_numpy()
    num_of_genes = len(gene_ids)

    ## Set domains for normalization (optional)
    if args.domain_for_grad_rev is not None:
        seq_batch_ids = torch.tensor(dat.obs[args.domain_for_grad_rev].factorize()[0])
    else: 
        seq_batch_ids = None

    ## Set conditions and cell types (combine logic)
    if args.condition_data is not None:
        condition_labels = torch.tensor(dat.obs[args.condition_data].factorize()[0])
        cell_type_labels = dat.obs[args.condition_data].factorize()[0]
    else:
        condition_labels = None
        raise ValueError("You must provide --condition_data for cell type labels.")

    ## Set other input parameters
    CLS_TOKEN_ID = args.reserved_cls_token
    PAD_TOKEN_ID   = args.reserved_pad_token
    MASK_TOKEN_ID  = args.reserved_mask_token
    reserved_tokens_count = 3

    num_expression_bins = args.num_expression_bins
    expression_vocab_size = num_expression_bins + reserved_tokens_count
    gene_vocab_size = num_of_genes + reserved_tokens_count

    if condition_labels is not None:
        CONDITION_VOCAB_SIZE = len(np.unique(condition_labels))
    else:
        CONDITION_VOCAB_SIZE = 1
        
    NUM_LIBRARY_BINS = args.num_library_bins

    if seq_batch_ids is not None:
        num_domains = len(np.unique(seq_batch_ids))
    else:
        num_domains = 1
        
    NUM_LIBRARY_BINS = args.num_library_bins

    # Initialize a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Convert torch.Tensors to numpy arrays for dataset
    condition_labels_np = condition_labels.cpu().numpy() if condition_labels is not None else np.zeros(dense_matrix.shape[0], dtype=int)
    seq_batch_ids_np = seq_batch_ids.cpu().numpy() if seq_batch_ids is not None else np.zeros(dense_matrix.shape[0], dtype=int)
    cell_type_labels_np = np.asarray(cell_type_labels, dtype=np.intp).reshape(-1)

    ds = SingleCellDatasetCellTyping(
        dense_matrix,
        NUM_LIBRARY_BINS,
        num_expression_bins,
        reserved_tokens_count,
        condition_labels_np,
        seq_batch_ids_np,
        np.asarray(cell_type_labels_np)
    )
    collate = partial(collate_fn_celltypes, cls_token_id=CLS_TOKEN_ID, pad_token_id=PAD_TOKEN_ID)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)

    # 2) Build the backbone (FullModel)
    token_embedding_layer = TokenEmbeddingLayer(
        gene_vocab_size=gene_vocab_size,
        expression_vocab_size=expression_vocab_size,
        condition_vocab_size=CONDITION_VOCAB_SIZE,
        library_vocab_size=NUM_LIBRARY_BINS,
        embedding_dim=512,  # match your training config
        max_seq_len=1200,   # match your training config
        use_positional=False,
        pad_token_id=PAD_TOKEN_ID
    ).to(device)

    flash_encoder_layers = nn.ModuleList([
        FlashTransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, causal=False)
        for _ in range(4)
    ]).to(device)

    masked_head = MaskedGeneExpressionPredictionHead(
        d_model=512,
        expression_vocab_size=expression_vocab_size
    ).to(device)

    whole_genome_head = WholeGenomeExpressionPredictionHead(
        d_model=512,
        total_gene_count=num_of_genes,
        expression_vocab_size=expression_vocab_size
    ).to(device)

    domain_classifier = DomainClassifier(input_dim=512, hidden_dim=256, num_domains=num_domains).to(device)

    masked_gene_head = MaskedGeneIDPredictionHead(
        d_model=512,
        gene_vocab_size=gene_vocab_size
    ).to(device)

    backbone = FullModel(
        token_embedding_layer, flash_encoder_layers, masked_head, whole_genome_head,
        domain_classifier, masked_gene_head, lambda_value=1.0  # or your training value
    ).to(device)

    # Load weights
    checkpoint = torch.load(args.in_model, map_location=device)
    backbone.load_state_dict(checkpoint['model_state_dict'])

    # Set up classifier
    cell_embedding_dim = 512  # or your CLS embedding size
    NUM_TYPES = len(np.unique(cell_type_labels_np))
    model = CellinguistForCellType(backbone, cell_embedding_dim, NUM_TYPES, freeze_backbone=True).to(device)

    # 3) optimizer over only classifier params (and un-frozen backbone if you like)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 4) train
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        n = 0
        for batch in dl:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch)
            labels = batch["cell_types"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)

        acc = correct / n
        print(f"Epoch {epoch+1:2d}  loss={total_loss/n:.3f}  acc={acc:.3%}")

        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                logits = model(batch)
                all_preds.append(logits.argmax(dim=-1).cpu())
        all_preds = torch.cat(all_preds, 0).numpy()

if __name__ == "__main__":
    main()