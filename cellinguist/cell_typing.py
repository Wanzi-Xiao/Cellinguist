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
from cellinguist.data.data_funcs import SingleCellDatasetUnified, collate_fn_unified
from cellinguist.models.base_model import TokenEmbeddingLayer, FlashTransformerEncoderLayer, MaskedGeneExpressionPredictionHead, WholeGenomeExpressionPredictionHead, DomainClassifier, FullModel, get_random_mask_positions, train_epoch_ddp

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

    ## Load anndata
    dat = ad.read_h5ad(args.input_anndata)
    dense_matrix = dat.X if isinstance(dat.X, np.ndarray) else dat.X.toarray()

    ## Set gene ids and vocab size
    gene_ids = dat.var.index.to_numpy()
    num_of_genes = len(gene_ids)

    ## Set domains for normalization (optional)
    if args.domain_for_grad_rev is not None:
        seq_batch_ids = torch.tensor(dat.obs[args.domain_for_grad_rev].factorize()[0])
    else: 
        seq_batch_ids = None

    ## Set conditions (optional)
    if args.condition_data is not None:
        condition_labels = torch.tensor(dat.obs[args.condition_data].factorize()[0])
    else:
        condition_labels = None

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

    ## Cell types
    cell_type_labels = dat.obs[args.domain_for_grad_rev].factorize()[0]

    # Initialize a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) build dataset that returns 'cell_type' label in __getitem__
    ds = SingleCellDatasetUnified(expr_matrix, condition_labels, domain_labels, cell_type_labels=your_labels)
    collate = partial(collate_fn_unified, cls_token_id=CLS, pad_token_id=PAD)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)

    # 2) load pretrained backbone
    backbone = DeepSetModel(...).to(device)
    backbone.load_state_dict(torch.load(args.in_model, map_location=device)) 
    model = CellinguistForCellType(backbone, num_cell_types=NUM_TYPES, freeze_backbone=True).to(device)

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
            optimizer.zero_grad()
            logits = model(batch["gene_ids"].to(device),
                       batch["expression_tokens"].to(device))
            labels = batch["cell_type"].to(device)
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
                logits = model(batch["gene_ids"].to(device),
                               batch["expression_tokens"].to(device))
                all_preds.append(logits.argmax(dim=-1).cpu())
        all_preds = torch.cat(all_preds, 0).numpy()