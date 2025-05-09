import sys
sys.path.append('/home/arc85/Desktop/Cellinguist/')

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
    parser = argparse.ArgumentParser(description="Cellinguist model for learning the language of cells.")
    parser.add_argument("--input_anndata", type=str, required=True, help="Path to anndata file.")
    parser.add_argument("--domain_for_grad_rev", type=str, help="Optional column name of metadata containing the domain info (such as sequencing batch) for gradient reversal.")
    parser.add_argument("--condition_data", type=str, help="Optional column name of metadata containing condition info.")
    parser.add_argument("--batch_size", type=int, default=32, help="Optional column name of metadata containing condition info.")
    parser.add_argument("--reserved_cls_token", type=int, default=0, help="Reserved token for CLS.")
    parser.add_argument("--reserved_pad_token", type=int, default=1, help="Reserved token for padding.")
    parser.add_argument("--reserved_mask_token", type=int, default=2, help="Reserved token for masking.")
    parser.add_argument("--num_expression_bins", type=int, default=128, help="Number of bins for expression data.")
    parser.add_argument("--num_library_bins", type=int, default=20, help="Number of bins for library size normalization.")
    parser.add_argument("--max_seq_length", type=int, default=1200, help="Maximum sequence length to consider for input.")
    parser.add_argument("--token_embedding_dim", type=int, default=512, help="Dimensions for gene embedding.")
    parser.add_argument("--flash_encoder_dim", type=int, default=512, help="Dimensions for flash encoder.")
    parser.add_argument("--flash_encoder_layers", type=int, default=4, help="Number of layers for flash encoder.")
    parser.add_argument("--flash_encoder_heads", type=int, default=8, help="Number of heads for flash encoder.")
    parser.add_argument("--prediction_domain_head_dim", type=int, default=512, help="Dimensions for prediction and domain classification heads.")
    parser.add_argument("--num_iterative_steps", type=int, default=3, help="Number of iterative steps within each batch of data.")
    parser.add_argument("--grad_rev_lambda", type=float, default=1.0, help="Lambda value for gradient reversal.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--out_model", type=str, required=True, help="Path to save output model, should end in pth extension.")
    args = parser.parse_args()

    ## Load anndata
    dat = ad.read_h5ad(args.input_anndata)
    dense_matrix = dat.X.toarray()

    ## Set gene ids and vocab size
    gene_ids = dat.var.gene.to_numpy()
    num_of_genes = len(gene_ids)

    ## Set domains for normalization (optional)
    seq_batch_ids = torch.tensor(dat.obs[args.domain_for_grad_rev].factorize()[0])

    ## Set conditions (optional)
    if args.condition_data is not None:
        condition_labels = torch.tensor(dat.obs[args.condition_data].factorize()[0])
    else:
        condition_labels = None
    print(condition_labels)

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

    # Initialize the process group.
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    ## Create dataset 
    dataset = SingleCellDatasetUnified(dense_matrix, condition_labels = condition_labels, domain_labels = seq_batch_ids, num_expression_bins = num_expression_bins, num_library_bins = NUM_LIBRARY_BINS, reserved_tokens_count = 3)
    sampler = DistributedSampler(dataset)
    collate = partial(collate_fn_unified, cls_token_id=CLS_TOKEN_ID, pad_token_id=PAD_TOKEN_ID)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler = sampler, collate_fn=collate)

    ## Token embedding 
    token_embedding_layer = TokenEmbeddingLayer(
        gene_vocab_size=gene_vocab_size,
        expression_vocab_size=expression_vocab_size,
        condition_vocab_size=CONDITION_VOCAB_SIZE,
        library_vocab_size=NUM_LIBRARY_BINS,
        embedding_dim=args.token_embedding_dim,
        max_seq_len=args.max_seq_length,
        use_positional=False,
        pad_token_id=PAD_TOKEN_ID
    ).to(device)

    ## Flash transformer
    flash_encoder_layers = nn.ModuleList([
        FlashTransformerEncoderLayer(d_model=args.flash_encoder_dim, nhead=args.flash_encoder_heads, dropout=0.1, causal=False)
        for _ in range(args.flash_encoder_layers)
    ])
    flash_encoder_layers = flash_encoder_layers.to(device)

    ## Prediction heads
    masked_head = MaskedGeneExpressionPredictionHead(
        d_model=args.prediction_domain_head_dim,
        expression_vocab_size=expression_vocab_size
    ).to(device)

    whole_genome_head = WholeGenomeExpressionPredictionHead(
        d_model=args.prediction_domain_head_dim,
        total_gene_count=num_of_genes,
        expression_vocab_size=expression_vocab_size
    ).to(device)
    
    ## Domain classifier
    domain_classifier = DomainClassifier(input_dim=args.prediction_domain_head_dim, hidden_dim=256, num_domains=num_domains).to(device)

    # Combine components into a full model.
    # Assume FullModel is a module that takes the token embedding layer, encoder layers, and prediction heads.
    full_model = FullModel(token_embedding_layer, flash_encoder_layers, masked_head, whole_genome_head, domain_classifier, lambda_value = args.grad_rev_lambda).to(device)
    
    # Wrap the model with DistributedDataParallel.
    ddp_model = DDP(full_model, device_ids=[local_rank])
    
    # Create your dataset and distributed sampler.
    # Assume dataset and collate_fn_unified are defined.
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)

    # Optionally, if you use teacher forcing, define the number of iterations.
    num_iterative_steps = args.num_iterative_steps

    # Main training loop.
    NUM_EPOCHS = args.epochs
    
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)  # shuffle dataset for distributed sampler
        avg_loss = train_epoch_ddp(
            dataloader, ddp_model, optimizer, device, num_iterative_steps=num_iterative_steps, mask_token_id=MASK_TOKEN_ID, pad_token_id=PAD_TOKEN_ID
        )
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    # Clean up.
    dist.destroy_process_group()

    # ---------------------------
    # Checkpoint the model components 
    # ---------------------------

    checkpoint = {
        'model_state_dict': ddp_model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch  # You might want to store the current epoch as well.
    }

    torch.save(checkpoint, args.out_model)
    print("Model saved!")

if __name__ == "__main__":
    main()