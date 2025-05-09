import os
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from cellinguist.training import get_random_mask_positions
from cellinguist.models.loss import mse_loss_for_expression, compute_similarity_loss
from cellinguist.data.data_funcs import SingleCellDatasetUnified, collate_fn_unified
from cellinguist.models.base_model import FlashTransformerEncoderLayer, MaskedGeneExpressionPredictionHead, WholeGenomeExpressionPredictionHead, DomainClassifier, FullModel

@click.group()
def cli():
    """Cellinguist: speak the language of cells"""
    pass

@click.command() 
@click.option("--batch-size", "-b", required=True, type=click.Path(exists=True),help="Size of batch for training.")
@click.option("--seq_length", "-s", required=True, default=1200, type=int, help="Maxmimum input sequence length to consider.")
## infer GENE_VOCAB_SIZE and TOTAL_GENE_COUNT from input dimensions
@click.option("--expression_vocab_size", "-v", required=True, default=128, type=int, help="Number of expression bins to create from count data.")
@click.option("--embedding_dim", "-e", required=True, default=512, type=int, help="Dimension of output embedding space.")
@click.option("--num_interative_steps", "-i", required=True, default=3, type=int, help="Number of iterative steps of learning per batch.")
@click.option("--flash_embed_dim", "-f", required=True, default=512, type=int, help="Dimension of flash embedding layer.")
@click.option("--flash_nhead", "-n", required=True, default=8, type=int, help="Number of flash embedding heads.")
@click.option("--flash_transform_layers", "-l", required=True, default=4, type=int, help="Number of flash transformer layers.")
@click.option("--grad_rev_lambda", "-g", required=True, default=1.0, type=float, help="Learning parameter for gradient reversal.")

