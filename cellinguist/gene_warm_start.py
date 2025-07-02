import numpy as np
import torch
import anndata as ad
from cellinguist.data.data_funcs import BinarizedExpressionDataset, accumulate_cooccurrence, truncated_svd_gpu
from torch.utils.data import Dataset, DataLoader
import pandas as pd

## Load anndata
dat = ad.read_h5ad("/home/arc85/Desktop/scrnaseq_transformer/01_input/HD_PBMC_5prime_3prime_250603b.h5ad")
dense_matrix = dat.X.toarray()

# Suppose `expr` is your huge (N_cells, G) NumPy array of raw counts.
dataset = BinarizedExpressionDataset(dense_matrix, threshold=1)
# We’ll shuffle (optional) and batch it. Pin memory and use multiple workers to speed I/O.
dataloader = DataLoader(
    dataset,
    batch_size=1024,       # tune this so that (batch_size × G) fits in GPU memory
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

## Step 1: accumulate co-occurence matrix on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_genes = dense_matrix.shape[1]  # G
C_gpu = accumulate_cooccurrence(dataloader, num_genes, device)
print("Finished accumulating co-occurrence. Frobenius norm:", C_gpu.norm().item())

## Step 2: log transform to normalize large values 
M_gpu = torch.log1p(C_gpu)
del C_gpu

## Step 3: perform SVD on GPU
emb_dim = 512
U_gpu = truncated_svd_gpu(M_gpu, emb_dim=emb_dim, n_iter=10)
print("Truncated SVD result shape:", U_gpu.shape)  # should be (G, emb_dim)

## Move to CPU
U_cpu = U_gpu.cpu().numpy()

RESERVED = 3
gene_embed_layer = torch.nn.Embedding(num_genes + RESERVED, emb_dim, padding_idx=1)
with torch.no_grad():
    gene_embed_layer.weight[:RESERVED].zero_()
    gene_embed_layer.weight[RESERVED:RESERVED+num_genes, :] = torch.from_numpy(U_cpu)

## Save output 
# Suppose `gene_emb_layer` is your nn.Embedding for gene‐IDs.
torch.save(gene_embed_layer.weight.data.cpu(), "gene_embeddings_250603.pth")

## Later in a new session:
# loaded_tensor = torch.load("gene_embeddings.pth",weights_only=True)  # yields a torch.Tensor of shape (total_tokens, emb_dim)

# test_save = pd.DataFrame(loaded_tensor.numpy())

# test_save.to_csv("/home/arc85/Desktop/Cellinguist/test_gene_embedding.csv")

# Create or grab your TokenEmbeddingLayer and overwrite its weights:
#token_emb = TokenEmbeddingLayer(
#    gene_vocab_size=total_tokens,
#    expression_vocab_size=…,
#    condition_vocab_size=…,
#    pad_token_id=…,
#    library_vocab_size=…,
#    embedding_dim=emb_dim,
#    max_seq_len=…,
#    use_positional=…
#)
#with torch.no_grad():
#    token_emb.gene_embeddings.weight.copy_(loaded_tensor.to(token_emb.gene_embeddings.weight.device))