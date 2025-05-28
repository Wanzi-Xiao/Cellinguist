# Cellinguist
Language of the cell

*This is a work in progress*

## Pretraining
Currently, we can pretrain on an anndata formatted datasets using the following incantation:

```
torchrun --nproc-per-node=2 training.py \
    --input_anndata=HD_PBMC_5prime_3prime_1k_hvf_250508.h5ad \
    --domain_for_grad_rev=chemistry \
    --out_model=/HD_PBMC_5prime_3prime_1k_hvf_model.pth
```
