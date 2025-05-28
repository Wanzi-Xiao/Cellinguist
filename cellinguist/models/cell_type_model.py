import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CellinguistForCellType(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 cell_embedding_dim: int,
                 num_cell_types: int,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # small MLP on top of the CLS embedding
        self.cls_head = nn.Sequential(
            nn.Linear(cell_embedding_dim, cell_embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cell_embedding_dim // 2, num_cell_types)
        )

    def forward(self, batch: dict):
        """
        batch must be exactly the same dict you pass to FullModel:
          {
            "gene_ids":           LongTensor (B, L),
            "expression_tokens":  LongTensor (B, L),
            "condition":          LongTensor (B,),
            "library_size":       LongTensor (B,),
            "whole_genome_target":LongTensor (B, G),
            "domain":             LongTensor (B,)
          }
        """
        # Call the backbone exactly as it expects
        masked_logits, whole_genome_logits, cls_token, domain_preds = self.backbone(batch)

        # cls_token is your (B, 512) cell embedding
        logits = self.cls_head(cls_token)  # (B, num_cell_types)
        return logits

