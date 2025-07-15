# resmlp_lightning.py
import torch, torch.nn as nn, torch.nn.functional as F
from timm.layers import DropPath
import torch.optim.lr_scheduler as sched
#from kan import KANLinear
import pytorch_lightning as pl
import torchmetrics

class ResBlock(pl.LightningModule):
    def __init__(self, d, p=0.1, drop_path=0.05, layer_scale=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.mlp  = nn.Sequential(
            #KANLinear(d, d*4, edge_activation='gelu'),
            nn.Linear(d, d*4, bias=False),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(d*4, d, bias=False),
        )
        self.gamma = nn.Parameter(layer_scale * torch.ones(d))
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return x + self.dp(self.gamma * self.mlp(self.norm(x)))

class XCovTokenMix(pl.LightningModule):
    def __init__(self, d, n_heads=4, drop_path=0.05, layer_scale=0.1):
        super().__init__()
        self.attn = nn.Conv1d(d, d, 1, groups=n_heads)
        self.dp   = DropPath(drop_path)
        self.gamma= nn.Parameter(layer_scale * torch.ones(d))

        # Initialize weights properly
        nn.init.xavier_uniform_(self.attn.weight)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch_size, 1, features)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, features)
        elif x.dim() > 3:
            original_shape = x.shape
            x = x.view(-1, original_shape[-2], original_shape[-1])
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor after reshaping, got {x.dim()}D tensor with shape {x.shape}")
        try:
            h = self.attn(x.transpose(1, 2)).transpose(1, 2)
        except RuntimeError as e:
            print(f"Warning: Transpose failed, applying attention directly. Error: {e}")
            if x.size(1) == 1:
                h = self.attn(x.squeeze(1).unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            else:
                h = x
        return x + self.dp(self.gamma * h)

class ResMLPLightning(pl.LightningModule):
    def __init__(self, n_features: int, d: int = 256, n_blocks: int = 8,
                 lr: float = 1e-3, wd: float = 1e-3, dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.proj   = nn.Linear(n_features, d)
        self.blocks = nn.Sequential(
            *[ResBlock(d, p=dropout) if i%2==0
              else XCovTokenMix(d)
              for i in range(n_blocks)]
        )
        self.head   = nn.Linear(d, 1)

        # Initialize weights properly
        self._init_weights()

        # metrics
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.NormalizedRootMeanSquaredError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.smape = torchmetrics.SymmetricMeanAbsolutePercentageError()

    def _init_weights(self):
        # Initialize projection layer
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Initialize head layer with smaller weights for stability
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)

    def loss_fn(self, y_hat, y):
        # Add stability checks for loss computation
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            print("Warning: NaN or Inf in predictions during loss computation")
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: NaN or Inf in targets during loss computation")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        # Use Huber loss for better stability with outliers
        loss = F.huber_loss(y_hat, y, delta=1.0)

        # Clip loss to prevent explosion
        loss = torch.clamp(loss, max=10.0)

        return loss

    def training_step(self, batch, _):
        x, y = batch

        # Add input stability checks
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf in training inputs")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)

        # Check for loss explosion
        if loss > 100.0:
            print(f"Warning: Very high training loss: {loss.item():.6f}")

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)

        # Add stability checks for validation metrics
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            print("Warning: NaN or Inf in validation predictions")
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: NaN or Inf in validation targets")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        if y_hat.dim() > 1 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)

        # Ensure y is also properly shaped
        if y.dim() > 1 and y.size(-1) == 1:
            y = y.squeeze(-1)

        val_loss = self.loss_fn(y_hat, y)
        val_mae = self.mae(y_hat, y)
        val_rmse = self.rmse(y_hat, y)
        val_mape = self.mape(y_hat, y)
        val_smape = self.smape(y_hat, y)

        self.log_dict(
            {
            "val_loss": val_loss,
             "val_mae": val_mae,
             "val_rmse": val_rmse,
             "val_mape": val_mape,
             "val_smape": val_smape
            },
            prog_bar=False)

    def configure_optimizers(self):
        warmup_steps = 1000  # Increased warmup for better stability
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            eps=1e-8,  # Increased epsilon for numerical stability
            betas=(0.9, 0.95)  # More conservative beta2
        )
        total_steps = self.trainer.estimated_stepping_batches

        # More conservative learning rate schedule
        sch = sched.SequentialLR(
            opt,
            schedulers=[
                # More gradual warmup
                sched.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps),
                # Less aggressive cosine annealing
                sched.CosineAnnealingLR(
                    opt,
                    T_max=max(total_steps - warmup_steps, 1),
                    eta_min=self.hparams.lr * 0.1,  # Higher minimum LR
                ),
            ],
            milestones=[warmup_steps],  # switch point
        )
        return [opt], [{"scheduler": sch, "interval": "step"}]
