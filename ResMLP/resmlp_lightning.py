# resmlp_lightning.py
# resmlp_lightning.py
import torch, torch.nn as nn, torch.nn.functional as F
from timm.models.layers import DropPath
from kan import KANLinear
import pytorch_lightning as pl
import torchmetrics

class ResBlock(nn.Module):
    def __init__(self, d, p=0.1, drop_path=0.05, layer_scale=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.mlp  = nn.Sequential(
            KANLinear(d, d*4, edge_activation='gelu'),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(d*4, d, bias=False),
        )
        self.gamma = nn.Parameter(layer_scale * torch.ones(d))
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()  
    def forward(self, x):
        return x + self.dp(self.gamma * self.mlp(self.norm(x)))

class XCovTokenMix(nn.Module):
    def __init__(self, d, n_heads=4, drop_path=0.05):
        super().__init__()
        self.attn = nn.Conv1d(d, d, 1, groups=n_heads)
        self.dp   = DropPath(drop_path)
        self.gamma= nn.Parameter(1e-5 * torch.ones(d))
    def forward(self, x):
        h = self.attn(x.transpose(1,2)).transpose(1,2)
        return x + self.dp(self.gamma * h)

class ResMLP(pl.LightningModule):
    def __init__(self, n_features: int, d: int = 256, n_blocks: int = 8,
                 lr: float = 1e-3, wd: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.proj   = nn.Linear(n_features, d)
        self.blocks = nn.Sequential(
            *[ResBlock(d, p=0.1) if i%2==0 
              else XCovTokenMix(d)
              for i in range(n_blocks)]
        )
        self.head   = nn.Linear(d, 1)
        # metrics
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.NormalizedRootMeanSquaredError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.smape = torchmetrics.SymmetricMeanAbsolutePercentageError()

    def forward(self, x):
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)

    def loss_fn(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        self.log_dict(
            {"val_loss": self.loss_fn(y_hat, y),
             "val_mae": self.mae(y_hat, y),
             "val_rmse": self.rmse(y_hat, y),
             "val_mape": self.mape(y_hat, y),
             "val_smape": self.smape(y_hat, y)
            },
            prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adafactor(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4_000, eta_min=1e-5)
        return [opt], [sch]