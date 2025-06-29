# resmlp_lightning.py
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class ResBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d * 4, d)
        )
    def forward(self, x):
        return x + self.net(x)

class ResMLP(pl.LightningModule):
    def __init__(self, n_features: int, d: int = 256, n_blocks: int = 6,
                 lr: float = 1e-3, wd: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.proj   = nn.Linear(n_features, d)
        self.blocks = nn.Sequential(*[ResBlock(d) for _ in range(n_blocks)])
        self.head   = nn.Linear(d, 1)
        # metrics
        self.r2 = torchmetrics.R2Score()
        self.mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)          # (batch,)

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
             "val_r2":  self.r2(y_hat, y),
             "val_mae": self.mae(y_hat, y)},
            prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return [opt], [sch]
