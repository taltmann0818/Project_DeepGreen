# train.py
import pytorch_lightning as pl
from resmlp_lightning import ResMLP
from datamodule_cs      import CrossSectionDM

dm = CrossSectionDM(root="features_cs", batch_size=1)    # 1 macro-batch/step
model = ResMLP(n_features=len(dm.feats))

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",       # lightning auto-casts to bf16 if supported
    deterministic=False,
    accumulate_grad_batches=4,    # ≈ 4×512 = 2 K tickers / step
    callbacks=[pl.callbacks.ModelCheckpoint(
                 monitor="val_loss", mode="min", filename="resmlp-best")]
)
trainer.fit(model, dm)
