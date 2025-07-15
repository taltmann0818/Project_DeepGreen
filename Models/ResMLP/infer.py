# infer.py
import torch, pandas as pd, pyarrow.parquet as pq, glob
from resmlp_lightning import ResMLP

ckpt   = "lightning_logs/.../checkpoints/resmlp-best.ckpt"
model  = ResMLP.load_from_checkpoint(ckpt).eval().to("cuda")

files  = glob.glob("features_cs/year=2025/month=06/day=26/*.parquet")
out    = []
with torch.no_grad():
    for f in files:
        df = pq.read_table(f).to_pandas()
        x  = torch.tensor(df[model.hparams.n_features].values,
                          dtype=torch.float32, device="cuda")
        α  = model(x).cpu().numpy()
        out.append(df[["date", "ticker"]].assign(alpha_resmlp=α))
pd.concat(out).to_parquet("alphas_resmlp_2025-06-26.parquet")
