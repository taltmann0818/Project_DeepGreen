# datamodule_cs.py
import pandas as pd, pyarrow.parquet as pq, glob, random
from torch.utils.data import Dataset, DataLoader
import torch, pytorch_lightning as pl
from pathlib import Path

class CrossSectionDS(Dataset):
    def __init__(self, files, feature_cols, target_col):
        self.files = files
        self.feats = feature_cols
        self.tgt   = target_col
    def __len__(self):   return len(self.files)
    def __getitem__(self, idx):
        df = pq.read_table(self.files[idx]).to_pandas()
        X  = df[self.feats].to_numpy(dtype="float32")
        y  = df[self.tgt ].to_numpy(dtype="float32")
        # random subsample to fixed batch of tickers
        subs = torch.randperm(len(X))[:512]
        return torch.from_numpy(X[subs]), torch.from_numpy(y[subs])

class CrossSectionDM(pl.LightningDataModule):
    def __init__(self, root="features_cs", split=0.9,
                 batch_size=1, num_workers=4):
        super().__init__()
        files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
        cut   = int(len(files)*split)
        self.train_files, self.val_files = files[:cut], files[cut:]
        # discover columns once
        sample = pq.read_table(files[0]).to_pandas()
        self.target = "fwd_3d_return"
        self.feats  = [c for c in sample.columns if c not in
                       ("date", "ticker", self.target)]

        self.bs, self.nw = batch_size, num_workers

    def train_dataloader(self):
        ds = CrossSectionDS(self.train_files, self.feats, self.target)
        return DataLoader(ds, batch_size=self.bs, num_workers=self.nw)

    def val_dataloader(self):
        ds = CrossSectionDS(self.val_files, self.feats, self.target)
        return DataLoader(ds, batch_size=self.bs, num_workers=self.nw)
