import pandas as pd
from glob import glob

# 1) read nightly alpha dumps  (produced by your existing inference scripts)
dfs = []
for f in glob("alphas_tft/*.parquet"):
    tft = pd.read_parquet(f)              # date, ticker, alpha_tft
    res = pd.read_parquet(f.replace("tft", "resmlp"))
    m   = tft.merge(res, on=["date", "ticker"])
    m["alpha_prod"]     = m["alpha_tft"] * m["alpha_resmlp"]
    m["abs_tft"]        = m["alpha_tft"].abs()
    m["abs_resmlp"]     = m["alpha_resmlp"].abs()
    m["fwd_3d_return"]  = load_label_for_date(m["date"].iloc[0])
    dfs.append(m)

data = pd.concat(dfs, ignore_index=True)

cut_date = "2025-04-01"                # oldest date reserved for test OOS
train = data[data["date"] <  cut_date]
test  = data[data["date"] >= cut_date]

X_train = train[["alpha_tft", "alpha_resmlp",
                 "abs_tft", "abs_resmlp", "alpha_prod"]]
y_train = train["fwd_3d_return"]

X_test  = test[X_train.columns]
y_test  = test["fwd_3d_return"]


import xgboost as xgb
params = dict(
    max_depth=2,              # shallow!
    n_estimators=80,          # tiny
    learning_rate=0.08,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42
)
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=10, verbose=False)

model.save_model("stacker_booster.json")

import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.tight_layout(); plt.show()

def blend_alphas(df_day: pd.DataFrame, booster_path="stacker_booster.json"):
    # df_day has columns: alpha_tft, alpha_resmlp, date, ticker
    booster = xgb.Booster()
    booster.load_model(booster_path)

    feats = df_day.assign(
        abs_tft=lambda d: d["alpha_tft"].abs(),
        abs_resmlp=lambda d: d["alpha_resmlp"].abs(),
        alpha_prod=lambda d: d["alpha_tft"] * d["alpha_resmlp"]
    )[["alpha_tft", "alpha_resmlp", "abs_tft",
       "abs_resmlp", "alpha_prod"]]

    dmat = xgb.DMatrix(feats)
    df_day["alpha_blend"] = booster.predict(dmat)
    return df_day


