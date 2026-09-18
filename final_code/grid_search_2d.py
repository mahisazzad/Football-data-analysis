import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import spatial_lib
from spatial_lib import carry_path_crosses_hulls

df = pd.read_pickle("base_carries_s1.pkl")
print(f"Base carries: {len(df)}")

feat_cols = ["carry_distance_m", "carry_speed_ms", "defender_proximity_m", "under_pressure"]
idx = np.arange(len(df))
idx_train, idx_temp = train_test_split(idx, test_size=0.4, random_state=42)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

y_weights = [0.15, 0.25, 0.35]
eps_grid = [4, 6, 8, 10, 12]

results = []
X = df[feat_cols].values

for yw in y_weights:
    spatial_lib.Y_WEIGHT = yw
    for eps in eps_grid:
        targets, n_lines_list = [], []
        for _, row in df.iterrows():
            crossed, n_lines = carry_path_crosses_hulls(row["start"], row["end"], row["opp_pts"], eps)
            targets.append(int(crossed >= 1))
            n_lines_list.append(n_lines)
        y = np.array(targets)
        rate = y.mean()
        avg_lines = np.mean(n_lines_list)

        Xtr, ytr = X[idx_train], y[idx_train]
        Xval, yval = X[idx_val], y[idx_val]
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
        gb.fit(Xtr, ytr)
        val_auc = roc_auc_score(yval, gb.predict_proba(Xval)[:, 1])

        results.append({"y_weight": yw, "eps": eps, "avg_n_lines": round(avg_lines,2),
                         "line_break_rate": round(100*rate,2), "val_auc": round(val_auc,4)})
        print(f"y_weight={yw}  eps={eps}  avg_lines={avg_lines:.2f}  rate={100*rate:.2f}%  val_AUC={val_auc:.4f}", flush=True)

best = max(results, key=lambda r: r["val_auc"])
print(f"\nBest combo by validation AUC: {best}")
json.dump({"grid": results, "best": best}, open("spatial_grid_search.json", "w"), indent=2)
