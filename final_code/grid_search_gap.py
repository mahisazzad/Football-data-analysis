import json, glob, math
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

MID_ATT_POSITIONS = {
    "Left Center Midfield", "Right Center Midfield", "Center Defensive Midfield",
    "Left Defensive Midfield", "Right Defensive Midfield", "Left Midfield", "Right Midfield",
    "Center Attacking Midfield", "Left Attacking Midfield", "Right Attacking Midfield",
    "Left Wing", "Right Wing", "Center Forward", "Left Center Forward", "Right Center Forward",
}
YARDS_TO_M = 0.9144

def dist_yd(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def cluster_lines(xs, gap):
    if not xs: return []
    xs = sorted(xs)
    lines, current = [], [xs[0]]
    for x in xs[1:]:
        if x - current[-1] > gap:
            lines.append(current); current = [x]
        else:
            current.append(x)
    lines.append(current)
    return [sum(c)/len(c) for c in lines]

# ---- Build the base carry table once (features + raw opponent x-lists), gap applied later ----
base_rows = []
for ef in sorted(glob.glob("events/*.json")):
    match_id = ef.split("/")[-1].replace(".json", "")
    events = json.load(open(ef))
    try:
        frames = json.load(open(f"threesixty/{match_id}.json"))
        frame_by_event = {f["event_uuid"]: f for f in frames}
    except FileNotFoundError:
        frame_by_event = {}
    for e in events:
        if e["type"]["name"] != "Carry": continue
        pos = e.get("position", {}).get("name")
        if pos not in MID_ATT_POSITIONS: continue
        if "location" not in e or "carry" not in e or "end_location" not in e["carry"]: continue
        start, end = e["location"], e["carry"]["end_location"]
        duration = e.get("duration")
        if duration is None or duration <= 0: continue
        dist_yards = dist_yd(start, end)
        dist_m = dist_yards * YARDS_TO_M
        speed_ms = dist_m / duration
        if dist_yards < 3.0 or duration < 0.4 or speed_ms > 12.0: continue
        frame = frame_by_event.get(e["id"])
        if not (frame and frame.get("freeze_frame")): continue
        opp_players = [pl for pl in frame["freeze_frame"] if not pl.get("teammate", True)]
        opp_locs = [pl["location"] for pl in opp_players]
        outfield_x = [pl["location"][0] for pl in opp_players if not pl.get("keeper", False)]
        if not opp_locs: continue
        defender_prox_m = min(dist_yd(start, o) * YARDS_TO_M for o in opp_locs)
        base_rows.append({
            "carry_distance_m": dist_m, "carry_duration_s": duration, "carry_speed_ms": speed_ms,
            "defender_proximity_m": defender_prox_m, "under_pressure": int(bool(e.get("under_pressure", False))),
            "start_x": start[0], "end_x": end[0], "outfield_x": outfield_x,
        })

df = pd.DataFrame(base_rows)
print(f"Base carries with 360 features: {len(df)}")

feat_cols = ["carry_distance_m", "carry_speed_ms", "defender_proximity_m", "under_pressure"]

# 60/20/20 train/val/test split, fixed once so every gap is evaluated on the SAME split
idx = np.arange(len(df))
idx_train, idx_temp = train_test_split(idx, test_size=0.4, random_state=42)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

gap_grid = [3, 4, 5, 6, 7, 8, 9, 10]
val_results = []
for gap in gap_grid:
    lines_broken = []
    for _, row in df.iterrows():
        if row["end_x"] > row["start_x"]:
            lo, hi = row["start_x"], row["end_x"]
            line_positions = cluster_lines(row["outfield_x"], gap)
            broken = sum(1 for lx in line_positions if lo < lx <= hi)
        else:
            broken = 0
        lines_broken.append(int(broken >= 1))
    y = np.array(lines_broken)
    rate = y.mean()

    X = df[feat_cols].values
    Xtr, ytr = X[idx_train], y[idx_train]
    Xval, yval = X[idx_val], y[idx_val]
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    gb.fit(Xtr, ytr)
    val_auc = roc_auc_score(yval, gb.predict_proba(Xval)[:, 1])
    val_results.append({"gap_yd": gap, "gap_m": round(gap*YARDS_TO_M,2), "line_break_rate": round(100*rate,2), "val_auc": round(val_auc,4)})
    print(f"gap={gap}yd ({gap*YARDS_TO_M:.1f}m)  rate={100*rate:.2f}%  val_AUC={val_auc:.4f}")

best = max(val_results, key=lambda r: r["val_auc"])
print(f"\nBest gap by validation AUC: {best}")

# Final test-set evaluation at the chosen gap (never touched during selection)
chosen_gap = best["gap_yd"]
lines_broken = []
for _, row in df.iterrows():
    if row["end_x"] > row["start_x"]:
        lo, hi = row["start_x"], row["end_x"]
        line_positions = cluster_lines(row["outfield_x"], chosen_gap)
        broken = sum(1 for lx in line_positions if lo < lx <= hi)
    else:
        broken = 0
    lines_broken.append(int(broken >= 1))
y = np.array(lines_broken)
X = df[feat_cols].values
Xtrval = np.concatenate([X[idx_train], X[idx_val]])
ytrval = np.concatenate([y[idx_train], y[idx_val]])
gb_final = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
gb_final.fit(Xtrval, ytrval)
test_auc = roc_auc_score(y[idx_test], gb_final.predict_proba(X[idx_test])[:, 1])
print(f"Final held-out TEST AUC at gap={chosen_gap}yd: {test_auc:.4f}")

json.dump({"grid": val_results, "chosen_gap_yd": chosen_gap, "chosen_gap_m": round(chosen_gap*YARDS_TO_M,2), "test_auc": round(test_auc,4)},
          open("gap_grid_search.json", "w"), indent=2)
