import json, glob, math
import pandas as pd

MID_ATT_POSITIONS = {
    "Left Center Midfield", "Right Center Midfield", "Center Defensive Midfield",
    "Left Defensive Midfield", "Right Defensive Midfield", "Left Midfield", "Right Midfield",
    "Center Attacking Midfield", "Left Attacking Midfield", "Right Attacking Midfield",
    "Left Wing", "Right Wing", "Center Forward", "Left Center Forward", "Right Center Forward",
}
YARDS_TO_M = 0.9144

def dist_yd(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def build_base(events_dir, threesixty_dir):
    rows = []
    for ef in sorted(glob.glob(f"{events_dir}/*.json")):
        match_id = ef.split("/")[-1].replace(".json", "")
        events = json.load(open(ef))
        try:
            frames = json.load(open(f"{threesixty_dir}/{match_id}.json"))
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
            opp_pts = [pl["location"] for pl in frame["freeze_frame"] if not pl.get("teammate", True) and not pl.get("keeper", False)]
            if not opp_pts: continue
            defender_prox_m = min(dist_yd(start, o) * YARDS_TO_M for o in opp_pts)
            rows.append({
                "match_id": match_id, "start": start, "end": end,
                "carry_distance_m": dist_m, "carry_duration_s": duration, "carry_speed_ms": speed_ms,
                "defender_proximity_m": defender_prox_m,
                "under_pressure": int(bool(e.get("under_pressure", False))),
                "opp_pts": opp_pts,
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df1 = build_base("events", "threesixty")
    print("Season 1 base carries:", len(df1))
    df1.to_pickle("base_carries_s1.pkl")
