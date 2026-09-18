import json, glob, math, csv
import spatial_lib
from spatial_lib import carry_path_crosses_hulls, voronoi_control_gained

spatial_lib.Y_WEIGHT = 0.35   # grid-search-validated, forward-progress-only version
EPS_2D = 6                     # grid-search-validated

MID_ATT_POSITIONS = {
    "Left Center Midfield", "Right Center Midfield", "Center Defensive Midfield",
    "Left Defensive Midfield", "Right Defensive Midfield", "Left Midfield", "Right Midfield",
    "Center Attacking Midfield", "Left Attacking Midfield", "Right Attacking Midfield",
    "Left Wing", "Right Wing", "Center Forward", "Left Center Forward", "Right Center Forward",
}
YARDS_TO_M = 0.9144
GOAL = (120.0, 40.0)
FINAL_40_X = 120 * 0.6
QUICK_TRANSITION_SEC = 6.0  # "counter-press window" heuristic for chaining possessions

def dist_yd(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
def dist_to_goal_yd(pt):
    return dist_yd(pt, GOAL)
def ts_to_sec(ts):
    h, m, s = ts.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

def build_possession_chains(events):
    """Group possession segments into chains: consecutive spells by the same team
    stay in one chain even through a brief (<QUICK_TRANSITION_SEC) opponent spell,
    modelling a loose-ball/counter-press regain rather than a genuine reset."""
    segments = []  # (period, possession, team_id, start_idx, end_idx, start_sec, end_sec)
    cur = None
    for e in events:
        key = (e.get("period"), e.get("possession"))
        team_id = e.get("possession_team", {}).get("id")
        if team_id is None:
            continue
        try:
            t = ts_to_sec(e["timestamp"])
        except Exception:
            continue
        if cur is None or (cur["period"], cur["possession"]) != key:
            if cur is not None:
                segments.append(cur)
            cur = {"period": e.get("period"), "possession": e.get("possession"), "team_id": team_id,
                   "start_idx": e["index"], "end_idx": e["index"], "start_sec": t, "end_sec": t}
        else:
            cur["end_idx"] = e["index"]
            cur["end_sec"] = t
    if cur is not None:
        segments.append(cur)

    chain_id_by_possession = {}
    last_team_segment = {}  # team_id -> (chain_id, end_sec) most recent segment for that team
    next_chain_id = 0
    for seg in segments:
        t = seg["team_id"]
        if t in last_team_segment:
            prev_chain, prev_end_sec = last_team_segment[t]
            gap = seg["start_sec"] - prev_end_sec
            if seg["period"] == segments[0]["period"] or True:  # same-period check applied via timestamps resetting; gap covers it
                pass
            if gap <= QUICK_TRANSITION_SEC and gap >= 0:
                chain_id = prev_chain
            else:
                chain_id = next_chain_id; next_chain_id += 1
        else:
            chain_id = next_chain_id; next_chain_id += 1
        chain_id_by_possession[(seg["period"], seg["possession"])] = chain_id
        last_team_segment[t] = (chain_id, seg["end_sec"])
    return chain_id_by_possession

def parse_season(events_dir, threesixty_dir, season_tag):
    rows = []
    for ef in sorted(glob.glob(f"{events_dir}/*.json")):
        match_id = ef.split("/")[-1].replace(".json", "")
        events = json.load(open(ef))
        try:
            frames = json.load(open(f"{threesixty_dir}/{match_id}.json"))
            frame_by_event = {f["event_uuid"]: f for f in frames}
        except FileNotFoundError:
            frame_by_event = {}

        chain_by_poss = build_possession_chains(events)

        # shots grouped by (team_id, chain_id) -> list of (index, xg)
        shots_by_chain = {}
        for e in events:
            if e["type"]["name"] == "Shot" and "shot" in e:
                key_poss = (e.get("period"), e.get("possession"))
                chain_id = chain_by_poss.get(key_poss)
                team_id = e.get("team", {}).get("id")
                shots_by_chain.setdefault((team_id, chain_id), []).append(
                    (e["index"], e["shot"].get("statsbomb_xg", 0.0))
                )
            # also same-possession only (for comparison / backward compatibility)

        shots_by_possession = {}
        for e in events:
            if e["type"]["name"] == "Shot" and "shot" in e:
                key = (e.get("possession"), e.get("team", {}).get("id"))
                shots_by_possession.setdefault(key, []).append((e["index"], e["shot"].get("statsbomb_xg", 0.0)))

        for e in events:
            if e["type"]["name"] != "Carry":
                continue
            pos = e.get("position", {}).get("name")
            if pos not in MID_ATT_POSITIONS:
                continue
            if "location" not in e or "carry" not in e or "end_location" not in e["carry"]:
                continue
            start, end = e["location"], e["carry"]["end_location"]
            duration = e.get("duration")
            if duration is None or duration <= 0:
                continue
            dist_yards = dist_yd(start, end)
            dist_m = dist_yards * YARDS_TO_M
            speed_ms = dist_m / duration
            if dist_yards < 3.0 or duration < 0.4 or speed_ms > 12.0:
                continue

            d_start_goal_m = dist_to_goal_yd(start) * YARDS_TO_M
            d_end_goal_m = dist_to_goal_yd(end) * YARDS_TO_M
            progression_m = d_start_goal_m - d_end_goal_m
            threshold_m = 5.0 if start[0] >= FINAL_40_X else 10.0
            progressive_carry = int(progression_m >= threshold_m)

            frame = frame_by_event.get(e["id"])
            defender_prox_m = None
            lines_broken_2d = None
            n_opponents = None
            pitch_control_gained = None
            if frame and frame.get("freeze_frame"):
                opp_pts = [pl["location"] for pl in frame["freeze_frame"] if not pl.get("teammate", True) and not pl.get("keeper", False)]
                team_pts = [pl["location"] for pl in frame["freeze_frame"] if pl.get("teammate", True) and not pl.get("actor", False)]
                n_opponents = len(opp_pts)
                if opp_pts:
                    defender_prox_m = min(dist_yd(start, o) * YARDS_TO_M for o in opp_pts)
                    crossed, _ = carry_path_crosses_hulls(start, end, opp_pts, EPS_2D)
                    lines_broken_2d = crossed
                    pitch_control_gained = voronoi_control_gained(start, end, team_pts, opp_pts, start)

            # ---- downstream value: single-possession (old) vs multi-phase chain (new) ----
            key_single = (e.get("possession"), e.get("team", {}).get("id"))
            downstream_xg_single = sum(xg for idx, xg in shots_by_possession.get(key_single, []) if idx > e["index"])

            key_poss = (e.get("period"), e.get("possession"))
            chain_id = chain_by_poss.get(key_poss)
            team_id = e.get("team", {}).get("id")
            downstream_xg_chain = sum(xg for idx, xg in shots_by_chain.get((team_id, chain_id), []) if idx > e["index"])

            rows.append({
                "season": season_tag, "match_id": match_id,
                "player": e["player"]["name"], "player_id": e["player"]["id"],
                "team": e["team"]["name"], "position": pos,
                "period": e["period"], "minute": e["minute"],
                "start_x": start[0], "start_y": start[1], "end_x": end[0], "end_y": end[1],
                "carry_distance_m": round(dist_m, 4), "carry_duration_s": round(duration, 4),
                "carry_speed_ms": round(speed_ms, 4),
                "under_pressure": int(bool(e.get("under_pressure", False))),
                "progressive_carry": progressive_carry,
                "defender_proximity_m": round(defender_prox_m, 4) if defender_prox_m is not None else "",
                "n_opponents_in_frame": n_opponents if n_opponents is not None else "",
                "lines_broken_2d": lines_broken_2d if lines_broken_2d is not None else "",
                "line_break_2d": int(lines_broken_2d >= 1) if lines_broken_2d is not None else "",
                "pitch_control_gained": round(pitch_control_gained, 5) if pitch_control_gained is not None else "",
                "downstream_xg_single_poss": round(downstream_xg_single, 4),
                "downstream_xg_chain": round(downstream_xg_chain, 4),
            })
    return rows

if __name__ == "__main__":
    rows = []
    rows += parse_season("events", "threesixty", "LaLiga_2020_21")
    rows += parse_season("season2/events", "season2/threesixty", "Ligue1_2021_22")
    print(f"Total qualifying carries: {len(rows)}")
    with open("carries_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved carries_final.csv")
