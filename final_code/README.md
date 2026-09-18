# Beyond Distance — Ball-Carrying Kinematics Pipeline

This is the complete, final, reproducible codebase behind the "Beyond Distance" research
(La Liga 2020/21 + Ligue 1 2021/22, StatsBomb open data, 16,472 real carries). It reflects
the **corrected** pipeline — including the forward-progress bug fix documented in the
thesis (Section 3.6) — not the earlier draft versions.

Everything runs on **StatsBomb's free open-data release**
(https://github.com/statsbomb/open-data). No API key or paid data access is needed.

## Pipeline order

Run these in sequence. Each step's output feeds the next.

| # | Script | What it does | Produces |
|---|--------|--------------|----------|
| 1 | `redownload.py` | Downloads real match event + 360° freeze-frame JSON files from StatsBomb open data, in rate-limit-safe batches. Usage: `python3 redownload.py <matches_file> <events_dir> <threesixty_dir> <start_idx> <batch_size>` | `events/*.json`, `threesixty/*.json` per season |
| 2 | `build_base.py` | Parses raw match files once into a cached table of carry-level features (used repeatedly by the grid searches below, so this avoids re-parsing JSON for every parameter tried). | `base_carries_s1.pkl` |
| 3 | `grid_search_gap.py` | **Initial (1-D) specification.** Validates the depth-only line-clustering gap threshold via held-out train/val/test grid search. This is the *first* version of the line-break metric — thesis Table 3.4. | `spatial_params` (1-D), printed results |
| 4 | `spatial_lib.py` | **Core geometry module.** DBSCAN clustering, convex-hull construction, forward-progress-only path intersection, and static Voronoi pitch control. This is the corrected version (see "Known issue, fixed" below) — imported by scripts 5–7 and 10. Not run directly. | — (library) |
| 5 | `grid_search_2d.py` | **2-D upgrade validation.** Grid-searches the two spatial parameters (lateral-distance weight, clustering gap) for the genuine 2-D method, restricted to tactically realistic line counts. Thesis Table 3.5. | `data/spatial_params_final.json` |
| 6 | `parse_final.py` | **Main pipeline.** Re-parses all real match files using the validated 2-D parameters: computes progressive-carry flags, 2-D line-breaking, static pitch control, and multi-phase (possession-chain) downstream xG. This is the single source of truth for all reported numbers. | `data/carries_final.csv` |
| 7 | `stats_inference.py` | **Formal statistics.** OLS (robust SE) for pitch-control-gained, logistic regression for line-breaking (6-feature baseline), player fixed-effects logistic regression with full p-values/CIs (both seasons), and bootstrap 95% CIs for classifier AUC. Thesis Tables 4.3–4.7. | `data/inference_results.json` |
| 8 | `make_dashboard_v2.py` | Season 1 descriptive dashboard (speed distribution, line-break rates, feature importance). Thesis Figure 4.1. | `real_kinematic_dashboard_v2.png` |
| 9 | `make_dashboard_v3.py` | Grid-search curve, xG-linkage comparison, Season 2 replication, AUC comparison. Addendum figure. | `dashboard_v3.png` |
| 10 | `make_spatial_example.py` | Renders one real carry's convex hulls and Voronoi pitch-control grid. Thesis Figure 3.2/3.3. | `spatial_example.png` |
| 11 | `make_forest_plot.py` | Player fixed-effects coefficients with 95% CIs, both seasons. Thesis Figure 4.2. | `forest_plot_fixed_effects.png` |

## `data/` — final outputs already generated

- `carries_final.csv` — the master dataset: 16,472 real carries, both seasons, every metric.
- `inference_results.json` — every regression table's coefficients, SEs, p-values, CIs.
- `spatial_params_final.json` — the validated 2-D clustering parameters (y_weight=0.35, eps=6).
- `gap_grid_search.json` — the 1-D grid search results (superseded by the 2-D method, kept for the methodology narrative in Section 3.5).

If you just want to re-run the analysis without re-downloading ~400MB of match data,
start from `data/carries_final.csv` at step 6 — steps 1–5 are only needed to regenerate
that file from scratch or to re-validate the parameters yourself.

## Known issue, fixed (keep this in your presentation notes)

An early version of `spatial_lib.py`'s `carry_path_crosses_hulls()` had no directionality
check, so backward/lateral carries could register as line-breaking by geometrically
grazing a hull (affected 31% of carries, 22.8% of which were wrongly flagged). The current
version requires `end[0] > start[0]` before any hull-intersection test runs. Every grid
search, dataset, and statistical result in this repository was regenerated after that fix
— none of it reflects the buggy version.

## Requirements

```
pip install pandas numpy scipy scikit-learn shapely statsmodels matplotlib --break-system-packages
```

## Data source

StatsBomb Open Data, competition_id 11 / season_id 90 (La Liga 2020/21) and
competition_id 7 / season_id 108 (Ligue 1 2021/22).
https://github.com/statsbomb/open-data
