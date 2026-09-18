
# Kinematic & Spatial Football Analysis Pipeline

An end-to-end data processing and statistical analysis pipeline for evaluating football (soccer) kinematic carry data and spatial positioning using StatsBomb open data. The system parses event and 360° freeze-frame data to detect progressive carries, line-breaking events, pitch-control dynamics, and downstream expected goals (xG) impact.

---

### Core Features

* **360° Spatial Geometry:** Utilizes DBSCAN clustering, convex hulls, path intersections, and Voronoi tessellation to compute pitch control and defensive line structures.


* **Hyperparameter Optimization:** Conducts 1-D depth and 2-D lateral grid searches to establish empirically validated thresholds for spatial defensive lines.


* **Rigorous Statistical Modeling:** Runs OLS, logistic regressions, player fixed-effects, and bootstrap confidence intervals to measure kinematic impact.


* **Automated Visualizations:** Generates comprehensive visual dashboards, single-carry spatial maps, and fixed-effects forest plots.



---

### Key File Outputs

| Component | Key Files / Scripts | Outputs Generated | Description |
| --- | --- | --- | --- |
| **Data Ingestion** | `01_download.py` <br>

<br> `02_build_base_carries.py` | `events/*.json`

<br>

<br> `threesixty/*.json`

<br>

<br> `base_carries_s1.pkl`<br> | Downloads raw StatsBomb open data and parses initial carry features.

 |
| **Spatial Engine** | `spatial_library.py` <br>

<br> `03_1d_gap_grid_search.py` <br>

<br> `04_2d_grid_search.py` | `gap_grid_search.json`

<br>

<br> `data/spatial_params_final.json`<br> | Contains core geometric algorithms and runs parameter validation grid searches.

 |
| **Main Pipeline** | `05_main_pipeline.py` | `data/carries_final.csv`<br> | Integrates line-breaking logic, progressive carry flags, and downstream xG.

 |
| **Inference** | `06_formal_statistics.py` | `data/inference_results.json`<br> | Executes formal statistical models, regressions, and bootstrap CIs.

 |
| **Dashboards** | `07_dashboard_v2.py` <br>

<br> `08_dashboard_v3.py` | `real_kinematic_dashboard_v2.png`

<br>

<br> `dashboard_v3.png`<br> | Produces diagnostic plots for speed, line-break rates, xG linkage, and model AUC.

 |
| **Visualizations** | `09_spatial_example.py` <br>

<br> `10_forest_plot.py` | `spatial_example.png`

<br>

<br> `forest_plot_fixed_effects.png`<br> | Generates single-carry pitch control maps and player effect forest plots.

 |

---

### Prerequisites

Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn

```

---

### Execution Guide

Run the pipeline in sequential numerical order to execute the full data processing and model evaluation workflow:

```bash
# 1. Download open data and parse base carry features
python 01_download.py
python 02_build_base_carries.py

# 2. Optimize spatial line-clustering parameters
python 03_1d_gap_grid_search.py
python 04_2d_grid_search.py

# 3. Process main feature extraction and run statistical models
python 05_main_pipeline.py
python 06_formal_statistics.py

# 4. Generate visual dashboards and spatial plots
python 07_dashboard_v2.py
python 08_dashboard_v3.py
python 09_spatial_example.py
python 10_forest_plot.py

```
