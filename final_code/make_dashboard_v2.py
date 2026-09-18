import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt

df = pd.read_csv("carries_full_v2.csv")
stats = pd.read_csv("player_stats_v2.csv")
results = json.load(open("model_results_v2.json"))

plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Ball-Carrying Kinematics v2 — Refined Definitions, Real StatsBomb Data", fontsize=14.5, fontweight="bold", color="#ffffff", y=0.995)
fig.text(0.5, 0.965, "Progressive carry (The Football Analyst / StatsBomb-FBref def.) + line-clustering line-break metric", ha="center", va="center", fontsize=10, color="#cbd5e1")
fig.text(0.5, 0.935, "REAL MATCH DATA — StatsBomb Open Data (events + 360 freeze frames)", ha="center", va="center", fontsize=10, fontweight="bold", color="#7ee787",
          bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d2818", edgecolor="#2ea043"))

featured = df[df["is_featured"]]
rest = df[~df["is_featured"]]

# Panel 1: carry speed distribution (unchanged, still real baseline)
ax1 = axes[0, 0]
ax1.hist(rest["carry_speed_ms"], bins=40, alpha=0.5, color="#64748b", label=f"Rest of sample (n={len(rest):,})", density=True)
ax1.hist(featured["carry_speed_ms"], bins=40, alpha=0.65, color="#00ff87", label=f"Featured 15 players (n={len(featured):,})", density=True)
ax1.set_title("1. Average Carry Speed — Real Distribution", fontsize=11, fontweight="bold", color="#e2e8f0", pad=10)
ax1.set_xlabel("Average carry speed, distance/duration (m/s)", fontsize=9, color="#94a3b8")
ax1.set_ylabel("Density", fontsize=9, color="#94a3b8")
ax1.legend(facecolor="#0b132b", edgecolor="#1e293b", labelcolor="white", fontsize=8)
ax1.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0")

# Panel 2: mass-sensitivity check -- KE under position-based vs flat mass model
ax2 = axes[0, 1]
lims = [0, max(df["ke_mid_j"].max(), df["ke_flat_j"].max()) * 1.03]
ax2.plot(lims, lims, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.8, label="y = x (no mass effect)")
ax2.scatter(rest["ke_flat_j"], rest["ke_mid_j"], color="#64748b", alpha=0.12, s=8, edgecolor="none")
ax2.scatter(featured["ke_flat_j"], featured["ke_mid_j"], color="#00ff87", alpha=0.35, s=14, edgecolor="none")
ax2.set_xlim(lims); ax2.set_ylim(lims)
ax2.set_title("2. Mass-Sensitivity Check: KE Proxy", fontsize=11, fontweight="bold", color="#e2e8f0", pad=10)
ax2.set_xlabel("KE using flat 80kg mass (J)", fontsize=9, color="#94a3b8")
ax2.set_ylabel("KE using position-based mass (J)", fontsize=9, color="#94a3b8")
ax2.legend(facecolor="#0b132b", edgecolor="#1e293b", labelcolor="white", fontsize=8, loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0")

# Panel 3: refined line-break rate by player
ax3 = axes[1, 0]
plot_stats = stats.sort_values("line_break_rate_v2_pct", ascending=True)
colors = ["#10b981" if pl != "Rest of sample (pooled)" else "#f59e0b" for pl in plot_stats["player"]]
ax3.barh(plot_stats["player"], plot_stats["line_break_rate_v2_pct"], color=colors, edgecolor="#334155", height=0.65)
ax3.axvline(stats.loc[stats["player"]=="Rest of sample (pooled)","line_break_rate_v2_pct"].values[0],
            color="#f59e0b", linestyle="--", linewidth=1, alpha=0.7)
ax3.set_title("3. Refined Line-Break Rate (Defensive-Line Clustering)", fontsize=10.5, fontweight="bold", color="#e2e8f0", pad=10)
ax3.set_xlabel("% of carries crossing \u22651 clustered opponent line", fontsize=8.5, color="#94a3b8")
ax3.tick_params(axis="y", labelsize=8)
ax3.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0", axis="x")

# Panel 4: feature importance, model 2b (with progressive_carry)
ax4 = axes[1, 1]
imp = results["model2b_importances"]
labels_map = {"carry_distance_m":"Carry distance","carry_duration_s":"Carry duration","carry_speed_ms":"Carry speed",
              "defender_proximity_m":"Defender proximity","under_pressure":"Under pressure (flag)","progressive_carry":"Progressive carry (flag)"}
names = [labels_map[k] for k in imp.keys()]
vals = list(imp.values())
order = np.argsort(vals)
names = [names[i] for i in order]; vals = [vals[i] for i in order]
bars = ax4.barh(names, vals, color=["#3b82f6" if n!="Progressive carry (flag)" else "#f59e0b" for n in names], edgecolor="#1d4ed8", height=0.55)
ax4.set_title(f"4. Feature Importance incl. Progressive Carry (AUC={results['model2b_auc']:.3f})", fontsize=10, fontweight="bold", color="#e2e8f0", pad=10)
ax4.set_xlabel("Relative importance (%) — gradient boosting, refined target", fontsize=8.5, color="#94a3b8")
for bar in bars:
    w = bar.get_width()
    ax4.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", ha="left", color="#ffffff", fontsize=8, fontweight="bold")
ax4.set_xlim(0, max(vals) + 15)
ax4.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0")

for ax in axes.flat:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1e293b"); ax.spines["bottom"].set_color("#1e293b")

fig.text(0.5, 0.012,
          "Source: StatsBomb open data, La Liga 2020/21. Progressive carry def.: The Football Analyst / StatsBomb-FBref. "
          "Line-break: opponent-line clustering, refined from a simpler bypass count.",
          ha="center", va="center", fontsize=8, color="#94a3b8", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.savefig("real_kinematic_dashboard_v2.png", dpi=300)
print("saved real_kinematic_dashboard_v2.png")
