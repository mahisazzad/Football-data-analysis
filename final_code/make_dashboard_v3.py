import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt

results = json.load(open("results_v3.json"))
grid = json.load(open("gap_grid_search.json"))
s2_stats = pd.read_csv("player_stats_season2.csv")

plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Ball-Carrying Kinematics v3 — Grid Search, xG Linkage, Season-2 Replication", fontsize=14, fontweight="bold", color="#ffffff", y=0.995)
fig.text(0.5, 0.965, "La Liga 2020/21 + Ligue 1 2021/22 — Real StatsBomb Event & 360° Data", ha="center", va="center", fontsize=10.5, color="#cbd5e1")
fig.text(0.5, 0.935, "REAL MATCH DATA — StatsBomb Open Data (two seasons, two leagues)", ha="center", va="center", fontsize=10, fontweight="bold", color="#7ee787",
          bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d2818", edgecolor="#2ea043"))

# Panel 1: grid search curve
ax1 = axes[0, 0]
g = pd.DataFrame(grid["grid"])
ax1.plot(g["gap_yd"], g["val_auc"], marker="o", color="#00ff87", linewidth=2, markersize=7)
ax1.axvline(grid["chosen_gap_yd"], color="#f59e0b", linestyle="--", linewidth=1.2, label=f"Chosen gap = {grid['chosen_gap_yd']} yd")
ax1.set_title("1. Line-Clustering Gap: Grid Search", fontsize=11, fontweight="bold", color="#e2e8f0", pad=10)
ax1.set_xlabel("Clustering gap (yards)", fontsize=9, color="#94a3b8")
ax1.set_ylabel("Validation AUC (line-break classifier)", fontsize=9, color="#94a3b8")
ax1.legend(facecolor="#0b132b", edgecolor="#1e293b", labelcolor="white", fontsize=8)
ax1.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0")

# Panel 2: xG linkage — progressive vs line-break
ax2 = axes[0, 1]
cats = ["Not\nprogressive", "Progressive", "No line\nbreak", "Line\nbreak"]
vals = [results["xg_combined"]["xg_not_prog"], results["xg_combined"]["xg_prog"],
        results["xg_combined"]["xg_no_break"], results["xg_combined"]["xg_break"]]
colors = ["#64748b", "#f59e0b", "#64748b", "#10b981"]
bars = ax2.bar(cats, vals, color=colors, edgecolor="#334155", width=0.6)
ax2.set_title("2. Downstream Shot Value (xG) — Combined, Real", fontsize=11, fontweight="bold", color="#e2e8f0", pad=10)
ax2.set_ylabel("Avg. xG from a shot later in same possession", fontsize=8.5, color="#94a3b8")
for bar in bars:
    h = bar.get_height()
    ax2.text(bar.get_x()+bar.get_width()/2, h+0.001, f"{h:.3f}", ha="center", color="white", fontsize=9, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0", axis="y")

# Panel 3: season-2 player line-break / progressive rates
ax3 = axes[1, 0]
plot_stats = s2_stats.sort_values("line_break_pct", ascending=True)
colors3 = ["#10b981" if pl != "Rest of sample" else "#f59e0b" for pl in plot_stats["player"]]
ax3.barh(plot_stats["player"], plot_stats["line_break_pct"], color=colors3, edgecolor="#334155", height=0.65)
ax3.set_title("3. Ligue 1 (PSG) Line-Break Rate — Real, Out-of-Sample", fontsize=10, fontweight="bold", color="#e2e8f0", pad=10)
ax3.set_xlabel("% of carries crossing \u22651 clustered opponent line", fontsize=8.5, color="#94a3b8")
ax3.tick_params(axis="y", labelsize=8)
ax3.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0", axis="x")

# Panel 4: AUC replication across seasons
ax4 = axes[1, 1]
labels4 = ["S1: 5 feat", "S1: +progressive", "S2: 5 feat", "S2: +progressive"]
vals4 = [results["auc_s1_5feat"], results["auc_s1_6feat"], results["auc_s2_5feat"], results["auc_s2_6feat"]]
colors4 = ["#3b82f6", "#f59e0b", "#3b82f6", "#f59e0b"]
bars4 = ax4.bar(labels4, vals4, color=colors4, edgecolor="#1d4ed8", width=0.6)
ax4.set_title("4. Model Replication: La Liga vs Ligue 1", fontsize=11, fontweight="bold", color="#e2e8f0", pad=10)
ax4.set_ylabel("ROC-AUC (line-break classifier)", fontsize=9, color="#94a3b8")
ax4.set_ylim(0.6, 0.85)
for bar in bars4:
    h = bar.get_height()
    ax4.text(bar.get_x()+bar.get_width()/2, h+0.005, f"{h:.3f}", ha="center", color="white", fontsize=9, fontweight="bold")
ax4.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0", axis="y")

for ax in axes.flat:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1e293b"); ax.spines["bottom"].set_color("#1e293b")

fig.text(0.5, 0.012, "Source: StatsBomb open data. Season 1 = Barcelona's La Liga 2020/21; Season 2 = PSG's Ligue 1 2021/22 (Messi, Mbapp\u00e9, Neymar).",
          ha="center", va="center", fontsize=8, color="#94a3b8", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.savefig("dashboard_v3.png", dpi=300)
print("saved dashboard_v3.png")
