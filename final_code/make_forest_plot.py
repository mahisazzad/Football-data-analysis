import json
import numpy as np
import matplotlib.pyplot as plt

res = json.load(open("inference_results.json"))

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.patch.set_facecolor("#0b132b")

for ax, key, title, baseline in [
    (axes[0], "fe_season1", "La Liga 2020/21\n(baseline: Aspas)", "Aspas"),
    (axes[1], "fe_season2", "Ligue 1 2021/22\n(baseline: Danilo Pereira)", "Danilo Pereira"),
]:
    ax.set_facecolor("#111827")
    tab = sorted(res[key]["table"], key=lambda r: r["coef"])
    names = [r["player"] for r in tab]
    coefs = [r["coef"] for r in tab]
    los = [r["coef"] - r["ci_low"] for r in tab]
    his = [r["ci_high"] - r["coef"] for r in tab]
    sig = [r["p_value"] < 0.05 for r in tab]
    colors = ["#10b981" if s else "#64748b" for s in sig]
    y_pos = np.arange(len(names))
    for i in range(len(names)):
        ax.errorbar([coefs[i]], [y_pos[i]], xerr=[[los[i]], [his[i]]], fmt="o", color="white",
                    ecolor=colors[i], elinewidth=2.5, capsize=4, markersize=7,
                    markerfacecolor="white", zorder=3)
    ax.axvline(0, color="#f59e0b", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Log-odds coefficient vs. baseline (95% CI)", fontsize=9, color="#94a3b8")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#e2e8f0")
    ax.grid(True, linestyle="--", alpha=0.15, color="#e2e8f0", axis="x")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Player Fixed Effects with 95% Confidence Intervals (Formal Hypothesis Testing)",
             fontsize=14, fontweight="bold", color="white", y=1.04)
fig.text(0.5, -0.03,
          "Green = statistically significant at p<0.05 (crosses zero = not distinguishable from baseline player). "
          "Season 1: no player reaches significance. Season 2: Messi (PSG) is the one significant positive effect.",
          ha="center", fontsize=9.5, color="#cbd5e1", style="italic")

plt.tight_layout()
plt.savefig("forest_plot_fixed_effects.png", dpi=200, bbox_inches="tight")
print("saved forest_plot_fixed_effects.png")
