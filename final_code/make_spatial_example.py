import pickle, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import spatial_lib
spatial_lib.Y_WEIGHT = 0.35
from spatial_lib import cluster_lines_2d, cluster_hull, voronoi_control_gained
from scipy.spatial import cKDTree

row = pickle.load(open("example_carry.pkl", "rb"))
start, end = row["start"], row["end"]
opp_pts = row["opp_pts"]

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# --- Panel 1: pitch, opponents, clusters, hulls, carry path ---
ax = axes[0]
ax.set_facecolor("#0b3d0b")
ax.add_patch(plt.Rectangle((0, 0), 120, 80, fill=False, edgecolor="white", linewidth=1.5))
ax.plot([60, 60], [0, 80], color="white", linewidth=0.8, alpha=0.5)

clusters = cluster_lines_2d(opp_pts, 6)
colors = plt.cm.autumn(np.linspace(0.1, 0.9, len(clusters)))
for c, col in zip(clusters, colors):
    hull = cluster_hull(c)
    if hull.geom_type == "Polygon":
        xs, ys = hull.exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor=col, alpha=0.35, edgecolor=col, linewidth=2))
    ax.scatter(c[:, 0], c[:, 1], color=col, s=90, edgecolor="white", zorder=5)

ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#00ff87", lw=3))
ax.scatter([start[0]], [start[1]], color="#00ff87", s=140, marker="o", edgecolor="white", zorder=6, label="Carry start")
ax.scatter([end[0]], [end[1]], color="#00ff87", s=140, marker="s", edgecolor="white", zorder=6, label="Carry end")
ax.set_xlim(0, 120); ax.set_ylim(0, 80)
ax.set_title(f"2-D Line-Break Detection: Convex Hulls\n({len(clusters)} lines detected, real 360\u00b0 opponent positions)", fontsize=11, fontweight="bold", color="#e2e8f0")
ax.set_xlabel("Attacking direction \u2192 (yards)", fontsize=9, color="#94a3b8")
ax.legend(loc="upper left", facecolor="#0b132b", edgecolor="#1e293b", labelcolor="white", fontsize=8)
ax.set_aspect("equal")

# --- Panel 2: Voronoi pitch control heatmap around the carry ---
ax2 = axes[1]
ax2.set_facecolor("#0b3d0b")
pad = 8
lo_x, hi_x = max(0, min(start[0], end[0]) - pad), min(120, max(start[0], end[0]) + pad)
lo_y, hi_y = max(0, min(start[1], end[1]) - pad), min(80, max(start[1], end[1]) + pad)
xs = np.arange(lo_x, hi_x, 1.0); ys = np.arange(lo_y, hi_y, 1.0)
gx, gy = np.meshgrid(xs, ys)
grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

team_pts = [[start[0]-8, start[1]+6], [start[0]-4, start[1]-10], [start[0]+2, start[1]+14]]  # illustrative teammates
all_after = np.array(team_pts + opp_pts + [end])
team_idx = set(range(len(team_pts) + 1))
tree = cKDTree(all_after)
_, nn = tree.query(grid_pts)
control = np.array([1 if i in team_idx else 0 for i in nn]).reshape(gx.shape)

ax2.contourf(gx, gy, control, levels=[-0.5, 0.5, 1.5], colors=["#7f1d1d", "#14532d"], alpha=0.55)
ax2.scatter([p[0] for p in opp_pts], [p[1] for p in opp_pts], color="#ef4444", s=70, edgecolor="white", zorder=5, label="Opponents")
ax2.scatter([p[0] for p in team_pts], [p[1] for p in team_pts], color="#22c55e", s=70, edgecolor="white", zorder=5, label="Teammates (illustrative)")
ax2.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#00ff87", lw=3))
ax2.scatter([start[0]], [start[1]], color="#facc15", s=140, marker="o", edgecolor="white", zorder=6, label="Carrier start\u2192end")
ax2.scatter([end[0]], [end[1]], color="#facc15", s=140, marker="s", edgecolor="white", zorder=6)
ax2.set_xlim(lo_x, hi_x); ax2.set_ylim(lo_y, hi_y)
ax2.set_title("Static Voronoi Pitch Control (after carry)\nGreen = attacking team's nearest-player control", fontsize=11, fontweight="bold", color="#e2e8f0")
ax2.set_xlabel("Attacking direction \u2192 (yards)", fontsize=9, color="#94a3b8")
ax2.legend(loc="upper left", facecolor="#0b132b", edgecolor="#1e293b", labelcolor="white", fontsize=7.5)
ax2.set_aspect("equal")

plt.tight_layout()
plt.savefig("spatial_example.png", dpi=200)
print("saved spatial_example.png")
