import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import unary_union

Y_WEIGHT = 0.25  # anisotropic scaling: 1 yard of lateral (y) separation counts as much
                 # less "different line" evidence than 1 yard of depth (x) separation

def cluster_lines_2d(points_xy, eps_x_equivalent):
    """DBSCAN on (x, y*Y_WEIGHT) so clustering is primarily depth-driven (tactically
    correct: a 'line' spans the pitch width but is grouped by how far downfield it is),
    while still using real 2-D positions (unlike the old x-only 1-D method)."""
    if len(points_xy) == 0:
        return []
    pts = np.array(points_xy)
    transformed = np.column_stack([pts[:, 0], pts[:, 1] * Y_WEIGHT])
    db = DBSCAN(eps=eps_x_equivalent, min_samples=1).fit(transformed)
    clusters = []
    for label in set(db.labels_):
        member_idx = np.where(db.labels_ == label)[0]
        clusters.append(pts[member_idx])
    return clusters

def cluster_hull(cluster_pts, buffer_yd=1.5):
    """Real 2-D convex hull for >=3 non-collinear points; a small buffered
    point/line for degenerate 1-2 point clusters (can't form a hull)."""
    if len(cluster_pts) >= 3:
        mp = MultiPoint([tuple(p) for p in cluster_pts])
        hull = mp.convex_hull
        if hull.geom_type == "Polygon":
            return hull
        return hull.buffer(buffer_yd)
    elif len(cluster_pts) == 2:
        return LineString([tuple(cluster_pts[0]), tuple(cluster_pts[1])]).buffer(buffer_yd)
    else:
        return Point(tuple(cluster_pts[0])).buffer(buffer_yd)

def carry_path_crosses_hulls(start, end, opp_points, eps):
    """Full 2-D pipeline for one carry: cluster real opponent positions into lines,
    build each line's convex hull, and count how many hulls the carry's actual
    2-D path (not just its x-range) geometrically intersects. Forward-progress
    only: 'breaking a line' means advancing through it, not grazing a defender's
    zone while moving sideways or backward (matches the cited tactical definition
    and the direction restriction used in the earlier 1-D method)."""
    if len(opp_points) == 0 or end[0] <= start[0]:
        return 0, 0
    clusters = cluster_lines_2d(opp_points, eps)
    path = LineString([tuple(start), tuple(end)])
    crossed = 0
    for c in clusters:
        hull = cluster_hull(c)
        if path.intersects(hull):
            crossed += 1
    return crossed, len(clusters)

def voronoi_control_gained(start, end, teammate_pts, opponent_pts, carrier_start, grid_step=2.0, pad=6.0):
    """Static, position-only Voronoi pitch control (no velocity data available from
    a single freeze frame, so this approximates -- not replicates -- dynamic models
    like Spearman et al.). Grid restricted to a corridor around the carry's own path
    so the metric reflects local space gained, not the whole pitch."""
    lo_x, hi_x = min(start[0], end[0]) - pad, max(start[0], end[0]) + pad
    lo_y, hi_y = min(start[1], end[1]) - pad, max(start[1], end[1]) + pad
    lo_x, hi_x = max(0, lo_x), min(120, hi_x)
    lo_y, hi_y = max(0, lo_y), min(80, hi_y)
    xs = np.arange(lo_x, hi_x, grid_step)
    ys = np.arange(lo_y, hi_y, grid_step)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    all_before = np.array(teammate_pts + opponent_pts + [carrier_start])
    n_team_before = len(teammate_pts) + 1  # + carrier
    team_before_idx = set(range(n_team_before))

    all_after = np.array(teammate_pts + opponent_pts + [end])  # carrier moved to end location
    team_after_idx = team_before_idx  # same index layout

    tree_before = cKDTree(all_before)
    _, nn_before = tree_before.query(grid_pts)
    control_before = np.mean([1 if i in team_before_idx else 0 for i in nn_before])

    tree_after = cKDTree(all_after)
    _, nn_after = tree_after.query(grid_pts)
    control_after = np.mean([1 if i in team_after_idx else 0 for i in nn_after])

    return float(control_after - control_before)
