import numpy as np

def batched_rotated_box_intersections(boxes):
    """
    Args:
        boxes: np.ndarray of shape (N, 5), each row is (cx, cy, w, h, angle_deg)
    Returns:
        intersections: np.ndarray of shape (N, 2, 2)
        on_short_sides: np.ndarray of shape (N,)
    """
    N = boxes.shape[0]
    cx, cy, w, h, angles = boxes.T

    angles_rad = np.deg2rad(angles)
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    # Define local corners: (4, 2)
    local_corners = np.array([
        [-0.5, -0.5],
        [ 0.5, -0.5],
        [ 0.5,  0.5],
        [-0.5,  0.5]
    ], dtype=np.float32)  # shape (4, 2)

    # Broadcast and scale corners: (N, 4, 2)
    wh = np.stack([w, h], axis=1)[:, None, :]  # (N, 1, 2)
    scaled_corners = local_corners[None, :, :] * wh  # (N, 4, 2)

    # Rotation matrices: (N, 2, 2)
    R = np.stack([
        np.stack([cos_a, -sin_a], axis=1),
        np.stack([sin_a,  cos_a], axis=1)
    ], axis=1)  # (N, 2, 2)

    # Rotate corners: (N, 4, 2)
    rotated_corners = np.einsum('nij,nkj->nki', R, scaled_corners)
    rotated_corners += np.stack([cx, cy], axis=1)[:, None, :]

    # Define the ray from (0, 0) to (2*cx, 2*cy): (N, 2)
    ray_dirs = np.stack([cx, cy], axis=1) * 2  # (N, 2)
    ray_orig = np.zeros((N, 2), dtype=np.float32)  # (N, 2)

    # Build edges from corners: (N, 4, 2)
    edges_start = rotated_corners
    edges_end = np.roll(rotated_corners, shift=-1, axis=1)

    # Edge direction vectors: (N, 4, 2)
    edge_vecs = edges_end - edges_start

    # Repeat ray for each edge: (N, 4, 2)
    ray_vecs = ray_dirs[:, None, :]  # (N, 1, 2)
    ray_vecs = np.repeat(ray_vecs, 4, axis=1)

    ray_orig_rep = ray_orig[:, None, :]  # (N, 1, 2)
    ray_orig_rep = np.repeat(ray_orig_rep, 4, axis=1)

    diff = edges_start - ray_orig_rep  # (N, 4, 2)
    denom = np.cross(ray_vecs, edge_vecs)  # (N, 4)

    # Avoid divide by zero
    denom_safe = np.where(np.abs(denom) < 1e-8, 1e-8, denom)

    t = np.cross(diff, edge_vecs) / denom_safe  # (N, 4)
    u = np.cross(diff, ray_vecs) / denom_safe  # (N, 4)

    # Intersection mask
    mask = (u >= 0) & (u <= 1) & (np.abs(denom) >= 1e-8)

    # Compute intersections: p1 + t * dir
    inter_pts = ray_orig_rep + t[..., None] * ray_vecs  # (N, 4, 2)

    # Filter two best intersection points by dot product
    cxcy = np.stack([cx, cy], axis=1)  # (N, 2)
    dot_scores = np.einsum('nij,nj->ni', inter_pts, cxcy)  # (N, 4)
    dot_scores = np.where(mask, dot_scores, -np.inf)

    top2_idx = np.argsort(-dot_scores, axis=1)[:, :2]  # (N, 2)
    batch_indices = np.arange(N)[:, None]  # (N, 1)

    # Get the top 2 intersection points
    best_points = inter_pts[batch_indices, top2_idx]  # (N, 2, 2)

    # Get the corresponding edge vectors
    edge_lengths = np.linalg.norm(edge_vecs, axis=-1)  # (N, 4)
    short_lengths = np.min(edge_lengths, axis=1, keepdims=True)  # (N, 1)

    # Get corresponding edge indices
    best_edges = edge_lengths[batch_indices, top2_idx]  # (N, 2)
    is_short = np.all(np.isclose(best_edges, short_lengths, atol=1e-6), axis=1)  # (N,)

    # Check direction: dot product of the two vectors
    vectors = best_points[:, 1] - best_points[:, 0]
    dot_directions = np.einsum('ni,ni->n', vectors, cxcy)
    is_positive = dot_directions > 0

    on_short_sides = is_short & is_positive
    return best_points, on_short_sides