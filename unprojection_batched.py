import numpy as np

def depth_to_world_xyz_numpy(depth, intrinsics, extrinsics):
    B, H, W = depth.shape
    # Create pixel grid (u, v, 1)
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W)
    uv1 = np.stack([u, v, np.ones_like(u)], axis=0)  # (3, H, W)
    uv1 = np.broadcast_to(uv1, (B, 3, H, W))         # (B, 3, H, W)

    # Compute inverse intrinsics
    K_inv = np.linalg.inv(intrinsics)  # (B, 3, 3)

    # Apply K^-1 to pixel coordinates
    cam_coords = np.einsum('bij,bjhw->bihw', K_inv, uv1)  # (B, 3, H, W)

    # Scale by depth
    xyz_cam = cam_coords * depth[:, np.newaxis, :, :]  # (B, 3, H, W)

    # Convert to homogeneous coords: (B, 4, H, W)
    ones = np.ones_like(depth[:, np.newaxis, :, :])  # (B, 1, H, W)
    xyz_hom = np.concatenate([xyz_cam, ones], axis=1)  # (B, 4, H, W)

    # Apply extrinsics (camera to world): (B, 3, 4) x (B, 4, H, W)
    xyz_world = np.einsum('bij,bjhw->bihw', extrinsics[:, :3, :], xyz_hom)  # (B, 3, H, W)

    return xyz_world
  