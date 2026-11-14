import numpy as np
import torch
import torch.nn.functional as F


def prepare_tile(tile, expected_size=None):

    if tile.ndim != 3:
        raise ValueError("Expected shape (C, H, W)")

    if tile.shape[0] < 3:
        raise ValueError("RGB image requires at least 3 channels")

    # Select RGB channels
    tile = tile[:3].astype(np.float32, copy=True)

    # Handle NaN and Inf
    valid_pixels = np.isfinite(tile).all(axis=0)

    if not valid_pixels.any():
        raise ValueError("Tile contains no valid pixels")

    tile[:, ~valid_pixels] = 0.0

    # Normalize to [0, 1]
    if tile.max() > 1.0:
        tile /= 255.0

    # Convert to Tensor and add batch dimension
    tensor = torch.from_numpy(tile).unsqueeze(0)

    # Resize if necessary
    if expected_size is not None:
        tensor = F.interpolate(tensor, size=expected_size, mode="bilinear", align_corners=False)

    return tensor