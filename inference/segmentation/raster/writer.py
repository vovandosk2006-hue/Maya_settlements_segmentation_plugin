from pathlib import Path
import numpy as np
import rasterio


def write_mask(output_path, mask, source_path):
    """Saves the binary mask with the georeferencing of the original raster"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_path) as src:
        if mask.shape != (src.height, src.width):
            raise ValueError("The mask size must match the dimensions of the original raster")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            width=src.width,
            height=src.height,
            nodata=None,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)

    return output_path