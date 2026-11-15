import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

def get_raster_info(path):
    """Returns dict with raster info"""
    with rasterio.open(path) as raster:

        # check pixel resolution
        x, y = raster.res
        # expected_res = 1
        # tolerance = 0.75
        # if (abs(x - expected_res) > tolerance or abs(y - expected_res) > tolerance) or x != y:
        #     raise ValueError(f"Wrong pixel resolution: {x:.4f} x {y:.4f} m/pixel. Expected approximately 1 x 1 m/pixel (tolerance ±{tolerance:.2f} m).")

        return {
            "path": str(path),
            "width": raster.width,
            "height": raster.height,
            "band_count": raster.count,
            "crs": raster.crs.to_string() if raster.crs else None,
            "resolution": raster.res,
            "bounds": (
                raster.bounds.left,
                raster.bounds.bottom,
                raster.bounds.right,
                raster.bounds.top,
            ),
            #"bounds": raster.bounds
            "colorinterp": raster.colorinterp
        }


def read_window(path, window, bands = [1, 2, 3], out_dtype= "float32"):
    """Returns np-array with pixel data from the window"""
    try:
        with rasterio.open(path) as src:
            if bands is None:
                indexes = list(range(1, src.count + 1))
            else:
                indexes = list(bands)

            if not indexes:
                raise ValueError("No raster bands selected")

            if any(index < 1 or index > src.count for index in indexes):
                raise ValueError(
                    f"Band indexes must be between 1 and {src.count}."
                )

            if (
                window.col_off < 0
                or window.row_off < 0
                or window.width <= 0
                or window.height <= 0
                or window.col_off + window.width > src.width
                or window.row_off + window.height > src.height
            ):
                raise ValueError(
                    "The requested window extends beyond the raster boundaries."
                )
            
            # read pixels data from window
            data = src.read(
                indexes=indexes,
                window=window,
                out_dtype=out_dtype,
                masked=True,
            )

            # Return np-array and replace NoData pixels with NaN
            return data.filled(np.nan)

    except RasterioIOError as exc:
        raise RuntimeError(f"Failed to read the raster window: {exc}") from exc