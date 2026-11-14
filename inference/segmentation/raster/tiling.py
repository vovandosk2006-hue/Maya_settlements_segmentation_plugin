from rasterio.windows import Window


def generate_windows(width, height, tile_size = 512, overlap = 64):
    """Generates windows for processing the entire raster"""

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 и < tile_size") 

    step = tile_size - overlap

    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            window_width = min(tile_size, width - col_off)
            window_height = min(tile_size, height - row_off)

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=window_width,
                height=window_height,
            )