import numpy as np


class MaskMerger:
    """Combines tile predictions into a complete raster"""

    def __init__(self, width, height):
        self.sum_predictions = np.zeros((height, width), dtype=np.float32)
        self.count_predictions = np.zeros((height, width), dtype=np.uint16)

    def add(self, prediction, window):
        """Adds a prediction for the corresponding window"""

        x = int(window.col_off)
        y = int(window.row_off)
        w = int(window.width)
        h = int(window.height)

        prediction = np.asarray(prediction, dtype = np.float32)

        if prediction.shape != (h, w):
            raise ValueError("The prediction size does not match the window size")

        self.sum_predictions[y:y+h, x:x+w] += prediction
        self.count_predictions[y:y+h, x:x+w] += 1

    def result(self, threshold=0.5):
        """Returns the final binary mask"""

        valid = self.count_predictions > 0

        averaged = np.zeros_like(self.sum_predictions)
        averaged[valid] = (self.sum_predictions[valid] / self.count_predictions[valid])

        return (averaged >= threshold).astype(np.uint8)