import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from segmentation.model import build_segmentation_model
from pathlib import Path
from segmentation.raster.reader import *
from segmentation.raster.tiling import *
from segmentation.raster.preprocessing import *
from segmentation.raster.merging import *
from segmentation.raster.writer import *

def load_model(checkpoint_path, device = "cpu"):
    device = torch.device(device)

    # build model 
    model = build_segmentation_model(
        model_name="deeplabv3_resnet50",
        pretrained_backbone=True,
        num_classes=1,
    )

    # load weights for model
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def predict_image(model, path, device="cpu"):
    device = torch.device(device)

    raster_info = get_raster_info(path)
    width = raster_info['width']
    height = raster_info['height']
    merger = MaskMerger(width, height)
    windows = generate_windows(width, height)
    
    with torch.inference_mode():
        for window in windows:
            window_data = read_window(path, window)
            input_tensor = prepare_tile(window_data)
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)["out"]
            prediction = torch.sigmoid(output)
            prediction = (prediction.squeeze(0).squeeze(0).cpu().numpy())
            merger.add(prediction, window)
    
    final_mask = merger.result(0.5)
    final_mask = (final_mask * 255).astype(np.uint8)

    return final_mask

if __name__ == "__main__":

    model = load_model("checkpoints/best_model.pt")

    mask = predict_image(model, f"test_rasters/google_satellite_1m_1.tif")
    write_mask(f"predictions/mask_new.tif", mask, f"test_rasters/google_satellite_1m_1.tif")

    # small raster images
    # images_count = 10
    # for x in range(images_count):
    #     mask = predict_image(model, f"test_rasters/google_satellite_1m_{x + 1}.tif")
    #     write_mask(f"predictions/mask{x + 1}.tif", mask, f"test_rasters/google_satellite_1m_{x + 1}.tif")
    
    # big raster images
    # for y in range(images_count):
    #      mask = predict_image(model, f"test_rasters/google_satellite_1m_{y + 11}.tif")
    #      write_mask(f"predictions/mask{y + 11}.tif", mask, f"test_rasters/google_satellite_1m_{y + 11}.tif")