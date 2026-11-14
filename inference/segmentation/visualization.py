from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def save_loss_plot(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss by epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_test_samples(model, device, threshold, run_dir, test_pairs, image_size, num_samples=10):
    model.eval()
    samples_dir = run_dir / "test_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )

    for sample_count, (image_path, mask_path) in enumerate(test_pairs[:num_samples]):
        image_raw = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path)
        if "A" in mask_image.getbands():
            mask_raw = mask_image.getchannel("A")
        else:
            mask_raw = mask_image.convert("L")

        image_tensor = image_transform(image_raw).unsqueeze(0).to(device)
        mask_tensor = mask_transform(mask_raw)
        mask_tensor = (mask_tensor > 0.5).float().squeeze(0)

        with torch.no_grad():
            output = model(image_tensor)["out"]
            pred_tensor = (torch.sigmoid(output) > threshold).float().squeeze(0).squeeze(0).cpu()

        img_np = np.array(image_raw.resize((image_size, image_size), resample=Image.Resampling.BILINEAR))
        pred_np = (pred_tensor.numpy() * 255).astype(np.uint8)
        true_np = (mask_tensor.numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        axes[1].imshow(pred_np, cmap="gray")
        axes[1].set_title("Model prediction")
        axes[1].axis("off")

        axes[2].imshow(true_np, cmap="gray")
        axes[2].set_title("Ground truth")
        axes[2].axis("off")

        legend_text = f"Filename: {Path(image_path).name}\nLegend: Black = Background (0), White = Target (255)"
        fig.text(0.5, 0.02, legend_text, ha="center", fontsize=11)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        sample_path = samples_dir / f"sample_{sample_count:02d}.png"
        plt.savefig(sample_path, dpi=100, bbox_inches="tight")
        plt.close()
