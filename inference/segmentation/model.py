import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50,
    fcn_resnet101,
    lraspp_mobilenet_v3_large,
)

# Optional: segmentation-models-pytorch (smp)
try:
    import segmentation_models_pytorch as smp  # type: ignore
except Exception:
    smp = None


class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_channels=32):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = _DoubleConv(in_channels, base_channels)
        self.enc2 = _DoubleConv(base_channels, base_channels * 2)
        self.enc3 = _DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = _DoubleConv(base_channels * 4, base_channels * 8)

        self.bottleneck = _DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    @staticmethod
    def _match_size(src, ref):
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return torch.nn.functional.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        out = self.head(d1)
        return {"out": out}


class SMPUNetWrapper(nn.Module):
    """Wrapper to make smp.Unet return {"out": tensor} like torchvision models."""
    def __init__(self, smp_model):
        super().__init__()
        self.model = smp_model

    def forward(self, x):
        out = self.model(x)
        return {"out": out}


MODEL_BUILDERS = {
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101,
    "fcn_resnet50": fcn_resnet50,
    "fcn_resnet101": fcn_resnet101,
    "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large,
    "unet": None,
    "unet_smp": None,
}


def _replace_classifier_heads(model, model_name, num_classes):
    if model_name == "lraspp_mobilenet_v3_large":
        low_in = model.classifier.low_classifier.in_channels
        high_in = model.classifier.high_classifier.in_channels
        model.classifier.low_classifier = nn.Conv2d(low_in, num_classes, kernel_size=1)
        model.classifier.high_classifier = nn.Conv2d(high_in, num_classes, kernel_size=1)
        return model

    cls_in = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(cls_in, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, num_classes, kernel_size=1)
    return model


def build_segmentation_model(model_name, pretrained_backbone, num_classes):
    if model_name not in MODEL_BUILDERS:
        supported = ", ".join(sorted(MODEL_BUILDERS.keys()))
        raise ValueError(f"Nepodporovany model '{model_name}'. Podporovane modely: {supported}")

    if model_name == "unet":
        return UNet(in_channels=3, num_classes=num_classes, base_channels=32)

    if model_name == "unet_smp":
        if smp is None:
            raise ImportError(
                "segmentation-models-pytorch is not installed. Install it with: `pip install segmentation-models-pytorch`"
            )
        # default encoder and weights - use resnet34 imagenet as a sensible default
        encoder_name = "resnet34"
        encoder_weights = "imagenet" if pretrained_backbone else None
        # activation=None -> return raw logits (we use BCEWithLogitsLoss / apply sigmoid manually)
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        return SMPUNetWrapper(model)

    builder = MODEL_BUILDERS[model_name]
    weights_backbone = "DEFAULT" if pretrained_backbone else None
    model = builder(weights=None, weights_backbone=weights_backbone)
    return _replace_classifier_heads(model, model_name, num_classes)
