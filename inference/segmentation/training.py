import torch
import torch.nn as nn
from tqdm import tqdm


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, eps=1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss(loss_name, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
    name = loss_name.lower().strip()

    if name == "bce":
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=pos_weight)

    raise ValueError("Neznamy loss_name. Pouzi jednu moznost: bce, dice, bce_dice")


def compute_metrics(logits, targets, threshold):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    dice_denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    iou = ((intersection + 1e-7) / (union + 1e-7)).mean().item()
    dice = ((2 * intersection + 1e-7) / (dice_denom + 1e-7)).mean().item()
    pixel_acc = (preds == targets).float().mean().item()

    return iou, dice, pixel_acc


def run_epoch(model, loader, criterion, optimizer, device, threshold, train_mode, use_amp=False, amp_dtype=torch.float16, scaler=None):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    total_batches = 0

    progress = tqdm(loader, desc="train" if train_mode else "eval", leave=False)

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            amp_active = use_amp and getattr(device, "type", "cpu").startswith("cuda") and torch.cuda.is_available()
            if amp_active:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(images)["out"]
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            if train_mode:
                if amp_active and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        iou, dice, acc = compute_metrics(outputs.detach(), masks, threshold)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        total_acc += acc
        total_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}", dice=f"{dice:.4f}")

    avg_loss = total_loss / max(total_batches, 1)
    avg_iou = total_iou / max(total_batches, 1)
    avg_dice = total_dice / max(total_batches, 1)
    avg_acc = total_acc / max(total_batches, 1)

    return avg_loss, avg_iou, avg_dice, avg_acc
