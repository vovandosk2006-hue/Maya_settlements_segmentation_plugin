from pathlib import Path
import random
import shutil
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, pairs, image_size, resize_mode="square", target_size=None):
        self.pairs = pairs
        mode = str(resize_mode).lower().strip()

        if mode == "native":
            image_ops = [transforms.ToTensor()]
            mask_ops = [transforms.ToTensor()]
        elif mode == "custom":
            if target_size is None:
                raise ValueError("Pre resize_mode='custom' treba zadat target_size=(height, width)")
            image_ops = [
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
            mask_ops = [
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        elif mode == "square":
            image_ops = [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
            mask_ops = [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        else:
            raise ValueError("Neznamy resize_mode. Pouzi: native, square, custom")

        self.image_transform = transforms.Compose(image_ops)
        self.mask_transform = transforms.Compose(mask_ops)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")

        mask_image = Image.open(mask_path)
        # Slovak: V tomto projekte alpha kanal ignorujeme a masku vzdy berieme z grayscale (L).
        mask = mask_image.convert("L")

        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


def get_mask_name(image_name, suffix, extension):
    stem = Path(image_name).stem
    return f"{stem}{suffix}{extension}"


def extract_location_from_image(image_name):
    # Slovak: Extrahuje ID lokality a casový rez z mena obrazka 'site{lokacia}_{cas}.jpg'
    match = re.match(r'site(\d+)_(\d+)\.(jpg|png)', image_name)
    if match:
        location = match.group(1)
        timeframe = match.group(2)
        return location, timeframe
    return None, None


def find_mask_by_location(location, masks_dir):
    # Slovak: Najde masku pre danu lokalitu.
    # Podpora formatov nazvov masiek: 'site_<loc>.png' alebo 'site<loc>.png'
    masks_dir = Path(masks_dir)
    loc_str1 = f"site_{location}.png"
    loc_str2 = f"site{location}.png"
    for mask_file in masks_dir.iterdir():
        name = mask_file.name.lower()
        if mask_file.suffix.lower() != '.png':
            continue
        # format: site_1.png alebo site1.png
        if name == loc_str1 or name == loc_str2 or mask_file.stem.lower() == f"site_{location}" or mask_file.stem.lower() == f"site{location}":
            return mask_file
    return None


def build_pairs_spatial_temporal(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_paths = sorted([path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in ['.jpg', '.png']])
    
    pairs = []
    missing_masks = []
    
    location_images = defaultdict(list)
    
    for image_path in image_paths:
        location, timeframe = extract_location_from_image(image_path.name)
        if location is not None:
            location_images[location].append(image_path)
    
    for location, images_for_location in sorted(location_images.items()):
        mask_path = find_mask_by_location(location, masks_dir)
        if mask_path and mask_path.exists():
            for image_path in images_for_location:
                pairs.append((image_path, mask_path))
        else:
            for image_path in images_for_location:
                missing_masks.append(image_path.name)
    
    return pairs, missing_masks


def build_pairs(images_dir, masks_dir, suffix):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_paths = sorted([path for path in images_dir.iterdir() if path.is_file()])
    
    try:
        first_image = image_paths[0] if image_paths else None
        if first_image and re.match(r'site\d+_\d+\.(jpg|png)', first_image.name):
            return build_pairs_spatial_temporal(images_dir, masks_dir)
    except:
        pass
    
    pairs = []
    missing_masks = []

    for image_path in image_paths:
        mask_name = get_mask_name(image_path.name, suffix, ".png")
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            missing_masks.append(image_path.name)

    return pairs, missing_masks


def split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Pomer split musi byt 1.0")

    rng = random.Random(seed)
    
    try:
        location_groups = group_pairs_by_location(pairs)
        if len(location_groups) > 1:
            return split_pairs_by_location(location_groups, train_ratio, val_ratio, test_ratio, seed)
    except:
        pass
    
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    total_count = len(shuffled)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    train_pairs = shuffled[:train_end]
    val_pairs = shuffled[train_end:val_end]
    test_pairs = shuffled[val_end:]

    return train_pairs, val_pairs, test_pairs


def group_pairs_by_location(pairs):
    location_groups = defaultdict(list)
    for image_path, mask_path in pairs:
        location, timeframe = extract_location_from_image(image_path.name)
        if location is not None:
            location_groups[location].append((image_path, mask_path))
    return location_groups


def split_pairs_by_location(location_groups, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Pomer split musi byt 1.0")
    
    rng = random.Random(seed)
    
    locations = sorted(location_groups.keys())
    shuffled_locations = locations[:]
    rng.shuffle(shuffled_locations)
    
    num_locations = len(shuffled_locations)
    num_train = int(num_locations * train_ratio)
    num_val = int(num_locations * val_ratio)
    
    train_locations = set(shuffled_locations[:num_train])
    val_locations = set(shuffled_locations[num_train:num_train + num_val])
    test_locations = set(shuffled_locations[num_train + num_val:])
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for location, pairs_for_location in location_groups.items():
        if location in train_locations:
            train_pairs.extend(pairs_for_location)
        elif location in val_locations:
            val_pairs.extend(pairs_for_location)
        else:
            test_pairs.extend(pairs_for_location)
    
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)
    
    # Slovak: vypis rozdelenia po lokalitach
    print(f"Priestorovo-casovy split:")
    print(f"  Train: {len(train_locations)} lokalit, {len(train_pairs)} obrazkov")
    print(f"  Val:   {len(val_locations)} lokalit, {len(val_pairs)} obrazkov")
    print(f"  Test:  {len(test_locations)} lokalit, {len(test_pairs)} obrazkov")
    
    return train_pairs, val_pairs, test_pairs


def split_pairs_stratified_by_location(location_groups, train_ratio, val_ratio, test_ratio, seed, num_bins=8):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Pomer split musi byt 1.0")
    
    rng = random.Random(seed)
    
    strata = {}
    for location, pairs_for_location in location_groups.items():
        if pairs_for_location:
            _, mask_path = pairs_for_location[0]
            fg_ratio = compute_mask_foreground_ratio(mask_path)
            bin_index = min(int(fg_ratio * num_bins), num_bins - 1)
            strata.setdefault(bin_index, []).append(location)
    
    train_locations = set()
    val_locations = set()
    test_locations = set()
    
    for bin_locations in strata.values():
        shuffled = bin_locations[:]
        rng.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
        
        train_locations.update(shuffled[:n_train])
        val_locations.update(shuffled[n_train:n_train + n_val])
        test_locations.update(shuffled[n_train + n_val:])
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for location, pairs_for_location in location_groups.items():
        if location in train_locations:
            train_pairs.extend(pairs_for_location)
        elif location in val_locations:
            val_pairs.extend(pairs_for_location)
        elif location in test_locations:
            test_pairs.extend(pairs_for_location)
    
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)
    
    # Slovak: vypis stratifikovaneho priest.-cas. rozdelenia
    print(f"Stratifikovany priestorovo-casovy split ({num_bins} binov):")
    print(f"  Train: {len(train_locations)} lokalit, {len(train_pairs)} obrazkov")
    print(f"  Val:   {len(val_locations)} lokalit, {len(val_pairs)} obrazkov")
    print(f"  Test:  {len(test_locations)} lokalit, {len(test_pairs)} obrazkov")
    
    return train_pairs, val_pairs, test_pairs


def compute_mask_foreground_ratio(mask_path):
    mask_image = Image.open(mask_path)
    mask = mask_image.convert("L")

    mask_np = np.array(mask, dtype=np.uint8)
    return float((mask_np > 0).mean())


def split_pairs_stratified(pairs, train_ratio, val_ratio, test_ratio, seed, num_bins=8):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Pomer split musi byt 1.0")

    if num_bins < 2:
        raise ValueError("Pocet binov pre stratifikaciu musi byt aspon 2")

    rng = random.Random(seed)

    try:
        location_groups = group_pairs_by_location(pairs)
        if len(location_groups) > 1:
            return split_pairs_stratified_by_location(location_groups, train_ratio, val_ratio, test_ratio, seed, num_bins)
    except:
        pass

    strata = {}
    for pair in pairs:
        _, mask_path = pair
        fg_ratio = compute_mask_foreground_ratio(mask_path)
        bin_index = min(int(fg_ratio * num_bins), num_bins - 1)
        strata.setdefault(bin_index, []).append(pair)

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for group in strata.values():
        shuffled = group[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1

        train_pairs.extend(shuffled[:n_train])
        val_pairs.extend(shuffled[n_train:n_train + n_val])
        test_pairs.extend(shuffled[n_train + n_val:])

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_pairs)

    return train_pairs, val_pairs, test_pairs


def prepare_split_dirs(run_dir):
    split_root = run_dir / "data"
    for split_name in ["train", "val", "test"]:
        (split_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (split_root / split_name / "masks").mkdir(parents=True, exist_ok=True)
    return split_root


def copy_split_data(split_root, split_name, pairs):
    image_dst = split_root / split_name / "images"
    mask_dst = split_root / split_name / "masks"

    for image_path, mask_path in pairs:
        shutil.copy2(image_path, image_dst / image_path.name)
        shutil.copy2(mask_path, mask_dst / mask_path.name)


def create_loaders(
    train_pairs,
    val_pairs,
    test_pairs,
    image_size,
    batch_size,
    num_workers,
    pin_memory,
    resize_mode="square",
    target_size=None,
):
    train_dataset = SegmentationDataset(train_pairs, image_size, resize_mode=resize_mode, target_size=target_size)
    val_dataset = SegmentationDataset(val_pairs, image_size, resize_mode=resize_mode, target_size=target_size)
    test_dataset = SegmentationDataset(test_pairs, image_size, resize_mode=resize_mode, target_size=target_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
