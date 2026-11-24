"""Datasets. CIFAR-10 is the dataset used in the paper's controlled-noise experiments.

`digits` (sklearn, 8x8 images, offline) is only a smoke-test dataset.
"""
from __future__ import annotations

import os

import numpy as np


def load_digits_dataset():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    X, y = d.data.astype(np.float32), d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    return {"X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte, "m": 10, "images_train": None}


def load_cifar10_images(root: str):
    """Return uint8 images (N,32,32,3) and labels for train and test via torchvision."""
    from torchvision.datasets import CIFAR10

    tr = CIFAR10(root=root, train=True, download=True)
    te = CIFAR10(root=root, train=False, download=True)
    return (np.asarray(tr.data), np.asarray(tr.targets, dtype=np.int64),
            np.asarray(te.data), np.asarray(te.targets, dtype=np.int64))


def extract_resnet_features(images: np.ndarray, img_size: int = 64, batch_size: int = 256,
                            device: str | None = None) -> np.ndarray:
    """512-d features from an ImageNet-pretrained ResNet-18 (frozen). Used with the logreg backend."""
    import torch
    import torch.nn.functional as F
    import torchvision

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for s in range(0, len(images), batch_size):
            xb = torch.from_numpy(images[s:s + batch_size]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            xb = F.interpolate(xb, size=(img_size, img_size), mode="bilinear", align_corners=False)
            feats.append(net((xb - mean) / std).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def load_cifar10_dataset(root: str, backend: str, img_size: int = 64):
    """Returns the dict used by run_experiments.py. Features are cached next to the data."""
    xtr, ytr, xte, yte = load_cifar10_images(root)
    out = {"y_train": ytr, "y_test": yte, "m": 10, "images_train": xtr}
    if backend != "logreg":  # the CNN backend works on raw images, no features needed
        return out
    cache = os.path.join(root, f"cifar10_resnet18_{img_size}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        out["X_train"], out["X_test"] = z["tr"], z["te"]
    else:
        print("extracting pretrained ResNet-18 features (one-off) ...")
        out["X_train"] = extract_resnet_features(xtr, img_size)
        out["X_test"] = extract_resnet_features(xte, img_size)
        np.savez(cache, tr=out["X_train"], te=out["X_test"])
    return out
