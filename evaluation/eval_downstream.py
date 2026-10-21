import argparse
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import ToTensor

if __package__:
    from .models import MLP, CNN
    from .classifier_utils import seed_everthing, set_logger, train_classifier, test_classifier
else:
    from models import MLP, CNN
    from classifier_utils import seed_everthing, set_logger, train_classifier, test_classifier


def main(args):
    # Logging configuration
    target_dir = Path(args.output_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    time = datetime.now().strftime("%m%d-%H:%M:%S")
    set_logger(f=target_dir / f"downstream_{time}.log")

    # Print configuration
    for key, val in vars(args).items():
        logging.info(f"{key}: {val}")
    seed_everthing(args.seed)

    # Load Data
    with np.load(args.synthesis_path, allow_pickle=False) as synthesis:
        images = synthesis["data"]
        labels = synthesis["labels"]

    # Data preparation
    assert type(images) == np.ndarray
    assert len(images.shape) == 4

    if images.dtype == np.uint8:
        images = images.astype(np.float32)
        images /= 255.
    if images.shape[3] == 3 or images.shape[3] == 1:
        images = np.transpose(images, [0, 3, 1, 2])

    if images.shape[1:] != (1, 28, 28) or len(images) < 10:
        raise ValueError("MNIST/FMNIST evaluation needs at least 10 images of shape (1, 28, 28)")
    if labels.shape != (len(images),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected one integer label per synthetic image")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Synthetic pixels must be finite and in [0, 1]")
    if labels.min() < 0 or labels.max() >= 10:
        raise ValueError("Synthetic labels must be in [0, 9]")

    gen_dataset = TensorDataset(
        torch.from_numpy(images.astype(np.float32, copy=False)),
        torch.from_numpy(labels.astype(np.int64, copy=False)),
    )

    # Load original data
    if args.dataset == "mnist":
        from torchvision.datasets import MNIST
        real_dataset = MNIST(root=args.data_root, train=False,
                             download=True, transform=ToTensor(),)
        img_ch = 1
        num_classes = 10
    elif args.dataset == "fmnist":
        from torchvision.datasets import FashionMNIST
        real_dataset = FashionMNIST(root=args.data_root, train=False,
                                    download=True, transform=ToTensor(),)
        img_ch = 1
        num_classes = 10
    else:
        raise NotImplementedError

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset not in ["mnist", "fmnist"]:
        raise NotImplementedError
    
    # Prepare train, val, test dataset
    val_size = max(1, int(np.ceil(len(gen_dataset) * 0.1)))
    train_dataset, val_dataset = random_split(
        gen_dataset, [len(gen_dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    models = {
        "MLP": MLP,
        "CNN": CNN,
    }

    metrics = {
        "dataset": args.dataset, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "synthesis_path": str(Path(args.synthesis_path).resolve()),
        "num_synthetic": len(gen_dataset), "num_test": len(real_dataset),
        "classifiers": {},
    }
    # Training classifier
    for name, model_class in models.items():
        # Match initialization and classifier training randomness across methods.
        seed_everthing(args.seed)
        model = model_class(num_classes=num_classes)
        logging.info(f"Training {name} classifier...")
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        val_acc = train_classifier(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            epochs=args.epochs,
        )
        acc = test_classifier(model, test_dataloader)
        logging.info(f"{name} classifier Accuracy: {acc:.4f}")
        metrics["classifiers"][name] = {
            "test_accuracy": acc, "validation_accuracy": val_acc,
        }

    # Only a complete evaluation creates the file used for experiment resume.
    metrics_path = target_dir / "metrics.json"
    temporary_path = metrics_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metrics, indent=2) + "\n")
    temporary_path.replace(metrics_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthesis_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="_results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument("--data_root", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "_data"))
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    main(args)
