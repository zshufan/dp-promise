"""PyTorch-only helpers for synthetic-data classifier evaluation."""
import copy
import logging
import random
import sys

import numpy as np
import torch
from tqdm import tqdm


def set_logger(f, level="INFO"):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
    for handler in (logging.FileHandler(f), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


def seed_everthing(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_classifier(model, train_dataloader, val_dataloader, optimizer, epochs=50):
    if epochs < 1:
        raise ValueError("Classifier epochs must be positive")
    loss_fn = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    best_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(X), y).backward()
            optimizer.step()
        acc = test_classifier(model, val_dataloader)
        logging.info("Epoch %d: synthetic validation accuracy %.4f", epoch + 1, acc)
        if acc > best_acc:
            best_acc = acc
            # state_dict().copy() aliases live tensors and loses the best epoch.
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_acc


@torch.no_grad()
def test_classifier(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    total = correct = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        correct += (model(X).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return correct / total
