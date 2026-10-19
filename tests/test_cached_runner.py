"""CPU smoke of the actual training runner with Phase-II dependencies stubbed."""
import importlib
import sys
import types

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader


class CountingDataset(Dataset):
    def __init__(self):
        self.images = torch.linspace(-1, 1, 32).reshape(8, 1, 2, 2)
        self.reads = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        self.reads += 1
        return self.images[i], 99


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.labels = []

    def forward(self, x, t, y):
        self.labels.append(y.detach().cpu().clone())
        return self.bias.expand_as(x)


def test_runner_uses_cache_only_across_epochs(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "src.opacus", types.ModuleType("src.opacus"))
    opacus = types.ModuleType("opacus")
    class PrivacyEngine:
        def make_private(self, module, optimizer, data_loader, **kw):
            return module, optimizer, data_loader
    opacus.PrivacyEngine = PrivacyEngine
    monkeypatch.setitem(sys.modules, "opacus", opacus)
    opacus_data = types.ModuleType("opacus.data_loader")
    class DPDataLoader:
        @staticmethod
        def from_data_loader(loader):
            raise AssertionError("cached Phase I must not sample the raw loader")
    opacus_data.DPDataLoader = DPDataLoader
    monkeypatch.setitem(sys.modules, "opacus.data_loader", opacus_data)
    opacus_utils = types.ModuleType("opacus.utils")
    opacus_utils.__path__ = []
    monkeypatch.setitem(sys.modules, "opacus.utils", opacus_utils)
    opacus_memory = types.ModuleType("opacus.utils.batch_memory_manager")
    class BatchMemoryManager:
        def __init__(self, **kw):
            raise AssertionError("Phase II not used in this test")
    opacus_memory.BatchMemoryManager = BatchMemoryManager
    monkeypatch.setitem(sys.modules, "opacus.utils.batch_memory_manager", opacus_memory)
    trainers = types.ModuleType("src.trainers")
    trainers.DPPromiseTrainer = lambda cfg: torch.nn.Identity()
    trainers.VanillaDDPMTrainer = lambda cfg: torch.nn.Identity()
    monkeypatch.setitem(sys.modules, "src.trainers", trainers)
    ema = types.ModuleType("src.ema")
    class EMA:
        def __init__(self, parameters, decay): pass
        def update(self, parameters): pass
        def copy_to(self, parameters): pass
    ema.ExponentialMovingAverage = EMA
    monkeypatch.setitem(sys.modules, "src.ema", ema)
    utils = types.ModuleType("src.utils")
    utils.sample_example_image = lambda model, config: torch.zeros(1,1,2,2)
    utils.sample_images = lambda model, config, num_samples: (
        torch.zeros(2,1,2,2), torch.zeros(2, dtype=torch.long))
    monkeypatch.setitem(sys.modules, "src.utils", utils)
    sys.modules.pop("src.runners", None)
    runner = importlib.import_module("src.runners")
    monkeypatch.setattr(runner, "tqdm", lambda x: x)

    cfg = OmegaConf.create({
        "diffusion": {"timesteps": 100, "beta_start": .0001, "beta_end": .02},
        "dp": {"S": 80, "max_grad_norm": 1., "sigma": 3.},
        "data": {"class_condition": True},
        "train": {"phase1_objective": "cached_intermediate",
                  "phase1_cache_step": 65, "phase1_gap_steps": 8,
                  "epochs1": 2, "epochs2": 0, "lr1": .01, "lr2": .01,
                  "ema_rate": .999, "num_noise_sample": 1},
    })
    ds = CountingDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = TinyModel()
    start_bias = model.bias.detach().clone()
    runner.train_dp_promise(model, loader, loader, config=cfg, workdir=tmp_path)
    assert ds.reads == len(ds)
    assert len(model.labels) == 4
    assert all(torch.equal(y, torch.zeros_like(y)) for y in model.labels)
    assert not torch.equal(model.bias.detach(), start_bias)
    assert (tmp_path / "evaluation" / "synthesis.npz").exists()
