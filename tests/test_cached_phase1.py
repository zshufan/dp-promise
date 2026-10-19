import math

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from src.cached_phase1 import (
    CachedIntermediateTrainer, cache_schedule, make_private_cache_loader,
    vp_alpha_bar,
)


def cfg(**train):
    return OmegaConf.create({
        "diffusion": {"timesteps": 100, "beta_start": 0.0001, "beta_end": 0.02},
        "dp": {"S": 80},
        "train": {"phase1_gap_steps": 8, "phase1_cache_step": 65, **train},
    })


class CountingDataset(Dataset):
    def __init__(self, n=7):
        self.images = torch.linspace(-1, 1, steps=n * 4).reshape(n, 1, 2, 2)
        self.reads = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        self.reads += 1
        return self.images[i], 100 + i


def test_cache_is_materialized_once_and_never_exposes_labels():
    dataset = CountingDataset()
    loader = DataLoader(dataset, batch_size=3, shuffle=True)
    cache = make_private_cache_loader(
        loader, vp_alpha_bar(cfg()), 65, generator=torch.Generator().manual_seed(456)
    )
    assert dataset.reads == len(dataset)
    assert len(cache.dataset.tensors) == 1
    initial = cache.dataset.tensors[0].clone()
    for _ in range(4):
        assert sum(len(batch[0]) for batch in cache) == len(dataset)
    assert dataset.reads == len(dataset)
    assert torch.equal(cache.dataset.tensors[0], initial)


def test_cache_gaussian_formula():
    ds = CountingDataset(2)
    loader = DataLoader(ds, batch_size=2)
    cached = make_private_cache_loader(
        loader, vp_alpha_bar(cfg()), 65, generator=torch.Generator().manual_seed(123)
    )
    z = torch.randn(ds.images.shape, generator=torch.Generator().manual_seed(123))
    alpha_c = float(vp_alpha_bar(cfg())[65])
    assert torch.allclose(
        cached.dataset.tensors[0],
        math.sqrt(alpha_c) * ds.images + math.sqrt(1-alpha_c) * z
    )


def test_two_stage_markov_kernel_and_inverse_epsilon_mapping():
    trainer = CachedIntermediateTrainer(cfg())
    x_c = torch.tensor([[[[0.5]]], [[[1.2]]]])
    v = torch.tensor([80, 95], dtype=torch.long)
    z1 = torch.tensor([[[[0.7]]], [[[-0.4]]]])
    z2 = torch.tensor([[[[-0.8]]], [[[0.3]]]])
    x_v, x_u, times = trainer.training_pair(x_c, v, z1, z2)
    a = vp_alpha_bar(cfg())
    a_u = a[v-8].float().view(2,1,1,1)
    a_v = a[v].float().view(2,1,1,1)
    expected_u = (a_u/a[65]).sqrt()*x_c + (1-a_u/a[65]).sqrt()*z1
    expected_v = (a_v/a_u).sqrt()*expected_u + (1-a_v/a_u).sqrt()*z2
    assert torch.equal(times, v)
    assert torch.allclose(x_u, expected_u, atol=1e-6)
    assert torch.allclose(x_v, expected_v, atol=1e-6)
    r = (a_v/a_u).sqrt()
    q = 1-a_v/a_u
    b = (1-a_v).sqrt()
    epsilon = (((x_v-r*x_u)*b/q).detach().clone()).requires_grad_()
    x_u_hat = trainer.intermediate_from_epsilon(x_v, v, epsilon)
    assert torch.allclose(x_u_hat, x_u, atol=1e-6)
    x_u_hat.square().mean().backward()
    assert torch.isfinite(epsilon.grad).all()


def test_invalid_ordering_and_input_bounds_fail_closed():
    with pytest.raises(ValueError):
        cache_schedule(cfg(phase1_cache_step=73))
    with pytest.raises(ValueError):
        cache_schedule(cfg(phase1_gap_steps=0))
    ds = CountingDataset(2)
    ds.images[0,0,0,0] = 2.0
    with pytest.raises(ValueError, match="Cache DP bound"):
        make_private_cache_loader(DataLoader(ds, batch_size=2), vp_alpha_bar(cfg()), 65)
