"""One-time private Gaussian cache and intermediate-state Phase-I training.

Only the materialization function accesses the original dataset.  All Phase-I
iterations use a TensorDataset containing *only* privatized images (no labels).
"""

import math
import secrets

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.schedules import linear_beta_schedule


def cache_schedule(config):
    """Return (cache_step, gap_steps), validating the Phase-I time ordering."""
    total = int(config.diffusion.timesteps)
    split = int(config.dp.S)
    gap = int(config.train.get("phase1_gap_steps", 32))
    step = config.train.get("phase1_cache_step", None)
    step = split - gap if step is None else int(step)
    if not (1 <= split < total):
        raise ValueError("DP-Promise requires 1 <= dp.S < diffusion.timesteps")
    if not (1 <= gap <= split):
        raise ValueError("phase1_gap_steps must be in [1, dp.S]")
    if not (0 <= step <= split - gap):
        raise ValueError("phase1_cache_step must satisfy 0 <= c <= dp.S - gap")
    return step, gap


def vp_alpha_bar(config):
    """Use the same zero-based VP/DDPM schedule as DPPromiseTrainer."""
    betas = linear_beta_schedule(
        config.diffusion.timesteps,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
    ).double()
    if torch.any((betas <= 0) | (betas >= 1)):
        raise ValueError("VP beta values must be strictly between zero and one")
    return torch.cumprod(1.0 - betas, dim=0)


def make_private_cache_loader(original_loader, alpha_bar, cache_step, *, generator=None):
    """Materialize exactly one noised image per original record in CPU memory.

    Production draws an unpredictable seed independently of train.seed. Passing
    a deterministic generator is intended only for tests; a known public seed
    must not be used to generate privacy noise.
    """
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(secrets.token_bytes(8), "big") % (2**63))

    # A sequential pass gives one cache entry per original record, even when
    # the original Phase-I DataLoader is shuffled or has drop_last enabled.
    source = DataLoader(
        original_loader.dataset,
        batch_size=original_loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=original_loader.num_workers,
    )
    a = math.sqrt(float(alpha_bar[cache_step]))
    b = math.sqrt(1.0 - float(alpha_bar[cache_step]))
    cached_data = None
    offset = 0
    with torch.no_grad():
        for batch in source:
            images = batch[0].detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(images).all() or torch.any(images.abs() > 1.000001):
                raise ValueError("Cache DP bound requires input pixels in [-1, 1]")
            if cached_data is None:
                cached_data = torch.empty(
                    (len(source.dataset),) + tuple(images.shape[1:]), dtype=images.dtype
                )
            noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
            end = offset + len(images)
            cached_data[offset:end].copy_(a * images + b * noise)
            offset = end
    if cached_data is None or offset != len(source.dataset):
        raise ValueError("Cannot construct a private cache from an empty or changing dataset")
    # No raw images or per-record labels are retained in the private cache.
    return DataLoader(
        TensorDataset(cached_data),
        batch_size=original_loader.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=original_loader.pin_memory,
    )


class CachedIntermediateTrainer(nn.Module):
    """Train epsilon-output UNet with an equivalent intermediate x_u MSE.

    Per iteration: cache x_c -> fresh x_u -> fresh x_v, v in [S,T),
    u=v-gap >= c. Network output remains the marginal-epsilon interface used
    by the unmodified Phase-II trainer and DDIM sampler.
    """

    def __init__(self, config):
        super().__init__()
        self.cache_step, self.gap_steps = cache_schedule(config)
        self.split = int(config.dp.S)
        self.timesteps = int(config.diffusion.timesteps)
        self.register_buffer("alpha_bar", vp_alpha_bar(config))

    def _alpha(self, times, image):
        coeff = self.alpha_bar[times].to(dtype=image.dtype)
        return coeff.view((-1,) + (1,) * (image.ndim - 1))

    def training_pair(self, cache, v=None, noise_u=None, noise_v=None):
        """Return x_v, x_u, v using independent Markov Gaussian increments."""
        batch_size = cache.shape[0]
        if v is None:
            v = torch.randint(self.split, self.timesteps, (batch_size,), device=cache.device)
        if v.shape != (batch_size,) or v.dtype != torch.long:
            raise ValueError("v must be a LongTensor with one input time per image")
        if torch.any((v < self.split) | (v >= self.timesteps)):
            raise ValueError("Phase-I input time v must lie in [S, T)")
        u = v - self.gap_steps
        alpha_c = self.alpha_bar[self.cache_step].to(dtype=cache.dtype)
        alpha_u = self._alpha(u, cache)
        alpha_v = self._alpha(v, cache)
        ratio_uc = alpha_u / alpha_c
        ratio_vu = alpha_v / alpha_u
        if noise_u is None:
            noise_u = torch.randn_like(cache)
        if noise_v is None:
            noise_v = torch.randn_like(cache)
        x_u = ratio_uc.sqrt() * cache + (1.0 - ratio_uc).clamp_min(0).sqrt() * noise_u
        x_v = ratio_vu.sqrt() * x_u + (1.0 - ratio_vu).clamp_min(0).sqrt() * noise_v
        return x_v, x_u, v

    def intermediate_from_epsilon(self, x_v, v, predicted_epsilon):
        """Convert DDPM marginal-epsilon output to an intermediate estimate."""
        u = v - self.gap_steps
        alpha_u = self._alpha(u, x_v)
        alpha_v = self._alpha(v, x_v)
        r = (alpha_v / alpha_u).sqrt()
        q = 1.0 - alpha_v / alpha_u
        b_v = (1.0 - alpha_v).sqrt()
        return (x_v - (q / b_v) * predicted_epsilon) / r

    def forward(self, model, cache, y=None):
        x_v, x_u, v = self.training_pair(cache)
        epsilon_prediction = model(x_v, v, y)
        x_u_prediction = self.intermediate_from_epsilon(x_v, v, epsilon_prediction)
        return F.mse_loss(x_u_prediction, x_u)
