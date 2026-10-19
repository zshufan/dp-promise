import math

import pytest
from omegaconf import OmegaConf

from src.cached_accounting import (
    approximate_phase2_gdp_mu, cache_gdp_mu, cached_privacy_components,
)
from src.cached_phase1 import vp_alpha_bar


def cfg(epochs1):
    return OmegaConf.create({
        "data": {"img_ch": 1, "img_size": 28},
        "diffusion": {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02},
        "train": {"phase1_cache_step": 868, "phase1_gap_steps": 32,
                  "epochs1": epochs1, "epochs2": 50, "batch_size2": 4096},
        "dp": {"S": 900, "sigma": 7.0},
    })


def test_gaussian_replace_one_sensitivity():
    assert cache_gdp_mu(0.5, 4) == pytest.approx(4)
    assert cache_gdp_mu(0.8, 1) == pytest.approx(4)


def test_cache_privacy_constant_across_epochs():
    a = cached_privacy_components(cfg(1), 60000)
    b = cached_privacy_components(cfg(100), 60000)
    alpha = float(vp_alpha_bar(cfg(1))[868])
    expected = 56 * math.sqrt(alpha/(1-alpha))
    assert a["mu_cache"] == pytest.approx(expected)
    assert a["mu_cache"] == b["mu_cache"]
    assert a["mu_phase2_approx"] == b["mu_phase2_approx"]
    assert a["mu_total_approx"] == pytest.approx(
        math.hypot(a["mu_cache"], a["mu_phase2_approx"])
    )


def test_zero_phase1_and_zero_phase2():
    assert cached_privacy_components(cfg(0), 60000)["mu_cache"] == 0
    assert approximate_phase2_gdp_mu(0.01, 0, 0) == 0


def test_invalid_accounting_inputs_rejected():
    with pytest.raises(ValueError):
        cache_gdp_mu(1, 4)
    with pytest.raises(ValueError):
        approximate_phase2_gdp_mu(.1, 3, 0)
