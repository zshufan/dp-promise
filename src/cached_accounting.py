"""One-time Gaussian-cache GDP plus approximate Phase-II GDP bookkeeping.

The cache bound is exact with ideal Gaussian randomness and bounded data.
The subsampled DP-SGD term follows DP-Promise's existing approximation and
is NOT a certified finite-step privacy bound.
"""
import math

from src.cached_phase1 import cache_schedule, vp_alpha_bar


def cache_gdp_mu(alpha_bar_c, image_dimension):
    """Replace-one sensitivity for pixels in [-1,1] is 2*sqrt(d)."""
    if image_dimension < 1:
        raise ValueError("image_dimension must be positive")
    alpha_bar_c = float(alpha_bar_c)
    if not (0 < alpha_bar_c < 1):
        raise ValueError("cache alpha_bar must be strictly between 0 and 1")
    return 2 * math.sqrt(image_dimension) * math.sqrt(alpha_bar_c / (1 - alpha_bar_c))


def approximate_phase2_gdp_mu(sample_rate, iterations, sigma):
    if not (0 < sample_rate <= 1) or iterations < 0:
        raise ValueError("invalid sample rate or iterations")
    if iterations == 0:
        return 0.0
    if sigma <= 0:
        raise ValueError("dp.sigma must be positive for Phase II")
    return sample_rate * math.sqrt(iterations * math.expm1(1 / (sigma * sigma)))


def cached_privacy_components(config, num_records):
    if num_records < 1:
        raise ValueError("num_records must be positive")
    step, gap = cache_schedule(config)
    dimension = int(config.data.img_ch) * int(config.data.img_size) ** 2
    mu_cache = (
        cache_gdp_mu(vp_alpha_bar(config)[step], dimension)
        if int(config.train.epochs1) > 0 else 0.0
    )
    batch_size2 = int(config.train.batch_size2)
    if batch_size2 <= 0:
        raise ValueError("train.batch_size2 must be positive")
    logical_batches = math.ceil(num_records / batch_size2)
    mu_phase2 = approximate_phase2_gdp_mu(
        1 / logical_batches,
        int(config.train.epochs2) * logical_batches,
        float(config.dp.sigma),
    )
    return {
        "cache_step": step,
        "gap_steps": gap,
        "mu_cache": mu_cache,
        "mu_phase2_approx": mu_phase2,
        "mu_total_approx": math.hypot(mu_cache, mu_phase2),
    }
