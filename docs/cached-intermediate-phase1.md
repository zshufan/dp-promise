# Cached intermediate-state Phase I (Scheme B)

This is an **opt-in** alternative Phase I for DP-Promise. The original path
remains the default. Phase II and the DDIM sampler are unchanged.

## Implementation

For the VP/DDPM schedule, let `alpha_bar[j]` be the product of (1-beta)
through timestep j. Use public cache step `c`, input timestep `v`,
gap `g`, and intermediate timestep `u=v-g`. Validate
`0 <= c <= S-g`, `1 <= g <= S < T`, and `S <= v < T`.

1. **Cache once:** for each normalized raw image `x0 in [-1,1]^d`,
   create `x_c = sqrt(alpha_bar[c])*x0 + sqrt(1-alpha_bar[c])*z_c`.
   The cache is RAM-only; no raw image or label is copied into its
   TensorDataset. Each record is read exactly once while constructing it.
   Noise comes from a new unpredictable generator, not the fixed train seed.
2. **Fresh two-stage noising per iteration:** uniformly draw `v` in
   `[S,T)`, let `u=v-g`, then independently sample
   `z_1,z_2 ~ N(0,I)`:
   ```
   x_u = sqrt(alpha_bar[u]/alpha_bar[c])*x_c
         + sqrt(1-alpha_bar[u]/alpha_bar[c])*z_1
   x_v = sqrt(alpha_bar[v]/alpha_bar[u])*x_u
         + sqrt(1-alpha_bar[v]/alpha_bar[u])*z_2
   ```
   No raw images or labels are used by the Phase-I optimizer.
3. **Intermediate-target MSE with epsilon-compatible UNet:** let
   `r=sqrt(alpha_bar[v]/alpha_bar[u])`,
   `q=1-r^2`, `b_v=sqrt(1-alpha_bar[v])`. Convert UNet's
   `epsilon_hat` into `x_u_hat=(x_v-(q/b_v)*epsilon_hat)/r`,
   and minimize `mean((x_u_hat-x_u)^2)`.
   This retains the marginal-noise output convention required by the
   original Phase-II DP-SGD and DDIM sampler.

This implements the *cached* variant of two-stage noising. Unlike the
writeup's fixed-supervision-time construction, the one-time private cache
is at `c` and the supervised intermediate target is the **fresh** `x_u`.
Both are generated solely from `x_c` and independent randomness, so
the same Gaussian post-processing argument applies.

## Privacy accounting and caveats

For replace-one adjacency and pixels bounded to `[-1,1]^d`, the cache
sensitivity is `Delta=2*sqrt(d)` and its ideal Gaussian GDP parameter is

```
mu_cache = Delta*sqrt(alpha_bar[c]/(1-alpha_bar[c]))
```

The number of Phase-I epochs does **not** increase this cache privacy
cost. This statement assumes ideal independent Gaussian sampling, stable
public preprocessing, and that the only private information accessible in
Phase I is through the cache.

**Phase II still accesses raw images and private labels.** Thus the
writeup's cache-only composition/amplification theorem does not apply
to end-to-end training. For comparison only, the repository now reports
the exact ideal-Gaussian cache mu together with the preexisting
*approximate* Phase-II GDP expression
`q_2*sqrt(K_2*(exp(1/sigma^2)-1))`, combining the two mu values in
quadrature. This is NOT a certified finite-step DP accountant; the
reported epsilon is an **estimate**, not a formal end-to-end guarantee.

Stock PyTorch RNG is not a cryptographically secure Gaussian sampler.
An OS-random seed prevents casual reproducibility but does not provide
cryptographic DP. A production proof/claim additionally requires a
reviewed secure Gaussian implementation, strict input/sensitivity
validation, and a rigorous Phase-II privacy accountant.

## Use

```bash
python train.py --config configs/dp_promise/mnist_28/cached_intermediate/config.yaml
PYTHONPATH=. pytest -q tests/test_cached_phase1.py tests/test_cached_accounting.py
```

MNIST settings: `T=1000,S=900,c=868,g=32`;
`v` in `[900,999]`, `u=v-32` in `[868,967]`.
The original pretrained checkpoint path is still required; Phase II
continues unchanged. RAM cache memory is about
`4 * records * channels * height * width` bytes.
