# Differentially Private Diffusion Probabilistic Models for Image Synthesis (dp-promise)

This is the implementation of "dp-promise: Differentially Private Diffusion Probabilistic Models for Image Synthesis". The architecture of model is based on the improved DDPM repository (https://github.com/openai/improved-diffusion).

## Requirements

The code is adapted for Python 3.9, PyTorch 2.0.0 and torchvison 0.15.1 with CUDA 11.7. Run the following command to install requirements

```shell
$ pip install -r requirements.txt
```

## Preparation

Create the data folder

```shell
$ mkdir _data
```

For MNIST, Fashion MNIST and CIFAR-10, dataset will be downloaded automatically.

- CelebA

    Download dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, then

    ```shell
    $ python prepare_celeba.py --source _data/CelebA/img_align_celeba --dest _data/CelebA/processed
    # For 64 x 64 resolution
    $ python prepare_celeba.py --source _data/CelebA/img_align_celeba --dest _data/CelebA/processed --width 64 --height 64
    ```

- ImageNet

    Download dataset from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data, then

    ```shell
    $ python prepare_imagenet.py --source _data/ImageNet --dest _data/ImageNet/processed
    # For 64 x 64 resolution
    $ python prepare_imagenet.py --source _data/ImageNet --dest _data/ImageNet/processed --width 64 --height 64
    ```

## Pre-training

```shell
$ python pretrain.py --config configs/vanilla/<dataset>/config.yaml
```

## Training

```shell
$ python train.py --config configs/dp_promise/<dataset>/config.yaml
```

## Evaluation

### One-command MNIST Phase-I utility comparison

From the repository root, activate the training environment and run:

```shell
conda activate dpromise
python compare_phase1_utility.py --checkpoint _pretrained_models/fmnist.ckpt
```

If you do not have this checkpoint, run this command instead (no placeholder paths):

```shell
python compare_phase1_utility.py --pretrain
```

`--pretrain` first trains one unconditional Fashion-MNIST model using the
repository's pretraining settings (300 epochs by default), saves its final EMA
weights to `_pretrained_models/fmnist.ckpt`, then runs both comparison methods
from that same checkpoint. An existing checkpoint is validated and reused.
Pretraining runs only once, even for multiple comparison seeds. Its configs
and logs are saved under `_pretrained_models/fmnist_pretraining/`. Downloading
Fashion-MNIST requires network access on the first run. An interrupted
pretraining attempt restarts from scratch; completed shared weights are reused.
For a short pipeline test, `--pretrain-epochs 1` is available, but it does not
replace the full pretrained experiment. `--pretrain` cannot be combined with
`--checkpoint` or `--from-scratch`.

This runs **original Phase I + Phase II** and **cached-intermediate Phase I +
Phase II**, generates 60,000 images per model, trains the existing MLP and CNN
classifiers on synthetic data, selects their weights using a 10% synthetic
validation split, and measures accuracy on the real MNIST test split. MNIST is
downloaded if needed. This classifier-only workflow needs the training
dependencies, including the custom Opacus fork; it does not need TensorFlow,
TensorFlow GAN, scikit-learn, or an Inception model.

The defaults are one training seed (`42`), 3 Phase-I epochs, 50 Phase-II epochs,
and 50 classifier epochs. Both configurations come from the same base YAML;
only `train.phase1_objective` differs within a pair. In particular, both use
`S=900`, `T=1000`, and Phase-II `sigma=7`. Phase-II RNG and final-synthesis RNG
are reset to the same training seed for each pair. The private cache retains
its independent, unpredictable randomness; the seed does not reproduce the
cache bit for bit. Both methods use the existing runner's final model weights
for synthesis, while model and EMA checkpoints are saved separately.

For three paired training seeds:

```shell
python compare_phase1_utility.py --pretrain \
  --seeds 42 43 44 --output _workdir/phase1_utility_3seeds
```

Results are stored under the chosen output directory (default
`_workdir/phase1_utility`):

- `results.csv`: per-seed, per-method MLP/CNN test and validation accuracies,
  expressed as fractions in `[0,1]`.
- `summary.json`: original/cached means and sample standard deviations, plus
  paired **cached minus original** differences in percentage points. Positive
  differences favor cached training. A standard deviation is `null` with only
  one pair; incomplete pairs are excluded from the summary.
- `seed_<seed>/<method>/`: exact config, training attempts, generated
  `synthesis.npz`, checkpoints, and `utility/metrics.json` plus evaluation logs.
- `manifest.json`: settings, checkpoint hash, and source hashes checked on resume.

Repeat the **same command with `--resume`** after interruption. Completed
training and evaluations are reused. An interrupted training stage starts a
new attempt from its initial weights; this is not optimizer/checkpoint resume.
Changed settings, weights, or source code require a new output directory.
Do not run two processes against the same output directory simultaneously.

Useful checks and a shorter pipeline trial:

```shell
# Preview paired configs and commands without writing files or needing a GPU.
python compare_phase1_utility.py --dry-run

# Check dependencies, checkpoint compatibility, and CUDA before training.
python compare_phase1_utility.py --pretrain --check-only

# Exercise the pipeline with fewer epochs/images; not the full utility experiment.
python compare_phase1_utility.py --pretrain \
  --epochs1 1 --epochs2 1 --num-samples 1000 --sample-steps 20 \
  --eval-epochs 2 --output _workdir/phase1_utility_trial
```

If no pretrained weights are available, `--from-scratch` explicitly runs both
methods from the same seeded random initialization. This is a different
experiment from the pretrained comparison; the script never silently falls
back to random initialization. CUDA is required unless `--allow-cpu` is given.

This comparison matches the single-input diffusion noise distribution
(marginalizing over cache randomness), not the privacy budget. Cache reuse and
the existing Phase-I batch samplers differ: original uses Poisson sampling,
cached uses shuffled batches. Results therefore compare the two full Phase-I
implementations, rather than isolating only the loss function. Phase-I losses
are not directly comparable between the two objectives.

### Other evaluation commands

Note that the FID and IS evaluation is adapted for TensorFlow 2.12.0 and tensorflow-gan 2.1.0

```shell
$ pip install requirements_eval.txt
```

First, compute dataset statistics

```shell
$ cd evaluation
# For MNIST
$ python compute_dataset_stat.py --dataset {mnist,fmnist,cifar10}
$ python compute_dataset_stat.py --dataset celeba --path ../_data/CelebA/processed
```

Then, run the following command to conduct evaluation

```shell
# downstream classifier
$ python eval_downstream.py --dataset <dataset> --synthesis_path <path/to/synthesis.npz> --output_path <output_path>
# scikit-learn classifier
$ python eval_scikit.py --dataset <dataset> --synthesis_path <path/to/synthesis.npz> --output_path <output_path>
# FID score and Inception Score
$ python eval_vision.py --dataset <dataset> --synthesis_path <path/to/synthesis.npz> --output_path <output_path>
```
