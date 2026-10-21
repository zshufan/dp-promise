"""Run matched-noise MNIST training pairs and synthetic-data utility evaluation.

Examples:
  python compare_phase1_utility.py --pretrain
  python compare_phase1_utility.py --checkpoint _pretrained_models/fmnist.ckpt
  python compare_phase1_utility.py --seeds 42 43 44 --resume
  python compare_phase1_utility.py --dry-run
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent
METHODS = ("original", "cached_intermediate")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT /
                        "configs/dp_promise/mnist_28/cached_intermediate/config.yaml")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--checkpoint", type=Path, help="Shared pretrained state_dict")
    initialization.add_argument("--from-scratch", action="store_true",
                                help="Explicitly compare both methods without pretraining")
    initialization.add_argument("--pretrain", action="store_true",
                                help="Create shared Fashion-MNIST weights if the configured checkpoint is missing")
    parser.add_argument("--pretrain-epochs", type=int, default=300,
                        help="Fashion-MNIST epochs when --pretrain creates weights (default: 300)")
    parser.add_argument("--output", type=Path, default=ROOT / "_workdir/phase1_utility")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs1", type=int, help="Override for both methods; default 3")
    parser.add_argument("--epochs2", type=int, help="Override for both methods; default 50")
    parser.add_argument("--num-samples", type=int, default=60000)
    parser.add_argument("--sample-steps", type=int, help="Override DDIM steps for both methods")
    parser.add_argument("--eval-epochs", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed stages only when the experiment manifest matches")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print settings and commands; no files, downloads or GPU required")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate dependencies, CUDA and checkpoint without running")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow slow CPU training when CUDA is unavailable")
    return parser


def make_configs(args):
    if args.pretrain and args.pretrain_epochs < 1:
        raise ValueError("Pretraining epochs must be positive")
    base = OmegaConf.load(args.config)
    if (base.data.name, base.data.img_ch, base.data.img_size, base.data.num_classes) != (
            "mnist", 1, 28, 10) or not base.data.class_condition:
        raise ValueError("This comparison requires class-conditioned 28x28 MNIST")
    if args.from_scratch:
        base.train.ckpt_path = None
    else:
        checkpoint = args.checkpoint or base.train.get("ckpt_path")
        if not checkpoint:
            raise ValueError("Supply --checkpoint, or explicitly choose --from-scratch")
        checkpoint = Path(checkpoint).expanduser()
        # Explicit CLI paths are relative to the caller; config paths to the repo.
        if not checkpoint.is_absolute() and args.checkpoint is None:
            checkpoint = ROOT / checkpoint
        base.train.ckpt_path = str(checkpoint.resolve())
    for field in ("epochs1", "epochs2"):
        if getattr(args, field) is not None:
            base.train[field] = getattr(args, field)
        if int(base.train[field]) < 1:
            raise ValueError("Both phases must have at least one epoch")
    if (args.num_samples < 10 or args.eval_epochs < 1 or args.eval_batch_size < 1
            or args.num_workers < 0):
        raise ValueError("Need >=10 samples, positive evaluation settings and >=0 workers")
    if len(set(args.seeds)) != len(args.seeds) or any(
            not 0 <= seed < 2**32 for seed in [*args.seeds, args.eval_seed]):
        raise ValueError("Seeds must be unique training seeds and integers in [0, 2**32)")
    base.data.num_workers = args.num_workers
    base.sample.num_samples = args.num_samples
    if args.sample_steps is not None:
        base.sample.samplesteps = args.sample_steps
    if not 1 <= int(base.sample.samplesteps) <= int(base.diffusion.timesteps):
        raise ValueError("sample steps must lie in [1, diffusion.timesteps]")
    # This field labels outputs; noise is fixed by the shared schedule and sigma.
    base.dp.epsilon = "estimated"
    split, total = int(base.dp.S), int(base.diffusion.timesteps)
    gap = int(base.train.get("phase1_gap_steps", 32))
    cache_step = int(base.train.get("phase1_cache_step", split - gap))
    if not (1 <= gap <= split < total and 0 <= cache_step <= split - gap):
        raise ValueError("Require 1 <= gap <= S < T and 0 <= cache_step <= S-gap")
    base.train.phase1_cache_step, base.train.phase1_gap_steps = cache_step, gap
    configs = {}
    for seed in args.seeds:
        for method in METHODS:
            config = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
            config.train.phase1_objective = method
            config.train.seed = seed
            config.train.phase2_seed = seed
            config.sample.seed = seed
            config.sample.final_seed = seed
            configs[f"seed_{seed}/{method}"] = config
    return configs


def preflight(config, allow_cpu=False, allow_missing_checkpoint=False):
    import torch
    from opacus import PrivacyEngine
    from src.models import UNetModel
    # Load actual entry points, catching missing runtime dependencies before work.
    import train  # noqa: F401
    from evaluation import eval_downstream  # noqa: F401

    issues = []
    if "noise_multiplicity" not in inspect.signature(PrivacyEngine.make_private).parameters:
        issues.append("Install the custom Opacus fork from requirements.txt (not stock Opacus)")
    if not allow_cpu and not torch.cuda.is_available():
        issues.append("CUDA is unavailable; use a GPU node, or explicitly pass --allow-cpu")
    checkpoint = config.train.ckpt_path
    if checkpoint and not Path(checkpoint).is_file() and not allow_missing_checkpoint:
        issues.append(f"Pretrained checkpoint missing: {checkpoint}. Use --pretrain to create "
                      "shared Fashion-MNIST weights, --checkpoint with an existing file, "
                      "or --from-scratch for an unpretrained comparison")
    if issues:
        raise RuntimeError("\n".join(issues))
    if checkpoint and Path(checkpoint).is_file():
        model = UNetModel(
            in_channels=config.data.img_ch, model_channels=config.model.ch,
            out_channels=config.data.img_ch, num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=config.model.attn, dropout=config.model.dropout,
            channel_mult=config.model.ch_mult, num_classes=config.data.num_classes,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    print(f"Preflight OK: torch {torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)


def pretraining_config(config, args):
    pre = OmegaConf.load(ROOT / "configs/vanilla/fmnist_28/config.yaml")
    # Use exactly the architecture and diffusion schedule of the comparison.
    pre.model = OmegaConf.to_container(config.model, resolve=True)
    pre.diffusion = OmegaConf.to_container(config.diffusion, resolve=True)
    pre.data.num_workers = args.num_workers
    pre.train.epochs = args.pretrain_epochs
    return pre


def pretraining_command(directory, attempt):
    return [sys.executable, "-u", str(ROOT / "pretrain.py"),
            "--config", str(directory / "config.yaml"), "--workdir", str(attempt),
            "--job-id", "shared_fmnist"]


def ensure_pretrained(config, args):
    checkpoint = Path(config.train.ckpt_path)
    if checkpoint.is_file():
        print(f"Reuse shared pretrained weights: {checkpoint}", flush=True)
        return
    pre = pretraining_config(config, args)
    directory = checkpoint.with_name(checkpoint.stem + "_pretraining")
    directory.mkdir(parents=True, exist_ok=True)
    saved_config = directory / "config.yaml"
    if saved_config.exists() and OmegaConf.load(saved_config) != pre:
        raise ValueError(f"Pretraining settings differ from {saved_config}; use a base config "
                         "with a different train.ckpt_path for a new pretraining run")
    OmegaConf.save(pre, saved_config)
    # A failed/interrupted attempt never becomes the shared comparison checkpoint.
    index = 1
    while (directory / f"attempt_{index:03d}").exists():
        index += 1
    attempt = directory / f"attempt_{index:03d}"
    attempt.mkdir()
    print(f"Pretraining shared Fashion-MNIST model for {pre.train.epochs} epochs. "
          f"Progress: {attempt}/vanilla_*/logs/log.txt", flush=True)
    run_command(pretraining_command(directory, attempt), attempt / "console.log")
    candidates = list(attempt.glob(f"*/checkpoints/ema_{pre.train.epochs}.ckpt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected the final EMA checkpoint for epoch {pre.train.epochs} in {attempt}")
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    shutil.copyfile(candidates[0], temporary)
    temporary.replace(checkpoint)
    print(f"Shared pretrained weights saved: {checkpoint}", flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_experiment(output, manifest, configs, resume):
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError(f"Experiment exists: {output}; use --resume or a new --output")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume settings, checkpoint or source code differ from manifest; "
                             "use a new --output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Refusing to mix results with nonempty directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
    for key, config in configs.items():
        run_dir = output / key
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "config.yaml"
        if path.exists():
            if OmegaConf.load(path) != config:
                raise ValueError(f"Saved config was modified: {path}")
        else:
            OmegaConf.save(config, path)


def training_command(run_dir, attempt):
    return [sys.executable, "-u", str(ROOT / "train.py"),
            "--config", str(run_dir / "config.yaml"), "--workdir", str(attempt),
            "--job-id", f"{run_dir.parent.name}_{run_dir.name}"]


def evaluation_command(run_dir, synthesis, args):
    return [sys.executable, "-u", str(ROOT / "evaluation/eval_downstream.py"),
            "--dataset", "mnist", "--synthesis_path", str(synthesis),
            "--output_path", str(run_dir / "utility"), "--data_root", str(ROOT / "_data"),
            "--epochs", str(args.eval_epochs), "--seed", str(args.eval_seed),
            "--batch_size", str(args.eval_batch_size), "--num_workers", str(args.num_workers)]


def run_command(command, log_path):
    print("$ " + shlex.join(command), flush=True)
    print(f"Console log: {log_path}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        # train.py redirects runtime stderr to its own logs/err.txt.
        raise RuntimeError(f"Command failed (exit {result.returncode}); see {log_path} "
                           f"and the run's logs/err.txt under {log_path.parent}. "
                           "Rerun with --resume after resolving the failure")


def run_one(run_dir, args):
    completion = run_dir / "training_complete.json"
    if completion.exists():
        synthesis = Path(json.loads(completion.read_text())["synthesis_path"])
        if not synthesis.is_file():
            raise ValueError(f"Completed training artifact is missing: {synthesis}")
        print(f"Reuse completed training: {run_dir}", flush=True)
    else:
        training_root = run_dir / "training"
        training_root.mkdir(exist_ok=True)
        index = 1
        while (training_root / f"attempt_{index:03d}").exists():
            index += 1
        attempt = training_root / f"attempt_{index:03d}"
        attempt.mkdir()
        print(f"Training progress: {attempt}/dp-promise_*/logs/log.txt", flush=True)
        run_command(training_command(run_dir, attempt), attempt / "console.log")
        candidates = list(attempt.glob("*/evaluation/synthesis.npz"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one synthesis.npz under {attempt}")
        synthesis = candidates[0]
        atomic_json(completion, {"synthesis_path": str(synthesis)})
    metrics_path = run_dir / "utility/metrics.json"
    if not metrics_path.exists():
        (run_dir / "utility").mkdir(exist_ok=True)
        run_command(evaluation_command(run_dir, synthesis, args), run_dir / "utility/console.log")
    if not metrics_path.is_file():
        raise RuntimeError(f"Evaluation did not produce {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    if (metrics["seed"], metrics["epochs"], metrics["num_synthetic"],
            metrics["synthesis_path"]) != (
            args.eval_seed, args.eval_epochs, args.num_samples, str(synthesis.resolve())):
        raise ValueError(f"Evaluation metadata do not match this run: {metrics_path}")
    for classifier in ("MLP", "CNN"):
        value = metrics["classifiers"][classifier]["test_accuracy"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid accuracy in {metrics_path}")
    return metrics


def summarize(records):
    summary = {}
    for classifier in ("MLP", "CNN"):
        by_seed = {}
        for row in records:
            if row["classifier"] == classifier:
                by_seed.setdefault(row["seed"], {})[row["method"]] = row["test_accuracy"]
        pairs = [pair for pair in by_seed.values() if all(m in pair for m in METHODS)]
        if not pairs:
            continue
        original = [pair[METHODS[0]] for pair in pairs]
        cached = [pair[METHODS[1]] for pair in pairs]
        differences = [100 * (b - a) for a, b in zip(original, cached)]
        stats = {"n_pairs": len(pairs)}
        for key, values in (("original_accuracy", original), ("cached_accuracy", cached),
                            ("delta_percentage_points", differences)):
            stats[key + "_mean"] = statistics.mean(values)
            stats[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        summary[classifier] = stats
    return summary


def save_results(output, records):
    temporary = output / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "seed", "method", "classifier", "test_accuracy", "validation_accuracy"])
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "results.csv")
    summary = summarize(records)
    atomic_json(output / "summary.json", summary)
    for classifier, stats in summary.items():
        print(f"{classifier}: original={100 * stats['original_accuracy_mean']:.2f}%, "
              f"cached={100 * stats['cached_accuracy_mean']:.2f}%, "
              f"cached-original={stats['delta_percentage_points_mean']:+.2f} pp "
              f"({stats['n_pairs']} pairs)", flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output = args.output.expanduser().resolve()
    configs = make_configs(args)
    first = next(iter(configs.values()))
    print(f"Matched input noise: S={first.dp.S}, T={first.diffusion.timesteps}; "
          f"Phase II sigma={first.dp.sigma}; seeds={args.seeds}", flush=True)
    print("Metric: train MLP/CNN on synthetic data, select on synthetic validation data, "
          "test on the real MNIST test split.", flush=True)
    if args.dry_run:
        if args.pretrain:
            checkpoint = Path(first.train.ckpt_path)
            if checkpoint.is_file():
                print(f"Reuse shared pretrained weights: {checkpoint}")
            else:
                directory = checkpoint.with_name(checkpoint.stem + "_pretraining")
                print(f"\nShared Fashion-MNIST pretraining ({args.pretrain_epochs} epochs):\n"
                      f"{OmegaConf.to_yaml(pretraining_config(first, args))}")
                print(shlex.join(pretraining_command(directory, directory / "attempt_001")))
        for key, config in configs.items():
            run_dir = args.output / key
            print(f"\n{key}\n{OmegaConf.to_yaml(config)}")
            print(shlex.join(training_command(run_dir, run_dir / "training/attempt_001")))
            print(shlex.join(evaluation_command(run_dir, Path("<completed-run>/evaluation/synthesis.npz"), args)))
        return
    preflight(first, args.allow_cpu, args.pretrain)
    if args.check_only:
        if args.pretrain and not Path(first.train.ckpt_path).is_file():
            print(f"Shared checkpoint will be created by {args.pretrain_epochs} epochs of "
                  "Fashion-MNIST pretraining when running without --check-only")
        return
    if args.pretrain:
        # Refuse obvious output conflicts before spending time on pretraining.
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise ValueError(f"Experiment exists: {args.output}; use --resume or a new --output")
        ensure_pretrained(first, args)
        preflight(first, args.allow_cpu)
    source_paths = [ROOT / "train.py", ROOT / "pretrain.py", Path(__file__).resolve(),
                    *sorted((ROOT / "src").glob("*.py")),
                    ROOT / "evaluation/eval_downstream.py", ROOT / "evaluation/classifier_utils.py",
                    ROOT / "evaluation/models.py"]
    manifest = {
        "schema_version": 1,
        "configs": {key: OmegaConf.to_container(config, resolve=True) for key, config in configs.items()},
        "checkpoint_sha256": file_hash(first.train.ckpt_path) if first.train.ckpt_path else None,
        "evaluation": {"epochs": args.eval_epochs, "seed": args.eval_seed,
                       "batch_size": args.eval_batch_size, "num_workers": args.num_workers},
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
    }
    prepare_experiment(args.output, manifest, configs, args.resume)
    # Download both splits up front so evaluation cannot fail due to absent test data.
    from torchvision.datasets import MNIST
    for training in (True, False):
        MNIST(root=str(ROOT / "_data"), train=training, download=True)
    records = []
    for key, config in configs.items():
        print(f"\n=== {key} ===", flush=True)
        metrics = run_one(args.output / key, args)
        for classifier, scores in metrics["classifiers"].items():
            records.append({"seed": config.train.seed, "method": config.train.phase1_objective,
                            "classifier": classifier, **scores})
        save_results(args.output, records)
    print(f"\nFinished. Results: {args.output / 'results.csv'}\n"
          f"Summary (means, sample standard deviations, paired differences): "
          f"{args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
