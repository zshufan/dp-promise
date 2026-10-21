import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import compare_phase1_utility as comparison
from evaluation import classifier_utils, eval_downstream


def test_configs_match_except_phase1_objective():
    args = comparison.build_parser().parse_args(["--from-scratch", "--seeds", "42", "43"])
    configs = comparison.make_configs(args)
    for seed in (42, 43):
        original = OmegaConf.to_container(configs[f"seed_{seed}/original"])
        cached = OmegaConf.to_container(configs[f"seed_{seed}/cached_intermediate"])
        assert original["train"].pop("phase1_objective") == "original"
        assert cached["train"].pop("phase1_objective") == "cached_intermediate"
        assert original == cached
        assert original["train"]["phase2_seed"] == seed
        assert original["sample"]["final_seed"] == seed
        assert original["dp"]["sigma"] == 7.0


def test_shared_pretraining_uses_final_ema_and_is_reused(monkeypatch, tmp_path):
    args = comparison.build_parser().parse_args(["--pretrain", "--pretrain-epochs", "1"])
    config = next(iter(comparison.make_configs(args).values()))
    config.train.ckpt_path = str(tmp_path / "weights/fmnist.ckpt")
    config.model.ch_mult = [1]
    calls = []

    def fake_pretrain(command, log_path):
        calls.append(command)
        saved = OmegaConf.load(command[command.index("--config") + 1])
        assert saved.data.name == "fmnist"
        assert saved.data.class_condition is False
        assert saved.model == config.model
        assert saved.diffusion == config.diffusion
        attempt = Path(command[command.index("--workdir") + 1])
        checkpoints = attempt / "vanilla-test/checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "ema_0.ckpt").write_bytes(b"initial weights")
        (checkpoints / "ema_1.ckpt").write_bytes(b"final EMA weights")

    monkeypatch.setattr(comparison, "run_command", fake_pretrain)
    comparison.ensure_pretrained(config, args)
    checkpoint = Path(config.train.ckpt_path)
    assert checkpoint.read_bytes() == b"final EMA weights"
    comparison.ensure_pretrained(config, args)
    assert len(calls) == 1


def test_failed_pretraining_does_not_publish_initial_weights(monkeypatch, tmp_path):
    args = comparison.build_parser().parse_args(["--pretrain", "--pretrain-epochs", "1"])
    config = next(iter(comparison.make_configs(args).values()))
    config.train.ckpt_path = str(tmp_path / "fmnist.ckpt")

    def incomplete_pretrain(command, log_path):
        attempt = Path(command[command.index("--workdir") + 1])
        checkpoints = attempt / "vanilla-test/checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "ema_0.ckpt").write_bytes(b"initial weights")

    monkeypatch.setattr(comparison, "run_command", incomplete_pretrain)
    with pytest.raises(RuntimeError, match="final EMA checkpoint"):
        comparison.ensure_pretrained(config, args)
    assert not Path(config.train.ckpt_path).exists()


def test_pretraining_saves_final_epoch_when_not_multiple_of_ten(monkeypatch, tmp_path):
    import src.runners as runner

    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, x, t, y):
            return self.scale * x

    monkeypatch.setattr(runner, "sample_example_image", lambda *args: torch.zeros(1, 1, 2, 2))
    monkeypatch.setattr(runner, "tqdm", lambda values: values)
    model = torch.nn.DataParallel(TinyDenoiser())
    loader = DataLoader(TensorDataset(torch.ones(2, 1, 2, 2), torch.zeros(2, dtype=torch.long)))
    cfg = OmegaConf.create({"data": {"class_condition": False},
                           "diffusion": {"timesteps": 100, "beta_start": .0001, "beta_end": .02},
                           "train": {"epochs": 1, "ema_rate": .9}})
    runner.train_ddpm(model, torch.optim.Adam(model.parameters(), lr=.01), loader,
                      workdir=tmp_path, config=cfg)
    assert (tmp_path / "checkpoints/ema_1.ckpt").is_file()
    assert (tmp_path / "checkpoints/model_1.ckpt").is_file()


def test_orchestration_resume_and_summary(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(comparison, "preflight", lambda *args: None)
    monkeypatch.setattr("torchvision.datasets.MNIST", lambda **kwargs: None)

    def fake_run(command, log_path):
        calls.append(command)
        if "--workdir" in command:
            workdir = Path(command[command.index("--workdir") + 1])
            path = workdir / "fake-run/evaluation"
            path.mkdir(parents=True)
            np.savez(path / "synthesis.npz", data=np.zeros((20, 1, 28, 28)), labels=np.zeros(20))
        else:
            output = Path(command[command.index("--output_path") + 1])
            synthesis = command[command.index("--synthesis_path") + 1]
            acc = .8 if output.parent.name == "cached_intermediate" else .7
            metrics = {
                "seed": 42, "epochs": 1, "num_synthetic": 20, "synthesis_path": synthesis,
                "classifiers": {name: {"test_accuracy": acc, "validation_accuracy": .6}
                                for name in ("MLP", "CNN")},
            }
            comparison.atomic_json(output / "metrics.json", metrics)

    monkeypatch.setattr(comparison, "run_command", fake_run)
    output = tmp_path / "experiment with spaces"
    args = ["--from-scratch", "--seeds", "42", "43", "--epochs1", "1", "--epochs2", "1",
            "--num-samples", "20", "--eval-epochs", "1", "--output", str(output)]
    comparison.main(args)
    assert len(calls) == 8
    summary = json.loads((output / "summary.json").read_text())
    assert summary["CNN"]["n_pairs"] == 2
    assert summary["CNN"]["delta_percentage_points_mean"] == pytest.approx(10)
    assert summary["CNN"]["delta_percentage_points_std"] == pytest.approx(0)
    assert len((output / "results.csv").read_text().splitlines()) == 9
    comparison.main([*args, "--resume"])
    assert len(calls) == 8
    (output / "seed_42/original/utility/metrics.json").unlink()
    comparison.main([*args, "--resume"])
    assert len(calls) == 9  # Only the missing evaluation is repeated.
    assert "--synthesis_path" in calls[-1]
    with pytest.raises(ValueError, match="differ from manifest"):
        comparison.main([*args, "--resume", "--epochs2", "2"])
    assert len(calls) == 9


def test_incomplete_training_retries_in_new_directory(monkeypatch, tmp_path):
    run_dir = tmp_path / "seed_42/original"
    run_dir.mkdir(parents=True)
    args = SimpleNamespace(eval_seed=42, eval_epochs=1, num_samples=20,
                           eval_batch_size=4, num_workers=0)
    calls = []

    def fail(command, log_path):
        calls.append(command)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(comparison, "run_command", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="interrupted"):
            comparison.run_one(run_dir, args)
    assert calls[0][calls[0].index("--workdir") + 1].endswith("attempt_001")
    assert calls[1][calls[1].index("--workdir") + 1].endswith("attempt_002")
    assert not (run_dir / "training_complete.json").exists()


def test_summary_requires_pairs_and_does_not_invent_single_run_std():
    row = {"seed": 42, "method": "original", "classifier": "CNN", "test_accuracy": .8}
    assert comparison.summarize([row]) == {}
    result = comparison.summarize([row, {**row, "method": "cached_intermediate", "test_accuracy": .7}])
    assert result["CNN"]["delta_percentage_points_mean"] == pytest.approx(-10)
    assert result["CNN"]["delta_percentage_points_std"] is None


def test_classifier_restores_best_validation_weights(monkeypatch):
    model = torch.nn.Linear(1, 2, bias=False)
    loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.zeros(1, dtype=torch.long)))

    class FixedSteps(torch.optim.SGD):
        count = 0

        def step(self):
            self.count += 1
            with torch.no_grad():
                model.weight.fill_(self.count)

    scores = iter([1.0, 0.0])
    monkeypatch.setattr(classifier_utils, "test_classifier", lambda *args: next(scores))
    best = classifier_utils.train_classifier(model, loader, loader, FixedSteps(model.parameters(), lr=.1), epochs=2)
    assert best == 1.0
    assert torch.equal(model.weight, torch.ones_like(model.weight))


def test_real_classifier_evaluation_on_offline_fixture(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    images = rng.random((40, 1, 28, 28), dtype=np.float32)
    labels = np.arange(40, dtype=np.int64) % 10
    synthesis = tmp_path / "synthesis.npz"
    np.savez(synthesis, data=images, labels=labels)
    test_data = TensorDataset(torch.from_numpy(images[:20]), torch.from_numpy(labels[:20]))
    monkeypatch.setattr("torchvision.datasets.MNIST", lambda **kwargs: test_data)
    args = SimpleNamespace(output_path=str(tmp_path / "metrics"), seed=42,
                           synthesis_path=str(synthesis), dataset="mnist", data_root=str(tmp_path),
                           batch_size=20, num_workers=0, lr=.001, epochs=1)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        metrics = eval_downstream.main(args)
    finally:
        torch.set_num_threads(previous_threads)
    assert metrics["num_synthetic"] == 40
    assert metrics["num_test"] == 20
    assert set(metrics["classifiers"]) == {"MLP", "CNN"}
    assert json.loads((tmp_path / "metrics/metrics.json").read_text()) == metrics


@pytest.mark.parametrize("objective", comparison.METHODS)
def test_real_two_phase_runner_smoke(monkeypatch, tmp_path, objective):
    # Real custom Opacus, optimizer, two phases and checkpoint saving; no downloads.
    import src.runners as runner

    class TinyDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x, t, y):
            return self.conv(x)

    seeds = []
    requested_samples = []
    actual_seed = runner.seed_everthing

    def seed(value):
        seeds.append(value)
        actual_seed(value)

    def sample(model, config, num_samples):
        requested_samples.append(num_samples)
        return torch.zeros(num_samples, 1, 2, 2), torch.zeros(num_samples, dtype=torch.long)

    monkeypatch.setattr(runner, "seed_everthing", seed)
    monkeypatch.setattr(runner, "sample_images", sample)
    monkeypatch.setattr(runner, "sample_example_image", lambda *args: torch.zeros(1, 1, 2, 2))
    monkeypatch.setattr(runner, "tqdm", lambda values: values)
    cfg = OmegaConf.create({
        "data": {"class_condition": True},
        "diffusion": {"timesteps": 100, "beta_start": .0001, "beta_end": .02},
        "train": {"phase1_objective": objective, "phase1_cache_step": 65, "phase1_gap_steps": 8,
                  "lr1": .001, "lr2": .001, "epochs1": 1, "epochs2": 1,
                  "ema_rate": .9, "num_noise_sample": 2, "max_physical_batch_size": 4,
                  "phase2_seed": 42},
        "dp": {"S": 80, "sigma": 2.0, "max_grad_norm": 1.0},
        "sample": {"num_samples": 12, "final_seed": 123},
    })
    torch.manual_seed(42)
    dataset = TensorDataset(torch.rand(16, 1, 2, 2) * 2 - 1, torch.zeros(16, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=8)
    model = TinyDenoiser()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        runner.train_dp_promise(model, loader, loader, config=cfg, workdir=tmp_path)
    finally:
        torch.set_num_threads(previous_threads)
    assert seeds == [42, 123]
    assert requested_samples == [12]
    assert (tmp_path / "checkpoints/model.ckpt").is_file()
    assert (tmp_path / "checkpoints/ema_1.ckpt").is_file()
    with np.load(tmp_path / "evaluation/synthesis.npz") as artifact:
        assert artifact["data"].shape == (12, 1, 2, 2)
