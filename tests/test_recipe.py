# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Tests for training/recipe.py: grad-accum policy, profiles, and launch-command construction."""
import pytest

import recipe


def test_grad_accum_is_power_of_two_within_bounds():
    """compute_grad_accum always returns a power of two inside [2, max_grad_accum]."""
    for n in (500, 5_000, 20_000, 60_000, 80_000, 120_000, 1_000_000):
        for batch in (2, 4):
            ga = recipe.compute_grad_accum(n, batch)
            assert 2 <= ga <= recipe.COMMON["max_grad_accum"]
            assert ga & (ga - 1) == 0


def test_grad_accum_known_values():
    """Spot-check the rounding rule against hand-computed raw values."""
    # steps/epoch target = 3000 / 20 = 150; raw = n / 150 / (batch * 8)
    assert recipe.compute_grad_accum(60_000, 4) == 16    # raw 12.5
    assert recipe.compute_grad_accum(10_000, 4) == 2     # raw 2.08
    assert recipe.compute_grad_accum(100, 4) == 2        # floor of 2
    assert recipe.compute_grad_accum(10_000_000, 4) == 32  # capped


def test_grad_accum_monotone_in_dataset_size():
    """Larger training sets never get a smaller accumulation factor."""
    gas = [recipe.compute_grad_accum(n, 4) for n in range(1_000, 200_000, 1_000)]
    assert gas == sorted(gas)


@pytest.mark.parametrize("model,lr", [
    ("Qwen3-8B", "1e-4"), ("Llama-3.1-8B-Instruct", "1e-4"),
    ("Qwen3-14B", "5e-5"), ("Qwen3-32B", "5e-5"),
    ("Qwen3-30B-A3B-Instruct-2507", "5e-5"),
])
def test_profile_lr(model, lr):
    """Each base model resolves to the paper's learning rate (8B: 1e-4, larger: 5e-5)."""
    assert recipe.profile_for(model)["lr"] == lr


def test_moe_uses_fsdp_attention_only():
    """The MoE profile uses FSDP and restricts LoRA to the attention projections."""
    prof = recipe.profile_for("Qwen3-30B-A3B-Instruct-2507")
    assert prof["accelerate"] == "fsdp"
    assert prof["modules"] == "q_proj,k_proj,v_proj,o_proj"


def test_build_command_names_and_paths():
    """build_command produces the run name, dataset paths, and batch/step arithmetic."""
    name, cmd, ga, gb, steps = recipe.build_command(
        "Qwen3-8B", "Hybrid-PN", 42, "datasets", "output", 60_000, repo_root="/repo")
    assert name == "Qwen3-8B_Hybrid-PN_lr-1e-4_seed-42"
    assert cmd[cmd.index("--train_file") + 1] == "datasets/train_Hybrid-PN.jsonl"
    assert cmd[cmd.index("--val_file") + 1] == "datasets/val_Hybrid-PN.jsonl"
    assert cmd[cmd.index("--output_dir") + 1] == f"output/{name}"
    assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == str(ga)
    assert cmd[cmd.index("--report_to") + 1] == "none"
    assert gb == 4 * ga * recipe.GPUS
    assert steps == (60_000 // gb) * recipe.NUM_EPOCHS
    assert "/repo/training/train.py" in cmd


def test_build_command_moe_script_and_mlflow():
    """The MoE model launches train_moe.py via FSDP flags and honors --mlflow."""
    _, cmd, *_ = recipe.build_command(
        "Qwen3-30B-A3B-Instruct-2507", "Adapt", 0, "d", "o", 40_000,
        use_mlflow=True, repo_root="/repo")
    assert "/repo/training/train_moe.py" in cmd
    assert cmd[cmd.index("--report_to") + 1] == "mlflow"
    assert "--config_file" in cmd


def test_variants_are_paper_names():
    """VARIANTS lists exactly the paper's variant names, in the paper's order."""
    assert recipe.VARIANTS[:3] == ["Aegis", "Wild", "Mix"]
    assert not any(v.startswith("M") and v[1:2].isdigit() for v in recipe.VARIANTS)
