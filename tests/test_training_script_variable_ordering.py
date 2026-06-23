"""
Regression test: _episode_offset (and related resume-block vars) must be assigned
BEFORE they are used in the env constructor.

Bug history: MATD3 and MASAC failed with:
    UnboundLocalError: local variable '_episode_offset' referenced before assignment
because the checkpoint-resume block was placed AFTER env = CityLearn*Env(...).
This test uses Python's AST to verify the fix holds for all 4 training scripts.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

SCRIPTS = [
    "train_citylearn_v3_matd3.py",
    "train_citylearn_v3_masac.py",
    "train_citylearn_v3_happo.py",
    "train_citylearn_v3_maac.py",
]

# For each script: (env_constructor_pattern, vars_that_must_be_assigned_before_it)
ORDERING_CHECKS = {
    "train_citylearn_v3_matd3.py": {
        "env_pattern": r"episode_offset=_episode_offset",
        "must_precede": ["_episode_offset", "_completed", "_matd3_models"],
    },
    "train_citylearn_v3_masac.py": {
        "env_pattern": r"episode_offset=_episode_offset",
        "must_precede": ["_episode_offset", "_completed", "_masac_has_ckpt"],
    },
    "train_citylearn_v3_happo.py": {
        "env_pattern": r"episode_offset=_episode_offset",
        "must_precede": ["_episode_offset", "_completed"],
    },
    "train_citylearn_v3_maac.py": {
        "env_pattern": r"episode_offset=_episode_offset",
        "must_precede": ["_episode_offset", "_completed"],
    },
}


def _find_in_main(src: str, pattern: str, is_assignment: bool) -> int | None:
    """Return first line number in main() matching pattern as assignment or use."""
    lines = src.splitlines()
    in_main = False
    main_start = 0
    for i, line in enumerate(lines):
        if re.match(r"^def main\(", line):
            in_main = True
            main_start = i
        elif in_main and i > main_start + 1 and re.match(r"^def ", line):
            break

    for lineno, line in enumerate(lines[main_start:], main_start + 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("from") or stripped.startswith("import"):
            continue
        if re.search(pattern, stripped):
            return lineno
    return None


def test_syntax_all_scripts():
    for name in SCRIPTS:
        path = SCRIPTS_DIR / name
        src = path.read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise AssertionError(f"{name}: SyntaxError — {e}") from e


def test_exactly_one_checkpoint_resume_block():
    for name in SCRIPTS:
        src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        count = src.count("Checkpoint-resume")
        assert count == 1, (
            f"{name}: expected 1 Checkpoint-resume block, found {count}. "
            "Duplicate blocks re-introduce the UnboundLocalError risk."
        )


def test_episode_offset_assigned_before_env_constructor():
    """Core regression: _episode_offset must be assigned before CityLearn*Env(...)."""
    for name, checks in ORDERING_CHECKS.items():
        src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        lines = src.splitlines()

        # Find env constructor line
        env_line = None
        for i, line in enumerate(lines, 1):
            if re.search(checks["env_pattern"], line.strip()):
                env_line = i
                break

        assert env_line is not None, f"{name}: env constructor pattern not found"

        for var in checks["must_precede"]:
            # Find first assignment of this var (word-boundary match, not substring)
            first_assign = None
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Assignment: var = ... at start of stripped line
                if re.match(rf"^{re.escape(var)}\s*=", stripped):
                    first_assign = i
                    break

            assert first_assign is not None, f"{name}: {var} never assigned"
            assert first_assign < env_line, (
                f"{name}: {var} assigned at L{first_assign} but env constructor "
                f"is at L{env_line} — UnboundLocalError will occur on fresh runs!"
            )


def test_matd3_no_duplicate_env_load():
    """MATD3 must not create a second CityLearnOffPolicyVecEnv (eval_env) since
    use_eval=False — loading the dataset twice per job wastes 8-12 GiB RAM × 3 jobs."""
    src = (SCRIPTS_DIR / "train_citylearn_v3_matd3.py").read_text(encoding="utf-8")
    # Count CityLearnOffPolicyVecEnv(...) constructor calls
    count = src.count("CityLearnOffPolicyVecEnv(")
    assert count == 1, (
        f"train_citylearn_v3_matd3.py: expected 1 CityLearnOffPolicyVecEnv constructor call, "
        f"found {count}. A second eval_env loads the full dataset twice per job — ~24-36 GiB wasted RAM."
    )
    # Also check that eval_env is reused (not allocated anew)
    assert "eval_env = env" in src, (
        "eval_env should be set to 'env' (reuse) when use_eval=False, not a new env."
    )


def test_ast_no_load_before_store_in_main():
    """AST-level check: no _ variable used before its first assignment in main()."""
    for name in SCRIPTS:
        src = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        tree = ast.parse(src)

        main_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                main_fn = node
                break

        assert main_fn is not None, f"{name}: no main() found"

        first_store: dict[str, int] = {}
        first_load: dict[str, int] = {}

        for node in ast.walk(main_fn):
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                lineno = node.lineno
                if isinstance(node.ctx, ast.Store):
                    if node.id not in first_store:
                        first_store[node.id] = lineno
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in first_load:
                        first_load[node.id] = lineno

        for var, use_line in first_load.items():
            if var in first_store:
                assign_line = first_store[var]
                assert assign_line <= use_line, (
                    f"{name}: {var} first used at L{use_line} but first assigned "
                    f"at L{assign_line} — potential UnboundLocalError!"
                )


def test_masac_gpu_buffer_installed():
    """MASAC train script must call install_gpu_replay_buffer after Runner init.

    When preload_batch_device=cuda, the 13.7 GiB numpy replay buffer must be
    moved to GPU VRAM to prevent system RAM OOM on 12-job concurrent runs.
    """
    src = (SCRIPTS_DIR / "train_citylearn_v3_masac.py").read_text(encoding="utf-8")

    assert "install_gpu_replay_buffer" in src, (
        "train_citylearn_v3_masac.py must import and call install_gpu_replay_buffer "
        "to move the MASAC replay buffer (~13.7 GiB) from system RAM to GPU VRAM."
    )
    assert "install_gpu_replay_buffer(runner" in src, (
        "install_gpu_replay_buffer must be called with 'runner' as the first argument "
        "after Runner(env, backend_args) is initialized."
    )

    # Verify that the call happens AFTER Runner initialization
    lines = src.splitlines()
    runner_init_line = None
    gpu_buf_call_line = None
    for i, line in enumerate(lines, 1):
        if "runner = Runner(env, backend_args)" in line:
            runner_init_line = i
        if "install_gpu_replay_buffer(runner" in line:
            gpu_buf_call_line = i

    assert runner_init_line is not None, "Runner(env, backend_args) not found"
    assert gpu_buf_call_line is not None, "install_gpu_replay_buffer(runner) call not found"
    assert gpu_buf_call_line > runner_init_line, (
        f"install_gpu_replay_buffer (L{gpu_buf_call_line}) must come AFTER "
        f"Runner initialization (L{runner_init_line})."
    )


def test_masac_runtime_optimizations_has_gpu_buffer():
    """masac_runtime_optimizations.py must export GpuBackedNdArray and install_gpu_replay_buffer."""
    src = (SCRIPTS_DIR / "masac_runtime_optimizations.py").read_text(encoding="utf-8")

    assert "class GpuBackedNdArray" in src, (
        "masac_runtime_optimizations.py must define GpuBackedNdArray — "
        "a numpy-compatible array backed by a CUDA tensor for system RAM reduction."
    )
    assert "def install_gpu_replay_buffer" in src, (
        "masac_runtime_optimizations.py must define install_gpu_replay_buffer."
    )
    assert "__setitem__" in src, (
        "GpuBackedNdArray must implement __setitem__ to accept numpy writes transparently."
    )
    assert "__getitem__" in src, (
        "GpuBackedNdArray must implement __getitem__ to return CUDA tensors for batch sampling."
    )


def test_launcher_masac_uses_cuda_preload():
    """Concurrent launcher must use preload_batch_device=cuda for MASAC GPU buffer offload."""
    launcher_path = SCRIPTS_DIR / "colab_a100_official_launcher.py"
    src = launcher_path.read_text(encoding="utf-8")

    # Find the two_phase_concurrent MASAC patching block
    assert '"--masac-preload-batch-device", "cuda"' in src, (
        "colab_a100_official_launcher.py must patch MASAC with preload_batch_device=cuda "
        "in run_two_phase_concurrent_jobs so the GPU buffer is activated. "
        "Found 'cpu' instead — the 41 GiB MASAC buffer stays in system RAM → OOM!"
    )
