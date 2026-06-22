"""Tests for the MASAC OOM-retry buffer fix in colab_a100_official_launcher.py.

Regression: with the Iquitos 2023-2025 dataset Building_7 has 42 EV chargers,
expanding obs_shape to ~370 dims. buffer_size=10 estimated 13.72 GiB, which
exceeded the 12 GiB OOM-retry limit → the retry aborted before training.
Fix: buffer_size dropped to 8 (→ ~10.98 GiB) and hidden dims halved.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Pure formula replicated here — avoids importing the heavy masac train script
# (which pulls in torch / numpy at module level).
# Keep in sync with train_citylearn_v3_masac.py::estimate_replay_buffer_gib.
# ---------------------------------------------------------------------------

def _estimate_replay_buffer_gib(
    *, buffer_size, episode_limit, n_agents, n_actions, obs_shape, state_shape
):
    transition_elements = (
        n_agents * obs_shape
        + n_agents
        + state_shape
        + 1
        + n_agents * obs_shape
        + state_shape
        + n_agents * n_actions
        + n_agents * n_actions
        + n_agents * n_actions
        + 1
        + 1
    )
    bytes_required = int(buffer_size) * int(episode_limit) * int(transition_elements) * 8
    return bytes_required / float(1024**3)


# Iquitos 2023-2025 dimensions that reproduce the exact Colab error (13.72 GiB @ buffer_size=10).
# Derived from the error message and the launcher comment (buffer_size=10 → ~13.7 GiB each):
#   transition_elements ≈ 21021
#   = 2*17*370 + 17 + 2*4134 + 3*17*3 + 3
# n_actions=3 comes from axis mode (3 bins) with 1 battery actuator per building.
# obs_shape=370 from the launcher comment: 7 templates × 42 chargers + 31 base + 42 type codes.
IQUITOS_KWARGS = dict(
    episode_limit=8760,
    n_agents=17,
    n_actions=3,       # axis mode, 3 bins, 1 battery actuator per building
    obs_shape=370,     # 7 obs templates × 42 EV chargers + 31 base + 42 type codes
    state_shape=4134,  # calibrated so buffer_size=10 → 13.72 GiB (matches Colab error)
)


# ---------------------------------------------------------------------------
# Launcher import — colab_a100_official_launcher.py only uses stdlib so no stubs needed
# ---------------------------------------------------------------------------

def _import_launcher():
    spec = importlib.util.spec_from_file_location(
        "colab_a100_official_launcher",
        SCRIPTS_DIR / "colab_a100_official_launcher.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return _import_launcher()


# ---------------------------------------------------------------------------
# Formula correctness tests
# ---------------------------------------------------------------------------

def test_buffer_size_10_exceeds_12gib():
    """Regression anchor: buffer_size=10 was too large for the OOM retry limit."""
    gib = _estimate_replay_buffer_gib(buffer_size=10, **IQUITOS_KWARGS)
    assert gib > 12.0, f"Expected >12 GiB with buffer_size=10, got {gib:.2f} GiB"


def test_buffer_size_8_fits_within_12gib():
    """The fixed buffer_size=8 must pass the 12 GiB pre-flight check."""
    gib = _estimate_replay_buffer_gib(buffer_size=8, **IQUITOS_KWARGS)
    assert gib <= 12.0, f"Expected <=12 GiB with buffer_size=8, got {gib:.2f} GiB"


def test_buffer_size_8_fits_within_20gib_default_limit():
    """Also passes the default (non-retry) 20 GiB limit."""
    gib = _estimate_replay_buffer_gib(buffer_size=8, **IQUITOS_KWARGS)
    assert gib <= 20.0, f"Expected <=20 GiB with buffer_size=8, got {gib:.2f} GiB"


def test_buffer_gib_scales_linearly_with_buffer_size():
    """Estimate must be proportional to buffer_size (formula sanity)."""
    g8 = _estimate_replay_buffer_gib(buffer_size=8, **IQUITOS_KWARGS)
    g10 = _estimate_replay_buffer_gib(buffer_size=10, **IQUITOS_KWARGS)
    ratio = g10 / g8
    assert abs(ratio - 10 / 8) < 1e-9, f"Linear scaling broken: ratio={ratio}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_masac_job(scenario="E2"):
    return {
        "name": "masac",
        "scenario": scenario,
        "script": "CityLearn/scripts/train_citylearn_v3_masac.py",
        "args": [
            "--scenario", scenario,
            "--episodes", "50",
            "--buffer-size", "10",
            "--critic-batch-size", "64",
            "--max-replay-buffer-gib", "20.0",
            "--masac-preload-batch-device", "auto",
            "--rnn-hidden-dim", "256",
            "--qmix-hidden-dim", "128",
            "--hyper-hidden-dim", "256",
        ],
    }


def _arg_value(args, flag):
    args = [str(a) for a in args]
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return None


# ---------------------------------------------------------------------------
# make_oom_retry_job tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["E1", "E2", "E3"])
def test_oom_retry_buffer_size_is_8(launcher, scenario):
    """Retry job must use buffer_size=8 (was 10, causing 13.72 GiB > 12 GiB)."""
    retry = launcher.make_oom_retry_job(_make_base_masac_job(scenario))
    assert retry is not None
    assert _arg_value(retry["args"], "--buffer-size") == "8"


@pytest.mark.parametrize("scenario", ["E1", "E2", "E3"])
def test_oom_retry_max_replay_buffer_gib_is_12(launcher, scenario):
    retry = launcher.make_oom_retry_job(_make_base_masac_job(scenario))
    assert _arg_value(retry["args"], "--max-replay-buffer-gib") == "12"


@pytest.mark.parametrize("scenario", ["E1", "E2", "E3"])
def test_oom_retry_preload_device_is_cpu(launcher, scenario):
    retry = launcher.make_oom_retry_job(_make_base_masac_job(scenario))
    assert _arg_value(retry["args"], "--masac-preload-batch-device") == "cpu"


@pytest.mark.parametrize("scenario", ["E1", "E2", "E3"])
def test_oom_retry_critic_batch_size_reduced(launcher, scenario):
    retry = launcher.make_oom_retry_job(_make_base_masac_job(scenario))
    assert int(_arg_value(retry["args"], "--critic-batch-size")) <= 32


@pytest.mark.parametrize("scenario", ["E1", "E2", "E3"])
def test_oom_retry_rnn_hidden_dim_reduced(launcher, scenario):
    """Halved hidden dim reduces GPU pressure when 12 jobs share one A100."""
    retry = launcher.make_oom_retry_job(_make_base_masac_job(scenario))
    assert int(_arg_value(retry["args"], "--rnn-hidden-dim")) <= 128


@pytest.mark.parametrize("scenario", ["E1", "E2", "E3"])
def test_oom_retry_qmix_hidden_dim_reduced(launcher, scenario):
    retry = launcher.make_oom_retry_job(_make_base_masac_job(scenario))
    assert int(_arg_value(retry["args"], "--qmix-hidden-dim")) <= 64


@pytest.mark.parametrize("scenario", ["E1", "E2", "E3"])
def test_oom_retry_buffer8_passes_12gib_check(launcher, scenario):
    """End-to-end: retry buffer_size=8 must satisfy --max-replay-buffer-gib 12."""
    retry = launcher.make_oom_retry_job(_make_base_masac_job(scenario))
    buf = int(_arg_value(retry["args"], "--buffer-size"))
    limit = float(_arg_value(retry["args"], "--max-replay-buffer-gib"))
    gib = _estimate_replay_buffer_gib(buffer_size=buf, **IQUITOS_KWARGS)
    assert gib <= limit, (
        f"Retry estimate {gib:.2f} GiB exceeds limit {limit:.2f} GiB (scenario={scenario})"
    )


def test_non_masac_happo_returns_none(launcher):
    """make_oom_retry_job returns None for algorithms without a retry recipe."""
    job = {
        "name": "happo",
        "scenario": "E1",
        "script": "CityLearn/scripts/train_citylearn_v3_happo.py",
        "args": ["--scenario", "E1"],
    }
    assert launcher.make_oom_retry_job(job) is None


def test_replace_arg_inserts_new_flag(launcher):
    result = launcher.replace_arg(["--foo", "1"], "--bar", "2")
    assert "--bar" in result and result[result.index("--bar") + 1] == "2"


def test_replace_arg_updates_existing_flag(launcher):
    result = launcher.replace_arg(["--foo", "old"], "--foo", "new")
    assert result.count("--foo") == 1
    assert result[result.index("--foo") + 1] == "new"
