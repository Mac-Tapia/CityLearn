"""Official MADRL source registry for thesis experiments.

This module does not reimplement algorithms. It records the external packages
or repositories that must back thesis-grade runs and exposes availability checks
so local prototype agents are not confused with official implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _candidate_external_roots(external_root: Path | str) -> tuple[Path, ...]:
    root = Path(external_root)

    if root.is_absolute():
        return (root,)

    return (
        Path.cwd() / root,
        PROJECT_ROOT / root,
        PROJECT_ROOT / "CityLearn" / root,
    )


@dataclass(frozen=True)
class OfficialMADRLSource:
    algorithm: str
    role: str
    paper_title: str
    paper_url: str
    repository_url: str
    backend_package: Optional[str]
    import_name: Optional[str]
    notes: str

    def is_available(self, external_root: Path | str = "external") -> bool:
        if self.import_name and find_spec(self.import_name) is not None:
            return True

        repo_name = self.repository_url.rstrip("/").split("/")[-1]
        repo_name = repo_name[:-4] if repo_name.endswith(".git") else repo_name

        return any((root / repo_name).exists() for root in _candidate_external_roots(external_root))


OFFICIAL_MADRL_SOURCES: Mapping[str, OfficialMADRLSource] = {
    "HAPPO": OfficialMADRLSource(
        algorithm="HAPPO",
        role="primary_official_backend",
        paper_title="Heterogeneous-Agent Reinforcement Learning",
        paper_url="http://jmlr.org/papers/v25/23-0488.html",
        repository_url="https://github.com/PKU-MARL/HARL",
        backend_package="harl",
        import_name="harl",
        notes=(
            "HARL is the official PyTorch implementation that includes HAPPO "
            "with heterogeneous policies and sequential agent updates."
        ),
    ),
    "MASAC": OfficialMADRLSource(
        algorithm="MASAC",
        role="paper_official_repository",
        paper_title="Decomposed Soft Actor-Critic Method for Cooperative Multi-Agent Reinforcement Learning",
        paper_url="https://arxiv.org/abs/2104.06655",
        repository_url="https://github.com/puyuan1996/MARL",
        backend_package=None,
        import_name=None,
        notes=(
            "The mSAC/MASAC paper points to this repository for its own "
            "implementation. XuanCe is a maintained modern alternative, but "
            "thesis-grade source parity should start from the paper repository."
        ),
    ),
    "MATD3": OfficialMADRLSource(
        algorithm="MATD3",
        role="paper_official_repository",
        paper_title="Reducing Overestimation Bias in Multi-Agent Domains Using Double Centralized Critics",
        paper_url="https://arxiv.org/abs/1910.01465",
        repository_url="https://github.com/JohannesAck/MATD3implementation",
        backend_package=None,
        import_name=None,
        notes=(
            "Author repository for MATD3. It targets TensorFlow 1.x/Gym 0.10; "
            "XuanCe provides a maintained PyTorch implementation for modern runs."
        ),
    ),
    "MATD3_PYTORCH": OfficialMADRLSource(
        algorithm="MATD3_PYTORCH",
        role="python39_pytorch_reference_backend",
        paper_title="Off-Policy Multi-Agent Reinforcement Learning Algorithms",
        paper_url="https://github.com/marlbenchmark/off-policy",
        repository_url="https://github.com/marlbenchmark/off-policy",
        backend_package=None,
        import_name=None,
        notes=(
            "marlbenchmark/off-policy provides PyTorch MLP MATD3 and recurrent "
            "RMATD3 implementations. It is not the original MATD3 author "
            "repository, but it is source-backed and imports in Python 3.9."
        ),
    ),
    "MAAC": OfficialMADRLSource(
        algorithm="MAAC",
        role="original_official_repository",
        paper_title="Actor-Attention-Critic for Multi-Agent Reinforcement Learning",
        paper_url="https://arxiv.org/abs/1810.02912",
        repository_url="https://github.com/shariqiqbal2810/MAAC",
        backend_package=None,
        import_name=None,
        notes=(
            "The original MAAC repository targets older PyTorch/Gym APIs; it "
            "must be vendored or isolated before thesis runs."
        ),
    ),
    "MARLLIB": OfficialMADRLSource(
        algorithm="MARLLIB",
        role="framework_backend",
        paper_title="MARLlib: A Scalable and Efficient Multi-agent Reinforcement Learning Library",
        paper_url="https://jmlr.org/papers/v24/23-0378.html",
        repository_url="https://github.com/Replicable-MARL/MARLlib",
        backend_package="marllib",
        import_name="marllib",
        notes=(
            "MARLlib is the requested Ray/RLlib-based MARL framework. Its "
            "upstream documentation states Linux compatibility for the official setup."
        ),
    ),
}


def official_backend_status(
    algorithms: Optional[Iterable[str]] = None,
    *,
    external_root: Path | str = "external",
) -> Dict[str, Dict[str, object]]:
    selected = [name.upper() for name in algorithms] if algorithms is not None else list(OFFICIAL_MADRL_SOURCES)
    status: Dict[str, Dict[str, object]] = {}

    for name in selected:
        source = OFFICIAL_MADRL_SOURCES[name]
        status[name] = {
            "available": source.is_available(external_root),
            "role": source.role,
            "backend_package": source.backend_package,
            "repository_url": source.repository_url,
            "paper_url": source.paper_url,
            "notes": source.notes,
        }

    return status


def missing_official_backends(
    algorithms: Optional[Iterable[str]] = None,
    *,
    external_root: Path | str = "external",
) -> Dict[str, OfficialMADRLSource]:
    selected = [name.upper() for name in algorithms] if algorithms is not None else list(OFFICIAL_MADRL_SOURCES)

    return {
        name: OFFICIAL_MADRL_SOURCES[name]
        for name in selected
        if not OFFICIAL_MADRL_SOURCES[name].is_available(external_root)
    }


def require_official_backends(
    algorithms: Optional[Iterable[str]] = None,
    *,
    external_root: Path | str = "external",
) -> None:
    missing = missing_official_backends(algorithms, external_root=external_root)

    if not missing:
        return

    details = ", ".join(
        f"{name} ({source.repository_url})"
        for name, source in missing.items()
    )
    raise ImportError(
        "Missing official MADRL backend(s). Install or vendor before thesis-grade runs: "
        f"{details}"
    )
