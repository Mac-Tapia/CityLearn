"""Save and restore full training state: optimizers, RNG seeds, and replay buffers."""
from __future__ import annotations

import functools
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

TRAINING_STATE_FILENAME = 'training_state.pt'
TRAINING_STATE_SCHEMA = 'madrl.training_state.v1'


class TrainingStateError(RuntimeError):
    """A resume cannot continue faithfully from the stored training state."""


def save_training_state(
    path: Path,
    optimizers: List[torch.optim.Optimizer],
    rng_states: Dict[str, object],
    replay: Optional[dict] = None,
) -> None:
    state = {
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'rng_states': rng_states,
        'replay': replay,
    }
    torch.save(state, path)


def load_training_state(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def capture_rng_states() -> Dict[str, Any]:
    states: Dict[str, Any] = {
        'torch': torch.random.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(states['torch'])
    np.random.set_state(states['numpy'])
    random.setstate(states['python'])
    if 'torch_cuda' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda'])


def _owned_optimizers(specs: Sequence[Mapping[str, Any]]) -> Dict[str, torch.optim.Optimizer]:
    return {
        str(spec['owner']): spec['optimizer']
        for spec in specs
        if spec.get('optimizer') is not None
    }


def save_owner_training_state(
    directory: Path,
    specs: Sequence[Mapping[str, Any]],
    *,
    backend: str,
) -> Path:
    """Write optimizer states keyed by owner plus RNG states next to the model weights."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    optimizers = _owned_optimizers(specs)
    payload = {
        'schema': TRAINING_STATE_SCHEMA,
        'backend': backend,
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'optimizer_states': {owner: opt.state_dict() for owner, opt in optimizers.items()},
        'rng_states': capture_rng_states(),
        # Off-policy replay buffers (multi-GiB) are not persisted; resumes refill them via warmup.
        'replay': None,
    }
    target = directory / TRAINING_STATE_FILENAME
    tmp = target.with_suffix('.pt.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return target


def restore_owner_training_state(
    directory: Path,
    specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Load optimizer and RNG states into live objects; fail closed on any mismatch."""
    path = Path(directory) / TRAINING_STATE_FILENAME
    if not path.is_file():
        raise TrainingStateError(f'resume requires {path} (optimizer/RNG state) but it is missing')
    payload = load_training_state(path)
    if payload.get('schema') != TRAINING_STATE_SCHEMA:
        raise TrainingStateError(f'{path} has unsupported schema {payload.get("schema")!r}')
    stored = payload['optimizer_states']
    optimizers = _owned_optimizers(specs)
    if set(stored) != set(optimizers):
        raise TrainingStateError(
            f'{path} owners {sorted(stored)} do not match live optimizers {sorted(optimizers)}'
        )
    for owner, optimizer in optimizers.items():
        optimizer.load_state_dict(stored[owner])
    restore_rng_states(payload['rng_states'])
    return {
        'path': str(path),
        'backend': payload.get('backend'),
        'saved_at': payload.get('saved_at'),
        'restored_owners': sorted(optimizers),
        'replay_restored': False,
    }


def wrap_save_with_training_state(
    owner_obj: Any,
    method_name: str,
    directory_getter: Callable[[], Path],
    specs_getter: Callable[[], Sequence[Mapping[str, Any]]],
    *,
    backend: str,
) -> None:
    """Make every backend weight save also write the matching training state."""
    original = getattr(owner_obj, method_name)

    @functools.wraps(original)
    def save_with_state(*args, **kwargs):
        result = original(*args, **kwargs)
        save_owner_training_state(directory_getter(), specs_getter(), backend=backend)
        return result

    setattr(owner_obj, method_name, save_with_state)
