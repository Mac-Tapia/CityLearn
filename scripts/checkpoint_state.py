"""Save and restore full training state: optimizers, RNG seeds, and replay buffers."""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


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
