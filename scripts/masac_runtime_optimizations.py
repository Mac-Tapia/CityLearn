"""Runtime optimizations for the external MARL MASAC backend.

The upstream MASAC implementation is kept in ``external/MARL``.  This module
patches its learner class at runtime so the project remains reproducible
without committing local-only changes inside the external submodule.

GPU replay buffer offload
─────────────────────────
The external MARL buffer pre-allocates ``buffer_size × episode_limit × *dims``
numpy float64 arrays in system RAM (~13.7 GiB per MASAC job with default args).
With 3 concurrent MASAC jobs that equals ~41 GiB of system RAM, causing OOM
on A100 Colab (167 GiB limit) when 12 jobs run simultaneously.

``install_gpu_replay_buffer`` replaces those numpy arrays with
``GpuBackedNdArray`` objects that store data as float32 CUDA tensors,
moving the 41 GiB from system RAM to GPU VRAM (80 GiB A100, ≈62% used).
The replacement is transparent: numpy-style indexing still works, and
``_prepare_batch`` receives CUDA tensors directly (zero extra copy during
training updates).
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PATCH_VERSION = "citylearn_masac_runtime_v1"
_GPU_BUF_PATCH_VERSION = "citylearn_masac_gpu_buf_v1"
_GPU_BUF_MIN_BYTES = 50 * 1024 * 1024   # only migrate arrays ≥ 50 MiB


# ── GPU-backed numpy-compatible array ──────────────────────────────────────

class GpuBackedNdArray:
    """Drop-in for a pre-allocated numpy float64 array stored as float32 CUDA tensor.

    The external MARL buffer uses numpy arrays as ring buffers, writing to
    them with slice assignment (``buf['o'][ep_idx, step_idx] = obs``) and
    reading with fancy indexing (``buf['o'][sample_indices]``).

    This class intercepts those operations transparently:
    - Writes: numpy → CPU tensor → CUDA tensor (non-blocking)
    - Reads: returns CUDA tensor directly (zero-copy when training on GPU)

    Only float-like dtypes are migrated; integer fields (padded, terminated)
    are left as numpy arrays since they are tiny (<< 1 MiB) and frequently
    used in numpy comparisons.
    """

    __slots__ = ("_tensor", "shape", "dtype", "ndim", "_device")

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.shape = tuple(tensor.shape)
        self.dtype = np.float32
        self.ndim = tensor.ndim
        self._device = tensor.device

    @classmethod
    def from_numpy(cls, arr: np.ndarray, device: torch.device) -> "GpuBackedNdArray":
        t = torch.as_tensor(arr.astype(np.float32, copy=False), device=device)
        return cls(t)

    @classmethod
    def zeros(cls, shape: tuple, device: torch.device) -> "GpuBackedNdArray":
        return cls(torch.zeros(shape, dtype=torch.float32, device=device))

    # ── numpy-compatible writes ──
    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, np.ndarray):
            src = torch.as_tensor(value.astype(np.float32, copy=False))
            self._tensor[key].copy_(src, non_blocking=True)
        elif isinstance(value, torch.Tensor):
            self._tensor[key].copy_(
                value.to(dtype=torch.float32, non_blocking=True),
                non_blocking=True,
            )
        else:
            self._tensor[key] = float(value)

    # ── numpy-compatible reads ──
    def __getitem__(self, key: Any) -> torch.Tensor:
        return self._tensor[key]

    def __len__(self) -> int:
        return self.shape[0]

    # Allow numpy ufuncs and np.array() to fall back to CPU copy
    def __array__(self, dtype: Any = None) -> np.ndarray:
        arr = self._tensor.cpu().numpy()
        return arr.astype(dtype) if dtype is not None else arr

    def __repr__(self) -> str:
        return f"GpuBackedNdArray(shape={self.shape}, device={self._device})"


# ── Buffer migration ────────────────────────────────────────────────────────

def _is_large_float_ndarray(value: Any) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.floating)
        and value.nbytes >= _GPU_BUF_MIN_BYTES
    )


def _migrate_dict(data: dict, device: torch.device) -> dict[str, int]:
    """Replace large float numpy arrays in a dict with GpuBackedNdArray."""
    migrated: dict[str, int] = {}
    for key in list(data.keys()):
        val = data[key]
        if _is_large_float_ndarray(val):
            nbytes = val.nbytes
            shape = val.shape
            # Drop the dict's reference BEFORE allocating GPU memory so the
            # original float64 array (13.7 GiB per job) is freed immediately
            # by Python's reference counter. Buffer starts filled with zeros
            # at epoch 0, so we don't need to copy the existing data.
            data[key] = None
            data[key] = GpuBackedNdArray.zeros(shape, device)
            migrated[str(key)] = nbytes
    return migrated


def _migrate_obj(obj: Any, device: torch.device, depth: int = 0) -> dict[str, int]:
    """Recursively replace large float numpy arrays on an object with GpuBackedNdArray."""
    migrated: dict[str, int] = {}
    if depth > 4 or obj is None:
        return migrated
    seen_ids: set[int] = set()

    def _walk(o: Any, d: int) -> None:
        if d > 4 or id(o) in seen_ids:
            return
        seen_ids.add(id(o))

        target = None
        if isinstance(o, dict):
            target = o
        elif hasattr(o, "__dict__"):
            target = vars(o)
        else:
            return

        for key in list(target.keys()):
            val = target[key]
            if _is_large_float_ndarray(val):
                nbytes = val.nbytes
                shape = val.shape
                target[key] = None   # free float64 before GPU alloc
                target[key] = GpuBackedNdArray.zeros(shape, device)
                migrated[f"{type(o).__name__}.{key}"] = nbytes
            elif isinstance(val, dict) and any(
                _is_large_float_ndarray(v) for v in val.values()
            ):
                migrated.update(_migrate_dict(val, device))
            elif val is not None and not isinstance(val, (str, int, float, bool, type)):
                _walk(val, d + 1)

    _walk(obj, 0)
    return migrated


def install_gpu_replay_buffer(runner: Any, device: torch.device) -> dict:
    """Migrate the external MARL replay buffer from system RAM to GPU VRAM.

    Walks the runner's object graph, finds pre-allocated numpy float arrays
    (≥50 MiB), and replaces them with GpuBackedNdArray tensors on ``device``.
    The replacement is transparent: the buffer's store/sample code continues
    to use numpy-style slice assignment/indexing without modification.

    Returns a metadata dict with ``{"enabled": bool, "migrated_bytes": int, ...}``.
    """
    if not torch.cuda.is_available():
        return {"enabled": False, "reason": "CUDA not available"}

    try:
        migrated = _migrate_obj(runner, device)
        # Force Python GC so any remaining references to the original float64
        # arrays (freed by setting dict/attr slots to None before GPU alloc)
        # are collected immediately rather than on the next GC cycle.
        gc.collect()
        total_bytes = sum(migrated.values())
        total_gib = total_bytes / 1024**3
        result = {
            "enabled": True,
            "patch_version": _GPU_BUF_PATCH_VERSION,
            "device": str(device),
            "migrated_arrays": migrated,
            "migrated_gib": round(total_gib, 2),
            "migrated_count": len(migrated),
        }
        print(
            f"[masac_gpu_buf] Moved {len(migrated)} replay buffer arrays "
            f"({total_gib:.2f} GiB float64 freed from RAM → {device} as float32. "
            f"Arrays: {list(migrated.keys())[:6]}",
            flush=True,
        )
        return result
    except Exception as exc:
        print(
            f"[masac_gpu_buf] WARNING: GPU buffer migration failed ({exc}). "
            f"Buffer remains in system RAM.",
            file=sys.stderr,
            flush=True,
        )
        return {"enabled": False, "reason": str(exc)}


def _device_for(args: Any) -> torch.device:
    if bool(getattr(args, "cuda", False)) and torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def _move_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype, non_blocking=True)

    return torch.as_tensor(value, dtype=dtype, device=device)


def _prepare_batch(self: Any, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert a sampled replay batch once, with CUDA fallback for 8GB GPUs."""

    target_device = _device_for(self.args)
    mode = str(getattr(self.args, "citylearn_preload_batch_device", "auto")).lower()
    prefer_cuda = target_device.type == "cuda" and mode != "cpu"
    device = target_device if prefer_cuda else torch.device("cpu")

    def convert(convert_device: torch.device) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            dtype = torch.long if key == "u" else torch.float32
            converted[key] = _move_tensor(value, dtype=dtype, device=convert_device)
        return converted

    if device.type == "cuda":
        try:
            prepared = convert(device)
            self._citylearn_batch_device = "cuda"
            return prepared
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or mode == "cuda":
                raise
            torch.cuda.empty_cache()

    prepared = convert(torch.device("cpu"))
    self._citylearn_batch_device = "cpu"
    return prepared


def _ensure_device(self: Any, value: torch.Tensor) -> torch.Tensor:
    device = _device_for(self.args)
    if value.device == device:
        return value

    return value.to(device=device, non_blocking=True)


def _agent_identity(self: Any, episode_num: int, like: torch.Tensor) -> torch.Tensor:
    key = (str(like.device), str(like.dtype), int(self.args.n_agents), int(episode_num))
    cache = getattr(self, "_citylearn_agent_identity_cache", None)
    if cache is None:
        cache = {}
        self._citylearn_agent_identity_cache = cache

    if key not in cache:
        cache[key] = torch.eye(
            self.args.n_agents,
            device=like.device,
            dtype=like.dtype,
        ).unsqueeze(0).expand(episode_num, -1, -1)

    return cache[key]


def _get_inputs(self: Any, batch: dict[str, torch.Tensor], transition_idx: int):
    obs = batch["o"][:, transition_idx]
    obs_next = batch["o_next"][:, transition_idx]
    u_onehot = batch["u_onehot"]
    episode_num = obs.shape[0]

    inputs = [obs]
    inputs_next = [obs_next]

    if self.args.last_action:
        if transition_idx == 0:
            inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
        else:
            inputs.append(u_onehot[:, transition_idx - 1])
        inputs_next.append(u_onehot[:, transition_idx])

    if self.args.reuse_network:
        agent_id = _agent_identity(self, episode_num, obs)
        inputs.append(agent_id)
        inputs_next.append(agent_id)

    inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
    inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)

    if bool(getattr(self.args, "cuda", False)):
        inputs = _ensure_device(self, inputs)
        inputs_next = _ensure_device(self, inputs_next)

    return inputs, inputs_next


def _init_hidden(self: Any, episode_num: int) -> None:
    device = _device_for(self.args)
    shape = (episode_num, self.n_agents, self.args.rnn_hidden_dim)
    self.eval_hidden = torch.zeros(shape, device=device)
    self.target_hidden = torch.zeros(shape, device=device)
    self.eval_hidden_2 = torch.zeros(shape, device=device)
    self.target_hidden_2 = torch.zeros(shape, device=device)
    self.policy_hidden = torch.zeros(shape, device=device)


def _get_q_values(self: Any, batch: dict[str, torch.Tensor], max_episode_len: int):
    episode_num = batch["o"].shape[0]
    q_evals, q_targets = [], []
    for transition_idx in range(max_episode_len):
        inputs, inputs_next = self._get_inputs(batch, transition_idx)
        q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
        q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
        q_evals.append(q_eval.view(episode_num, self.n_agents, -1))
        q_targets.append(q_target.view(episode_num, self.n_agents, -1))

    return torch.stack(q_evals, dim=1), torch.stack(q_targets, dim=1)


def _get_q_values_2(self: Any, batch: dict[str, torch.Tensor], max_episode_len: int):
    episode_num = batch["o"].shape[0]
    q_evals, q_targets = [], []
    for transition_idx in range(max_episode_len):
        inputs, inputs_next = self._get_inputs(batch, transition_idx)
        q_eval, self.eval_hidden_2 = self.eval_rnn_2(inputs, self.eval_hidden_2)
        q_target, self.target_hidden_2 = self.target_rnn_2(inputs_next, self.target_hidden_2)
        q_evals.append(q_eval.view(episode_num, self.n_agents, -1))
        q_targets.append(q_target.view(episode_num, self.n_agents, -1))

    return torch.stack(q_evals, dim=1), torch.stack(q_targets, dim=1)


def _train_actor(self: Any, batch: dict[str, Any], max_episode_len: int, actor_sample_times=None):
    batch = _prepare_batch(self, batch)
    episode_num = batch["o"].shape[0]
    self.init_hidden(episode_num)

    s = _ensure_device(self, batch["s"]) if self.args.cuda else batch["s"]
    terminated = batch["terminated"]
    mask = 1 - batch["padded"].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    mask = mask.repeat(1, 1, self.n_agents)
    actions = _ensure_device(self, batch["u"]) if self.args.cuda else batch["u"]
    avail_u = _ensure_device(self, batch["avail_u"]) if self.args.cuda else batch["avail_u"]
    mask = _ensure_device(self, mask) if self.args.cuda else mask

    q_evals = []
    actions_prob = []
    actions_logprobs = []
    for transition_idx in range(max_episode_len):
        inputs, _ = self._get_inputs(batch, transition_idx)
        q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
        q_eval = q_eval.view(episode_num, self.n_agents, -1)
        q_evals.append(q_eval)

        agent_outs, self.policy_hidden = self.agent.policy(inputs, self.policy_hidden)
        avail_actions = avail_u[:, transition_idx]
        reshaped_avail_actions = avail_actions.reshape(episode_num * self.n_agents, -1)
        agent_outs[reshaped_avail_actions == 0] = -1e11
        agent_outs = F.softmax(agent_outs, dim=1).view(episode_num, self.n_agents, -1)
        actions_prob.append(agent_outs)
        actions_logprobs.append(torch.log(agent_outs + (agent_outs == 0.0).float() * 1e-8))

    del actions
    q_vals = torch.stack(q_evals, dim=1)
    actions_prob = torch.stack(actions_prob, dim=1)
    log_prob_pi = torch.stack(actions_logprobs, dim=1)

    q_i_mean_negi_mean = torch.sum(actions_prob * (self.alpha * log_prob_pi - q_vals), dim=-1)
    q_i_mean_negi_mean = self.eval_qmix_net(q_i_mean_negi_mean, s)
    q_i_mean_negi_mean = q_i_mean_negi_mean.repeat(repeats=(1, 1, self.n_agents))
    policy_loss = (q_i_mean_negi_mean * mask).sum() / mask.sum()

    self.policy_optimiser.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_params, self.args.grad_norm_clip)
    self.policy_optimiser.step()
    self.soft_update()


def _train_critic(self: Any, batch: dict[str, Any], max_episode_len: int, train_step: int, epsilon=None):
    batch = _prepare_batch(self, batch)
    episode_num = batch["o"].shape[0]
    self.init_hidden(episode_num)

    s = batch["s"]
    s_next = batch["s_next"]
    u = batch["u"]
    r = batch["r"]
    terminated = batch["terminated"]
    mask = 1 - batch["padded"].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    mask = mask.repeat(1, 1, self.n_agents)
    r = 10.0 * (r - r.mean()) / (r.std() + 1.0e-6)

    q_evals, q_targets = self.get_q_values(batch, max_episode_len)
    q_evals_2, q_targets_2 = self.get_q_values_2(batch, max_episode_len)

    if self.args.cuda:
        s = _ensure_device(self, s)
        s_next = _ensure_device(self, s_next)
        u = _ensure_device(self, u)
        r = _ensure_device(self, r)
        terminated = _ensure_device(self, terminated)
        mask = _ensure_device(self, mask)
        avail_u = _ensure_device(self, batch["avail_u"])
    else:
        avail_u = batch["avail_u"]

    q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
    q_evals_2 = torch.gather(q_evals_2, dim=3, index=u).squeeze(3)

    actions_prob = []
    actions_logprobs = []
    for transition_idx in range(max_episode_len):
        _, inputs_next = self._get_inputs(batch, transition_idx)
        agent_outs, self.policy_hidden = self.agent.policy(inputs_next, self.policy_hidden)
        avail_actions = avail_u[:, transition_idx]
        reshaped_avail_actions = avail_actions.reshape(episode_num * self.n_agents, -1)
        agent_outs[reshaped_avail_actions == 0] = -1e11
        agent_outs = F.softmax(agent_outs, dim=1).view(episode_num, self.n_agents, -1)
        actions_prob.append(agent_outs)
        actions_logprobs.append(torch.log(agent_outs + (agent_outs == 0.0).float() * 1e-8))

    actions_prob = torch.stack(actions_prob, dim=1)
    log_prob_pi = torch.stack(actions_logprobs, dim=1)

    target_entropy = -1.0 * self.n_actions
    if self.args.auto_entropy is True:
        alpha_loss = (
            torch.sum(
                actions_prob.detach() * (-self.log_alpha * (log_prob_pi + target_entropy).detach()),
                dim=-1,
            )
            * mask
        ).sum() / mask.sum()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
    else:
        self.alpha = 1.0

    q_targets_sample = torch.sum(actions_prob * (q_targets - self.alpha * log_prob_pi), dim=-1).view(
        episode_num,
        max_episode_len,
        -1,
    ).detach()
    q_targets_sample_2 = torch.sum(actions_prob * (q_targets_2 - self.alpha * log_prob_pi), dim=-1).view(
        episode_num,
        max_episode_len,
        -1,
    ).detach()

    q_total_eval = self.eval_qmix_net(q_evals, s)
    q_total_target = self.target_qmix_net(q_targets_sample, s_next)
    q_total_eval_2 = self.eval_qmix_net_2(q_evals_2, s)
    q_total_target_2 = self.target_qmix_net_2(q_targets_sample_2, s_next)
    q_total_target = torch.min(q_total_target, q_total_target_2)
    targets = r + self.args.gamma * q_total_target * (1 - terminated)

    loss = ((mask * (q_total_eval - targets.detach())) ** 2).sum() / mask.sum()
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
    self.optimizer.step()

    loss_2 = ((mask * (q_total_eval_2 - targets.detach())) ** 2).sum() / mask.sum()
    self.optimizer_2.zero_grad()
    loss_2.backward()
    torch.nn.utils.clip_grad_norm_(self.eval_parameters_2, self.args.grad_norm_clip)
    self.optimizer_2.step()

    if train_step % 10000 == 0:
        self.save_model(train_step)


def install_masac_runtime_optimizations() -> dict[str, Any]:
    """Patch external MASAC learner methods in-place and return metadata."""

    from ac_discrete import qmix_msac

    cls = qmix_msac.QMIX_PG
    if getattr(cls, "_citylearn_patch_version", None) == PATCH_VERSION:
        return {
            "enabled": True,
            "patch_version": PATCH_VERSION,
            "already_installed": True,
        }

    cls._prepare_citylearn_batch = _prepare_batch
    cls._ensure_citylearn_device = _ensure_device
    cls._citylearn_agent_identity = _agent_identity
    cls._get_inputs = _get_inputs
    cls.init_hidden = _init_hidden
    cls.get_q_values = _get_q_values
    cls.get_q_values_2 = _get_q_values_2
    cls.train_actor = _train_actor
    cls.train_critic = _train_critic
    cls._citylearn_patch_version = PATCH_VERSION

    return {
        "enabled": True,
        "patch_version": PATCH_VERSION,
        "already_installed": False,
        "changes": [
            "sampled replay batches are tensorized once per update",
            "CUDA batch preload uses auto fallback to CPU after OOM",
            "agent identity matrix is cached per device/dtype/batch",
            "hidden-state allocation is created directly on target device",
            "per-timestep .cuda() calls are removed when batch preload succeeds",
        ],
    }

