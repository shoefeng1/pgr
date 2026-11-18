"""DeepMind Control Suite -> Gymnasium adapter.

Replaces the unmaintained `gym` + `dmcgym` pair (and `mujoco-py`) used by the original
code. Env ids keep the old naming, e.g. `cheetah-run-v0`, `quadruped-walk-v0`,
`reacher-hard-v0`. Anything that is not a DMC task id falls through to `gymnasium.make`
(e.g. `Hopper-v4`, which needs `pip install "gymnasium[mujoco]"`).

The REDQ code in `synther/REDQ` still speaks the *classic* Gym API
(`reset() -> obs`, `step() -> (obs, reward, done, info)`, `env.seed(s)`), so `LegacyStepAPI`
presents a Gymnasium env that way. REDQ itself is left unmodified.
"""
import re
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_DMC_ID = re.compile(r"^(?P<domain>[A-Za-z0-9_]+)-(?P<task>[A-Za-z0-9_]+?)(?:-v\d+)?$")


def _spec_to_space(spec) -> spaces.Space:
    import dm_env  # imported lazily: only needed for DMC envs

    if isinstance(spec, Mapping):
        # Keep DMC's key order (OrderedDict is not re-sorted by gymnasium), so a
        # flattened observation has the same layout as with the original dmcgym.
        return spaces.Dict(OrderedDict((k, _spec_to_space(v)) for k, v in spec.items()))
    if isinstance(spec, dm_env.specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    if isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.integer):
            low, high = np.iinfo(spec.dtype).min, np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low, high = float("-inf"), float("inf")
        else:
            raise ValueError(f"Unsupported dtype in spec: {spec.dtype}")
        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    raise NotImplementedError(f"Unsupported spec type: {type(spec)}")


def _obs_to_numpy(obs):
    if isinstance(obs, Mapping):
        return OrderedDict((k, _obs_to_numpy(v)) for k, v in obs.items())
    return np.asarray(obs)


class DMCEnv(gym.Env):
    """A DeepMind Control Suite task as a Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self, domain_name: str, task_name: str):
        from dm_control import suite

        self._env = suite.load(domain_name=domain_name, task_name=task_name)
        self.action_space = _spec_to_space(self._env.action_spec())
        self.observation_space = _spec_to_space(self._env.observation_spec())
        # All benchmark tasks run 1000 control steps; read it from dm_control when possible.
        step_limit = getattr(self._env, "_step_limit", float("inf"))
        self.max_episode_steps = int(step_limit) if np.isfinite(step_limit) else 1000

    def seed(self, seed: int) -> None:
        if hasattr(self._env, "random_state"):
            self._env.random_state.seed(seed)
        else:
            self._env.task.random.seed(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        time_step = self._env.reset()
        return _obs_to_numpy(time_step.observation), {}

    def step(self, action):
        action = np.clip(np.asarray(action), self.action_space.low, self.action_space.high)
        time_step = self._env.step(action)
        reward = float(time_step.reward or 0.0)
        last = time_step.last()
        # DMC signals a time-limit cut-off with discount == 1.0 and a real terminal with 0.0.
        terminated = bool(last and time_step.discount == 0.0)
        truncated = bool(last and not terminated)
        return _obs_to_numpy(time_step.observation), reward, terminated, truncated, {}


def make_env(env_name: str) -> gym.Env:
    """`cheetah-run-v0` -> DMCEnv('cheetah', 'run'); anything else -> gymnasium.make."""
    match = _DMC_ID.match(env_name)
    if match:
        from dm_control import suite

        domain, task = match.group("domain"), match.group("task")
        if (domain, task) in set(suite.ALL_TASKS):
            return DMCEnv(domain, task)
    return gym.make(env_name)


class ClipToActionSpace(gym.ActionWrapper):
    """Clip actions to the Box bounds of the wrapped env *and keep those bounds* as the action space.

    Gymnasium>=1.0's own `ClipAction` advertises an unbounded action space, which would break
    `act_limit = env.action_space.high[0]` and make `action_space.sample()` unbounded.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = env.action_space

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class LegacyStepAPI:
    """Expose a Gymnasium env through the classic Gym API used by REDQ.

    reset() -> obs;  step(a) -> (obs, reward, done, info) with done = terminated or truncated,
    and info['truncated'] carrying that distinction back for callers that need it (e.g. to
    tell real episode boundaries apart from time-limit cutoffs when `done` alone can't).
    seed(s) takes effect at the next reset (which is when seeds are consumed anyway).
    Everything else (spaces, spec, unwrapped, ...) is forwarded to the wrapped env.
    """

    def __init__(self, env: gym.Env):
        self.env = env
        self._pending_seed: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        if name == "env":  # avoid infinite recursion before __init__ has run
            raise AttributeError(name)
        return getattr(self.env, name)

    def seed(self, seed: int) -> None:
        self._pending_seed = seed

    def reset(self):
        obs, _ = self.env.reset(seed=self._pending_seed)
        self._pending_seed = None
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info['truncated'] = truncated
        return obs, reward, terminated or truncated, info
