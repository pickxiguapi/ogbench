"""Low-overhead, synchronized inference timing for LeWM controllers."""

from __future__ import annotations

import functools
import platform
import statistics
import time
from collections import defaultdict

import jax
import numpy as np


def _block_until_ready(value):
    """Synchronize every asynchronous JAX leaf before stopping a timer."""
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, 'block_until_ready')
        else leaf,
        value,
    )


def _percentile(values, percentile):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _distribution(values, drop_first=0):
    values = [float(value) for value in values]
    steady = values[int(drop_first):]
    return {
        'count': len(values),
        'cold_start_count': min(int(drop_first), len(values)),
        'cold_start_ms': [value * 1_000.0 for value in values[:drop_first]],
        'steady_count': len(steady),
        'steady_mean_ms': (
            statistics.fmean(steady) * 1_000.0 if steady else None
        ),
        'steady_std_ms': (
            statistics.stdev(steady) * 1_000.0 if len(steady) > 1 else 0.0
        ),
        'steady_median_ms': (
            statistics.median(steady) * 1_000.0 if steady else None
        ),
        'steady_p95_ms': (
            _percentile(steady, 95.0) * 1_000.0 if steady else None
        ),
        'steady_min_ms': min(steady) * 1_000.0 if steady else None,
        'steady_max_ms': max(steady) * 1_000.0 if steady else None,
        'sample_ms': [value * 1_000.0 for value in values],
    }


class LeWMInferenceProfiler:
    """Attach timers to one LeWM-CEM policy without changing its outputs."""

    _DROP_FIRST = {
        # History and goal encoding use different leading dimensions and each
        # triggers one JIT compilation.
        'subgoal_encoder': 2,
        'latent_path_flow': 1,
        'subgoal_total': 1,
        'action_prior': 1,
        'cem': 1,
    }

    def __init__(self, policy):
        self.policy = policy
        self.samples = defaultdict(list)
        self.control_steps = []
        self._cem_calls = 0
        self._attach(policy)

    def _wrap(self, owner, attribute, label):
        function = getattr(owner, attribute)

        @functools.wraps(function)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            result = function(*args, **kwargs)
            result = _block_until_ready(result)
            elapsed = time.perf_counter() - started
            self.samples[label].append(elapsed)
            if label == 'cem':
                self._cem_calls += 1
            return result

        setattr(owner, attribute, timed)

    def _attach(self, policy):
        if hasattr(policy, 'local_policy'):
            raise ValueError(
                'Inference timing currently requires a single-stage LeWM policy.'
            )

        self._wrap(policy, '_plan_one', 'cem')
        if policy.guidance_policy is not None:
            self._wrap(policy, '_guidance_block', 'action_prior')

        generator = policy.subgoal_generator
        if generator is not None:
            self._wrap(generator, 'encode_pixels', 'subgoal_encoder')
            self._wrap(generator, '_predict', 'latent_path_flow')
            self._wrap(generator, 'predict_path', 'subgoal_total')

        get_actions = policy.get_actions

        @functools.wraps(get_actions)
        def timed_get_actions(pixels, goals, alive):
            alive_count = int(np.count_nonzero(alive))
            cem_before = self._cem_calls
            component_labels = ('subgoal_total', 'action_prior', 'cem')
            sample_counts_before = {
                label: len(self.samples.get(label, ()))
                for label in component_labels
            }
            started = time.perf_counter()
            result = get_actions(pixels, goals, alive)
            result = _block_until_ready(result)
            elapsed = time.perf_counter() - started
            component_seconds = sum(
                sum(self.samples.get(label, ())[sample_counts_before[label]:])
                for label in component_labels
            )
            self.control_steps.append(
                {
                    'elapsed_seconds': elapsed,
                    'alive_actions': alive_count,
                    'replans': self._cem_calls - cem_before,
                    'profiled_component_seconds': component_seconds,
                }
            )
            return result

        policy.get_actions = timed_get_actions

    def summary(self):
        module_summary = {
            label: _distribution(
                values,
                drop_first=self._DROP_FIRST.get(label, 0),
            )
            for label, values in sorted(self.samples.items())
        }

        replan_steps = [step for step in self.control_steps if step['replans']]
        top_level_labels = ('subgoal_total', 'action_prior', 'cem')
        steady_component_ms = sum(
            module_summary[label]['steady_mean_ms']
            for label in top_level_labels
            if label in module_summary
            and module_summary[label]['steady_mean_ms'] is not None
        )
        steady_replans = max(self._cem_calls - self._DROP_FIRST['cem'], 0)
        # Random-key helpers and other Python/JAX glue can also compile during
        # the first replan batch, outside the individually wrapped modules.
        # Drop that batch when estimating steady-state bookkeeping overhead.
        steady_other_steps = replan_steps[1:]
        other_replan_seconds = sum(
            max(
                0.0,
                step['elapsed_seconds'] - step['profiled_component_seconds'],
            )
            for step in steady_other_steps
        )
        steady_other_replans = sum(step['replans'] for step in steady_other_steps)
        other_replan_ms = (
            other_replan_seconds * 1_000.0 / steady_other_replans
            if steady_other_replans
            else 0.0
        )

        buffer_steps = [step for step in self.control_steps if not step['replans']]
        buffer_seconds = sum(step['elapsed_seconds'] for step in buffer_steps)
        buffer_actions = sum(step['alive_actions'] for step in buffer_steps)

        replan_ms = (
            steady_component_ms + other_replan_ms
            if steady_replans
            else None
        )
        buffer_action_ms = (
            buffer_seconds * 1_000.0 / buffer_actions if buffer_actions else 0.0
        )
        action_block = int(self.policy.action_block)
        amortized_action_ms = (
            (replan_ms + (action_block - 1) * buffer_action_ms) / action_block
            if replan_ms is not None
            else None
        )

        devices = jax.devices()
        return {
            'protocol': {
                'clock': 'time.perf_counter',
                'jax_synchronized': True,
                'cold_start_policy': (
                    'Report JIT compilation separately and exclude it from '
                    'steady-state distributions.'
                ),
                'python': platform.python_version(),
                'jax': jax.__version__,
                'devices': [
                    {
                        'platform': device.platform,
                        'device_kind': device.device_kind,
                        'id': int(device.id),
                    }
                    for device in devices
                ],
            },
            'counts': {
                'control_step_calls': len(self.control_steps),
                'alive_actions': sum(
                    step['alive_actions'] for step in self.control_steps
                ),
                'replan_events': self._cem_calls,
                'steady_replan_events': steady_replans,
            },
            'end_to_end': {
                'steady_replan_ms_per_environment': replan_ms,
                'steady_component_ms_per_environment': steady_component_ms,
                'other_replan_ms_per_environment': other_replan_ms,
                'buffer_action_ms_per_environment': buffer_action_ms,
                'steady_amortized_ms_per_environment_action': amortized_action_ms,
                'steady_actions_per_second': (
                    1_000.0 / amortized_action_ms
                    if amortized_action_ms not in (None, 0.0)
                    else None
                ),
                'replan_step_samples': replan_steps,
                'buffer_step_samples': buffer_steps,
            },
            'modules': module_summary,
        }
