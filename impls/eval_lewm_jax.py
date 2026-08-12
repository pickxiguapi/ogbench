"""Evaluate JAX LeWM with the reference dataset-goal CEM protocol."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import re
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from eval_lewm import json_safe, sample_eval_starts, task_spec

_HEADER = struct.Struct('!Q')


def checkpoint_epoch(path):
    match = re.search(r'weights_epoch_(\d+)\.msgpack$', str(path))
    if match is None:
        raise ValueError(f'Cannot infer epoch from checkpoint path: {path}')
    return int(match.group(1))


def _read_exact(stream, size):
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError('JAX LeWM worker exited unexpectedly.')
        chunks.append(chunk)
        size -= len(chunk)
    return b''.join(chunks)


def _send(stream, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _receive(stream):
    size = _HEADER.unpack(_read_exact(stream, _HEADER.size))[0]
    return pickle.loads(_read_exact(stream, size))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--task', choices=('cube', 'pusht', 'tworoom', 'reacher'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stable-wm-root', default='/root/data/yyf/stable-worldmodel')
    parser.add_argument('--ogbench-root', default='/root/data/yyf/ogbench')
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--eval-budget', type=int, default=50)
    parser.add_argument('--cem-horizon', type=int, default=5)
    parser.add_argument('--cem-receding-horizon', type=int, default=5)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--video-dir')
    parser.add_argument('--output')
    return parser.parse_args()


def worker_main(args):
    protocol_stdout = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        import flax
        import jax
        import jax.numpy as jnp
        from lewm_jax import ARCHITECTURE, LeWM

        payload = flax.serialization.msgpack_restore(Path(args.checkpoint).read_bytes())
        config = payload['config']
        if config.get('architecture') != ARCHITECTURE:
            raise ValueError(
                f'Checkpoint architecture {config.get("architecture")!r} is not {ARCHITECTURE!r}.'
            )
        model = LeWM(
            image_size=int(config['image_size']),
            embed_dim=int(config['embed_dim']),
            history_size=int(config['history_size']),
            projector_hidden_dim=int(config.get('projector_hidden_dim', 2048)),
            action_smoothed_dim=int(config.get('action_smoothed_dim', 10)),
            action_mlp_scale=int(config.get('action_mlp_scale', 4)),
            predictor_depth=int(config.get('predictor_depth', 6)),
            predictor_heads=int(config.get('predictor_heads', 16)),
            predictor_dim_head=int(config.get('predictor_dim_head', 64)),
            predictor_mlp_dim=int(config.get('predictor_mlp_dim', 2048)),
            predictor_dropout=float(config.get('predictor_dropout', 0.1)),
            predictor_emb_dropout=float(config.get('predictor_emb_dropout', 0.0)),
            dtype=jnp.float32,
        )
        variables = {'params': payload['params'], 'batch_stats': payload['batch_stats']}

        def plan_one(key, pixels, goals, initial_mean):
            # The reference names this tensor `var`, but samples with
            # candidates = noise * var + mean and updates it with torch.std.
            # It is therefore a standard deviation, initialized by var_scale.
            std = jnp.full_like(initial_mean, args.cem_var_scale)

            def cem_step(_, carry):
                key, mean, std = carry
                key, sample_key = jax.random.split(key)
                candidates = (
                    jax.random.normal(
                        sample_key,
                        (args.cem_num_samples, args.cem_horizon, initial_mean.shape[-1]),
                        dtype=jnp.float32,
                    )
                    * std[None]
                    + mean[None]
                )
                candidates = candidates.at[0].set(mean)
                costs = model.apply(
                    variables,
                    pixels[None, None],
                    goals[None, None],
                    candidates[None],
                    method=model.rollout_cost,
                )[0]
                _, elite_indices = jax.lax.top_k(-costs, args.cem_topk)
                elites = candidates[elite_indices]
                mean = elites.mean(axis=0)
                # torch.std(dim=1) in the reference CEM uses Bessel correction.
                std = elites.std(axis=0, ddof=1)
                return key, mean, std

            _, mean, std = jax.lax.fori_loop(
                0, args.cem_steps, cem_step, (key, initial_mean, std)
            )
            return mean, std

        plan_one = jax.jit(plan_one)
        rng = jax.random.PRNGKey(args.seed)

    stdin, stdout = sys.stdin.buffer, protocol_stdout
    _send(stdout, {'ready': True, 'epoch': payload['epoch']})
    while True:
        request = _receive(stdin)
        if request is None:
            return
        try:
            pixels = np.asarray(request['pixels'])
            goals = np.asarray(request['goals'])
            # Stable WM image transforms are NCHW; Flax ViT is NHWC.
            pixels = np.transpose(pixels, (0, 1, 3, 4, 2))
            goals = np.transpose(goals, (0, 1, 3, 4, 2))
            initial_mean = np.asarray(request['initial_mean'], dtype=np.float32)
            action_rows = []
            variance_rows = []
            # Reference CEM uses batch_size=1.  Calling one fixed-shape jitted
            # function per live environment preserves that behavior and avoids
            # recompilation when terminated environments shrink the replan set.
            for row in range(len(pixels)):
                rng, plan_key = jax.random.split(rng)
                actions, variance = plan_one(
                    plan_key, pixels[row], goals[row], initial_mean[row]
                )
                action_rows.append(np.asarray(actions))
                variance_rows.append(np.asarray(variance))
            _send(
                stdout,
                {
                    'actions': np.stack(action_rows),
                    'variance': np.stack(variance_rows),
                },
            )
        except Exception:  # noqa: BLE001 - return the remote traceback.
            _send(stdout, {'error': traceback.format_exc()})


class JAXLeWMCEMSolver:
    """Stable WM Solver surface backed by one fully-jitted JAX CEM."""

    def __init__(
        self,
        checkpoint,
        ogbench_root,
        *,
        seed,
        horizon,
        num_samples,
        steps,
        topk,
        var_scale,
    ):
        import torch

        self._torch = torch
        self._n_envs = None
        self._action_dim = None
        self._horizon = None
        self._var_scale = float(var_scale)
        child_env = os.environ.copy()
        child_env['PYTHONPATH'] = str(Path(ogbench_root) / 'impls')
        child_env.pop('LD_LIBRARY_PATH', None)
        child_env.pop('MUJOCO_GL', None)
        command = [
            str(Path(ogbench_root) / '.venv/bin/python'),
            str(Path(__file__).resolve()),
            '--worker',
            '--checkpoint',
            str(checkpoint),
            '--seed',
            str(seed),
            '--cem-horizon',
            str(horizon),
            '--cem-num-samples',
            str(num_samples),
            '--cem-steps',
            str(steps),
            '--cem-topk',
            str(topk),
            '--cem-var-scale',
            str(var_scale),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=child_env,
        )
        ready = _receive(self.process.stdout)
        if not isinstance(ready, dict) or not ready.get('ready'):
            raise RuntimeError(f'Unexpected JAX LeWM worker response: {ready}')

    def configure(self, *, action_space, n_envs, config):
        self._n_envs = n_envs
        self._horizon = config.horizon
        self._action_dim = int(np.prod(action_space.shape[1:])) * config.action_block

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def horizon(self):
        return self._horizon

    @property
    def action_dim(self):
        return self._action_dim

    def __call__(self, info_dict, init_action=None):
        return self.solve(info_dict, init_action=init_action)

    def solve(self, info_dict, init_action=None):
        batch_size = len(info_dict['pixels'])
        if init_action is None:
            initial_mean = np.zeros(
                (batch_size, self.horizon, self.action_dim), dtype=np.float32
            )
        else:
            initial_mean = init_action.detach().cpu().numpy().astype(np.float32, copy=False)
            remaining = self.horizon - initial_mean.shape[1]
            if remaining > 0:
                initial_mean = np.concatenate(
                    [
                        initial_mean,
                        np.zeros((batch_size, remaining, self.action_dim), dtype=np.float32),
                    ],
                    axis=1,
                )
        request = {
            'pixels': info_dict['pixels'].detach().cpu().numpy(),
            'goals': info_dict['goal'].detach().cpu().numpy(),
            'initial_mean': initial_mean,
        }
        _send(self.process.stdin, request)
        response = _receive(self.process.stdout)
        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(response['error'])
        return {
            'actions': self._torch.from_numpy(response['actions']),
            'mean': [self._torch.from_numpy(response['actions'])],
            'var': [self._torch.from_numpy(response['variance'])],
            'costs': [],
        }

    def close(self):
        if self.process.poll() is not None:
            return
        try:
            _send(self.process.stdin, None)
            self.process.wait(timeout=10)
        except (BrokenPipeError, EOFError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def main(args):
    if args.task is None or args.output is None:
        raise ValueError('--task and --output are required outside worker mode.')

    stable_root = Path(args.stable_wm_root).resolve()
    ogbench_root = Path(args.ogbench_root).resolve()
    sys.path.insert(0, str(stable_root))

    import stable_worldmodel as swm
    import torch
    from sklearn.preprocessing import StandardScaler
    from torchvision.transforms import v2 as transforms

    spec = task_spec(args.task, stable_root / 'datasets')
    import flax
    checkpoint_payload = flax.serialization.msgpack_restore(Path(args.checkpoint).read_bytes())
    dataset = swm.data.load_dataset(str(spec['hdf5']))
    episodes, starts = sample_eval_starts(
        dataset, args.num_eval, args.goal_offset_steps, args.seed
    )
    actions = dataset.get_col_data('action')
    actions = actions[~np.isnan(actions).any(axis=1)]
    action_scaler = StandardScaler().fit(actions)
    image_transforms = [transforms.ToImage(), transforms.Resize(size=224)]
    image_transform = transforms.Compose(image_transforms)

    solver = JAXLeWMCEMSolver(
        args.checkpoint,
        ogbench_root,
        seed=args.seed,
        horizon=args.cem_horizon,
        num_samples=args.cem_num_samples,
        steps=args.cem_steps,
        topk=args.cem_topk,
        var_scale=args.cem_var_scale,
    )
    world = None
    try:
        plan_config = swm.PlanConfig(
            horizon=args.cem_horizon,
            receding_horizon=args.cem_receding_horizon,
            history_len=1,
            action_block=args.action_block,
            warm_start=True,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan_config,
            process={'action': action_scaler},
            transform={'pixels': image_transform, 'goal': image_transform},
        )
        world = swm.World(
            **spec['world'],
            num_envs=args.num_eval,
            max_episode_steps=2 * args.eval_budget,
            image_shape=(224, 224),
            render_mode='rgb_array',
        )
        world.set_policy(policy)
        started = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=starts.tolist(),
            goal_offset=args.goal_offset_steps,
            eval_budget=args.eval_budget,
            episodes_idx=episodes.tolist(),
            callables=spec['callables'],
            video=args.video_dir,
        )
        elapsed = time.time() - started
    finally:
        if world is not None:
            world.close()
        solver.close()

    result = {
        'task': args.task,
        'method': 'lewm_jax',
        'encoder': checkpoint_payload['config'].get('encoder', 'impala_small'),
        'checkpoint': args.checkpoint,
        'checkpoint_step': checkpoint_epoch(args.checkpoint),
        'seed': args.seed,
        'num_eval': args.num_eval,
        'goal_offset_steps': args.goal_offset_steps,
        'eval_budget': args.eval_budget,
        'cem': {
            'horizon': args.cem_horizon,
            'receding_horizon': args.cem_receding_horizon,
            'action_block': args.action_block,
            'num_samples': args.cem_num_samples,
            'steps': args.cem_steps,
            'topk': args.cem_topk,
            'var_scale': args.cem_var_scale,
            'batch_size': 1,
            'history_len': 1,
            'warm_start': True,
        },
        'eval_episodes': episodes,
        'eval_start_steps': starts,
        'evaluation_time': elapsed,
        'metrics': metrics,
        # Keep the complete Stable WM metrics while also exposing the two
        # canonical dashboard fields at the top level.
        'success_rate': metrics.get('success_rate'),
        'episodes': args.num_eval,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.worker:
        worker_main(parsed_args)
    else:
        main(parsed_args)
