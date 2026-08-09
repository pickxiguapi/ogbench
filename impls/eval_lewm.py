"""Evaluate OGBench agents with Stable WM's eval_ff dataset protocol."""

import argparse
import contextlib
import json
import os
import pickle
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HEADER = struct.Struct('!Q')


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError('Policy worker exited unexpectedly.')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def _send(stream, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _receive(stream):
    size = _HEADER.unpack(_read_exact(stream, _HEADER.size))[0]
    return pickle.loads(_read_exact(stream, size))


def _pack_array(value):
    value = np.ascontiguousarray(value)
    return {
        'dtype': value.dtype.str,
        'shape': value.shape,
        'data': value.tobytes(),
    }


def _unpack_array(value):
    return np.frombuffer(value['data'], dtype=value['dtype']).reshape(
        value['shape']
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('cube', 'pusht'), required=True)
    parser.add_argument('--method', choices=('gciql', 'hiql'), required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--checkpoint-step', type=int, default=100000)
    parser.add_argument('--stable-wm-root', default='/home/dzb/stable-worldmodel')
    parser.add_argument('--ogbench-root', default='/home/dzb/ogbench')
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--eval-budget', type=int, default=50)
    parser.add_argument('--video-dir', default=None)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def task_spec(task, data_root):
    if task == 'cube':
        return {
            'hdf5': data_root / 'cube_single_expert.h5',
            'lance': data_root / 'cube_single_expert.lance',
            'world': {
                'env_name': 'swm/OGBCube-v0',
                'env_type': 'single',
                'ob_type': 'states',
                'multiview': False,
                'width': 224,
                'height': 224,
                'visualize_info': False,
                'terminate_at_goal': True,
            },
            'callables': [
                {
                    'method': 'set_state',
                    'args': {
                        'qpos': {'value': 'qpos'},
                        'qvel': {'value': 'qvel'},
                    },
                },
                {
                    'method': 'set_target_pos',
                    'args': {
                        'cube_id': {'value': 0, 'in_dataset': False},
                        'target_pos': {
                            'value': 'goal_privileged_block_0_pos'
                        },
                        'target_quat': {
                            'value': 'goal_privileged_block_0_quat'
                        },
                    },
                },
            ],
        }
    return {
        'hdf5': data_root / 'pusht_expert_train.h5',
        'lance': data_root / 'pusht_expert_train.lance',
        'world': {'env_name': 'swm/PushT-v1'},
        'callables': [
            {'method': '_set_state', 'args': {'state': {'value': 'state'}}},
            {
                'method': '_set_goal_state',
                'args': {'goal_state': {'value': 'goal_state'}},
            },
        ],
    }


def agent_config(method):
    if method == 'gciql':
        from agents.gciql import get_config

        config = get_config()
        config.alpha = 1.0
    else:
        from agents.hiql import get_config

        config = get_config()
        config.high_alpha = 3.0
        config.low_actor_rep_grad = True
        config.low_alpha = 3.0
        config.subgoal_steps = 10
    config.batch_size = 256
    config.encoder = 'impala_small'
    config.p_aug = 0.5
    return config


def load_agent(method, lance_path, checkpoint_dir, checkpoint_step):
    from agents import agents
    from utils.datasets import GCDataset, HGCDataset
    from utils.flax_utils import restore_agent
    from utils.lewm_dataset import LeWMLanceDataset

    config = agent_config(method)
    base = LeWMLanceDataset(
        lance_path, split='train', validation_fraction=0.05
    )
    wrapper = GCDataset if method == 'gciql' else HGCDataset
    dataset = wrapper(base, config, preprocess_frame_stack=False)
    example = dataset.sample(1, evaluation=True)
    agent = agents[config.agent_name].create(
        0, example['observations'], example['actions'], config
    )
    return restore_agent(agent, checkpoint_dir, checkpoint_step)


def worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--method', choices=('gciql', 'hiql'), required=True)
    parser.add_argument('--lance-path', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--checkpoint-step', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    import jax

    # Keep stdout exclusively for the framed IPC protocol.
    with contextlib.redirect_stdout(sys.stderr):
        agent = load_agent(
            args.method,
            args.lance_path,
            args.checkpoint_dir,
            args.checkpoint_step,
        )
    rng = jax.random.PRNGKey(args.seed)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    _send(stdout, {'ready': True})

    while True:
        request = _receive(stdin)
        if request is None:
            break
        try:
            rng, action_rng = jax.random.split(rng)
            actions = agent.sample_actions(
                observations=_unpack_array(request['observations']),
                goals=_unpack_array(request['goals']),
                seed=action_rng,
                temperature=0.0,
            )
            _send(stdout, _pack_array(np.asarray(actions)))
        except Exception:  # noqa: BLE001 - return worker failures to parent.
            _send(stdout, {'error': traceback.format_exc()})


class OGBenchPolicy:
    """Stable WM policy backed by an isolated OGBench JAX process."""

    def __init__(
        self,
        action_scaler,
        method,
        lance_path,
        checkpoint_dir,
        checkpoint_step,
        seed,
        ogbench_root,
    ):
        self.action_scaler = action_scaler
        self.env = None
        self.type = 'ogbench'

        child_env = os.environ.copy()
        child_env['PYTHONPATH'] = str(Path(ogbench_root) / 'impls')
        child_env.pop('LD_LIBRARY_PATH', None)
        child_env.pop('MUJOCO_GL', None)
        command = [
            str(Path(ogbench_root) / '.venv/bin/python'),
            str(Path(__file__).resolve()),
            '--worker',
            '--method',
            method,
            '--lance-path',
            str(lance_path),
            '--checkpoint-dir',
            str(checkpoint_dir),
            '--checkpoint-step',
            str(checkpoint_step),
            '--seed',
            str(seed),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=child_env,
        )
        ready = _receive(self.process.stdout)
        if ready != {'ready': True}:
            raise RuntimeError(f'Unexpected policy worker response: {ready}')

    def set_env(self, env):
        self.env = env

    @staticmethod
    def _latest_frame(value):
        value = np.asarray(value)
        return value[:, -1] if value.ndim == 5 else value

    def get_action(self, info_dict, **kwargs):
        request = {
            'observations': _pack_array(
                self._latest_frame(info_dict['pixels'])
            ),
            'goals': _pack_array(self._latest_frame(info_dict['goal'])),
        }
        _send(self.process.stdin, request)
        normalized = _receive(self.process.stdout)
        if isinstance(normalized, dict) and 'error' in normalized:
            raise RuntimeError(normalized['error'])
        return self.action_scaler.inverse_transform(_unpack_array(normalized))

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


def get_episode_lengths(dataset, episodes):
    column = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    episode_idx = dataset.get_col_data(column)
    step_idx = dataset.get_col_data('step_idx')
    return np.asarray(
        [np.max(step_idx[episode_idx == episode]) + 1 for episode in episodes]
    )


def sample_eval_starts(dataset, num_eval, goal_offset, seed):
    column = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    episode_ids = dataset.get_col_data(column)
    episodes = np.unique(episode_ids)
    max_start = get_episode_lengths(dataset, episodes) - goal_offset - 1
    by_episode = {
        episode: max_start[index] for index, episode in enumerate(episodes)
    }
    max_start_per_row = np.asarray(
        [by_episode[episode] for episode in episode_ids]
    )
    valid = np.nonzero(
        dataset.get_col_data('step_idx') <= max_start_per_row
    )[0]
    if len(valid) <= num_eval:
        raise ValueError(
            f'Only {len(valid)} valid evaluation starts for {num_eval} runs'
        )

    # Match scripts/plan/eval_ff.py exactly, including its exclusive upper
    # bound of len(valid) - 1.
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(valid) - 1, size=num_eval, replace=False)
    rows = np.sort(valid[positions])
    selected = dataset.get_row_data(rows)
    return selected[column].astype(int), selected['step_idx'].astype(int)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def main():
    args = parse_args()
    stable_root = Path(args.stable_wm_root).resolve()
    sys.path.insert(0, str(stable_root))

    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler

    # Stable WM owns the evaluator and Torch runtime. Append (rather than
    # prepend) OGBench's environment dependencies only after Stable WM has
    # loaded, so its CUDA 13 Torch libraries cannot be shadowed by JAX's
    # CUDA 12 packages. JAX policy inference runs in the isolated worker.
    ogbench_root = Path(args.ogbench_root).resolve()
    sys.path.extend(
        [
            str(ogbench_root),
            str(ogbench_root / '.venv/lib/python3.10/site-packages'),
        ]
    )

    spec = task_spec(args.task, stable_root / 'datasets')
    dataset = swm.data.load_dataset(str(spec['hdf5']))
    episodes, starts = sample_eval_starts(
        dataset, args.num_eval, args.goal_offset_steps, args.seed
    )
    scaler = StandardScaler().fit(dataset.get_col_data('action'))
    policy = OGBenchPolicy(
        scaler,
        args.method,
        spec['lance'],
        args.checkpoint_dir,
        args.checkpoint_step,
        args.seed,
        ogbench_root,
    )
    world = None
    try:
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
        evaluation_time = time.time() - started
    finally:
        if world is not None:
            world.close()
        policy.close()

    result = {
        'task': args.task,
        'method': args.method,
        'checkpoint_dir': args.checkpoint_dir,
        'checkpoint_step': args.checkpoint_step,
        'seed': args.seed,
        'num_eval': args.num_eval,
        'goal_offset_steps': args.goal_offset_steps,
        'eval_budget': args.eval_budget,
        'eval_episodes': episodes,
        'eval_start_steps': starts,
        'evaluation_time': evaluation_time,
        'metrics': metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker_main()
    else:
        main()
