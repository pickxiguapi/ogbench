import ast
import json
import subprocess
from pathlib import Path

from eval_lewm_4tasks import DEFAULT_CEM_ITERATIONS, DEFAULT_CEM_SAMPLES, DEFAULT_FLOW_STEPS, VARIANTS

ROOT = Path(__file__).resolve().parents[2]


def test_release_has_only_paper_experiment_launchers():
    scripts = sorted(path.name for path in (ROOT / 'exp' / 'lewmpp').glob('*.sh'))
    assert scripts == [
        'common.sh',
        'evaluate.sh',
        'evaluate_main_one_seed.sh',
        'precompute_latents.sh',
        'reproduce_paper.sh',
        'train_action_prior.sh',
        'train_lewm.sh',
        'train_subgoal_generator.sh',
    ]
    for retired in ('backup', 'reports', 'results'):
        assert not (ROOT / retired).exists()


def test_python_entrypoints_match_the_release_pipeline():
    names = {path.name for path in (ROOT / 'impls').glob('*.py') if path.name.startswith(('train_', 'eval_'))}
    assert names == {
        'train_lewm_jax.py',
        'train_action_prior.py',
        'train_subgoal_generator.py',
        'eval_lewm_4tasks.py',
    }


def test_release_has_exactly_three_subgoal_model_types():
    tree = ast.parse((ROOT / 'impls' / 'subgoal_generators.py').read_text())
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes == ['LatentSubgoalMLP', 'AdaLNTransformerBlock', 'LatentPathFlow']


def test_evaluator_exposes_required_release_variants_and_defaults():
    assert VARIANTS == ('full', 'no_subgoal', 'no_action_prior', 'no_moh', 'lewm', 'gciql_chunk')
    assert (DEFAULT_CEM_ITERATIONS, DEFAULT_CEM_SAMPLES, DEFAULT_FLOW_STEPS) == (5, 300, 16)


def test_bash_rejects_generator_family_or_type_mismatch(tmp_path):
    checkpoint = tmp_path / 'checkpoint.msgpack'
    checkpoint.write_bytes(b'checkpoint')
    config = {
        'architecture': 'latent_path_flow_transformer_encoder',
        'goal_sampling': 'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25',
        'max_goal_steps': 25,
    }
    (tmp_path / 'config.json').write_text(json.dumps(config))
    common = ROOT / 'exp' / 'lewmpp' / 'common.sh'

    valid = subprocess.run(
        ['bash', '-c', 'source "$1"; verify_generator goalmax25 latent_path_flow "$2"', 'test', common, checkpoint],
        capture_output=True,
        text=True,
    )
    mismatch = subprocess.run(
        [
            'bash',
            '-c',
            'source "$1"; verify_generator general_uniform_future latent_path_flow "$2"',
            'test',
            common,
            checkpoint,
        ],
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0
    assert mismatch.returncode != 0
