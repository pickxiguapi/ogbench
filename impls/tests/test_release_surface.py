import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from eval_lewm_4tasks import (
    DEFAULT_CEM_ITERATIONS,
    DEFAULT_CEM_SAMPLES,
    DEFAULT_FLOW_STEPS,
    VARIANTS,
    validate_release_files,
)

ROOT = Path(__file__).resolve().parents[2]


def test_release_has_only_paper_experiment_launchers():
    scripts = sorted(path.name for path in ROOT.glob('*.sh'))
    assert scripts == ['eval.sh', 'train.sh']
    assert not list((ROOT / 'exp' / 'lewmpp').glob('*.sh'))
    for script in scripts:
        text = (ROOT / script).read_text()
        assert '\nif ' not in text
        assert '\ncase ' not in text
    for retired in ('backup', 'reports', 'results'):
        assert not (ROOT / retired).exists()


def test_release_has_complete_config_templates():
    names = {path.name for path in (ROOT / 'configs').glob('*.example.env')}
    assert names == {
        'eval_gciql_chunk.example.env',
        'eval_lewm_baseline.example.env',
        'eval_lewmpp_general.example.env',
        'eval_lewmpp_h25.example.env',
        'eval_no_action_prior.example.env',
        'eval_no_moh.example.env',
        'eval_no_subgoal.example.env',
        'lewmpp_paths.example.env',
        'precompute_latents.example.env',
        'train_action_prior.example.env',
        'train_lewm.example.env',
        'train_subgoal_generator.example.env',
    }


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


def test_preflight_rejects_generator_family_mismatch(tmp_path):
    (tmp_path / 'cube_single_expert.h5').touch()
    (tmp_path / 'cube_single_expert.lance').touch()
    lewm = tmp_path / 'lewm.msgpack'
    lewm.touch()
    checkpoint = tmp_path / 'checkpoint.msgpack'
    checkpoint.write_bytes(b'checkpoint')
    config = {
        'architecture': 'latent_path_flow_transformer_encoder',
        'goal_sampling': 'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25',
        'max_goal_steps': 25,
    }
    (tmp_path / 'config.json').write_text(json.dumps(config))
    args = SimpleNamespace(
        variant='no_action_prior',
        task='cube',
        data_root=str(tmp_path),
        lewm_checkpoint=str(lewm),
        action_prior_checkpoint_dir=None,
        action_prior_checkpoint_step=100_000,
        subgoal_generator_checkpoint=str(checkpoint),
        goal_offset_steps=25,
        generator_family='goalmax25',
        generator_type='latent_path_flow',
    )
    validate_release_files(args)
    args.generator_family = 'general_uniform_future'
    with pytest.raises(ValueError, match='H25 requires goalmax25'):
        validate_release_files(args)
