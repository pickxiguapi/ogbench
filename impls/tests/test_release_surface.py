import ast
import hashlib
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
    assert not list(ROOT.glob('*.sh'))
    scripts = {path.name for path in (ROOT / 'experiments').glob('*.sh')}
    assert scripts == {
        'eval_ablation_action_prior_h25_4tasks.sh',
        'eval_ablation_moh_h25_4tasks.sh',
        'eval_ablation_subgoal_path_h25_4tasks.sh',
        'eval_action-prior-chunk_h25_4tasks.sh',
        'eval_lewm_baseline_h100_4tasks.sh',
        'eval_lewm_baseline_h25_4tasks.sh',
        'eval_lewm_baseline_h50_4tasks.sh',
        'eval_lewm_baseline_h75_4tasks.sh',
        'eval_lewmpp_h100_4tasks.sh',
        'eval_lewmpp_h25_4tasks.sh',
        'eval_lewmpp_h50_4tasks.sh',
        'eval_lewmpp_h75_4tasks.sh',
        'eval_policy_mode_anchor_h25_4tasks.sh',
        'eval_subgoal_generators_h25_4tasks.sh',
        'precompute_lewm_latents_4tasks.sh',
        'train_action-prior-chunk_4tasks.sh',
        'train_lewm_4tasks.sh',
        'train_subgoal_endpoint_flow_general_uniform_future_4tasks.sh',
        'train_subgoal_endpoint_flow_goalmax25_4tasks.sh',
        'train_subgoal_latent_path_flow_general_uniform_future_4tasks.sh',
        'train_subgoal_latent_path_flow_goalmax25_4tasks.sh',
        'train_subgoal_mlp_general_uniform_future_4tasks.sh',
        'train_subgoal_mlp_goalmax25_4tasks.sh',
    }
    for script in scripts:
        text = (ROOT / 'experiments' / script).read_text()
        assert '\nif ' not in text
        assert '\ncase ' not in text
        assert 'TASKS=(cube pusht reacher tworoom)' in text
        assert 'GPU_IDS=(0 1 2 3)' in text
        assert 'pids+=("$!")' in text
        assert 'wait "$pid" || status=1' in text
        assert 'exit "$status"' in text
    for retired in ('backup', 'reports', 'results'):
        assert not (ROOT / retired).exists()


def test_release_has_complete_config_templates():
    names = {path.name for path in (ROOT / 'configs').glob('*.example.env')}
    assert names == {'lewmpp_paths.example.env'}


def test_action_prior_training_launcher_records_release_hyperparameters():
    text = (ROOT / 'experiments' / 'train_action-prior-chunk_4tasks.sh').read_text()
    for argument in (
        '--train_steps=100000',
        '--save_interval=100000',
        '--log_interval=5000',
        '--batch_size=256',
        '--seed=777',
        '--lr=3e-4',
        '--discount=0.99',
        '--expectile=0.9',
        '--tau=0.005',
        '--chunk_size=5',
        '--alpha=3.0',
        '--p_aug=0.0',
        '--validation_fraction=0.05',
    ):
        assert argument in text


def test_generator_launchers_record_and_validate_family_invariants():
    for path in (ROOT / 'experiments').glob('train_subgoal_*_goalmax25_4tasks.sh'):
        text = path.read_text()
        assert '--goal-sampling=uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25' in text
        assert '--max-goal-steps=25' in text
    for path in (ROOT / 'experiments').glob('train_subgoal_*_general_uniform_future_4tasks.sh'):
        text = path.read_text()
        assert '--goal-sampling=hiql_uniform_future_same_trajectory' in text
        assert '--max-goal-steps' not in text
    generator_evals = [
        path
        for path in (ROOT / 'experiments').glob('eval_*.sh')
        if '--subgoal-generator-checkpoint' in path.read_text()
    ]
    assert len(generator_evals) == 9
    for path in generator_evals:
        text = path.read_text()
        assert text.count('validate_generator_checkpoint.py') == 1
        assert text.index('validate_generator_checkpoint.py') < text.index('pids=()')


def test_evaluation_launchers_refuse_to_overwrite_results():
    for path in (ROOT / 'experiments').glob('eval_*.sh'):
        text = path.read_text()
        assert 'test ! -e "$result_dir/result.json"' in text or (
            'test ! -e "$full_dir/result.json"' in text and 'test ! -e "$ablation_dir/result.json"' in text
        )


def test_python_entrypoints_match_the_release_pipeline():
    names = {path.name for path in (ROOT / 'impls').glob('*.py') if path.name.startswith(('train_', 'eval_'))}
    names.add('action-prior-chunk.py')
    assert names == {
        'action-prior-chunk.py',
        'train_lewm_jax.py',
        'train_subgoal_generator.py',
        'eval_lewm_4tasks.py',
    }


def test_release_has_exactly_three_subgoal_model_types():
    tree = ast.parse((ROOT / 'impls' / 'subgoal_generators.py').read_text())
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes == ['LatentSubgoalMLP', 'AdaLNTransformerBlock', 'LatentPathFlow']


def test_evaluator_exposes_required_release_variants_and_defaults():
    assert VARIANTS == ('full', 'no_subgoal', 'no_action_prior', 'no_moh', 'lewm', 'action_prior_chunk')
    assert (DEFAULT_CEM_ITERATIONS, DEFAULT_CEM_SAMPLES, DEFAULT_FLOW_STEPS) == (5, 300, 16)


def test_action_prior_public_surface_uses_neutral_name():
    retired_method_name = 'gci' + 'ql'
    retired_loss_name = 'a' + 'wr'
    paths = [
        ROOT / 'README.md',
        ROOT / 'CHANGELOG.md',
        ROOT / 'impls' / 'action-prior-chunk.py',
        ROOT / 'impls' / 'action_prior_chunk.py',
        ROOT / 'impls' / 'agents' / 'action_prior_chunk.py',
        ROOT / 'impls' / 'eval_lewm_4tasks.py',
    ]
    paths.extend((ROOT / 'configs').glob('*.example.env'))
    paths.extend((ROOT / 'experiments').glob('*.sh'))
    for path in paths:
        for line in path.read_text().splitlines():
            lowered = line.lower()
            assert retired_method_name not in lowered, path
            if 'actor_loss' not in lowered:
                assert retired_loss_name not in lowered, path


def test_preflight_rejects_generator_family_mismatch(tmp_path):
    (tmp_path / 'cube_single_expert.h5').touch()
    (tmp_path / 'cube_single_expert.lance').touch()
    lewm = tmp_path / 'lewm.msgpack'
    lewm.touch()
    checkpoint = tmp_path / 'checkpoint.msgpack'
    checkpoint.write_bytes(b'checkpoint')
    config = {
        'architecture': 'latent_path_flow_transformer_encoder',
        'task': 'cube',
        'latent_dataset': str(tmp_path / 'cube.h5'),
        'lewm_checkpoint_sha256': hashlib.sha256(lewm.read_bytes()).hexdigest(),
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
