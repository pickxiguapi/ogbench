import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPLS = ROOT / 'impls'
EXP = ROOT / 'exp'


def _literal_assignment(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not found in {path}')


def test_final_python_entrypoints_are_converged():
    assert sorted(path.name for path in IMPLS.glob('train_*.py')) == [
        'train_gciql_chunk.py',
        'train_lewm_jax.py',
    ]
    assert sorted(path.name for path in IMPLS.glob('eval_*.py')) == [
        'eval_lewm_4tasks.py',
        'eval_ogbench_env_8tasks.py',
    ]


def test_representation_modes_encode_the_four_design_choices():
    modes = _literal_assignment(IMPLS / 'train_gciql_chunk.py', 'REPRESENTATION_MODES')
    assert modes == {
        'independent': (False, False, False),
        'pi': (False, False, True),
        'qv': (True, True, False),
        'all': (True, True, True),
    }


def test_formal_bashes_expose_modes_and_disable_augmentation_by_default():
    policy_bashes = sorted(EXP.rglob('*gciql_chunk*.sh'))
    eval_bashes = sorted(EXP.rglob('*eval*.sh'))
    assert len(policy_bashes) == 2
    assert len(eval_bashes) == 2
    for path in policy_bashes:
        text = path.read_text()
        assert 'REPRESENTATION_MODE=${REPRESENTATION_MODE:-independent}' in text
        assert 'P_AUG=${P_AUG:-0.0}' in text
        assert '--representation_mode="$REPRESENTATION_MODE"' in text
        assert '--p_aug="$P_AUG"' in text
    for path in eval_bashes:
        text = path.read_text()
        assert 'MODE=${MODE:-guided}' in text
        assert 'policy|lewm|guided|native_q' in text
        assert 'REPRESENTATION_MODE=${REPRESENTATION_MODE:-independent}' in text


def test_policy_experiment_names_are_compact_and_guarded():
    policy_bashes = sorted(EXP.rglob('*gciql_chunk*.sh'))
    eval_bashes = sorted(EXP.rglob('*eval*.sh'))
    for path in policy_bashes + eval_bashes:
        text = path.read_text()
        assert 'MODE_TAG=ind' in text
        assert '${#exp_name} >= 64' in text or '${#policy_name} >= 64' in text

    names = [
        f'gc8_{tag}_{mode}_n500000_b512_a0.0_sd0'
        for tag in 'cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy'.split()
        for mode in 'ind pi qv all'.split()
    ]
    assert max(map(len, names)) < 64


def test_only_evaluation_uses_reproduction_wrappers():
    train_wrappers = sorted((EXP / 'train').glob('*reproduce*main_matrix.sh'))
    eval_wrappers = sorted((EXP / 'eval').rglob('*reproduce*main_matrix.sh'))
    assert train_wrappers == []
    assert len(eval_wrappers) == 2
    for path in eval_wrappers:
        text = path.read_text()
        assert 'MODE=lewm REPRESENTATION_MODE=independent' in text
        assert 'for representation in independent pi qv all' in text
        for mode in ('policy', 'guided', 'native_q'):
            assert f'MODE={mode} REPRESENTATION_MODE="$representation"' in text


def test_policy_bashes_only_require_lewm_settings_for_shared_modes():
    policy_bashes = sorted(EXP.rglob('*gciql_chunk*.sh'))
    for path in policy_bashes:
        text = path.read_text()
        independent = text.index('independent)')
        shared = text.index('pi|qv|all)')
        checkpoint = text.index('lewm_args=(--lewm_checkpoint=')
        assert independent < shared < checkpoint
        assert ': "${LEWM_SEED:?' in text
        assert ': "${LEWM_BATCH_SIZE:?' in text
        assert 'Frozen LeWM checkpoint not found:' in text
        assert text.index('Frozen LeWM checkpoint not found:') < text.index(
            'for i in "${!'
        )
        assert 'train_lewm_jax.py' not in text


def test_executors_source_client_env_from_the_current_checkout():
    executors = sorted(EXP.rglob('*train_*.sh')) + sorted(EXP.rglob('*eval_*.sh'))
    executors = [path for path in executors if 'reproduce_' not in path.name]
    assert len(executors) == 6
    for path in executors:
        text = path.read_text()
        assert 'source "$OGBENCH_ROOT/scripts/client_env.sh"' in text
        assert 'source /home/' not in text
        assert 'source /root/' not in text


def test_backup_launchers_are_not_executable():
    for path in (ROOT / 'backup').rglob('*.sh'):
        assert not os.access(path, os.X_OK), path
