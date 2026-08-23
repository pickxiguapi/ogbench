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


def test_reproduction_wrappers_cover_the_main_matrices():
    train_wrappers = sorted((EXP / 'train').glob('*reproduce*main_matrix.sh'))
    eval_wrappers = sorted((EXP / 'eval').rglob('*reproduce*main_matrix.sh'))
    assert len(train_wrappers) == 2
    assert len(eval_wrappers) == 2
    for path in train_wrappers:
        text = path.read_text()
        assert text.index('REPRESENTATION_MODE=independent') < text.index(
            'if [[ "$RUN_LEWM" == 1 ]]'
        )
        assert text.index('if [[ "$RUN_LEWM" == 1 ]]') < text.index(
            'for mode in pi qv all'
        )
        assert 'for mode in pi qv all' in text
        assert 'REPRESENTATION_MODE="$mode"' in text
    for path in eval_wrappers:
        text = path.read_text()
        assert 'MODE=lewm REPRESENTATION_MODE=independent' in text
        assert 'for representation in independent pi qv all' in text
        for mode in ('policy', 'guided', 'native_q'):
            assert f'MODE={mode} REPRESENTATION_MODE="$representation"' in text


def test_backup_launchers_are_not_executable():
    for path in (ROOT / 'backup').rglob('*.sh'):
        assert not os.access(path, os.X_OK), path
