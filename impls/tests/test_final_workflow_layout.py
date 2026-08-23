import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPLS = ROOT / 'impls'
EXP = ROOT / 'exp'

POLICY_BASHES = [
    EXP / 'train' / '20260823_train_node2_gciql_chunk_ogbench_env_8tasks.sh',
    EXP / 'train' / '20260823_train_yb_gciql_chunk_4tasks.sh',
    EXP / 'train' / '20260823_train_yb_gciql_chunk_4tasks_independent.sh',
]
EVAL_BASHES = [
    EXP / 'eval' / 'lewm_4tasks' / '20260823_eval_yb_lewm_4tasks.sh',
    EXP / 'eval' / 'ogbench_env_8tasks' / '20260823_eval_node2_ogbench_env_8tasks.sh',
]


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
    for path in POLICY_BASHES:
        text = path.read_text()
        assert '--p_aug="$P_AUG"' in text
    independent = (EXP / 'train' / '20260823_train_yb_gciql_chunk_4tasks_independent.sh').read_text()
    assert '--representation_mode=independent' in independent
    assert '--lewm_checkpoint' not in independent
    shared = (EXP / 'train' / '20260823_train_yb_gciql_chunk_4tasks.sh').read_text()
    assert 'P_AUG=${P_AUG:-0.0}' in shared
    assert 'REPRESENTATION_MODE=${REPRESENTATION_MODE:?' in shared
    assert '--representation_mode="$REPRESENTATION_MODE"' in shared
    for path in EVAL_BASHES:
        text = path.read_text()
        assert 'MODE=${MODE:-guided}' in text
        assert 'policy|lewm|guided|native_q' in text
        assert 'REPRESENTATION_MODE=${REPRESENTATION_MODE:-independent}' in text


def test_policy_experiment_names_are_compact_and_guarded():
    for path in POLICY_BASHES + EVAL_BASHES:
        text = path.read_text()
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
    independent_path = EXP / 'train' / '20260823_train_yb_gciql_chunk_4tasks_independent.sh'
    independent = independent_path.read_text()
    assert 'REPRESENTATION_MODE' not in independent
    assert 'LEWM_' not in independent
    assert 'lewm_checkpoint' not in independent.lower()

    yb_shared = (EXP / 'train' / '20260823_train_yb_gciql_chunk_4tasks.sh').read_text()
    assert 'independent)' not in yb_shared
    for unused in ('LEWM_EPOCH', 'LEWM_BATCH_SIZE', 'LEWM_SEED'):
        assert unused not in yb_shared
    for task in ('CUBE', 'PUSHT', 'REACHER', 'TWOROOM'):
        assert f'LEWM_{task}_CHECKPOINT' in yb_shared
    assert 'Frozen LeWM checkpoint not found:' in yb_shared
    assert yb_shared.index('Frozen LeWM checkpoint not found:') < yb_shared.index(
        'for i in "${!datasets[@]}"'
    )

    node2_shared = (EXP / 'train' / '20260823_train_node2_gciql_chunk_ogbench_env_8tasks.sh').read_text()
    assert ': "${LEWM_SEED:?' in node2_shared
    assert ': "${LEWM_BATCH_SIZE:?' in node2_shared
    assert 'Frozen LeWM checkpoint not found:' in node2_shared
    for text in (yb_shared, node2_shared):
        assert 'train_lewm_jax.py' not in text


def test_executors_source_client_env_from_the_current_checkout():
    executors = POLICY_BASHES + EVAL_BASHES + [
        EXP / 'train' / '20260823_train_node2_lewm_ogbench_env_8tasks.sh',
        EXP / 'train' / '20260823_train_yb_lewm_4tasks.sh',
    ]
    for path in executors:
        text = path.read_text()
        assert 'source "$OGBENCH_ROOT/scripts/client_env.sh"' in text
        assert 'source /home/' not in text
        assert 'source /root/' not in text


def test_backup_launchers_are_not_executable():
    for path in (ROOT / 'backup').rglob('*.sh'):
        assert not os.access(path, os.X_OK), path
