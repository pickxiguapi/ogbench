import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_visual_ogbench8_pipeline_is_path_connected():
    names = (
        'train_lewm_visual_ogbench8.sh',
        'precompute_visual_ogbench8_latents.sh',
        'train_action-prior-chunk_visual_ogbench8.sh',
        'train_latent_path_flow_visual_ogbench8.sh',
        'eval_lewmpp_visual_ogbench8.sh',
    )
    scripts = {
        name: (
            ROOT
            / 'experiments'
            / ('train' if name.startswith('train_') else 'eval' if name.startswith('eval_') else '')
            / name
        ).read_text()
        for name in names
    }
    assert 'weights_epoch_10.msgpack' in scripts['train_lewm_visual_ogbench8.sh'] or '--epochs=10' in scripts[
        'train_lewm_visual_ogbench8.sh'
    ]
    for name in scripts:
        assert 'GPU_IDS=(0 1 2 3 4 5 6 7)' in scripts[name]
    assert 'LEWM_CHECKPOINT_ROOT=""' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert 'ACTION_PRIOR_CHECKPOINT_ROOT=""' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert 'LATENT_PATH_FLOW_CHECKPOINT_ROOT=""' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert 'checkpoint_200000.msgpack' in scripts['eval_lewmpp_visual_ogbench8.sh']
    assert '--policy-checkpoint-step=500000' in scripts['eval_lewmpp_visual_ogbench8.sh']


def test_visual_ogbench8_eval_records_paper_protocol():
    text = (ROOT / 'experiments' / 'eval' / 'eval_lewmpp_visual_ogbench8.sh').read_text()
    for fragment in (
        'EVAL_SEEDS=(0 1 42)',
        '--num-eval=50',
        '--cem-horizon=2',
        '--cem-receding-horizon=1',
        '--action-block=5',
        '--cem-num-samples=300',
        '--cem-iterations=5',
        '--cem-topk=30',
        '--guidance-population-size=250',
        '--guidance-random-elite-cap=5',
        '--guidance-goal-mode=final',
        '--cem-cost-mode=moh',
    ):
        assert fragment in text


def test_visual_ogbench8_aggregator_rejects_incomplete_matrix(tmp_path):
    output = tmp_path / 'summary.json'
    result = subprocess.run(
        [
            str(ROOT / '.venv' / 'bin' / 'python'),
            str(ROOT / 'impls' / 'aggregate_visual_ogbench8_results.py'),
            f'--results-root={tmp_path}',
            f'--output={output}',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'Incomplete matrix' in result.stderr
    assert not output.exists()


def test_visual_ogbench8_aggregator_uses_sample_standard_deviation(tmp_path):
    tags = ('cs_play', 'cd_play', 'ct_play', 'scene_play', 'cs_noisy', 'cd_noisy', 'ct_noisy', 'scene_noisy')
    for seed, value in zip((0, 1, 42), (0.4, 0.5, 0.6)):
        for tag in tags:
            path = tmp_path / f'seed{seed}' / tag / 'result.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        'seed': seed,
                        'episodes_per_task': 50,
                        'use_subgoal': True,
                        'policy_guidance': 'policy_random_mixture',
                        'overall_success': value,
                    }
                )
            )
    output = tmp_path / 'summary.json'
    subprocess.run(
        [
            str(ROOT / '.venv' / 'bin' / 'python'),
            str(ROOT / 'impls' / 'aggregate_visual_ogbench8_results.py'),
            f'--results-root={tmp_path}',
            f'--output={output}',
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary['macro']['mean'] == 0.5
    assert summary['macro']['sample_std'] == pytest.approx(0.1)
