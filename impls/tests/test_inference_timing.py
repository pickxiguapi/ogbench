import numpy as np

from inference_timing import LeWMInferenceProfiler, _distribution


class _FakePolicy:
    action_block = 5
    guidance_policy = None
    subgoal_generator = None

    def __init__(self):
        self.should_replan = True

    def _plan_one(self):
        return np.ones((2, 3), dtype=np.float32)

    def get_actions(self, pixels, goals, alive):
        if self.should_replan:
            for _ in np.flatnonzero(alive):
                self._plan_one()
        return np.zeros((len(alive), 1), dtype=np.float32)


def test_distribution_separates_cold_start():
    summary = _distribution([0.003, 0.001, 0.002], drop_first=1)
    assert summary['count'] == 3
    assert summary['cold_start_ms'] == [3.0]
    assert summary['steady_mean_ms'] == 1.5
    assert summary['steady_median_ms'] == 1.5


def test_profiler_reports_per_environment_replan_and_buffer_costs():
    policy = _FakePolicy()
    profiler = LeWMInferenceProfiler(policy)
    pixels = np.zeros((3, 1, 2), dtype=np.float32)
    goals = np.zeros_like(pixels)
    alive = np.ones(3, dtype=bool)

    # The first replan batch is treated as cold start.  The second contributes
    # three steady-state replans, followed by one buffer-only action step.
    policy.get_actions(pixels, goals, alive)
    policy.get_actions(pixels, goals, alive)
    policy.should_replan = False
    policy.get_actions(pixels, goals, alive)

    summary = profiler.summary()
    assert summary['counts']['replan_events'] == 6
    assert summary['counts']['steady_replan_events'] == 3
    assert summary['counts']['alive_actions'] == 9
    assert summary['modules']['cem']['count'] == 6
    assert summary['end_to_end']['steady_replan_ms_per_environment'] >= 0.0
    assert summary['end_to_end']['buffer_action_ms_per_environment'] >= 0.0
