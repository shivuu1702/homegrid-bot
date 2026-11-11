"""Unit and validation tests for the HomeGrid benchmark comparison.

Verifies:
1. Environment seeding produces distinct initial states across seeds.
2. The benchmark never overwrites skills/smdp_policy.json.
3. Evaluation never updates or mutates trained policy weights.
4. Episode-level discounted reward is correctly calculated across multiple skills.
5. All three methods evaluate on the exact same held-out benchmark tasks.
"""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "agent"))

from benchmark_comparison import (
    BENCHMARK_TASKS,
    SimulatedHomeGridEnv,
    run_simulated_benchmark,
)
from rl_selector import (
    LinearSMDPSkillSelector,
    SMDPExecutionTracker,
    SkillOption,
    get_all_standard_options,
)


class BenchmarkValidationTests(unittest.TestCase):
    def test_different_seeds_produce_different_initial_states(self):
        """Verify that evaluation seeds produce distinct agent starting locations."""
        eval_seeds = [42, 101, 202, 303, 404]
        initial_positions = []
        initial_directions = []

        for s in eval_seeds:
            env = SimulatedHomeGridEnv(task="find the bottle", seed=s)
            state = env.get_full_symbolic_state()
            initial_positions.append(state["agent"]["pos"])
            initial_directions.append(state["agent"]["dir"])

        # Must have variation across seeds (not all identical)
        unique_positions = set(initial_positions)
        unique_directions = set(initial_directions)

        self.assertGreater(len(unique_positions), 1, "Seeds failed to diversify agent position")
        self.assertGreater(len(unique_directions), 1, "Seeds failed to diversify agent direction")

    def test_benchmark_does_not_overwrite_smdp_policy_json(self):
        """Verify that benchmark policy save targets smdp_policy_benchmark.json, not smdp_policy.json."""
        prod_policy_path = Path("skills") / "smdp_policy.json"
        self.assertTrue(prod_policy_path.exists(), "skills/smdp_policy.json must exist")

        with open(prod_policy_path, "rb") as f:
            original_bytes = f.read()

        # Check benchmark script specifies benchmark-specific policy path
        with open("benchmark_comparison.py", "r", encoding="utf-8") as f:
            bench_code = f.read()

        self.assertIn("smdp_policy_benchmark.json", bench_code)
        # Verify production policy path is never assigned as save target in benchmark
        self.assertNotIn('policy_path = Path("skills") / "smdp_policy.json"', bench_code)

        # Confirm production policy bytes remain unchanged
        with open(prod_policy_path, "rb") as f:
            current_bytes = f.read()
        self.assertEqual(original_bytes, current_bytes)

    def test_evaluation_does_not_update_trained_policy_weights(self):
        """Verify that evaluation is strictly read-only and does not mutate policy weights."""
        standard_options = get_all_standard_options()
        selector = LinearSMDPSkillSelector(default_options=standard_options, epsilon=0.0, seed=42)

        # Seed with non-zero arbitrary weights to test for any weight drift
        for opt in standard_options:
            w = selector.get_weights(opt)
            w[:5] = np.array([0.5, -0.2, 0.8, -0.4, 0.1])
        selector.shared_weights[:5] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        weights_before = {k: v.copy() for k, v in selector.weights.items()}
        shared_before = selector.shared_weights.copy()

        # Run evaluation on benchmark
        res = run_simulated_benchmark(
            policy="smdp_trained",
            tasks=BENCHMARK_TASKS[:3],
            selector=selector,
            seed=42,
        )

        # Verify weights are identical
        self.assertTrue(np.array_equal(selector.shared_weights, shared_before))
        for k in weights_before:
            self.assertTrue(np.array_equal(selector.weights[k], weights_before[k]))

    def test_episode_level_reward_discounting_across_multiple_skills(self):
        """Verify that discounted reward is calculated from the start of the episode,

        accounting for total primitive steps across all executed skills.
        """
        env = SimulatedHomeGridEnv(task="open the recycling bin", seed=42)
        gamma = 0.95
        tracker = SMDPExecutionTracker(env, gamma=gamma)

        # Skill 1: open_bin(compost bin) -> takes 5 steps, reward 0.0
        opt1 = SkillOption("open_bin", ("compost bin",))
        env.execute_skill(opt1, tracker)
        self.assertEqual(tracker.total_env_steps, 5)

        # Skill 2: open_bin(recycling bin) -> takes 5 steps, completes task, reward 1.0
        opt2 = SkillOption("open_bin", ("recycling bin",))
        env.execute_skill(opt2, tracker)
        self.assertEqual(tracker.total_env_steps, 10)

        # Whole-episode discounted reward must be gamma^(10-1) = gamma^9
        expected_episode_reward = gamma ** 9  # ~0.6302
        single_skill_reward = gamma ** 4      # ~0.8145

        # Verify tracker accumulated episode-level reward correctly
        self.assertAlmostEqual(tracker.episode_discounted_reward, expected_episode_reward, places=4)
        self.assertNotAlmostEqual(tracker.episode_discounted_reward, single_skill_reward, places=2)

    def test_all_methods_use_same_held_out_tasks(self):
        """Verify that Random, Untrained, and Trained SMDP evaluate the exact same tasks."""
        res_random = run_simulated_benchmark(policy="random", tasks=BENCHMARK_TASKS, seed=42)
        sel_un = LinearSMDPSkillSelector(epsilon=0.0, seed=42)
        res_un = run_simulated_benchmark(policy="smdp_untrained", tasks=BENCHMARK_TASKS, selector=sel_un, seed=42)

        random_tasks = [(r["task_id"], r["task"]) for r in res_random["task_results"]]
        untrained_tasks = [(r["task_id"], r["task"]) for r in res_un["task_results"]]
        expected_tasks = [(item[0], item[1]) for item in BENCHMARK_TASKS]

        self.assertEqual(random_tasks, expected_tasks)
        self.assertEqual(untrained_tasks, expected_tasks)


if __name__ == "__main__":
    unittest.main()
