"""Unit tests for the Linear SMDP Skill Selector, StateFeaturizer, and ActionMasker."""

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

# Add agent/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "agent"))

from rl_selector import (
    ActionMasker,
    LinearSMDPSkillSelector,
    SkillOption,
    StateFeaturizer,
    Subgoal,
    SubgoalTracker,
    get_all_standard_options,
    parse_compound_task,
    task_is_complete,
)


def _make_mock_state(
    agent_pos=(7, 5),
    agent_room="K",
    agent_dir=2,
    carrying=None,
    front_obj=None,
    objects=None,
):
    """Helper to construct a valid symbolic_state dictionary."""
    if objects is None:
        objects = [
            {
                "name": "bottle",
                "type": "Pickable",
                "pos": (3, 8),
                "room": "K",
                "state": None,
                "action": None,
                "invisible": False,
                "contains": None,
            },
            {
                "name": "fruit",
                "type": "Pickable",
                "pos": (4, 4),
                "room": "L",
                "state": None,
                "action": None,
                "invisible": False,
                "contains": None,
            },
            {
                "name": "trash bin",
                "type": "Storage",
                "pos": (11, 1),
                "room": "L",
                "state": "closed",
                "action": "pedal",
                "invisible": None,
                "contains": [],
            },
            {
                "name": "recycling bin",
                "type": "Storage",
                "pos": (10, 2),
                "room": "L",
                "state": "open",
                "action": "grasp",
                "invisible": None,
                "contains": ["papers"],
            },
            {
                "name": "compost bin",
                "type": "Storage",
                "pos": (12, 1),
                "room": "L",
                "state": "closed",
                "action": "lift",
                "invisible": None,
                "contains": [],
            },
            {
                "name": "plates",
                "type": "Pickable",
                "pos": (2, 9),
                "room": "K",
                "state": None,
                "action": None,
                "invisible": False,
                "contains": None,
            },
            {
                "name": "papers",
                "type": "Pickable",
                "pos": (-1, -1),
                "room": None,
                "state": None,
                "action": None,
                "invisible": False,
                "contains": None,
            },
        ]

    return {
        "step": 10,
        "agent": {
            "pos": agent_pos,
            "room": agent_room,
            "dir": agent_dir,
            "carrying": carrying,
        },
        "objects": objects,
        "front_obj": front_obj,
    }


class SkillOptionTests(unittest.TestCase):
    def test_canonical_string_roundtrip(self):
        opt = SkillOption.from_str("put_in_bin(bottle, trash bin)")
        self.assertEqual(opt.name, "put_in_bin")
        self.assertEqual(opt.args, ("bottle", "trash bin"))
        self.assertEqual(opt.id, "put_in_bin(bottle, trash bin)")

    def test_tuple_conversion_matches_agent_format(self):
        opt = SkillOption("pickup_object", ("fruit",))
        tup_name, tup_args = opt.to_tuple()
        self.assertEqual(tup_name, "pickup_object")
        self.assertEqual(tup_args, ["fruit"])

        reconstructed = SkillOption.from_tuple((tup_name, tup_args))
        self.assertEqual(opt, reconstructed)

    def test_standard_options_coverage(self):
        options = get_all_standard_options()
        # 7 find + 4 pickup + 3 open + 12 put_in_bin + 12 move_to_room = 38
        self.assertEqual(len(options), 38)
        names = {opt.name for opt in options}
        self.assertEqual(
            names,
            {"find_object", "pickup_object", "open_bin", "put_in_bin", "move_to_room"},
        )


class StateFeaturizerTests(unittest.TestCase):
    def setUp(self):
        self.featurizer = StateFeaturizer()
        self.state = _make_mock_state()

    def test_feature_vector_shape_and_range(self):
        task = "put the bottle in the trash bin"
        phi = self.featurizer.featurize(self.state, task)

        self.assertIsInstance(phi, np.ndarray)
        self.assertEqual(phi.ndim, 1)
        self.assertEqual(len(phi), self.featurizer.feature_dim)
        # All features should be within [0.0, 1.0]
        self.assertTrue(np.all(phi >= 0.0), f"Min value was {phi.min()}")
        self.assertTrue(np.all(phi <= 1.0), f"Max value was {phi.max()}")

    def test_deterministic_encoding(self):
        task = "move the fruit to the living room"
        phi1 = self.featurizer.featurize(self.state, task)
        phi2 = self.featurizer.featurize(self.state, task)
        self.assertTrue(np.array_equal(phi1, phi2))

    def test_task_type_and_target_features(self):
        phi_cleanup = self.featurizer.featurize(
            self.state, "put the bottle in the trash bin"
        )
        task_is_cleanup_idx = self.featurizer.feature_names.index("task_is_cleanup")
        target_bottle_idx = self.featurizer.feature_names.index(
            "task_target_obj_bottle"
        )
        target_trash_idx = self.featurizer.feature_names.index(
            "task_target_bin_trash_bin"
        )

        self.assertEqual(phi_cleanup[task_is_cleanup_idx], 1.0)
        self.assertEqual(phi_cleanup[target_bottle_idx], 1.0)
        self.assertEqual(phi_cleanup[target_trash_idx], 1.0)

        # Contrast with a find task
        phi_find = self.featurizer.featurize(self.state, "find the fruit")
        task_is_find_idx = self.featurizer.feature_names.index("task_is_find")
        target_fruit_idx = self.featurizer.feature_names.index("task_target_obj_fruit")
        self.assertEqual(phi_find[task_is_find_idx], 1.0)
        self.assertEqual(phi_find[target_fruit_idx], 1.0)
        self.assertEqual(phi_find[task_is_cleanup_idx], 0.0)

    def test_agent_carrying_relation_feature(self):
        state_carrying = _make_mock_state(carrying="bottle")
        phi = self.featurizer.featurize(state_carrying, "get the bottle")

        carrying_matches_idx = self.featurizer.feature_names.index(
            "carrying_matches_target_obj"
        )
        task_complete_idx = self.featurizer.feature_names.index("task_already_complete")

        self.assertEqual(phi[carrying_matches_idx], 1.0)
        self.assertEqual(phi[task_complete_idx], 1.0)

    def test_intercept_bias_always_one(self):
        phi = self.featurizer.featurize(self.state, "find the bottle")
        bias_idx = self.featurizer.feature_names.index("bias_intercept")
        self.assertEqual(phi[bias_idx], 1.0)


class ActionMaskerTests(unittest.TestCase):
    def setUp(self):
        self.masker = ActionMasker()
        self.state = _make_mock_state(carrying=None)

    def test_pickup_valid_when_hands_empty(self):
        opt = SkillOption("pickup_object", ("bottle",))
        self.assertTrue(self.masker.is_valid(opt, self.state))

    def test_pickup_invalid_when_carrying_another_item(self):
        state_full = _make_mock_state(carrying="fruit")
        opt = SkillOption("pickup_object", ("bottle",))
        self.assertFalse(self.masker.is_valid(opt, state_full))

    def test_open_bin_preconditions(self):
        # Trash bin is closed in mock state -> valid to open
        open_trash = SkillOption("open_bin", ("trash bin",))
        self.assertTrue(self.masker.is_valid(open_trash, self.state))

        # Recycling bin is already open in mock state -> invalid to open
        open_recycling = SkillOption("open_bin", ("recycling bin",))
        self.assertFalse(self.masker.is_valid(open_recycling, self.state))

    def test_put_in_bin_already_contained(self):
        # Papers are already inside recycling bin -> invalid
        opt = SkillOption("put_in_bin", ("papers", "recycling bin"))
        self.assertFalse(self.masker.is_valid(opt, self.state))

        # Bottle is on the floor and recycling bin is open -> valid
        opt_valid = SkillOption("put_in_bin", ("bottle", "recycling bin"))
        self.assertTrue(self.masker.is_valid(opt_valid, self.state))

    def test_move_to_room_already_there(self):
        # Fruit is already in living room ('L') and not carried -> invalid
        opt = SkillOption("move_to_room", ("fruit", "living room"))
        self.assertFalse(self.masker.is_valid(opt, self.state))

        # Fruit to kitchen -> valid
        opt_valid = SkillOption("move_to_room", ("fruit", "kitchen"))
        self.assertTrue(self.masker.is_valid(opt_valid, self.state))

    def test_find_object_carried_is_invalid(self):
        state_carried = _make_mock_state(carrying="bottle")
        opt = SkillOption("find_object", ("bottle",))
        self.assertFalse(self.masker.is_valid(opt, state_carried))

    def test_nonexistent_object_is_invalid(self):
        opt = SkillOption("pickup_object", ("nonexistent_sword",))
        self.assertFalse(self.masker.is_valid(opt, self.state))

    def test_empty_and_all_invalid_candidates(self):
        # Empty candidate list
        self.assertEqual(self.masker.get_valid_mask([], self.state), [])
        self.assertEqual(self.masker.get_valid_options([], self.state), [])

        # List of only invalid options
        invalid_candidates = [
            SkillOption("pickup_object", ("unknown_item",)),
            SkillOption("open_bin", ("recycling bin",)),  # already open
        ]
        self.assertEqual(
            self.masker.get_valid_mask(invalid_candidates, self.state),
            [False, False],
        )
        self.assertEqual(
            self.masker.get_valid_options(invalid_candidates, self.state),
            [],
        )


class LinearSMDPSkillSelectorTests(unittest.TestCase):
    def setUp(self):
        self.featurizer = StateFeaturizer()
        self.masker = ActionMasker()
        self.selector = LinearSMDPSkillSelector(
            featurizer=self.featurizer,
            masker=self.masker,
            epsilon=0.0,
            seed=42,
        )
        self.state = _make_mock_state()

    def test_zero_weights_initialization(self):
        opt = SkillOption("pickup_object", ("bottle",))
        w = self.selector.get_weights(opt)
        self.assertEqual(w.shape, (self.featurizer.feature_dim,))
        self.assertTrue(np.all(w == 0.0))

    def test_linear_q_value_computation(self):
        opt = SkillOption("pickup_object", ("bottle",))
        features = np.zeros(self.featurizer.feature_dim, dtype=np.float64)
        features[0] = 2.0
        features[1] = 3.0

        w = np.zeros(self.featurizer.feature_dim, dtype=np.float64)
        w[0] = 0.5
        w[1] = -1.0
        self.selector.set_weights(opt, w)

        expected_q = 0.5 * 2.0 + (-1.0) * 3.0  # -2.0
        q = self.selector.get_q_value(opt, features)
        self.assertAlmostEqual(q, expected_q)

    def test_greedy_selection_picks_highest_q(self):
        opt_a = SkillOption("pickup_object", ("bottle",))
        opt_b = SkillOption("find_object", ("bottle",))

        # Assign higher Q-value to opt_b
        w_b = np.zeros(self.featurizer.feature_dim, dtype=np.float64)
        bias_idx = self.featurizer.feature_names.index("bias_intercept")
        w_b[bias_idx] = 10.0
        self.selector.set_weights(opt_b, w_b)

        selected = self.selector.select_option(
            self.state,
            "find the bottle",
            candidates=[opt_a, opt_b],
            epsilon=0.0,
        )
        self.assertEqual(selected, opt_b)

    def test_action_masking_blocks_high_q_invalid_option(self):
        # Recycling bin is already open; set a massive Q-value on opening it
        opt_open_recycling = SkillOption("open_bin", ("recycling bin",))
        opt_open_trash = SkillOption("open_bin", ("trash bin",))

        w_rec = np.ones(self.featurizer.feature_dim, dtype=np.float64) * 100.0
        self.selector.set_weights(opt_open_recycling, w_rec)

        # Because recycling bin is already open, ActionMasker must prune it
        selected = self.selector.select_option(
            self.state,
            "open the recycling bin",
            candidates=[opt_open_recycling, opt_open_trash],
            epsilon=0.0,
        )
        self.assertEqual(selected, opt_open_trash)

    def test_empty_or_all_invalid_candidates_returns_none(self):
        # All invalid candidates
        invalid_candidates = [
            SkillOption("pickup_object", ("nonexistent",)),
            SkillOption("open_bin", ("recycling bin",)),  # already open
        ]
        chosen = self.selector.select_option(
            self.state,
            "dummy task",
            candidates=invalid_candidates,
        )
        self.assertIsNone(chosen)

        # Empty candidate list
        chosen_empty = self.selector.select_option(
            self.state,
            "dummy task",
            candidates=[],
        )
        self.assertIsNone(chosen_empty)

    def test_smdp_td_update(self):
        opt = SkillOption("pickup_object", ("bottle",))
        features = np.zeros(self.featurizer.feature_dim, dtype=np.float64)
        features[5] = 1.0

        # Perform TD update
        alpha = 0.1
        td_error = 2.0
        self.selector.smdp_td_update(opt, features, td_error, alpha)

        w = self.selector.get_weights(opt)
        self.assertAlmostEqual(w[5], alpha * td_error * 1.0)
        self.assertAlmostEqual(self.selector.get_q_value(opt, features), 0.2)

    def test_serialization_save_and_load(self):
        opt = SkillOption("put_in_bin", ("bottle", "trash bin"))
        w = np.random.default_rng(123).normal(size=self.featurizer.feature_dim)
        self.selector.set_weights(opt, w)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "rl_policy.json"
            self.selector.save(save_path)
            self.assertTrue(save_path.exists())

            # Load into new selector
            loaded_selector = LinearSMDPSkillSelector(
                featurizer=self.featurizer,
                masker=self.masker,
            )
            loaded_selector.load(save_path)

            loaded_w = loaded_selector.get_weights(opt)
            self.assertTrue(np.allclose(w, loaded_w))

    def test_explain_decision(self):
        opt = SkillOption("pickup_object", ("bottle",))
        features = self.featurizer.featurize(self.state, "get the bottle")

        w = np.zeros(self.featurizer.feature_dim, dtype=np.float64)
        target_idx = self.featurizer.feature_names.index("task_target_obj_bottle")
        w[target_idx] = 3.5
        self.selector.set_weights(opt, w)

        explanation = self.selector.explain_decision(
            self.state, "get the bottle", opt, top_k=3
        )
        self.assertEqual(explanation["option"], opt.id)
        self.assertAlmostEqual(explanation["q_value"], 3.5)
        self.assertEqual(len(explanation["top_features"]), 3)
        self.assertEqual(
            explanation["top_features"][0]["feature"], "task_target_obj_bottle"
        )
        self.assertAlmostEqual(explanation["top_features"][0]["contribution"], 3.5)


class MockEnvForSMDP:
    """Mock environment with deterministic step transitions and reward emission."""

    def __init__(self, step_rewards=None, state=None, task="put the bottle in the trash bin"):
        self.step_rewards = step_rewards or [0.0, 0.5, 1.0]
        self.step_idx = 0
        self.task = task
        self.state = state or _make_mock_state()

        class Unwrapped:
            def __init__(self, parent):
                self.parent = parent

            def get_full_symbolic_state(self):
                return self.parent.state

        self.unwrapped = Unwrapped(self)

    def reset(self):
        self.step_idx = 0
        return None, {"symbolic_state": self.state}

    def step(self, action):
        reward = self.step_rewards[self.step_idx] if self.step_idx < len(self.step_rewards) else 0.0
        self.step_idx += 1
        terminated = (self.step_idx >= len(self.step_rewards))
        truncated = False
        return None, reward, terminated, truncated, {"symbolic_state": self.state}


class SMDPTrackerAndTrainingTests(unittest.TestCase):
    def setUp(self):
        self.featurizer = StateFeaturizer()
        self.masker = ActionMasker()
        self.selector = LinearSMDPSkillSelector(
            featurizer=self.featurizer,
            masker=self.masker,
            epsilon=0.0,
            seed=101,
        )
        self.state = _make_mock_state()

    def test_reward_accumulation(self):
        from rl_selector import SMDPExecutionTracker, SkillOption

        mock_env = MockEnvForSMDP(step_rewards=[0.0, 0.5, 1.0])
        tracker = SMDPExecutionTracker(mock_env, gamma=0.9)

        opt = SkillOption("put_in_bin", ("bottle", "trash bin"))
        tracker.start_skill(opt)

        # 3 environment steps
        tracker.step(1)  # r=0.0, discount=0.9^0=1.0 -> 0.0
        tracker.step(1)  # r=0.5, discount=0.9^1=0.9 -> 0.45
        tracker.step(5)  # r=1.0, discount=0.9^2=0.81 -> 0.81

        transition = tracker.end_skill(success=True, task="put the bottle in the trash bin")

        self.assertEqual(transition.duration, 3)
        expected_discounted_reward = 0.0 * 1.0 + 0.5 * 0.9 + 1.0 * 0.81  # 1.26
        self.assertAlmostEqual(transition.reward, expected_discounted_reward)
        self.assertTrue(transition.done)

    def test_duration_discounting_in_update(self):
        from rl_selector import SMDPTransition, SkillOption

        opt = SkillOption("pickup_object", ("bottle",))
        next_state = _make_mock_state(carrying="bottle")

        # Set a Q-value of 10.0 on a valid next-state option
        next_opt = SkillOption("put_in_bin", ("bottle", "trash bin"))
        w_next = np.zeros(self.featurizer.feature_dim, dtype=np.float64)
        bias_idx = self.featurizer.feature_names.index("bias_intercept")
        w_next[bias_idx] = 10.0
        self.selector.set_weights(next_opt, w_next)

        duration = 3
        gamma = 0.9
        reward = 1.0
        # Target y = reward + (gamma^duration) * max_Q(s', a') = 1.0 + (0.9^3) * 10.0 = 1.0 + 7.29 = 8.29
        expected_target = reward + (gamma ** duration) * 10.0

        transition = SMDPTransition(
            state=self.state,
            option=opt,
            reward=reward,
            duration=duration,
            next_state=next_state,
            done=False,
            task="put the bottle in the trash bin",
            success=True,
        )

        alpha = 0.5
        td_error = self.selector.update_smdp(transition, alpha=alpha, gamma=gamma)

        self.assertAlmostEqual(td_error, expected_target - 0.0)
        # Weight should update by alpha * td_error * phi(s, task, option)
        phi_s = self.featurizer.featurize(self.state, "put the bottle in the trash bin", option=opt)
        expected_w = alpha * td_error * phi_s
        self.assertTrue(np.allclose(self.selector.get_weights(opt), expected_w))

    def test_terminal_transition_no_bootstrap(self):
        from rl_selector import SMDPTransition, SkillOption

        opt = SkillOption("put_in_bin", ("bottle", "trash bin"))
        # Next state has high Q-value
        next_state = _make_mock_state()
        w_high = np.ones(self.featurizer.feature_dim, dtype=np.float64) * 100.0
        self.selector.set_weights(opt, w_high)

        # Terminal transition with done=True
        reward = 1.0
        transition = SMDPTransition(
            state=self.state,
            option=opt,
            reward=reward,
            duration=5,
            next_state=next_state,
            done=True,  # Terminal!
            task="put the bottle in the trash bin",
            success=True,
        )

        # Re-zero weight for opt
        self.selector.set_weights(opt, np.zeros(self.featurizer.feature_dim))
        td_error = self.selector.update_smdp(transition, alpha=1.0, gamma=0.9)

        # Since done=True, target must be strictly reward (no bootstrap) -> td_error = reward - 0.0 = 1.0
        self.assertAlmostEqual(td_error, reward)

    def test_invalid_next_actions_fallback_to_zero_bootstrap(self):
        from rl_selector import SMDPTransition, SkillOption

        opt = SkillOption("pickup_object", ("bottle",))
        # Empty objects in next state so all candidates are invalid
        next_state = {
            "agent": {"pos": (0, 0), "room": "K", "dir": 0, "carrying": "bottle"},
            "objects": [],
            "front_obj": None,
        }

        transition = SMDPTransition(
            state=self.state,
            option=opt,
            reward=0.5,
            duration=2,
            next_state=next_state,
            done=False,  # Not terminal, but no valid next actions exist
            task="put the bottle in the trash bin",
            success=True,
        )

        self.selector.set_weights(opt, np.zeros(self.featurizer.feature_dim))
        td_error = self.selector.update_smdp(transition, alpha=1.0, gamma=0.9)

        # When no next actions are valid, Q_next_max = 0, so target = 0.5 + 0 = 0.5
        self.assertAlmostEqual(td_error, 0.5)

    def test_training_vs_evaluation_behavior(self):
        from rl_selector import SkillOption

        # Create two valid options: A and B
        opt_a = SkillOption("pickup_object", ("bottle",))
        opt_b = SkillOption("pickup_object", ("fruit",))

        # Give option A a higher Q-value
        w_a = np.zeros(self.featurizer.feature_dim, dtype=np.float64)
        bias_idx = self.featurizer.feature_names.index("bias_intercept")
        w_a[bias_idx] = 10.0
        self.selector.set_weights(opt_a, w_a)

        # Evaluation mode (epsilon=0.0): should ALWAYS select option A
        for _ in range(20):
            chosen = self.selector.select_option(
                self.state, "get the bottle", candidates=[opt_a, opt_b], epsilon=0.0
            )
            self.assertEqual(chosen, opt_a)

        # High exploration mode (epsilon=1.0): should choose both A and B over multiple samples
        selections = set()
        for _ in range(50):
            chosen = self.selector.select_option(
                self.state, "get the bottle", candidates=[opt_a, opt_b], epsilon=1.0
            )
            selections.add(chosen)
        self.assertEqual(selections, {opt_a, opt_b})

    def test_meaningful_choices_at_decision_point(self):
        from rl_selector import get_all_standard_options

        all_options = get_all_standard_options()
        valid = self.masker.get_valid_options(all_options, self.state, "put the bottle in the trash bin")

        # In standard HomeGrid states, there should be multiple valid skills available
        self.assertGreater(len(valid), 1, "The RL selector must have meaningful choices (>1 valid skill)")
        # Verify valid includes both pickup, find, and put_in_bin
        valid_names = {opt.name for opt in valid}
        self.assertTrue(len(valid_names) >= 2)

    def test_safe_weight_persistence_does_not_overwrite_skills_json(self):
        from config import SKILLS_FILE

        # Read skills.json before
        with open(SKILLS_FILE, "r", encoding="utf-8") as f:
            original_skills_json = f.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            policy_file = Path(tmp_dir) / "smdp_policy.json"
            self.selector.save(policy_file)

            self.assertTrue(policy_file.exists())
            # Ensure skills.json was untouched
            with open(SKILLS_FILE, "r", encoding="utf-8") as f:
                current_skills_json = f.read()
            self.assertEqual(original_skills_json, current_skills_json)

    def test_integration_agent_modes(self):
        from unittest.mock import patch
        import agent as agent_module

        # Mock environment that completes immediately
        done_state = _make_mock_state()
        # Set target object inside bin so task_is_complete is True
        for obj in done_state["objects"]:
            if obj["name"] == "trash bin":
                obj["contains"] = ["bottle"]

        env = MockEnvForSMDP(state=done_state, task="put the bottle in the trash bin")

        with patch.object(agent_module, "save_skill"):
            # 1. RL mode
            res_rl = agent_module.run_task(env, mode="rl", selector=self.selector)
            self.assertTrue(res_rl["success"])
            self.assertEqual(res_rl["mode"], "rl")

            # 2. Random mode
            res_rand = agent_module.run_task(env, mode="random", selector=self.selector)
            self.assertTrue(res_rand["success"])
            self.assertEqual(res_rand["mode"], "random")

            # 3. Hybrid mode
            res_hybrid = agent_module.run_task(env, mode="hybrid", selector=self.selector)
            self.assertTrue(res_hybrid["success"])
            self.assertIn("hybrid", res_hybrid["mode"])

    def test_task_skill_interaction_features(self):
        """Phase 3: Verify that task-skill interaction features accurately detect matched vs mismatched options."""
        task = "find the compost bin"
        opt_matched = SkillOption("find_object", ("compost bin",))
        opt_mismatched = SkillOption("put_in_bin", ("bottle", "trash bin"))

        phi_match = self.featurizer.featurize(self.state, task, option=opt_matched)
        phi_mismatch = self.featurizer.featurize(self.state, task, option=opt_mismatched)

        verb_match_idx = self.featurizer.feature_names.index("skill_verb_matches_task_verb")
        goal_act_idx = self.featurizer.feature_names.index("skill_is_direct_goal_action")
        mismatch_idx = self.featurizer.feature_names.index("skill_verb_mismatches_task")
        arg_bin_match_idx = self.featurizer.feature_names.index("skill_arg_matches_target_bin")

        self.assertEqual(phi_match[verb_match_idx], 1.0)
        self.assertEqual(phi_match[goal_act_idx], 1.0)
        self.assertEqual(phi_match[arg_bin_match_idx], 1.0)
        self.assertEqual(phi_match[mismatch_idx], 0.0)

        self.assertEqual(phi_mismatch[verb_match_idx], 0.0)
        self.assertEqual(phi_mismatch[goal_act_idx], 0.0)
        self.assertEqual(phi_mismatch[mismatch_idx], 1.0)

    def test_shared_weights_cross_task_generalization(self):
        """Phase 3: Verify that shared interaction weights allow zero-shot transfer to unseen options."""
        selector = LinearSMDPSkillSelector(featurizer=self.featurizer, masker=self.masker, epsilon=0.0)

        # Train on 'find the fruit' with find_object(fruit)
        opt_fruit = SkillOption("find_object", ("fruit",))
        phi_fruit = self.featurizer.featurize(self.state, "find the fruit", option=opt_fruit)
        # Update with positive reward TD error
        selector.smdp_td_update(opt_fruit, phi_fruit, td_error=2.0, alpha=0.5)

        # Now test on unseen task 'find the bottle'
        # find_object(bottle) was NEVER directly updated!
        opt_bottle = SkillOption("find_object", ("bottle",))
        opt_irrelevant = SkillOption("put_in_bin", ("bottle", "trash bin"))

        q_bottle = selector.get_q_value(
            opt_bottle,
            self.featurizer.featurize(self.state, "find the bottle", option=opt_bottle),
        )
        q_irrelevant = selector.get_q_value(
            opt_irrelevant,
            self.featurizer.featurize(self.state, "find the bottle", option=opt_irrelevant),
        )

        self.assertGreater(
            q_bottle,
            q_irrelevant,
            f"Unseen option find_object(bottle) Q={q_bottle} should exceed irrelevant Q={q_irrelevant}",
        )

    def test_candidate_argument_preservation(self):
        """Phase 3: Verify that candidate generation and masking preserve correct targets for all 5 task types."""
        tasks_and_expected_opts = [
            ("find the compost bin", SkillOption("find_object", ("compost bin",))),
            ("get the fruit", SkillOption("pickup_object", ("fruit",))),
            ("open the trash bin", SkillOption("open_bin", ("trash bin",))),
            ("put the bottle in the trash bin", SkillOption("put_in_bin", ("bottle", "trash bin"))),
            ("move the plates to the dining room", SkillOption("move_to_room", ("plates", "dining room"))),
        ]
        all_standard = get_all_standard_options()
        for task_str, expected_opt in tasks_and_expected_opts:
            self.assertIn(expected_opt, all_standard, f"{expected_opt} missing from standard options for {task_str}")
            valid_opts = self.masker.get_valid_options(all_standard, self.state, task_str)
            self.assertIn(expected_opt, valid_opts, f"{expected_opt} was incorrectly pruned for {task_str}")


class CompoundTaskSubgoalTests(unittest.TestCase):
    """Unit and regression tests for compound task parsing, subgoal tracking, and action masking."""

    def setUp(self):
        self.featurizer = StateFeaturizer()
        self.masker = ActionMasker()
        self.state = _make_mock_state(carrying=None)

    def test_compound_task_parsing_ordered_subgoals(self):
        # 1. fruit to recycling bin
        t1 = "find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside"
        sg1 = parse_compound_task(t1)
        self.assertIsNotNone(sg1)
        self.assertEqual(len(sg1), 5)
        self.assertEqual(sg1[0].name, "find_object")
        self.assertEqual(sg1[0].args, ("fruit",))
        self.assertEqual(sg1[1].name, "pickup_object")
        self.assertEqual(sg1[1].args, ("fruit",))
        self.assertEqual(sg1[2].name, "move_to_room")
        self.assertEqual(sg1[2].args, ("fruit", "kitchen"))
        self.assertEqual(sg1[3].name, "open_bin")
        self.assertEqual(sg1[3].args, ("recycling bin",))
        self.assertEqual(sg1[4].name, "put_in_bin")
        self.assertEqual(sg1[4].args, ("fruit", "recycling bin"))

        # 2. papers to trash bin
        t2 = "find the papers, pick them up, move to the kitchen, open the trash bin, and put the papers inside"
        sg2 = parse_compound_task(t2)
        self.assertIsNotNone(sg2)
        self.assertEqual(len(sg2), 5)
        self.assertEqual(sg2[0].id, "find_object(papers)")
        self.assertEqual(sg2[1].id, "pickup_object(papers)")
        self.assertEqual(sg2[2].id, "move_to_room(papers, kitchen)")
        self.assertEqual(sg2[3].id, "open_bin(trash bin)")
        self.assertEqual(sg2[4].id, "put_in_bin(papers, trash bin)")

        # 3. bottle to compost bin
        t3 = "find the bottle, pick it up, move to the living room, open the compost bin, and put the bottle inside"
        sg3 = parse_compound_task(t3)
        self.assertIsNotNone(sg3)
        self.assertEqual(len(sg3), 5)
        self.assertEqual(sg3[0].id, "find_object(bottle)")
        self.assertEqual(sg3[1].id, "pickup_object(bottle)")
        self.assertEqual(sg3[2].id, "move_to_room(bottle, living room)")
        self.assertEqual(sg3[3].id, "open_bin(compost bin)")
        self.assertEqual(sg3[4].id, "put_in_bin(bottle, compost bin)")

    def test_canonical_tasks_return_none_from_parse_compound(self):
        canonical = [
            "find the fruit",
            "get the fruit",
            "open the recycling bin",
            "put the fruit in the recycling bin",
            "move the plates to the kitchen",
        ]
        for c in canonical:
            self.assertIsNone(parse_compound_task(c))

    def test_unrelated_skills_excluded_by_active_subgoal(self):
        # When active subgoal is pickup_object(fruit)
        subgoal = Subgoal("pickup_object", ("fruit",), "Pick up fruit")
        all_options = get_all_standard_options()
        valid = self.masker.get_valid_options(all_options, self.state, active_subgoal=subgoal)

        # Fruit pickup should be valid
        self.assertIn(SkillOption("pickup_object", ("fruit",)), valid)
        # Unrelated pickables must NEVER be valid
        self.assertNotIn(SkillOption("pickup_object", ("papers",)), valid)
        self.assertNotIn(SkillOption("pickup_object", ("bottle",)), valid)
        self.assertNotIn(SkillOption("pickup_object", ("plates",)), valid)
        # Unrelated skills on other entities must NEVER be valid
        self.assertNotIn(SkillOption("find_object", ("recycling bin",)), valid)
        self.assertNotIn(SkillOption("find_object", ("bottle",)), valid)
        self.assertNotIn(SkillOption("move_to_room", ("papers", "living room")), valid)
        self.assertNotIn(SkillOption("open_bin", ("trash bin",)), valid)
        self.assertNotIn(SkillOption("put_in_bin", ("bottle", "compost bin")), valid)

    def test_subgoal_tracker_advances_only_on_verified_success(self):
        t = "find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside"
        subgoals = parse_compound_task(t)
        tracker = SubgoalTracker(t, subgoals)

        # Initial subgoal: find_object(fruit)
        self.assertEqual(tracker.active_subgoal.id, "find_object(fruit)")

        # Case 1: Failed skill execution -> must NOT advance
        state_unfacing = _make_mock_state(front_obj=None)
        advanced = tracker.step(
            SkillOption("find_object", ("fruit",)),
            skill_success=False,
            next_state=state_unfacing,
        )
        self.assertFalse(advanced)
        self.assertEqual(tracker.active_subgoal.id, "find_object(fruit)")

        # Case 2: Skill claims success, but postcondition not met -> must NOT advance
        advanced = tracker.step(
            SkillOption("find_object", ("fruit",)),
            skill_success=True,
            next_state=state_unfacing,
        )
        self.assertFalse(advanced)
        self.assertEqual(tracker.active_subgoal.id, "find_object(fruit)")

        # Case 3: Verified success -> advances to pickup_object(fruit)
        state_facing = _make_mock_state(front_obj="fruit")
        advanced = tracker.step(
            SkillOption("find_object", ("fruit",)),
            skill_success=True,
            next_state=state_facing,
        )
        self.assertTrue(advanced)
        self.assertEqual(tracker.active_subgoal.id, "pickup_object(fruit)")

        # Case 4: Pick up fruit succeeds (in living room)
        state_carrying = _make_mock_state(carrying="fruit", agent_room="L")
        advanced = tracker.step(
            SkillOption("pickup_object", ("fruit",)),
            skill_success=True,
            next_state=state_carrying,
        )
        self.assertTrue(advanced)
        self.assertEqual(tracker.active_subgoal.id, "move_to_room(fruit, kitchen)")

        # Case 5: Move to kitchen succeeds (bin is closed)
        state_kitchen = _make_mock_state(
            carrying="fruit",
            agent_room="K",
            objects=[
                {
                    "name": "recycling bin",
                    "type": "Storage",
                    "pos": (10, 2),
                    "room": "K",
                    "state": "closed",
                    "action": "pedal",
                    "invisible": None,
                    "contains": [],
                }
            ],
        )
        advanced = tracker.step(
            SkillOption("move_to_room", ("fruit", "kitchen")),
            skill_success=True,
            next_state=state_kitchen,
        )
        self.assertTrue(advanced)
        self.assertEqual(tracker.active_subgoal.id, "open_bin(recycling bin)")

    def test_compound_task_is_complete_verification(self):
        t = "find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside"
        # Not complete when fruit is on floor
        s_initial = _make_mock_state(carrying=None)
        self.assertFalse(task_is_complete(t, s_initial))

        # Complete when fruit is inside recycling bin
        s_final = _make_mock_state(
            carrying=None,
            objects=[
                {
                    "name": "recycling bin",
                    "type": "Storage",
                    "pos": (10, 2),
                    "room": "L",
                    "state": "open",
                    "contains": ["fruit"],
                }
            ],
        )
        self.assertTrue(task_is_complete(t, s_final))


if __name__ == "__main__":
    unittest.main()
