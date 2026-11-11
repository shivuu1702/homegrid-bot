"""Unit tests for agent parsing and graceful LLM-failure handling."""
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "agent")

import agent as agent_module
from llm import LLMError


class AgentTests(unittest.TestCase):
    def test_parse_plan_ignores_unknown_skills(self):
        plan = agent_module._parse_plan(
            "1. pickup_object(fruit)\n2. invented_skill(nope)\n3. open_bin(compost bin)"
        )
        self.assertEqual(plan, [("pickup_object", ["fruit"]), ("open_bin", ["compost bin"])])

    def test_transient_llm_failure_returns_failed_task_not_traceback(self):
        class FakeEnv:
            task = "find the bottle"

            class unwrapped:
                @staticmethod
                def get_full_symbolic_state():
                    return {"agent": {"carrying": None}, "objects": [], "front_obj": None}

        messages = []
        with patch.object(agent_module, "ask_llm", side_effect=LLMError("temporarily unavailable", retryable=True)), \
             patch.object(agent_module, "save_skill"):
            result = agent_module.run_task(FakeEnv(), log=messages.append)

        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], 3)
        self.assertTrue(any("LLM unavailable" in message for message in messages))




class TaskPreservationRegressionTests(unittest.TestCase):
    """Regression tests verifying that explicitly requested CLI tasks reach RL,

    hybrid, random, and LLM execution flows unchanged.
    """

    class MockMultitaskWrapper:
        def __init__(self, initial_task="put the bottle in the compost bin"):
            self.task = initial_task
            self.reward_fn = lambda s: 0
            self.dist_goal = lambda s: 0
            self.subtask_done = False
            self.step_cnt = 0

        def sample_task(self):
            pass

    class MockEnv:
        def __init__(self, initial_task="put the bottle in the compost bin"):
            self.task = initial_task
            self.wrapper = TaskPreservationRegressionTests.MockMultitaskWrapper(initial_task)
            self.env = self.wrapper

        class unwrapped:
            @staticmethod
            def get_full_symbolic_state():
                return {
                    "agent": {"carrying": None, "pos": (7, 5), "room": "K"},
                    "objects": [
                        {"name": "fruit", "type": "Pickable", "pos": (4, 4), "room": "L"},
                        {"name": "bottle", "type": "Pickable", "pos": (3, 8), "room": "K"},
                        {"name": "recycling bin", "type": "Storage", "pos": (10, 2), "room": "L", "state": "closed", "contains": []},
                        {"name": "compost bin", "type": "Storage", "pos": (12, 1), "room": "L", "state": "closed", "contains": []},
                    ],
                    "front_obj": None,
                }

    def test_run_task_preserves_explicit_task_in_rl_mode(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        captured_tasks = []

        def mock_run_smdp(e, selector=None, task=None, **kwargs):
            captured_tasks.append(task)
            return {
                "task": task,
                "success": True,
                "total_reward": 1.0,
                "total_steps": 4,
                "skills_executed": ["find_object(fruit)"],
                "failed_skills": 0,
            }

        with patch.object(agent_module, "run_smdp_episode", side_effect=mock_run_smdp), \
             patch.object(agent_module, "save_skill"):
            result = agent_module.run_task(
                env,
                mode="rl",
                task=requested_task,
                log=lambda *a: None,
            )

        self.assertEqual(captured_tasks, [requested_task])
        self.assertEqual(result["task"], requested_task)
        self.assertEqual(env.task, requested_task)
        self.assertEqual(env.wrapper.task, requested_task)

    def test_run_task_preserves_explicit_task_in_random_mode(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        captured_tasks = []

        def mock_run_smdp(e, selector=None, task=None, **kwargs):
            captured_tasks.append(task)
            return {
                "task": task,
                "success": True,
                "total_reward": 1.0,
                "total_steps": 4,
                "skills_executed": [],
                "failed_skills": 0,
            }

        with patch.object(agent_module, "run_smdp_episode", side_effect=mock_run_smdp), \
             patch.object(agent_module, "save_skill"):
            result = agent_module.run_task(
                env,
                mode="random",
                task=requested_task,
                log=lambda *a: None,
            )

        self.assertEqual(captured_tasks, [requested_task])
        self.assertEqual(result["task"], requested_task)
        self.assertEqual(env.task, requested_task)

    def test_run_task_preserves_explicit_task_in_hybrid_mode(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        captured_tasks = []

        def mock_run_smdp(e, selector=None, task=None, **kwargs):
            captured_tasks.append(task)
            return {
                "task": task,
                "success": True,
                "total_reward": 1.0,
                "total_steps": 4,
                "skills_executed": [],
                "failed_skills": 0,
            }

        with patch.object(agent_module, "run_smdp_episode", side_effect=mock_run_smdp), \
             patch.object(agent_module, "save_skill"):
            result = agent_module.run_task(
                env,
                mode="hybrid",
                selector=object(),
                task=requested_task,
                log=lambda *a: None,
            )

        self.assertEqual(captured_tasks, [requested_task])
        self.assertEqual(result["task"], requested_task)
        self.assertEqual(env.task, requested_task)

    def test_run_task_preserves_explicit_task_in_llm_mode(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        captured_planned_tasks = []

        def mock_plan_task(task, state, log):
            captured_planned_tasks.append(task)
            return [("open_bin", ["recycling bin"])]

        def mock_execute_plan(e, plan, task, log):
            return True, 1.0, ""

        with patch.object(agent_module, "plan_task", side_effect=mock_plan_task), \
             patch.object(agent_module, "execute_plan", side_effect=mock_execute_plan), \
             patch.object(agent_module, "save_skill"):
            result = agent_module.run_task(
                env,
                mode="llm",
                task=requested_task,
                log=lambda *a: None,
            )

        self.assertEqual(captured_planned_tasks, [requested_task])
        self.assertEqual(result["task"], requested_task)
        self.assertEqual(env.task, requested_task)

    def test_set_env_task_configures_multitask_wrapper_reward_fn(self):
        env = self.MockEnv("find the bottle")
        agent_module.set_env_task(env, "put the fruit in the recycling bin")

        self.assertEqual(env.task, "put the fruit in the recycling bin")
        self.assertEqual(env.wrapper.task, "put the fruit in the recycling bin")

        # Test reward_fn
        # 1. Carrying fruit gives 0.5 subtask reward
        state_carrying = {
            "agent": {"carrying": "fruit", "pos": (7, 5)},
            "objects": [{"name": "recycling bin", "contains": []}],
        }
        self.assertEqual(env.wrapper.reward_fn(state_carrying), 0.5)

        # 2. Fruit in recycling bin gives 1.0 terminal reward
        state_done = {
            "agent": {"carrying": None, "pos": (7, 5)},
            "objects": [{"name": "recycling bin", "contains": ["fruit"]}],
        }
        self.assertEqual(env.wrapper.reward_fn(state_done), 1.0)

        # 3. Unrelated state gives 0
        state_empty = {
            "agent": {"carrying": None, "pos": (7, 5)},
            "objects": [{"name": "recycling bin", "contains": []}],
        }
        self.assertEqual(env.wrapper.reward_fn(state_empty), 0)

    def test_prepare_env_for_task_finds_required_objects_and_sets_task(self):
        
        class AlternatingEnv:
            def __init__(self):
                self.reset_count = 0
                self.task = "initial_random_task"
                self.env = TaskPreservationRegressionTests.MockMultitaskWrapper(self.task)

            def reset(self):
                self.reset_count += 1
                if self.reset_count < 3:
                    # Missing fruit and recycling bin
                    sym = {
                        "objects": [
                            {"name": "bottle", "type": "Pickable"},
                            {"name": "trash bin", "type": "Storage"},
                        ]
                    }
                else:
                    # Required objects present
                    sym = {
                        "objects": [
                            {"name": "fruit", "type": "Pickable"},
                            {"name": "recycling bin", "type": "Storage"},
                        ]
                    }
                return None, {"symbolic_state": sym}

        test_env = AlternatingEnv()
        ok = agent_module.prepare_env_for_task(test_env, "put the fruit in the recycling bin")

        self.assertTrue(ok)
        self.assertEqual(test_env.reset_count, 3)
        self.assertEqual(test_env.task, "put the fruit in the recycling bin")
        self.assertEqual(test_env.env.task, "put the fruit in the recycling bin")


    def test_hybrid_mode_success_without_api_key(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        # Mock RL to succeed
        def mock_run_smdp(e, selector=None, task=None, **kwargs):
            return {
                "task": task,
                "success": True,
                "total_reward": 1.0,
                "total_steps": 4,
                "skills_executed": ["find_object(fruit)", "pickup_object(fruit)"],
                "failed_skills": 0,
            }

        validate_called = []
        def mock_validate():
            validate_called.append(True)
            raise agent_module.LLMConfigurationError("LLM_API_KEY is missing for provider 'openai'. Set it in .env.")

        with patch.object(agent_module, "run_smdp_episode", side_effect=mock_run_smdp), \
             patch.object(agent_module, "validate_llm_config", side_effect=mock_validate), \
             patch.object(agent_module, "save_skill"):
            result = agent_module.run_task(
                env,
                mode="hybrid",
                selector=object(),
                task=requested_task,
                log=lambda *a: None,
            )

        # RL succeeded, so LLM config should NEVER be validated
        self.assertEqual(validate_called, [])
        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "hybrid (rl-success)")
        self.assertEqual(result["task"], requested_task)
        self.assertEqual(result["steps"], 4)

    def test_hybrid_mode_fallback_with_missing_api_key_reports_clear_error(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        # Mock RL to fail
        def mock_run_smdp(e, selector=None, task=None, **kwargs):
            return {
                "task": task,
                "success": False,
                "total_reward": 0.2,
                "total_steps": 12,
                "skills_executed": ["find_object(fruit)"],
                "failed_skills": 1,
            }

        logged_messages = []
        def mock_validate():
            raise agent_module.LLMConfigurationError("LLM_API_KEY is missing for provider 'openai'. Set it in .env.")

        with patch.object(agent_module, "run_smdp_episode", side_effect=mock_run_smdp), \
             patch.object(agent_module, "validate_llm_config", side_effect=mock_validate):
            result = agent_module.run_task(
                env,
                mode="hybrid",
                selector=object(),
                task=requested_task,
                log=logged_messages.append,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["mode"], "hybrid (llm-fallback-unavailable)")
        self.assertEqual(result["task"], requested_task)
        self.assertIn("LLM fallback unavailable: LLM_API_KEY is missing for provider 'openai'", result["error"])
        self.assertEqual(result["steps"], 12)
        self.assertEqual(result["plan"], ["find_object(fruit)"])
        self.assertTrue(any("LLM fallback unavailable" in msg for msg in logged_messages))

    def test_hybrid_mode_fallback_with_valid_api_key_proceeds_to_llm(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        # Mock RL to fail
        def mock_run_smdp(e, selector=None, task=None, **kwargs):
            return {
                "task": task,
                "success": False,
                "total_reward": 0.0,
                "total_steps": 5,
                "skills_executed": [],
                "failed_skills": 1,
            }

        def mock_validate():
            return True

        def mock_plan_task(task, state, log):
            return [("find_object", ["fruit"])]

        def mock_execute_plan(e, plan, task, log):
            return True, 1.0, ""

        with patch.object(agent_module, "run_smdp_episode", side_effect=mock_run_smdp), \
             patch.object(agent_module, "validate_llm_config", side_effect=mock_validate), \
             patch.object(agent_module, "plan_task", side_effect=mock_plan_task), \
             patch.object(agent_module, "execute_plan", side_effect=mock_execute_plan), \
             patch.object(agent_module, "save_skill"):
            result = agent_module.run_task(
                env,
                mode="hybrid",
                selector=object(),
                task=requested_task,
                log=lambda *a: None,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "hybrid (llm-fallback-success)")
        self.assertEqual(result["task"], requested_task)

    def test_hybrid_mode_fallback_with_missing_gemini_key_reports_clear_error(self):
        env = self.MockEnv("put the bottle in the compost bin")
        requested_task = "put the fruit in the recycling bin"

        def mock_run_smdp(e, selector=None, task=None, **kwargs):
            return {
                "task": task,
                "success": False,
                "total_reward": 0.0,
                "total_steps": 4,
                "skills_executed": [],
                "failed_skills": 1,
            }

        logged_messages = []
        def mock_validate():
            raise agent_module.LLMConfigurationError(
                "GEMINI_API_KEY (or LLM_API_KEY) is missing for provider 'gemini'. Set it in .env."
            )

        with patch.object(agent_module, "run_smdp_episode", side_effect=mock_run_smdp), \
             patch.object(agent_module, "validate_llm_config", side_effect=mock_validate):
            result = agent_module.run_task(
                env,
                mode="hybrid",
                selector=object(),
                task=requested_task,
                log=logged_messages.append,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["mode"], "hybrid (llm-fallback-unavailable)")
        self.assertIn("GEMINI_API_KEY", result["error"])
        self.assertTrue(any("LLM fallback unavailable" in msg for msg in logged_messages))

    def test_prepare_env_for_compound_task_finds_required_entities(self):
        class AlternatingEnv:
            def __init__(self):
                self.reset_count = 0
                self.task = "initial_random_task"
                self.env = TaskPreservationRegressionTests.MockMultitaskWrapper(self.task)

            def reset(self):
                self.reset_count += 1
                if self.reset_count < 2:
                    sym = {"objects": [{"name": "bottle", "type": "Pickable"}]}
                else:
                    sym = {
                        "objects": [
                            {"name": "fruit", "type": "Pickable"},
                            {"name": "recycling bin", "type": "Storage"},
                        ]
                    }
                return None, {"symbolic_state": sym}

        test_env = AlternatingEnv()
        compound_task = "find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside"
        success = agent_module.prepare_env_for_task(test_env, compound_task, max_resets=10, log=lambda *a: None)
        self.assertTrue(success)
        self.assertEqual(test_env.task, compound_task)
        self.assertEqual(test_env.reset_count, 2)

    def test_task_is_complete_for_compound_cleanup_task(self):
        compound_task = "find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside"
        state_incomplete = self.MockEnv.unwrapped.get_full_symbolic_state()
        self.assertFalse(agent_module.task_is_complete(compound_task, state_incomplete))

        state_complete = {
            "agent": {"carrying": None, "pos": (10, 2), "room": "K"},
            "objects": [
                {"name": "recycling bin", "type": "Storage", "pos": (10, 2), "room": "K", "state": "open", "contains": ["fruit"]},
                {"name": "fruit", "type": "Pickable", "pos": (-1, -1), "room": None},
            ],
            "front_obj": "recycling bin",
        }
        self.assertTrue(agent_module.task_is_complete(compound_task, state_complete))


if __name__ == "__main__":
    unittest.main()
