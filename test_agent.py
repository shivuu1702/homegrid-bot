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


if __name__ == "__main__":
    unittest.main()
