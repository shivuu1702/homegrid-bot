"""Provider-agnostic LLM configuration and adapter tests. No network calls."""
import sys
import unittest

sys.path.insert(0, "agent")

from config import LLMConfig, load_llm_config
from llm import (
    GeminiAdapter,
    LLMConfigurationError,
    OpenAICompatibleAdapter,
    _normalise_provider_error,
    get_provider_adapter,
    validate_llm_config,
)


class LLMConfigurationTests(unittest.TestCase):
    def test_loads_unified_env_values(self):
        config = load_llm_config({
            "LLM_PROVIDER": "groq",
            "LLM_MODEL": "a-model",
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://example.test/v1",
        })
        self.assertEqual(config, LLMConfig("groq", "a-model", "test-key", "https://example.test/v1"))

    def test_preserves_legacy_gemini_configuration(self):
        config = load_llm_config({"OPENAI_MODEL": "gemini-2.5-flash", "OPENAI_API_KEY": "test-key"})
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, "gemini-2.5-flash")

    def test_validation_errors_are_clear(self):
        cases = [
            (LLMConfig("unknown", "model", "key"), "Unsupported"),
            (LLMConfig("openai", "model", ""), "LLM_API_KEY"),
            (LLMConfig("openai", "", "key"), "LLM_MODEL"),
            (LLMConfig("groq", "model", "key"), "LLM_BASE_URL"),
        ]
        for config, expected in cases:
            with self.subTest(config=config), self.assertRaisesRegex(LLMConfigurationError, expected):
                validate_llm_config(config)


class ProviderSelectionTests(unittest.TestCase):
    def test_gemini_uses_its_native_adapter(self):
        adapter = get_provider_adapter(LLMConfig("gemini", "test-model", "test-key"))
        self.assertIsInstance(adapter, GeminiAdapter)

    def test_openai_compatible_providers_share_one_adapter(self):
        configs = [
            LLMConfig("openai", "test-model", "test-key"),
            LLMConfig("openrouter", "test-model", "test-key", "https://example.test/v1"),
            LLMConfig("groq", "test-model", "test-key", "https://example.test/v1"),
        ]
        for config in configs:
            with self.subTest(provider=config.provider):
                self.assertIsInstance(get_provider_adapter(config), OpenAICompatibleAdapter)

    def test_temporary_statuses_become_retryable_errors(self):
        for status in (429, 503):
            with self.subTest(status=status):
                error = _normalise_provider_error("mock", Exception(f"HTTP {status}"))
                self.assertTrue(error.retryable)
                self.assertIn("Try again", str(error))


if __name__ == "__main__":
    unittest.main()
