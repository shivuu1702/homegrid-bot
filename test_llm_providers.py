"""Provider-agnostic LLM configuration and adapter tests. No network calls."""
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "agent")

from config import DEFAULT_GEMINI_MODEL, LLMConfig, load_llm_config
from llm import (
    GeminiAdapter,
    LLMConfigurationError,
    LLMError,
    OpenAICompatibleAdapter,
    _normalise_provider_error,
    get_provider_adapter,
    test_llm_connection,
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
        self.assertEqual(config.api_key, "test-key")

    def test_loads_gemini_with_gemini_api_key_and_default_model(self):
        config = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-gemini-key",
        })
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, DEFAULT_GEMINI_MODEL)
        self.assertEqual(config.api_key, "test-gemini-key")

    def test_loads_gemini_with_custom_gemini_model(self):
        config = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-gemini-key",
            "GEMINI_MODEL": "gemini-2.0-flash",
        })
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, "gemini-2.0-flash")
        self.assertEqual(config.api_key, "test-gemini-key")

    def test_gemini_model_precedence_over_llm_model(self):
        config = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.0-flash",
            "LLM_MODEL": "gemini-1.5-flash",
            "GEMINI_API_KEY": "test-key",
        })
        self.assertEqual(config.model, "gemini-2.0-flash")

    def test_gemini_api_key_precedence_over_llm_api_key(self):
        config = load_llm_config({
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "gemini-specific-key",
            "LLM_API_KEY": "generic-key",
        })
        self.assertEqual(config.api_key, "gemini-specific-key")

    def test_gemini_auto_detect_from_gemini_api_key(self):
        config = load_llm_config({"GEMINI_API_KEY": "auto-key"})
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, DEFAULT_GEMINI_MODEL)
        self.assertEqual(config.api_key, "auto-key")

    def test_validation_errors_are_clear(self):
        cases = [
            (LLMConfig("unknown", "model", "key"), "Unsupported"),
            (LLMConfig("openai", "model", ""), "LLM_API_KEY"),
            (LLMConfig("openai", "", "key"), "LLM_MODEL"),
            (LLMConfig("gemini", "model", ""), "GEMINI_API_KEY"),
            (LLMConfig("gemini", "", "key"), "LLM_MODEL"),
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


class GeminiAdapterMockedTests(unittest.TestCase):
    """Unit tests for GeminiAdapter using mocks (no network calls)."""

    @patch("google.genai.Client")
    def test_gemini_complete_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "pickup_object(bottle)"
        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
        result = adapter.complete("Find the bottle")

        self.assertEqual(result, "pickup_object(bottle)")
        self.assertTrue(mock_client.models.generate_content.called)
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(call_kwargs["contents"], "Find the bottle")

    @patch("google.genai.Client")
    def test_gemini_complete_empty_response_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "   "
        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
        with self.assertRaisesRegex(LLMError, "empty response"):
            adapter.complete("prompt")

    @patch("google.genai.Client")
    def test_gemini_complete_none_text_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
        with self.assertRaisesRegex(LLMError, "empty response"):
            adapter.complete("prompt")

    def test_gemini_missing_api_key_at_adapter(self):
        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", ""))
        with self.assertRaisesRegex(LLMError, "API key is missing"):
            adapter.complete("prompt")

    @patch("google.genai.Client")
    def test_gemini_complete_quota_error_is_retryable(self, mock_client_cls):
        from google.genai import errors
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = errors.APIError(
            429, {"error": {"message": "RESOURCE_EXHAUSTED"}}
        )

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
        with self.assertRaises(LLMError) as ctx:
            adapter.complete("prompt")
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("rate limit", str(ctx.exception).lower())

    @patch("google.genai.Client")
    def test_gemini_complete_auth_error_is_not_retryable(self, mock_client_cls):
        from google.genai import errors
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = errors.APIError(
            401, {"error": {"message": "API_KEY_INVALID"}}
        )

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
        with self.assertRaises(LLMError) as ctx:
            adapter.complete("prompt")
        self.assertFalse(ctx.exception.retryable)
        self.assertIn("Authentication failed", str(ctx.exception))

    @patch("google.genai.Client")
    def test_gemini_complete_permission_denied_is_not_retryable(self, mock_client_cls):
        from google.genai import errors
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = errors.APIError(
            403, {"error": {"message": "PERMISSION_DENIED"}}
        )

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
        with self.assertRaises(LLMError) as ctx:
            adapter.complete("prompt")
        self.assertFalse(ctx.exception.retryable)
        self.assertIn("Permission denied", str(ctx.exception))

    @patch("google.genai.Client")
    def test_gemini_complete_server_error_is_retryable(self, mock_client_cls):
        from google.genai import errors
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = errors.APIError(
            500, {"error": {"message": "INTERNAL_SERVER_ERROR"}}
        )

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
        with self.assertRaises(LLMError) as ctx:
            adapter.complete("prompt")
        self.assertTrue(ctx.exception.retryable)

    def test_gemini_complete_import_error_handled(self):
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", "test-key"))
            with self.assertRaisesRegex(LLMError, "google-genai"):
                adapter.complete("prompt")

    @patch("google.genai.Client")
    def test_secret_key_never_leaked_in_exceptions(self, mock_client_cls):
        fake_secret = "AIzaSySecretKeyValueDoNotLeak999"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = RuntimeError(
            f"Network error calling https://generativelanguage.googleapis.com/v1beta/models?key={fake_secret}"
        )

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-2.5-flash", fake_secret))
        with self.assertRaises(LLMError) as ctx:
            adapter.complete("prompt")

        error_message = str(ctx.exception)
        self.assertNotIn(fake_secret, error_message)
        self.assertIn("gemini: request failed (RuntimeError).", error_message)


class GeminiConnectionTestTests(unittest.TestCase):
    """Mocked regression tests specifically for the Gemini connection-test path."""

    @patch("google.genai.Client")
    def test_gemini_test_connection_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "pong"
        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-3.8-flash", "test-key"))
        result = adapter.test_connection()

        self.assertTrue(result)
        self.assertTrue(mock_client.models.generate_content.called)
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs["model"], "gemini-3.8-flash")
        self.assertEqual(call_kwargs["contents"], "ping")
        # Ensure connection check does not bundle heavy task-planning parameters
        self.assertNotIn("config", call_kwargs)

    @patch("google.genai.Client")
    def test_gemini_test_connection_candidates_without_text(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [MagicMock()]
        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-3.8-flash", "test-key"))
        result = adapter.test_connection()
        self.assertTrue(result)

    @patch("google.genai.Client")
    def test_gemini_test_connection_empty_response_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = None
        mock_client.models.generate_content.return_value = mock_response

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-3.8-flash", "test-key"))
        with self.assertRaisesRegex(LLMError, "empty response"):
            adapter.test_connection()

    @patch("google.genai.Client")
    def test_gemini_test_connection_404_does_not_mention_base_url(self, mock_client_cls):
        from google.genai import errors
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = errors.APIError(
            404, {"error": {"message": "models/gemini-3.8-flash is not found"}}
        )

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-3.8-flash", "test-key"))
        with self.assertRaises(LLMError) as ctx:
            adapter.test_connection()

        err_msg = str(ctx.exception)
        self.assertNotIn("LLM_BASE_URL", err_msg)
        self.assertIn("GEMINI_MODEL", err_msg)
        self.assertFalse(ctx.exception.retryable)

    @patch("google.genai.Client")
    def test_gemini_test_connection_auth_error(self, mock_client_cls):
        from google.genai import errors
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = errors.APIError(
            401, {"error": {"message": "API_KEY_INVALID"}}
        )

        adapter = GeminiAdapter(LLMConfig("gemini", "gemini-3.8-flash", "test-key"))
        with self.assertRaises(LLMError) as ctx:
            adapter.test_connection()

        err_msg = str(ctx.exception)
        self.assertIn("GEMINI_API_KEY", err_msg)
        self.assertFalse(ctx.exception.retryable)

    @patch("google.genai.Client")
    def test_test_llm_connection_helper_success_and_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_client.models.generate_content.return_value = mock_response

        logged = []
        cfg = LLMConfig("gemini", "gemini-3.8-flash", "test-key")
        ok = test_llm_connection(config=cfg, log=logged.append)
        self.assertTrue(ok)
        self.assertTrue(any("Connection test passed" in m for m in logged))

        # Test failure path with logger
        from google.genai import errors
        mock_client.models.generate_content.side_effect = errors.APIError(
            404, {"error": {"message": "NOT_FOUND"}}
        )
        logged_fail = []
        fail_ok = test_llm_connection(config=cfg, log=logged_fail.append)
        self.assertFalse(fail_ok)
        self.assertTrue(any("Connection test failed" in m for m in logged_fail))
        self.assertNotIn("LLM_BASE_URL", logged_fail[0])

    def test_openai_test_connection_preserved(self):
        adapter = OpenAICompatibleAdapter(LLMConfig("openai", "gpt-4o-mini", "test-key"))
        with patch.object(adapter, "complete", return_value="OK") as mock_comp:
            ok = adapter.test_connection()
            self.assertTrue(ok)
            mock_comp.assert_called_once_with("Reply with exactly the word: OK")


class OllamaProviderTests(unittest.TestCase):
    def test_ollama_accepted_without_api_key(self):
        config = LLMConfig("ollama", "qwen2.5:3b", "", "http://localhost:11434/v1")
        validated = validate_llm_config(config)
        self.assertEqual(validated.provider, "ollama")

    def test_ollama_selects_openai_compatible_adapter(self):
        config = LLMConfig("ollama", "qwen2.5:3b", "", "http://localhost:11434/v1")
        adapter = get_provider_adapter(config)
        self.assertIsInstance(adapter, OpenAICompatibleAdapter)

    def test_ollama_requires_base_url(self):
        config = LLMConfig("ollama", "qwen2.5:3b", "")
        with self.assertRaisesRegex(LLMConfigurationError, "LLM_BASE_URL"):
            validate_llm_config(config)


if __name__ == "__main__":
    unittest.main()
