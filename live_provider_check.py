"""Manual, optional live check. It is deliberately not named test_*.py."""
import sys

sys.path.insert(0, "agent")

from llm import LLMConfigurationError, test_llm_connection, validate_llm_config


if __name__ == "__main__":
    try:
        validate_llm_config()
    except LLMConfigurationError as exc:
        print(f"[LLM ERROR] {exc}")
        raise SystemExit(1)
    raise SystemExit(0 if test_llm_connection() else 1)
