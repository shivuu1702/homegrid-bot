"""
main.py — Entry point for the Voyager-inspired HomeGrid agent.

Run:
    cd AI-Project
    python agent/main.py

This script:
  1. Creates the HomeGrid environment
  2. Runs a curriculum of task episodes
  3. Shows the agent planning, executing, failing, and self-correcting
  4. Demonstrates skill library growing over time
"""

import sys
import os
import json
import logging
import datetime
import gym

# Add agent/ to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

import homegrid  # registers gym environments
from agent import run_task
from skills import load_skills
from config import (
    ENV_ID, LOGS_DIR, CURRICULUM_TASK_TYPES,
    LLM_PROVIDER, LLM_MODEL,
)
from llm import LLMConfigurationError, validate_llm_config, test_llm_connection


# ── Logging Setup ──────────────────────────────────────────────────────────────

def setup_logger():
    """Create a logger that writes to both terminal and a log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"run_{timestamp}.log"

    # Root logger
    logger = logging.getLogger("homegrid_agent")
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO level — clean output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    # File handler (DEBUG level — full details)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger, log_file


# ── Environment Setup ──────────────────────────────────────────────────────────

def make_env():
    """
    Create the HomeGrid environment.
    We use 'homegrid-task' which includes:
      - MultitaskWrapper (samples a task, gives reward on completion)
      - LanguageWrapper (streams language hints as tokens)
    """
    env = gym.make(ENV_ID, disable_env_checker=True)
    return env


def force_task_type(env, task_type: str, max_resets: int = 20) -> bool:
    """
    Reset the environment until we get a task of the requested type.
    HomeGrid randomly samples tasks on reset, so we may need a few tries.
    Returns True if we found the right task type.
    """
    for _ in range(max_resets):
        obs, info = env.reset()
        task = env.task
        task_lower = task.lower()
        if task_type == "find" and task_lower.startswith("find"):
            return True
        elif task_type == "get" and task_lower.startswith("get"):
            return True
        elif task_type == "cleanup" and task_lower.startswith("put"):
            return True
        elif task_type == "open" and task_lower.startswith("open"):
            return True
        elif task_type == "rearrange" and task_lower.startswith("move"):
            return True
    return False  # Couldn't get the right task type


# ── Main Agent Loop ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voyager-inspired HomeGrid Agent")
    parser.add_argument("--visual", action="store_true",
                        help="Open a HomeGrid window to watch the agent in real-time")
    parser.add_argument("--demo", action="store_true",
                        help="Run a single reliable demo task (cleanup) instead of full curriculum")
    parser.add_argument("--task", type=str, default=None,
                        help='Run a specific task, e.g. --task "put the fruit in the recycling bin"')
    args = parser.parse_args()

    logger, log_file = setup_logger()
    log = logger.info

    # Banner
    log("=" * 60)
    log("  Voyager-Inspired HomeGrid Agent")
    log("=" * 60)
    log(f"  LLM Provider : {LLM_PROVIDER}")
    log(f"  LLM Model    : {LLM_MODEL}")
    log(f"  Log file     : {log_file}")
    log(f"  Skills       : {LOGS_DIR.parent / 'skills' / 'skills.json'}")
    if args.demo:
        log(f"  Mode         : DEMO (single cleanup task)")
    elif args.task:
        log(f"  Mode         : SINGLE TASK")
    else:
        log(f"  Mode         : CURRICULUM ({len(CURRICULUM_TASK_TYPES)} task types)")
    log("=" * 60)

    # ── LLM configuration validation (before anything else) ──────────────────
    # This checks provider, API key, and model BEFORE creating HomeGrid,
    # so a misconfiguration fails fast with a clear message.
    try:
        validate_llm_config()
    except LLMConfigurationError as exc:
        log(f"[LLM ERROR] {exc}")
        sys.exit(1)

    # Optional connection test — sends a tiny prompt to verify the key works.
    # Set LLM_SKIP_TEST=1 in .env to skip this if you want to save API quota.
    import os
    if os.getenv("LLM_SKIP_TEST", "").strip() not in ("1", "true", "yes"):
        ok = test_llm_connection()
        if not ok:
            log("[LLM ERROR] Connection test failed. Curriculum will not start.")
            log("  Fix the LLM configuration in .env and re-run.")
            sys.exit(1)
    else:
        log("[LLM] Connection test skipped (LLM_SKIP_TEST=1).")
    log("")
    log(f"[Agent] Creating HomeGrid environment ({ENV_ID})...")
    env = make_env()
    log("[Agent] Environment created (ok)")

    # ── Visual rendering (optional) ──────────────────────────────────────────
    if args.visual:
        from visual import attach_visual
        env = attach_visual(env, enabled=True)
        log("[Agent] Visual window attached (--visual mode)")
    log("")

    # Show initial skill library
    library = load_skills()
    learned_count = len(library["skills"])
    log(f"[Agent] Skill library loaded: {learned_count} learned skill(s)")
    if learned_count > 0:
        for s in library["skills"]:
            log(f"  - {s['name']} (used {s['success_count']} time(s))")
    log("")

    # ── Demo / Single-Task Mode ──────────────────────────────────────────────
    if args.demo or args.task:
        if args.demo:
            target_task = "put the fruit in the recycling bin"
            log(f"[Demo] Target task: {target_task}")
        else:
            target_task = args.task
            log(f"[Task] Target task: {target_task}")

        # Reset until we get a matching task (or use whatever we get)
        task_prefix = target_task.split()[0].lower()  # e.g. "put", "find"
        matched = False
        for _ in range(30):
            obs, info = env.reset()
            if env.task.lower() == target_task.lower():
                matched = True
                break
        if not matched:
            # Try prefix match as fallback
            for _ in range(20):
                obs, info = env.reset()
                if env.task.lower().startswith(task_prefix):
                    matched = True
                    break
        if not matched:
            obs, info = env.reset()
            log(f"[Warning] Could not get exact task; using: {env.task}")
        else:
            log(f"[Agent] Got task: {env.task}")

        result = run_task(env, log=log)

        # Show result
        library = load_skills()
        log(f"\n[Library] Skills saved: {len(library['skills'])}")
        for s in library["skills"]:
            log(f"  [saved] {s['name']} (success_count={s['success_count']})")

        log(f"\n{'='*60}")
        status = "[OK]" if result["success"] else "[FAIL]"
        log(f"  {status} | attempts={result['attempts']} | {result['task']}")
        log(f"  Full log saved to: {log_file}")
        log(f"{'='*60}")

        env.close()
        return

    # ── Curriculum Loop ──────────────────────────────────────────────────────
    results = []
    log("[Agent] Starting curriculum...\n")
    log(f"  Task types to attempt: {CURRICULUM_TASK_TYPES}\n")

    for task_type in CURRICULUM_TASK_TYPES:
        log(f"\n{'-'*60}")
        log(f"[Curriculum] Task type: {task_type.upper()}")

        # Reset env and try to get the right task type
        found = force_task_type(env, task_type)
        if not found:
            # If we can't get exact type, just use whatever task we have
            obs, info = env.reset()
            log(f"[Curriculum] Could not get '{task_type}' task; using: {env.task}")
        else:
            log(f"[Curriculum] Got task: {env.task}")

        # Run the task with the agent
        result = run_task(env, log=log)
        results.append(result)

        # Show updated skill library after each task
        library = load_skills()
        log(f"\n[Library] Skills saved: {len(library['skills'])}")
        for s in library["skills"]:
            log(f"  [saved] {s['name']} (success_count={s['success_count']})")

    # ── Final Summary ─────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("  CURRICULUM COMPLETE - SUMMARY")
    log(f"{'='*60}")

    successes = sum(1 for r in results if r["success"])
    log(f"  Tasks attempted : {len(results)}")
    log(f"  Tasks succeeded : {successes}")
    log(f"  Tasks failed    : {len(results) - successes}")
    log("")

    for i, r in enumerate(results, 1):
        status = "[OK]    " if r["success"] else "[FAIL]  "
        log(f"  [{i}] {status} | attempts={r['attempts']} | {r['task']}")

    log(f"\n  Full log saved to: {log_file}")

    # Save run summary to JSON
    summary_file = LOGS_DIR / f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Run summary: {summary_file}")

    env.close()


if __name__ == "__main__":
    main()
