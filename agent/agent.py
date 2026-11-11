"""
agent.py — Core agent logic.

The agent does this on every task:
  1. Read the task from the environment
  2. Describe the current world state in plain English
  3. Retrieve any relevant learned skills from the library
  4. Ask the LLM for a plan (ordered list of skill calls)
  5. Execute the plan step by step
  6. If successful → save skill to library
  7. If failed → ask LLM to self-correct, retry (up to MAX_RETRIES times)

This is the Voyager pipeline, simplified for HomeGrid.
"""

import re
from typing import Any, Dict, Optional
from llm import LLMConfigurationError, LLMError, ask_llm, validate_llm_config
from skills import (
    SKILL_FUNCTIONS,
    get_relevant_skills,
    list_available_primitive_skills,
    save_skill,
)
from config import MAX_RETRIES, MAX_STEPS_PER_RUN
from rl_selector import (
    LinearSMDPSkillSelector,
    StateFeaturizer,
    run_smdp_episode,
    task_is_complete as rl_task_is_complete,
)


# ══════════════════════════════════════════════════════════════════════════════
# STATE DESCRIPTION — Convert symbolic_state to plain English for the LLM
# ══════════════════════════════════════════════════════════════════════════════

def describe_state(state: dict) -> str:
    """
    Turn the HomeGrid symbolic_state dict into a human-readable summary.
    This is what gets sent to the LLM so it can reason about the world.
    """
    lines = []

    # Agent info
    agent = state["agent"]
    carrying = agent["carrying"] or "nothing"
    room_map = {"K": "kitchen", "L": "living room", "D": "dining room"}
    agent_room = room_map.get(agent.get("room"), "unknown room")
    lines.append(f"Agent is in the {agent_room}, carrying: {carrying}.")
    lines.append(f"Agent is facing: {state.get('front_obj') or 'empty space'}.")

    # Objects
    pickables = []
    bins = []
    for obj in state["objects"]:
        pos = obj["pos"]
        if pos == (-1, -1):
            continue  # being carried, already noted
        obj_room = room_map.get(obj.get("room"), "unknown")
        if obj["type"] == "Pickable" and not obj.get("invisible"):
            pickables.append(f"{obj['name']} in the {obj_room}")
        elif obj["type"] == "Storage":
            bin_state = obj.get("state", "?")
            bin_action = obj.get("action", "?")
            contains = obj.get("contains") or []
            contains_str = f", contains: {', '.join(contains)}" if contains else ""
            lines.append(
                f"Bin '{obj['name']}' is in the {obj_room} "
                f"[state: {bin_state}, opens with: {bin_action}{contains_str}]."
            )

    if pickables:
        lines.append("Visible objects: " + "; ".join(pickables) + ".")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LLM PLANNING — Ask the LLM for a skill plan
# ══════════════════════════════════════════════════════════════════════════════

def _build_plan_prompt(task: str, state_desc: str, past_skills: list,
                       error_feedback: str = "") -> str:
    """Build the prompt sent to the LLM for planning."""
    primitive_skills = list_available_primitive_skills()

    learned = ""
    if past_skills:
        names = [s["name"] for s in past_skills]
        learned = f"\nPreviously learned skills that may be relevant: {', '.join(names)}"

    error_section = ""
    if error_feedback:
        error_section = f"\n\nPREVIOUS ATTEMPT FAILED:\n{error_feedback}\nPlease generate a corrected plan."

    return f"""TASK: {task}

CURRENT WORLD STATE:
{state_desc}

AVAILABLE PRIMITIVE SKILLS:
{primitive_skills}
{learned}

Generate a plan as a numbered list of skill calls.
Each line must be exactly: <number>. skill_name(arg1, arg2, ...)
Use only the skill names listed above.
Use exact object/bin/room names from the world state.

Example:
1. pickup_object(bottle)
2. open_bin(trash bin)
3. put_in_bin(bottle, trash bin)

Plan:{error_section}"""


def plan_task(task: str, state: dict, log=print) -> list:
    """
    Ask the LLM for a plan given the current task and world state.
    Returns a list of (skill_name, args_list) tuples.
    """
    log("[Agent] Asking LLM for plan...")
    state_desc = describe_state(state)
    past_skills = get_relevant_skills(task)
    prompt = _build_plan_prompt(task, state_desc, past_skills)
    response = ask_llm(prompt)
    log(f"[Agent] LLM response:\n{response}\n")
    return _parse_plan(response)


def plan_correction(task: str, state: dict, failed_plan: list,
                    error_msg: str, log=print) -> list:
    """
    Ask the LLM to self-correct after a failure.
    This is the self-correction loop inspired by Voyager's critic.
    """
    log("[Agent] Asking LLM for corrected plan...")
    state_desc = describe_state(state)
    past_skills = get_relevant_skills(task)
    failed_str = "\n".join(
        f"{i+1}. {name}({', '.join(args)})"
        for i, (name, args) in enumerate(failed_plan)
    )
    error_feedback = f"Failed plan:\n{failed_str}\nError: {error_msg}"
    prompt = _build_plan_prompt(task, state_desc, past_skills, error_feedback)
    response = ask_llm(prompt)
    log(f"[Agent] Corrected plan:\n{response}\n")
    return _parse_plan(response)


def _parse_plan(response: str) -> list:
    """
    Parse LLM output into a list of (skill_name, args) tuples.
    Example input line: "1. pickup_object(bottle)"
    Returns: [("pickup_object", ["bottle"])]
    """
    plan = []
    # Match lines like "1. skill_name(arg1, arg2)" or "skill_name(arg1)"
    pattern = re.compile(
        r"(?:\d+\.\s*)?([a-z_]+)\(([^)]*)\)"
    )
    for line in response.splitlines():
        line = line.strip()
        match = pattern.search(line)
        if match:
            skill_name = match.group(1).strip()
            args_str = match.group(2).strip()
            args = [a.strip() for a in args_str.split(",") if a.strip()]
            if skill_name in SKILL_FUNCTIONS:
                plan.append((skill_name, args))
    return plan


def task_is_complete(task: str, state: dict) -> bool:
    """Return whether *the original task* is complete in ``state``."""
    return rl_task_is_complete(task, state)


# ══════════════════════════════════════════════════════════════════════════════
# PLAN EXECUTION — Run each skill in the plan
# ══════════════════════════════════════════════════════════════════════════════

def execute_plan(env, plan: list, task: Optional[str] = None, log=print) -> tuple:
    """
    Execute a list of (skill_name, args) skill calls on the environment.
    Returns (success: bool, total_reward: float, error_message: str)
    """
    total_reward = 0.0
    total_steps = 0

    if not plan:
        return False, 0.0, "LLM returned an empty plan"

    for skill_name, args in plan:
        fn = SKILL_FUNCTIONS.get(skill_name)
        if fn is None:
            error = f"Unknown skill: {skill_name}"
            log(f"[Agent] {error}")
            return False, total_reward, error

        log(f"[Agent] Executing skill: {skill_name}({', '.join(args)})")

        try:
            success, steps = fn(env, *args, log=log)
            total_steps += steps
        except TypeError as e:
            error = f"Skill {skill_name} called with wrong args {args}: {e}"
            log(f"[Agent] Error: {error}")
            return False, total_reward, error
        except Exception as e:
            error = f"Skill {skill_name} raised exception: {e}"
            log(f"[Agent] Error: {error}")
            return False, total_reward, error

        log(f"[HomeGrid] Skill '{skill_name}' result: {'[OK]' if success else '[FAIL]'}")

        if not success:
            return False, total_reward, f"Skill '{skill_name}' failed to complete"

        # Enforce MAX_STEPS_PER_RUN to prevent runaway execution
        if total_steps >= MAX_STEPS_PER_RUN:
            log(f"[Agent] Step limit reached ({total_steps}/{MAX_STEPS_PER_RUN}). Stopping execution.")
            return False, total_reward, f"Step limit reached ({total_steps} steps)"

    if task is not None:
        final_state = env.unwrapped.get_full_symbolic_state()
        if not task_is_complete(task, final_state):
            return False, total_reward, "Plan completed, but the assigned task is not complete"

    return True, total_reward, ""


def _configure_multitask_wrapper(wrapper: Any, task: str) -> None:
    """Configure MultitaskWrapper's reward_fn and dist_goal for a specific task."""
    parsed = StateFeaturizer.parse_task(task)
    task_type = parsed["task_type"]
    obj_name = parsed["target_obj"]
    bin_name = parsed["target_bin"]
    room_name = parsed["target_room"]

    room_code = {"kitchen": "K", "living room": "L", "dining room": "D"}.get(room_name)

    def reward_fn(symbolic_state):
        if not symbolic_state:
            return 0
        if task_type == "find":
            return int(symbolic_state.get("front_obj") == obj_name)
        elif task_type == "get":
            return int(symbolic_state.get("agent", {}).get("carrying") == obj_name)
        elif task_type == "open":
            for o in symbolic_state.get("objects", []):
                if o.get("name") == bin_name:
                    return int(o.get("state") == "open")
            return 0
        elif task_type == "cleanup":
            if symbolic_state.get("agent", {}).get("carrying") == obj_name:
                return 0.5
            for o in symbolic_state.get("objects", []):
                if o.get("name") == bin_name:
                    return int(obj_name in (o.get("contains") or []))
            return 0
        elif task_type == "rearrange":
            if symbolic_state.get("agent", {}).get("carrying") == obj_name:
                return 0.5
            for o in symbolic_state.get("objects", []):
                if o.get("name") == obj_name:
                    return int(o.get("room") == room_code)
            return 0
        return 0

    def dist_goal(symbolic_state):
        if not symbolic_state:
            return 0
        goal_name = bin_name if (task_type == "cleanup" and symbolic_state.get("agent", {}).get("carrying") == obj_name) else obj_name
        if not goal_name:
            return 0
        objs = [o for o in symbolic_state.get("objects", []) if o.get("name") == goal_name]
        if objs and "pos" in objs[0]:
            pos = objs[0]["pos"]
            agent_pos = symbolic_state.get("agent", {}).get("pos") or getattr(wrapper, "agent_pos", (0, 0))
            return abs(agent_pos[0] - pos[0]) + abs(agent_pos[1] - pos[1])
        return 0

    try:
        wrapper.task = task
        wrapper.reward_fn = reward_fn
        wrapper.dist_goal = dist_goal
        wrapper.subtask_done = False
        wrapper.start_step = getattr(wrapper, "step_cnt", 0)
    except Exception:
        pass



def prepare_env_for_task(env: Any, target_task: str, max_resets: int = 50, log=print) -> bool:
    """Ensure the environment layout contains the necessary objects/bins for target_task,

    and explicitly configure target_task across the environment hierarchy.
    """
    parsed = StateFeaturizer.parse_task(target_task)
    req_obj = parsed.get("target_obj")
    req_bin = parsed.get("target_bin")

    found_state = False
    for attempt in range(max_resets):
        obs, info = env.reset()

        state = None
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_full_symbolic_state"):
            state = env.unwrapped.get_full_symbolic_state()
        elif hasattr(env, "get_full_symbolic_state"):
            state = env.get_full_symbolic_state()
        elif isinstance(info, dict) and "symbolic_state" in info:
            state = info["symbolic_state"]

        if state is not None and "objects" in state:
            obj_names = {o["name"] for o in state["objects"]}
            if req_obj and req_obj not in obj_names:
                continue
            if req_bin and req_bin not in obj_names:
                continue
            found_state = True
            break
        else:
            found_state = True
            break

    if not found_state:
        log(f"[Warning] Could not confirm required objects for '{target_task}' after {max_resets} resets; continuing anyway.")

    set_env_task(env, target_task)
    return found_state


def set_env_task(env: Any, task: str) -> None:
    """Set the active task across the environment and its wrapper hierarchy."""
    if not task:
        return

    try:
        env.task = task
    except Exception:
        pass

    curr = env
    while curr is not None:
        try:
            curr.task = task
        except Exception:
            pass

        # Protect HomeGrid LanguageWrapper from crashing on arbitrary compound task strings
        if hasattr(curr, "cache") and hasattr(curr, "embed_cache") and isinstance(curr.cache, dict):
            import numpy as np
            if task not in curr.cache:
                pad_token = curr.cache.get("<pad>", 0)
                curr.cache[task] = [pad_token]
            if task not in curr.embed_cache:
                pad_embed = curr.embed_cache.get("<pad>", np.zeros(getattr(curr, "embed_size", 512), dtype=np.float32))
                curr.embed_cache[task] = [pad_embed]

        if hasattr(curr, "reward_fn") and hasattr(curr, "sample_task"):
            _configure_multitask_wrapper(curr, task)

        curr = getattr(curr, "env", None)


def run_task(
    env,
    log=print,
    mode: str = "llm",
    selector: Optional[LinearSMDPSkillSelector] = None,
    training: bool = False,
    alpha: float = 0.05,
    gamma: float = 0.95,
    epsilon: Optional[float] = None,
    task: Optional[str] = None,
) -> dict:
    """Run a complete task with the current environment task.

    Supports:
      - mode='llm': Original Voyager loop (LLM planning + self-correction)
      - mode='rl': Direct Linear SMDP Skill Selection
      - mode='hybrid': RL selector first; fallback to LLM planner if RL fails/unavailable
      - mode='random': Uniform random selection among valid skills (baseline)
    """
    if task is not None:
        set_env_task(env, task)
    else:
        task = getattr(env, "task", "")
    log(f"\n{'='*60}")
    log(f"[Agent] Task: {task} | Mode: {mode.upper()}")
    log(f"{'='*60}")

    # 1. RL or Random Mode
    if mode in ("rl", "random"):
        result = run_smdp_episode(
            env,
            selector=selector,
            task=task,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            training=training,
            random_baseline=(mode == "random"),
            log=log,
        )
        if result["success"]:
            skill_name = _task_to_skill_name(task)
            save_skill(skill_name, f"Solved: {task}", _infer_task_type(task))
            log(f"[Agent] Saving skill: '{skill_name}' to library")
        return {
            "task": task,
            "success": result["success"],
            "attempts": 1,
            "plan": result["skills_executed"],
            "reward": result["total_reward"],
            "steps": result["total_steps"],
            "failed_skills": result["failed_skills"],
            "mode": mode,
        }

    # 2. Hybrid Mode: Try RL first if selector is provided
    if mode == "hybrid":
        rl_res = None
        if selector is not None:
            log("[Agent] [Hybrid] Attempting SMDP skill selection first...")
            rl_res = run_smdp_episode(
                env,
                selector=selector,
                task=task,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                training=training,
                log=log,
            )
            if rl_res["success"]:
                log("[Agent] [Hybrid] SMDP skill selection succeeded!")
                skill_name = _task_to_skill_name(task)
                save_skill(skill_name, f"Solved: {task}", _infer_task_type(task))
                return {
                    "task": task,
                    "success": True,
                    "attempts": 1,
                    "plan": rl_res["skills_executed"],
                    "reward": rl_res["total_reward"],
                    "steps": rl_res["total_steps"],
                    "failed_skills": rl_res["failed_skills"],
                    "mode": "hybrid (rl-success)",
                }
            log("[Agent] [Hybrid] RL attempt did not complete task; falling back to LLM planner...")
        else:
            log("[Agent] [Hybrid] No RL selector available; falling back to LLM planner...")

        # Validate LLM configuration at fallback time
        try:
            validate_llm_config()
        except LLMConfigurationError as exc:
            error_msg = f"LLM fallback unavailable: {exc}"
            log(f"[Agent] [Hybrid] {error_msg}")
            return {
                "task": task,
                "success": False,
                "attempts": 1,
                "plan": rl_res["skills_executed"] if rl_res else [],
                "reward": rl_res["total_reward"] if rl_res else 0.0,
                "steps": rl_res["total_steps"] if rl_res else 0,
                "failed_skills": rl_res["failed_skills"] if rl_res else 0,
                "mode": "hybrid (llm-fallback-unavailable)",
                "error": error_msg,
            }

    # 3. LLM Planning Loop (Default / Fallback)
    state = env.unwrapped.get_full_symbolic_state()

    plan = None
    error_msg = ""
    success = False

    for attempt in range(1, MAX_RETRIES + 2):  # +2 because range is exclusive
        log(f"\n{'-'*40}")
        log(f"[Agent] Attempt {attempt}/{MAX_RETRIES + 1}")
        log(f"{'-'*40}")

        # Plan
        try:
            if attempt == 1:
                plan = plan_task(task, state, log)
            else:
                # If no plan exists, the previous failure happened during LLM
                # planning, so retry normal planning instead of self-correction.
                if plan is None:
                    log(f"[Agent] Previous attempt had no valid plan - requesting fresh plan")
                    plan = plan_task(task, state, log)
                else:
                    # Self-correction: send the failed plan and error info back to LLM.
                    plan = plan_correction(task, state, plan, error_msg, log)

        except LLMError as exc:
            # A transient provider outage must behave like a failed planning
            # attempt, not terminate the HomeGrid run with a traceback.
            plan = None
            error_msg = f"LLM unavailable: {exc}"
            log(f"[Agent] {error_msg}")
            if attempt <= MAX_RETRIES:
                log(f"[Agent] Retrying...")
            continue

        if not plan:
            error_msg = "LLM could not generate a valid plan"
            log(f"[Agent] {error_msg}")
            continue

        log(f"[Agent] Plan: {[(n, a) for n, a in plan]}")

        # Execute
        success, reward, error_msg = execute_plan(env, plan, task, log)

        # Re-read state after execution (environment may have changed)
        state = env.unwrapped.get_full_symbolic_state()

        if success:
            log(f"\n[Agent] Task successful!")
            break
        else:
            log(f"\n[Agent] Task failed. Reason: {error_msg}")
            if attempt <= MAX_RETRIES:
                log(f"[Agent] Self-correcting... (attempt {attempt+1})")

    # Save skill on success
    if success:
        skill_name = _task_to_skill_name(task)
        skill_desc = f"Solved: {task}"
        task_type = _infer_task_type(task)
        save_skill(skill_name, skill_desc, task_type)
        log(f"[Agent] Saving skill: '{skill_name}' to library")

    return {
        "task": task,
        "success": success,
        "attempts": attempt,
        "plan": plan,
        "mode": ("hybrid (llm-fallback-success)" if success else "hybrid (llm-fallback-failed)") if mode == "hybrid" else mode,
    }


def _task_to_skill_name(task: str) -> str:
    """Convert a task string to a snake_case skill name for the library."""
    # e.g. "put the bottle in the trash bin" → "put_bottle_in_trash_bin"
    words = task.lower().replace(" the ", "_").replace(" in ", "_in_")
    words = re.sub(r"[^a-z0-9_]", "_", words)
    words = re.sub(r"_+", "_", words).strip("_")
    return words[:50]  # cap length


def _infer_task_type(task: str) -> str:
    """Infer broad task type for skill categorization."""
    task_lower = task.lower()
    if task_lower.startswith("find"):
        return "find"
    elif task_lower.startswith("get"):
        return "get"
    elif task_lower.startswith("put"):
        return "cleanup"
    elif task_lower.startswith("move"):
        return "rearrange"
    elif task_lower.startswith("open"):
        return "open"
    return "general"
