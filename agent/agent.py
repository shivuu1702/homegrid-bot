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
from typing import Optional
from llm import LLMError, ask_llm
from skills import (
    SKILL_FUNCTIONS,
    get_relevant_skills,
    list_available_primitive_skills,
    save_skill,
)
from config import MAX_RETRIES, MAX_STEPS_PER_RUN


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
    """Return whether *the original task* is complete in ``state``.

    Primitive skills report whether their local action worked.  That is not
    enough to establish that an LLM plan solved the assigned task (for
    example, ``find_object(bottle)`` can succeed during ``get the fruit``).
    The environment samples only the task forms handled below.
    """
    task_lower = task.lower().strip()
    objects = {obj["name"]: obj for obj in state["objects"]}

    if task_lower.startswith("find the "):
        return state.get("front_obj") == task_lower.removeprefix("find the ")
    if task_lower.startswith("get the "):
        return state["agent"].get("carrying") == task_lower.removeprefix("get the ")
    if task_lower.startswith("open the "):
        bin_info = objects.get(task_lower.removeprefix("open the "))
        return bool(bin_info and bin_info.get("state") == "open")
    if task_lower.startswith("put the ") and " in the " in task_lower:
        obj_name, bin_name = task_lower.removeprefix("put the ").split(" in the ", 1)
        bin_info = objects.get(bin_name)
        return bool(bin_info and obj_name in (bin_info.get("contains") or []))
    if task_lower.startswith("move the ") and " to the " in task_lower:
        obj_name, room_name = task_lower.removeprefix("move the ").split(" to the ", 1)
        room_codes = {"kitchen": "K", "living room": "L", "dining room": "D"}
        return bool(objects.get(obj_name) and objects[obj_name].get("room") == room_codes.get(room_name))
    return False


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


def run_task(env, log=print) -> dict:
    """
    Run a complete task with the current environment task.
    Implements the full Voyager loop:
      observe → plan → execute → feedback → correct → save skill

    Returns a result dict with keys: task, success, attempts, reward
    """
    # Get the current task from env
    task = env.task
    log(f"\n{'='*60}")
    log(f"[Agent] Task: {task}")
    log(f"{'='*60}")

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
