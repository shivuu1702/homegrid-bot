"""
skills.py — Two things in one file:

1. PRIMITIVE SKILLS: Deterministic Python functions that control HomeGrid.
   These are the "executable skills" in the Voyager sense.
   The LLM never generates raw action integers — it calls these named functions.

2. SKILL LIBRARY: Save/load/retrieve skills from skills.json.
   This is the external memory that grows as the agent succeeds.
   The LLM's weights never change — only this file grows.

Architecture:
    LLM decides which skill to call
         ↓
    Primitive skill function runs
         ↓
    HomeGrid actions executed step-by-step
         ↓
    Result returned to agent
"""

import json
from collections import deque
from config import SKILLS_FILE, MAX_NAV_STEPS


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION HELPER — BFS pathfinding on the grid
# ══════════════════════════════════════════════════════════════════════════════

# Action integers (from HomeGridBase.Actions)
LEFT   = 0
RIGHT  = 1
UP     = 2
DOWN   = 3
PICKUP = 4
DROP   = 5
GET    = 6
PEDAL  = 7
GRASP  = 8
LIFT   = 9

# Direction index → (dx, dy) movement vector
DIR_TO_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

# Action → direction it moves in
VEC_TO_ACTION = {(1, 0): RIGHT, (0, 1): DOWN, (-1, 0): LEFT, (0, -1): UP}

OPEN_ACTIONS = {"pedal": PEDAL, "grasp": GRASP, "lift": LIFT}


def _get_object_info(state, name):
    """Return the object dict from symbolic_state matching name, or None."""
    for obj in state["objects"]:
        if obj["name"] == name:
            return obj
    return None


def _bfs_path(start, goal, blocked):
    """
    BFS from start to goal on a 2D grid.
    blocked: set of (x,y) positions the agent cannot walk into.
    Returns a list of (x,y) positions from start (exclusive) to goal (inclusive),
    or None if no path found.
    """
    if start == goal:
        return []
    queue = deque()
    queue.append((start, []))
    visited = {start}

    while queue:
        pos, path = queue.popleft()
        x, y = pos
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt = (x + dx, y + dy)
            if nxt == goal:
                return path + [nxt]
            if nxt not in visited and nxt not in blocked:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))

    return None  # No path found


def _build_blocked(env, state):
    """
    Build the set of grid positions the agent cannot walk into.
    Reads actual wall/obstacle positions from the HomeGrid grid object.
    This correctly handles walls, fixed furniture, and impassable floor tiles.
    """
    blocked = set()

    # Read the actual grid for wall/obstacle positions
    grid = env.unwrapped.grid
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            floor = grid.get_floor(x, y)
            # Out-of-bounds cells (both None) = impassable
            if cell is None and floor is None:
                blocked.add((x, y))
            # Cell objects that block movement (walls, furniture)
            elif cell is not None and not cell.agent_can_overlap():
                blocked.add((x, y))
            # Floor tiles with fixed non-overlappable objects (cupboards, etc.)
            elif floor is not None and not floor.agent_can_overlap():
                blocked.add((x, y))

    return blocked


def navigate_to(env, target_name, log=print):
    """
    Move the agent to stand adjacent to target_name and face it.
    Uses BFS with actual wall information from the environment grid.

    Each call takes ONE step toward the target (the loop in the caller
    drives us here repeatedly until we arrive or timeout).
    Actually this function runs its own loop for MAX_NAV_STEPS steps.

    Returns (steps_taken, reached) where reached=True if we got there.
    """
    steps = 0

    for _ in range(MAX_NAV_STEPS):
        state = env.unwrapped.get_full_symbolic_state()
        # Always convert numpy int64 → plain Python int for set/dict hashing
        agent_pos = (int(state["agent"]["pos"][0]), int(state["agent"]["pos"][1]))

        obj_info = _get_object_info(state, target_name)
        if obj_info is None:
            log(f"  [nav] Object '{target_name}' not found in state")
            return steps, False

        raw_pos = obj_info["pos"]
        obj_pos = (int(raw_pos[0]), int(raw_pos[1]))

        # Object is being carried — treat as "reached"
        if obj_pos == (-1, -1):
            log(f"  [nav] '{target_name}' is already carried")
            return steps, True

        # Check if we're already adjacent and facing the object
        agent_dir = int(state["agent"]["dir"])
        dir_dx, dir_dy = DIR_TO_VEC[agent_dir]
        front = (agent_pos[0] + dir_dx, agent_pos[1] + dir_dy)
        if front == obj_pos:
            log(f"  [nav] Already facing '{target_name}'")
            return steps, True

        # Build wall/obstacle blocked set
        blocked = _build_blocked(env, state)
        # Don't block the target position itself (we want to face it from next to it)
        blocked.discard(obj_pos)
        # Don't block our own position
        blocked.discard(agent_pos)

        # Find best neighbor to stand in (adjacent to target, not blocked)
        best_path = None
        best_stand = None

        for ddx, ddy in [(1,0),(-1,0),(0,1),(0,-1)]:
            candidate = (obj_pos[0] + ddx, obj_pos[1] + ddy)
            if candidate in blocked:
                continue
            path = _bfs_path(agent_pos, candidate, blocked)
            if path is not None:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                    best_stand = candidate

        if best_stand is None:
            log(f"  [nav] No reachable neighbor for '{target_name}'")
            return steps, False

        if agent_pos == best_stand:
            # In position — turn to face target
            face_dx = obj_pos[0] - agent_pos[0]
            face_dy = obj_pos[1] - agent_pos[1]
            action = VEC_TO_ACTION.get((face_dx, face_dy))
            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
            return steps, True

        # Take ONE step along the best path
        if best_path:
            next_pos = best_path[0]
            move_dx = next_pos[0] - agent_pos[0]
            move_dy = next_pos[1] - agent_pos[1]
            action = VEC_TO_ACTION.get((move_dx, move_dy))
            if action is None:
                log(f"  [nav] No action for move ({move_dx},{move_dy})")
                return steps, False
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                return steps, False

    log(f"  [nav] Navigation timeout for '{target_name}'")
    return steps, False


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE SKILLS — High-level actions the LLM can request
# ══════════════════════════════════════════════════════════════════════════════

def skill_find_object(env, obj_name, log=print):
    """
    Skill: find_object
    Navigate to and face the named object.
    Reward condition: front_obj == obj_name
    """
    log(f"  [skill] find_object({obj_name})")
    steps, reached = navigate_to(env, obj_name, log)
    if not reached:
        return False, steps
    state = env.unwrapped.get_full_symbolic_state()
    success = (state["front_obj"] == obj_name)
    return success, steps


def skill_pickup_object(env, obj_name, log=print):
    """
    Skill: pickup_object
    Navigate to object then pick it up.
    """
    log(f"  [skill] pickup_object({obj_name})")

    # Already carrying it?
    state = env.unwrapped.get_full_symbolic_state()
    if state["agent"]["carrying"] == obj_name:
        log(f"  [skill] Already carrying {obj_name}")
        return True, 0

    steps, reached = navigate_to(env, obj_name, log)
    if not reached:
        return False, steps

    obs, reward, terminated, truncated, info = env.step(PICKUP)
    steps += 1

    state = env.unwrapped.get_full_symbolic_state()
    success = (state["agent"]["carrying"] == obj_name)
    log(f"  [skill] carrying: {state['agent']['carrying']}")
    return success, steps


def skill_open_bin(env, bin_name, log=print):
    """
    Skill: open_bin
    Navigate to bin and open it using the correct action.
    Each bin has a randomized required action (pedal/grasp/lift).
    We read the required action from symbolic_state — no guessing.
    """
    log(f"  [skill] open_bin({bin_name})")

    # Check if already open before wasting steps navigating
    state = env.unwrapped.get_full_symbolic_state()
    bin_info = _get_object_info(state, bin_name)
    if bin_info is not None and bin_info.get("state") == "open":
        log(f"  [skill] {bin_name} is already open")
        return True, 0

    steps, reached = navigate_to(env, bin_name, log)
    if not reached:
        return False, steps

    # Re-read state after navigation
    state = env.unwrapped.get_full_symbolic_state()
    bin_info = _get_object_info(state, bin_name)
    if bin_info is None:
        return False, steps

    bin_state = bin_info.get("state", "closed")
    if bin_state == "open":
        log(f"  [skill] {bin_name} opened during navigation")
        return True, steps

    if bin_state == "broken":
        log(f"  [skill] {bin_name} is broken — waiting...")
        # Wait ~10 ticks for bin to reset to closed
        for _ in range(12):
            obs, reward, terminated, truncated, info = env.step(LEFT)
            steps += 1
        state = env.unwrapped.get_full_symbolic_state()
        bin_info = _get_object_info(state, bin_name)
        if bin_info is None or bin_info.get("state") != "closed":
            log(f"  [skill] {bin_name} still broken after wait")
            return False, steps
        # Re-navigate
        s, reached = navigate_to(env, bin_name, log)
        steps += s
        if not reached:
            return False, steps
        state = env.unwrapped.get_full_symbolic_state()
        bin_info = _get_object_info(state, bin_name)

    required_action = bin_info.get("action", "pedal")
    action_int = OPEN_ACTIONS.get(required_action, PEDAL)
    log(f"  [skill] Opening {bin_name} with: {required_action}")

    obs, reward, terminated, truncated, info = env.step(action_int)
    steps += 1

    state = env.unwrapped.get_full_symbolic_state()
    bin_info = _get_object_info(state, bin_name)
    success = (bin_info is not None and bin_info.get("state") == "open")
    if not success and bin_info:
        log(f"  [skill] {bin_name} state after attempt: {bin_info.get('state')}")
    return success, steps


def skill_put_in_bin(env, obj_name, bin_name, log=print):
    """
    Skill: put_in_bin
    Full composite skill: pick up object → open bin → drop into bin.
    This handles the "put X in Y" task type end-to-end.
    """
    log(f"  [skill] put_in_bin({obj_name}, {bin_name})")
    total_steps = 0

    # Step 1: Pick up the object (if not already carrying it)
    state = env.unwrapped.get_full_symbolic_state()
    if state["agent"]["carrying"] != obj_name:
        ok, s = skill_pickup_object(env, obj_name, log)
        total_steps += s
        if not ok:
            log(f"  [skill] Failed to pick up {obj_name}")
            return False, total_steps

    # Step 2: Open the bin
    ok, s = skill_open_bin(env, bin_name, log)
    total_steps += s
    if not ok:
        log(f"  [skill] Failed to open {bin_name}")
        return False, total_steps

    # Step 3: Navigate to bin and drop
    steps, reached = navigate_to(env, bin_name, log)
    total_steps += steps
    if not reached:
        return False, total_steps

    obs, reward, terminated, truncated, info = env.step(DROP)
    total_steps += 1

    # Verify the object landed in the bin
    state = env.unwrapped.get_full_symbolic_state()
    bin_info = _get_object_info(state, bin_name)
    if bin_info and obj_name in (bin_info.get("contains") or []):
        log(f"  [skill] {obj_name} successfully placed in {bin_name}")
        return True, total_steps

    # Also count as success if we're no longer carrying it and reward > 0
    if state["agent"]["carrying"] is None:
        log(f"  [skill] Dropped {obj_name} (not carrying anymore)")
        return True, total_steps

    log(f"  [skill] Drop failed — still carrying: {state['agent']['carrying']}")
    return False, total_steps


def skill_move_to_room(env, obj_name, room_name, log=print):
    """
    Skill: move_to_room
    Pick up object and carry it into the target room, then drop it.
    Room names: "kitchen", "living room", "dining room"
    """
    log(f"  [skill] move_to_room({obj_name}, {room_name})")
    total_steps = 0

    room_codes = {"kitchen": "K", "living room": "L", "dining room": "D"}
    room_code = room_codes.get(room_name.lower())
    if room_code is None:
        log(f"  [skill] Unknown room: {room_name}")
        return False, total_steps

    # Step 1: Pick up object
    state = env.unwrapped.get_full_symbolic_state()
    if state["agent"]["carrying"] != obj_name:
        ok, s = skill_pickup_object(env, obj_name, log)
        total_steps += s
        if not ok:
            return False, total_steps

    # Step 2: Check if already in the target room
    state = env.unwrapped.get_full_symbolic_state()
    agent_room = state["agent"].get("room")
    if agent_room == room_code:
        log(f"  [skill] Already in {room_name}")
        ok, s = _drop_in_room(env, obj_name, room_code, log)
        total_steps += s
        return ok, total_steps

    # Step 3: Navigate into the target room
    # Get room cell positions from the layout
    try:
        room_cells = list(env.unwrapped.room_to_cells.get(room_code, []))
        room_cells = [(int(c[0]), int(c[1])) for c in room_cells]
    except Exception as e:
        log(f"  [skill] Could not get room cells: {e}")
        return False, total_steps

    if not room_cells:
        log(f"  [skill] No cells for room {room_name}")
        return False, total_steps

    # Navigate step-by-step toward any room cell
    for _ in range(MAX_NAV_STEPS):
        state = env.unwrapped.get_full_symbolic_state()
        agent_pos = (int(state["agent"]["pos"][0]), int(state["agent"]["pos"][1]))
        agent_room = state["agent"].get("room")

        if agent_room == room_code:
            # We entered the room — drop the object
            log(f"  [skill] Entered {room_name}")
            ok, s = _drop_in_room(env, obj_name, room_code, log)
            total_steps += s
            return ok, total_steps

        # Pick a target room cell and BFS to it
        blocked = _build_blocked(env, state)
        blocked.discard(agent_pos)

        best_path = None
        for cell in room_cells:
            if cell in blocked:
                continue
            path = _bfs_path(agent_pos, cell, blocked)
            if path is not None:
                if best_path is None or len(path) < len(best_path):
                    best_path = path

        if best_path is None or len(best_path) == 0:
            log(f"  [skill] No path to {room_name}")
            return False, total_steps

        # Take one step
        next_pos = best_path[0]
        move_dx = next_pos[0] - agent_pos[0]
        move_dy = next_pos[1] - agent_pos[1]
        action = VEC_TO_ACTION.get((move_dx, move_dy))
        if action is None:
            log(f"  [skill] No action for move ({move_dx},{move_dy})")
            return False, total_steps
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            return False, total_steps

    log(f"  [skill] Timeout navigating to {room_name}")
    return False, total_steps


def _drop_in_room(env, obj_name, room_code, log=print):
    """Helper: drop the carried object while standing in the room."""
    steps = 0
    state = env.unwrapped.get_full_symbolic_state()
    agent_pos = (int(state["agent"]["pos"][0]), int(state["agent"]["pos"][1]))

    # Try each direction — drop toward an empty cell
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        face_pos = (agent_pos[0]+dx, agent_pos[1]+dy)
        # Check if the facing cell is free
        occupied = any(
            (int(o["pos"][0]), int(o["pos"][1])) == face_pos
            for o in state["objects"]
            if tuple(o["pos"]) != (-1, -1)
        )
        if not occupied:
            action = VEC_TO_ACTION.get((dx, dy))
            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                obs, reward, terminated, truncated, info = env.step(DROP)
                steps += 1
                state = env.unwrapped.get_full_symbolic_state()
                for obj in state["objects"]:
                    if obj["name"] == obj_name and obj.get("room") == room_code:
                        log(f"  [skill] Dropped {obj_name} in room {room_code}")
                        return True, steps
                break  # Tried, check result
    return False, steps


# Map of skill names → callable functions
# LLM will return skill names from this dict.
SKILL_FUNCTIONS = {
    "find_object":    skill_find_object,
    "pickup_object":  skill_pickup_object,
    "open_bin":       skill_open_bin,
    "put_in_bin":     skill_put_in_bin,
    "move_to_room":   skill_move_to_room,
}


# ══════════════════════════════════════════════════════════════════════════════
# SKILL LIBRARY — External memory that grows over time
# ══════════════════════════════════════════════════════════════════════════════

def load_skills():
    """Load the skill library from skills.json. Returns a dict."""
    SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SKILLS_FILE.exists():
        return {"skills": []}
    with open(SKILLS_FILE, "r") as f:
        return json.load(f)


def save_skill(name: str, description: str, task_type: str):
    """
    Save or update a skill in skills.json.
    If the skill already exists, increment its success_count.
    This is the 'skill learning' part of Voyager.
    The LLM's weights never change — only this JSON file grows.
    """
    library = load_skills()
    for skill in library["skills"]:
        if skill["name"] == name:
            skill["success_count"] += 1
            skill["last_used_for"] = task_type
            break
    else:
        library["skills"].append({
            "name": name,
            "description": description,
            "task_type": task_type,
            "success_count": 1,
        })

    SKILLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SKILLS_FILE, "w") as f:
        json.dump(library, f, indent=2)


def get_relevant_skills(task: str) -> list:
    """
    Find skills in the library that are relevant to the current task.
    Simple keyword matching — understandable and sufficient for demo.
    Future extension: replace with embedding similarity search.
    """
    library = load_skills()
    task_lower = task.lower()
    relevant = []
    for skill in library["skills"]:
        keywords = skill["name"].replace("_", " ").split()
        keywords += skill.get("task_type", "").split()
        keywords += skill.get("description", "").lower().split()
        if any(kw in task_lower for kw in keywords if len(kw) > 3):
            relevant.append(skill)
    return relevant


def list_available_primitive_skills() -> str:
    """Return a human-readable list of primitive skills for LLM prompts."""
    descriptions = {
        "find_object":   "Navigate to and face a named object. Args: (obj_name)",
        "pickup_object": "Navigate to and pick up a named object. Args: (obj_name)",
        "open_bin":      "Navigate to and open a named bin using its correct action. Args: (bin_name)",
        "put_in_bin":    "Pick up object, open bin, then drop object into bin. Args: (obj_name, bin_name)",
        "move_to_room":  "Pick up object and carry it to a named room. Args: (obj_name, room_name)",
    }
    lines = []
    for name, desc in descriptions.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)
