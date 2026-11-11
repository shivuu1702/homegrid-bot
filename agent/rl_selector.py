"""Linear SMDP Skill Selector for HomeGrid-Bot.

This module implements the foundational components for a Semi-Markov Decision
Process (SMDP) Skill Selector with Linear Function Approximation:

1. SkillOption: Canonical representation of high-level skills/options.
2. StateFeaturizer: Deterministic, normalized featurization of symbolic state + task.
3. ActionMasker: Rigorous precondition checking for legal skills in the current state.
4. LinearSMDPSkillSelector: Linear Q-function Q(s, option) = theta_option^T @ phi(s)
   with epsilon-greedy exploration and SMDP TD update support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DOMAIN CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PICKABLE_OBJECTS: Tuple[str, ...] = ("bottle", "fruit", "papers", "plates")
STORAGE_BINS: Tuple[str, ...] = ("recycling bin", "trash bin", "compost bin")
ROOM_NAMES: Tuple[str, ...] = ("kitchen", "living room", "dining room")

ROOM_TO_CODE: Dict[str, str] = {
    "kitchen": "K",
    "living room": "L",
    "dining room": "D",
}
CODE_TO_ROOM: Dict[str, str] = {v: k for k, v in ROOM_TO_CODE.items()}

# Maximum coordinate bounds from homegrid layout (width=14, height=12)
GRID_MAX_WIDTH: float = 14.0
GRID_MAX_HEIGHT: float = 12.0
MAX_MANHATTAN_DIST: float = GRID_MAX_WIDTH + GRID_MAX_HEIGHT  # 26.0

SUPPORTED_SKILL_NAMES: Tuple[str, ...] = (
    "find_object",
    "pickup_object",
    "open_bin",
    "put_in_bin",
    "move_to_room",
)


# ══════════════════════════════════════════════════════════════════════════════
# SKILL OPTION REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SkillOption:
    """Canonical representation of an SMDP macro-action / skill option."""

    name: str
    args: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))

    @property
    def id(self) -> str:
        """Formatted string identifier, e.g. 'pickup_object(fruit)'."""
        return f"{self.name}({', '.join(self.args)})"

    def to_tuple(self) -> Tuple[str, List[str]]:
        """Format matching agent.py plan structure: (skill_name, [arg1, ...])."""
        return self.name, list(self.args)

    @classmethod
    def from_tuple(cls, item: Tuple[str, Sequence[str]]) -> "SkillOption":
        """Construct from (skill_name, args_list) tuple."""
        return cls(name=item[0].strip(), args=tuple(str(a).strip() for a in item[1]))

    @classmethod
    def from_str(cls, s: str) -> "SkillOption":
        """Parse string such as 'put_in_bin(fruit, recycling bin)' into SkillOption."""
        s = s.strip()
        if "(" not in s or not s.endswith(")"):
            raise ValueError(f"Invalid skill option string: '{s}'")
        name = s[:s.find("(")].strip()
        args_str = s[s.find("(") + 1:-1].strip()
        args = tuple(a.strip() for a in args_str.split(",") if a.strip()) if args_str else ()
        return cls(name=name, args=args)

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"SkillOption({self.id})"


def get_all_standard_options() -> List[SkillOption]:
    """Generate all valid canonical concrete options across HomeGrid domain entities."""
    options: List[SkillOption] = []

    # 1. find_object(obj) for all pickables and bins (7 options)
    for obj in PICKABLE_OBJECTS + STORAGE_BINS:
        options.append(SkillOption("find_object", (obj,)))

    # 2. pickup_object(obj) for all pickables (4 options)
    for obj in PICKABLE_OBJECTS:
        options.append(SkillOption("pickup_object", (obj,)))

    # 3. open_bin(bin) for all bins (3 options)
    for b in STORAGE_BINS:
        options.append(SkillOption("open_bin", (b,)))

    # 4. put_in_bin(obj, bin) for all (obj, bin) pairs (12 options)
    for obj in PICKABLE_OBJECTS:
        for b in STORAGE_BINS:
            options.append(SkillOption("put_in_bin", (obj, b)))

    # 5. move_to_room(obj, room) for all (obj, room) pairs (12 options)
    for obj in PICKABLE_OBJECTS:
        for r in ROOM_NAMES:
            options.append(SkillOption("move_to_room", (obj, r)))

    return options


# ══════════════════════════════════════════════════════════════════════════════
# SUBGOAL AND COMPOUND TASK REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Subgoal:
    """Represents a discrete, verifiable subgoal within a simple or compound task."""

    name: str  # e.g. "find_object", "pickup_object", "move_to_room", "open_bin", "put_in_bin"
    args: Tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))

    @property
    def id(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    @property
    def target_obj(self) -> Optional[str]:
        if self.name in ("find_object", "pickup_object", "put_in_bin", "move_to_room"):
            if self.args and self.args[0] in PICKABLE_OBJECTS:
                return self.args[0]
        return None

    @property
    def target_bin(self) -> Optional[str]:
        if self.name == "open_bin" and self.args:
            return self.args[0]
        if self.name == "put_in_bin" and len(self.args) >= 2:
            return self.args[1]
        if self.name == "find_object" and self.args and self.args[0] in STORAGE_BINS:
            return self.args[0]
        return None

    @property
    def target_room(self) -> Optional[str]:
        if self.name == "move_to_room" and len(self.args) >= 2:
            return self.args[1]
        return None

    def is_satisfied(self, state: Dict[str, Any]) -> bool:
        """Evaluate whether this subgoal's postcondition is verified in the symbolic state."""
        objects = {obj["name"]: obj for obj in state.get("objects", [])}
        front_obj = state.get("front_obj")
        carrying = state.get("agent", {}).get("carrying")
        agent_room_code = state.get("agent", {}).get("room")

        if self.name == "find_object":
            target = self.args[0]
            # Satisfied if facing target, or already carrying it (for pickables),
            # or if target is already inside destination bin
            return front_obj == target or carrying == target

        elif self.name == "pickup_object":
            target = self.args[0]
            return carrying == target

        elif self.name == "move_to_room":
            obj_name = self.args[0]
            room_name = self.args[1]
            target_code = ROOM_TO_CODE.get(room_name, room_name)
            if carrying == obj_name and agent_room_code == target_code:
                return True
            obj_info = objects.get(obj_name)
            if obj_info and obj_info.get("room") == target_code:
                return True
            return False

        elif self.name == "open_bin":
            bin_name = self.args[0]
            b = objects.get(bin_name)
            return bool(b and b.get("state") == "open")

        elif self.name == "put_in_bin":
            obj_name = self.args[0]
            bin_name = self.args[1]
            b = objects.get(bin_name)
            return bool(b and obj_name in (b.get("contains") or []))

        return False


def parse_compound_task(task_str: Optional[str]) -> Optional[List[Subgoal]]:
    """Parse a multi-clause compound task string into an ordered sequence of Subgoals.

    Returns None if task_str is a canonical single-clause task.
    """
    if not task_str:
        return None

    t = task_str.strip().lower()

    # Determine if this is a compound task
    has_comma = "," in t
    has_and_clause = any(
        conj in t for conj in (
            " and then ", " then ",
            " and pick", " and get", " and move", " and open", " and put", " and find"
        )
    )
    if not (has_comma or has_and_clause):
        return None

    # Extract target entities present in the entire prompt
    target_obj: Optional[str] = None
    for obj in PICKABLE_OBJECTS:
        if obj in t:
            target_obj = obj
            break

    target_bin: Optional[str] = None
    for b in STORAGE_BINS:
        if b in t:
            target_bin = b
            break

    target_room: Optional[str] = None
    for r in ROOM_NAMES:
        if r in t:
            target_room = r
            break

    # Normalize clause separators to commas
    norm = t
    for conj in (" and then ", " then ", " and pick", " and get", " and move", " and open", " and put", " and find"):
        if conj.startswith(" and "):
            verb = conj.removeprefix(" and ")
            norm = norm.replace(conj, f", {verb}")
        else:
            norm = norm.replace(conj, ", ")

    raw_clauses = [c.strip() for c in norm.split(",") if c.strip()]
    subgoals: List[Subgoal] = []

    for clause in raw_clauses:
        # 1. Find clause
        if clause.startswith("find") or " find " in clause:
            clause_obj = None
            for obj in PICKABLE_OBJECTS:
                if obj in clause:
                    clause_obj = obj
                    break
            clause_bin = None
            for b in STORAGE_BINS:
                if b in clause:
                    clause_bin = b
                    break
            chosen = clause_obj or clause_bin or target_obj or target_bin
            if chosen:
                subgoals.append(Subgoal("find_object", (chosen,), f"Find the {chosen}"))

        # 2. Pick up clause
        elif "pick" in clause or "get" in clause:
            clause_obj = None
            for obj in PICKABLE_OBJECTS:
                if obj in clause:
                    clause_obj = obj
                    break
            chosen = clause_obj or target_obj
            if chosen:
                subgoals.append(Subgoal("pickup_object", (chosen,), f"Pick up the {chosen}"))

        # 3. Move / navigate clause
        elif "move" in clause or "go to" in clause or "navigate" in clause:
            clause_room = None
            for r in ROOM_NAMES:
                if r in clause:
                    clause_room = r
                    break
            chosen_room = clause_room or target_room
            clause_obj = None
            for obj in PICKABLE_OBJECTS:
                if obj in clause:
                    clause_obj = obj
                    break
            chosen_obj = clause_obj or target_obj
            if chosen_obj and chosen_room:
                subgoals.append(Subgoal("move_to_room", (chosen_obj, chosen_room), f"Move {chosen_obj} to {chosen_room}"))

        # 4. Open bin clause
        elif "open" in clause:
            clause_bin = None
            for b in STORAGE_BINS:
                if b in clause:
                    clause_bin = b
                    break
            chosen_bin = clause_bin or target_bin
            if chosen_bin:
                subgoals.append(Subgoal("open_bin", (chosen_bin,), f"Open the {chosen_bin}"))

        # 5. Put in bin clause
        elif "put" in clause or "drop" in clause or "place" in clause or "inside" in clause:
            clause_obj = None
            for obj in PICKABLE_OBJECTS:
                if obj in clause:
                    clause_obj = obj
                    break
            chosen_obj = clause_obj or target_obj
            clause_bin = None
            for b in STORAGE_BINS:
                if b in clause:
                    clause_bin = b
                    break
            chosen_bin = clause_bin or target_bin
            if chosen_obj and chosen_bin:
                subgoals.append(Subgoal("put_in_bin", (chosen_obj, chosen_bin), f"Put {chosen_obj} in {chosen_bin}"))

    return subgoals if len(subgoals) >= 2 else None


class SubgoalTracker:
    """Manages active subgoal state, verification, and progressive advancement for compound tasks."""

    def __init__(self, task: str, subgoals: Sequence[Subgoal]) -> None:
        self.task = task
        self.subgoals = list(subgoals)
        self.current_idx: int = 0
        self.history: List[Dict[str, Any]] = []

    @property
    def active_subgoal(self) -> Optional[Subgoal]:
        if self.current_idx < len(self.subgoals):
            return self.subgoals[self.current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_idx >= len(self.subgoals)

    def check_initial_satisfaction(self, state: Dict[str, Any], log=print) -> None:
        """Advance past any subgoals already satisfied in the initial state (e.g. bin already open)."""
        while self.current_idx < len(self.subgoals):
            sg = self.subgoals[self.current_idx]
            if sg.is_satisfied(state):
                log(f"  [SubgoalTracker] Initial state satisfies subgoal {self.current_idx + 1}/{len(self.subgoals)}: '{sg.description}'")
                self.current_idx += 1
            else:
                break

    def step(
        self,
        option: SkillOption,
        skill_success: bool,
        next_state: Dict[str, Any],
        log=print,
    ) -> bool:
        """Verify whether the skill execution satisfied the active subgoal.

        Advances to the next subgoal ONLY upon verified postcondition satisfaction.
        Returns True if active subgoal succeeded and advanced, False otherwise.
        """
        if self.is_complete:
            return True

        sg = self.active_subgoal
        # Verification: skill must report success AND state must verify postcondition
        satisfied = bool(skill_success and sg.is_satisfied(next_state))

        self.history.append({
            "subgoal_index": self.current_idx,
            "subgoal": sg.description,
            "option": option.id,
            "skill_success": skill_success,
            "satisfied": satisfied,
        })

        if satisfied:
            log(f"  [SubgoalTracker] Subgoal {self.current_idx + 1}/{len(self.subgoals)} completed: '{sg.description}'")
            self.current_idx += 1

            # Auto-advance past any subsequent subgoals that are already satisfied
            while self.current_idx < len(self.subgoals):
                next_sg = self.subgoals[self.current_idx]
                if next_sg.is_satisfied(next_state):
                    log(f"  [SubgoalTracker] Subgoal {self.current_idx + 1}/{len(self.subgoals)} already satisfied: '{next_sg.description}'")
                    self.current_idx += 1
                else:
                    break
            return True
        else:
            log(f"  [SubgoalTracker] Subgoal {self.current_idx + 1}/{len(self.subgoals)} NOT satisfied by '{option.id}' "
                f"(skill_success={skill_success}). Staying on '{sg.description}'.")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# STATE FEATURIZER
# ══════════════════════════════════════════════════════════════════════════════

class StateFeaturizer:
    """Converts symbolic HomeGrid state and task context into a compact,

    deterministic, and normalized numerical feature vector.
    """

    def __init__(self) -> None:
        self.feature_names = self._build_feature_names()
        self.feature_dim = len(self.feature_names)

    @staticmethod
    def _build_feature_names() -> List[str]:
        """Exhaustive list of all named feature indices for interpretability."""
        names: List[str] = []

        # 1. Task type one-hot (5)
        for t in ("find", "get", "open", "cleanup", "rearrange"):
            names.append(f"task_is_{t}")

        # 2. Task target object one-hot (5)
        for obj in PICKABLE_OBJECTS + ("none",):
            names.append(f"task_target_obj_{obj.replace(' ', '_')}")

        # 3. Task target bin one-hot (4)
        for b in STORAGE_BINS + ("none",):
            names.append(f"task_target_bin_{b.replace(' ', '_')}")

        # 4. Task target room one-hot (4)
        for r in ROOM_NAMES + ("none",):
            names.append(f"task_target_room_{r.replace(' ', '_')}")

        # 5. Agent normalized coordinates (2)
        names.extend(["agent_pos_x_norm", "agent_pos_y_norm"])

        # 6. Agent room one-hot (4)
        for r in ROOM_NAMES + ("unknown",):
            names.append(f"agent_in_room_{r.replace(' ', '_')}")

        # 7. Agent orientation one-hot (4)
        for d in (0, 1, 2, 3):
            names.append(f"agent_dir_{d}")

        # 8. Agent carrying features (7)
        names.append("agent_is_carrying")
        for obj in PICKABLE_OBJECTS + ("none",):
            names.append(f"agent_carrying_{obj.replace(' ', '_')}")
        names.append("carrying_matches_target_obj")

        # 9. Facing object relation (1)
        names.append("facing_matches_target_obj")

        # 10. Target object features (4)
        names.extend([
            "target_obj_available",
            "target_obj_is_carried",
            "target_obj_dist_norm",
            "target_obj_in_same_room",
        ])

        # 11. Target bin features (5)
        names.extend([
            "target_bin_is_open",
            "target_bin_is_closed",
            "target_bin_is_broken",
            "target_bin_dist_norm",
            "target_bin_contains_target_obj",
        ])

        # 12. Target room features (2)
        names.extend([
            "agent_in_target_room",
            "target_obj_in_target_room",
        ])

        # 13. Progress & Bias (2)
        names.extend([
            "task_already_complete",
            "bias_intercept",
        ])

        # 14. Task-Skill Interaction Features (12)
        names.extend([
            "skill_arg_matches_target_obj",
            "skill_arg_matches_target_bin",
            "skill_arg_matches_target_room",
            "skill_verb_matches_task_verb",
            "skill_is_direct_goal_action",
            "skill_is_prerequisite_action",
            "skill_verb_mismatches_task",
            "skill_arg_mismatches_target_obj",
            "skill_arg_mismatches_target_bin",
            "skill_arg_mismatches_target_room",
            "skill_all_args_match_task",
            "skill_already_satisfied",
        ])

        return names

    @staticmethod
    def parse_task(task_str: Optional[str]) -> Dict[str, Optional[str]]:
        """Parse natural-language task into task_type, target_obj, target_bin, target_room."""
        res: Dict[str, Optional[str]] = {
            "task_type": None,
            "target_obj": None,
            "target_bin": None,
            "target_room": None,
        }
        if not task_str:
            return res

        t = task_str.strip().lower()

        # Check if this is a compound task
        subgoals = parse_compound_task(task_str)
        if subgoals:
            for obj in PICKABLE_OBJECTS:
                if obj in t:
                    res["target_obj"] = obj
                    break
            for b in STORAGE_BINS:
                if b in t:
                    res["target_bin"] = b
                    break
            for r in ROOM_NAMES:
                if r in t:
                    res["target_room"] = r
                    break

            if "put" in t or res["target_bin"]:
                res["task_type"] = "cleanup"
            elif "move" in t and res["target_room"]:
                res["task_type"] = "rearrange"
            elif "pick" in t or "get" in t:
                res["task_type"] = "get"
            elif "open" in t:
                res["task_type"] = "open"
            else:
                res["task_type"] = "find"
            return res

        # Canonical single-clause task parsing
        if t.startswith("find"):
            res["task_type"] = "find"
            for obj in PICKABLE_OBJECTS:
                if obj in t:
                    res["target_obj"] = obj
                    break
            for b in STORAGE_BINS:
                if b in t:
                    res["target_bin"] = b
                    if res["target_obj"] is None:
                        res["target_obj"] = b
                    break
        elif t.startswith("get"):
            res["task_type"] = "get"
            for obj in PICKABLE_OBJECTS:
                if obj in t:
                    res["target_obj"] = obj
                    break
        elif t.startswith("open"):
            res["task_type"] = "open"
            for b in STORAGE_BINS:
                if b in t:
                    res["target_bin"] = b
                    break
        elif t.startswith("put"):
            res["task_type"] = "cleanup"
            for obj in PICKABLE_OBJECTS:
                if obj in t:
                    res["target_obj"] = obj
                    break
            for b in STORAGE_BINS:
                if b in t:
                    res["target_bin"] = b
                    break
        elif t.startswith("move"):
            res["task_type"] = "rearrange"
            for obj in PICKABLE_OBJECTS:
                if obj in t:
                    res["target_obj"] = obj
                    break
            for r in ROOM_NAMES:
                if r in t:
                    res["target_room"] = r
                    break

        return res

    def featurize(
        self,
        state: Dict[str, Any],
        task: Optional[str] = "",
        option: Optional[Union[SkillOption, str]] = None,
        active_subgoal: Optional[Subgoal] = None,
    ) -> np.ndarray:
        """Extract a deterministic, normalized 1D numpy vector from symbolic state, task, and optional skill option."""
        feat = np.zeros(self.feature_dim, dtype=np.float64)
        parsed_task = self.parse_task(task)

        task_type = parsed_task["task_type"]
        target_obj = parsed_task["target_obj"]
        target_bin = parsed_task["target_bin"]
        target_room = parsed_task["target_room"]

        # If active_subgoal is provided, condition featurization on the active subgoal
        if active_subgoal is not None:
            if active_subgoal.target_obj:
                target_obj = active_subgoal.target_obj
            if active_subgoal.target_bin:
                target_bin = active_subgoal.target_bin
            if active_subgoal.target_room:
                target_room = active_subgoal.target_room

            if active_subgoal.name == "find_object":
                task_type = "find"
            elif active_subgoal.name == "pickup_object":
                task_type = "get"
            elif active_subgoal.name == "open_bin":
                task_type = "open"
            elif active_subgoal.name == "put_in_bin":
                task_type = "cleanup"
            elif active_subgoal.name == "move_to_room":
                task_type = "rearrange"

        agent = state.get("agent", {})
        agent_pos_raw = agent.get("pos", (0, 0))
        agent_x = float(agent_pos_raw[0])
        agent_y = float(agent_pos_raw[1])

        raw_room = agent.get("room")
        agent_room_name = CODE_TO_ROOM.get(raw_room, raw_room)

        agent_dir = agent.get("dir", -1)
        carrying = agent.get("carrying")
        front_obj = state.get("front_obj")

        objects = {obj["name"]: obj for obj in state.get("objects", [])}

        idx = 0

        # 1. Task type one-hot (5)
        for t in ("find", "get", "open", "cleanup", "rearrange"):
            feat[idx] = 1.0 if task_type == t else 0.0
            idx += 1

        # 2. Task target object one-hot (5)
        for obj in PICKABLE_OBJECTS:
            feat[idx] = 1.0 if target_obj == obj else 0.0
            idx += 1
        feat[idx] = 1.0 if target_obj is None else 0.0
        idx += 1

        # 3. Task target bin one-hot (4)
        for b in STORAGE_BINS:
            feat[idx] = 1.0 if target_bin == b else 0.0
            idx += 1
        feat[idx] = 1.0 if target_bin is None else 0.0
        idx += 1

        # 4. Task target room one-hot (4)
        for r in ROOM_NAMES:
            feat[idx] = 1.0 if target_room == r else 0.0
            idx += 1
        feat[idx] = 1.0 if target_room is None else 0.0
        idx += 1

        # 5. Agent normalized coordinates (2)
        feat[idx] = np.clip(agent_x / GRID_MAX_WIDTH, 0.0, 1.0)
        idx += 1
        feat[idx] = np.clip(agent_y / GRID_MAX_HEIGHT, 0.0, 1.0)
        idx += 1

        # 6. Agent room one-hot (4)
        for r in ROOM_NAMES:
            feat[idx] = 1.0 if agent_room_name == r else 0.0
            idx += 1
        feat[idx] = 1.0 if agent_room_name not in ROOM_NAMES else 0.0
        idx += 1

        # 7. Agent orientation one-hot (4)
        for d in (0, 1, 2, 3):
            feat[idx] = 1.0 if agent_dir == d else 0.0
            idx += 1

        # 8. Agent carrying features (7)
        is_carrying = 1.0 if carrying else 0.0
        feat[idx] = is_carrying
        idx += 1

        for obj in PICKABLE_OBJECTS:
            feat[idx] = 1.0 if carrying == obj else 0.0
            idx += 1
        feat[idx] = 1.0 if not carrying else 0.0
        idx += 1

        feat[idx] = 1.0 if (carrying and target_obj and carrying == target_obj) else 0.0
        idx += 1

        # 9. Facing object relation (1)
        feat[idx] = 1.0 if (front_obj and target_obj and front_obj == target_obj) else 0.0
        idx += 1

        # 10. Target object features (4)
        target_obj_info = objects.get(target_obj) if target_obj else None
        target_available = 0.0
        target_carried = 0.0
        target_dist_norm = 1.0
        target_same_room = 0.0

        if target_obj_info and not target_obj_info.get("invisible"):
            target_available = 1.0
            pos = target_obj_info.get("pos", (-1, -1))
            if pos == (-1, -1) or carrying == target_obj:
                target_carried = 1.0
                target_dist_norm = 0.0
                target_same_room = 1.0
            else:
                dist = abs(agent_x - pos[0]) + abs(agent_y - pos[1])
                target_dist_norm = np.clip(dist / MAX_MANHATTAN_DIST, 0.0, 1.0)
                obj_room = target_obj_info.get("room")
                obj_room_name = CODE_TO_ROOM.get(obj_room, obj_room)
                target_same_room = 1.0 if (obj_room_name and obj_room_name == agent_room_name) else 0.0

        feat[idx] = target_available
        idx += 1
        feat[idx] = target_carried
        idx += 1
        feat[idx] = target_dist_norm
        idx += 1
        feat[idx] = target_same_room
        idx += 1

        # 11. Target bin features (5)
        target_bin_info = objects.get(target_bin) if target_bin else None
        bin_is_open = 0.0
        bin_is_closed = 0.0
        bin_is_broken = 0.0
        bin_dist_norm = 1.0
        bin_contains_target = 0.0

        if target_bin_info:
            bin_state = target_bin_info.get("state", "closed")
            bin_is_open = 1.0 if bin_state == "open" else 0.0
            bin_is_closed = 1.0 if bin_state == "closed" else 0.0
            bin_is_broken = 1.0 if bin_state == "broken" else 0.0

            pos = target_bin_info.get("pos", (0, 0))
            dist = abs(agent_x - pos[0]) + abs(agent_y - pos[1])
            bin_dist_norm = np.clip(dist / MAX_MANHATTAN_DIST, 0.0, 1.0)

            contains = target_bin_info.get("contains") or []
            if target_obj and target_obj in contains:
                bin_contains_target = 1.0

        feat[idx] = bin_is_open
        idx += 1
        feat[idx] = bin_is_closed
        idx += 1
        feat[idx] = bin_is_broken
        idx += 1
        feat[idx] = bin_dist_norm
        idx += 1
        feat[idx] = bin_contains_target
        idx += 1

        # 12. Target room features (2)
        agent_in_target_room = 1.0 if (target_room and agent_room_name == target_room) else 0.0
        target_obj_in_target_room = 0.0
        if target_obj_info and target_room:
            obj_room = target_obj_info.get("room")
            obj_room_name = CODE_TO_ROOM.get(obj_room, obj_room)
            target_obj_in_target_room = 1.0 if obj_room_name == target_room else 0.0

        feat[idx] = agent_in_target_room
        idx += 1
        feat[idx] = target_obj_in_target_room
        idx += 1

        # 13. Progress & Bias (2)
        task_complete = 0.0
        if task_type == "find" and target_obj:
            task_complete = 1.0 if front_obj == target_obj else 0.0
        elif task_type == "get" and target_obj:
            task_complete = 1.0 if carrying == target_obj else 0.0
        elif task_type == "open" and target_bin_info:
            task_complete = 1.0 if target_bin_info.get("state") == "open" else 0.0
        elif task_type == "cleanup" and target_bin_info and target_obj:
            task_complete = 1.0 if target_obj in (target_bin_info.get("contains") or []) else 0.0
        elif task_type == "rearrange" and target_obj_info and target_room:
            target_code = ROOM_TO_CODE.get(target_room)
            task_complete = 1.0 if target_obj_info.get("room") == target_code else 0.0

        feat[idx] = task_complete
        idx += 1

        # Intercept / bias
        feat[idx] = 1.0
        idx += 1

        # 14. Task-Skill Interaction Features (12)
        if option is not None:
            if isinstance(option, str):
                s_name = option.split("(")[0].strip()
                args_str = option.split("(")[1].rstrip(")").strip() if "(" in option else ""
                s_args = tuple(a.strip() for a in args_str.split(",")) if args_str else ()
            elif hasattr(option, "name") and hasattr(option, "args"):
                s_name = option.name
                s_args = tuple(option.args)
            else:
                s_name = ""
                s_args = ()

            # 1. skill_arg_matches_target_obj
            feat[idx] = 1.0 if (target_obj and target_obj in s_args) else 0.0
            idx += 1

            # 2. skill_arg_matches_target_bin
            feat[idx] = 1.0 if (target_bin and target_bin in s_args) else 0.0
            idx += 1

            # 3. skill_arg_matches_target_room
            target_room_code = ROOM_TO_CODE.get(target_room) if target_room else None
            feat[idx] = 1.0 if (target_room and (target_room in s_args or target_room_code in s_args)) else 0.0
            idx += 1

            # 4. skill_verb_matches_task_verb
            verb_match = (
                (task_type == "find" and s_name == "find_object") or
                (task_type == "get" and s_name == "pickup_object") or
                (task_type == "open" and s_name == "open_bin") or
                (task_type == "cleanup" and s_name == "put_in_bin") or
                (task_type == "rearrange" and s_name == "move_to_room")
            )
            feat[idx] = 1.0 if verb_match else 0.0
            idx += 1

            # 5. skill_is_direct_goal_action
            is_goal = False
            if task_type == "find" and s_name == "find_object":
                is_goal = bool((target_obj and target_obj in s_args) or (target_bin and target_bin in s_args))
            elif task_type == "get" and s_name == "pickup_object":
                is_goal = bool(target_obj and target_obj in s_args)
            elif task_type == "open" and s_name == "open_bin":
                is_goal = bool(target_bin and target_bin in s_args)
            elif task_type == "cleanup" and s_name == "put_in_bin":
                is_goal = bool(target_obj and target_bin and len(s_args) == 2 and s_args[0] == target_obj and s_args[1] == target_bin)
            elif task_type == "rearrange" and s_name == "move_to_room":
                is_goal = bool(target_obj and target_room and len(s_args) == 2 and s_args[0] == target_obj and (s_args[1] == target_room or s_args[1] == target_room_code))
            feat[idx] = 1.0 if is_goal else 0.0
            idx += 1

            # 6. skill_is_prerequisite_action
            is_prereq = False
            if task_type == "cleanup":
                if carrying != target_obj and s_name == "pickup_object" and target_obj in s_args:
                    is_prereq = True
                elif carrying == target_obj and target_bin_info and target_bin_info.get("state") != "open" and s_name == "open_bin" and target_bin in s_args:
                    is_prereq = True
            elif task_type == "rearrange":
                if carrying != target_obj and s_name == "pickup_object" and target_obj in s_args:
                    is_prereq = True
            feat[idx] = 1.0 if is_prereq else 0.0
            idx += 1

            # 7. skill_verb_mismatches_task
            mismatch = False
            if task_type == "find" and s_name in ("put_in_bin", "move_to_room", "pickup_object", "open_bin"):
                mismatch = True
            elif task_type == "get" and s_name in ("put_in_bin", "move_to_room", "open_bin"):
                mismatch = True
            elif task_type == "open" and s_name in ("put_in_bin", "move_to_room", "pickup_object"):
                mismatch = True
            feat[idx] = 1.0 if mismatch else 0.0
            idx += 1

            # 8. skill_arg_mismatches_target_obj
            arg_obj_mismatch = False
            if target_obj and len(s_args) > 0 and s_args[0] in PICKABLE_OBJECTS and s_args[0] != target_obj:
                arg_obj_mismatch = True
            feat[idx] = 1.0 if arg_obj_mismatch else 0.0
            idx += 1

            # 9. skill_arg_mismatches_target_bin
            arg_bin_mismatch = False
            if target_bin:
                for a in s_args:
                    if a in STORAGE_BINS and a != target_bin:
                        arg_bin_mismatch = True
            feat[idx] = 1.0 if arg_bin_mismatch else 0.0
            idx += 1

            # 10. skill_arg_mismatches_target_room
            arg_room_mismatch = False
            if target_room:
                for a in s_args:
                    if (a in ROOM_NAMES or a in CODE_TO_ROOM) and (a != target_room and a != target_room_code):
                        arg_room_mismatch = True
            feat[idx] = 1.0 if arg_room_mismatch else 0.0
            idx += 1

            # 11. skill_all_args_match_task
            feat[idx] = 1.0 if is_goal else 0.0
            idx += 1

            # 12. skill_already_satisfied
            already_sat = False
            if s_name == "find_object" and len(s_args) == 1 and front_obj == s_args[0]:
                already_sat = True
            elif s_name == "pickup_object" and len(s_args) == 1 and carrying == s_args[0]:
                already_sat = True
            elif s_name == "open_bin" and len(s_args) == 1 and objects.get(s_args[0], {}).get("state") == "open":
                already_sat = True
            elif s_name == "move_to_room" and len(s_args) == 2:
                r_code = ROOM_TO_CODE.get(s_args[1], s_args[1])
                if objects.get(s_args[0], {}).get("room") == r_code and carrying != s_args[0]:
                    already_sat = True
            feat[idx] = 1.0 if already_sat else 0.0
            idx += 1
        else:
            # option is None: default all 12 interaction features to 0.0
            idx += 12

        assert idx == self.feature_dim, f"Feature dimension mismatch: {idx} vs {self.feature_dim}"
        return feat


# ══════════════════════════════════════════════════════════════════════════════
# ACTION MASKER
# ══════════════════════════════════════════════════════════════════════════════

class ActionMasker:
    """Evaluates concrete skill preconditions against symbolic state to prevent

    invalid skill selection.
    """

    @staticmethod
    def is_valid(
        option: SkillOption,
        state: Dict[str, Any],
        task: Optional[str] = None,
        active_subgoal: Optional[Subgoal] = None,
    ) -> bool:
        """Check whether option satisfies physical and domain preconditions in state,
        as well as task-grounding and active-subgoal constraints.
        """
        name = option.name
        args = option.args
        objects = {obj["name"]: obj for obj in state.get("objects", [])}
        agent = state.get("agent", {})
        carrying = agent.get("carrying")
        front_obj = state.get("front_obj")

        # 1. Physical & domain preconditions
        if name == "find_object":
            if len(args) != 1:
                return False
            obj_name = args[0]
            obj = objects.get(obj_name)
            if not obj or obj.get("invisible"):
                return False
            # Cannot find an object that is currently carried (pos is -1,-1)
            if carrying == obj_name or obj.get("pos") == (-1, -1):
                return False
            # Redundant if already facing and adjacent to the object
            if front_obj == obj_name:
                return False

        elif name == "pickup_object":
            if len(args) != 1:
                return False
            obj_name = args[0]
            obj = objects.get(obj_name)
            if not obj or obj.get("type") != "Pickable" or obj.get("invisible"):
                return False
            # If agent already carrying a different item, hands are full
            if carrying is not None and carrying != obj_name:
                return False
            # If already carrying this item, picking up is redundant/not executable from floor
            if carrying == obj_name:
                return False
            # Object must be available in the grid (not inside a bin or carried)
            pos = obj.get("pos", (-1, -1))
            if pos == (-1, -1):
                return False

        elif name == "open_bin":
            if len(args) != 1:
                return False
            bin_name = args[0]
            b = objects.get(bin_name)
            if not b or b.get("type") != "Storage":
                return False
            # Only valid to open if not already open
            if b.get("state") == "open":
                return False

        elif name == "put_in_bin":
            if len(args) != 2:
                return False
            obj_name, bin_name = args[0], args[1]
            obj = objects.get(obj_name)
            b = objects.get(bin_name)
            if not obj or obj.get("type") != "Pickable" or obj.get("invisible"):
                return False
            if not b or b.get("type") != "Storage":
                return False
            # Object already in bin?
            if obj_name in (b.get("contains") or []):
                return False
            # Hands constraint: must be carrying obj_name or carrying nothing
            if carrying is not None and carrying != obj_name:
                return False
            # If not carrying obj_name, object must be available in the grid to pick up
            if carrying != obj_name and obj.get("pos") == (-1, -1):
                return False

        elif name == "move_to_room":
            if len(args) != 2:
                return False
            obj_name, room_name = args[0], args[1].lower()
            if room_name not in ROOM_NAMES and room_name not in CODE_TO_ROOM:
                return False
            target_code = ROOM_TO_CODE.get(room_name, room_name)

            obj = objects.get(obj_name)
            if not obj or obj.get("type") != "Pickable" or obj.get("invisible"):
                return False
            # If already in the target room and not carried, skill is already satisfied
            if obj.get("room") == target_code and carrying != obj_name:
                return False
            # Hands constraint: must be carrying obj_name or carrying nothing
            if carrying is not None and carrying != obj_name:
                return False
            # If not carrying, object must be in the grid to pick up
            if carrying != obj_name and obj.get("pos") == (-1, -1):
                return False
        else:
            return False

        # 2. Task Grounding & Target Preservation (Active Subgoal)
        if active_subgoal is not None:
            task_obj = active_subgoal.target_obj
            task_bin = active_subgoal.target_bin
            task_room = active_subgoal.target_room

            # A. Pickable Object Grounding
            if task_obj and len(args) > 0 and args[0] in PICKABLE_OBJECTS:
                if args[0] != task_obj:
                    return False

            # B. Storage Bin Grounding
            if task_bin:
                if name == "open_bin" and len(args) > 0 and args[0] != task_bin:
                    return False
                if name == "put_in_bin" and len(args) >= 2 and args[1] != task_bin:
                    return False
                if name == "find_object" and len(args) > 0 and args[0] in STORAGE_BINS and args[0] != task_bin:
                    return False

            # C. Room Grounding
            if task_room and name == "move_to_room" and len(args) >= 2:
                r_code = ROOM_TO_CODE.get(args[1], args[1])
                target_code = ROOM_TO_CODE.get(task_room, task_room)
                if r_code != target_code:
                    return False

            # D. Active Subgoal Phase Filtering
            if active_subgoal.name == "find_object":
                target_find = active_subgoal.target_obj or active_subgoal.target_bin
                if name != "find_object" or (len(args) > 0 and args[0] != target_find):
                    return False
            elif active_subgoal.name == "pickup_object":
                if name == "pickup_object":
                    if len(args) > 0 and args[0] != active_subgoal.target_obj:
                        return False
                elif name == "find_object":
                    if len(args) > 0 and args[0] != active_subgoal.target_obj:
                        return False
                else:
                    return False
            elif active_subgoal.name == "move_to_room":
                target_room_code = ROOM_TO_CODE.get(active_subgoal.target_room, active_subgoal.target_room)
                if carrying == active_subgoal.target_obj:
                    if name != "move_to_room":
                        return False
                    if len(args) >= 2 and (args[0] != active_subgoal.target_obj or ROOM_TO_CODE.get(args[1], args[1]) != target_room_code):
                        return False
                else:
                    if name == "pickup_object":
                        if len(args) > 0 and args[0] != active_subgoal.target_obj:
                            return False
                    elif name == "find_object":
                        if len(args) > 0 and args[0] != active_subgoal.target_obj:
                            return False
                    else:
                        return False
            elif active_subgoal.name == "open_bin":
                if name == "open_bin":
                    if len(args) > 0 and args[0] != active_subgoal.target_bin:
                        return False
                elif name == "find_object":
                    if len(args) > 0 and args[0] != active_subgoal.target_bin:
                        return False
                else:
                    return False
            elif active_subgoal.name == "put_in_bin":
                if name == "put_in_bin":
                    if len(args) >= 2 and (args[0] != active_subgoal.target_obj or args[1] != active_subgoal.target_bin):
                        return False
                elif name == "open_bin":
                    if len(args) > 0 and args[0] != active_subgoal.target_bin:
                        return False
                elif name in ("pickup_object", "find_object"):
                    if len(args) > 0 and args[0] != active_subgoal.target_obj:
                        return False
                else:
                    return False

        return True

    def get_valid_mask(
        self,
        candidates: Sequence[SkillOption],
        state: Dict[str, Any],
        task: Optional[str] = None,
        active_subgoal: Optional[Subgoal] = None,
    ) -> List[bool]:
        """Compute boolean validity mask for candidates in the given state."""
        return [self.is_valid(opt, state, task, active_subgoal) for opt in candidates]

    def get_valid_options(
        self,
        candidates: Sequence[SkillOption],
        state: Dict[str, Any],
        task: Optional[str] = None,
        active_subgoal: Optional[Subgoal] = None,
    ) -> List[SkillOption]:
        """Filter candidates returning only legal, precondition-satisfying options."""
        return [opt for opt in candidates if self.is_valid(opt, state, task, active_subgoal)]


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR SMDP SKILL SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

class LinearSMDPSkillSelector:
    """Linear SMDP Skill Selector with Linear Function Approximation.

    Evaluates Q(s, option) = theta_option^T @ phi(s, task) and supports
    epsilon-greedy exploration over precondition-masked options.
    """

    def __init__(
        self,
        featurizer: Optional[StateFeaturizer] = None,
        masker: Optional[ActionMasker] = None,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        default_options: Optional[Sequence[SkillOption]] = None,
    ) -> None:
        self.featurizer = featurizer or StateFeaturizer()
        self.masker = masker or ActionMasker()
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.feature_dim = self.featurizer.feature_dim

        # Mapping: option_id (str) -> np.ndarray of shape (feature_dim,)
        self.weights: Dict[str, np.ndarray] = {}
        # Shared linear weights across all options for cross-task feature transfer
        self.shared_weights: np.ndarray = np.zeros(self.feature_dim, dtype=np.float64)

        if default_options:
            for opt in default_options:
                self.get_weights(opt)

    def _normalize_option(self, option: Union[SkillOption, str]) -> SkillOption:
        if isinstance(option, str):
            return SkillOption.from_str(option)
        return option

    def get_weights(self, option: Union[SkillOption, str]) -> np.ndarray:
        """Get the option-specific linear weight vector theta for an option (initialized to zeros if new)."""
        opt = self._normalize_option(option)
        opt_id = opt.id
        if opt_id not in self.weights:
            self.weights[opt_id] = np.zeros(self.feature_dim, dtype=np.float64)
        return self.weights[opt_id]

    def set_weights(self, option: Union[SkillOption, str], w: np.ndarray) -> None:
        """Set weights for an option, enforcing dimension match."""
        opt = self._normalize_option(option)
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (self.feature_dim,):
            raise ValueError(f"Weight vector shape {w.shape} != expected ({self.feature_dim},)")
        self.weights[opt.id] = w

    def get_q_value(self, option: Union[SkillOption, str], features: np.ndarray) -> float:
        """Calculate Q(s, option) = (theta_option + w_shared)^T @ features."""
        w = self.get_weights(option).copy()
        w[49:] += self.shared_weights[49:]
        return float(np.dot(w, features))

    def get_q_values(
        self,
        options: Sequence[Union[SkillOption, str]],
        features: Optional[np.ndarray] = None,
        state: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute Q(s, option) across multiple candidate options."""
        res: Dict[str, float] = {}
        for opt in options:
            norm_opt = self._normalize_option(opt)
            if state is not None and task is not None:
                opt_feat = self.featurizer.featurize(state, task, option=norm_opt)
            elif features is not None:
                opt_feat = features
            else:
                opt_feat = np.zeros(self.feature_dim, dtype=np.float64)
            res[norm_opt.id] = self.get_q_value(norm_opt, opt_feat)
        return res

    def select_option(
        self,
        state: Dict[str, Any],
        task: str,
        candidates: Optional[Sequence[Union[SkillOption, str]]] = None,
        epsilon: Optional[float] = None,
        active_subgoal: Optional[Subgoal] = None,
    ) -> Optional[SkillOption]:
        """Select a skill option from valid candidates using epsilon-greedy exploration.

        Returns None if no candidates are valid in the current state.
        """
        eps = self.epsilon if epsilon is None else float(epsilon)

        if candidates is None:
            cand_list = get_all_standard_options()
        else:
            cand_list = [self._normalize_option(c) for c in candidates]

        # Action masking
        valid_options = self.masker.get_valid_options(
            cand_list, state, task, active_subgoal=active_subgoal
        )
        if not valid_options:
            return None

        # Epsilon-greedy exploration
        if len(valid_options) > 1 and self.rng.random() < eps:
            idx = int(self.rng.integers(0, len(valid_options)))
            return valid_options[idx]

        # Greedy exploitation: argmax_a Q(s, a)
        q_vals = [
            self.get_q_value(
                opt,
                self.featurizer.featurize(
                    state, task, option=opt, active_subgoal=active_subgoal
                ),
            )
            for opt in valid_options
        ]
        max_q = max(q_vals)

        # Break ties uniformly at random
        best_indices = [i for i, q in enumerate(q_vals) if np.isclose(q, max_q, atol=1e-8)]
        chosen_idx = int(self.rng.choice(best_indices))
        return valid_options[chosen_idx]

    def smdp_td_update(
        self,
        option: Union[SkillOption, str],
        features: np.ndarray,
        td_error: float,
        alpha: float,
    ) -> None:
        """Linear SMDP Q-Learning weight update: theta <- theta + alpha * td_error * phi."""
        opt = self._normalize_option(option)
        w = self.get_weights(opt)
        delta = float(alpha) * float(td_error) * features
        w += delta
        self.shared_weights[49:] += delta[49:]

    def explain_decision(
        self,
        state: Dict[str, Any],
        task: str,
        option: Union[SkillOption, str],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Decompose Q(s, option) into its most influential linear component features."""
        opt = self._normalize_option(option)
        features = self.featurizer.featurize(state, task, option=opt)
        w = self.get_weights(opt).copy()
        w[49:] += self.shared_weights[49:]
        contributions = w * features
        q_val = float(np.sum(contributions))

        top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]
        top_factors = [
            {
                "feature": self.featurizer.feature_names[i],
                "feature_value": float(features[i]),
                "weight": float(w[i]),
                "contribution": float(contributions[i]),
            }
            for i in top_indices
        ]

        return {
            "option": opt.id,
            "q_value": q_val,
            "top_features": top_factors,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist selector weights and configuration to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "Linear_SMDP_Q_Learning",
            "feature_dim": self.feature_dim,
            "feature_names": self.featurizer.feature_names,
            "epsilon": self.epsilon,
            "shared_weights": self.shared_weights.tolist(),
            "weights": {k: v.tolist() for k, v in self.weights.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, filepath: Union[str, Path]) -> None:
        """Load selector weights and configuration from a JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("feature_dim") != self.feature_dim:
            raise ValueError(
                f"Saved feature_dim {payload.get('feature_dim')} != "
                f"expected {self.feature_dim}"
            )

        self.epsilon = float(payload.get("epsilon", self.epsilon))
        if "shared_weights" in payload:
            self.shared_weights = np.array(payload["shared_weights"], dtype=np.float64)
        else:
            self.shared_weights = np.zeros(self.feature_dim, dtype=np.float64)
        self.weights = {
            k: np.array(v, dtype=np.float64)
            for k, v in payload.get("weights", {}).items()
        }

    def update_smdp(
        self,
        transition: "SMDPTransition",
        alpha: float = 0.05,
        gamma: float = 0.95,
        candidates: Optional[Sequence[Union[SkillOption, str]]] = None,
    ) -> float:
        """Perform linear SMDP Q-Learning weight update:

        y = R + (gamma^tau * max_{a' in Valid(s')} Q(s', a') if not done else 0)
        delta = y - Q(s, a)
        theta_a <- theta_a + alpha * delta * phi(s, task, a)

        Returns:
            delta: TD error for this transition
        """
        phi_s = self.featurizer.featurize(transition.state, transition.task, option=transition.option)
        q_current = self.get_q_value(transition.option, phi_s)

        if transition.done:
            # Terminal state transition: strictly no bootstrap!
            target = float(transition.reward)
        else:
            # Non-terminal state: bootstrap with max Q over valid options in s'
            if candidates is None:
                cand_list = get_all_standard_options()
            else:
                cand_list = [self._normalize_option(c) for c in candidates]

            valid_next = self.masker.get_valid_options(
                cand_list, transition.next_state, transition.task
            )
            if valid_next:
                q_next_max = max(
                    self.get_q_value(opt, self.featurizer.featurize(transition.next_state, transition.task, option=opt))
                    for opt in valid_next
                )
            else:
                q_next_max = 0.0

            # Duration discounting by gamma^tau
            duration_discount = float(gamma) ** int(transition.duration)
            target = float(transition.reward) + duration_discount * float(q_next_max)

        td_error = target - q_current
        self.smdp_td_update(transition.option, phi_s, td_error, alpha)
        return float(td_error)


# ══════════════════════════════════════════════════════════════════════════════
# SMDP TRANSITIONS & STEP TRACKER
# ══════════════════════════════════════════════════════════════════════════════

def task_is_complete(task: str, state: Dict[str, Any]) -> bool:
    """Return whether the original task is satisfied in symbolic state."""
    task_lower = (task or "").lower().strip()
    objects = {obj["name"]: obj for obj in state.get("objects", [])}
    front_obj = state.get("front_obj")
    carrying = state.get("agent", {}).get("carrying")

    # Compound task verification
    subgoals = parse_compound_task(task)
    if subgoals:
        target_bin = None
        target_obj = None
        target_room = None
        for sg in subgoals:
            if sg.target_obj:
                target_obj = sg.target_obj
            if sg.target_bin:
                target_bin = sg.target_bin
            if sg.target_room:
                target_room = sg.target_room

        last_sg = subgoals[-1]
        if last_sg.name == "put_in_bin":
            b_name = last_sg.args[1] if len(last_sg.args) >= 2 else target_bin
            b = objects.get(b_name)
            obj = last_sg.args[0] if len(last_sg.args) >= 1 else target_obj
            return bool(b and obj in (b.get("contains") or []))
        elif last_sg.name == "move_to_room":
            obj = last_sg.args[0] if len(last_sg.args) >= 1 else target_obj
            room = last_sg.args[1] if len(last_sg.args) >= 2 else target_room
            target_code = ROOM_TO_CODE.get(room)
            return bool(objects.get(obj) and objects[obj].get("room") == target_code)
        elif last_sg.name == "pickup_object":
            obj = last_sg.args[0] if len(last_sg.args) >= 1 else target_obj
            return carrying == obj
        elif last_sg.name == "find_object":
            target = last_sg.args[0] if len(last_sg.args) >= 1 else (target_obj or target_bin)
            return front_obj == target
        elif last_sg.name == "open_bin":
            b_name = last_sg.args[0] if len(last_sg.args) >= 1 else target_bin
            b = objects.get(b_name)
            return bool(b and b.get("state") == "open")

    if task_lower.startswith("find the "):
        target = task_lower.removeprefix("find the ").strip()
        return front_obj == target
    if task_lower.startswith("get the "):
        target = task_lower.removeprefix("get the ").strip()
        return carrying == target
    if task_lower.startswith("open the "):
        target = task_lower.removeprefix("open the ").strip()
        b = objects.get(target)
        return bool(b and b.get("state") == "open")
    if task_lower.startswith("put the ") and " in the " in task_lower:
        parts = task_lower.removeprefix("put the ").split(" in the ", 1)
        if len(parts) == 2:
            obj_name, bin_name = parts[0].strip(), parts[1].strip()
            b = objects.get(bin_name)
            return bool(b and obj_name in (b.get("contains") or []))
    if task_lower.startswith("move the ") and " to the " in task_lower:
        parts = task_lower.removeprefix("move the ").split(" to the ", 1)
        if len(parts) == 2:
            obj_name, room_name = parts[0].strip(), parts[1].strip()
            target_code = ROOM_TO_CODE.get(room_name)
            return bool(objects.get(obj_name) and objects[obj_name].get("room") == target_code)
    return False


@dataclass
class SMDPTransition:
    """Semi-Markov Decision Process transition record for one skill execution."""

    state: Dict[str, Any]
    option: SkillOption
    reward: float  # Cumulative discounted reward: sum_{k=0}^{tau-1} gamma^k r_k
    duration: int  # Duration in primitive environment steps: tau
    next_state: Dict[str, Any]
    done: bool  # Terminal flag (task complete or episode ended)
    task: str
    success: bool = True


class SMDPExecutionTracker:
    """Environment step interceptor that accumulates primitive environment rewards and durations."""

    def __init__(self, env: Any, gamma: float = 0.95) -> None:
        self.env = env
        self.gamma = float(gamma)
        self.current_option: Optional[SkillOption] = None
        self.current_duration: int = 0
        self.current_discounted_reward: float = 0.0
        self.is_terminal: bool = False
        self.total_env_steps: int = 0
        self.episode_discounted_reward: float = 0.0
        self.start_state: Optional[Dict[str, Any]] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _get_symbolic_state(self) -> Dict[str, Any]:
        if hasattr(self.env, "unwrapped") and hasattr(
            self.env.unwrapped, "get_full_symbolic_state"
        ):
            return self.env.unwrapped.get_full_symbolic_state()
        if hasattr(self.env, "get_full_symbolic_state"):
            return self.env.get_full_symbolic_state()
        if hasattr(self.env, "symbolic_state"):
            return self.env.symbolic_state
        return {"agent": {"carrying": None, "pos": (0, 0), "room": None}, "objects": []}

    def start_skill(self, option: SkillOption) -> Dict[str, Any]:
        """Begin tracking a skill execution."""
        self.current_option = option
        self.current_duration = 0
        self.current_discounted_reward = 0.0
        self.start_state = self._get_symbolic_state()
        return self.start_state

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Intercept low-level step to accumulate discounted rewards and step counts."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_duration += 1
        self.total_env_steps += 1
        # Discount reward by steps elapsed within this skill execution: gamma^(tau-1) * r
        discount = self.gamma ** (self.current_duration - 1)
        self.current_discounted_reward += discount * float(reward)
        self.episode_discounted_reward += (self.gamma ** (self.total_env_steps - 1)) * float(reward)
        if terminated or truncated:
            self.is_terminal = True
        return obs, reward, terminated, truncated, info

    def end_skill(self, success: bool, task: str) -> SMDPTransition:
        """Finalize skill tracking and return the complete SMDP transition."""
        next_state = self._get_symbolic_state()
        done = self.is_terminal or task_is_complete(task, next_state)
        transition = SMDPTransition(
            state=self.start_state or next_state,
            option=self.current_option or SkillOption("unknown", ()),
            reward=float(self.current_discounted_reward),
            duration=int(self.current_duration),
            next_state=next_state,
            done=bool(done),
            task=task,
            success=bool(success),
        )
        return transition


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION, TRAINING, AND EVALUATION HARNESS
# ══════════════════════════════════════════════════════════════════════════════

def get_skill_function(skill_name: str) -> Any:
    """Safely import and retrieve the primitive skill function from agent/skills.py."""
    try:
        from skills import SKILL_FUNCTIONS
        return SKILL_FUNCTIONS.get(skill_name)
    except Exception:
        return None


def execute_smdp_skill(
    env: Any,
    option: SkillOption,
    task: str,
    tracker: Optional[SMDPExecutionTracker] = None,
    gamma: float = 0.95,
    log=print,
) -> Tuple[bool, SMDPTransition]:
    """Execute a single skill option on the environment while tracking SMDP metrics."""
    if tracker is None:
        tracker = SMDPExecutionTracker(env, gamma=gamma)

    fn = get_skill_function(option.name)
    if fn is None:
        log(f"  [SMDP] Unknown skill: {option.name}")
        tracker.start_skill(option)
        trans = tracker.end_skill(success=False, task=task)
        return False, trans

    tracker.start_skill(option)
    try:
        success, _ = fn(tracker, *option.args, log=log)
    except Exception as exc:
        log(f"  [SMDP] Exception executing {option}: {exc}")
        success = False

    trans = tracker.end_skill(success=success, task=task)
    return success, trans


def run_smdp_episode(
    env: Any,
    selector: Optional[LinearSMDPSkillSelector] = None,
    task: Optional[str] = None,
    max_skills: int = 10,
    alpha: float = 0.05,
    gamma: float = 0.95,
    epsilon: Optional[float] = None,
    training: bool = False,
    random_baseline: bool = False,
    candidates: Optional[Sequence[SkillOption]] = None,
    log=print,
) -> Dict[str, Any]:
    """Run an entire episode with SMDP skill selection.

    Supports training (with TD updates), evaluation (greedy), and random baseline modes.
    """
    if selector is None:
        selector = LinearSMDPSkillSelector(epsilon=0.0)

    # Use task from env if not explicitly provided
    if task is None:
        task = getattr(env, "task", "")

    tracker = SMDPExecutionTracker(env, gamma=gamma)
    cand_list = candidates or get_all_standard_options()

    transitions: List[SMDPTransition] = []
    total_reward: float = 0.0
    failed_skills: int = 0
    executed_skills: List[SkillOption] = []

    # Initialize subgoal tracking for compound tasks
    subgoals = parse_compound_task(task)
    tracker_sg: Optional[SubgoalTracker] = SubgoalTracker(task, subgoals) if subgoals else None

    # Get initial state
    state = tracker._get_symbolic_state()
    initial_complete = task_is_complete(task, state)

    if initial_complete:
        return {
            "task": task,
            "success": True,
            "total_reward": 1.0,
            "total_steps": 0,
            "skills_executed": [],
            "failed_skills": 0,
            "transitions": [],
        }

    if tracker_sg:
        log(f"  [SMDP] Compound task detected: {len(tracker_sg.subgoals)} subgoals planned:")
        for idx_sg, sg in enumerate(tracker_sg.subgoals, 1):
            log(f"    {idx_sg}. {sg.description} [{sg.id}]")
        tracker_sg.check_initial_satisfaction(state, log=log)

    for skill_step in range(1, max_skills + 1):
        active_sg = tracker_sg.active_subgoal if tracker_sg else None

        # 1. Action Masking: identify valid skills respecting active subgoal and state
        valid_options = selector.masker.get_valid_options(
            cand_list, state, task, active_subgoal=active_sg
        )
        if not valid_options:
            # If active subgoal appears to have no candidates, check if already satisfied
            if tracker_sg and active_sg and active_sg.is_satisfied(state):
                tracker_sg.step(SkillOption(active_sg.name, active_sg.args), True, state, log=log)
                if tracker_sg.is_complete:
                    break
                active_sg = tracker_sg.active_subgoal
                valid_options = selector.masker.get_valid_options(
                    cand_list, state, task, active_subgoal=active_sg
                )

            if not valid_options:
                log(f"  [SMDP] Step {skill_step}: No valid skills available for state"
                    + (f" (active subgoal: {active_sg.description})" if active_sg else ""))
                break

        # 2. Skill Selection
        if random_baseline:
            # Uniform random choice among valid options
            idx = int(selector.rng.integers(0, len(valid_options)))
            chosen_option = valid_options[idx]
        else:
            # Greedy when evaluating; epsilon-greedy when training
            eps = (selector.epsilon if epsilon is None else epsilon) if training else 0.0
            chosen_option = selector.select_option(
                state, task, candidates=valid_options, epsilon=eps, active_subgoal=active_sg
            )
            if chosen_option is None:
                break

        subgoal_info = f" [Subgoal: {active_sg.description}]" if active_sg else ""
        log(f"  [SMDP] Step {skill_step}: Selected skill '{chosen_option}' "
            f"(valid candidates: {len(valid_options)}){subgoal_info}")
        executed_skills.append(chosen_option)

        # 3. Execute Skill with SMDP tracking
        success, transition = execute_smdp_skill(
            tracker, chosen_option, task, tracker=tracker, gamma=gamma, log=log
        )
        transitions.append(transition)
        total_reward += transition.reward
        if not success:
            failed_skills += 1

        # 4. Advance Subgoal Tracker
        if tracker_sg:
            tracker_sg.step(chosen_option, success, transition.next_state, log=log)

        # 5. SMDP Q-Learning weight update (if training)
        if training:
            td_error = selector.update_smdp(
                transition, alpha=alpha, gamma=gamma, candidates=cand_list
            )
            log(f"  [SMDP] TD Error: {td_error:+.4f} | "
                f"Reward: {transition.reward:.2f} | Duration: {transition.duration} steps")

        # Advance state
        state = transition.next_state

        # Check termination
        if tracker_sg and tracker_sg.is_complete:
            log(f"  [SMDP] All compound subgoals completed at skill step {skill_step}!")
            break
        elif transition.done:
            log(f"  [SMDP] Task completed or episode terminated at skill step {skill_step}")
            break

        # Safe loop-prevention: halt if same skill is executed consecutively without state change
        if len(executed_skills) >= 2 and executed_skills[-1] == executed_skills[-2]:
            if transition.state == transition.next_state:
                log(f"  [SMDP] Loop prevented: '{chosen_option}' executed without state progress. Halting episode.")
                break

    final_success = (tracker_sg.is_complete if tracker_sg else False) or task_is_complete(task, state)
    return {
        "task": task,
        "success": bool(final_success),
        "total_reward": float(total_reward),
        "total_steps": int(tracker.total_env_steps),
        "skills_executed": [opt.id for opt in executed_skills],
        "failed_skills": int(failed_skills),
        "transitions": transitions,
    }


def train_smdp(
    env: Any,
    selector: LinearSMDPSkillSelector,
    num_episodes: int = 50,
    alpha: float = 0.05,
    gamma: float = 0.95,
    epsilon: float = 0.15,
    policy_save_path: Optional[Union[str, Path]] = None,
    log=print,
) -> List[Dict[str, Any]]:
    """Train the Linear SMDP Skill Selector over a series of environment episodes."""
    results: List[Dict[str, Any]] = []

    log(f"[SMDP Training] Starting {num_episodes} episodes (alpha={alpha}, gamma={gamma}, eps={epsilon})...")

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        task = getattr(env, "task", "")
        log(f"\n--- Episode {ep}/{num_episodes} | Task: {task} ---")

        result = run_smdp_episode(
            env,
            selector=selector,
            task=task,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            training=True,
            log=log,
        )
        results.append(result)
        status = "[OK]" if result["success"] else "[FAIL]"
        log(f"Episode {ep} {status} | Steps: {result['total_steps']} | Reward: {result['total_reward']:.2f}")

    if policy_save_path:
        selector.save(policy_save_path)
        log(f"[SMDP Training] Policy saved to: {policy_save_path}")

    return results


def evaluate_smdp(
    env: Any,
    selector: LinearSMDPSkillSelector,
    num_episodes: int = 20,
    candidates: Optional[Sequence[SkillOption]] = None,
    log=print,
) -> Dict[str, Any]:
    """Evaluate a trained Linear SMDP Skill Selector with purely greedy selection."""
    results: List[Dict[str, Any]] = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        task = getattr(env, "task", "")
        res = run_smdp_episode(
            env,
            selector=selector,
            task=task,
            training=False,
            candidates=candidates,
            log=log,
        )
        results.append(res)

    success_count = sum(1 for r in results if r["success"])
    total_steps = sum(r["total_steps"] for r in results)
    total_reward = sum(r["total_reward"] for r in results)
    total_failed_skills = sum(r["failed_skills"] for r in results)

    n = max(1, len(results))
    return {
        "episodes": len(results),
        "success_rate": success_count / n,
        "avg_steps": total_steps / n,
        "avg_reward": total_reward / n,
        "avg_failed_skills": total_failed_skills / n,
        "results": results,
    }


def evaluate_random_baseline(
    env: Any,
    num_episodes: int = 20,
    candidates: Optional[Sequence[SkillOption]] = None,
    seed: Optional[int] = 42,
    log=print,
) -> Dict[str, Any]:
    """Evaluate a baseline agent that selects valid skills uniformly at random."""
    selector = LinearSMDPSkillSelector(seed=seed)
    results: List[Dict[str, Any]] = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        task = getattr(env, "task", "")
        res = run_smdp_episode(
            env,
            selector=selector,
            task=task,
            training=False,
            random_baseline=True,
            candidates=candidates,
            log=log,
        )
        results.append(res)

    success_count = sum(1 for r in results if r["success"])
    total_steps = sum(r["total_steps"] for r in results)
    total_reward = sum(r["total_reward"] for r in results)
    total_failed_skills = sum(r["failed_skills"] for r in results)

    n = max(1, len(results))
    return {
        "episodes": len(results),
        "success_rate": success_count / n,
        "avg_steps": total_steps / n,
        "avg_reward": total_reward / n,
        "avg_failed_skills": total_failed_skills / n,
        "results": results,
    }

