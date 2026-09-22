# PROJECT_NOTES.md — Developer Notes for the Voyager-Inspired HomeGrid Agent

---

## 1. Project Goal

Build the **smallest possible working Voyager-inspired agent** for the HomeGrid environment.

The original Voyager (Wang et al., 2023) was an LLM-powered agent for Minecraft that:
- Used GPT-4 to write JavaScript code snippets (skills)
- Stored those code snippets in a skill library
- Retrieved and reused skills using embedding-based search
- Had an automated critic that verified whether skills worked

Our project keeps the **core ideas** but dramatically simplifies the implementation to make it understandable in a college AI course setting.

---

## 2. HomeGrid Analysis

### Environment Summary

HomeGrid (Lin et al., 2023) is a 3-room gridworld environment for studying language-grounded agents.

**Layout:**
```
Kitchen (K) | Living Room (L)
            |
Dining Room (D)
```

**Objects that can be picked up (Pickable):**
- bottle, fruit, papers, plates

**Bins that store objects (Storage):**
- recycling bin, trash bin, compost bin
- Each bin opens with ONE specific action: `pedal`, `grasp`, or `lift`
- This action is RANDOMIZED each episode
- Wrong action → bin becomes "broken" temporarily

**Action space (integers 0–9):**
```
0  left    — turn left and move
1  right   — turn right and move
2  up      — turn up and move
3  down    — turn down and move
4  pickup  — pick up the object in front
5  drop    — drop carried object / put into open bin
6  get     — take object out of open bin
7  pedal   — open bin (if this is its required action)
8  grasp   — open bin (if this is its required action)
9  lift    — open bin (if this is its required action)
```

**Observation space:**
- `obs["image"]` — 96×96×3 RGB pixel image (partial agent view)
- `obs["token"]` — T5 token ID (language stream, one token per step)
- `obs["token_embed"]` — 512-dim T5 embedding of the token
- `obs["log_language_info"]` — human-readable current language string
- `obs["is_read_step"]` — True during pre-episode reading phase

**`info` dict (crucial for our agent):**
```python
info["symbolic_state"]  # Complete symbolic world state
info["success"]         # None / True / False
info["events"]          # Things that happened this step
```

**`info["symbolic_state"]` contents:**
```python
{
  "step": 42,
  "agent": {
    "pos": (7, 5),
    "room": "K",          # K=kitchen, L=living room, D=dining room
    "dir": 2,             # 0=right, 1=down, 2=left, 3=up
    "carrying": "bottle"  # or None
  },
  "objects": [
    {
      "name": "bottle",
      "type": "Pickable",
      "pos": (3, 8),
      "room": "K",
      "state": None,
      "action": None,
      "invisible": False,
      "contains": None
    },
    {
      "name": "trash bin",
      "type": "Storage",
      "pos": (11, 1),
      "room": "L",
      "state": "closed",    # open / closed / broken
      "action": "pedal",    # the correct opening action
      "invisible": None,
      "contains": []
    }
  ],
  "front_obj": "bottle",   # name of object directly in front, or None
}
```

**Task types (from MultitaskWrapper):**
- `"find the <object>"` → reward=1 when `front_obj == object`
- `"get the <object>"` → reward=1 when `carrying == object`
- `"put the <object> in the <bin>"` → reward=1 when bin contains object
- `"move the <object> to the <room>"` → reward=1 when object.room == room
- `"open the <bin>"` → reward=1 when bin.state == "open"

Reward is **0.5** for subtasks (picking up object en route to bin).
The task is **truncated** after 100 steps.

**Relevant source files:**
- `homegrid/homegrid_base.py` — Environment, actions, symbolic_state, reset/step
- `homegrid/language_wrappers.py` — MultitaskWrapper (task sampling, reward), LanguageWrapper
- `homegrid/layout.py` — Room layout, TRASH, CANS lists
- `homegrid/base.py` — Storage, Pickable, Grid classes
- `homegrid/manual_control.py` — Human keyboard controller (reference for actions)

---

## 3. Architecture

```
HomeGrid Environment
        │
        │ env.reset() → obs, info
        │ env.step(int) → obs, reward, terminated, truncated, info
        │
        ↓
describe_state(symbolic_state)
        │  Converts symbolic_state to plain English
        ↓
get_relevant_skills(task)
        │  Keyword search in skills.json
        ↓
ask_llm(prompt)
        │  OpenAI Chat API call
        │  Returns ordered list of skill calls
        ↓
_parse_plan(response)
        │  Regex parsing: "1. pickup_object(bottle)" → [("pickup_object", ["bottle"])]
        ↓
execute_plan(env, plan)
        │  For each (skill_name, args) in plan:
        │    Call primitive skill function
        ↓
Primitive Skill Function (e.g., skill_put_in_bin)
        │  BFS navigation + HomeGrid actions
        ↓
HomeGrid (obs, reward, ...)
        │
        ↓
Success? ──Yes──→ save_skill() → skills.json grows
        │
        No
        ↓
plan_correction() ──LLM──→ new plan → retry
```

---

## 4. File-by-File Explanation

### `agent/config.py`
Central configuration. All constants live here:
- `OPENAI_MODEL` — which LLM to use
- `MAX_RETRIES` — how many times to retry a failed task
- `MAX_NAV_STEPS` — timeout for navigation BFS
- `CURRICULUM_TASK_TYPES` — the progression of task types
- `SKILLS_FILE`, `LOGS_DIR` — paths

**Why config.py exists:** So students can change settings in one place instead of hunting through code.

---

### `agent/llm.py`
The LLM interface. One function: `ask_llm(prompt) -> str`.

Sends the prompt to OpenAI's Chat API with a system message that explains the agent's context (3-room house, objects, bins).

Uses `temperature=0.2` for low randomness — we want consistent, structured plans.

**The LLM is never fine-tuned.** It is used as-is, off-the-shelf.

---

### `agent/skills.py`
Two things combined in one file for simplicity:

**Part 1 — Primitive Skills (deterministic Python functions):**
- `skill_find_object(env, obj_name)` — BFS navigate and face object
- `skill_pickup_object(env, obj_name)` — navigate then pickup
- `skill_open_bin(env, bin_name)` — navigate then use correct open action
- `skill_put_in_bin(env, obj_name, bin_name)` — pickup → open bin → drop
- `skill_move_to_room(env, obj_name, room_name)` — pickup → walk to room → drop

Navigation uses **BFS (Breadth-First Search)** on the 2D grid using positions from `symbolic_state`. This is deterministic and reliable — the LLM does not need to generate individual movement steps.

**Part 2 — Skill Library (JSON-based external memory):**
- `load_skills()` — read skills.json
- `save_skill(name, description, task_type)` — add/update skill in skills.json
- `get_relevant_skills(task)` — keyword search for relevant past skills

`SKILL_FUNCTIONS` dict maps skill name strings (what the LLM outputs) to the actual Python functions.

---

### `agent/agent.py`
Core agent logic:

- `describe_state(state)` → English description of world state (sent to LLM)
- `plan_task(task, state)` → builds prompt, calls `ask_llm()`, parses response
- `plan_correction(task, state, failed_plan, error)` → self-correction prompt
- `_parse_plan(response)` → regex parsing of LLM skill list
- `execute_plan(env, plan)` → runs each skill, collects results
- `run_task(env)` → the main loop: plan → execute → feedback → correct → save

---

### `agent/main.py`
Entry point:
- Creates HomeGrid environment
- Sets up logging (terminal + file)
- Runs through `CURRICULUM_TASK_TYPES`
- Calls `run_task()` for each episode
- Prints summary and saves JSON log

---

### `skills/skills.json`
The external skill memory. Starts empty (`{"skills": []}`).

Grows as the agent succeeds at tasks:
```json
{
  "skills": [
    {
      "name": "find_bottle",
      "description": "Solved: find the bottle",
      "task_type": "find",
      "success_count": 2
    },
    {
      "name": "put_bottle_in_trash_bin",
      "description": "Solved: put the bottle in the trash bin",
      "task_type": "cleanup",
      "success_count": 1
    }
  ]
}
```

**The LLM's weights never change.** Only this file grows.

---

## 5. Important Technical Concepts

### Agent
A program that observes an environment and takes actions to achieve goals.

### Environment
HomeGrid — the 3-room house simulation. Exposes `reset()` and `step()`.

### Observation
What the agent perceives each step: pixel image, language tokens, and (crucially for us) `symbolic_state` — the structured world description.

### Action
One of 10 integers (0–9) corresponding to movement or object interactions.

### Reward
A scalar signal from the environment. HomeGrid gives:
- `1.0` on task completion
- `0.5` on subtask completion (picking up an object for a cleanup task)
- `-1.0` if the agent walks into a hazardous area (spill)
- `0` otherwise

### LLM (Large Language Model)
A pretrained neural network (GPT-4o-mini) that can understand and generate text. We use it as a **black-box reasoner** — we give it a task description and world state, and it tells us what skills to use.

### Prompting
How we "talk" to the LLM. We craft text prompts that include:
- The task
- The current world state (in plain English)
- Available skills
- Examples of the expected output format

### Skill
A named, reusable action sequence. In our system, a skill is a Python function that calls HomeGrid primitives.

### Skill Library
`skills/skills.json` — external memory that stores which skills have worked before. The LLM sees the names of past successful skills in its prompt, so it can try to reuse them.

### Skill Retrieval
Finding relevant skills from the library for the current task. We use **keyword matching** (simple, fast, understandable). Future work: embedding-based semantic search.

### Planning
The LLM's job: given a task and world state, decide which skills to run and in what order. Output is a numbered list like:
```
1. pickup_object(bottle)
2. open_bin(trash bin)
3. put_in_bin(bottle, trash bin)
```

### Execution
Running the plan: calling each skill function in order, which in turn sends primitive actions to HomeGrid.

### Environment Feedback
What HomeGrid returns after each action: new observation, reward, and whether the episode ended. We use this to detect success or failure.

### Self-Correction
When a plan fails, we send the failure information back to the LLM and ask for a corrected plan. This loop runs up to `MAX_RETRIES` times.

### Curriculum
Running tasks in order of increasing difficulty:
`find → get → open → cleanup → rearrange`

### Why this is NOT fine-tuning
Fine-tuning means updating the LLM's weights using gradient descent on a training dataset. We never do this. The LLM's weights are frozen. The agent "learns" only by:
1. Saving successful skill names to `skills.json`
2. Including those skill names in future prompts

This is **in-context adaptation**, not fine-tuning.

### Why this is Voyager-inspired
Voyager (original):
- GPT-4 writes JavaScript code (skills)
- Stores skills in a vector database
- Retrieves skills with embeddings
- Has an automated code-testing critic

Our system:
- GPT-4o-mini selects pre-built Python skills (not writing new code)
- Stores skill names in a JSON file
- Retrieves skills with keyword matching
- Success/failure from the environment is the critic

Same conceptual pipeline, much simpler implementation.

---

## 6. Voyager Comparison

| Feature | Original Voyager | Our HomeGrid Agent |
|---------|-----------------|-------------------|
| Environment | Minecraft | HomeGrid |
| LLM | GPT-4 | GPT-4o-mini |
| Skill format | JavaScript code generated by LLM | Pre-built Python functions |
| Skill storage | Vector database (Chroma) | JSON file |
| Skill retrieval | Embedding similarity | Keyword matching |
| Critic | LLM checks if code runs correctly | Environment reward (reward=1 = success) |
| Navigation | Minecraft bot API | BFS pathfinding on symbolic grid |
| Self-correction | LLM rewrites code | LLM selects different skill sequence |
| Curriculum | Auto-generated by LLM | Fixed task type list |
| Fine-tuning | None | None |

Both systems: **LLM never changes. Skill library grows.**

---

## 7. API Usage Flow

```
1. env.reset() called
      ↓
2. env.task extracted → e.g., "put the bottle in the trash bin"
      ↓
3. symbolic_state read → describe_state() → plain English
      ↓
4. skills.json loaded → get_relevant_skills(task) → keyword search
      ↓
5. Prompt built:
   "TASK: put the bottle in the trash bin
    CURRENT WORLD STATE: ...
    AVAILABLE SKILLS: ...
    PREVIOUSLY LEARNED SKILLS: ..."
      ↓
6. OpenAI API call: POST /v1/chat/completions
   model: gpt-4o-mini
   temperature: 0.2
      ↓
7. Response parsed:
   "1. pickup_object(bottle)
    2. open_bin(trash bin)
    3. put_in_bin(bottle, trash bin)"
      ↓
8. Plan executed (no more LLM calls during execution)
      ↓
9. If failed: repeat from step 5 with failure info added to prompt
      ↓
10. If success: save_skill() updates skills.json
```

**LLM calls per task: 1–3** (1 initial plan + up to MAX_RETRIES corrections)
**LLM calls during navigation/execution: 0**

---

## 8. Design Decisions

### JSON instead of vector database
A vector database (Chroma, FAISS) would allow semantic similarity search: "put the bottle in bin" could retrieve "cleanup_skill" even without keyword overlap. But for a mid-sem demo with ~10 skills, keyword matching is sufficient and much simpler to understand and debug.

### BFS navigation instead of LLM-generated movements
If we asked the LLM to generate 30 `step(action)` calls, it would hallucinate positions and make mistakes constantly. BFS is deterministic and always finds the shortest path. This is the right tradeoff: **use LLM for high-level planning, use deterministic code for low-level control.**

### Pre-built skills instead of LLM-generated code
Voyager has GPT-4 write new JavaScript functions from scratch. This requires a very capable model and complex verification. For our demo, the skills are pre-built Python functions and the LLM only decides which ones to call. This is more reliable and still demonstrates the Voyager concept.

### homegrid-task (no language wrappers for agent)
The `"homegrid-task"` env ID wraps HomeGrid with language streaming. We don't use `obs["token"]` for planning — instead we read `info["symbolic_state"]` directly. The language wrapper is still loaded (it provides `env.task`), but we bypass the token stream for our agent logic.

### No fine-tuning
The LLM is never trained. Its parameters are frozen. The only "learning" is the external skill library growing. This is simpler, cheaper, faster, and still demonstrates the concept.

---

## 9. Known Limitations

1. **Skill retrieval is basic** — keyword matching can miss relevant skills if wording differs (e.g., "cleanup" vs "put in bin"). Embedding-based retrieval would fix this.

2. **Navigation can get stuck** — If an object is blocked by another object or in a corner with no BFS-reachable neighbor, navigation fails. This is handled by the retry loop but could be improved with more sophisticated path planning.

3. **Bin opening can fail** — If the wrong action was tried on a bin, it becomes "broken" for several steps. The agent will then fail the task and retry.

4. **Randomized task type** — The environment randomly samples task types on reset. `force_task_type()` resets up to 20 times to get the right type, but occasionally gives up.

5. **No persistent state across episodes** — Each `env.reset()` gives a new random layout. The skill library remembers which *types* of tasks have been solved, but not the specific layout.

6. **LLM can hallucinate object names** — If the LLM invents an object name not in the world state (e.g., "apple" instead of "fruit"), the skill will fail gracefully and trigger self-correction.

---

## 10. Future Extensions

| Extension | Difficulty | Impact |
|-----------|-----------|--------|
| Embedding-based skill retrieval (FAISS/Chroma) | Medium | Better skill reuse |
| LLM-generated Python skill code (true Voyager) | Hard | More flexible skills |
| Local LLM (Ollama + Llama 3) | Easy | No API costs |
| Better curriculum (LLM-generated task ordering) | Medium | Closer to Voyager |
| Skill verification (run skill in sandbox first) | Hard | More reliable execution |
| Multi-object task handling | Medium | More complex demos |
| Persistent agent memory across sessions | Easy | Better long-term learning |
| Web dashboard for skill library visualization | Medium | Better demo presentation |

For the final semester project, embedding-based retrieval and LLM-generated skills are the most impactful extensions.
