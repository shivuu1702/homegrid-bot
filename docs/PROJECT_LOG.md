# PROJECT_LOG.md — HomeGrid-Bot Development Log

---

## 1. Project Overview

**HomeGrid-Bot** is a Voyager-inspired autonomous agent for the HomeGrid environment, developed as an AI course project at IIIT Vadodara.

**Research objective:** Demonstrate that an LLM-based agent can interpret natural-language tasks, plan sequences of actions, execute them in a simulated home environment, self-correct on failure, and accumulate a reusable skill library — all without fine-tuning the LLM.

**Repository:** <https://github.com/shivuu1702/homegrid-bot>

---

## 2. Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| HomeGrid environment integration | ✅ Implemented | `homegrid-task` Gym env, symbolic state access |
| Natural-language task input | ✅ Implemented | 5 task types: find, get, open, cleanup, rearrange |
| LLM-based planning | ✅ Implemented | Provider-neutral (Gemini, OpenAI, OpenRouter, Groq) |
| BFS navigation | ✅ Implemented | Grid-based pathfinding with wall detection |
| Primitive skill execution | ✅ Implemented | 5 skills: find, pickup, open_bin, put_in_bin, move_to_room |
| Self-correction loop | ✅ Implemented | Up to MAX_RETRIES re-plans on failure |
| Skill library (JSON) | ✅ Implemented | 30 learned skills accumulated from runs |
| Task completion verification | ✅ Implemented | `task_is_complete()` checks symbolic state |
| Curriculum loop | ✅ Implemented | Progresses through 5 task types |
| Visual rendering (--visual) | ✅ Implemented | Matplotlib window with per-action logging |
| Demo mode (--demo) | ✅ Implemented | Single-task execution mode |
| Logging (file + console) | ✅ Implemented | `logs/run_*.log` and `logs/summary_*.json` |
| Unit tests | ✅ Implemented | Plan parsing, LLM failure handling, provider config |
| Multi-provider LLM abstraction | ✅ Implemented | Adapter pattern: Gemini native + OpenAI-compatible |
| Embedding-based skill retrieval | ❌ Not implemented | Currently uses keyword matching |
| LLM-generated code skills | ❌ Not implemented | Skills are pre-built Python functions |
| Formal benchmark / evaluation | ✅ Defined (not yet executed) | 10-task fixed benchmark in `docs/HOMEGRID_BENCHMARK.md` |

---

## 3. Architecture and Execution Flow

```
Natural-language task (e.g., "put the bottle in the recycling bin")
    │
    ▼
agent/main.py — Entry point
    │  Creates HomeGrid env, validates LLM config, runs curriculum or demo
    ▼
agent/agent.py — Core agent loop (run_task)
    │  1. Read task from env.task
    │  2. describe_state() → plain-English world description
    │  3. get_relevant_skills() → keyword search in skills.json
    │  4. plan_task() → build prompt → ask_llm() → parse plan
    │  5. execute_plan() → call primitive skills in sequence
    │  6. Success → save_skill() to skills.json
    │  7. Failure → plan_correction() → retry (up to MAX_RETRIES)
    ▼
agent/skills.py — Primitive skills + skill library
    │  Deterministic Python functions: find_object, pickup_object,
    │  open_bin, put_in_bin, move_to_room
    │  BFS navigation on symbolic grid with wall detection
    │  JSON-based skill library: load/save/retrieve
    ▼
agent/llm.py — LLM abstraction
    │  Provider-neutral interface: ask_llm(prompt) → str
    │  Adapters: GeminiAdapter (google-genai), OpenAICompatibleAdapter
    │  Error normalization, connection testing
    ▼
agent/config.py — Configuration
    │  Reads .env via dotenv; provides LLMConfig dataclass
    │  Constants: MAX_RETRIES=2, MAX_STEPS_PER_RUN=200, MAX_NAV_STEPS=150
    ▼
homegrid/ — External environment (do not modify)
    HomeGrid gridworld: 3 rooms, 4 pickable objects, 3 bins
    gym.make("homegrid-task") → MultitaskWrapper + LanguageWrapper
```

---

## 4. Environment, Dependencies, and Configuration

**Python:** 3.8+ (per homegrid's `pyproject.toml`)

**Dependencies** (`requirements.txt`):
- `openai>=1.0.0` — OpenAI-compatible LLM calls
- `google-genai>=1.0.0` — Gemini native SDK
- `python-dotenv>=1.0.0` — `.env` file loading
- `-e ./homegrid` — Local HomeGrid package (which brings `gym==0.26`, `numpy`, `matplotlib`, `tokenizers`, `sentencepiece`)

**Configuration** (`.env`, template in `.env.example`):
- `LLM_PROVIDER` — `gemini` | `openai` | `openrouter` | `groq`
- `LLM_MODEL` — Model identifier (e.g., `gemini-2.5-flash`, `gpt-4o-mini`)
- `LLM_API_KEY` — API key for selected provider
- `LLM_BASE_URL` — Required for OpenRouter and Groq
- `LLM_SKIP_TEST` — Set `1` to skip startup connection test
- `HOMEGRID_RENDER_DELAY` — Seconds between rendered actions (default 0.55)

---

## 5. Existing Behavior and Known Limitations

### Verified by log inspection (not re-tested in this audit)

- The agent has been run extensively between 2026-09-21 and 2026-10-28.
- 69 log files exist in `logs/` (53 run logs + 16 summary JSONs).
- The skill library has accumulated 30 distinct learned skills covering all 5 task types.
- The most recent run (2026-10-28) used `gemini-3.6-flash` as the LLM provider and succeeded on a cleanup task in 1 attempt.
- Curriculum runs show mixed success: simpler tasks (find, get, open) succeed reliably; cleanup and rearrange tasks sometimes fail.
- Summary logs show occasional `null` plans (LLM could not generate a valid plan).

### Known limitations (from `PROJECT_NOTES.md` and code inspection)

1. **Keyword-based skill retrieval** — Can miss semantically related skills.
2. **Navigation can fail** — Objects blocked by other objects or in corners with no BFS-reachable neighbor.
3. **Bin auto-close race** — Bins close on a timer; agent must reopen if it takes too long to navigate.
4. **Bin "broken" state** — Wrong open action breaks bin temporarily; handled with wait-and-retry.
5. **Randomized task sampling** — `force_task_type()` may not get the desired task type within max resets.
6. **100-step episode limit** — Imposed by HomeGrid; complex tasks can be truncated.
7. **LLM hallucination** — Can invent object names not present in state; handled by skill failure + self-correction.
8. **README.md project structure is outdated** — Lists directories (`environment/`, `prompts/`, `memory/`, `utils/`) that do not exist.
9. **README_TESTING.md references only OpenAI** — Does not reflect the current multi-provider LLM abstraction.

---

## 6. Git Preservation and Checkpoint Details

### Audit date: 2026-10-28

**State at audit start:**
- Branch: `main`
- HEAD: `35e242a` ("skills update", 2026-10-28 14:31:10 +0530)
- Working tree: clean (no staged, unstaged, or untracked changes)
- Stash: empty
- Remote: `origin/main` up to date with local `main`

**Backup created:**
- Branch: `backup/pre-audit-20261028`
- Points to commit: `35e242a`
- Pushed to remote: ✅ `origin/backup/pre-audit-20261028`

**Commit history (6 total commits):**

| Hash | Date | Message |
|------|------|---------|
| `35e242a` | 2026-10-28 | skills update |
| `9091e70` | 2026-09-23 | readme added |
| `738855f` | 2026-09-23 | skills updated |
| `44fc0e7` | 2026-09-23 | demo version added |
| `32a3fa7` | 2026-09-23 | Track homegrid files directly |
| `5a08d03` | 2026-09-23 | initial commit |

**Gitignored content (not in Git):**
- `.env` — Contains API keys (correctly excluded)
- `logs/` — 69 log files from agent runs (gitignored by choice)
- `.venv/` — Python virtual environment
- `__pycache__/` — Python bytecode cache

> [!WARNING]
> The `logs/` directory is gitignored. Its 69 files are **not preserved in Git** and exist only on this local machine. Consider committing selected logs or backing them up separately if they contain important experimental records.

---

## 7. Existing Logs and Experiment Artifact Index

### Log files (in `logs/`, gitignored)

**Run logs** (`run_*.log`): 53 files spanning 2026-09-21 to 2026-10-28.
- Earliest: `run_20260921_203541.log`
- Latest: `run_20261028_142844.log`
- Content: Timestamped agent execution traces (task, LLM prompts/responses, skill execution, navigation, results)

**Summary JSONs** (`summary_*.json`): 16 files.
- Content: Per-run structured results (task, success, attempts, plan)
- Cover curriculum and demo runs

### Key observations from logs

- The agent was actively developed and tested between 2026-09-21 and 2026-09-24, then had a gap until 2026-10-28.
- Most recent provider: `gemini` with `gemini-3.6-flash` model.
- The skills library grew organically across runs; `skills.json` currently has 30 entries.
- No formal success-rate statistics have been computed across all runs.

---

## 8. Development Log (Chronological)

### 2026-09-23 — Initial development (inferred from commits)

- Initial commit: base project structure, HomeGrid environment integration
- Added homegrid tracking, demo version, skills updates, README
- 6 commits total in this session establishing the working prototype

### 2026-09-21 to 2026-09-24 — Active experimentation (inferred from logs)

- Extensive agent runs covering all 5 task types
- Skill library grew from 0 to ~28 entries
- Multiple curriculum runs and individual task tests
- Various LLM providers/models tested (evidence of both OpenAI and Gemini usage)

### 2026-10-28 — Repository audit (this session)

**What was done:**
- Inspected all 6 source modules, 3 test files, 4 documentation files, skills.json, .env.example, .gitignore, and representative logs
- Verified Git state: clean working tree on `main` at `35e242a`, synced with `origin/main`
- Created backup branch `backup/pre-audit-20261028` at `35e242a` and pushed to remote
- Created `AGENTS.md` with mandatory instructions for future coding agents
- Created this file (`docs/PROJECT_LOG.md`) with project knowledge and history

**What was NOT done (intentionally out of scope):**
- ❌ No HomeGrid environment execution or Gemini testing
- ❌ No benchmark or evaluation run
- ❌ No dependency installation or code changes
- ❌ No refactoring or feature development

### 2026-10-28 — Fixed HomeGrid Benchmark creation

**What was done:**
- Inspected git state: clean working tree on `main` at `dedafe9`, 1 commit ahead of `origin/main`
- Reviewed all agent source files (`agent/agent.py`, `agent/skills.py`, `agent/config.py`, `agent/main.py`, `agent/llm.py`), project docs (`PROJECT_NOTES.md`, `PROJECT_LOG.md`, `AGENTS.md`), and the skill library (`skills/skills.json`, 30 entries)
- Examined HomeGrid environment internals (`homegrid/layout.py`, `homegrid/language_wrappers.py`) to confirm exact task formats, object/bin/room names, reward conditions, and action space
- Designed a fixed benchmark of **10 tasks** covering all 5 supported task types:
  - 2 × find (BM-01, BM-02)
  - 2 × get (BM-03, BM-04)
  - 2 × open (BM-05, BM-06)
  - 2 × cleanup (BM-07, BM-08)
  - 2 × rearrange (BM-09, BM-10)
- Each task documented with: unique ID, exact natural-language input, preconditions, expected plan, expected outcome, objective pass/fail criterion, difficulty level, known risks
- Tasks use exact strings from `MultitaskWrapper.sample_task()` format, runnable via `python agent/main.py --task "..."`
- Created `docs/HOMEGRID_BENCHMARK.md` (v1.0)
- Updated status table in `docs/PROJECT_LOG.md` to reflect benchmark existence
- Updated this chronological log

**Files created:**
- `docs/HOMEGRID_BENCHMARK.md` — 10-task fixed benchmark

**Files modified:**
- `docs/PROJECT_LOG.md` — Updated status, chronology, and next steps

**What was NOT done:**
- ❌ No benchmark tasks were executed (no experiments run)
- ❌ No agent code, prompts, skills, or configuration were modified
- ❌ No dependencies installed
- ❌ No statistical pass rates computed — benchmark is defined but unexecuted

**Unverified items (marked in benchmark):**
- Whether `find_object` works for bins (evidence: `find_compost_bin` has 3 successes in skills.json)
- Whether `move_to_room` handles objects already in the target room gracefully
- Whether `--task` matching reliably finds rearrange tasks (may require many resets)

**Design decisions:**
- 10 tasks chosen (2 per type) to balance coverage vs. execution cost
- Difficulty graded 🟢 Easy → 🟡 Medium → 🔴 Hard, matching single-skill to composite-skill tasks
- All 4 pickable objects and all 3 bins used at least once for broad coverage
- Tasks use environment-native strings to avoid format mismatch issues

---

## 9. Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| Backup branch, not tag | Branch can be pushed to remote; tag would also work but branch is more visible |
| Did not commit `logs/` | Logs are gitignored by project convention; changing this requires user decision |
| Did not modify `README.md` | Out of scope; README has stale project structure but is not blocking |
| Keyword skill retrieval kept | Sufficient for current scale (30 skills); embedding-based retrieval is future work |
| Pre-built skills, not LLM-generated code | Design decision documented in `PROJECT_NOTES.md`; appropriate for course project scope |

---

## 10. Known Issues and Unresolved Questions

1. **Logs are not version-controlled.** 69 experiment logs exist only locally. If the machine is lost, this experiment history is gone. Consider: selectively committing important summaries, or backing up `logs/` elsewhere.

2. **README.md project structure is inaccurate.** Lists `environment/`, `prompts/`, `memory/`, `utils/` directories that do not exist. Should be updated to reflect actual structure.

3. **README_TESTING.md is partially outdated.** References only OpenAI for LLM setup. The codebase now supports Gemini, OpenRouter, and Groq. The `.env.example` is the accurate reference.

4. **No systematic evaluation executed.** A fixed 10-task benchmark has been defined in `docs/HOMEGRID_BENCHMARK.md`, but no runs have been performed against it. Aggregate success rates remain unknown.

5. **`PROJECT_NOTES.md` references "GPT-4o-mini"** in several places but the active provider is now Gemini. The notes are still architecturally accurate but provider-specific details are stale.

6. **`find_bottle` skill is absent from skills.json** despite being a basic task. This may indicate early runs used a different skills.json that was later reset, or that the "find" tasks happened to sample non-bottle objects.

---

## 11. Next Steps

1. **Run the fixed benchmark** — Execute all 10 tasks in `docs/HOMEGRID_BENCHMARK.md` and record pass/fail results, attempt counts, and failure modes. This is the immediate next step.

2. **Automate benchmark execution** — Create a script that runs each benchmark task N times (e.g., N=5) and computes aggregate pass rates per task type.

3. **Back up or selectively commit experiment logs** — The 69 log files in `logs/` represent the only record of past experiments. Consider committing the `summary_*.json` files at minimum.

4. **Update README.md project structure** — Align the documented structure with the actual file layout.

5. **Update README_TESTING.md** — Reflect multi-provider support and current `.env` conventions.
