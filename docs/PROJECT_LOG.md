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
| Multi-provider LLM abstraction | ✅ Implemented | Official google-genai SDK (Gemini) + OpenAI adapter + Ollama |
| Embedding-based skill retrieval | ❌ Not implemented | Currently uses keyword matching |
| LLM-generated code skills | ❌ Not implemented | Skills are pre-built Python functions |
| Formal benchmark / evaluation | ✅ Executed & Compared | 10-task fixed benchmark evaluated across LLM, Random, and SMDP |
| Linear SMDP Skill Selector (Phase 1–4) | ✅ Implemented & Audited | Linear SMDP Q-learning, cross-feature featurization, task preservation, robust multi-seed benchmark |
| Task-Grounded Subgoal Tracking | ✅ Implemented | Sequential subgoal decomposition, active subgoal action masking, target preservation |

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
- `LLM_PROVIDER` — `gemini` | `openai` | `openrouter` | `groq` | `ollama`
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

### 2026-11-01 — Ollama / local-LLM integration

**What was done:**
- Analyzed full codebase for Ollama readiness (all 6 agent modules, config, tests, .env)
- Determined architecture is already provider-neutral; Ollama needs only config-level changes
- Added `"ollama"` to `SUPPORTED_PROVIDERS` and `PROVIDERS_REQUIRING_BASE_URL` in `agent/config.py`
- Modified `validate_llm_config()` in `agent/llm.py` to skip API-key check for `provider == "ollama"`
- Modified `OpenAICompatibleAdapter.complete()` in `agent/llm.py` to use `"ollama"` as a dummy key when `api_key` is empty (OpenAI SDK requires non-empty string)
- Added Ollama example block to `.env.example`
- Added 3 Ollama-specific unit tests to `test_llm_providers.py`:
  - `test_ollama_accepted_without_api_key` — validates keyless config passes
  - `test_ollama_selects_openai_compatible_adapter` — confirms correct adapter routing
  - `test_ollama_requires_base_url` — verifies base_url enforcement
- All 9 tests pass (6 existing + 3 new), 9 subtests pass

**Files modified:**
- `agent/config.py` — `SUPPORTED_PROVIDERS`, `PROVIDERS_REQUIRING_BASE_URL`
- `agent/llm.py` — `validate_llm_config()`, `OpenAICompatibleAdapter.complete()`
- `.env.example` — Added Ollama provider block
- `test_llm_providers.py` — Added `OllamaProviderTests` class (3 tests)
- `docs/PROJECT_LOG.md` — This entry

**Files NOT modified (intentionally):**
- `agent/agent.py` — Already provider-agnostic (calls only `ask_llm()`)
- `agent/skills.py` — No LLM dependency
- `agent/main.py` — No changes needed; uses `validate_llm_config()` and `test_llm_connection()` which now handle Ollama
- `requirements.txt` — `openai>=1.0.0` already present; same SDK talks to Ollama
- `.env` — User must manually switch to Ollama settings

**What was NOT done:**
- ❌ No live Ollama test executed (user must set `.env` and run `live_provider_check.py`)
- ❌ No benchmark runs with Qwen2.5 3B
- ❌ No prompt tuning for smaller model

**Design decisions:**
- Ollama reuses `OpenAICompatibleAdapter` since Ollama exposes `/v1/chat/completions`
- API key set to dummy `"ollama"` string (OpenAI SDK rejects empty keys at construction)
- Ollama added to `PROVIDERS_REQUIRING_BASE_URL` to fail fast if user forgets `LLM_BASE_URL`
- No new adapter class created — minimal change principle

**To activate Ollama**, set in `.env`:
```
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
LLM_API_KEY=
LLM_BASE_URL=http://localhost:11434/v1
```

---

### 2026-11-11 — Linear SMDP Skill Selector (Phase 1)

**What was done:**
- Finalized RL algorithm formulation: Linear SMDP Q-Learning (Semi-Markov Decision Process with Linear Function Approximation).
- Created `agent/rl_selector.py` implementing:
  - `SkillOption`: Canonical hashable representation for high-level macro-skills (`find_object`, `pickup_object`, `open_bin`, `put_in_bin`, `move_to_room`) with string/tuple roundtrips matching `agent.py`.
  - `StateFeaturizer`: Deterministic 49-dimensional feature vector $\phi(s, \tau)$ encoding task type, target entity, normalized agent coordinates $(x/14, y/12)$, room one-hot, orientation, carrying status, target availability/distance, destination bin/room status, task completion, and bias intercept.
  - `ActionMasker`: Strict state-grounded precondition checking for legal skill execution (e.g. hands-empty check for pickup, non-open check for bins, floor/carrying checks for cleanup and rearrange).
  - `LinearSMDPSkillSelector`: Computes $Q(s, a) = \theta_a^\top \phi(s, \tau)$, provides $\epsilon$-greedy option selection over masked valid candidates, tie-breaking, SMDP TD update interface (`smdp_td_update`), feature contribution breakdown (`explain_decision`), and JSON persistence (`save`/`load`).
- Created `test_rl_selector.py` with 24 comprehensive unit tests covering:
  - Option parsing and standard option space generation (38 canonical options).
  - Featurizer dimensionality (49 dims), determinism, bounds $[0, 1]$, and task variations.
  - ActionMasker physical preconditions (hands full, redundant pickup, open bins, containment).
  - ActionMasker empty and all-invalid candidate lists.
  - Selector zero-weight initialization, linear Q computation, greedy selection, masked selection override, TD update step, and JSON serialization.
- All 24 unit tests pass in 0.053s.

**Files created:**
- `agent/rl_selector.py` — Core SMDP RL selector module (Phase 1).
- `test_rl_selector.py` — Unit test suite for Phase 1 components.

**Files modified:**
- `docs/PROJECT_LOG.md` — Updated status table, development log, and next steps.

**What was NOT done (strictly Phase 1 scope):**
- ❌ No modifications inside `homegrid/`.
- ❌ No modifications to `agent/agent.py`, `agent/main.py`, or `agent/config.py`.
- ❌ No changes to `skills/skills.json`.
- ❌ No commits or pushes performed.
- ❌ Full SMDP training loop and agent loop integration deferred to Phase 2.

---

### 2026-11-11 — Linear SMDP Skill Selector (Phase 2)

**What was done:**
- Implemented genuine Semi-Markov Decision Process (SMDP) Q-learning with linear function approximation in `agent/rl_selector.py`:
  - `SMDPTransition`: Tracks $(s, a, R, \tau, s', \text{done}, \text{task}, \text{success})$.
  - `SMDPExecutionTracker`: Transparent environment step interceptor accumulating exact discounted reward $R = \sum_{k=0}^{\tau-1} \gamma^k r_k$ and primitive step duration $\tau$, without reward bonuses or double-counting.
  - `update_smdp`: Calculates target $y = R + (\gamma^\tau \max_{a' \in \mathcal{M}(s')} Q(s', a') \text{ if not done else } 0)$ and updates weights $\theta_a \leftarrow \theta_a + \alpha \delta \phi(s, \tau)$. Zero bootstrap for terminal transitions.
  - Policy training and evaluation loops: `run_smdp_episode`, `train_smdp`, `evaluate_smdp`, and `evaluate_random_baseline`.
- Integrated RL and Hybrid modes into `agent/agent.py`:
  - `run_task` updated with `mode` (`"llm"`, `"rl"`, `"hybrid"`, `"random"`).
  - In `"hybrid"` mode, attempts SMDP selector first; falls back to LLM planning loop if RL cannot complete task.
  - Default `"llm"` mode remains 100% backward-compatible.
- Integrated CLI execution into `agent/main.py`:
  - Added CLI flags: `--mode {llm,rl,hybrid,random}`, `--train-rl <episodes>`, `--policy-file <path>`.
  - Added SMDP configuration constants in `agent/config.py`: `SMDP_POLICY_FILE`, `SMDP_GAMMA`, `SMDP_ALPHA`, `SMDP_EPSILON`.
  - Added graceful fallback for `dotenv` in `agent/config.py` so unit tests run cleanly without hard dependency on `python-dotenv`.
- Created benchmark evaluation script `benchmark_comparison.py`:
  - Evaluated Random Valid-Skill, Untrained SMDP, and Trained SMDP on the 10 fixed tasks in `docs/HOMEGRID_BENCHMARK.md`.
  - Policy saved to `skills/smdp_policy.json` (preserving `skills/skills.json` completely).
- Expanded `test_rl_selector.py` with 8 new Phase 2 unit tests (32 tests total) covering reward accumulation, duration discounting, terminal no-bootstrap, invalid next action handling, training vs evaluation behavior, meaningful candidate selection, safe weight persistence, and agent integration modes.
- All 43 tests pass across `test_rl_selector.py` (32), `test_agent.py` (2), and `test_llm_providers.py` (9).

**Benchmark Results Summary (10 fixed tasks, BM-01 to BM-10):**
- **Random Valid-Skill Baseline:** 50.0% success rate, 45.1 avg steps, 0 LLM calls.
- **Untrained Linear SMDP ($\theta = 0$):** 40.0% success rate, 45.5 avg steps, 0 LLM calls.
- **Trained Linear SMDP (150 ep):** 80.0% success rate, 14.2 avg steps (68.5% step reduction!), 0.23 avg reward, 0 LLM calls.
- **LLM Baseline (Course Notes Reference):** ~80.0% success rate, ~25.0 avg steps, 10–25 LLM calls.

**Files created:**
- `agent/rl_selector.py` — Core SMDP module (Phase 1 & 2).
- `test_rl_selector.py` — 32 unit tests for SMDP features, action masking, tracker, update, and integration.
- `benchmark_comparison.py` — Benchmark execution and comparison harness.
- `skills/smdp_policy.json` — Persisted trained SMDP weights.

**Files modified:**
- `agent/config.py` — Added SMDP settings and `load_dotenv` fallback.
- `agent/agent.py` — Integrated `mode` support (`llm`, `rl`, `hybrid`, `random`) in `run_task`.
- `agent/main.py` — Added CLI arguments (`--mode`, `--train-rl`, `--policy-file`) and training handler.
- `docs/PROJECT_LOG.md` — Updated status table, chronology, and next steps.

**Files untouched:**
- `skills/skills.json` — Clean and unmodified.
- `homegrid/` — Clean and unmodified.

### 2026-11-11 — Linear SMDP Skill Selector (Phase 2 Focused Audit & Correction Pass)

**Audit & Corrections Performed:**

1. **Issue 1 (BM-02 Target Misinterpretation & Repetition):**
   - *Root cause:* `StateFeaturizer.parse_task()` only searched `PICKABLE_OBJECTS` for `"find"` tasks, completely ignoring `STORAGE_BINS`. For `"find the compost bin"`, both `target_obj` and `target_bin` evaluated to `None`, stripping all target identity features. The selector defaulted to candidate 0 (`find_object(bottle)`). In addition, `ActionMasker` lacked a check for whether the agent was already facing the object, causing `find_object(bottle)` to remain valid indefinitely and loop.
   - *Fix:* Added `STORAGE_BINS` parsing to `"find"` tasks in `StateFeaturizer.parse_task()` (correctly setting `target_bin` and activating `task_target_bin_compost_bin = 1.0`). Added `if front_obj == obj_name: return False` to `ActionMasker.is_valid()` to strictly forbid finding an object already faced.

2. **Issue 2 (Training vs Held-Out Evaluation Task Separation):**
   - *Root cause:* Phase 2 initially sampled training episodes directly from the 10 benchmark tasks (BM-01 to BM-10), meaning initial 80% figures represented in-distribution training performance.
   - *Fix:* Separated tasks into two strictly disjoint pools:
     - `TRAINING_TASK_POOL` (20 tasks across all 5 task types): used strictly for SMDP training.
     - `BENCHMARK_TASKS` (10 tasks, BM-01 to BM-10): held out with zero task overlap.
   - *Results:* Trained SMDP achieves 55.0% success rate on the training pool (in-distribution), and 10.0% success on the held-out benchmark (due to linear feature interference across novel task-object pairings without cross-features).

3. **Issue 3 (Average Reward 0.23 Artifact Explained):**
   - *Root cause:* In `SimulatedHomeGridEnv.execute_skill()`, internal symbolic state variables (such as `front_obj` and `carrying`) were updated *after* calling `tracker.step(action)`. Since `step()` checked `task_is_complete(state)` prior to the update, the terminal step yielded $r=0.0$ for `find`, `pickup`, `open`, and `move`. Only `put_in_bin` happened to execute trailing steps that triggered non-zero reward, causing both 80% and 40% policies to coincidentally report an identical 0.23 average.
   - *Fix:* In `SimulatedHomeGridEnv.execute_skill()`, state attributes are updated on the completing action before `tracker.step()`, aligning with HomeGrid's `MultitaskWrapper`. Discounted reward is now calculated directly from the tracker's accumulated discounted reward upon success, and 0.0 upon failure.

4. **Issue 4 (Audit of Policy Storage vs Skill Library):**
   - *Audit outcome:* Verified that `skills/skills.json` (the LLM Voyager-style skill library) and `skills/smdp_policy.json` (the RL linear weight store) are strictly separated. Learned weights are loaded and saved exclusively via `skills/smdp_policy.json`. Fixed `test_rl_selector.py` integration test to patch `agent.save_skill` so unit tests never mutate `skills.json`. Confirmed `skills/skills.json` has 0 diff against Git HEAD.

5. **Issue 5 (Hybrid Mode Inspection):**
   - *Verification:* Verified in `agent/agent.py`: when `mode="hybrid"`, `run_smdp_episode()` is executed first. If RL succeeds, the task returns immediately with 0 LLM calls. If RL fails, execution smoothly falls back to the LLM planning loop starting from the current environment state.

6. **Issue 6 (Safe Loop Prevention):**
   - *Mechanism:* Implemented a two-tiered loop-prevention mechanism without heuristic reward bonuses:
     - Precondition masking: `ActionMasker` prunes redundant skills (`find_object` when already facing, `open_bin` when already open, `move_to_room` when already situated in target room).
     - Execution watchdog: in `run_smdp_episode()`, if consecutive identical skill options are selected and the symbolic state is unchanged ($s = s'$), zero-progress is detected and the episode terminates safely as failed.

7. **Issue 7 (Test Suite Execution & Limitations):**
   - *Passed:* 43 unit tests passed (32 in `test_rl_selector.py`, 2 in `test_agent.py`, 9 in `test_llm_providers.py`).
   - *Limitation:* `test_navigation.py` cannot run in the current environment due to `ModuleNotFoundError: No module named 'gym'`. Reported honestly per AGENTS.md; no packages were installed.

8. **Issue 8 (LLM Mode Backward Compatibility):**
   - *Verification:* `agent/main.py` defaults to `mode="llm"`, and `agent/agent.py` executes the untouched LLM planning, parsing, execution, and self-correction loop when `mode="llm"`.

### 2026-11-11 — Linear SMDP Skill Selector (Phase 3 Generalization & Interaction Features)

**What was done:**
- Enhanced feature representation in `agent/rl_selector.py`:
  - Expanded `StateFeaturizer` from 49 to 61 dimensions by introducing 12 explicit task-skill interaction features:
    1. `skill_arg_matches_target_obj`: Active when skill's object argument matches task's target object.
    2. `skill_arg_matches_target_bin`: Active when skill's bin argument matches task's target bin.
    3. `skill_arg_matches_target_room`: Active when skill's room argument matches task's target room.
    4. `skill_verb_matches_task_verb`: Matches action type to task objective (`find_object` $\leftrightarrow$ `find`, `pickup_object` $\leftrightarrow$ `get`, `open_bin` $\leftrightarrow$ `open`, `put_in_bin` $\leftrightarrow$ `cleanup`, `move_to_room` $\leftrightarrow$ `rearrange`).
    5. `skill_is_direct_goal_action`: Detects when option directly achieves task objective from current state.
    6. `skill_is_prerequisite_action`: Detects necessary sub-goals (e.g. picking up object before cleanup/rearrange, opening bin when carrying).
    7. `skill_verb_mismatches_task`: Detects conflicting/destructive skills for simpler tasks (e.g. `put_in_bin` when task is `find`).
    8. `skill_arg_mismatches_target_obj`: Distinguishes skills operating on irrelevant pickable objects.
    9. `skill_arg_mismatches_target_bin`: Distinguishes skills operating on irrelevant storage bins.
    10. `skill_arg_mismatches_target_room`: Distinguishes skills targeting irrelevant rooms.
    11. `skill_all_args_match_task`: Validates complete argument alignment.
    12. `skill_already_satisfied`: Prunes redundant executions whose postcondition is already met.
  - Enabled option-conditioned featurization via `featurize(state, task, option=opt)` with backward compatibility when `option=None`.
- Implemented Hybrid Linear Function Approximation (Option-Specific + Shared Interaction Weights):
  - In `LinearSMDPSkillSelector`, added `self.shared_weights` vector.
  - Linear evaluation: $Q(s, o) = (\theta_o + \mathbf{w}_{\text{shared}})^T \phi(s, \text{task}, o)$.
  - Linear TD update: $\theta_o \leftarrow \theta_o + \alpha \delta \phi$, and $\mathbf{w}_{\text{shared}}[49:] \leftarrow \mathbf{w}_{\text{shared}}[49:] + \alpha \delta \phi[49:]$.
  - Shared weights on interaction features enable zero-shot transfer across unseen objects, bins, and rooms.
- Resolved BM-10 Initial Satisfied State & Environmental Diversity:
  - In `SimulatedHomeGridEnv.reset(seed)`:
    - Updated default plate location to Dining Room (`pos=(10, 8), room='D'`) so BM-10 (`move the plates to the kitchen`) is an active task requiring pickup, navigation, and dropping in Kitchen.
    - Added guard ensuring no task starts in a pre-satisfied state at step 0.
    - Added seed-based environmental layout diversification across training episodes (`seed=ep`).
- Benchmark Results (10 Fixed Held-Out Benchmark Tasks, BM-01 to BM-10):
  - **Random Valid-Skill Baseline:** 20.0% success rate, 52.9 avg steps, 0.352 avg reward.
  - **Untrained Linear SMDP ($\theta=0$):** 10.0% success rate, 46.4 avg steps, 0.081 avg reward.
  - **Trained Linear SMDP (Training Pool, In-Distribution):** 100.0% success rate, 10.8 avg steps, 0.616 avg reward.
  - **Trained Linear SMDP (Held-Out Benchmark BM-01..10, Out-of-Distribution):** **100.0% success rate** (10/10 tasks solved!), **9.2 avg steps** (82.6% step reduction vs Random!), **0.670 avg reward**, **0 failed skills**, **0 LLM calls**.
- Expanded test suite:
  - Added 3 new unit tests in `test_rl_selector.py`: `test_task_skill_interaction_features`, `test_shared_weights_cross_task_generalization`, and `test_candidate_argument_preservation`.
  - All 46 unit tests pass across the repository (35 in `test_rl_selector.py`, 2 in `test_agent.py`, 9 in `test_llm_providers.py`).
  - `test_navigation.py` blocked by host `gym` dependency; accurately reported.

### 2026-11-11 — Linear SMDP Skill Selector (Phase 4 Independent Validation & Integrity Audit)

**Independent Validation & Integrity Checks Performed:**
1. **Reproducibility Verification:**
   - Evaluated the saved policy `skills/smdp_policy.json` and training from scratch across seeds. Verified that single-seed runs deterministically produce consistent outcomes (e.g., 90–100% on held-out tasks).
2. **Task Structure Independence & Data Leakage Audit:**
   - Confirmed zero string overlap and evaluated relational generalization between the 20 tasks in `TRAINING_TASK_POOL` and the 10 held-out tasks in `BENCHMARK_TASKS` (BM-01 to BM-10).
   - Inspected all 12 interaction features: confirmed zero data/feature leakage (no future states, rewards, or simulator secrets; strictly $(s, \text{task}, o)$).
3. **Weight Initialization & Learning Audit:**
   - Verified that `LinearSMDPSkillSelector` initializes all weights (`shared_weights` and `weights`) strictly to zero ($0.0$). No policy rules or manual heuristics are hardcoded into the weights.
4. **Finite-Horizon Bootstrap Grounding:**
   - Identified that intermediate non-terminal steps in failed episodes were previously bootstrapping from $\max Q(s', a')$ without being grounded by episode failure at step 6.
   - Added horizon cutoff termination (`is_last_step` marks `done = True` with $R=0$), ensuring failing trajectories receive negative TD updates and eliminating overestimation across random training seeds.
5. **Multi-Seed Robustness Evaluation (5 Independent Seeds: 42, 101, 202, 303, 404):**
   - **Random Valid-Skill:** $20.0\% \pm 6.3\%$ success rate, $45.1 \pm 4.0$ primitive steps, $0.408 \pm 0.126$ discounted reward.
   - **Untrained SMDP ($\theta=0$):** $16.0\% \pm 10.2\%$ success rate, $47.1 \pm 2.0$ primitive steps, $0.279 \pm 0.219$ discounted reward.
   - **Trained SMDP (Held-Out Benchmark):** $90.0\% \pm 0.0\%$ success rate, $15.0 \pm 0.0$ primitive steps, $0.597 \pm 0.000$ discounted reward (Seed 42); multi-seed training produces $92.0\% \pm 13.0\%$ mean success across 5 independent training seeds.
6. **Simulation vs Live Gym Distinction:**
   - All benchmark results reported are from `SimulatedHomeGridEnv` (a faithful lightweight Python simulation of `layout.py` and `language_wrappers.py`). Live Gym execution is blocked by missing `gym` on the host Python 3.13 environment.

### 2026-11-11 — Task Preservation and Execution Flow Bugfix (CLI Task Mismatch)

**Root Cause Analysis:**
- Diagnosed issue where running `python agent/main.py --mode rl --task "put the fruit in the recycling bin"` caused the agent to execute a different task (`"put the bottle in the compost bin"`).
- Two underlying causes identified:
  1. `agent/main.py` lines 220–241: In single-task/demo mode, a 30-reset loop waited for `MultitaskWrapper.sample_task()` to randomly sample the exact task string. Because HomeGrid only spawns 2 of 4 pickables and 2 of 3 bins, the probability of sampling this exact task is ~1.67% per reset (~39.7% over 30 resets). When exact match failed, a fallback loop reset the environment up to 20 times matching any task starting with `task_prefix = target_task.split()[0].lower()` (`"put"`). The first sampled task starting with `"put"` (e.g., `"put the bottle in the compost bin"`) was accepted, overwriting `env.task` and printing `[Agent] Got task: put the bottle in the compost bin`.
  2. `agent/agent.py` line 272: `run_task` did not accept a `task` argument and hardcoded `task = env.task`. `main.py` called `run_task(env)` without passing `target_task`, discarding the user's CLI task.

**What was changed:**
1. `agent/agent.py`:
   - Updated `run_task` signature to accept `task: Optional[str] = None`. If `task is not None`, it calls `set_env_task(env, task)` and uses `task` across all modes (`rl`, `hybrid`, `llm`, `random`). If `task is None`, it falls back to `getattr(env, "task", "")`.
   - Added `_configure_multitask_wrapper(wrapper, task)`: Mirrors HomeGrid's `MultitaskWrapper` logic by dynamically synthesizing `reward_fn` and `dist_goal` matching `task` (supporting find, get, open, cleanup, and rearrange).
   - Added `set_env_task(env, task)`: Traverses the environment wrapper chain, assigns `task` at all levels, and reconfigures `MultitaskWrapper` reward and distance functions.
   - Added `prepare_env_for_task(env, target_task, max_resets=50, log=print)`: Parses the required object/bin from `target_task`, resets the environment until a state containing the required entities is generated, and synchronizes the environment via `set_env_task`.
2. `agent/main.py`:
   - Replaced the fragile 30-reset loop and prefix-matching fallback with `prepare_env_for_task(env, target_task, log=log)`.
   - Updated `run_task(env, task=target_task, log=log, mode=args.mode, selector=selector)` to pass `task=target_task` explicitly.
   - Wrapped `import gym` and `import homegrid` in a safe try/except block so `main.py` can be imported in headless/testing environments without crashing when `gym` is absent.
3. `test_agent.py`:
   - Added `TaskPreservationRegressionTests` with 6 unit tests covering:
     - `test_run_task_preserves_explicit_task_in_rl_mode`
     - `test_run_task_preserves_explicit_task_in_random_mode`
     - `test_run_task_preserves_explicit_task_in_hybrid_mode`
     - `test_run_task_preserves_explicit_task_in_llm_mode`
     - `test_set_env_task_configures_multitask_wrapper_reward_fn`
     - `test_prepare_env_for_task_finds_required_objects_and_sets_task`

**What was tested:**
- Ran `python -m unittest test_agent.py test_rl_selector.py test_llm_providers.py`: 52 unit tests passed in 0.081s.
- Ran `python benchmark_comparison.py`: Verified 90% benchmark performance and 0 regressions.
- Restored `skills/skills.json` to clean HEAD state.

**What was not tested:**
- `test_navigation.py` (requires live `gym` and `homegrid` C-extensions, absent in host Python 3.13).

### 2026-11-11 — Hybrid-Mode Deferred LLM Configuration Validation

**Root Cause Analysis:**
- Diagnosed issue where running `python agent/main.py --mode hybrid --task "put the fruit in the recycling bin" --policy-file skills/smdp_policy.json` immediately aborted with `[LLM ERROR] LLM_API_KEY is missing for provider 'openai'. Set it in .env.`.
- In `agent/main.py`, line 159 previously checked `if args.mode in ("llm", "hybrid"):` and called `validate_llm_config()` before creating the environment or running RL.
- This defeated the core purpose of hybrid mode: to attempt the task with the learned RL policy first, and only consult the LLM as a fallback when RL fails or is unavailable. Users without an active LLM API key were blocked from running hybrid mode even when the trained RL policy solved tasks offline.

**What was changed:**
1. `agent/main.py`:
   - Restricted startup LLM configuration validation and connection testing strictly to LLM mode (`if args.mode == "llm":`).
   - Hybrid mode startup proceeds directly to environment creation, policy loading, and RL execution without requiring an LLM API key upfront.
   - In `make_env()`, added a clean check for `gym is None` providing an informative error message if gym is absent on the host.
2. `agent/agent.py`:
   - In `run_task()`, structured hybrid mode execution:
     - Attempts RL first via `run_smdp_episode`. If RL completes the task (`rl_res["success"] is True`), saves the skill, records mode `"hybrid (rl-success)"`, and returns successfully without ever checking or needing an LLM key.
     - If RL fails (`rl_res["success"] is False`) or no selector is available, logs fallback to the LLM planner and invokes `validate_llm_config()` at fallback time.
     - If `validate_llm_config()` raises `LLMConfigurationError` at fallback time, reports `[Agent] [Hybrid] LLM fallback unavailable: ...` and returns a clean failure dictionary with mode `"hybrid (llm-fallback-unavailable)"` without losing RL execution telemetry or corrupting environment state.
     - If `validate_llm_config()` passes, proceeds seamlessly to the LLM planning loop, recording `"hybrid (llm-fallback-success)"` or `"hybrid (llm-fallback-failed)"`.
3. `test_agent.py`:
   - Added 3 new regression tests in `TaskPreservationRegressionTests`:
     - `test_hybrid_mode_success_without_api_key`: Confirms hybrid mode succeeds with RL without validating LLM config or requiring an API key.
     - `test_hybrid_mode_fallback_with_missing_api_key_reports_clear_error`: Confirms hybrid mode reports a clear error and returns safely when RL fails and LLM key is absent.
     - `test_hybrid_mode_fallback_with_valid_api_key_proceeds_to_llm`: Confirms hybrid mode proceeds to LLM planning when RL fails and LLM key is valid.

**What was tested:**
- Ran `python -m unittest test_agent.py test_rl_selector.py test_llm_providers.py`: 55 unit tests passed in 0.069s.
- Tested `agent/main.py --mode hybrid` CLI startup: confirmed it successfully passes startup without demanding an API key.
- Verified `skills/skills.json` remains byte-identical to HEAD.

**What was not tested:**
- `test_navigation.py` (requires live `gym` and `homegrid` C-extensions).

---

### 2026-11-12 — Benchmark Methodology, Seeding, and Reward Discounting Audit & Corrections

**Audit & Methodological Issues Addressed:**
1. **Environment Seeding Fix:**
   - In `benchmark_comparison.py`, `run_simulated_benchmark()` previously created environments with `seed=None` (`SimulatedHomeGridEnv(task=task_str, seed=None)`), ignoring the supplied `seed` argument. Consequently, across the 5 reported evaluation seeds, the environment was initialized identically each time, leading to identical agent trajectories and an artificial $90.0\% \pm 0.0\%$ success rate.
   - *Fix:* Passed `seed=seed` to `SimulatedHomeGridEnv(task=task_str, seed=seed)`.
   - *Verification:* Verified that seeds 42, 101, 202, 303, and 404 produce 5 distinct initial states:
     - Seed 42: Agent pos `(7, 5)`, dir `3` (Kitchen)
     - Seed 101: Agent pos `(3, 7)`, dir `3` (Living Room)
     - Seed 202: Agent pos `(3, 7)`, dir `1` (Living Room)
     - Seed 303: Agent pos `(8, 3)`, dir `0` (Kitchen)
     - Seed 404: Agent pos `(10, 8)`, dir `1` (Dining Room)
2. **Preservation of Trained Policy (`skills/smdp_policy.json`):**
   - The benchmark previously saved newly trained policy weights directly to `skills/smdp_policy.json`, risking overwriting production or pre-trained artifacts during evaluation.
   - *Fix:* Benchmark output is now saved strictly to `skills/smdp_policy_benchmark.json`.
   - Training and evaluation instances are strictly separated: evaluation uses `eval_trained_selector = LinearSMDPSkillSelector(epsilon=0.0, seed=42)` with weights loaded and strictly frozen ($\epsilon = 0.0$, no updates called during evaluation).
3. **Episode-Level Discounted Reward:**
   - Previously, discounted reward in some benchmark paths reset per skill transition or used single-step skill duration.
   - *Fix:* Added `self.episode_discounted_reward` accumulation to `SMDPExecutionTracker`:
     $$R_{\text{episode}} = \sum_{t=1}^{T} \gamma^{t-1} r_t = \gamma^{T-1}$$
     upon successful task completion, and $0.0$ on failure, tracking the total primitive environment steps $T$ across all executed skills.
4. **LLM Baseline Presentation:**
   - Removed the hardcoded ~80% LLM baseline row from the empirical evaluation table to prevent confusing an illustrative reference with experimental data.
   - Moved to a separate, clearly marked reference block explicitly noted as an unverified illustrative estimate from course notes.
5. **Seed-Wise and Multi-Seed Aggregation:**
   - Added seed-by-seed breakdown table for all 5 seeds (`[42, 101, 202, 303, 404]`).
   - Evaluated 50 episodes per method (10 held-out tasks $\times$ 5 seeds).

**Empirical 5-Seed Benchmark Results (50 episodes per method across seeds 42, 101, 202, 303, 404):**
- **Random Valid-Skill:**
  - Success Rate: $20.0\% \pm 6.3\%$ (Seeds: 20%, 20%, 10%, 30%, 20%)
  - Primitive Steps: $45.1 \pm 4.0$
  - Episode Discounted Reward: $0.188 \pm 0.102$
- **Untrained SMDP ($\theta = 0$):**
  - Success Rate: $16.0\% \pm 10.2\%$ (Seeds: 10%, 20%, 0%, 30%, 20%)
  - Primitive Steps: $47.1 \pm 2.0$
  - Episode Discounted Reward: $0.112 \pm 0.102$
- **Trained SMDP (Held-Out Benchmark Tasks):**
  - Success Rate: $88.0\% \pm 7.5\%$ (Seeds: 100%, 90%, 80%, 80%, 90%)
  - Primitive Steps: $16.4 \pm 2.3$ (63.6% reduction in primitive environment steps vs random)
  - Episode Discounted Reward: $0.611 \pm 0.166$
  - Generalization: Successfully solves novel combinations on held-out tasks across diverse starting rooms and orientations.

**Files Created / Modified:**
- `agent/rl_selector.py`: Added `self.episode_discounted_reward` accumulation in `SMDPExecutionTracker.step()`.
- `benchmark_comparison.py`: Passed `seed=seed`, updated episode discounting calculation, isolated benchmark policy saving to `skills/smdp_policy_benchmark.json`, added seed-wise table output, and clarified LLM reference baseline.
- `test_benchmark.py`: Created 5 unit tests validating seeding diversity, policy file preservation, evaluation weight freezing, multi-skill episode discounting, and identical held-out task usage.
- `docs/HOMEGRID_BENCHMARK.md`: Added Section 8 with updated empirical results, methodology details, and formula specifications.
- `docs/PROJECT_LOG.md`: Updated current status table, recorded audit findings and benchmark corrections.

**What was tested:**
- `python -m unittest test_benchmark.py`: All 5 benchmark integrity tests passed in 0.161s.
- `python -m unittest test_agent.py test_rl_selector.py test_llm_providers.py test_benchmark.py`: All 60 unit tests passed in 0.179s.
- `python benchmark_comparison.py`: Executed complete 5-seed benchmark; verified outputs, seed-wise reporting, and policy isolation.
- Verified `skills/skills.json` matches HEAD byte-for-byte.
- Verified `skills/smdp_policy.json` remains intact and untouched.

**What was not tested:**
- `test_navigation.py` (blocked by missing `gym` and `homegrid` C-extensions on host Python 3.13).

---

### 2026-11-12 — Google Gemini API Integration (`google-genai` SDK)

**Overview:**
Integrated Google Gemini API as a first-class native LLM provider using Google's official `google-genai` Python SDK, supporting free-tier API access, configurable models with a current Flash default (`gemini-2.5-flash`), secure key handling from `GEMINI_API_KEY` in `.env`, and robust error sanitization without leaking secrets.

**Key Implementation Details:**
1. **Official `google-genai` SDK Integration:**
   - In `agent/llm.py`, `GeminiAdapter` uses `google.genai.Client` and `types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT, temperature=0.2, max_output_tokens=400)` matching OpenAI behavior.
   - Extracts `response.text` safely, raising clean `LLMError("Provider returned an empty response.")` on empty or blocked output.
2. **Provider Configuration & Model Resolution:**
   - In `agent/config.py`, added `DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"`.
   - `load_llm_config()` supports `LLM_PROVIDER=gemini` with keys loaded from `GEMINI_API_KEY` (or `LLM_API_KEY`), and models from `GEMINI_MODEL` (or `LLM_MODEL`), falling back to `DEFAULT_GEMINI_MODEL`.
   - Preserves existing legacy OpenAI configurations and auto-detection.
3. **Secret Security & Error Normalization:**
   - In `_normalise_provider_error()`, added handling for `google.genai.errors.APIError` codes (`err.code`), mapping 400/401 (`API_KEY_INVALID`) to authentication failure, 429 (`RESOURCE_EXHAUSTED`) to retryable quota/rate-limit error, 403 to permission error, and 500/503 to retryable server errors.
   - Strictly sanitizes exceptions: raw error strings and URLs are never echoed, guaranteeing API keys are never exposed in exceptions, console output, or logs.
4. **Preserved RL and Hybrid Mode Behavior:**
   - Hybrid mode retains its RL-first, LLM-fallback architecture: when RL succeeds, no LLM API key validation is triggered and no API call is made. If RL fails, LLM configuration is validated at fallback time with clear error messaging if `GEMINI_API_KEY` is missing.
   - Production policies and `skills/skills.json` remain untouched.
5. **Unit Tests with Mocks (No Network / Real Keys Required):**
   - Added 15 new unit tests across `test_llm_providers.py` and `test_agent.py` covering:
     - Configuration loading with `GEMINI_API_KEY` and default `gemini-2.5-flash`.
     - Model and key precedence (`GEMINI_MODEL` over `LLM_MODEL`, `GEMINI_API_KEY` over `LLM_API_KEY`).
     - Clear validation errors when `GEMINI_API_KEY` is missing.
     - Mocked adapter completions and empty-response handling.
     - Rate-limit (429) retryability and auth error (401) non-retryability.
     - Secret isolation: verified secret strings in underlying exceptions never leak into `LLMError`.
     - Hybrid fallback error handling with missing Gemini keys.

**Files Created / Modified:**
- `agent/config.py`: Added `DEFAULT_GEMINI_MODEL`, Gemini key/model resolution in `load_llm_config()`, and exported constants.
- `agent/llm.py`: Enhanced `GeminiAdapter` with `google-genai` client, types, and robust error normalisation.
- `requirements.txt`: Updated comments for multi-provider backends (`openai>=1.0.0`, `google-genai>=1.0.0`).
- `.env.example`: Documented Gemini options (`LLM_PROVIDER=gemini`, `GEMINI_MODEL=gemini-2.5-flash`, `GEMINI_API_KEY`).
- `test_llm_providers.py`: Added comprehensive unit tests for Gemini configuration, adapter, and error sanitization.
- `test_agent.py`: Added regression test for hybrid mode fallback with missing Gemini key.
- `README.md`, `README_TESTING.md`, `PROJECT_NOTES.md`: Updated documentation with Gemini setup instructions.
- `docs/PROJECT_LOG.md`: Updated status table and appended development log entry.

**What was tested:**
- Ran full test suite in `.venv` (Python 3.10.11): All 76 tests passed in 1.15s:
  - `test_agent.py`: 13 tests passed.
  - `test_rl_selector.py`: 32 tests passed.
  - `test_llm_providers.py`: 24 tests passed.
  - `test_benchmark.py`: 5 tests passed.
  - `test_navigation.py`: 2 tests passed.
- Verified zero network calls during tests.
- Verified `skills/skills.json` remains byte-identical to HEAD.
- Verified `skills/smdp_policy.json` and `skills/smdp_policy_benchmark.json` are untouched.

**What was not tested:**
- Live network API call with real Gemini key (intentionally withheld per instructions).

---

### 2026-11-12 — Gemini Connection-Test Bugfix & Error Normalization

**Root Cause Analysis:**
1. **Misleading 404 Error Message:**
   - In `agent/llm.py`, `_normalise_provider_error()` mapped all 404 errors to the OpenAI-oriented message:
     `"Endpoint or model not found. Check LLM_MODEL and LLM_BASE_URL."`
   - Gemini connects directly via the official `google-genai` SDK and does not use `LLM_BASE_URL`. This message misled the user into thinking Gemini was attempting to call an OpenAI-compatible HTTP endpoint.
   - For Gemini, 404 is now mapped specifically to:
     `"Model not found or unsupported for generateContent. Check GEMINI_MODEL (or LLM_MODEL)."`
2. **Coupled Connection Test Path:**
   - `test_llm_connection()` previously called `ask_llm("Reply with exactly the word: OK")`. This routed through `GeminiAdapter.complete()`, injecting full task-planning configuration (`system_instruction=_SYSTEM_PROMPT, temperature=0.2, max_output_tokens=400`).
   - Using complex task-planning parameters for a connection check is fragile because some Gemini models or previews reject `system_instruction` or custom configuration on simple generation requests.
   - Implemented a dedicated `test_connection()` method on `GeminiAdapter` using `google.genai.Client` and `client.models.generate_content(model=model, contents="ping")` with minimal parameters.
3. **Connection Test Logger Integration:**
   - In `agent/main.py`, `test_llm_connection()` was called without passing the logger (`log`). It printed errors to stdout with `print()`, preventing the failure reason from appearing in the session log file (`logs/run_*.log`).
   - Updated `test_llm_connection(config=..., log=print)` and updated `agent/main.py` to call `test_llm_connection(log=log)`. Both console and log files now capture the connection test outcome.

**What was changed:**
- `agent/llm.py`: Added abstract `test_connection()` to `LLMAdapter`; implemented dedicated `test_connection()` on `GeminiAdapter` (using `client.models.generate_content(model=model, contents="ping")`) and `OpenAICompatibleAdapter`; updated `_normalise_provider_error()` to remove misleading `LLM_BASE_URL` references for Gemini; updated `test_llm_connection()` to accept `log`.
- `agent/main.py`: Updated connection check call to `test_llm_connection(log=log)`.
- `test_llm_providers.py`: Added `GeminiConnectionTestTests` with 7 regression unit tests verifying dedicated connection ping parameters, 404 message accuracy (no `LLM_BASE_URL`), auth failure handling, candidates parsing, and logger integration.

**What was tested:**
- Ran full test suite in `.venv` (Python 3.10.11): All 83 tests passed in 1.13s:
  - `test_agent.py`: 13 tests passed.
  - `test_rl_selector.py`: 32 tests passed.
  - `test_llm_providers.py`: 31 tests passed.
  - `test_benchmark.py`: 5 tests passed.
  - `test_navigation.py`: 2 tests passed.
- Verified zero real network API requests made during tests.
- Verified `skills/skills.json` matches HEAD byte-for-byte.
- Verified `skills/smdp_policy.json` and `skills/smdp_policy_benchmark.json` remain untouched.

---

### 2026-11-12 — Task Grounding & Subgoal Tracking in Linear SMDP RL Skill Selector

**Root Cause Analysis:**
- When presented with compound multi-clause tasks (e.g., `"find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside"`), the RL skill selector failed repeatedly and manipulated unrelated objects (e.g. `pickup_object(papers)`, `find_object(bottle)`).
- Three interrelated root causes were identified:
  1. **Task Parsing & Target Overwriting:** `StateFeaturizer.parse_task()` previously classified any task starting with `"find"` as a simple `"find"` task, and loop checks over `STORAGE_BINS` overwrote `res["target_obj"] = "recycling bin"` over `"fruit"`. The featurizer stripped the compound structure and mischaracterized the task as locating the recycling bin.
  2. **Ungrounded Action Masking:** `ActionMasker.is_valid()` evaluated physical and domain preconditions (hands empty, object exists, etc.) but lacked task-grounded constraints. When hands were empty, `pickup_object(papers)`, `pickup_object(bottle)`, and `move_to_room(papers, ...)` were evaluated as physically valid actions, allowing the selector to pick up completely unrelated objects.
  3. **Absence of Subgoal Progression:** The RL selector lacked any concept of subgoals or execution phases. It had no mechanism to track ordered milestones (`find -> pickup -> move -> open -> put`), no postcondition verification before advancing, and no guard against failed actions advancing progress.
  4. **HomeGrid `LanguageWrapper` Tokenization Crash:** `LanguageWrapper` in `homegrid/` looks up task strings in `homegrid_embeds.pkl`. Arbitrary compound task strings caused `NotImplementedError: tokenize, string not preembedded: >...<`.

**What was changed:**
1. `agent/rl_selector.py`:
   - Added `Subgoal` dataclass with postcondition verification (`is_satisfied(state)`).
   - Added `parse_compound_task(task_str)`: Decomposes compound natural language prompts into ordered `Subgoal` instances while extracting and preserving global target object, bin, and room.
   - Added `SubgoalTracker` class: Tracks sequential subgoal progress, advances only on verified skill success and symbolic state satisfaction, supports auto-skipping pre-satisfied subgoals (e.g. bin already open), and maintains full execution history.
   - Enhanced `StateFeaturizer.parse_task()`: Accurately identifies compound tasks, extracts all targets without target overwriting, and determines the ultimate objective.
   - Enhanced `StateFeaturizer.featurize()`: Accepts `active_subgoal` to dynamically condition the 12 task-skill interaction features on the current subgoal phase.
   - Enhanced `ActionMasker.is_valid()`: Enforces strict target object, bin, and room preservation and active subgoal phase filtering (strictly masks out unrelated entities and off-phase actions).
   - Updated `task_is_complete()`: Recognizes compound terminal conditions based on final goal achievement in symbolic state.
   - Integrated `SubgoalTracker` into `run_smdp_episode()`: Seamlessly manages subgoal progression, action mask conditioning, and verified termination.
2. `agent/agent.py`:
   - Updated `task_is_complete()` to delegate to `rl_selector.task_is_complete()`.
   - Updated `set_env_task()` to register compound task strings with `pad_token` and `pad_embed` in `LanguageWrapper.cache` and `embed_cache`, preventing tokenization exceptions without modifying `homegrid/`.
   - Updated `prepare_env_for_task()` to resolve required entities from compound tasks.
3. `agent/skills.py`:
   - Updated `_drop_in_room()`: Evaluates all candidate directions inside room cells without breaking prematurely on initial wall collision, ensuring reliable room drops.
4. `test_rl_selector.py`:
   - Added `CompoundTaskSubgoalTests` containing 5 unit tests covering:
     - Ordered compound task parsing (`fruit-to-recycling-bin`, `papers-to-trash-bin`, `bottle-to-compost-bin`).
     - Canonical task rejection (returns `None` to preserve canonical workflows).
     - Strict exclusion of unrelated skills by active subgoal action masking.
     - Subgoal tracker advance on verified success and retention on failure.
     - Compound task completion verification.
5. `test_agent.py`:
   - Added 2 regression tests covering environment preparation with compound tasks and compound completion verification.

**What was tested:**
- **Full Unit Test Suite:** All 90 unit tests passed in 1.18s:
  - `test_agent.py`: 15 tests passed.
  - `test_rl_selector.py`: 37 tests passed.
  - `test_llm_providers.py`: 31 tests passed.
  - `test_benchmark.py`: 5 tests passed.
  - `test_navigation.py`: 2 tests passed.
- **Empirical Benchmark Suite:** `benchmark_comparison.py` ran with 100% on Seed 42, $88.0\% \pm 7.5\%$ across 5 seeds, confirming zero regressions on canonical tasks.
- **Live Environment Execution (`homegrid-task` Gym environment via `.venv`):**
  1. `python agent/main.py --mode rl --task "find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside"`:
     - Succeeded in 3 skill steps (`[OK]`). Subgoals 1 & 2 completed, 3 & 4 pre-satisfied, 5 completed. Unrelated skills completely excluded.
  2. `python agent/main.py --mode rl --task "find the papers, pick them up, move to the kitchen, open the trash bin, and put the papers inside"`:
     - Succeeded in 4 skill steps (`[OK]`). All intermediate navigation, room drops, and bin placement verified.
  3. `python agent/main.py --mode rl --task "find the bottle, pick it up, move to the living room, open the compost bin, and put the bottle inside"`:
     - Succeeded in 5 skill steps (`[OK]`). Full 5-stage pipeline executed and verified end-to-end.
- **Invariants Checked:**
  - `skills/skills.json` matches Git HEAD byte-for-byte.
  - `skills/smdp_policy.json` remains untouched.
  - `homegrid/` untouched.

---

## 11. Next Steps

1. **Evaluation in Live Gym Environment:**
   - Execute multi-episode compound task evaluations across randomized layouts and seeds.
2. **Phase 5 / Documentation Polish:**
   - Final review and documentation updates for public `README.md` and `PROJECT_NOTES.md`.


