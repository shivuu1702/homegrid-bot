# README_TESTING.md — How to Run the Voyager-Inspired HomeGrid Agent

## 1. Prerequisites

- **Python 3.8 or higher** (tested on Python 3.8–3.11)
- **An OpenAI API key** (see Section 4)
- **Git** (only needed if you haven't cloned the repo yet)

Check your Python version:
```bash
python --version
```

---

## 2. Installation

All commands are run from the **project root** (`AI-Project/`).

### Step 1 — Install HomeGrid (from the local repo)
```bash
pip install -e homegrid/
```

This installs HomeGrid and all its dependencies:
- `gym==0.26`
- `numpy`
- `matplotlib`
- `tokenizers`
- `sentencepiece`

### Step 2 — Install agent dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `openai>=1.0.0` — to call the LLM API
- `python-dotenv` — to load the `.env` file

---

## 3. HomeGrid Setup

HomeGrid is already installed in the `homegrid/` subfolder.
**Do not modify any files inside `homegrid/`.**

The agent uses HomeGrid as an external library:
```python
import gym
import homegrid
env = gym.make("homegrid-task", disable_env_checker=True)
```

To verify HomeGrid works:
```bash
python -c "
import gym, homegrid
env = gym.make('homegrid-task', disable_env_checker=True)
obs, info = env.reset()
print('HomeGrid OK!')
print('Task:', env.task)
print('State keys:', list(info['symbolic_state'].keys()))
"
```

Expected output:
```
HomeGrid OK!
Task: find the bottle
State keys: ['step', 'agent', 'objects', 'front_obj', 'unsafe']
```

---

## 4. LLM / API Setup

### Which API is used?
**OpenAI Chat API** — `gpt-4o-mini` (default, recommended)

### How to get an API key
1. Go to https://platform.openai.com/api-keys
2. Sign in and click **"Create new secret key"**
3. Copy the key (it starts with `sk-`)

### Where to put the key
Create a `.env` file in the **project root** (`AI-Project/.env`):

```bash
# On Windows (PowerShell):
copy .env.example .env
# Then open .env in a text editor and add your key

# On Linux/Mac:
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
```

### How to change the model
Edit `.env` and change `OPENAI_MODEL`:
```
OPENAI_MODEL=gpt-4o        # more powerful, more expensive
OPENAI_MODEL=gpt-3.5-turbo # cheaper, less reliable
```

### How the LLM is used
The LLM is called **only for planning** — not for every action.
- Called once at the **start of each task** to generate a skill plan
- Called again if a task **fails** (self-correction)
- Never called for individual movement steps (BFS handles navigation)

**The LLM is never fine-tuned.** It is used as a black-box reasoner.

### What happens if the API key is missing
The agent will print a clear error and exit:
```
[ERROR] OPENAI_API_KEY is not set.
  1. Copy .env.example to .env
  2. Add your OpenAI API key to .env
  3. Re-run the agent
```

---

## 5. Running the Agent

```bash
python agent/main.py
```

That's it. Run from the **project root** (`AI-Project/`).

---

## 6. Expected Output

```
============================================================
  Voyager-Inspired HomeGrid Agent
============================================================
  LLM Model : gpt-4o-mini
  Log file  : logs/run_20260921_192300.log
  Skills    : skills/skills.json
============================================================

[Agent] API key detected ✓
[Agent] Creating HomeGrid environment (homegrid-task)...
[Agent] Environment created ✓

[Agent] Skill library loaded: 0 learned skill(s)

[Agent] Starting curriculum...
  Task types to attempt: ['find', 'get', 'open', 'cleanup', 'rearrange']

────────────────────────────────────────────────────────────
[Curriculum] Task type: FIND
[Curriculum] Got task: find the bottle

============================================================
[Agent] Task: find the bottle
============================================================

[Agent] Attempt 1/3
[Agent] Asking LLM for plan...
[Agent] LLM response:
1. find_object(bottle)

[Agent] Plan: [('find_object', ['bottle'])]
[Agent] Executing skill: find_object(bottle)
  [nav] Already facing 'bottle'
[HomeGrid] Skill 'find_object' result: success

[Agent] ✓ Task successful!
[Agent] Saving skill: 'find_bottle' to library

[Library] Skills saved: 1
  ✓ find_bottle (success_count=1)

... (continues for each task type) ...

============================================================
  CURRICULUM COMPLETE — SUMMARY
============================================================
  Tasks attempted : 5
  Tasks succeeded : 4
  Tasks failed    : 1

  [1] ✓ SUCCESS | attempts=1 | find the bottle
  [2] ✓ SUCCESS | attempts=1 | get the fruit
  [3] ✓ SUCCESS | attempts=2 | open the trash bin
  [4] ✓ SUCCESS | attempts=1 | put the papers in the recycling bin
  [5] ✗ FAILED  | attempts=3 | move the plates to the kitchen
```

---

## 7. Testing Individual Components

### Test HomeGrid connection
```bash
python -c "
import gym, homegrid
env = gym.make('homegrid-task', disable_env_checker=True)
obs, info = env.reset()
print('Task:', env.task)
state = info['symbolic_state']
print('Objects:', [o['name'] for o in state['objects']])
print('Agent pos:', state['agent']['pos'])
"
```

### Test LLM connection
```bash
python -c "
import sys; sys.path.insert(0, 'agent')
from llm import ask_llm
r = ask_llm('Say: LLM connection successful')
print(r)
"
```

### Test skill library
```bash
python -c "
import sys; sys.path.insert(0, 'agent')
from skills import save_skill, load_skills, get_relevant_skills
save_skill('test_skill', 'A test skill', 'find')
lib = load_skills()
print('Skills:', lib['skills'])
relevant = get_relevant_skills('find the bottle')
print('Relevant:', relevant)
"
```

### Test the complete agent on one task
```bash
python -c "
import sys; sys.path.insert(0, 'agent')
import gym, homegrid
env = gym.make('homegrid-task', disable_env_checker=True)
env.reset()
print('Task:', env.task)
from agent import run_task
result = run_task(env)
print('Result:', result)
"
```

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'homegrid'`
**Fix:** Install HomeGrid from the local repo:
```bash
pip install -e homegrid/
```

### `ModuleNotFoundError: No module named 'openai'`
**Fix:** Install agent requirements:
```bash
pip install -r requirements.txt
```

### `[ERROR] OPENAI_API_KEY is not set.`
**Fix:** Create your `.env` file:
```bash
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```
Then edit `.env` and add your key.

### `openai.AuthenticationError: Invalid API key`
**Fix:** Check that your API key is correct in `.env`. It should start with `sk-`.

### `openai.RateLimitError`
**Fix:** Your API key has hit its rate limit. Wait a moment and retry, or use a key with a higher quota.

### `gym.error.NameNotFound: Environment 'homegrid-task' doesn't exist`
**Fix:** The homegrid package needs to be imported to register environments:
```python
import homegrid  # This line is required before gym.make()
```

### Navigation seems stuck / agent doesn't reach the object
This can happen with objects placed in corners. The agent will report a timeout and the LLM will be asked to self-correct. Increase `MAX_NAV_STEPS` in `agent/config.py` if needed.

### Skills file not updating
Check that `skills/` directory exists and is writable. The agent creates it automatically, but check permissions if on Linux.
