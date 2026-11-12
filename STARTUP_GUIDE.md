# HomeGrid-Bot Startup Guide

This guide explains how to start HomeGrid-Bot from a fresh Windows PowerShell terminal.

## 1. Open PowerShell and navigate to the project

Use a standalone Windows PowerShell terminal (not the Antigravity integrated terminal when launching the visual environment).

```powershell
cd C:\Users\sanja\homegrid-bot
```

## 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt should show `(.venv)`.

If PowerShell blocks activation because of the execution policy, run this in the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Check the branch and working tree

```powershell
git branch --show-current
git status
```

The expected branch is `main`. A clean working tree should report:

```text
nothing to commit, working tree clean
```

To download the latest committed changes from GitHub when needed:

```powershell
git pull origin main
```

You do not need to pull every time you start the project.

## 4. Check Python and dependencies

```powershell
python --version
python -m pip check
```

The project environment was set up with Python 3.10.11.

If dependencies need to be installed or `requirements.txt` has changed:

```powershell
python -m pip install -r requirements.txt
```

## 5. Check Gemini configuration

The local `.env` file should contain your Gemini configuration. Example variable names:

```dotenv
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_API_KEY=your_actual_key
```

Keep the real API key in `.env`. Do not paste it into source code, commit it, or share it in screenshots. Do not replace your real key with the placeholder above.

## 6. Run HomeGrid-Bot

Run these commands from the project root with `(.venv)` activated.

### RL mode — trained SMDP policy

```powershell
python agent/main.py --mode rl --task "find the fruit, pick it up, and put it in the recycling bin" --policy-file skills/smdp_policy.json --visual
```

### LLM mode — Gemini

```powershell
python agent/main.py --mode llm --task "find the bottle"
```

### Hybrid mode — RL with LLM fallback

```powershell
python agent/main.py --mode hybrid --task "find the fruit and put it in the recycling bin" --policy-file skills/smdp_policy.json --visual
```

### Random mode — baseline

```powershell
python agent/main.py --mode random --task "find the fruit" --visual
```

## 7. Run a compound task

Example compound task using the RL policy:

```powershell
python agent/main.py --mode rl --task "find the fruit, pick it up, move to the kitchen, open the recycling bin, and put the fruit inside" --policy-file skills/smdp_policy.json --visual
```

## 8. Run tests

Run the complete test suite:

```powershell
python -m pytest -q
```

Run selected test files:

```powershell
python -m pytest test_rl_selector.py -v
python -m pytest test_benchmark.py -v
python -m pytest test_agent.py -v
python -m pytest test_llm_providers.py -v
```

## 9. Useful Git commands

```powershell
git branch --show-current
git status
git pull origin main
git log -1 --oneline
```

To deactivate the virtual environment:

```powershell
deactivate
```

## Quick start — usual RL run

For a normal run after the environment has already been created and dependencies installed, these are the essential commands:

```powershell
cd C:\Users\sanja\homegrid-bot
.\.venv\Scripts\Activate.ps1
python agent/main.py --mode rl --task "find the fruit, pick it up, and put it in the recycling bin" --policy-file skills/smdp_policy.json --visual
```

## Important notes

- Use standalone Windows PowerShell for visual runs.
- Activate `.venv` before running Python commands.
- Run commands from the project root: `C:\Users\sanja\homegrid-bot`.
- The trained RL policy is stored in `skills/smdp_policy.json`.
- Keep `.env` private and never commit API keys.
- Do not delete or overwrite policy JSON files unless you intentionally want to replace the learned policy.
- Install dependencies again only when required, such as after dependency changes.
- The exact available CLI flags can be checked with:

```powershell
python agent/main.py --help
```
