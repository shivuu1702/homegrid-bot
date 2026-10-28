# AGENTS.md — Mandatory Instructions for Coding Agents

> Every coding agent working on this repository **must** follow these rules automatically.
> These are not optional recommendations.

---

## Before Making Changes

1. **Read this file** (`AGENTS.md`) and `docs/PROJECT_LOG.md` before editing code.
2. **Inspect the current Git state** (`git status`, `git log -5`, `git branch`) before modifying files.
3. **Understand relevant code** before changing behavior. Trace the execution flow through `agent/main.py → agent/agent.py → agent/skills.py → agent/llm.py → agent/config.py`. Read `PROJECT_NOTES.md` for architecture context.
4. **Do not modify files inside `homegrid/`** — this is an external environment library.

---

## During Work

5. **Commit safely.** Never amend existing commits, force-push, or rebase published history without explicit user approval.
6. **Never commit secrets.** Do not write API keys, tokens, passwords, or `.env` file contents into any tracked file, documentation, log, or terminal output.
7. **Do not blindly stage everything.** Inspect `git diff` and `git status` before committing. Exclude generated artifacts, caches, virtual environments, and secrets.

---

## Documentation Requirements

8. **Update `docs/PROJECT_LOG.md`** in the same work session whenever meaningful work occurs. Do this before finishing, even if the user did not ask for it.
9. In the project log, record:
   - What was inspected, changed, added, removed, tested, and **not tested**.
   - Commands run and their actual outcomes, including failures.
   - Architectural decisions and their rationale.
   - Bugs discovered, fixes attempted, and unresolved issues.
   - Changes to dependencies, configuration, model/provider, prompts, skills, execution logic, verification, or evaluation.
10. **Reference source files, commits, and logs** where useful.
11. **Maintain a current-status section** and an **actionable next-steps section**.
12. **Preserve previous history.** Append chronological entries. Never silently replace or rewrite earlier findings.
13. **Never invent results.** Do not claim tests passed when they were not run. Do not label assumptions as verified facts.
14. **Report documentation changes** alongside code changes in the final summary.

---

## Key Project Conventions

- **Entry point:** `python agent/main.py` (run from project root)
- **Configuration:** All settings in `agent/config.py`, loaded from `.env`
- **LLM provider:** Configured via `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` in `.env`
- **Skills library:** `skills/skills.json` — external memory, grows on successful tasks
- **Logs directory:** `logs/` — gitignored; contains per-run `.log` and `.json` summaries
- **Environment:** `homegrid-task` Gym environment (from `homegrid/` package)
- **Tests:** `test_agent.py`, `test_llm_providers.py` (unit, no network); `test_navigation.py` (requires environment); `live_provider_check.py` (requires API key)

---

## Related Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | This file — agent instructions |
| `docs/PROJECT_LOG.md` | Evolving project knowledge and history |
| `PROJECT_NOTES.md` | Architecture deep-dive and design decisions |
| `README.md` | Public-facing project overview |
| `README_TESTING.md` | Setup and testing instructions |
| `.env.example` | Environment variable template |
