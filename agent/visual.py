"""
visual.py — Visual rendering wrapper for the HomeGrid agent.

Wraps the real HomeGrid environment so that every env.step() call
automatically renders the frame to a matplotlib Window and prints
a concise per-action log line.

This file reuses the existing HomeGrid Window class and get_frame()
method — no new renderer is created.

Usage:
    from visual import attach_visual
    env = attach_visual(env)   # wraps the env; window opens immediately
"""

import time
import gym

from config import HOMEGRID_RENDER_DELAY

# Action integer → human-readable name (matches skills.py constants)
ACTION_NAMES = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
    4: "PICKUP",
    5: "DROP",
    6: "GET",
    7: "PEDAL",
    8: "GRASP",
    9: "LIFT",
}


class VisualEnvWrapper(gym.Wrapper):
    """Transparent wrapper that renders after every step.

    • Intercepts ``step()`` and ``reset()``
    • Renders the frame to the existing HomeGrid Window
    • Prints a concise per-action terminal log
    • Sleeps for ``HOMEGRID_RENDER_DELAY`` between frames
    • All other env attributes/methods pass through unchanged
    """

    def __init__(self, env, window, delay: float = HOMEGRID_RENDER_DELAY):
        super().__init__(env)
        self.window = window
        self.delay = delay
        self._step_count = 0

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._render_and_log(action, reward, terminated, truncated)
        return obs, reward, terminated, truncated, info

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._step_count = 0
        self._render_frame()
        return result

    # ── internal helpers ─────────────────────────────────────────────────

    def _render_frame(self):
        """Render the current frame to the Window (if still open)."""
        if self.window.closed:
            return
        try:
            img = self.env.unwrapped.get_frame()
            self.window.show_img(img)
        except Exception:
            pass  # don't crash the agent if rendering fails

    def _render_and_log(self, action, reward, terminated, truncated):
        """Render frame, print action log, sleep for delay."""
        # Render
        self._render_frame()

        # Per-action terminal log
        action_name = ACTION_NAMES.get(action, str(action))
        try:
            state = self.env.unwrapped.get_full_symbolic_state()
            pos = tuple(int(c) for c in state["agent"]["pos"])
            front = state.get("front_obj") or "empty"
            carrying = state["agent"].get("carrying") or "nothing"
        except Exception:
            pos = "?"
            front = "?"
            carrying = "?"

        print(
            f"  [HomeGrid] action={action_name:<7s} "
            f"pos={pos}  reward={reward:.2f}  "
            f"front={front}  carrying={carrying}"
        )

        # Update window caption
        if not self.window.closed:
            try:
                task = getattr(self.env, "task", "")
                self.window.set_caption(
                    f"Task: {task}  |  Step {self._step_count}  |  "
                    f"Action: {action_name}"
                )
            except Exception:
                pass

        # Configurable delay so the human can watch
        if self.delay > 0 and not self.window.closed:
            time.sleep(self.delay)

    # ── cleanup ──────────────────────────────────────────────────────────

    def close(self):
        """Close the window and the underlying environment."""
        if not self.window.closed:
            self.window.close()
        self.env.close()


def attach_visual(env, enabled: bool = True):
    """Wrap *env* with visual rendering.  Returns the wrapped env.

    If ``enabled`` is False, returns the env unchanged (headless mode).
    """
    if not enabled:
        return env

    # Import the existing HomeGrid Window class
    from homegrid.window import Window
    import matplotlib.pyplot as plt

    # Disable default matplotlib keybindings so they don't interfere
    for k in plt.rcParams:
        if "keymap" in k:
            plt.rcParams[k] = []

    window = Window("HomeGrid Agent")
    window.show(block=False)  # non-blocking interactive mode

    # Render the initial frame
    try:
        img = env.unwrapped.get_frame()
        window.show_img(img)
    except Exception:
        pass

    return VisualEnvWrapper(env, window)
