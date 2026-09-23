# 🏠 HomeGrid — Voyager-Inspired AI Agent

> **An experimental autonomous agent that understands natural-language tasks and learns to act inside a simulated home environment.**

HomeGrid is a small-scale, real-world-inspired AI agent project developed as part of our **Artificial Intelligence course project**.

The project is inspired by the idea behind **Voyager**: instead of building a system that simply performs predefined actions, we are exploring how an agent can **interpret a task, plan a sequence of actions, interact with an environment, and improve its behavior over time**.

Rather than using a computationally expensive environment like Minecraft, HomeGrid provides a lightweight environment that allows us to experiment with the core ideas of an autonomous agent on consumer hardware.

---

## ✨ What Can It Do?

The agent receives a task in natural language, such as:

```text
Find the bottle.
```

or a more complex task:

```text
Find the bottle and put it in the recycling bin.
```

It then attempts to:

**Understand → Plan → Act → Observe → Continue**

The environment contains objects, locations, and possible interactions that the agent must reason about to complete the given task.

---

## 🧠 Voyager-Inspired Architecture

The current system follows a simplified agent pipeline:

```text
             Natural Language Task
                      │
                      ▼
              🧠 Task Understanding
                      │
                      ▼
                 📋 Planning
                      │
                      ▼
               ⚙️ Action Selection
                      │
                      ▼
              🏠 HomeGrid Environment
                      │
                      ▼
                 👁️ Observation
                      │
                      └──────────► Next Action
```

The long-term goal is to extend this into a more capable autonomous agent with better:

- 🧠 Task understanding
- 📋 Multi-step planning
- 🔄 Feedback and replanning
- 💾 Memory
- 🛠️ Skill generation/reuse
- 📈 Task performance

---

## 🎯 Why HomeGrid?

Large environments such as Minecraft are excellent for studying autonomous agents, but they can also introduce significant computational and implementation complexity.

HomeGrid gives us a **controlled and lightweight environment** where we can focus on the agent itself.

This makes it easier to experiment with:

- LLM-based planning
- Agent–environment interaction
- Tool/action selection
- Multi-step tasks
- Memory
- Error handling
- Autonomous decision making

without requiring a high-end GPU.

---

## 🛠️ Tech Stack

- **Python**
- **HomeGrid / MiniGrid environment**
- **Large Language Model (LLM)**
- **OpenAI-compatible API**
- **Gymnasium**
- **dotenv**
- **Git & GitHub**

The architecture is designed so that the underlying AI model and API configuration can be changed without rewriting the entire project.

---

## 📂 Project Structure

```text
HomeGrid/
│
├── agent/              # Agent logic and decision making
├── environment/        # HomeGrid environment interaction
├── prompts/            # LLM prompts
├── memory/             # Agent memory components
├── utils/              # Utility functions
│
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
├── main.py             # Main entry point
├── README.md
└── ...
```

> The structure may evolve as the agent architecture develops.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the environment

Create a `.env` file based on `.env.example` and add the required API configuration.

```env
API_KEY=your_api_key
MODEL=your_model
```

### 5. Run the agent

```bash
python main.py
```

---

## 🧪 Current Status

🚧 **Active Development**

The current version is an early working prototype.

At this stage, the primary focus is establishing a reliable agent pipeline and testing how well an LLM can translate natural-language instructions into actions inside HomeGrid.

Some complex or ambiguous instructions may still be interpreted incorrectly.

For example:

```text
Find the papers and put them in the recycling bin.
```

may require the agent to correctly identify **both the target object and the destination** before executing the complete sequence.

Improving this kind of task understanding and multi-step execution is one of the next major development goals.

---

## 🗺️ Roadmap

### Phase 1 — Core Agent
- [x] HomeGrid environment
- [x] Natural-language task input
- [x] LLM-based decision making
- [x] Environment interaction
- [x] Basic task execution

### Phase 2 — Better Reasoning
- [ ] Improve task interpretation
- [ ] Reliable multi-step planning
- [ ] Better action validation
- [ ] Error detection & recovery

### Phase 3 — Agent Memory
- [ ] Short-term memory
- [ ] Long-term memory
- [ ] Skill storage
- [ ] Skill reuse

### Phase 4 — Voyager-Inspired Extensions
- [ ] Automatic skill generation
- [ ] Experience-based improvement
- [ ] Task decomposition
- [ ] Self-reflection / feedback loop

---

## 📊 Example

**Input**

```text
Find the bottle.
```

**Agent**

```text
Task → Identify bottle
     → Locate bottle
     → Navigate to bottle
     → Complete task
```

For a multi-step task:

```text
Find the bottle and put it in the recycling bin.
```

the agent needs to reason about:

```text
Task
 │
 ├── Find bottle
 │
 ├── Locate recycling bin
 │
 └── Move bottle → recycling bin
```

This transition from **single-action tasks to reliable multi-step behavior** is a central part of the project.

---

## 🌱 Project Vision

HomeGrid is not intended to be a full reproduction of Voyager.

Instead, it is a **small-scale experimental platform** for understanding and implementing the ideas behind autonomous LLM-based agents.

The ultimate goal is to move from:

> **"Give the agent an action."**

to:

> **"Give the agent a goal — let it figure out how to achieve it."** 🤖

---

## 👨‍💻 Team

**AI Course Project — IIIT Vadodara**

Built with curiosity, experiments, and probably too many debugging sessions. 😭

---

## ⭐ Future Updates

This README will evolve alongside the project.

Future versions may include:

- 🎥 Demo video
- 🖼️ Environment screenshots
- 🏗️ Detailed architecture diagram
- 📊 Evaluation results
- 🧪 Experiment logs
- 🧠 Agent memory visualization
- 📈 Performance comparisons

---

### 📜 License

This project is currently intended for **academic and educational purposes**.
