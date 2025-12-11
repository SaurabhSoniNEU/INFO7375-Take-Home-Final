# Reinforcement Learning for Agentic AI Systems

**Multi-Agent Content Creation with Deep Q-Networks and Thompson Sampling**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Author:** Saurabh Soni  
**Course:** INFO 7375 - Prompt Engineering for Generative AI  
**Institution:** Northeastern University  
**Date:** December 10, 2025

---

## 🎯 Project Overview

This project implements a reinforcement learning-enhanced multi-agent orchestration system that learns optimal coordination strategies for specialized AI agents. Using **Deep Q-Networks (DQN)** and **Thompson Sampling**, the system achieves **16.18% improvement** over baseline performance while maintaining **94.2% quality**.

### Key Features

- ✅ **Two RL Approaches**: DQN (value-based learning) + Thompson Sampling (exploration strategy)
- ✅ **5 Coordination Patterns**: Sequential, Parallel, Hierarchical, Collaborative, Adaptive
- ✅ **4 Specialized Agents**: Research, Writing, Editor, Technical
- ✅ **Novel Exploration**: Three-phase strategy ensuring pattern diversity
- ✅ **Multi-Objective Rewards**: Balances quality, efficiency, coordination, diversity
- ✅ **Statistical Validation**: Proven improvement (p < 0.001)

---

## 📊 Results Summary (500 Episodes)

| Metric | Value |
|--------|-------|
| **Average Reward** | 0.9286 (92.9%) |
| **Final Performance** | 0.9649 (96.5%) |
| **Average Quality** | 0.9420 (94.2%) |
| **Improvement** | 16.18% |
| **Pattern Diversity** | 96% (all 5 patterns used) |
| **Statistical Significance** | p < 0.001 |
| **Best Episode** | #256 (perfect 1.0 reward) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# 1. Clone or download the project
cd rl-agentic-ai-system

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Training

```bash
# Train for 200 episodes (recommended)
python main.py
# When prompted, enter: 200

# Or use default 200 episodes
python main.py
# Just press Enter
```

### Run Tests

```bash
# Verify installation
python test_system.py

# Should see: ✓ ALL TESTS PASSED!
```

### Generate Sample Content

```bash
# See what coordinated agents would produce
python content_generator.py
# Check: output/content/ folder
```

---

## 📁 Project Structure

```
rl-agentic-ai-system/
├── main.py                     # Main entry point
├── rl_orchestrator.py          # RL-based agent coordination
├── dqn_agent.py               # Deep Q-Network implementation
├── thompson_sampler.py        # Thompson Sampling & UCB
├── agents.py                  # 4 specialized agents
├── reward_function.py         # Multi-objective reward calculation
├── config.py                  # Configuration management
├── visualization.py           # Results visualization
├── evaluation.py              # Performance evaluation
├── test_system.py             # Test suite
├── content_generator.py       # Sample content generation
├── preserve_outputs.py        # Backup utility
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
└── output/                    # Generated during training
    ├── plots/                 # Visualizations (PNG)
    ├── results/               # Metrics (JSON)
    ├── models/                # Saved RL models
    ├── content/               # Sample generated content
    └── logs/                  # Training logs
```

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────┐
│         RL Agent System                 │
│                                         │
│  DQN Agent ←→ State Encoder             │
│      ↓              ↓                   │
│  RL Orchestrator                        │
│  • Pattern Selection (5 types)          │
│  • Three-Phase Exploration              │
│  • Multi-Objective Rewards              │
│      ↓              ↓                   │
│  Thompson Sampler ←→ Agent Team         │
│  • Beta(α,β) per agent                  │
│  • Bayesian Selection                   │
│                                         │
│  4 Specialized Agents:                  │
│  • Research  • Writing                  │
│  • Editor    • Technical                │
└─────────────────────────────────────────┘
```

---

## 🔬 Technical Approach

### Approach 1: Deep Q-Networks (DQN)

**Purpose:** Learn which coordination pattern to use for each task

**Components:**
- Q-Network: [256→128→64] hidden layers
- Experience Replay: 10,000 capacity
- Target Network: Updated every 10 steps
- Epsilon-Greedy: ε = 1.0 → 0.01 (decay: 0.97)

**State Space:** 32 dimensions (task features, agent states, context)  
**Action Space:** 5 coordination patterns

### Approach 2: Thompson Sampling

**Purpose:** Select best agents within coordination patterns

**Mechanism:**
- Beta distribution per agent: θᵢ ~ Beta(αᵢ, βᵢ)
- Sample and select: i* = argmax(θ̂ᵢ)
- Bayesian update: α += 1 (success), β += 1 (failure)

**Integration:** Used in Hierarchical and Adaptive patterns

---

## 📈 Key Innovation: Three-Phase Exploration

| Phase | Episodes | Strategy | Purpose |
|-------|----------|----------|---------|
| **Forced** | 1-30 | Cycle all patterns | Guarantee exploration |
| **Guided** | 31-60 | 50% DQN, 50% random | Safe transition |
| **Normal** | 61+ | Epsilon-greedy DQN | Exploit learned policy |

**Impact:** Increased pattern diversity from 0% → 96%

---

## 🎓 Usage Examples

### Basic Training

```python
from rl_orchestrator import RLAgentOrchestrator
from config import Config

# Initialize
config = Config()
orchestrator = RLAgentOrchestrator(config)

# Execute single task
task = {
    'type': 'blog_post',
    'topic': 'AI Trends',
    'requirements': {
        'length': 800,
        'tone': 'informative',
        'target_audience': 'general'
    }
}

result = orchestrator.execute_task(task)
print(f"Reward: {result['reward']:.3f}")
print(f"Quality: {result['quality_score']:.3f}")
print(f"Pattern Used: {result['coordination_pattern']}")
```

### Backup Your Results

```bash
# Before running new experiments, backup current results
python preserve_outputs.py backup 500ep

# List all backups
python preserve_outputs.py list

# Restore if needed
python preserve_outputs.py restore output_backup_500ep
```

---

## 📊 Understanding the Outputs

### Training Outputs

After training, check `output/` folder:

**1. Visualizations** (`output/plots/`)
- `training_curves.png` - Learning progress over episodes
- `agent_utilization.png` - Which agents were used (balanced 21-28%)
- `exploration_rate.png` - Epsilon decay (1.0 → 0.01)
- `final_results.png` - Complete 6-panel summary

**2. Metrics** (`output/results/`)
- `final_evaluation.json` - All 28 performance metrics
  - Average reward, quality, improvement percentage
  - Statistical test results (t-test, p-value)
  - Convergence analysis

**3. Models** (`output/models/`)
- `model_ep500_dqn.pth` - Trained DQN neural network
- Can load and deploy for production

**4. Sample Content** (`output/content/`)
- Examples of what coordinated agents would produce
- Shows different patterns in action

---

## 🔧 Configuration

Edit `config.py` or create `config.json`:

```python
{
    "num_episodes": 200,           # Training episodes
    "dqn_learning_rate": 0.001,    # DQN learning rate
    "dqn_epsilon_decay": 0.97,     # Epsilon decay rate
    "reward_quality_weight": 0.4,  # Quality importance
    "reward_diversity_weight": 0.2 # Diversity importance
}
```

---

## 📝 Assignment Requirements

**This project fulfills:**

✅ **Two RL Approaches**: DQN + Thompson Sampling  
✅ **Agentic System Integration**: Multi-agent orchestration  
✅ **Complete Implementation**: 2,700+ lines of production code  
✅ **Comprehensive Testing**: Full test suite included  
✅ **Experimental Design**: 500 episodes with statistical validation  
✅ **Results Analysis**: Learning curves, pattern analysis, significance tests  
✅ **Documentation**: Technical report, setup guides, code comments  

---

## 🐛 Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'torch'`
```bash
pip install torch numpy scipy matplotlib seaborn
```

**Issue:** Tests failing
```bash
rm -rf __pycache__
python test_system.py
```

**Issue:** Want to run quick test
```bash
python main.py
# Enter: 20  (just 20 episodes for testing)
```

---

## 📚 Documentation

- **Technical Report**: See `FINAL_TECHNICAL_REPORT.md` (or PDF version)
- **Quick Start**: See `QUICKSTART.md` for 5-minute setup
- **Experimental Design**: See `EXPERIMENTAL_DESIGN.md` for methodology
- **Demo Script**: See `PRACTICAL_DEMO_SCRIPT.md` for presentation guide

---

## 🎯 Key Results

### Pattern Performance

| Pattern | Usage | Avg Reward | Best For |
|---------|-------|------------|----------|
| **Adaptive** | 21.2% | **0.945** | Complex/uncertain tasks |
| **Hierarchical** | 14.0% | **0.938** | Coordination-intensive |
| **Collaborative** | 21.4% | 0.925 | Iterative refinement |
| **Sequential** | 23.4% | 0.921 | Simple linear tasks |
| **Parallel** | 20.0% | 0.914 | Independent subtasks |

### Agent Performance

- Agent 0 (Research): 21.7% usage, 0.928 quality
- Agent 1 (Writing): 23.7% usage, 0.945 quality
- **Agent 2 (Editor): 27.0% usage, 0.958 quality** ← Best
- Agent 3 (Technical): 27.6% usage, 0.941 quality

---

## 🔮 Future Enhancements

- **Real LLM Integration**: Replace simulated agents with GPT-4/Claude
- **Continuous Actions**: Fine-grained coordination control
- **Multi-Task Learning**: Transfer across domains
- **Meta-Learning**: Fast adaptation to new tasks
- **Hierarchical RL**: Temporal abstraction

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@project{soni2025rl,
  author = {Saurabh Soni},
  title = {Reinforcement Learning for Agentic AI Systems},
  year = {2025},
  institution = {Northeastern University},
  course = {INFO 7375 - Prompt Engineering for Generative AI}
}
```

---

## 📧 Contact

**Saurabh Soni**  
Northeastern University  
Course: INFO 7375 - Prompt Engineering for Generative AI

---

## 📜 License

This project is created for academic purposes as part of INFO 7375 coursework.
