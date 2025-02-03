# Unified Reasoning and Aggregation Framework (URAF)

## 🚀 Overview
**URAF (Unified Reasoning and Aggregation Framework)** is an evaluation framework designed to measure **structured reasoning, logical depth, and decision-making quality** of LLMs (Large Language Models) in agent-based workflows.

It supports:
 - ✅ **Benchmarking LLMs** for structured thinking and multi-step problem-solving.
 - ✅ **Evaluating agent readiness** (e.g., Backtracking Agents, Decision-Making Agents).
 - ✅ **Comparing multiple LLMs** over time to track performance.
 - ✅ **Custom model configurations** via YAML files for flexible evaluations.
 - ✅ **A CLI for easy execution and history tracking**.

---

## 📂 Project Structure
```
uraf/
│── __init__.py
│── core.py
│── llm_client.py
│── response_processor.py
│── evaluator.py
│── scorer.py
│── benchmark.py
│── benchmark_generator.py
│── benchmark_tracker.py
│── config_loader.py
│── cli.py  # CLI for dynamic agent evaluation
│── evaluate_agents.py  # Main evaluation script
examples/
│── config.yaml  # Default config file
│── qwen2.5-7b-instruct-1m-config.yaml  # Model-specific config
│── gpt-4-config.yaml  # Another model-specific config
│── run_evaluation.py  # Example script using CLI
data/
│── benchmark_results.json
```

---

## 🛠️ Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/URAF.git
cd URAF
```

### **2️⃣ Install Dependencies**
```bash
poetry install
```

---

## 🔥 Usage
### **Run an Agent-Based Evaluation**
```bash
poetry run python -m uraf.cli --run --config qwen2.5-7b-instruct-1m-config.yaml
```

### **Check Model Evaluation History**
```bash
poetry run python -m uraf.cli --history
```

### **Compare Model Performances**
```bash
poetry run python -m uraf.cli --compare
```

### **Export Results to CSV**
```bash
poetry run python -m uraf.cli --export
```

---

## ⚙️ Configurations
URAF supports **custom model configurations** via YAML files.

🔹 **Example: `examples/qwen2.5-7b-instruct-1m-config.yaml`**
```yaml
llm:
  model: "qwen2.5-7b-instruct-1m"
  api_url: "http://localhost:1234/v1/completions"
  max_tokens: 10000
  temperature: 0.5
  top_p: 0.85
  top_k: 50
  min_p: 0.2
  presence_penalty: 1.0

evaluation:
  readiness_thresholds:
    Multi-Step Critical Thinking Agent: 7.0
    Backtracking & Self-Correcting Agent: 6.5
    Multi-Perspective Analysis Agent: 7.5
    Decision-Making Agent: 7.0
    Autonomous Planning Agent: 8.0

storage:
  benchmark_results_path: "data/qwen2.5-7b-results.json"
```

Use different model configs by passing `--config <config-file>` to the CLI.

---

## 🏆 Supported Agent Evaluations
URAF evaluates LLMs across **specific agent types**:
| **Agent Type** | **Key Capabilities** |
|---------------|----------------------|
| **Multi-Step Critical Thinking Agent** | Logical breakdown of complex problems |
| **Backtracking & Self-Correcting Agent** | Identifies errors & optimizes responses |
| **Multi-Perspective Analysis Agent** | Evaluates problems from different viewpoints |
| **Decision-Making Agent** | Weighs trade-offs and selects optimal paths |
| **Autonomous Planning Agent** | Structures tasks & executes multi-step workflows |

---

## 🛠️ Development & Contribution
### **Run URAF in Development Mode**
```bash
export PYTHONPATH=$(pwd)
poetry run python -m uraf.cli --run
```

### **Testing**
```bash
pytest tests/
```

### **Code Formatting & Linting**
```bash
black .
flake8 .
mypy uraf/
```

### **Want to Contribute?**
🚀 Open an issue or submit a PR to improve URAF!

---

## 📜 License
This project is licensed under the MIT License.

---

## 🤝 Contact
For questions, contact **[Your Name]** at **your.email@example.com**.

