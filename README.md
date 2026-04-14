# 🔬 Hallucination Detection Benchmark — Ollama Edition

Test **any LLM** for hallucination using 4 detection methods from the  
[AWS ML blog: "Detect hallucinations for RAG-based systems" (2025)](https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems/).

**100% local. No API keys. No GPU setup. Just Ollama.**

---

## ⚡ Quick Start (3 commands)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh   # Linux/Mac
# Windows: download from https://ollama.com/download

# 2. Pull models you want to test
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull gemma3:4b

# 3. Run benchmark
pip install -r requirements.txt
python main.py
```

---

## 🤖 Supported Models (just add more to config.yaml!)

| Family | Models in config.yaml |
|--------|----------------------|
| **Meta / Llama** | llama3.2:3b, llama3.1:8b, llama3.3:70b |
| **Google / Gemma** | gemma3:1b, gemma3:4b, gemma3:12b |
| **Mistral** | mistral:7b, mistral-nemo:12b, mixtral:8x7b |
| **Microsoft / Phi** | phi4:14b, phi3.5:3.8b |
| **Alibaba / Qwen** | qwen2.5:7b, qwen2.5:14b, qwen2.5:72b |
| **DeepSeek** | deepseek-r1:7b, deepseek-r1:14b |
| **Cohere** | command-r7b:7b, aya-expanse:8b |
| **TII / Falcon** | falcon3:7b |
| **HuggingFace** | smollm2:1.7b |

**Add ANY model** from https://ollama.com/library by editing `config.yaml`:
```yaml
models:
  - name: "my-model"
    model: "ollama-tag:size"   # exact tag from ollama.com/library
    family: "company"
    auto_pull: false
```

---

## 🛠 CLI Commands

```bash
# See what's installed in Ollama
python main.py --list-models

# See all models defined in config.yaml
python main.py --list-available

# Pull models then run benchmark
python main.py --pull llama3.2:3b mistral:7b gemma3:4b

# Run specific models only
python main.py --models llama3.2-3b mistral-7b

# Fast test (20 samples, synthetic only)
python main.py --quick

# Skip slow BERT stochastic detector
python main.py --no-bert

# Use remote Ollama server
python main.py --host http://192.168.1.10:11434

# Check setup without running inference
python main.py --dry-run

# Smoke-test (no Ollama needed)
python quick_demo.py

# Smoke-test with Ollama
python quick_demo.py --with-ollama llama3.2:3b
```

---

## 📐 Detection Methods

| # | Method | Accuracy | Precision | Recall | LLM Calls |
|---|--------|----------|-----------|--------|-----------|
| 1 | **Token Similarity** | 0.47 | **0.96** | 0.03 | 0 — free |
| 2 | **Semantic Similarity** | 0.48 | 0.90 | 0.02 | 0 (embeddings only) |
| 3 | **LLM Prompt-Based** | 0.75 | 0.94 | 0.53 | 1 per sample |
| 4 | **BERT Stochastic** | **0.76** | 0.72 | **0.90** | N+1 per sample |
| + | **Ensemble** | best combined | — | — | N+2 |

*Metrics from AWS blog, averaged across Wikipedia + synthetic datasets.*

**Recommendation:**
- Use **Token Similarity** as a cheap pre-filter (catches obvious hallucinations)
- Use **LLM-Based** for the best accuracy/cost ratio
- Use **BERT Stochastic** when recall is critical (medical, legal)
- Use **Ensemble** for production systems

---

## 📁 Project Structure

```
hallucination_benchmark/
├── main.py                  # Main entry point
├── quick_demo.py            # Smoke-test (no Ollama required)
├── config.yaml              # All settings — models, datasets, detectors
├── requirements.txt
│
├── models/
│   ├── base_model.py        # Abstract interface
│   ├── ollama_model.py      # Ollama backend (works for ALL models)
│   └── model_factory.py     # Builds model list, checks what's installed
│
├── data/
│   └── datasets.py          # HaluEval, RAGBench, synthetic loader
│
├── detectors/
│   ├── llm_detector.py      # Method 1: LLM judge prompt
│   ├── semantic_detector.py # Method 2: Cosine similarity
│   ├── bert_detector.py     # Method 3: BERT stochastic checker
│   ├── token_detector.py    # Method 4: BLEU + ROUGE-L + intersection
│   └── ensemble.py          # Weighted combination
│
└── benchmark/
    ├── runner.py             # Orchestrates all model × dataset runs
    ├── evaluator.py          # Accuracy, Precision, Recall, F1, AUC-ROC
    └── reporter.py           # Console + charts + HTML report
```

---

## ⚙️ config.yaml Key Settings

```yaml
# Which models to run ([] = all installed models)
selected_models: ["llama3.2-3b", "mistral-7b"]

# Ollama server (change for remote)
ollama:
  host: "http://localhost:11434"

# Number of stochastic samples for BERT method (more = better recall, slower)
detectors:
  bert_stochastic:
    n_samples: 5

# Max samples per dataset
datasets:
  - name: "halueval_qa"
    max_samples: 150
```

---

## 📊 Output Files

```
results/
├── report.html              # Self-contained HTML report with all charts
├── all_results.csv          # Every prediction from every model/detector
├── summary.json             # Per-(model × method) metrics
├── method_comparison.png    # Bar chart: all 4 methods side by side
├── model_method_heatmap.png # Heatmap: F1 score for each model × method
├── score_distributions.png  # Violin plots: scores split by ground truth
├── precision_recall.png     # PR scatter with F1 annotations
└── benchmark.log            # Full run log
```

---

## 📚 References

- [AWS ML Blog: Detect hallucinations for RAG-based systems (2025)](https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems/)
- [HaluEval benchmark](https://arxiv.org/abs/2305.11747)
- [RAGBench benchmark](https://arxiv.org/abs/2407.11005)
- [BERTScore paper](https://arxiv.org/abs/1904.09675)
- [Ollama model library](https://ollama.com/library)
