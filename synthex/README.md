# SYNTHEX

> Created by **Sharaf Samiur Rahman** — [github.com/impulsiveaura](https://github.com/impulsiveaura)

**Dual-stream inference pipeline for model-agnostic AI reasoning.**

Most LLMs process a query in a single forward pass — one reasoning path, one output. SYNTHEX runs two parallel processing pipelines and synthesizes their outputs through cross-stream coherence fusion before emitting a final response.

```bash
pip install synthex[anthropic]
```

```python
from synthex import Pipeline

p = Pipeline(model="claude-sonnet-4-6")
result = p.run("What are the tradeoffs of microservices vs monoliths?")
print(result.consensus)
```

---

## Why SYNTHEX?

Standard single-pass inference has a well-known failure mode: the model's literal interpretation of a query and its implicit understanding of what you *really* need can diverge silently. You get an answer to the words, not the question.

SYNTHEX separates these two processing paths explicitly:

| Pipeline A | Pipeline B |
|---|---|
| Tokenization · Literal | Inference Core · Temporal |
| Tokenization · Latent | Inference Core · Generative |
| Embedding Fusion | Signal Distillation |

**Pipeline A** runs Literal and Latent tokenization in parallel, then computes their coherence intersection — the fused embedding space where both streams agree.

**Pipeline B** fires two inference cores in parallel: Temporal (context-invariant principles) and Generative (cross-domain synthesis). Signal Distillation rejects artifacts that only appear in one stream.

The **Inference Mode Selector** profiles each query and idles unused modes — simple factual queries skip expensive generative synthesis entirely.

The **Consensus Layer** merges all distilled signals with zero embedding bleed.

---

## Architecture

```
Query
  │
  ├── [PIPELINE A]
  │     ├── Tokenization · Literal  ─┐
  │     ├── Tokenization · Latent   ─┤ (parallel)
  │     └── Embedding Fusion ←───────┘ (coherence intersection)
  │
  ├── [PIPELINE B]
  │     ├── Inference Core · Temporal   ─┐
  │     ├── Inference Core · Generative ─┤ (parallel)
  │     └── Signal Distillation ←────────┘ (dual-confidence gate)
  │
  ├── [MODE SELECTOR]  — per-query inference budget
  │
  └── [CONSENSUS LAYER] — zero-bleed synthesis → final output
```

---

## Installation

```bash
pip install synthex[anthropic]   # Claude
pip install synthex[openai]      # GPT
pip install synthex[ollama]      # Local models
pip install synthex[all]         # Everything
```

---

## Usage

```python
from synthex import Pipeline

# Basic
p = Pipeline(model="claude-sonnet-4-6")
result = p.run("What is the future of energy storage?")
print(result.consensus)

# Inspect every layer
print(result.tokenization_literal.content)
print(result.tokenization_latent.content)
print(result.embedding_fusion.content)
print(result.inference_temporal.content)
print(result.inference_generative.content)
print(result.signal_distillation.content)

# Check inference modes
print(result.mode_profile.active_modes)
print(result.mode_profile.efficiency_ratio)

# Async
result = await p.run_async("your query")

# Batch
results = await p.run_batch_async(["q1", "q2", "q3"])

# Other models
Pipeline(model="gpt-4o")
Pipeline(model="ollama/llama3")

# Verbose
Pipeline(model="claude-sonnet-4-6", verbose=True)

# Export JSON
import json
print(json.dumps(result.to_dict(), indent=2))
```

---

## CLI

```bash
synthex "What is the future of battery storage?"
synthex "Explain attention" --model gpt-4o
synthex "What is consciousness?" --model ollama/llama3
synthex "Best API practices" --output json
synthex "query" --output layers
synthex --batch queries.txt
synthex "query" --verbose
```

---

## Inference Modes

| Mode | Description |
|---|---|
| `LITERAL` | Explicit token semantics |
| `LATENT` | Implicit embedding space |
| `TEMPORAL` | Context-invariant principles |
| `GENERATIVE` | Cross-domain synthesis |

---

## Requirements

- Python 3.9+
- One provider: `anthropic`, `openai`, or `ollama`

---

## Development

```bash
git clone https://github.com/impulsiveaura/synthex
cd synthex
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — Created by Sharaf Samiur Rahman
