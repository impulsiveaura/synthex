# SYNTHEX

> Created by **Sharaf Samiur Rahman** — [github.com/impulsiveaura](https://github.com/impulsiveaura)

**A structured prompt orchestration system for more reliable AI reasoning.**

Most LLMs process a query in a single unstructured pass — one reasoning path, one output. SYNTHEX routes queries through parallel specialised processing steps and synthesises only what they agree on before returning a final response.

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

Single-pass inference has a consistent failure mode: the model's literal interpretation of a query and its implicit understanding of what you actually need can diverge silently. You get an answer to the words, not the question.

SYNTHEX separates these two processing paths explicitly across two parallel sequences:

| Path A | Path B |
|---|---|
| Literal processing | Temporal reasoning |
| Latent processing | Generative synthesis |
| Coherence fusion | Signal distillation |

**Path A** runs literal and latent processing steps in parallel, then fuses only what both agree on.

**Path B** runs temporal and generative reasoning steps in parallel against the fused output. Distillation retains only what both steps agree on.

The **Mode Selector** profiles each query and skips processing steps it doesn't need — simple queries route through lightweight paths.

The **Consensus step** merges all distilled outputs into a single final response.

> Whether this approach produces measurably better outputs than single-pass inference is an open empirical question. Benchmarking is the next step.

---

## Architecture

```
Query
  │
  ├── [PATH A]
  │     ├── Literal Processing   ─┐
  │     ├── Latent Processing    ─┤ (parallel)
  │     └── Coherence Fusion ←───┘
  │
  ├── [PATH B]
  │     ├── Temporal Reasoning   ─┐
  │     ├── Generative Synthesis ─┤ (parallel)
  │     └── Signal Distillation ←┘
  │
  ├── [MODE SELECTOR] — per-query processing budget
  │
  └── [CONSENSUS] — final output
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

# Inspect every step
print(result.tokenization_literal.content)
print(result.tokenization_latent.content)
print(result.embedding_fusion.content)
print(result.inference_temporal.content)
print(result.inference_generative.content)
print(result.signal_distillation.content)

# Check which modes activated
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

## Processing Steps

| Step | Description |
|---|---|
| `LITERAL` | What is explicitly being asked |
| `LATENT` | What is implicitly being asked |
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

## Benchmarking

Independent benchmark results are welcome. If you run SYNTHEX against single-pass inference on a meaningful dataset, open a pull request with your methodology and results.

---

## License

MIT — Created by Sharaf Samiur Rahman
