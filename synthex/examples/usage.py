"""
SYNTHEX usage examples.
Created by Sharaf Samiur Rahman — github.com/impulsiveaura
"""
import asyncio
from synthex import Pipeline


# 1. Basic
def basic():
    p = Pipeline(model="claude-sonnet-4-6")
    result = p.run("What is the future of solid-state batteries?")
    print(result.consensus)


# 2. Inspect all layers
def inspect_layers():
    p = Pipeline(model="claude-sonnet-4-6")
    result = p.run("How does transformer attention work?")
    print("LITERAL:    ", result.tokenization_literal.content)
    print("LATENT:     ", result.tokenization_latent.content)
    print("FUSION:     ", result.embedding_fusion.content)
    print("TEMPORAL:   ", result.inference_temporal.content)
    print("GENERATIVE: ", result.inference_generative.content)
    print("DISTILLED:  ", result.signal_distillation.content)
    print("CONSENSUS:  ", result.consensus)


# 3. Check which modes fired
def check_modes():
    p = Pipeline(model="claude-sonnet-4-6")
    result = p.run("What year was Python created?")
    print(f"Active modes: {[m.value for m in result.mode_profile.active_modes]}")
    print(f"Efficiency:   {result.mode_profile.efficiency_ratio:.0%} modes idle")
    print(f"Budget:       {result.mode_profile.budget_summary}")


# 4. Use OpenAI
def use_openai():
    p = Pipeline(model="gpt-4o")
    result = p.run("Explain the CAP theorem")
    print(result.consensus)


# 5. Use local Ollama model
def use_ollama():
    p = Pipeline(model="ollama/llama3")
    result = p.run("What are the tradeoffs of microservices?")
    print(result.consensus)


# 6. Async batch
async def batch():
    p = Pipeline(model="claude-sonnet-4-6")
    results = await p.run_batch_async([
        "What is RAG?",
        "How does RLHF work?",
        "What are the tradeoffs of fine-tuning?",
    ])
    for r in results:
        print(f"Q: {r.query}\nA: {r.consensus}\n")


# 7. Layer callback
def with_callback():
    def on_layer(layer_id, output):
        print(f"  ✓ {layer_id}: {output[:60]}")
    p = Pipeline(model="claude-sonnet-4-6", on_layer_complete=on_layer)
    result = p.run("What makes a great API design?")
    print(f"\nFinal: {result.consensus}")


# 8. Export to JSON
def export_json():
    import json
    p = Pipeline(model="claude-sonnet-4-6")
    result = p.run("Explain gradient descent")
    print(json.dumps(result.to_dict(), indent=2))


# 9. Custom adapter
def custom_adapter():
    from synthex.adapters import BaseAdapter

    class MyAdapter(BaseAdapter):
        @property
        def model_id(self): return "my-model"
        async def complete(self, prompt, max_tokens=512): return "response", 0
        def complete_sync(self, prompt, max_tokens=512):
            import asyncio
            return asyncio.get_event_loop().run_until_complete(self.complete(prompt, max_tokens))

    p = Pipeline(adapter=MyAdapter())
    result = p.run("test")
    print(result.consensus)


if __name__ == "__main__":
    basic()
