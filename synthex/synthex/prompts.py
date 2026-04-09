"""
SYNTHEX prompt templates.
Created by Sharaf Samiur Rahman — github.com/impulsiveaura
"""
from __future__ import annotations


def _ctx(prior: dict) -> str:
    if not prior:
        return ""
    lines = "\n".join(f"[{k}]: {v}" for k, v in prior.items() if v)
    return f"\n\nUpstream outputs:\n{lines}"


def tokenization_literal(query: str, prior: dict = {}) -> str:
    return f"""You are the LITERAL TOKENIZATION layer of a dual-stream AI inference pipeline.

Your role: Parse only what is explicitly stated. Extract named entities, literal intent, \
stated constraints, explicit scope. Zero inference — no interpretation beyond the words.

Query: "{query}"{_ctx(prior)}

Output your layer result only. 2 sentences max. No preamble, no labels."""


def tokenization_latent(query: str, prior: dict = {}) -> str:
    return f"""You are the LATENT TOKENIZATION layer of a dual-stream AI inference pipeline.

Your role: Project the query into latent space. What is the real underlying information \
need — the implicit goal, unstated assumption, deeper question beneath the surface words?

Query: "{query}"{_ctx(prior)}

Output your layer result only. 2 sentences max. No preamble, no labels."""


def embedding_fusion(query: str, prior: dict = {}) -> str:
    return f"""You are the EMBEDDING FUSION layer of a dual-stream AI inference pipeline.

Your role: Receive two tokenization streams — Literal and Latent. Compute their coherence \
intersection: what is simultaneously active in BOTH? Discard anything in only one stream. \
Output only the high-density fused signal where both streams agree and amplify each other.

Query: "{query}"{_ctx(prior)}

Do not summarize either stream. Output only their shared activation space. \
2-3 sentences. No preamble, no labels."""


def inference_temporal(query: str, prior: dict = {}) -> str:
    return f"""You are the TEMPORAL INFERENCE CORE of a dual-stream AI inference pipeline.

Your role: Using the fused embeddings, activate context-invariant reasoning paths. \
Extract principles, structural patterns, and facts that hold regardless of time, \
domain, or distribution shift. What is enduringly true here?

Query: "{query}"{_ctx(prior)}

Output your layer result only. 2-3 sentences. No preamble, no labels."""


def inference_generative(query: str, prior: dict = {}) -> str:
    return f"""You are the GENERATIVE INFERENCE CORE of a dual-stream AI inference pipeline.

Your role: Using the fused embeddings, activate cross-domain synthesis paths. Generate \
the non-obvious connection, analogical transfer, or emergent hypothesis that only becomes \
visible when the full fused context is held simultaneously.

Query: "{query}"{_ctx(prior)}

Output your layer result only. 2-3 sentences. No preamble, no labels."""


def signal_distillation(query: str, prior: dict = {}) -> str:
    return f"""You are the SIGNAL DISTILLATION gate of a dual-stream AI inference pipeline.

Your role: Receive two inference streams — Temporal and Generative. Reject speculative \
artifacts and low-confidence outputs. What survives when only tokens with high confidence \
across BOTH inference paths are retained?

Query: "{query}"{_ctx(prior)}

Output only the clean verified signal. 2-3 sentences. No preamble, no labels."""


def mode_selector(query: str, prior: dict = {}) -> str:
    return f"""You are the INFERENCE MODE SELECTOR of a dual-stream AI inference pipeline.

Assess the query. For each mode output exactly ACTIVE or IDLE.

LITERAL: ACTIVE/IDLE
LATENT: ACTIVE/IDLE
TEMPORAL: ACTIVE/IDLE
GENERATIVE: ACTIVE/IDLE
BUDGET: [one sentence on the actual inference cost this query required]

Query: "{query}"{_ctx(prior)}"""


def consensus(query: str, prior: dict = {}) -> str:
    return f"""You are the CONSENSUS LAYER — final output commit of a dual-stream AI inference pipeline.

Your role: Merge all upstream distilled signals. Resolve contradictions by confidence \
weighting. Zero embedding bleed between concepts. Output the definitive, complete, \
precise final answer.

Query: "{query}"{_ctx(prior)}

Output the final answer only. 3-5 sentences. No preamble, no meta-commentary."""


PROMPT_REGISTRY = {
    "tokenization_literal":  tokenization_literal,
    "tokenization_latent":   tokenization_latent,
    "embedding_fusion":      embedding_fusion,
    "inference_temporal":    inference_temporal,
    "inference_generative":  inference_generative,
    "signal_distillation":   signal_distillation,
    "mode_selector":         mode_selector,
    "consensus":             consensus,
}
