"""
SYNTHEX dual-stream inference pipeline — core execution engine.
Created by Sharaf Samiur Rahman — github.com/impulsiveaura
"""
from __future__ import annotations
import asyncio
import time
from typing import Optional, Callable

from ..types import SynthexResult, LayerOutput, FusionOutput, ModeProfile, InferenceMode
from ..adapters import BaseAdapter, create_adapter
from .. import prompts


class Pipeline:
    """
    SYNTHEX dual-stream inference pipeline.

    Quick start:
        from synthex import Pipeline
        p = Pipeline(model="claude-sonnet-4-6")
        result = p.run("What is the future of battery storage?")
        print(result.consensus)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        adapter: Optional[BaseAdapter] = None,
        max_tokens_per_layer: int = 400,
        on_layer_complete: Optional[Callable[[str, str], None]] = None,
        verbose: bool = False,
        **adapter_kwargs,
    ):
        self._adapter = adapter or create_adapter(model, **adapter_kwargs)
        self._max_tokens = max_tokens_per_layer
        self._on_layer_complete = on_layer_complete
        self._verbose = verbose

    def run(self, query: str) -> SynthexResult:
        """Synchronous pipeline run."""
        return asyncio.get_event_loop().run_until_complete(self.run_async(query))

    async def run_async(self, query: str) -> SynthexResult:
        """Async pipeline run — preferred for production."""
        t0 = time.perf_counter()
        collected: dict[str, str] = {}
        total_tokens = 0

        def _log(layer_id: str, content: str):
            if self._verbose:
                print(f"  [{layer_id}] {content[:80].replace(chr(10), ' ')}...")
            if self._on_layer_complete:
                self._on_layer_complete(layer_id, content)

        # Pipeline A — dual tokenization (parallel)
        tok_lit_text, tok_lat_text, tok_tokens = await self._parallel(
            prompts.tokenization_literal(query),
            prompts.tokenization_latent(query),
        )
        total_tokens += tok_tokens
        collected["tokenization_literal"] = tok_lit_text
        collected["tokenization_latent"]  = tok_lat_text
        _log("tokenization_literal", tok_lit_text)
        _log("tokenization_latent",  tok_lat_text)

        # Pipeline A — embedding fusion
        emb_text, emb_tokens = await self._adapter.complete(
            prompts.embedding_fusion(query, collected), self._max_tokens)
        total_tokens += emb_tokens
        collected["embedding_fusion"] = emb_text
        _log("embedding_fusion", emb_text)

        # Pipeline B — dual inference core (parallel)
        inf_tmp_text, inf_gen_text, inf_tokens = await self._parallel(
            prompts.inference_temporal(query, collected),
            prompts.inference_generative(query, collected),
        )
        total_tokens += inf_tokens
        collected["inference_temporal"]   = inf_tmp_text
        collected["inference_generative"] = inf_gen_text
        _log("inference_temporal",   inf_tmp_text)
        _log("inference_generative", inf_gen_text)

        # Pipeline B — signal distillation
        sig_text, sig_tokens = await self._adapter.complete(
            prompts.signal_distillation(query, collected), self._max_tokens)
        total_tokens += sig_tokens
        collected["signal_distillation"] = sig_text
        _log("signal_distillation", sig_text)

        # Mode selector
        sel_text, sel_tokens = await self._adapter.complete(
            prompts.mode_selector(query, collected), 300)
        total_tokens += sel_tokens
        collected["mode_selector"] = sel_text
        _log("mode_selector", sel_text)
        mode_profile = _parse_mode_profile(sel_text)

        # Consensus layer
        con_text, con_tokens = await self._adapter.complete(
            prompts.consensus(query, collected), self._max_tokens)
        total_tokens += con_tokens
        _log("consensus", con_text)

        return SynthexResult(
            query=query,
            consensus=con_text,
            tokenization_literal=LayerOutput("tokenization_literal", tok_lit_text, InferenceMode.LITERAL),
            tokenization_latent=LayerOutput("tokenization_latent",   tok_lat_text, InferenceMode.LATENT),
            embedding_fusion=FusionOutput("embedding_fusion", emb_text, tok_lit_text, tok_lat_text),
            inference_temporal=LayerOutput("inference_temporal",     inf_tmp_text, InferenceMode.TEMPORAL),
            inference_generative=LayerOutput("inference_generative", inf_gen_text, InferenceMode.GENERATIVE),
            signal_distillation=FusionOutput("signal_distillation",  sig_text, inf_tmp_text, inf_gen_text),
            mode_profile=mode_profile,
            latency_seconds=round(time.perf_counter() - t0, 3),
            total_tokens=total_tokens,
            model=self._adapter.model_id,
        )

    async def run_batch_async(self, queries: list[str]) -> list[SynthexResult]:
        return await asyncio.gather(*[self.run_async(q) for q in queries])

    def run_batch(self, queries: list[str]) -> list[SynthexResult]:
        return asyncio.get_event_loop().run_until_complete(self.run_batch_async(queries))

    async def _parallel(self, prompt_a: str, prompt_b: str) -> tuple[str, str, int]:
        results = await asyncio.gather(
            self._adapter.complete(prompt_a, self._max_tokens),
            self._adapter.complete(prompt_b, self._max_tokens),
        )
        return results[0][0], results[1][0], results[0][1] + results[1][1]


def _parse_mode_profile(raw: str) -> ModeProfile:
    active, idle = [], []
    upper = raw.upper()
    budget = ""
    for mode in InferenceMode:
        key = mode.value + ":"
        if key in upper:
            idx = upper.index(key)
            seg = upper[idx: idx + 30]
            (active if "ACTIVE" in seg else idle).append(mode)
    for line in raw.splitlines():
        if line.upper().startswith("BUDGET:"):
            budget = line[7:].strip()
            break
    if not active:
        active = list(InferenceMode)
        idle = []
    return ModeProfile(active_modes=active, idle_modes=idle, budget_summary=budget, raw_output=raw)
