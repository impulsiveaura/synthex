"""
SYNTHEX test suite.
Created by Sharaf Samiur Rahman — github.com/impulsiveaura
Run: pytest tests/
"""
from __future__ import annotations
import pytest
from synthex import Pipeline, SynthexResult, InferenceMode
from synthex.adapters import BaseAdapter
from synthex.types import ModeProfile
from synthex.pipeline import _parse_mode_profile


class MockAdapter(BaseAdapter):
    """Deterministic mock adapter — no real API calls needed for tests."""
    def __init__(self):
        self._call_count = 0

    @property
    def model_id(self): return "mock/test"

    async def complete(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        self._call_count += 1
        p = prompt.upper()
        if "LITERAL TOKENIZATION" in p:
            return "The query explicitly asks about battery storage technology.", 80
        if "LATENT TOKENIZATION" in p:
            return "The underlying need is understanding long-term energy infrastructure planning.", 80
        if "EMBEDDING FUSION" in p:
            return "Both streams converge on energy storage as a strategic infrastructure question.", 90
        if "TEMPORAL INFERENCE" in p:
            return "Energy storage follows consistent cost curves and adoption S-curves across history.", 90
        if "GENERATIVE INFERENCE" in p:
            return "Battery storage parallels the evolution of computing memory — each generation enables new use cases.", 90
        if "SIGNAL DISTILLATION" in p:
            return "Solid-state batteries and grid-scale deployment represent the most confident signal.", 90
        if "MODE SELECTOR" in p:
            return "LITERAL: ACTIVE\nLATENT: ACTIVE\nTEMPORAL: ACTIVE\nGENERATIVE: ACTIVE\nBUDGET: All modes required.", 80
        if "CONSENSUS" in p:
            return "Battery storage is advancing rapidly. Solid-state batteries are the next inflection point.", 100
        return f"Mock response {self._call_count}", 50

    def complete_sync(self, prompt: str, max_tokens: int = 512) -> tuple[str, int]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.complete(prompt, max_tokens))


@pytest.mark.asyncio
async def test_returns_synthex_result():
    result = await Pipeline(adapter=MockAdapter()).run_async("What is the future of battery storage?")
    assert isinstance(result, SynthexResult)

@pytest.mark.asyncio
async def test_consensus_populated():
    result = await Pipeline(adapter=MockAdapter()).run_async("test query")
    assert result.consensus and len(result.consensus) > 10

@pytest.mark.asyncio
async def test_all_layers_populated():
    result = await Pipeline(adapter=MockAdapter()).run_async("test")
    assert result.tokenization_literal.content
    assert result.tokenization_latent.content
    assert result.embedding_fusion.content
    assert result.inference_temporal.content
    assert result.inference_generative.content
    assert result.signal_distillation.content

@pytest.mark.asyncio
async def test_mode_profile_parsed():
    result = await Pipeline(adapter=MockAdapter()).run_async("test")
    assert result.mode_profile is not None
    assert len(result.mode_profile.active_modes) > 0

@pytest.mark.asyncio
async def test_model_id_set():
    result = await Pipeline(adapter=MockAdapter()).run_async("test")
    assert result.model == "mock/test"

@pytest.mark.asyncio
async def test_latency_recorded():
    result = await Pipeline(adapter=MockAdapter()).run_async("test")
    assert result.latency_seconds >= 0

@pytest.mark.asyncio
async def test_to_dict():
    result = await Pipeline(adapter=MockAdapter()).run_async("test")
    d = result.to_dict()
    assert all(k in d for k in ["query", "consensus", "layers", "mode_profile", "meta"])

@pytest.mark.asyncio
async def test_batch():
    results = await Pipeline(adapter=MockAdapter()).run_batch_async(["q1", "q2", "q3"])
    assert len(results) == 3
    assert all(isinstance(r, SynthexResult) for r in results)

@pytest.mark.asyncio
async def test_callback_fires():
    fired = []
    await Pipeline(adapter=MockAdapter(), on_layer_complete=lambda l, o: fired.append(l)).run_async("test")
    assert len(fired) >= 6

def test_parse_mode_all_active():
    p = _parse_mode_profile("LITERAL: ACTIVE\nLATENT: ACTIVE\nTEMPORAL: ACTIVE\nGENERATIVE: ACTIVE\nBUDGET: All.")
    assert len(p.active_modes) == 4

def test_parse_mode_some_idle():
    p = _parse_mode_profile("LITERAL: ACTIVE\nLATENT: IDLE\nTEMPORAL: ACTIVE\nGENERATIVE: IDLE\nBUDGET: Simple.")
    assert InferenceMode.LITERAL in p.active_modes
    assert InferenceMode.LATENT  in p.idle_modes

def test_parse_mode_fallback():
    p = _parse_mode_profile("unparseable nonsense")
    assert len(p.active_modes) == 4

def test_efficiency_ratio():
    p = _parse_mode_profile("LITERAL: ACTIVE\nLATENT: IDLE\nTEMPORAL: IDLE\nGENERATIVE: IDLE\nBUDGET: Simple.")
    assert p.efficiency_ratio == 0.75
