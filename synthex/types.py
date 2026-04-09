"""
SYNTHEX core types and data models.
Created by Sharaf Samiur Rahman — github.com/impulsiveaura
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InferenceMode(str, Enum):
    LITERAL    = "LITERAL"
    LATENT     = "LATENT"
    TEMPORAL   = "TEMPORAL"
    GENERATIVE = "GENERATIVE"


class PipelineStatus(str, Enum):
    IDLE   = "idle"
    ACTIVE = "active"
    DONE   = "done"
    FAILED = "failed"


@dataclass
class LayerOutput:
    layer_id: str
    content: str
    mode: Optional[InferenceMode] = None
    confidence: float = 1.0
    tokens_used: int = 0

    def __bool__(self):
        return bool(self.content and self.content != "[layer fault]")


@dataclass
class FusionOutput:
    layer_id: str
    content: str
    stream_a: str = ""
    stream_b: str = ""
    coherence_score: float = 0.0
    tokens_used: int = 0

    def __bool__(self):
        return bool(self.content)


@dataclass
class ModeProfile:
    active_modes: list[InferenceMode]
    idle_modes: list[InferenceMode]
    budget_summary: str
    raw_output: str

    @property
    def mode_count(self) -> int:
        return len(self.active_modes)

    @property
    def efficiency_ratio(self) -> float:
        total = len(InferenceMode)
        return 1.0 - (self.mode_count / total)


@dataclass
class SynthexResult:
    """
    The complete output of a SYNTHEX pipeline run.
    Access result.consensus for the final answer.
    """
    query: str
    consensus: str

    tokenization_literal:  LayerOutput  = field(default_factory=lambda: LayerOutput("", ""))
    tokenization_latent:   LayerOutput  = field(default_factory=lambda: LayerOutput("", ""))
    embedding_fusion:      FusionOutput = field(default_factory=lambda: FusionOutput("", ""))
    inference_temporal:    LayerOutput  = field(default_factory=lambda: LayerOutput("", ""))
    inference_generative:  LayerOutput  = field(default_factory=lambda: LayerOutput("", ""))
    signal_distillation:   FusionOutput = field(default_factory=lambda: FusionOutput("", ""))

    mode_profile:     Optional[ModeProfile] = None
    latency_seconds:  float = 0.0
    total_tokens:     int   = 0
    model:            str   = ""

    def __repr__(self) -> str:
        modes = [m.value for m in (self.mode_profile.active_modes if self.mode_profile else [])]
        return (
            f"SynthexResult(\n"
            f"  query={self.query!r},\n"
            f"  consensus={self.consensus[:120]!r}{'...' if len(self.consensus) > 120 else ''},\n"
            f"  active_modes={modes},\n"
            f"  latency={self.latency_seconds:.2f}s,\n"
            f"  tokens={self.total_tokens}\n"
            f")"
        )

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "consensus": self.consensus,
            "layers": {
                "tokenization_literal":  self.tokenization_literal.content,
                "tokenization_latent":   self.tokenization_latent.content,
                "embedding_fusion":      self.embedding_fusion.content,
                "inference_temporal":    self.inference_temporal.content,
                "inference_generative":  self.inference_generative.content,
                "signal_distillation":   self.signal_distillation.content,
            },
            "mode_profile": {
                "active": [m.value for m in (self.mode_profile.active_modes if self.mode_profile else [])],
                "idle":   [m.value for m in (self.mode_profile.idle_modes   if self.mode_profile else [])],
                "budget": self.mode_profile.budget_summary if self.mode_profile else "",
            },
            "meta": {
                "latency_seconds": self.latency_seconds,
                "total_tokens":    self.total_tokens,
                "model":           self.model,
            }
        }
