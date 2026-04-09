"""
SYNTHEX — Dual-Stream Inference Pipeline
Created by Sharaf Samiur Rahman — github.com/impulsiveaura

Quick start:
    from synthex import Pipeline
    p = Pipeline(model="claude-sonnet-4-6")
    result = p.run("What is the future of energy storage?")
    print(result.consensus)
"""
from .pipeline import Pipeline
from .types import SynthexResult, LayerOutput, FusionOutput, ModeProfile, InferenceMode, PipelineStatus
from .adapters import BaseAdapter, AnthropicAdapter, OpenAIAdapter, OllamaAdapter, create_adapter

__version__ = "0.1.0"
__author__  = "Sharaf Samiur Rahman"
__license__ = "MIT"

__all__ = [
    "Pipeline",
    "SynthexResult", "LayerOutput", "FusionOutput", "ModeProfile",
    "InferenceMode", "PipelineStatus",
    "BaseAdapter", "AnthropicAdapter", "OpenAIAdapter", "OllamaAdapter", "create_adapter",
]
