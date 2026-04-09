"""
SYNTHEX CLI
Created by Sharaf Samiur Rahman — github.com/impulsiveaura

Usage:
    synthex "your query here"
    synthex "your query" --model gpt-4o
    synthex "your query" --output json
    synthex --batch queries.txt
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import os


def build_parser():
    p = argparse.ArgumentParser(
        prog="synthex",
        description="SYNTHEX — Dual-Stream Inference Pipeline by Sharaf Samiur Rahman",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  synthex "What is the future of battery storage?"
  synthex "Explain attention mechanisms" --model gpt-4o
  synthex "What is consciousness?" --model ollama/llama3 --verbose
  synthex "Best API design practices" --output json
  synthex --batch queries.txt
        """,
    )
    p.add_argument("query",    nargs="?",  help="Query to process")
    p.add_argument("--model",  "-m",       default="claude-sonnet-4-6")
    p.add_argument("--output", "-o",       choices=["text", "json", "layers"], default="text")
    p.add_argument("--verbose","-v",       action="store_true")
    p.add_argument("--batch",  "-b",       help="Path to .txt file with one query per line")
    p.add_argument("--api-key",            help="API key")
    p.add_argument("--version",            action="store_true")
    return p


def print_result(result):
    W = 70
    def rule(c="─"): print(c * W)
    def section(label, content, color=""):
        rule()
        if color and sys.stdout.isatty():
            print(f"\033[{color}m  {label}\033[0m")
        else:
            print(f"  {label}")
        rule("·")
        for line in content.splitlines():
            print(f"  {line}")
        print()

    rule("═")
    print("  SYNTHEX · DUAL-STREAM INFERENCE PIPELINE")
    print("  github.com/impulsiveaura/synthex")
    rule("═")
    print(f"  Query   : {result.query}")
    print(f"  Model   : {result.model}")
    print(f"  Latency : {result.latency_seconds}s  |  Tokens: {result.total_tokens}")
    if result.mode_profile:
        modes = " · ".join(m.value for m in result.mode_profile.active_modes)
        print(f"  Modes   : {modes}")
    print()
    section("TOKENIZATION · LITERAL",  result.tokenization_literal.content,  "36")
    section("TOKENIZATION · LATENT",   result.tokenization_latent.content,   "35")
    section("EMBEDDING FUSION",        result.embedding_fusion.content,      "33")
    section("INFERENCE · TEMPORAL",    result.inference_temporal.content,    "31")
    section("INFERENCE · GENERATIVE",  result.inference_generative.content,  "32")
    section("SIGNAL DISTILLATION",     result.signal_distillation.content,   "34")
    rule("═")
    print("  CONSENSUS OUTPUT")
    rule("═")
    for line in result.consensus.splitlines():
        print(f"  {line}")
    rule("═")


async def _run(args):
    from synthex import Pipeline
    kwargs = {}
    if args.api_key:
        kwargs["api_key"] = args.api_key

    def on_layer(layer_id, output):
        if args.verbose:
            print(f"  \033[90m[{layer_id}]\033[0m {output[:60].replace(chr(10),' ')}...", file=sys.stderr)

    pipeline = Pipeline(model=args.model, on_layer_complete=on_layer if args.verbose else None, **kwargs)

    if args.batch:
        with open(args.batch) as f:
            queries = [l.strip() for l in f if l.strip()]
        results = await pipeline.run_batch_async(queries)
        if args.output == "json":
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            for r in results:
                print_result(r)
                print()
    else:
        if not args.query:
            print("Error: provide a query or use --batch", file=sys.stderr)
            sys.exit(1)
        result = await pipeline.run_async(args.query)
        if args.output == "json":
            print(json.dumps(result.to_dict(), indent=2))
        elif args.output == "layers":
            for k, v in result.to_dict()["layers"].items():
                print(f"\n[{k.upper()}]\n{v}")
            print(f"\n[CONSENSUS]\n{result.consensus}")
        else:
            print_result(result)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        from synthex import __version__
        print(f"synthex {__version__} by Sharaf Samiur Rahman")
        return
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if os.environ.get("SYNTHEX_DEBUG"):
            import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
