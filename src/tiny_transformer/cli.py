from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tiny_transformer.config import ModelConfig, TrainConfig
from tiny_transformer.train import load_checkpoint, train_from_text
from tiny_transformer.visualize import save_attention_heatmap
from tiny_transformer.web import serve_playground


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and sample a tiny GPT-style Transformer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a model on a plain-text corpus.")
    train.add_argument("--data", required=True, help="Path to a UTF-8 text file.")
    train.add_argument("--output", default="runs/tiny-transformer.pt", help="Checkpoint path.")
    train.add_argument("--device", default="cpu", help="Device such as cpu, cuda, or mps.")
    train.add_argument("--steps", type=int, default=1_000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--block-size", type=int, default=32)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--embedding", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    train.add_argument("--bpe-vocab-size", type=int, default=256)
    train.add_argument("--grad-accum-steps", type=int, default=1)
    train.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA or MPS.")

    generate = subparsers.add_parser("generate", help="Generate text from a trained checkpoint.")
    generate.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    generate.add_argument("--prompt", default="\n", help="Prompt text.")
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--max-new-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=20)

    attention = subparsers.add_parser("attention", help="Export an attention heatmap SVG.")
    attention.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    attention.add_argument("--prompt", required=True, help="Prompt text to inspect.")
    attention.add_argument("--output", default="runs/attention.svg", help="SVG output path.")
    attention.add_argument("--device", default="cpu")
    attention.add_argument("--layer", type=int, default=-1, help="Layer index to visualize.")
    attention.add_argument("--head", type=int, default=0, help="Attention head index to visualize.")

    serve = subparsers.add_parser("serve", help="Launch a local text-generation playground.")
    serve.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="cpu")

    return parser


def train_command(args: argparse.Namespace) -> None:
    text = Path(args.data).read_text(encoding="utf-8")
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.amp,
        output_path=args.output,
    )
    model_config = ModelConfig(
        vocab_size=1,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embedding,
        dropout=args.dropout,
    )
    train_from_text(
        text,
        model_config=model_config,
        train_config=train_config,
        device=args.device,
        tokenizer_name=args.tokenizer,
        bpe_vocab_size=args.bpe_vocab_size,
    )
    print(f"Saved checkpoint to {args.output}")


def generate_command(args: argparse.Namespace) -> None:
    model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    encoded = tokenizer.encode(args.prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=args.device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


def attention_command(args: argparse.Namespace) -> None:
    model, tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    encoded = tokenizer.encode(args.prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=args.device)
    save_attention_heatmap(
        model=model,
        tokenizer=tokenizer,
        idx=idx,
        output_path=args.output,
        layer=args.layer,
        head=args.head,
    )
    print(f"Saved attention heatmap to {args.output}")


def serve_command(args: argparse.Namespace) -> None:
    serve_playground(
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "attention":
        attention_command(args)
    elif args.command == "serve":
        serve_command(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
