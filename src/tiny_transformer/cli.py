from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tiny_transformer.config import ModelConfig, TrainConfig
from tiny_transformer.train import load_checkpoint, train_from_text


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

    generate = subparsers.add_parser("generate", help="Generate text from a trained checkpoint.")
    generate.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    generate.add_argument("--prompt", default="\n", help="Prompt text.")
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--max-new-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=20)

    return parser


def train_command(args: argparse.Namespace) -> None:
    text = Path(args.data).read_text(encoding="utf-8")
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
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
    train_from_text(text, model_config=model_config, train_config=train_config, device=args.device)
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "generate":
        generate_command(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
