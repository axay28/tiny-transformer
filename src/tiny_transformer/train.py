from __future__ import annotations

from pathlib import Path

import torch
from tqdm import trange

from tiny_transformer.config import ModelConfig, TrainConfig
from tiny_transformer.data import TextDataset, split_tokens
from tiny_transformer.model import TinyTransformer
from tiny_transformer.tokenizer import CharTokenizer


@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    train_data: TextDataset,
    val_data: TextDataset,
    batch_size: int,
    eval_batches: int,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split, dataset in {"train": train_data, "val": val_data}.items():
        split_losses = []
        for _ in range(eval_batches):
            x, y = dataset.get_batch(batch_size)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Expected a loss during evaluation")
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def train_from_text(
    text: str,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    device: str = "cpu",
) -> TinyTransformer:
    train_config = train_config or TrainConfig()
    torch.manual_seed(train_config.seed)

    tokenizer = CharTokenizer.train(text)
    token_ids = tokenizer.encode(text)
    train_ids, val_ids = split_tokens(token_ids)

    if model_config is None:
        model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    else:
        model_config = ModelConfig(**{**model_config.to_dict(), "vocab_size": tokenizer.vocab_size})

    train_data = TextDataset(train_ids, model_config.block_size, device)
    val_data = TextDataset(val_ids, model_config.block_size, device)
    model = TinyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    progress = trange(train_config.max_steps, desc="training", leave=True)
    for step in progress:
        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:
            losses = estimate_loss(
                model, train_data, val_data, train_config.batch_size, train_config.eval_batches
            )
            progress.set_postfix(train=f"{losses['train']:.3f}", val=f"{losses['val']:.3f}")

        x, y = train_data.get_batch(train_config.batch_size)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Expected a loss during training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(model, tokenizer, train_config.output_path)
    return model


def save_checkpoint(model: TinyTransformer, tokenizer: CharTokenizer, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
        },
        output,
    )


def load_checkpoint(path: str, device: str = "cpu") -> tuple[TinyTransformer, CharTokenizer]:
    payload = torch.load(path, map_location=device)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    config = ModelConfig(**payload["model_config"])
    model = TinyTransformer(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, tokenizer
