from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from tqdm import trange

from tiny_transformer.config import ModelConfig, TrainConfig
from tiny_transformer.data import TextDataset, split_tokens
from tiny_transformer.model import TinyTransformer
from tiny_transformer.tokenizer import BytePairTokenizer, CharTokenizer, Tokenizer, tokenizer_from_dict


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


def save_loss_history(history: list[dict[str, float | int]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def train_from_text(
    text: str,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    device: str = "cpu",
    tokenizer_name: str = "char",
    bpe_vocab_size: int = 256,
    resume_path: str | None = None,
) -> TinyTransformer:
    train_config = train_config or TrainConfig()
    torch.manual_seed(train_config.seed)

    resume_payload = torch.load(resume_path, map_location=device) if resume_path else None
    tokenizer = (
        tokenizer_from_dict(resume_payload["tokenizer"])
        if resume_payload is not None
        else train_tokenizer(text, tokenizer_name, bpe_vocab_size)
    )
    token_ids = tokenizer.encode(text)
    train_ids, val_ids = split_tokens(token_ids)

    if resume_payload is not None:
        model_config = ModelConfig(**resume_payload["model_config"])
    elif model_config is None:
        model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    else:
        model_config = ModelConfig(**{**model_config.to_dict(), "vocab_size": tokenizer.vocab_size})

    train_data = TextDataset(train_ids, model_config.block_size, device)
    val_data = TextDataset(val_ids, model_config.block_size, device)
    model = TinyTransformer(model_config).to(device)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    if resume_payload is not None and "optimizer_state" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
    start_step = int(resume_payload.get("step", 0)) if resume_payload is not None else 0
    device_type = "cuda" if device.startswith("cuda") else "mps" if device == "mps" else "cpu"
    amp_enabled = train_config.use_amp and device_type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device_type == "cuda")
    loss_history: list[dict[str, float | int]] = []

    progress = trange(start_step, train_config.max_steps, desc="training", leave=True)
    for step in progress:
        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:
            losses = estimate_loss(
                model, train_data, val_data, train_config.batch_size, train_config.eval_batches
            )
            progress.set_postfix(train=f"{losses['train']:.3f}", val=f"{losses['val']:.3f}")
            loss_history.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                }
            )

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_config.grad_accum_steps):
            x, y = train_data.get_batch(train_config.batch_size)
            with torch.autocast(device_type=device_type, enabled=amp_enabled):
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("Expected a loss during training")
                loss = loss / train_config.grad_accum_steps
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if train_config.loss_history_path:
        save_loss_history(loss_history, train_config.loss_history_path)
    save_checkpoint(
        model,
        tokenizer,
        train_config.output_path,
        optimizer=optimizer,
        step=train_config.max_steps,
        train_config=train_config,
    )
    return model


def train_tokenizer(text: str, tokenizer_name: str, bpe_vocab_size: int) -> Tokenizer:
    if tokenizer_name == "char":
        return CharTokenizer.train(text)
    if tokenizer_name == "bpe":
        return BytePairTokenizer.train(text, vocab_size=bpe_vocab_size)
    raise ValueError("tokenizer_name must be 'char' or 'bpe'")


def save_checkpoint(
    model: TinyTransformer,
    tokenizer: Tokenizer,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    train_config: TrainConfig | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "step": step,
            "parameter_count": model.parameter_count(),
            "weight_tied": model.lm_head.weight is model.token_embedding.weight,
        }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if train_config is not None:
        payload["train_config"] = train_config.to_dict()
    torch.save(payload, output)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[TinyTransformer, Tokenizer]:
    payload = torch.load(path, map_location=device)
    tokenizer = tokenizer_from_dict(payload["tokenizer"])
    config = ModelConfig(**payload["model_config"])
    model = TinyTransformer(config).to(device)
    if not payload.get("weight_tied", False):
        # Checkpoints created before weight tying stored independent embedding and
        # output matrices. Preserve that behavior so older demos remain compatible.
        model.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, tokenizer


def evaluate_checkpoint(
    path: str,
    text: str,
    batch_size: int = 32,
    eval_batches: int = 20,
    device: str = "cpu",
) -> dict[str, float | int]:
    model, tokenizer = load_checkpoint(path, device=device)
    token_ids = tokenizer.encode(text)
    train_ids, val_ids = split_tokens(token_ids)
    train_data = TextDataset(train_ids, model.config.block_size, device)
    val_data = TextDataset(val_ids, model.config.block_size, device)
    losses = estimate_loss(model, train_data, val_data, batch_size, eval_batches)
    return {
        "train_loss": losses["train"],
        "validation_loss": losses["val"],
        "validation_perplexity": math.exp(min(losses["val"], 20)),
        "parameter_count": model.parameter_count(),
    }
