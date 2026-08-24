import torch

from tiny_transformer.config import ModelConfig, TrainConfig
from tiny_transformer.train import evaluate_checkpoint, train_from_text


def test_checkpoint_can_resume_and_report_perplexity(tmp_path) -> None:
    text = "the tiny model learns from repeated text. " * 20
    checkpoint = tmp_path / "model.pt"
    model_config = ModelConfig(
        vocab_size=1, block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=0.0
    )
    first = TrainConfig(
        batch_size=2,
        max_steps=1,
        eval_interval=1,
        eval_batches=1,
        output_path=str(checkpoint),
    )
    train_from_text(text, model_config=model_config, train_config=first)

    resumed = TrainConfig(
        batch_size=2,
        max_steps=2,
        eval_interval=1,
        eval_batches=1,
        output_path=str(checkpoint),
    )
    train_from_text(text, train_config=resumed, resume_path=str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu")
    metrics = evaluate_checkpoint(str(checkpoint), text, batch_size=2, eval_batches=1)

    assert payload["step"] == 2
    assert "optimizer_state" in payload
    assert metrics["validation_perplexity"] > 0
    assert metrics["parameter_count"] == payload["parameter_count"]
