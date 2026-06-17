from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tiny_transformer.train import load_checkpoint


CHECKPOINT = Path("demo/tiny-transformer-demo.pt")
DEVICE = "cpu"


model, tokenizer = load_checkpoint(str(CHECKPOINT), device=DEVICE)


def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    if not prompt:
        prompt = "\n"
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())


with gr.Blocks(title="Tiny Transformer") as demo:
    gr.Markdown("# Tiny Transformer")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(value="To be", label="Prompt", lines=5)
            max_new_tokens = gr.Slider(8, 240, value=120, step=1, label="New tokens")
            temperature = gr.Slider(0.2, 1.5, value=0.35, step=0.05, label="Temperature")
            top_k = gr.Slider(1, 30, value=3, step=1, label="Top-k")
            button = gr.Button("Generate", variant="primary")
        output = gr.Textbox(label="Output", lines=16)

    gr.Examples(
        examples=[
            ["To be", 120, 0.35, 3],
            ["Attention", 120, 0.35, 3],
            ["The model", 120, 0.35, 3],
        ],
        inputs=[prompt, max_new_tokens, temperature, top_k],
    )

    button.click(
        generate_text,
        inputs=[prompt, max_new_tokens, temperature, top_k],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
