from __future__ import annotations

from html import escape
from pathlib import Path

import torch

from tiny_transformer.model import TinyTransformer
from tiny_transformer.tokenizer import Tokenizer


@torch.no_grad()
def save_attention_heatmap(
    model: TinyTransformer,
    tokenizer: Tokenizer,
    idx: torch.Tensor,
    output_path: str,
    layer: int = -1,
    head: int = 0,
) -> None:
    attentions = model.attention_maps(idx)
    if not attentions:
        raise ValueError("Model did not return attention maps")

    selected = attentions[layer][0]
    if head < 0 or head >= selected.shape[0]:
        raise ValueError(f"head must be between 0 and {selected.shape[0] - 1}")

    weights = selected[head].detach().cpu()
    token_ids = idx[0].detach().cpu().tolist()
    labels = [_display_token(tokenizer.id_to_token(token_id)) for token_id in token_ids]
    svg = _attention_svg(weights, labels, layer=layer, head=head)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def _attention_svg(weights: torch.Tensor, labels: list[str], layer: int, head: int) -> str:
    cell = 24
    margin_left = 120
    margin_top = 96
    size = len(labels)
    width = margin_left + size * cell + 24
    height = margin_top + size * cell + 40
    max_weight = max(float(weights.max()), 1e-9)

    cells = []
    for row in range(size):
        for col in range(size):
            value = float(weights[row, col]) / max_weight
            color = 255 - int(value * 210)
            cells.append(
                f'<rect x="{margin_left + col * cell}" y="{margin_top + row * cell}" '
                f'width="{cell}" height="{cell}" fill="rgb({color},{color},255)">'
                f"<title>{escape(labels[row])} attends to {escape(labels[col])}: "
                f"{float(weights[row, col]):.3f}</title></rect>"
            )

    x_labels = [
        f'<text x="{margin_left + idx * cell + 12}" y="{margin_top - 10}" '
        f'transform="rotate(-45 {margin_left + idx * cell + 12},{margin_top - 10})">'
        f"{escape(label)}</text>"
        for idx, label in enumerate(labels)
    ]
    y_labels = [
        f'<text x="{margin_left - 8}" y="{margin_top + idx * cell + 16}" text-anchor="end">'
        f"{escape(label)}</text>"
        for idx, label in enumerate(labels)
    ]

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            "<style>text{font:12px system-ui,sans-serif} rect{stroke:#fff;stroke-width:1}</style>",
            f'<text x="16" y="28" style="font-size:18px;font-weight:700">'
            f"Attention heatmap: layer {layer}, head {head}</text>",
            *x_labels,
            *y_labels,
            *cells,
            "</svg>",
        ]
    )


def _display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "space")
