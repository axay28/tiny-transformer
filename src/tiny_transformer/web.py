from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from tiny_transformer.train import load_checkpoint


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tiny Transformer Playground</title>
  <style>
    body { margin: 0; font: 16px system-ui, sans-serif; background: #f7f7f4; color: #1b1b1b; }
    main { max-width: 920px; margin: 0 auto; padding: 32px 20px; }
    h1 { margin: 0 0 18px; font-size: 32px; }
    textarea, pre { box-sizing: border-box; width: 100%; border: 1px solid #c8c8c2;
      border-radius: 8px; padding: 14px; background: white; color: #1b1b1b; }
    textarea { min-height: 110px; resize: vertical; }
    pre { min-height: 220px; white-space: pre-wrap; line-height: 1.45; }
    .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin: 14px 0; }
    label { display: grid; gap: 4px; font-size: 13px; }
    input { width: 86px; padding: 8px; border: 1px solid #c8c8c2; border-radius: 6px; }
    button { border: 0; border-radius: 8px; padding: 10px 16px; color: white;
      background: #1f5f5b; font-weight: 700; cursor: pointer; }
  </style>
</head>
<body>
  <main>
    <h1>Tiny Transformer Playground</h1>
    <textarea id="prompt">To be</textarea>
    <div class="controls">
      <label>New tokens <input id="tokens" type="number" min="1" max="500" value="120"></label>
      <label>Temperature <input id="temperature" type="number" min="0.1" step="0.1" value="0.8"></label>
      <label>Top-k <input id="topk" type="number" min="1" value="20"></label>
      <button id="generate">Generate</button>
    </div>
    <pre id="output"></pre>
  </main>
  <script>
    const button = document.getElementById("generate");
    button.addEventListener("click", async () => {
      button.disabled = true;
      document.getElementById("output").textContent = "Generating...";
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          prompt: document.getElementById("prompt").value,
          max_new_tokens: Number(document.getElementById("tokens").value),
          temperature: Number(document.getElementById("temperature").value),
          top_k: Number(document.getElementById("topk").value)
        })
      });
      const payload = await response.json();
      document.getElementById("output").textContent = payload.text || payload.error;
      button.disabled = false;
    });
  </script>
</body>
</html>
"""


def serve_playground(checkpoint: str, host: str, port: int, device: str) -> None:
    model, tokenizer = load_checkpoint(checkpoint, device=device)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(404)
                return
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                prompt = str(payload.get("prompt", ""))
                idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
                out = model.generate(
                    idx,
                    max_new_tokens=int(payload.get("max_new_tokens", 120)),
                    temperature=float(payload.get("temperature", 0.8)),
                    top_k=int(payload.get("top_k", 20)),
                )
                body = json.dumps({"text": tokenizer.decode(out[0].tolist())}).encode("utf-8")
                self._send(200, body, "application/json")
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(400, body, "application/json")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving playground at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped playground server.")
    finally:
        server.server_close()
