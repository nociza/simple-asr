"""Simple Flask-based GUI for managing Simple ASR settings."""

from __future__ import annotations

import threading
import webbrowser
from typing import Optional

from flask import Flask, redirect, render_template_string, request, url_for

from .app import AppConfig
from .config import ConfigManager
from .vocabulary import DEFAULT_VOCAB_FILE, load_vocabulary, save_vocabulary

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Simple ASR Control Panel</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background: #f4f6fb; }
      h1 { margin-bottom: 0.5rem; }
      form { background: #fff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
      label { display: block; margin-bottom: 0.25rem; font-weight: 600; }
      input[type=text], input[type=number], textarea { width: 100%; padding: 0.5rem; margin-bottom: 1rem; border: 1px solid #cfd4ea; border-radius: 6px; }
      textarea { min-height: 120px; }
      button { padding: 0.6rem 1.2rem; background: #0057ff; border: none; border-radius: 6px; color: #fff; cursor: pointer; font-size: 1rem; }
      button:hover { background: #003fcc; }
      .meta { margin-top: 0.5rem; color: #4a4f64; font-size: 0.95rem; }
      .flash { padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem; background: #e8f5e9; color: #33691e; }
    </style>
  </head>
  <body>
    <h1>Simple ASR Control Panel</h1>
    {% if message %}<div class="flash">{{ message }}</div>{% endif %}
    <form method="post">
      <label for="hotkey">Hotkey</label>
      <input type="text" id="hotkey" name="hotkey" value="{{ config.hotkey }}" />

      <label for="sample_rate">Sample Rate</label>
      <input type="number" id="sample_rate" name="sample_rate" value="{{ config.sample_rate }}" min="8000" step="1000" />

      <label for="max_new_tokens">Max New Tokens</label>
      <input type="number" id="max_new_tokens" name="max_new_tokens" value="{{ max_new_tokens }}" min="16" step="8" />

      <label for="vocabulary">Custom Vocabulary (one phrase per line)</label>
      <textarea id="vocabulary" name="vocabulary">{{ vocabulary_text }}</textarea>

      <button type="submit">Save Settings</button>
      <p class="meta">Changes apply immediately and are persisted to your local configuration file.</p>
    </form>
  </body>
</html>
"""


def _parse_vocabulary(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def start_gui(
    app_instance,
    config_manager: ConfigManager,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    vocab_path=DEFAULT_VOCAB_FILE,
) -> threading.Thread:
    flask_app = Flask(__name__)
    flask_app.config["SECRET_KEY"] = "simple-asr-ui"

    message_box: dict[str, Optional[str]] = {"message": None}

    @flask_app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            loaded = config_manager.load()
            hotkey = (request.form.get("hotkey") or loaded.hotkey).strip()
            sample_rate = request.form.get("sample_rate") or loaded.sample_rate
            max_new_tokens = request.form.get(
                "max_new_tokens"
            ) or loaded.provider_options.get("max_new_tokens", 128)
            vocabulary_text = request.form.get("vocabulary", "")

            try:
                sample_rate_int = int(sample_rate)
            except (TypeError, ValueError):
                sample_rate_int = loaded.sample_rate

            try:
                max_new_tokens_int = int(max_new_tokens)
            except (TypeError, ValueError):
                max_new_tokens_int = int(
                    loaded.provider_options.get("max_new_tokens", 128)
                )

            new_options = dict(loaded.provider_options)
            new_options["max_new_tokens"] = max_new_tokens_int
            new_vocab = _parse_vocabulary(vocabulary_text)
            new_options["vocabulary"] = new_vocab

            updated = AppConfig(
                provider_name=loaded.provider_name,
                hotkey=hotkey or loaded.hotkey,
                model_id=loaded.model_id,
                sample_rate=sample_rate_int,
                provider_options=new_options,
            )

            config_manager.save(updated)
            save_vocabulary(new_vocab, vocab_path)
            try:
                app_instance.apply_settings(updated.copy())
                message_box["message"] = "Settings saved successfully."
            except ValueError as exc:
                message_box["message"] = str(exc)

            return redirect(url_for("index"))

        config = config_manager.load()
        vocabulary = load_vocabulary(vocab_path) or config.provider_options.get(
            "vocabulary", []
        )
        message = message_box["message"]
        message_box["message"] = None

        return render_template_string(
            TEMPLATE,
            config=config,
            vocabulary_text="\n".join(vocabulary),
            max_new_tokens=config.provider_options.get("max_new_tokens", 128),
            message=message,
        )

    def run_server() -> None:
        flask_app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    if open_browser:
        webbrowser.open(f"http://{host}:{port}/", new=2, autoraise=True)

    return thread
