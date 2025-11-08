## Simple ASR

A very, very simple locally hosted press-to-talk speech transcription helper that types wherever your cursor is focused and lets you switch between open-source ASR models (default: NVIDIA's Canary on Hugging Face). Minimal alternative to wispr flow. Feel free to contribute by opening a PR. 

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA-capable drivers (recommended for Canary)
- Microphone accessible on the host system

### Setup

```bash
uv sync
```

The installation pulls in PyTorch, NVIDIA NeMo, and related audio libraries. Ensure your CUDA, GPU drivers, and Python runtime satisfy the Canary requirements. Consult the [NeMo installation guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/installation.html) if you need GPU-specific wheels.

### Usage

```bash
uv run main.py --provider canary --hotkey f10 --vocab "PyTorch" "CUDA" "Hopper"
```

Launch the built-in web UI for live configuration management:

```bash
uv run python main.py --with-gui
```

1. Wait for the model to finish loading (first run may download weights from Hugging Face).
2. Hold the chosen hotkey (default: `F8`) to capture audio. Release it to start transcription.
3. The transcript prints to the console and is pasted into the foreground application once ready. If the clipboard is unavailable, the app falls back to simulating keystrokes.
4. Press `Ctrl+C` in the terminal to quit. The listener cleans up its background hooks before exiting.

Additional flags:

- `--model-id`: use a different Canary checkpoint hosted on Hugging Face.
- `--sample-rate`: override the default 16 kHz microphone sample rate.
- `--max-new-tokens`: control the generation length during decoding.
- `--log-level`: adjust logging verbosity (e.g., DEBUG).
- `--provider`: switch to additional providers as you add them.
- `--with-gui`: spin up the local web UI for interactive controls and vocabulary management.

### Extending Providers

Add new provider implementations under `simple_asr/providers/` and register them in `simple_asr/providers/__init__.py`. Each provider implements `BaseASRProvider` and exposes a `transcribe` method that returns plain text.

### Custom Vocabulary via TOML

Place a `vocab.toml` file in the project root to maintain custom terminology across runs. Example:

```toml
phrases = [
  "PyTorch",
  "CUDA",
  "Hopper"
]
```

Edits to `vocab.toml` are picked up automatically on the next start, and any vocabulary changes made through the CLI or web UI keep the file in sync.
