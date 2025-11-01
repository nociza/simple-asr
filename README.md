## Simple ASR

Press-to-talk speech transcription powered by NVIDIA's Canary model from Hugging Face.

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA-capable drivers (recommended for Canary)
- Microphone accessible on the host system

### Setup
```bash
pip install .
```

The installation pulls in PyTorch, NVIDIA NeMo, and related audio libraries. Consult the [NeMo installation guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/installation.html) if you need GPU-specific wheels.

### Usage
```bash
python main.py --provider canary --hotkey f8
```

With [uv](https://github.com/astral-sh/uv) you can skip manual installs:
```bash
uv run start -- --provider canary --hotkey f8
```

1. Wait for the model to finish loading (first run may download weights from Hugging Face).
2. Hold the chosen hotkey (default: `F8`) to capture audio. Release it to start transcription.
3. The transcript prints to the console once ready.

Additional flags:
- `--model-id`: use a different Canary checkpoint hosted on Hugging Face.
- `--sample-rate`: override the default 16 kHz microphone sample rate.
- `--max-new-tokens`: control the generation length during decoding.
- `--log-level`: adjust logging verbosity (e.g., DEBUG).

### Extending Providers
Add new provider implementations under `simple_asr/providers/` and register them in `simple_asr/providers/__init__.py`. Each provider implements `BaseASRProvider` and exposes a `transcribe` method that returns plain text.
