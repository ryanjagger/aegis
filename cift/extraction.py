"""Activation-extraction core: the white-box model wrapper (U2).

Loads a small instruct model once (process-lifetime singleton) and returns
per-layer hidden states at the readout position for a prompt. This is the only
module that runs a forward pass; it is kept independent of the
``BaseResponsesAdapter`` protocol so the lab never depends on the live pipeline
(the stretch gate, U9, is the only thing that would wrap this in an adapter).

Key correctness points (see the paper, arXiv:2606.04141 §4.2):

- **dtype** must be ``float32`` on Apple Silicon — transformers v5 defaults to
  ``dtype="auto"`` which loads Qwen2.5's bf16 weights and crashes on MPS. We pin
  float32 and assert it after load.
- **readout position** is the final prompt token after the chat template appends
  the assistant header (``add_generation_prompt=True``) — the pre-generation
  decision position. We pool the last ``K = floor(0.25 * num_layers)`` layers at
  that single position. Pooling at static credential-token positions is the
  classic fake-good-result mistake and is never done by default.
- a *generated*-token readout would need a ``generate(max_new_tokens=1, ...)``
  step; this single forward captures the final-prompt-token position only.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# Route any op without an MPS kernel to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from app.config import get_settings  # noqa: E402


def resolve_device(setting: str) -> str:
    """Map the ``AIS_CIFT_DEVICE`` setting to a concrete device string."""

    if setting == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return setting


@lru_cache(maxsize=2)
def _load(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # v5 kwarg is `dtype=`, not `torch_dtype=`. float32 is mandatory on MPS.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model = model.to(device).eval()
    actual = next(model.parameters()).dtype
    if actual != torch.float32:
        raise RuntimeError(
            f"expected float32 weights but loaded {actual}; a transformers version "
            "regression may have ignored the dtype= kwarg (pin transformers>=5,<6)"
        )
    return tokenizer, model


def get_model(model_name: str | None = None, device: str | None = None):
    settings = get_settings()
    name = model_name or settings.cift_model_name
    dev = resolve_device(device or settings.cift_device)
    torch.manual_seed(settings.cift_seed)
    return _load(name, dev), dev


def last_k(num_layers: int) -> int:
    """K = floor(0.25 * L); the paper's default monitored window of late layers."""

    return max(1, num_layers // 4)


def fingerprint(device: str) -> dict:
    """Identity of the extraction stack, stored alongside a fitted baseline.

    MPS float32 features are not guaranteed bitwise-stable across torch /
    transformers versions, so a persisted baseline is only trustworthy against the
    same fingerprint.
    """

    import transformers

    settings = get_settings()
    return {
        "model": settings.cift_model_name,
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "float32",
    }


def extract_features(
    text: str, *, token_index: int = -1, model_name: str | None = None, device: str | None = None
) -> np.ndarray:
    """Return last-K-layer hidden states at one token position: ``[K, hidden]``.

    ``token_index`` defaults to ``-1`` (the final prompt token = readout position).
    Passing an earlier index lets AE2 contrast the readout position against a
    credential-token position. Features come back as CPU float32 NumPy.
    """

    (tokenizer, model), dev = get_model(model_name, device)
    messages = [{"role": "user", "content": text}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(dev)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)

    k = last_k(model.config.num_hidden_layers)
    # hidden_states is length L+1 (index 0 = embeddings); take the last K layers.
    layers = out.hidden_states[-k:]
    readout = torch.stack([h[0, token_index, :] for h in layers], dim=0)  # [K, hidden]
    return readout.to(torch.float32).cpu().numpy()


def generate_text(
    text: str,
    *,
    max_new_tokens: int = 96,
    model_name: str | None = None,
    device: str | None = None,
) -> str:
    """Greedy-decode the model's reply to ``text`` (for the text-scanner arm).

    The contrast needs the model's actual output so the text scanner has something
    to scan; the CIFT arm reads the prompt-side activations from
    ``extract_features``. Greedy decoding keeps the lab reproducible.
    """

    (tokenizer, model), dev = get_model(model_name, device)
    messages = [{"role": "user", "content": text}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(dev)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    completion = out[0, input_ids.shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True)


def extract_many(
    texts: list[str],
    *,
    token_index: int = -1,
    model_name: str | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Stack ``extract_features`` over many prompts into ``[N, K, hidden]``.

    batch=1 per prompt (no padding) keeps the readout-position selection exact and
    avoids the left/right-pad position class of bugs in a research lab.
    """

    return np.stack(
        [
            extract_features(t, token_index=token_index, model_name=model_name, device=device)
            for t in texts
        ]
    )
