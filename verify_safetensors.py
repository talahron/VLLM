#!/usr/bin/env python3
"""
Verify that model weights at MODEL_PATH are stored as safetensors.

Exits with code 0 if all weight files are safetensors.
Exits with code 1 (with a clear error message) otherwise.

This enforces an organizational policy: only safetensors-format weights may
be deployed. Pickle-based formats (.bin, .pt, .pth, .ckpt) can execute
arbitrary code on load and are not permitted.
"""

import os
import sys
from pathlib import Path

# Extensions known to indicate model weights (vs config, tokenizer, etc.)
# .safetensors is the only one our policy allows.
ALLOWED_WEIGHT_EXTENSIONS = {".safetensors"}

DISALLOWED_WEIGHT_EXTENSIONS = {
    ".bin",      # pickle-based weights (older HF format)
    ".pt",       # PyTorch checkpoint (pickle)
    ".pth",      # PyTorch checkpoint (pickle)
    ".ckpt",     # generic pickle checkpoint
    ".msgpack",  # flax/jax weights (not safetensors)
    ".h5",       # TensorFlow/Keras HDF5
    ".pkl",      # raw pickle
}

# Files that look like weights by extension. Any "pytorch_model*" or
# "model*" file ending in .bin/.pt/.pth/.ckpt should trigger the check.
WEIGHT_NAME_HINTS = (
    "pytorch_model",
    "model-",
    "model.",
    "consolidated.",
    "weights",  # some FP8/quantized dumps use "weights-00001.bin"
    "quant",    # catch quantized pickle dumps e.g. "quant_model.pt"
)


def find_weight_files(model_dir: Path) -> tuple[list[Path], list[Path]]:
    """Walk the model directory and classify files as allowed or disallowed.

    Returns (allowed_files, disallowed_files).
    """
    allowed = []
    disallowed = []

    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        name = path.name.lower()

        # Skip files that clearly aren't weights (configs, tokenizers, etc.)
        # Tokenizers can legitimately be .bin (sentencepiece models), so
        # filter on filename to avoid false positives.
        if suffix not in ALLOWED_WEIGHT_EXTENSIONS | DISALLOWED_WEIGHT_EXTENSIONS:
            continue

        if "tokenizer" in name or "vocab" in name or "merges" in name:
            # These are tokenizer artifacts, not model weights — even if they
            # happen to be .bin (sentencepiece is .model or .bin sometimes).
            continue

        if suffix in ALLOWED_WEIGHT_EXTENSIONS:
            allowed.append(path)
        elif suffix in DISALLOWED_WEIGHT_EXTENSIONS:
            # Only flag as a weight file if the name looks like weights
            # (avoids false positives like config.bin from random tools).
            if any(hint in name for hint in WEIGHT_NAME_HINTS):
                disallowed.append(path)

    return allowed, disallowed


def main() -> int:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("ERROR: MODEL_PATH environment variable is not set.", file=sys.stderr)
        return 1

    model_dir = Path(model_path)
    if not model_dir.exists():
        print(
            f"ERROR: MODEL_PATH does not exist: {model_path}\n"
            f"Is the GCS FUSE volume mounted? Does the directory exist in the bucket?",
            file=sys.stderr,
        )
        return 1

    if not model_dir.is_dir():
        print(f"ERROR: MODEL_PATH is not a directory: {model_path}", file=sys.stderr)
        return 1

    allowed, disallowed = find_weight_files(model_dir)

    if disallowed:
        print(
            "=" * 70 + "\n"
            "POLICY VIOLATION: Non-safetensors model weights detected.\n"
            "\n"
            "Organizational policy permits only .safetensors format for model\n"
            "weights. Pickle-based formats (.bin/.pt/.pth/.ckpt) can execute\n"
            "arbitrary code at load time and are not allowed.\n"
            "\n"
            "Disallowed files found:",
            file=sys.stderr,
        )
        for path in disallowed:
            rel = path.relative_to(model_dir)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  - {rel} ({size_mb:.1f} MB)", file=sys.stderr)
        print(
            "\n"
            "To fix:\n"
            f"  1. Re-download the model with only safetensors files, or\n"
            f"  2. Use Hugging Face's safetensors conversion tool to convert\n"
            f"     existing weights, then re-upload to your model bucket.\n"
            f"  3. Verify your bucket contents with:\n"
            f"     gcloud storage ls {model_path.replace('/models/', 'gs://YOUR_BUCKET/')}\n"
            + "=" * 70,
            file=sys.stderr,
        )
        return 1

    if not allowed:
        print(
            f"ERROR: No model weight files found in {model_path}.\n"
            f"Expected at least one .safetensors file.\n"
            f"Directory contents:",
            file=sys.stderr,
        )
        for path in sorted(model_dir.iterdir()):
            print(f"  - {path.name}", file=sys.stderr)
        return 1

    total_size_gb = sum(p.stat().st_size for p in allowed) / (1024 ** 3)
    print(
        f"Safetensors policy check passed: "
        f"found {len(allowed)} .safetensors file(s), "
        f"total {total_size_gb:.2f} GB."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
