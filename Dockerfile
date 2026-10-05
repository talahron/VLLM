# =============================================================================
# Single-stage build using the -devel CUDA image.
# Trades image size (~6-7 GB) and a larger CVE surface for the ability to
# JIT-compile Triton/flashinfer/torch.compile kernels at runtime, which vLLM
# expects by default.
#
# GPU target: nvidia-rtx-pro-6000 ONLY (Cloud Run, europe-west4).
#
# Host driver 580.x provides native CUDA 13.0 support, which matches this
# image's CUDA 13 userspace exactly. The CUDA forward-compat libraries are
# therefore NOT needed and are actively harmful here: loading the compat
# libcuda.so.1 on top of a same-branch (580) host driver shadows the host
# driver and fails with:
#   cudaGetDeviceCount() Error 803: system has unsupported display driver
#   / cuda driver combination
# So this image DELETES /usr/local/cuda/compat entirely and rebuilds the
# linker cache without it -- the host driver's libcuda.so.1 (injected by
# Cloud Run at /usr/local/nvidia/lib64) becomes the only one available.
#
# Build context: this Dockerfile + verify_safetensors.py. No heredocs /
# BuildKit-only syntax -- parses on the legacy gcr.io/cloud-builders/docker.
# =============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        build-essential \
        ninja-build \
        curl \
        git \
        libgomp1 \
        libucx0 \
        ca-certificates \
        tini \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── uv for fast, deterministic Python package installs ──────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── Python venv at /opt/venv ─────────────────────────────────────────────────
RUN python3.12 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pin setuptools >= 78.1.1 to address CVE-2025-47273 (path traversal in
# PackageIndex). Pip and wheel pinned to recent versions for hygiene.
RUN uv pip install --upgrade \
        "pip>=25.0" \
        "setuptools>=78.1.1" \
        "wheel>=0.45.0"

# vLLM 0.21.0 with the default CUDA 13 wheel (no --torch-backend override).
# Matches the base image's libcudart.so.13.
RUN uv pip install \
        vllm==0.21.0 \
        --no-cache

# ── transformers floor for Qwen3.8 ───────────────────────────────────────────
# The vLLM recipe for this model requires transformers >= 5.8.0, matching the
# version Qwen3.8's config.json was written by. vLLM parses the config with its
# own Qwen3_5Config, so the real dependency is the Qwen3-VL processor class.
#
# THIS IS THE CHANGE MOST LIKELY TO BREAK YOUR BUILD. If vLLM 0.21.0 pins
# transformers below 5.8, uv will either fail to resolve or silently downgrade.
# Check the build log for a resolution conflict, and check at runtime with:
#   /opt/venv/bin/python -c "import transformers; print(transformers.__version__)"
# If 0.21.0 cannot accommodate it, the fix is a newer vLLM, not a forced pin --
# see the notes in README.md before bumping.
RUN uv pip install --no-cache "transformers>=5.8.0"

# Cap FastAPI below 0.137 — 0.137.0 (released 2026-06-14) introduced lazy
# include_router that stores _IncludedRouter objects without a .path attribute,
# which crashes prometheus-fastapi-instrumentator on every request (incl.
# /health), failing the Cloud Run startup probe. vLLM applied the same cap.
# Bumping the instrumentator does NOT fix it; only capping FastAPI does.
RUN uv pip install --no-cache "fastapi<0.137"

# vLLM expects flashinfer at runtime on CUDA platforms but doesn't always
# install it as a transitive dependency. Install explicitly from FlashInfer's
# index, which has CUDA-architecture-matched wheels.
RUN uv pip install \
        flashinfer-python \
        --extra-index-url https://flashinfer.ai/whl/cu130/torch2.11 \
        --no-cache

# Scrub any leftover vulnerable setuptools wheels inside the venv.
RUN find /opt/venv -type f \( -name "setuptools-*.whl" -o -name "pip-*.whl" \) -delete \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ── Scrub apt-shipped setuptools artifacts (CVE-2025-47273) ──────────────────
RUN dpkg --purge --force-depends \
        python3-setuptools \
        python3-pkg-resources \
        python3-setuptools-whl 2>/dev/null || true \
    && rm -rf /usr/lib/python3/dist-packages/setuptools* \
              /usr/lib/python3/dist-packages/pkg_resources* \
              /usr/lib/python3/dist-packages/_distutils_hack* \
              /usr/lib/python3/dist-packages/distutils-precedence.pth \
              /usr/share/python-wheels \
              /usr/lib/python3.12/test/wheeldata \
    && find / -type f \( -name "setuptools-*.whl" -o -name "setuptools-*.tar.gz" \) \
         -not -path "/opt/venv/*" -not -path "/proc/*" -delete 2>/dev/null || true \
    && find / -type d -name "__pycache__" -not -path "/opt/venv/*" -not -path "/proc/*" \
         -exec rm -rf {} + 2>/dev/null || true

# ── Remove CUDA forward-compat libraries (Error 803 fix) ─────────────────────
# This is the core fix. The base image bundles cuda-compat and references
# /usr/local/cuda/compat in its own ld.so.conf, so the build-time ldconfig
# cache binds libcuda.so.1 to the COMPAT copy regardless of LD_LIBRARY_PATH.
# On an RTX PRO 6000 host (driver 580, native CUDA 13) that compat libcuda
# shadows the host driver and produces cudaGetDeviceCount() Error 803.
# Deleting the directory AND its ld.so.conf reference, then rebuilding the
# cache, guarantees the only libcuda.so.1 at runtime is the host driver's
# (injected by Cloud Run at /usr/local/nvidia/lib64).
RUN rm -rf /usr/local/cuda/compat \
    && rm -f /etc/ld.so.conf.d/compat.conf \
    && find /etc/ld.so.conf.d -type f -exec sed -i '\#/usr/local/cuda/compat#d' {} +

# ── Dynamic linker configuration ─────────────────────────────────────────────
# /usr/local/nvidia/lib64 is where Cloud Run mounts the host driver's
# userspace libraries (libcuda.so.1, libnvidia-ml.so.1, etc.). With compat
# gone, this is the sole source of libcuda at runtime.
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64"
RUN echo "/opt/venv/lib/python3.12/site-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf \
    && echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf \
    && echo "/usr/local/nvidia/lib64" > /etc/ld.so.conf.d/nvidia.conf \
    && ldconfig

# ── JIT kernel target architecture ───────────────────────────────────────────
# Triton / flashinfer / torch.compile JIT paths compile kernels at runtime.
# RTX PRO 6000 Blackwell = compute capability 12.0 (sm_120). +PTX for forward
# compatibility / JIT fallback.
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"

# ── Security hardening: non-root user ────────────────────────────────────────
RUN userdel -r ubuntu 2>/dev/null || true \
    && useradd -m -u 1000 -s /bin/bash vllm \
    && mkdir -p /app /models \
    && chown -R vllm:vllm /app /models

# ── Remove NVIDIA Nsight Compute (CVE remediation) ───────────────────────────
RUN rm -rf /opt/nvidia/nsight-compute \
    && rm -rf /opt/nvidia/nsight-systems \
    && rm -rf /opt/nvidia/nsight* \
    && find /usr/local/cuda -path "*nsight*" -exec rm -rf {} + 2>/dev/null || true

# ── Strip other -devel-only tooling not needed for inference ─────────────────
RUN rm -rf /usr/local/cuda/samples \
           /usr/local/cuda/doc \
           /usr/local/cuda/extras/CUPTI/samples \
           /usr/local/cuda/extras/Debugger \
           /usr/local/cuda/tools \
    && rm -rf /opt/nvidia/cuda-gdb \
    && find /usr/local/cuda -name "*.a" -delete 2>/dev/null \
    && find /usr/local/cuda -name "*.la" -delete 2>/dev/null

USER vllm
WORKDIR /app

# ── Policy enforcement: safetensors-only check ───────────────────────────────
COPY --chown=vllm:vllm verify_safetensors.py /app/verify_safetensors.py
RUN chmod +x /app/verify_safetensors.py

# ── vLLM runtime configuration ───────────────────────────────────────────────
ENV HF_HOME=/home/vllm/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/vllm/.cache/huggingface
ENV TRITON_CACHE_DIR=/home/vllm/.cache/triton

# FP8 weights (~31 GB) on a 96 GB RTX PRO 6000 leave ~55-60 GB for KV cache
# after activations/overhead, so both context length and concurrency can rise.
ENV MODEL_PATH=/models/Qwen3.8-27B-FP8
ENV SERVED_MODEL_NAME=qwen38-27b-fp8

# ── Tunables overridable at deploy time (--set-env-vars) ─────────────────────
#   GPU_MEMORY_UTILIZATION   (default 0.90)
#   MAX_MODEL_LEN            (default 262144 = 256K; Qwen3.8 native context)
#   MAX_NUM_SEQS             (default 64)
#   MAX_NUM_BATCHED_TOKENS   (default 16384)   <-- NEW
#   KV_CACHE_DTYPE           (default fp8)
#   TOOL_CALL_PARSER         (default qwen3_coder)  <-- CHANGED from qwen3_xml
#
# CORRECTION to the previous comment here: Qwen3.8-27B is NOT standard
# attention. It is hybrid: only 16 of its 64 layers run full attention
# (full_attention_interval: 4); the other 48 run LINEAR attention with a
# constant-size recurrent state. That state is constant per sequence rather
# than growing with context, so unlike Qwen3.6's Mamba cache it does not
# impose a hard block ceiling on MAX_NUM_SEQS -- but it is not free either.
# The practical limit is still KV-cache memory. Read the startup log.
ENV VLLM_LOGGING_LEVEL=INFO
ENV PYTORCH_DISABLE_NVFUSER_CUSTOMOP_REGISTRATION=1

# ── Prefill batch budget (throughput lever #1) ───────────────────────────────
# vLLM's default chunked-prefill token budget is conservative. For long Hebrew
# legal documents the workload is prefill-bound, not decode-bound, so raising
# this is the change most likely to move your wall-clock numbers. 16384 is the
# value the vLLM recipe uses. Chunked prefill is on by default in the V1
# engine, so this does not need to be >= MAX_MODEL_LEN.
ENV MAX_NUM_BATCHED_TOKENS=16384

# ── Speculative decoding: in-checkpoint MTP head (throughput lever #2) ───────
# Qwen3.8-27B ships an MTP draft head inside the checkpoint. No second model to
# download, no extra weight memory. On CUDA the method string is "mtp".
# ("qwen3_5_mtp" is the Huawei Ascend spelling and will not work here.)
#
# Honest caveat: MTP is a LATENCY win at low concurrency and can become a
# THROUGHPUT loss at high concurrency, because rejected draft tokens burn
# compute that batched requests would have used productively. At
# --concurrency=10 this is genuinely ambiguous. Benchmark it both ways and read
# the acceptance rate (see README) rather than assuming.
#
# Set to "" to disable.
ENV SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'

# ── CUDA graph mode ──────────────────────────────────────────────────────────
# FULL_DECODE_ONLY captures CUDA graphs for decode only and leaves prefill
# eager. This is the safe mode for hybrid/linear-attention models and avoids
# the class of graph-capture OOM seen on Qwen3.6. Set to "" to use vLLM's
# default mode.
ENV COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

# ── Server-wide thinking default ─────────────────────────────────────────────
# Belt and braces. You already pass {"enable_thinking": false} from client
# code; this makes it the server default so any client that forgets the kwarg
# (JetBrains AI Assistant, Continue.dev, an unaudited pipeline path) still gets
# no-think instead of silently paying for a reasoning block.
#
# Note: enable_thinking and reasoning_effort are ALTERNATIVES, not a stack.
# reasoning_effort (xhigh default / medium / low) tunes ADAPTIVE thinking, so
# it only has an effect when thinking is on. With enable_thinking false you are
# already at the cheapest setting and adding reasoning_effort buys nothing.
# If you want some reasoning back, replace the value below with
# '{"reasoning_effort": "low"}' -- do not set both.
ENV DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

# ── Optional: drop the vision tower ──────────────────────────────────────────
# Qwen3.8-27B is multimodal (Qwen3_5ForConditionalGeneration, with a
# vision_config in config.json). If THIS endpoint only ever serves text,
# setting LANGUAGE_MODEL_ONLY=1 frees the vision tower's memory and buys a
# meaningfully larger KV pool.
# Do NOT enable it if this is the same service you route page images through.
# Default off, because that routing question is yours to answer.
ENV LANGUAGE_MODEL_ONLY=0

# ── Tool call parser ─────────────────────────────────────────────────────────
# CHANGED from qwen3_xml to qwen3_coder: qwen3_coder is what the vLLM recipe
# specifies for Qwen3.8. If your extraction agents' tool calls start failing to
# parse after this deploy, that is the first thing to roll back
# (--set-env-vars TOOL_CALL_PARSER=qwen3_xml).
ENV TOOL_CALL_PARSER=qwen3_coder

EXPOSE 8000

# ── Startup: diagnostics + safetensors check + vLLM ──────────────────────────
# Shell form so env vars expand at runtime (legacy-builder compatible).
# Optional JSON-valued flags are accumulated into EXTRA_ARGS with literal
# single quotes, then `eval` re-parses them so the JSON survives intact and an
# empty variable drops its flag entirely rather than passing "".
# MODEL_PATH / SERVED_MODEL_NAME must not contain spaces (eval re-splits).
CMD echo "[startup] libcuda resolution:"; \
    ldconfig -p | grep -i libcuda || echo "[startup] WARNING: no libcuda in cache"; \
    echo "[startup] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"; \
    echo "[startup] transformers: $(/opt/venv/bin/python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo UNKNOWN)"; \
    EXTRA_ARGS=""; \
    if [ -n "${COMPILATION_CONFIG}" ]; then \
        EXTRA_ARGS="${EXTRA_ARGS} --compilation-config '${COMPILATION_CONFIG}'"; \
    fi; \
    if [ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS}" ]; then \
        EXTRA_ARGS="${EXTRA_ARGS} --default-chat-template-kwargs '${DEFAULT_CHAT_TEMPLATE_KWARGS}'"; \
    fi; \
    if [ -n "${SPECULATIVE_CONFIG}" ]; then \
        EXTRA_ARGS="${EXTRA_ARGS} --speculative-config '${SPECULATIVE_CONFIG}'"; \
    else \
        echo "[startup] speculative decoding DISABLED"; \
    fi; \
    if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then \
        EXTRA_ARGS="${EXTRA_ARGS} --language-model-only"; \
        echo "[startup] vision tower DISABLED (text-only serving)"; \
    fi; \
    echo "[startup] extra args:${EXTRA_ARGS}"; \
    /opt/venv/bin/python /app/verify_safetensors.py \
    && eval exec python3.12 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host 0.0.0.0 \
        --port 8000 \
        --dtype auto \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
        --max-model-len "${MAX_MODEL_LEN:-262144}" \
        --max-num-seqs "${MAX_NUM_SEQS:-64}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
        --enable-auto-tool-choice \
        --reasoning-parser qwen3 \
        --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
        ${EXTRA_ARGS}
