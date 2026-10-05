# Qwen3.8-27B-FP8 on vLLM — Cloud Run GPU Deployment

Serves the Qwen3.8-27B-FP8 model with vLLM on Google Cloud Run using a single NVIDIA RTX PRO 6000 Blackwell GPU (96 GB VRAM). Weights are loaded from a GCS bucket mounted via Cloud Storage FUSE, and an OpenAI-compatible API is exposed on port 8000.

## Contents

- [Architecture overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Repository layout](#repository-layout)
- [Model weights in GCS](#model-weights-in-gcs)
- [Building the image](#building-the-image)
- [Deploying to Cloud Run](#deploying-to-cloud-run)
- [Configuration reference](#configuration-reference)
- [Performance tuning](#performance-tuning)
- [Verifying the deployment](#verifying-the-deployment)
- [Calling the service](#calling-the-service)
- [Tuning context length and concurrency](#tuning-context-length-and-concurrency)
- [Troubleshooting](#troubleshooting)

## Architecture overview

| Component | Detail |
|---|---|
| Model | Qwen3.8-27B-FP8 (FP8-quantized, ~31 GB of weights) |
| Architecture | Hybrid attention: 16 of 64 layers full attention, 48 linear attention; multimodal (vision tower present) |
| Serving engine | vLLM 0.21.0, OpenAI-compatible server |
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM, compute capability sm_120) |
| Platform | Cloud Run service (GPU, scale-to-zero) |
| Weights source | GCS bucket mounted at `/models` via Cloud Storage FUSE |
| Base image | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` |
| KV cache | FP8 (`--kv-cache-dtype fp8`) for extended context / concurrency |
| Speculative decoding | In-checkpoint MTP draft head, 3 tokens (toggleable) |
| Thinking | No-think by default, server-side (`--default-chat-template-kwargs`) |
| Ingress | Internal only, VPC-attached, unauthenticated access disabled |

The container deletes the CUDA forward-compat libraries at build time. On an RTX PRO 6000 host the driver (580.x) natively provides CUDA 13, so the compat `libcuda.so.1` would shadow the host driver and fail with `cudaGetDeviceCount() Error 803`. Removing it lets the host driver's library (injected by Cloud Run at `/usr/local/nvidia/lib64`) be the only one resolved.

A startup policy check (`verify_safetensors.py`) refuses to launch if any model weights are stored in a pickle-based format (`.bin`, `.pt`, `.pth`, `.ckpt`). Only `.safetensors` are permitted.

### A note on the vLLM recipe

The upstream recipe at <https://recipes.vllm.ai/Qwen/Qwen3.8-27B> carries launch commands for several backends. The `--quantization ascend` command is for Huawei Ascend NPUs and must not be copied here. Specifically, these flags from that block are wrong or inert on CUDA:

| Flag | Why not here |
|---|---|
| `--quantization ascend` | Selects the ModelSlim INT8 scheme on the Ascend backend. No CUDA equivalent. |
| `Eco-Tech/Qwen3.8-27B-w8a8` | ModelScope checkpoint built for that backend. |
| `"method":"qwen3_5_mtp"` | The Ascend spelling of MTP. On CUDA the method is `"mtp"`. |
| `--mm-encoder-tp-mode data` | No-op at TP1. |
| `--additional-config '{"enable_cpu_binding":true}'` | Ascend-specific. |
| `--enable-prefix-caching` | Already the default in the V1 engine. |
| `enforce_eager: true` | Disables CUDA graphs. Wanted on a 32 GB card that cannot fit graph capture; not wanted on 96 GB. |

The three things worth porting from it — `--max-num-batched-tokens`, the compilation config, and MTP under its CUDA name — are all in the Dockerfile.

## Prerequisites

- `gcloud` CLI authenticated with permission to deploy to Cloud Run in the target project.
- Two GCP projects (typical MoJ split):
  - **Image/registry project** (`moj-gcp-shared-devops`) — hosts the Artifact Registry repo.
  - **Deployment project** (`moj-gcp-d-rag`) — where the Cloud Run service runs.
- The Cloud Run runtime service account must have `roles/artifactregistry.reader` on the registry repo in the image project (cross-project pull).
- GPU quota for `nvidia-rtx-pro-6000` in the target region (granted as milliGPU; 3,000 milliGPU = 3 GPUs on first use).
- A GCS bucket containing the model weights in `.safetensors` format.
- `transformers >= 5.8.0` inside the image. This is a hard requirement of the Qwen3.8 recipe: it matches the version `config.json` was written by, and vLLM needs the Qwen3-VL processor class it provides. See [Troubleshooting](#troubleshooting) if the build fights you on this.

> **Region availability:** RTX PRO 6000 is only offered in specific Cloud Run regions. Confirm `europe-west4` (or your region) supports it before deploying.

## Repository layout

```
.
├── Dockerfile              # CUDA 13 devel base, vLLM 0.21.0, compat-lib removal
├── verify_safetensors.py   # Startup gate: rejects non-safetensors weights
└── README.md               # This file
```

## Model weights in GCS

Weights live under a single folder in the model bucket, one subdirectory per model. The folder name must match `MODEL_PATH` byte-for-byte — including case, dots, and hyphens. Mismatches are the most common cause of a failed startup (`MODEL_PATH does not exist`).

List the bucket to find the exact name:

```bash
gcloud storage ls gs://moj-gcp-d-rag-qwen-models/
```

Example output:

```
gs://moj-gcp-d-rag-qwen-models/Qwen-3.8-27b-FP8/
gs://moj-gcp-d-rag-qwen-models/Qwen3.6-27B-BF16/
...
```

Confirm the shards and config are present:

```bash
gcloud storage ls gs://moj-gcp-d-rag-qwen-models/Qwen-3.8-27b-FP8/
```

You should see `*.safetensors` shards plus `config.json`. The `config.json` should contain a `quantization_config` block identifying the FP8 method — vLLM auto-detects this with `--dtype auto`. It should also contain a `vision_config` block; that is expected, the model is multimodal.

In the example above the actual folder is `Qwen-3.8-27b-FP8` (hyphen after `Qwen`, lowercase `b`). Set `MODEL_PATH=/models/Qwen-3.8-27b-FP8` to match. Adjust to whatever your bucket actually shows.

## Building the image

Build with the legacy Cloud Build docker builder (the Dockerfile avoids BuildKit-only syntax):

```bash
export PROJECT=moj-gcp-shared-devops
export REGION_IMAGE=me-west1
export REPO=green
export VERSION=qwen-3.8-27b-FP8

gcloud builds submit \
  --project=${PROJECT} \
  --tag=${REGION_IMAGE}-docker.pkg.dev/${PROJECT}/${REPO}/vllm:${VERSION} \
  .
```

Check the build log for the `transformers` resolution. The image installs `transformers>=5.8.0` after vLLM. If `uv` reports a conflict, or if it resolves to something below 5.8, read the [Troubleshooting](#troubleshooting) entry before deploying.

## Deploying to Cloud Run

The full deploy script. Adjust the `MODEL_PATH` to match your bucket folder exactly, then run:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── Projects / registry ──────────────────────────────────────────────────────
export PROJECT=moj-gcp-shared-devops        # image / Artifact Registry project
export LOCAL_PROJECT=moj-gcp-d-rag          # Cloud Run deployment project
export REGION=europe-west4                  # Cloud Run region (must offer the GPU)
export REGION_IMAGE=me-west1                # Artifact Registry region
export REPO=green                           # Artifact Registry repo
export VERSION=qwen-3.8-27b-FP8             # image tag
export BUCKET=moj-gcp-d-rag-qwen-models     # model weights bucket

# ── Resolve the image digest (pin the exact image) ───────────────────────────
DIGEST=$(gcloud artifacts docker images describe \
  ${REGION_IMAGE}-docker.pkg.dev/${PROJECT}/${REPO}/vllm:${VERSION} \
  --format='value(image_summary.digest)')
echo "Deploying: ${DIGEST}"

# ── Deploy ───────────────────────────────────────────────────────────────────
gcloud beta run deploy qwen38-vllm-service \
  --region=${REGION} \
  --project=${LOCAL_PROJECT} \
  --image=${REGION_IMAGE}-docker.pkg.dev/${PROJECT}/${REPO}/vllm:${VERSION}@${DIGEST} \
  --gpu=1 \
  --gpu-type=nvidia-rtx-pro-6000 \
  --cpu=20 \
  --memory=80Gi \
  --no-cpu-throttling \
  --no-gpu-zonal-redundancy \
  --cpu-boost \
  --add-volume=name=models,type=cloud-storage,bucket=${BUCKET} \
  --add-volume-mount=volume=models,mount-path=/models \
  --set-env-vars=^@^MODEL_PATH=/models/Qwen-3.8-27b-FP8@SERVED_MODEL_NAME=qwen3.8-27b-FP8@MAX_MODEL_LEN=131072@MAX_NUM_SEQS=10@MAX_NUM_BATCHED_TOKENS=16384 \
  --network=projects/moj-gcp-dev-net-spoke-0/global/networks/dev-spoke-0 \
  --subnet=projects/moj-gcp-dev-net-spoke-0/regions/europe-west4/subnetworks/dev-rag-ewest4 \
  --vpc-egress=private-ranges-only \
  --service-account=458040623421-compute@developer.gserviceaccount.com \
  --max-instances=3 \
  --min-instances=0 \
  --concurrency=10 \
  --timeout=3600 \
  --port=8000 \
  --startup-probe=httpGet.path=/health,httpGet.port=8000,initialDelaySeconds=60,periodSeconds=30,timeoutSeconds=25,failureThreshold=120 \
  --no-allow-unauthenticated \
  --ingress=internal
```

**Note on `--set-env-vars` syntax:** the `^@^` prefix changes the delimiter from comma to `@`. This matters as soon as you want to pass a JSON-valued variable such as `SPECULATIVE_CONFIG`, because the JSON contains commas that gcloud would otherwise treat as variable separators. Example:

```bash
--set-env-vars='^@^SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":5}@MAX_NUM_SEQS=10'
```

To turn speculative decoding off for an A/B comparison:

```bash
gcloud beta run services update qwen38-vllm-service \
  --region=europe-west4 --project=moj-gcp-d-rag \
  --set-env-vars=SPECULATIVE_CONFIG=
```

**CPU/memory ratio:** On the RTX PRO 6000, Cloud Run pins memory to the CPU count. At 20 CPU the allowed memory is exactly 80Gi (both floor and ceiling). To get more host RAM you must raise CPU (roughly +4Gi ceiling per extra vCPU), e.g. `--cpu=32 --memory=128Gi`, subject to CPU quota.

## Configuration reference

Environment variables consumed by the container (override via `--set-env-vars`):

| Variable | Default (image) | Purpose |
|---|---|---|
| `MODEL_PATH` | `/models/Qwen3.8-27B-FP8` | Path to weights inside the FUSE mount. Must match the bucket folder exactly. |
| `SERVED_MODEL_NAME` | `qwen38-27b-fp8` | Model name clients use in API calls. |
| `MAX_MODEL_LEN` | `262144` | Max context length (tokens). |
| `MAX_NUM_SEQS` | `64` | Max concurrent in-flight sequences in vLLM's scheduler. |
| `MAX_NUM_BATCHED_TOKENS` | `16384` | Prefill token budget per scheduler step. The main throughput lever for long documents. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM vLLM may use. |
| `KV_CACHE_DTYPE` | `fp8` | KV cache precision. Set to `auto` for bf16 KV. |
| `SPECULATIVE_CONFIG` | `{"method":"mtp","num_speculative_tokens":3}` | MTP speculative decoding. Empty string disables. |
| `COMPILATION_CONFIG` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | CUDA graph capture mode. Empty string uses vLLM's default. |
| `DEFAULT_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking": false}` | Server-wide thinking default. |
| `LANGUAGE_MODEL_ONLY` | `0` | Set to `1` to drop the vision tower on a text-only endpoint. |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Tool-call parser. Was `qwen3_xml`; see note below. |

Key vLLM server flags baked into the image:

| Flag | Value | Notes |
|---|---|---|
| `--dtype` | `auto` | Reads FP8 quant method from the checkpoint config. |
| `--kv-cache-dtype` | `fp8` | Halves KV-cache footprint; slight precision cost. |
| `--max-num-batched-tokens` | `16384` | Value used by the upstream recipe. |
| `--reasoning-parser` | `qwen3` | **Not optional even with thinking off.** The chat template opens every assistant turn with `<think>`; without the parser the whole reasoning block lands in `message.content`. |
| `--tool-call-parser` | `qwen3_coder` | Changed from `qwen3_xml`. The recipe specifies `qwen3_coder` for Qwen3.8. If your extraction agents' tool calls stop parsing after this deploy, roll back with `--set-env-vars=TOOL_CALL_PARSER=qwen3_xml` — that is the first suspect. |
| `--enable-auto-tool-choice` | — | Auto tool selection. |
| `--speculative-config` | MTP, 3 tokens | In-checkpoint draft head. No extra weights. |
| `--compilation-config` | `FULL_DECODE_ONLY` | Decode-only CUDA graphs; safe for hybrid-attention models. |
| `--tensor-parallel-size` | (1, implicit) | Cloud Run attaches one GPU per instance, so TP must be 1. |

**Tensor parallelism note:** Upstream Qwen guidance sometimes suggests `--tensor-parallel-size 4`. That assumes a 4-GPU node (the recipe's TP4 command targets a GB300 tray). Cloud Run attaches a single GPU per instance, so keep TP at 1 here.

## Performance tuning

Changes in this revision, roughly in descending order of expected impact for a long-document extraction workload.

### 1. `--max-num-batched-tokens 16384`

Hebrew legal documents produce long prompts and short completions, so the service is prefill-bound rather than decode-bound. vLLM's default chunked-prefill budget is conservative. Raising it lets more prompt tokens go through per scheduler step. This is the change most likely to move wall-clock numbers, and it costs nothing but a little activation memory.

Chunked prefill is on by default in the V1 engine, so this value does not need to be greater than or equal to `MAX_MODEL_LEN`.

### 2. Speculative decoding via the in-checkpoint MTP head

Qwen3.8-27B ships an MTP draft head inside the checkpoint. There is no second model to download and no extra weight memory. Reported acceptance rates upstream sit between roughly 0.75 and 0.90 depending on the checkpoint build.

This is a latency win at low concurrency and can be a throughput loss at high concurrency. Rejected draft tokens consume compute that batched requests would otherwise have used. At `--concurrency=10` the outcome is genuinely ambiguous, so measure it rather than assuming. See [Verifying the deployment](#verifying-the-deployment) for the acceptance metric, and the `SPECULATIVE_CONFIG=` update command above for the A/B.

If acceptance is high but throughput is flat, try `num_speculative_tokens: 5`. If acceptance is below ~0.6, turn it off.

### 3. Thinking modes

`enable_thinking` and `reasoning_effort` are alternatives, not a stack:

- `{"enable_thinking": false}` — no thinking at all. The model answers directly.
- `{"reasoning_effort": "low"}` — adaptive thinking, one step down from the default. The scale is `xhigh` (default), `medium`, `low`.

`reasoning_effort` only has an effect when thinking is on. If client code already sends `enable_thinking: false`, it is already at the cheapest setting and adding `reasoning_effort: low` buys nothing — if anything it risks re-enabling reasoning depending on template precedence.

What this revision changes is where the setting lives. Passing it server-side via `--default-chat-template-kwargs` means a client that forgets the kwarg still gets no-think, instead of silently paying for a reasoning block. Per-request `chat_template_kwargs` still override the server default, so existing client code keeps working unchanged.

### 4. `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`

Captures CUDA graphs for decode and leaves prefill eager. Safe mode for hybrid-attention models, and it sidesteps the class of graph-capture OOM seen on Qwen3.6.

### 5. `LANGUAGE_MODEL_ONLY=1` (optional, off by default)

Qwen3.8-27B carries a vision tower. On a text-only endpoint, dropping it frees that memory for KV cache — upstream measurements on a memory-constrained card showed a KV pool going from roughly 91K to 136K tokens with this flag alone.

Left off by default because whether this endpoint ever receives image input is a question about your PDF routing, not about the model. If the vision path is a separate service, turn it on.

### Not changed, and why

- **NVFP4 instead of FP8.** NVFP4 uses a real CUTLASS kernel on sm_120 rather than an emulation fallback, and is smaller and faster still. It is worth benchmarking, but second — the thing to validate is extraction quality on Hebrew legal text, not the kernel path, and that is a real evaluation rather than a flag flip.
- **`--enable-prefix-caching`.** Already the V1 default. Harmless to pass, but it is not what will speed you up. If your extraction prompts share a long fixed system prefix, confirm it is actually active in the startup log rather than assuming either way.

## Verifying the deployment

Watch the startup logs — vLLM prints the resolved KV-cache size and the implied maximum concurrency:

```bash
gcloud beta run services logs read qwen38-vllm-service \
  --region=europe-west4 \
  --project=moj-gcp-d-rag \
  --limit=200
```

Look for:

- `Safetensors policy check passed` — the weight-format gate cleared.
- `[startup] transformers: 5.8.x` — the version floor is satisfied.
- `[startup] extra args:` — confirms which optional flags actually made it through.
- A `libcuda` line resolving under `/usr/local/nvidia/lib64` (host driver, not compat).
- `GPU KV cache size: N tokens` / `Maximum concurrency for N tokens per request`.
- A line confirming the speculative drafter loaded.
- The Uvicorn/vLLM server reporting it is listening on `0.0.0.0:8000`.

### Reading MTP acceptance

Throughput alone cannot tell a working drafter from one that loaded and was ignored. Read the acceptance rate directly from the Prometheus endpoint:

```bash
curl -sS "${SERVICE_URL}/metrics" -H "Authorization: Bearer ${TOKEN}" \
  | grep spec_decode
```

Compute `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total`. Do this after a representative batch of real extraction requests, not after a single "hello" — acceptance on your actual Hebrew legal prompts is the number that matters, and it will not match the English benchmarks upstream.

## Calling the service

The service is internal-only and requires authentication. From inside the VPC (or via an authenticated proxy), call the OpenAI-compatible endpoint:

```bash
SERVICE_URL="https://<your-cloud-run-url>"   # from: gcloud run services describe ...
TOKEN=$(gcloud auth print-identity-token)

curl -sS "${SERVICE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-FP8",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 1.0,
    "top_p": 0.95
  }'
```

The `model` field must match `SERVED_MODEL_NAME` exactly, including case and the dots (`qwen3.8-27b-FP8`).

**Sampling defaults.** The model's `generation_config.json` ships `temperature: 1.0`, `top_p: 0.95`, `top_k: 20`. For structured extraction you will likely want lower temperature than that; just be aware you are departing from the model's tuned defaults when you do.

**Thinking control per request** (overrides the server default):

```json
{
  "model": "qwen3.8-27b-FP8",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Existing client code that already sends this needs no change — it now agrees with the server default rather than being the only thing standing between you and a reasoning block.

For a local proxy during development:

```bash
gcloud run services proxy qwen38-vllm-service \
  --region=europe-west4 --project=moj-gcp-d-rag --port=8080
# then call http://localhost:8080/v1/chat/completions
```

## Tuning context length and concurrency

FP8 weights (~31 GB) on a 96 GB card leave roughly 45–50 GB for KV cache after activations and overhead at `--gpu-memory-utilization 0.90`. With FP8 KV cache, that comfortably supports 128K context across 10 concurrent sequences, which is why the deploy sets `MAX_MODEL_LEN=131072` and `MAX_NUM_SEQS=10` (matching `--concurrency=10`).

Guidelines:

- `MAX_NUM_SEQS` should not exceed `--concurrency` — Cloud Run gates requests at the concurrency setting, so extra scheduler slots go unused.
- Speculative decoding consumes some of the same budget. If you enable MTP and then hit a KV-cache error at init, that is the interaction; lower `MAX_MODEL_LEN` or drop `num_speculative_tokens`.
- To push context toward the model's native 256K (`262144`), lower effective concurrency or verify the startup KV-cache log shows enough room.
- Beyond 262144, up to 1M, you also need an `--hf-overrides` flag. Note the nesting under `text_config` for this model (the 2.4T variant takes it flat):

  ```
  --hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}'
  ```

- If vLLM logs a "not enough KV cache" error at init, lower `MAX_MODEL_LEN` (e.g. `65536`).
- For precision-sensitive extraction, benchmark output quality with and without `--kv-cache-dtype fp8`; drop it if you see degradation. Note that fp8 KV is a choice here, not a requirement — the model serves correctly with bf16 KV at roughly half the pool size.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Invalid value specified for container memory ... must be between 80Gi and 80Gi` | Memory is pinned to CPU count; 20 CPU → 80Gi only. | Use `--memory=80Gi`, or raise `--cpu` to lift the ceiling. |
| `MODEL_PATH does not exist: /models/...` | Folder name doesn't match the bucket byte-for-byte. | `gcloud storage ls` the bucket; copy the exact name into `MODEL_PATH`. |
| `POLICY VIOLATION: Non-safetensors model weights detected` | Pickle-format weights in the bucket. | Re-download/convert to `.safetensors` and re-upload. |
| `cudaGetDeviceCount() Error 803` | CUDA compat lib shadowing the host driver. | Handled by the image (compat dir deleted); rebuild if you modified it. |
| Image pull 403 | Runtime SA lacks registry read on the image project. | Grant `roles/artifactregistry.reader` on the repo in `moj-gcp-shared-devops`. |
| Startup probe fails, container restarts during load | Host RAM OOM during weight load (distinct from KV-cache errors). | Raise `--cpu`/`--memory`, or reduce FUSE page-cache pressure. |
| `not enough KV cache` at vLLM init | `MAX_MODEL_LEN` × concurrency too large for VRAM, possibly worsened by MTP. | Lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS`, or disable `SPECULATIVE_CONFIG`. |
| Deploy fails with GPU quota/availability error | RTX PRO 6000 not available/quota in region. | Confirm region support and request GPU quota. |
| `quantization not specified` | Checkpoint quant method not auto-detected. | Add `--quantization fp8` or `--quantization compressed-tensors`. |
| Build fails resolving `transformers>=5.8.0`, or runtime prints a version below 5.8 | vLLM 0.21.0's own pin conflicts with the Qwen3.8 requirement. | Do not force the pin past vLLM's constraint — that produces subtle processor errors rather than a clean failure. Bump vLLM instead. The recipe lists 0.17.0+ as the floor for this model, and verifies sm_120 behaviour on much newer builds; 0.21.0 may simply predate full support. |
| Tool calls stop parsing after this deploy | `TOOL_CALL_PARSER` changed from `qwen3_xml` to `qwen3_coder`. | `--set-env-vars=TOOL_CALL_PARSER=qwen3_xml` to roll back, then reconcile with what your agents actually emit. |
| `--speculative-config` rejected as an unknown method | `"qwen3_5_mtp"` copied from the Ascend recipe. | Use `"mtp"` on CUDA. |
| Server starts but reasoning text appears in `message.content` | `--reasoning-parser qwen3` missing, or thinking re-enabled per request. | Keep the parser; check what `chat_template_kwargs` the client is actually sending. |
| Errors mentioning DeepGemm on the FP8 checkpoint | vLLM normally auto-disables DeepGemm for `model_type=qwen3_5_text` on Blackwell and falls back to CUTLASS. Older builds may not. | Try `VLLM_USE_DEEP_GEMM=0` as an env var. If that is needed, it is a signal your vLLM version predates the fix. |
| MXFP4 checkpoint fails to load | The vLLM MXFP4 implementation on NVIDIA devices is missing linear method support. | Use NVFP4 (or FP8) on NVIDIA. |

---

Serving stack: vLLM 0.21.0 · CUDA 13 · RTX PRO 6000 Blackwell · Cloud Run GPU.
